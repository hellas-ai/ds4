#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include "quants.h"

#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#error "ds4-tp-shard.c targets POSIX systems"
#endif

#define DS4_TP_RANKS 4u
#define DS4_GGUF_DEFAULT_ALIGNMENT 32u
#define COPY_BUF_BYTES (8u * 1024u * 1024u)

typedef enum {
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_UINT64 = 10,
} gguf_value_type;

typedef enum {
    SHARD_REPLICATED,
    SHARD_COLUMN_DIM1,
    SHARD_ROW_DIM0,
} shard_kind;

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t ne[DS4Q_MAX_DIMS];
    uint32_t type;
    uint64_t old_offset;
    uint64_t old_size;

    shard_kind kind;
    uint64_t shard_ne[DS4Q_MAX_DIMS];
    uint64_t shard_offset;
    uint64_t shard_size;
} tensor_meta;

typedef struct {
    char *path;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
    uint8_t *kv_raw;
    uint64_t kv_raw_len;
    uint64_t alignment;
    uint64_t data_offset;
    tensor_meta *tensors;
} gguf_file;

typedef struct {
    char *model;
    char *out_prefix;
    bool overwrite;
    bool dry_run;
} params;

static void die(const char *msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

static void die_errno(const char *what, const char *path) {
    fprintf(stderr, "error: %s %s: %s\n", what, path ? path : "", strerror(errno));
    exit(1);
}

static void *xmalloc(uint64_t n) {
    if (n > (uint64_t)SIZE_MAX) die("allocation too large");
    void *p = malloc((size_t)(n ? n : 1));
    if (!p) die("out of memory");
    return p;
}

static void *xcalloc(uint64_t n, uint64_t sz) {
    if (sz && n > (uint64_t)SIZE_MAX / sz) die("allocation too large");
    void *p = calloc((size_t)(n ? n : 1), (size_t)(sz ? sz : 1));
    if (!p) die("out of memory");
    return p;
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static bool u64_add(uint64_t a, uint64_t b, uint64_t *out) {
    if (a > UINT64_MAX - b) return false;
    *out = a + b;
    return true;
}

static bool u64_mul(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) return false;
    *out = a * b;
    return true;
}

static uint64_t checked_add(uint64_t a, uint64_t b, const char *what) {
    uint64_t out = 0;
    if (!u64_add(a, b, &out)) {
        fprintf(stderr, "error: integer overflow while computing %s\n", what);
        exit(1);
    }
    return out;
}

static uint64_t checked_mul(uint64_t a, uint64_t b, const char *what) {
    uint64_t out = 0;
    if (!u64_mul(a, b, &out)) {
        fprintf(stderr, "error: integer overflow while computing %s\n", what);
        exit(1);
    }
    return out;
}

static uint64_t pad_u64(uint64_t x, uint64_t n) {
    if (n == 0) die("bad GGUF alignment");
    uint64_t r = x % n;
    return r ? checked_add(x, n - r, "padding") : x;
}

static uint64_t read_u64(FILE *fp, const char *what) {
    uint8_t b[8];
    if (fread(b, 1, sizeof(b), fp) != sizeof(b)) {
        fprintf(stderr, "error: short read while reading %s\n", what);
        exit(1);
    }
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (8 * i);
    return v;
}

static uint32_t read_u32(FILE *fp, const char *what) {
    uint8_t b[4];
    if (fread(b, 1, sizeof(b), fp) != sizeof(b)) {
        fprintf(stderr, "error: short read while reading %s\n", what);
        exit(1);
    }
    return (uint32_t)b[0] |
           ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) |
           ((uint32_t)b[3] << 24);
}

static void write_u64(FILE *fp, uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; i++) b[i] = (uint8_t)(v >> (8 * i));
    if (fwrite(b, 1, sizeof(b), fp) != sizeof(b)) die("write u64 failed");
}

static void write_u32(FILE *fp, uint32_t v) {
    uint8_t b[4] = {
        (uint8_t)v,
        (uint8_t)(v >> 8),
        (uint8_t)(v >> 16),
        (uint8_t)(v >> 24),
    };
    if (fwrite(b, 1, sizeof(b), fp) != sizeof(b)) die("write u32 failed");
}

static void seek_abs(FILE *fp, uint64_t off, const char *path) {
    if (off > (uint64_t)INT64_MAX) die("file offset exceeds off_t range");
    if (fseeko(fp, (off_t)off, SEEK_SET) != 0) die_errno("seek", path);
}

static uint64_t tell_u64(FILE *fp) {
    off_t pos = ftello(fp);
    if (pos < 0) die("ftell failed");
    return (uint64_t)pos;
}

static char *read_gguf_string(FILE *fp) {
    uint64_t n = read_u64(fp, "GGUF string length");
    char *s = xmalloc(n + 1);
    if (n && fread(s, 1, (size_t)n, fp) != (size_t)n) die("short GGUF string read");
    s[n] = '\0';
    return s;
}

static void write_gguf_string(FILE *fp, const char *s) {
    uint64_t n = strlen(s);
    write_u64(fp, n);
    if (n && fwrite(s, 1, (size_t)n, fp) != (size_t)n) die("write GGUF string failed");
}

static uint64_t gguf_string_size(const char *s) {
    return checked_add(8u, strlen(s), "GGUF string size");
}

static uint64_t gguf_scalar_size(uint32_t type) {
    switch (type) {
    case 0:  /* UINT8 */
    case 1:  /* INT8 */
    case 7:  /* BOOL */
        return 1;
    case 2:  /* UINT16 */
    case 3:  /* INT16 */
        return 2;
    case GGUF_TYPE_UINT32:
    case 5:  /* INT32 */
    case 6:  /* FLOAT32 */
        return 4;
    case GGUF_TYPE_UINT64:
    case 11: /* INT64 */
    case 12: /* FLOAT64 */
        return 8;
    default:
        return 0;
    }
}

static void skip_bytes(FILE *fp, uint64_t n) {
    uint64_t pos = checked_add(tell_u64(fp), n, "skip position");
    seek_abs(fp, pos, NULL);
}

static void skip_gguf_value(FILE *fp, uint32_t type) {
    if (type == GGUF_TYPE_STRING) {
        uint64_t n = read_u64(fp, "GGUF string length");
        skip_bytes(fp, n);
        return;
    }
    if (type == GGUF_TYPE_ARRAY) {
        uint32_t elem_type = read_u32(fp, "GGUF array type");
        uint64_t n = read_u64(fp, "GGUF array count");
        if (elem_type == GGUF_TYPE_STRING) {
            for (uint64_t i = 0; i < n; i++) {
                uint64_t len = read_u64(fp, "GGUF array string length");
                skip_bytes(fp, len);
            }
        } else {
            uint64_t sz = gguf_scalar_size(elem_type);
            if (!sz) die("unsupported GGUF array type");
            skip_bytes(fp, checked_mul(n, sz, "GGUF array byte size"));
        }
        return;
    }
    uint64_t sz = gguf_scalar_size(type);
    if (!sz) die("unsupported GGUF metadata value type");
    skip_bytes(fp, sz);
}

static bool str_ends(const char *s, const char *suffix) {
    size_t ns = strlen(s);
    size_t nf = strlen(suffix);
    return ns >= nf && memcmp(s + ns - nf, suffix, nf) == 0;
}

static bool file_exists(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    fclose(fp);
    return true;
}

static char *rank_path(const char *prefix, uint32_t rank) {
    const size_t n = strlen(prefix) + 32;
    char *out = xmalloc(n);
    snprintf(out, n, "%s-rank%u.gguf", prefix, rank);
    return out;
}

static shard_kind classify_tensor(const char *name) {
    if (strstr(name, "indexer.")) return SHARD_REPLICATED;
    if (str_ends(name, ".attn_q_b.weight") ||
        str_ends(name, ".attn_output_a.weight") ||
        str_ends(name, ".ffn_gate_exps.weight") ||
        str_ends(name, ".ffn_up_exps.weight") ||
        str_ends(name, ".ffn_gate_shexp.weight") ||
        str_ends(name, ".ffn_up_shexp.weight")) {
        return SHARD_COLUMN_DIM1;
    }
    if (str_ends(name, ".attn_output_b.weight") ||
        str_ends(name, ".ffn_down_exps.weight") ||
        str_ends(name, ".ffn_down_shexp.weight")) {
        return SHARD_ROW_DIM0;
    }
    return SHARD_REPLICATED;
}

static uint64_t q8_0_row_size(uint64_t ne0) {
    return checked_mul((ne0 + 31u) / 32u, 34u, "Q8_0 row size");
}

static uint64_t tensor_row_size(uint32_t type, uint64_t ne0) {
    if (type == DS4Q_TYPE_Q8_0) return q8_0_row_size(ne0);
    if (ne0 > (uint64_t)INT64_MAX) die("tensor dim is too large");
    size_t row = ds4q_row_size((ds4q_type)type, (int64_t)ne0);
    if (row == 0) {
        const char *name = ds4q_type_name((ds4q_type)type);
        fprintf(stderr,
                "error: unsupported or unaligned tensor row type=%u%s%s%s ne0=%" PRIu64 "\n",
                type,
                name ? " (" : "",
                name ? name : "",
                name ? ")" : "",
                ne0);
        exit(1);
    }
    return (uint64_t)row;
}

static uint64_t tensor_outer_count(const tensor_meta *t) {
    uint64_t outer = 1;
    for (uint32_t i = 2; i < t->n_dims; i++) {
        outer = checked_mul(outer, t->ne[i], "tensor outer count");
    }
    return outer;
}

static uint64_t tensor_nbytes(uint32_t type, const uint64_t *ne, uint32_t n_dims) {
    if (n_dims == 0) die("bad tensor rank");
    uint64_t n = tensor_row_size(type, ne[0]);
    for (uint32_t i = 1; i < n_dims; i++) n = checked_mul(n, ne[i], "tensor byte size");
    return n;
}

static uint64_t tensor_info_size(const tensor_meta *t) {
    uint64_t n = gguf_string_size(t->name);
    n = checked_add(n, 4, "tensor info size");
    n = checked_add(n, checked_mul(t->n_dims, 8, "tensor dims size"), "tensor info size");
    n = checked_add(n, 4, "tensor info size");
    n = checked_add(n, 8, "tensor info size");
    return n;
}

static gguf_file load_gguf_metadata(const char *path) {
    gguf_file g = {0};
    g.path = xstrdup(path);
    g.alignment = DS4_GGUF_DEFAULT_ALIGNMENT;

    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open GGUF", path);

    char magic[4];
    if (fread(magic, 1, sizeof(magic), fp) != sizeof(magic) ||
        memcmp(magic, "GGUF", 4) != 0) {
        die("bad GGUF file");
    }
    g.version = read_u32(fp, "GGUF version");
    g.n_tensors = read_u64(fp, "GGUF tensor count");
    g.n_kv = read_u64(fp, "GGUF KV count");

    uint64_t kv_start = tell_u64(fp);
    for (uint64_t i = 0; i < g.n_kv; i++) {
        char *key = read_gguf_string(fp);
        uint32_t type = read_u32(fp, "GGUF KV type");
        if (!strcmp(key, "general.alignment") && type == GGUF_TYPE_UINT32) {
            uint32_t a = read_u32(fp, "GGUF alignment");
            if (a) g.alignment = a;
        } else {
            skip_gguf_value(fp, type);
        }
        free(key);
    }
    uint64_t tensor_start = tell_u64(fp);
    g.kv_raw_len = tensor_start - kv_start;
    g.kv_raw = xmalloc(g.kv_raw_len);
    seek_abs(fp, kv_start, path);
    if (g.kv_raw_len &&
        fread(g.kv_raw, 1, (size_t)g.kv_raw_len, fp) != (size_t)g.kv_raw_len) {
        die_errno("read GGUF metadata", path);
    }
    seek_abs(fp, tensor_start, path);

    g.tensors = xcalloc(g.n_tensors, sizeof(g.tensors[0]));
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        tensor_meta *t = &g.tensors[i];
        t->name = read_gguf_string(fp);
        t->n_dims = read_u32(fp, "GGUF tensor rank");
        if (t->n_dims == 0 || t->n_dims > DS4Q_MAX_DIMS) die("bad GGUF tensor rank");
        for (uint32_t j = 0; j < t->n_dims; j++) {
            t->ne[j] = read_u64(fp, "GGUF tensor dim");
            t->shard_ne[j] = t->ne[j];
        }
        t->type = read_u32(fp, "GGUF tensor type");
        t->old_offset = read_u64(fp, "GGUF tensor offset");
        t->old_size = tensor_nbytes(t->type, t->ne, t->n_dims);
    }
    g.data_offset = pad_u64(tell_u64(fp), g.alignment);
    fclose(fp);
    return g;
}

static void build_shard_plan(gguf_file *g) {
    uint64_t off = 0;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        tensor_meta *t = &g->tensors[i];
        t->kind = classify_tensor(t->name);
        memcpy(t->shard_ne, t->ne, sizeof(t->shard_ne));

        if (t->kind == SHARD_COLUMN_DIM1) {
            if (t->n_dims < 2) die("column-sharded tensor rank is less than 2");
            if ((t->ne[1] % DS4_TP_RANKS) != 0) {
                fprintf(stderr, "error: tensor %s dim[1]=%" PRIu64 " is not divisible by %u\n",
                        t->name, t->ne[1], DS4_TP_RANKS);
                exit(1);
            }
            t->shard_ne[1] = t->ne[1] / DS4_TP_RANKS;
        } else if (t->kind == SHARD_ROW_DIM0) {
            if (t->n_dims < 2) die("row-sharded tensor rank is less than 2");
            if ((t->ne[0] % DS4_TP_RANKS) != 0) {
                fprintf(stderr, "error: tensor %s dim[0]=%" PRIu64 " is not divisible by %u\n",
                        t->name, t->ne[0], DS4_TP_RANKS);
                exit(1);
            }
            t->shard_ne[0] = t->ne[0] / DS4_TP_RANKS;
            const uint64_t src_row = tensor_row_size(t->type, t->ne[0]);
            const uint64_t shard_row = tensor_row_size(t->type, t->shard_ne[0]);
            if (checked_mul(shard_row, DS4_TP_RANKS, "row shard span") != src_row) {
                fprintf(stderr,
                        "error: tensor %s dim[0] shard does not split on quant block boundaries "
                        "(src row=%" PRIu64 ", shard row=%" PRIu64 ")\n",
                        t->name, src_row, shard_row);
                exit(1);
            }
        }

        t->shard_size = tensor_nbytes(t->type, t->shard_ne, t->n_dims);
        t->shard_offset = off;
        off = checked_add(off, pad_u64(t->shard_size, g->alignment), "tensor data offset");
    }
}

static uint64_t shard_meta_size(const gguf_file *g) {
    uint64_t n = 4 + 4 + 8 + 8;
    n = checked_add(n, g->kv_raw_len, "GGUF metadata size");
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        n = checked_add(n, tensor_info_size(&g->tensors[i]), "GGUF metadata size");
    }
    return n;
}

static uint64_t shard_data_offset(const gguf_file *g) {
    return pad_u64(shard_meta_size(g), g->alignment);
}

static void write_padding(FILE *fp, uint64_t n) {
    static const uint8_t zeros[4096] = {0};
    while (n) {
        size_t chunk = n < sizeof(zeros) ? (size_t)n : sizeof(zeros);
        if (fwrite(zeros, 1, chunk, fp) != chunk) die("write padding failed");
        n -= chunk;
    }
}

static void write_header(FILE *fp, const gguf_file *g, uint64_t out_data_offset) {
    if (fwrite("GGUF", 1, 4, fp) != 4) die("write GGUF magic failed");
    write_u32(fp, g->version);
    write_u64(fp, g->n_tensors);
    write_u64(fp, g->n_kv);
    if (g->kv_raw_len &&
        fwrite(g->kv_raw, 1, (size_t)g->kv_raw_len, fp) != (size_t)g->kv_raw_len) {
        die("write GGUF metadata failed");
    }
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const tensor_meta *t = &g->tensors[i];
        write_gguf_string(fp, t->name);
        write_u32(fp, t->n_dims);
        for (uint32_t j = 0; j < t->n_dims; j++) write_u64(fp, t->shard_ne[j]);
        write_u32(fp, t->type);
        write_u64(fp, t->shard_offset);
    }
    uint64_t pos = tell_u64(fp);
    if (pos > out_data_offset) die("GGUF metadata larger than planned");
    write_padding(fp, out_data_offset - pos);
}

static void copy_exact(FILE *in, FILE *out, uint8_t *buf, uint64_t n, const char *in_path) {
    while (n) {
        size_t chunk = n < COPY_BUF_BYTES ? (size_t)n : COPY_BUF_BYTES;
        if (fread(buf, 1, chunk, in) != chunk) die_errno("read tensor", in_path);
        if (fwrite(buf, 1, chunk, out) != chunk) die("write tensor failed");
        n -= chunk;
    }
}

static void copy_range(FILE *in, FILE *out, uint8_t *buf,
                       uint64_t offset, uint64_t n, const char *in_path) {
    seek_abs(in, offset, in_path);
    copy_exact(in, out, buf, n, in_path);
}

static void write_replicated_tensor(FILE *in, FILE *out[DS4_TP_RANKS], uint8_t *buf,
                                    const gguf_file *g, const tensor_meta *t) {
    seek_abs(in, checked_add(g->data_offset, t->old_offset, "source tensor offset"), g->path);
    uint64_t n = t->old_size;
    while (n) {
        size_t chunk = n < COPY_BUF_BYTES ? (size_t)n : COPY_BUF_BYTES;
        if (fread(buf, 1, chunk, in) != chunk) die_errno("read tensor", g->path);
        for (uint32_t r = 0; r < DS4_TP_RANKS; r++) {
            if (fwrite(buf, 1, chunk, out[r]) != chunk) die("write replicated tensor failed");
        }
        n -= chunk;
    }
}

static void write_column_tensor(FILE *in, FILE *out[DS4_TP_RANKS], uint8_t *buf,
                                const gguf_file *g, const tensor_meta *t) {
    const uint64_t row_bytes = tensor_row_size(t->type, t->ne[0]);
    const uint64_t rows = t->ne[1];
    const uint64_t rows_per_rank = rows / DS4_TP_RANKS;
    const uint64_t plane_bytes = checked_mul(rows, row_bytes, "source tensor plane size");
    const uint64_t rank_bytes = checked_mul(rows_per_rank, row_bytes, "column shard size");
    const uint64_t outer = tensor_outer_count(t);
    const uint64_t src_base = checked_add(g->data_offset, t->old_offset, "source tensor offset");

    for (uint64_t plane = 0; plane < outer; plane++) {
        uint64_t plane_base = checked_add(src_base,
                                          checked_mul(plane, plane_bytes, "source plane offset"),
                                          "source plane offset");
        for (uint32_t r = 0; r < DS4_TP_RANKS; r++) {
            uint64_t row0 = checked_mul(r, rows_per_rank, "rank row start");
            uint64_t off = checked_add(plane_base,
                                       checked_mul(row0, row_bytes, "rank row byte start"),
                                       "rank source offset");
            copy_range(in, out[r], buf, off, rank_bytes, g->path);
        }
    }
}

static void write_row_tensor(FILE *in, FILE *out[DS4_TP_RANKS], uint8_t *buf,
                             const gguf_file *g, const tensor_meta *t) {
    const uint64_t src_row_bytes = tensor_row_size(t->type, t->ne[0]);
    const uint64_t shard_row_bytes = tensor_row_size(t->type, t->shard_ne[0]);
    const uint64_t rows = checked_mul(t->ne[1], tensor_outer_count(t), "row-sharded row count");
    const uint64_t src_base = checked_add(g->data_offset, t->old_offset, "source tensor offset");
    uint8_t *row = xmalloc(src_row_bytes);

    for (uint64_t y = 0; y < rows; y++) {
        uint64_t off = checked_add(src_base,
                                   checked_mul(y, src_row_bytes, "source row offset"),
                                   "source row offset");
        seek_abs(in, off, g->path);
        if (fread(row, 1, (size_t)src_row_bytes, in) != (size_t)src_row_bytes) {
            die_errno("read tensor row", g->path);
        }
        for (uint32_t r = 0; r < DS4_TP_RANKS; r++) {
            const uint8_t *slice = row + (uint64_t)r * shard_row_bytes;
            if (fwrite(slice, 1, (size_t)shard_row_bytes, out[r]) != (size_t)shard_row_bytes) {
                die("write row-sharded tensor failed");
            }
        }
    }

    free(row);
    (void)buf;
}

static const char *kind_name(shard_kind kind) {
    switch (kind) {
    case SHARD_REPLICATED:  return "replicated";
    case SHARD_COLUMN_DIM1: return "column-dim1";
    case SHARD_ROW_DIM0:    return "row-dim0";
    }
    return "unknown";
}

static void write_shards(const gguf_file *g, const params *p) {
    char *paths[DS4_TP_RANKS] = {0};
    FILE *out[DS4_TP_RANKS] = {0};
    FILE *in = fopen(g->path, "rb");
    if (!in) die_errno("open input", g->path);

    const uint64_t out_data_offset = shard_data_offset(g);
    for (uint32_t r = 0; r < DS4_TP_RANKS; r++) {
        paths[r] = rank_path(p->out_prefix, r);
        if (file_exists(paths[r]) && !p->overwrite) {
            fprintf(stderr, "error: output exists: %s (use --overwrite)\n", paths[r]);
            exit(1);
        }
        out[r] = fopen(paths[r], "wb");
        if (!out[r]) die_errno("open output", paths[r]);
        write_header(out[r], g, out_data_offset);
    }

    uint8_t *buf = xmalloc(COPY_BUF_BYTES);
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const tensor_meta *t = &g->tensors[i];
        fprintf(stderr, "[%4" PRIu64 "/%4" PRIu64 "] %-12s %s\n",
                i + 1, g->n_tensors, kind_name(t->kind), t->name);
        switch (t->kind) {
        case SHARD_REPLICATED:
            write_replicated_tensor(in, out, buf, g, t);
            break;
        case SHARD_COLUMN_DIM1:
            write_column_tensor(in, out, buf, g, t);
            break;
        case SHARD_ROW_DIM0:
            write_row_tensor(in, out, buf, g, t);
            break;
        }
        const uint64_t padded = pad_u64(t->shard_size, g->alignment);
        for (uint32_t r = 0; r < DS4_TP_RANKS; r++) {
            write_padding(out[r], padded - t->shard_size);
        }
    }
    free(buf);

    fclose(in);
    for (uint32_t r = 0; r < DS4_TP_RANKS; r++) {
        if (fclose(out[r]) != 0) die_errno("close output", paths[r]);
        fprintf(stderr, "wrote %s\n", paths[r]);
        free(paths[r]);
    }
}

static void free_gguf(gguf_file *g) {
    free(g->path);
    free(g->kv_raw);
    for (uint64_t i = 0; i < g->n_tensors; i++) free(g->tensors[i].name);
    free(g->tensors);
}

static void usage(const char *argv0) {
    printf("usage: %s --model MODEL.gguf --out-prefix PREFIX [--overwrite] [--dry-run]\n", argv0);
    printf("\nWrites PREFIX-rank0.gguf through PREFIX-rank3.gguf for DS4 tensor parallelism.\n");
}

static char *need_arg(int argc, char **argv, int *i, const char *arg) {
    if (++*i >= argc) {
        fprintf(stderr, "error: missing value for %s\n", arg);
        exit(1);
    }
    return argv[*i];
}

static params parse_args(int argc, char **argv) {
    params p = {0};
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(argv[0]);
            exit(0);
        } else if (!strcmp(arg, "--model") || !strcmp(arg, "--in")) {
            p.model = need_arg(argc, argv, &i, arg);
        } else if (!strcmp(arg, "--out-prefix")) {
            p.out_prefix = need_arg(argc, argv, &i, arg);
        } else if (!strcmp(arg, "--overwrite")) {
            p.overwrite = true;
        } else if (!strcmp(arg, "--dry-run")) {
            p.dry_run = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg);
            exit(1);
        }
    }
    if (!p.model) die("--model is required");
    if (!p.out_prefix) die("--out-prefix is required");
    return p;
}

int main(int argc, char **argv) {
    params p = parse_args(argc, argv);
    gguf_file g = load_gguf_metadata(p.model);
    build_shard_plan(&g);

    uint64_t replicated = 0, column = 0, row = 0;
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        if (g.tensors[i].kind == SHARD_REPLICATED) replicated++;
        else if (g.tensors[i].kind == SHARD_COLUMN_DIM1) column++;
        else row++;
    }
    fprintf(stderr,
            "ds4-tp-shard: tensors=%" PRIu64 " replicated=%" PRIu64
            " column=%" PRIu64 " row=%" PRIu64 " alignment=%" PRIu64 "\n",
            g.n_tensors, replicated, column, row, g.alignment);
    fprintf(stderr,
            "ds4-tp-shard: input_data_offset=%" PRIu64 " output_data_offset=%" PRIu64 "\n",
            g.data_offset, shard_data_offset(&g));

    if (!p.dry_run) write_shards(&g, &p);
    free_gguf(&g);
    return 0;
}
