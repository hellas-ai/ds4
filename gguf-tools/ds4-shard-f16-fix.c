/*
 * ds4-shard-f16-fix: rewrite a ds4 TP shard GGUF, converting specific
 * Q8_0 tensors to F16 as required by the ds4 Metal graph engine.
 *
 * Tensors that must be F16 (compressor projections, HC mixing projections,
 * and the token embedding table) are dequantized Q8_0→F32→F16 in-memory.
 * All other tensors are copied byte-for-byte.
 *
 * Usage: ds4-shard-f16-fix INPUT.gguf OUTPUT.gguf [--dry-run]
 */

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

#define GGUF_MAGIC         "GGUF"
#define GGUF_ALIGNMENT     32u
#define COPY_BUF_BYTES     (8u * 1024u * 1024u)  /* 8 MiB streaming buffer */
#define CONV_BUF_ELEMS     (256u * 1024u)          /* 256K floats for conversion */

typedef enum {
    GGUF_TYPE_UINT8   = 0,  GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,  GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,  GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,  GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,  GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10, GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_value_type;

typedef struct {
    char     *name;
    uint32_t  n_dims;
    uint64_t  ne[DS4Q_MAX_DIMS];
    uint32_t  type;       /* original type from source shard */
    uint64_t  src_offset; /* offset within the source data region */
    uint64_t  src_size;   /* byte size in source */
    bool      repack;     /* true => convert Q8_0→F16 */
    uint64_t  dst_type;   /* type written to output (F16 if repack, else same) */
    uint64_t  dst_size;   /* byte size in output */
    uint64_t  dst_offset; /* offset within the output data region */
} tensor_info;

/* ── utilities ─────────────────────────────────────────────────── */

static void die(const char *msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

static void die_errno(const char *what) {
    fprintf(stderr, "error: %s: %s\n", what, strerror(errno));
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static bool str_ends(const char *s, const char *suffix) {
    size_t ns = strlen(s), nf = strlen(suffix);
    return ns >= nf && memcmp(s + ns - nf, suffix, nf) == 0;
}

static uint64_t pad_u64(uint64_t x, uint64_t n) {
    uint64_t r = x % n;
    return r ? x + (n - r) : x;
}

/* ── GGUF I/O helpers ───────────────────────────────────────────── */

static uint32_t read_u32(FILE *f, const char *ctx) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) { fprintf(stderr, "short read: %s\n", ctx); exit(1); }
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

static uint64_t read_u64(FILE *f, const char *ctx) {
    uint8_t b[8];
    if (fread(b, 1, 8, f) != 8) { fprintf(stderr, "short read: %s\n", ctx); exit(1); }
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (i * 8);
    return v;
}

static void write_u32(FILE *f, uint32_t v) {
    uint8_t b[4] = { (uint8_t)v, (uint8_t)(v>>8), (uint8_t)(v>>16), (uint8_t)(v>>24) };
    if (fwrite(b, 1, 4, f) != 4) die("write u32");
}

static void write_u64(FILE *f, uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; i++) b[i] = (uint8_t)(v >> (i*8));
    if (fwrite(b, 1, 8, f) != 8) die("write u64");
}

static char *read_gguf_string(FILE *f) {
    uint64_t n = read_u64(f, "string len");
    char *s = xmalloc((size_t)n + 1);
    if (n && fread(s, 1, (size_t)n, f) != (size_t)n) die("short string read");
    s[n] = '\0';
    return s;
}

static void write_gguf_string(FILE *f, const char *s) {
    uint64_t n = strlen(s);
    write_u64(f, n);
    if (n && fwrite(s, 1, (size_t)n, f) != (size_t)n) die("write string");
}

static uint64_t gguf_string_nbytes(const char *s) { return 8 + strlen(s); }

static off_t tell_off(FILE *f) {
    off_t p = ftello(f);
    if (p < 0) die("ftello");
    return p;
}

static void seek_abs(FILE *f, uint64_t off) {
    if (fseeko(f, (off_t)off, SEEK_SET) != 0) die_errno("fseeko");
}

static void write_zeros(FILE *f, uint64_t n) {
    static const uint8_t zero[4096] = {0};
    while (n) {
        size_t chunk = (n < sizeof(zero)) ? (size_t)n : sizeof(zero);
        if (fwrite(zero, 1, chunk, f) != chunk) die("write padding");
        n -= chunk;
    }
}

/* Skip a GGUF value without copying it (used to skip KV metadata). */
static void skip_gguf_value(FILE *f, uint32_t type) {
    uint64_t n;
    switch (type) {
    case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: fseeko(f,1,SEEK_CUR); return;
    case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: fseeko(f,2,SEEK_CUR); return;
    case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: fseeko(f,4,SEEK_CUR); return;
    case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: fseeko(f,8,SEEK_CUR); return;
    case GGUF_TYPE_STRING:
        n = read_u64(f, "kv str len");
        fseeko(f, (off_t)n, SEEK_CUR);
        return;
    case GGUF_TYPE_ARRAY: {
        uint32_t et = read_u32(f, "arr type");
        uint64_t cnt = read_u64(f, "arr count");
        if (et == GGUF_TYPE_STRING) {
            for (uint64_t i = 0; i < cnt; i++) {
                uint64_t sl = read_u64(f, "arr str len");
                fseeko(f, (off_t)sl, SEEK_CUR);
            }
        } else {
            /* scalar sizes for array element types */
            static const uint8_t asz[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
            if (et >= 13 || asz[et] == 0) die("unsupported array element type");
            fseeko(f, (off_t)(cnt * asz[et]), SEEK_CUR);
        }
        return;
    }
    default:
        fprintf(stderr, "error: unknown KV value type %u\n", type);
        exit(1);
    }
}

/* ── Tensor type utilities ──────────────────────────────────────── */

static uint64_t f16_tensor_nbytes(const uint64_t *ne, uint32_t n_dims) {
    uint64_t n = 2; /* 2 bytes per F16 element */
    for (uint32_t i = 0; i < n_dims; i++) n *= ne[i];
    return n;
}

static uint64_t q8_0_tensor_nbytes(const uint64_t *ne, uint32_t n_dims) {
    /* Q8_0: 34 bytes per block of 32 elements; ne[0] must be % 32 == 0 */
    if (n_dims == 0) die("q8_0: zero-dim tensor");
    uint64_t n = ((ne[0] + 31u) / 32u) * 34u; /* bytes per row */
    for (uint32_t i = 1; i < n_dims; i++) n *= ne[i];
    return n;
}

static uint64_t tensor_nbytes(uint32_t type, const uint64_t *ne, uint32_t n_dims) {
    if (type == DS4Q_TYPE_F16) return f16_tensor_nbytes(ne, n_dims);
    if (type == DS4Q_TYPE_Q8_0) return q8_0_tensor_nbytes(ne, n_dims);
    if (type == DS4Q_TYPE_F32) {
        uint64_t n = 4;
        for (uint32_t i = 0; i < n_dims; i++) n *= ne[i];
        return n;
    }
    /* For other types, use the generic row_size function */
    uint64_t row = (uint64_t)ds4q_row_size((ds4q_type)type, (int64_t)ne[0]);
    if (!row) { fprintf(stderr, "error: unknown tensor type %u\n", type); exit(1); }
    uint64_t n = row;
    for (uint32_t i = 1; i < n_dims; i++) n *= ne[i];
    return n;
}

/* ── Tensor name list: must be stored as F16 ────────────────────── */

static const char * const MUST_F16[] = {
    "attn_compressor_kv.weight",
    "attn_compressor_gate.weight",
    "attn_compressor_ape.weight",
    "indexer_compressor_kv.weight",
    "indexer_compressor_gate.weight",
    "indexer_compressor_ape.weight",
    "indexer.proj.weight",
    "hc_attn_fn.weight",
    "hc_ffn_fn.weight",
    "output_hc_fn.weight",
    "token_embd.weight",
    NULL
};

static bool needs_f16(const char *name) {
    for (const char * const *p = MUST_F16; *p; p++)
        if (str_ends(name, *p)) return true;
    return false;
}

/* ── Q8_0 → F16 conversion ──────────────────────────────────────── */

/*
 * Convert Q8_0 data to F16.
 * src: packed Q8_0 blocks, each 34 bytes (2-byte f16 scale + 32 int8 values).
 * dst: output F16 (uint16) array with n_elements entries.
 * n_elements must be a multiple of 32.
 */
static void q8_0_to_f16(const uint8_t *src, uint16_t *dst, uint64_t n_elements) {
    const uint64_t n_blocks = n_elements / 32u;
    float tmp[32];
    for (uint64_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * 34u;
        uint16_t hd;
        memcpy(&hd, block, 2);
        const float d = ds4q_f16_to_f32(hd);
        const int8_t *qs = (const int8_t *)(block + 2);
        for (int j = 0; j < 32; j++) tmp[j] = (float)qs[j] * d;
        ds4q_f32_to_f16_row(tmp, dst + b * 32u, 32);
    }
}

/* ── Core conversion logic ──────────────────────────────────────── */

static void copy_exact(FILE *in, FILE *out, uint8_t *buf, uint64_t n) {
    while (n) {
        size_t chunk = (n < COPY_BUF_BYTES) ? (size_t)n : COPY_BUF_BYTES;
        if (fread(buf, 1, chunk, in) != chunk) die("read tensor data");
        if (fwrite(buf, 1, chunk, out) != chunk) die("write tensor data");
        n -= chunk;
    }
}

static void convert_and_write(FILE *in, FILE *out, uint64_t q8_bytes, uint64_t n_elements) {
    /*
     * Read Q8_0 data from in (q8_bytes total) and write F16 to out.
     * Processes CONV_BUF_ELEMS elements at a time to bound memory usage.
     */
    const uint64_t chunk_elems = CONV_BUF_ELEMS;
    const uint64_t chunk_q8    = (chunk_elems / 32u) * 34u;
    const uint64_t chunk_f16   = chunk_elems * 2u;

    uint8_t  *q8buf  = xmalloc((size_t)chunk_q8);
    uint16_t *f16buf = xmalloc((size_t)chunk_f16);

    uint64_t elems_left = n_elements;
    while (elems_left) {
        uint64_t batch = (elems_left < chunk_elems) ? elems_left : chunk_elems;
        if (batch % 32 != 0) die("Q8_0 element count not divisible by 32");
        uint64_t qbytes = (batch / 32u) * 34u;
        if (fread(q8buf, 1, (size_t)qbytes, in) != (size_t)qbytes) die("read Q8_0 data");
        q8_0_to_f16(q8buf, f16buf, batch);
        uint64_t fbytes = batch * 2u;
        if (fwrite(f16buf, 1, (size_t)fbytes, out) != (size_t)fbytes) die("write F16 data");
        elems_left -= batch;
    }
    (void)q8_bytes; /* consumed implicitly by the above loop */
    free(q8buf);
    free(f16buf);
}

/* ── Main ───────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    const char *in_path  = NULL;
    const char *out_path = NULL;
    bool dry_run = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dry-run") == 0) { dry_run = true; continue; }
        if (!in_path)       { in_path  = argv[i]; continue; }
        if (!out_path)      { out_path = argv[i]; continue; }
        fprintf(stderr, "unexpected argument: %s\n", argv[i]);
        return 1;
    }
    if (!in_path || !out_path) {
        fprintf(stderr, "usage: ds4-shard-f16-fix INPUT.gguf OUTPUT.gguf [--dry-run]\n");
        return 1;
    }

    FILE *in = fopen(in_path, "rb");
    if (!in) { perror(in_path); return 1; }

    /* ── 1. Parse header ── */
    char magic[4];
    if (fread(magic, 1, 4, in) != 4 || memcmp(magic, GGUF_MAGIC, 4) != 0)
        die("not a GGUF file");
    uint32_t version   = read_u32(in, "version");
    uint64_t n_tensors = read_u64(in, "n_tensors");
    uint64_t n_kv      = read_u64(in, "n_kv");
    fprintf(stderr, "GGUF v%u, %" PRIu64 " tensors, %" PRIu64 " KV pairs\n",
            version, n_tensors, n_kv);

    /* ── 2. Remember KV metadata byte range, then skip it ── */
    uint64_t kv_start = (uint64_t)tell_off(in);
    for (uint64_t i = 0; i < n_kv; i++) {
        free(read_gguf_string(in));
        uint32_t vtype = read_u32(in, "kv type");
        skip_gguf_value(in, vtype);
    }
    uint64_t kv_end = (uint64_t)tell_off(in);
    uint64_t kv_bytes = kv_end - kv_start;

    /* ── 3. Parse tensor info ── */
    tensor_info *tensors = xmalloc((size_t)n_tensors * sizeof(tensor_info));
    for (uint64_t i = 0; i < n_tensors; i++) {
        tensor_info *t = &tensors[i];
        t->name = read_gguf_string(in);
        t->n_dims = read_u32(in, "n_dims");
        if (t->n_dims > DS4Q_MAX_DIMS) die("too many dims");
        for (uint32_t d = 0; d < t->n_dims; d++) t->ne[d] = read_u64(in, "dim");
        t->type = read_u32(in, "tensor type");
        t->src_offset = read_u64(in, "tensor offset");
        t->src_size   = tensor_nbytes(t->type, t->ne, t->n_dims);

        /* Determine whether to repack */
        t->repack = (t->type == DS4Q_TYPE_Q8_0) && needs_f16(t->name);
        if (t->repack) {
            t->dst_type = DS4Q_TYPE_F16;
            t->dst_size = f16_tensor_nbytes(t->ne, t->n_dims);
            /* Check n_elements % 32 == 0 (required by Q8_0 block layout) */
            uint64_t n_elems = 1;
            for (uint32_t d = 0; d < t->n_dims; d++) n_elems *= t->ne[d];
            if (n_elems % 32 != 0) {
                fprintf(stderr, "error: %s has %" PRIu64 " elements, not divisible by 32 "
                        "(cannot dequantize Q8_0)\n", t->name, n_elems);
                exit(1);
            }
        } else {
            t->dst_type = t->type;
            t->dst_size = t->src_size;
        }
    }
    uint64_t tensor_info_end = (uint64_t)tell_off(in);
    uint64_t src_data_offset = pad_u64(tensor_info_end, GGUF_ALIGNMENT);

    /* ── 4. Compute output tensor offsets ── */
    uint64_t dst_off = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        tensor_info *t = &tensors[i];
        t->dst_offset = dst_off;
        dst_off = pad_u64(dst_off + t->dst_size, GGUF_ALIGNMENT);
    }

    /* ── 5. Compute output header size and data offset ── */
    /* Header = magic(4) + version(4) + n_tensors(8) + n_kv(8) + KV bytes
     *        + tensor info entries */
    uint64_t dst_header = 4 + 4 + 8 + 8 + kv_bytes;
    for (uint64_t i = 0; i < n_tensors; i++) {
        tensor_info *t = &tensors[i];
        dst_header += gguf_string_nbytes(t->name);
        dst_header += 4;                        /* n_dims */
        dst_header += 8 * t->n_dims;            /* dims */
        dst_header += 4;                        /* type */
        dst_header += 8;                        /* offset */
    }
    uint64_t dst_data_offset = pad_u64(dst_header, GGUF_ALIGNMENT);

    /* ── 6. Stats and dry-run ── */
    uint64_t n_repacked = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        if (tensors[i].repack) n_repacked++;
    }
    fprintf(stderr, "repacking %" PRIu64 " / %" PRIu64 " tensors from Q8_0 → F16\n",
            n_repacked, n_tensors);
    if (n_repacked == 0) {
        fprintf(stderr, "nothing to repack; output would be identical to input.\n");
        fclose(in);
        for (uint64_t i = 0; i < n_tensors; i++) free(tensors[i].name);
        free(tensors);
        return 0;
    }

    for (uint64_t i = 0; i < n_tensors; i++) {
        if (!tensors[i].repack) continue;
        fprintf(stderr, "  repack: %s  Q8_0→F16  "
                "%" PRIu64 " B → %" PRIu64 " B\n",
                tensors[i].name, tensors[i].src_size, tensors[i].dst_size);
    }
    if (dry_run) {
        fprintf(stderr, "dry-run: no output written.\n");
        fclose(in);
        for (uint64_t i = 0; i < n_tensors; i++) free(tensors[i].name);
        free(tensors);
        return 0;
    }

    /* ── 7. Write output GGUF ── */
    FILE *out = fopen(out_path, "wb");
    if (!out) { perror(out_path); return 1; }

    /* 7a. Magic + version + counts */
    fwrite(GGUF_MAGIC, 1, 4, out);
    write_u32(out, version);
    write_u64(out, n_tensors);
    write_u64(out, n_kv);

    /* 7b. KV metadata: copy verbatim from input */
    seek_abs(in, kv_start);
    uint8_t *copybuf = xmalloc(COPY_BUF_BYTES);
    copy_exact(in, out, copybuf, kv_bytes);

    /* 7c. Tensor info */
    for (uint64_t i = 0; i < n_tensors; i++) {
        tensor_info *t = &tensors[i];
        write_gguf_string(out, t->name);
        write_u32(out, t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; d++) write_u64(out, t->ne[d]);
        write_u32(out, (uint32_t)t->dst_type);
        write_u64(out, t->dst_offset);
    }

    /* 7d. Padding to data offset */
    {
        uint64_t cur = (uint64_t)tell_off(out);
        if (cur > dst_data_offset) die("output header exceeded planned data offset");
        write_zeros(out, dst_data_offset - cur);
    }

    /* 7e. Tensor data */
    for (uint64_t i = 0; i < n_tensors; i++) {
        tensor_info *t = &tensors[i];
        uint64_t src_abs = src_data_offset + t->src_offset;
        seek_abs(in, src_abs);

        if (t->repack) {
            /* Convert Q8_0 → F16 */
            uint64_t n_elems = 1;
            for (uint32_t d = 0; d < t->n_dims; d++) n_elems *= t->ne[d];
            convert_and_write(in, out, t->src_size, n_elems);
        } else {
            copy_exact(in, out, copybuf, t->src_size);
        }

        /* Pad to next alignment boundary */
        uint64_t cur = (uint64_t)tell_off(out);
        uint64_t expected = dst_data_offset + pad_u64(t->dst_offset + t->dst_size, GGUF_ALIGNMENT);
        if (cur < expected) write_zeros(out, expected - cur);
        else if (cur > expected) {
            fprintf(stderr, "error: overran tensor %s by %" PRIu64 " bytes\n",
                    t->name, cur - expected);
            exit(1);
        }

        if ((i + 1) % 100 == 0 || i + 1 == n_tensors)
            fprintf(stderr, "  [%" PRIu64 "/%" PRIu64 "] %s\n", i+1, n_tensors, t->name);
    }

    if (fclose(out) != 0) die_errno("close output");
    fclose(in);
    free(copybuf);
    for (uint64_t i = 0; i < n_tensors; i++) free(tensors[i].name);
    free(tensors);

    fprintf(stderr, "done: %s\n", out_path);
    return 0;
}
