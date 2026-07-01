/*
 * ds4_tp.c — Tensor parallelism: RCCL data plane + TCP control plane.
 *
 * Bootstrap sequence:
 *   Rank 0: ds4_tp_ctx_create → listens on bootstrap_port, accepts tp_size-1
 *           peers, receives HELLO from each, calls ncclGetUniqueId, sends NCCL_ID
 *           to all peers, calls ncclCommInitRank(rank=0), waits for READY.
 *   Rank N: ds4_tp_ctx_create → connects to rank 0, sends HELLO, receives
 *           NCCL_ID, calls ncclCommInitRank(rank=N), sends READY, enters
 *           worker loop waiting for TOKEN/DRAFT/ACCEPT/RESET/SHUTDOWN.
 */

#define _POSIX_C_SOURCE 200809L

#include "ds4_tp.h"
#include "ds4.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#define DS4_TP_HAVE_RCCL 1
#else
/* Stub types for non-ROCm builds (e.g. macOS Metal build). TP requires ROCm. */
typedef void *ncclComm_t;
typedef struct { char internal[128]; } ncclUniqueId;
#define ncclSuccess 0
#define DS4_TP_HAVE_RCCL 0
#endif

/* ── struct definition ─────────────────────────────────────────────────────── */

struct ds4_tp_ctx {
    bool     enabled;
    uint32_t rank;
    uint32_t tp_size;

#if DS4_TP_HAVE_RCCL
    ncclComm_t  comm;
    hipStream_t stream;
    bool        comm_ready;
#endif

    /* Control-plane sockets.
     * Rank 0: peer_fds[0..tp_size-2] are the accepted worker connections.
     * Rank N: peer_fds[0] is the connection to rank 0. */
    int    *peer_fds;
    uint32_t n_peers;

    /* Monotonically increasing message sequence. */
    uint32_t seq;
    pthread_mutex_t mu;
};

/* ── helpers ────────────────────────────────────────────────────────────────── */

static int tp_err(char *err, size_t errlen, const char *fmt, ...) {
    if (err && errlen) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    return -1;
}

static int tp_send_exact(int fd, const void *buf, size_t n) {
    const char *p = (const char *)buf;
    while (n) {
        ssize_t r = write(fd, p, n);
        if (r <= 0) return -1;
        p += r; n -= (size_t)r;
    }
    return 0;
}

static int tp_recv_exact(int fd, void *buf, size_t n) {
    char *p = (char *)buf;
    while (n) {
        ssize_t r = read(fd, p, n);
        if (r <= 0) return -1;
        p += r; n -= (size_t)r;
    }
    return 0;
}

static int tp_send_frame(int fd, uint32_t type, uint32_t seq,
                         const void *payload, uint32_t payload_bytes) {
    ds4_tp_frame hdr;
    hdr.magic = DS4_TP_MSG_MAGIC;
    hdr.type  = type;
    hdr.bytes = payload_bytes;
    hdr.seq   = seq;
    if (tp_send_exact(fd, &hdr, sizeof(hdr)) < 0) return -1;
    if (payload_bytes && tp_send_exact(fd, payload, payload_bytes) < 0) return -1;
    return 0;
}

static int tp_recv_frame(int fd, ds4_tp_frame *hdr, void *payload_buf,
                         uint32_t payload_cap) {
    if (tp_recv_exact(fd, hdr, sizeof(*hdr)) < 0) return -1;
    if (hdr->magic != DS4_TP_MSG_MAGIC) return -1;
    if (hdr->bytes > payload_cap) return -1;
    if (hdr->bytes && tp_recv_exact(fd, payload_buf, hdr->bytes) < 0) return -1;
    return 0;
}

static int tp_set_nodelay(int fd) {
    int one = 1;
    return setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

static int tp_connect(const char *host, int port) {
    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);
    struct addrinfo hints = {0}, *res = NULL;
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, port_str, &hints, &res) != 0) return -1;
    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        close(fd); fd = -1;
    }
    freeaddrinfo(res);
    if (fd >= 0) tp_set_nodelay(fd);
    return fd;
}

static int tp_listen(int port) {
    int fd = socket(AF_INET6, SOCK_STREAM, 0);
    if (fd < 0) fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in6 addr6 = {0};
    addr6.sin6_family = AF_INET6;
    addr6.sin6_port   = htons((uint16_t)port);
    addr6.sin6_addr   = in6addr_any;
    if (bind(fd, (struct sockaddr *)&addr6, sizeof(addr6)) < 0) {
        struct sockaddr_in addr4 = {0};
        addr4.sin_family = AF_INET;
        addr4.sin_port   = htons((uint16_t)port);
        addr4.sin_addr.s_addr = INADDR_ANY;
        if (bind(fd, (struct sockaddr *)&addr4, sizeof(addr4)) < 0) {
            close(fd); return -1;
        }
    }
    if (listen(fd, 16) < 0) { close(fd); return -1; }
    return fd;
}

/* ── rank 0: accept all workers, exchange NCCL id ─────────────────────────── */

static int tp_rank0_bootstrap(ds4_tp_ctx *tp, const ds4_tp_options *opt,
                               char *err, size_t errlen) {
#if !DS4_TP_HAVE_RCCL
    return tp_err(err, errlen, "TP requires ROCm/RCCL");
#else
    const uint32_t n_workers = opt->tp_size - 1;
    tp->n_peers = n_workers;
    tp->peer_fds = (int *)calloc(n_workers, sizeof(int));
    if (!tp->peer_fds) return tp_err(err, errlen, "TP: alloc peer_fds");

    int listen_fd = tp_listen(opt->bootstrap_port);
    if (listen_fd < 0)
        return tp_err(err, errlen, "TP rank 0: listen on port %d: %s",
                      opt->bootstrap_port, strerror(errno));

    fprintf(stderr, "ds4-tp: rank 0 listening on port %d\n", opt->bootstrap_port);

    /* Accept exactly n_workers connections; verify HELLO from each. */
    uint8_t buf[512];
    uint32_t connected = 0;
    while (connected < n_workers) {
        int cfd = accept(listen_fd, NULL, NULL);
        if (cfd < 0) { close(listen_fd); return tp_err(err, errlen, "TP accept: %s", strerror(errno)); }
        tp_set_nodelay(cfd);

        ds4_tp_frame hdr;
        ds4_tp_hello hello;
        if (tp_recv_frame(cfd, &hdr, buf, sizeof(buf)) < 0 ||
            hdr.type != DS4_TP_MSG_HELLO ||
            hdr.bytes != sizeof(ds4_tp_hello)) {
            close(cfd); close(listen_fd);
            return tp_err(err, errlen, "TP rank 0: bad HELLO from peer");
        }
        memcpy(&hello, buf, sizeof(hello));
        if (hello.tp_size != opt->tp_size) {
            close(cfd); close(listen_fd);
            return tp_err(err, errlen, "TP rank 0: tp_size mismatch: got %u expected %u",
                          hello.tp_size, opt->tp_size);
        }
        /* Store by rank (rank 1 → slot 0, etc.) */
        if (hello.rank == 0 || hello.rank >= opt->tp_size) {
            close(cfd); close(listen_fd);
            return tp_err(err, errlen, "TP rank 0: bad peer rank %u", hello.rank);
        }
        tp->peer_fds[hello.rank - 1] = cfd;
        fprintf(stderr, "ds4-tp: rank 0 accepted rank %u\n", hello.rank);
        connected++;
    }
    close(listen_fd);

    /* Generate NCCL unique id and broadcast to all workers. */
    ncclUniqueId nccl_id;
    if (ncclGetUniqueId(&nccl_id) != ncclSuccess)
        return tp_err(err, errlen, "TP: ncclGetUniqueId failed");

    for (uint32_t i = 0; i < n_workers; i++) {
        if (tp_send_frame(tp->peer_fds[i], DS4_TP_MSG_NCCL_ID, tp->seq++,
                          &nccl_id, sizeof(nccl_id)) < 0)
            return tp_err(err, errlen, "TP rank 0: send NCCL_ID to rank %u failed", i + 1);
    }

    /* Use the default (null) stream so allreduce serializes with compute. */
    tp->stream = NULL;
    if (ncclCommInitRank(&tp->comm, (int)opt->tp_size, nccl_id, 0) != ncclSuccess)
        return tp_err(err, errlen, "TP rank 0: ncclCommInitRank failed");
    tp->comm_ready = true;

    /* Wait for READY from all workers. */
    for (uint32_t i = 0; i < n_workers; i++) {
        ds4_tp_frame hdr;
        if (tp_recv_frame(tp->peer_fds[i], &hdr, buf, sizeof(buf)) < 0 ||
            hdr.type != DS4_TP_MSG_READY)
            return tp_err(err, errlen, "TP rank 0: no READY from rank %u", i + 1);
    }

    fprintf(stderr, "ds4-tp: rank 0 RCCL communicator ready, tp_size=%u\n", opt->tp_size);
    return 0;
#endif
}

/* ── rank N: connect to rank 0, receive NCCL id ──────────────────────────── */

static int tp_rankN_bootstrap(ds4_tp_ctx *tp, const ds4_tp_options *opt,
                               char *err, size_t errlen) {
#if !DS4_TP_HAVE_RCCL
    return tp_err(err, errlen, "TP requires ROCm/RCCL");
#else
    tp->n_peers = 1;
    tp->peer_fds = (int *)calloc(1, sizeof(int));
    if (!tp->peer_fds) return tp_err(err, errlen, "TP: alloc peer_fds");

    /* Retry connect for up to 30 seconds — rank 0 may not be ready yet. */
    int fd = -1;
    for (int attempt = 0; attempt < 60 && fd < 0; attempt++) {
        fd = tp_connect(opt->bootstrap_host, opt->bootstrap_port);
        if (fd < 0) { usleep(500000); }
    }
    if (fd < 0)
        return tp_err(err, errlen, "TP rank %u: connect to rank 0 (%s:%d): %s",
                      opt->rank, opt->bootstrap_host, opt->bootstrap_port, strerror(errno));
    tp->peer_fds[0] = fd;

    /* Send HELLO. */
    ds4_tp_hello hello = { .rank = opt->rank, .tp_size = opt->tp_size };
    if (tp_send_frame(fd, DS4_TP_MSG_HELLO, tp->seq++, &hello, sizeof(hello)) < 0)
        return tp_err(err, errlen, "TP rank %u: send HELLO failed", opt->rank);

    /* Receive NCCL_ID. */
    ds4_tp_frame hdr;
    uint8_t buf[512];
    ncclUniqueId nccl_id;
    if (tp_recv_frame(fd, &hdr, buf, sizeof(buf)) < 0 ||
        hdr.type != DS4_TP_MSG_NCCL_ID ||
        hdr.bytes != sizeof(nccl_id))
        return tp_err(err, errlen, "TP rank %u: bad NCCL_ID", opt->rank);
    memcpy(&nccl_id, buf, sizeof(nccl_id));

    /* Use the default (null) stream so allreduce serializes with compute. */
    tp->stream = NULL;
    if (ncclCommInitRank(&tp->comm, (int)opt->tp_size, nccl_id, (int)opt->rank) != ncclSuccess)
        return tp_err(err, errlen, "TP rank %u: ncclCommInitRank failed", opt->rank);
    tp->comm_ready = true;

    /* Send READY. */
    if (tp_send_frame(fd, DS4_TP_MSG_READY, tp->seq++, NULL, 0) < 0)
        return tp_err(err, errlen, "TP rank %u: send READY failed", opt->rank);

    fprintf(stderr, "ds4-tp: rank %u RCCL communicator ready\n", opt->rank);
    return 0;
#endif
}

/* ── public: create/destroy ─────────────────────────────────────────────────── */

int ds4_tp_ctx_create(ds4_tp_ctx **out, const ds4_tp_options *opt,
                      char *err, size_t errlen) {
    if (!out || !opt) return tp_err(err, errlen, "ds4_tp_ctx_create: null arg");
    *out = NULL;

    ds4_tp_ctx *tp = (ds4_tp_ctx *)calloc(1, sizeof(*tp));
    if (!tp) return tp_err(err, errlen, "ds4_tp_ctx_create: alloc");

    tp->enabled = opt->enabled;
    tp->rank    = opt->rank;
    tp->tp_size = opt->tp_size;
    pthread_mutex_init(&tp->mu, NULL);

    if (!opt->enabled || opt->tp_size <= 1) {
        *out = tp;
        return 0;
    }

    /* Apply defaults. */
    ds4_tp_options eff = *opt;
    if (eff.bootstrap_port == 0) eff.bootstrap_port = 54321;
    if (!eff.bootstrap_host || !eff.bootstrap_host[0]) eff.bootstrap_host = "127.0.0.1";

    int rc;
    if (eff.rank == 0)
        rc = tp_rank0_bootstrap(tp, &eff, err, errlen);
    else
        rc = tp_rankN_bootstrap(tp, &eff, err, errlen);

    if (rc < 0) { free(tp->peer_fds); free(tp); return rc; }
    *out = tp;
    return 0;
}

void ds4_tp_ctx_destroy(ds4_tp_ctx *tp) {
    if (!tp) return;
#if DS4_TP_HAVE_RCCL
    if (tp->comm_ready) {
        hipStreamSynchronize(tp->stream);  /* NULL = hipDeviceSynchronize */
        ncclCommDestroy(tp->comm);
    }
#endif
    if (tp->peer_fds) {
        for (uint32_t i = 0; i < tp->n_peers; i++)
            if (tp->peer_fds[i] >= 0) close(tp->peer_fds[i]);
        free(tp->peer_fds);
    }
    pthread_mutex_destroy(&tp->mu);
    free(tp);
}

bool ds4_tp_enabled(const ds4_tp_ctx *tp) { return tp && tp->enabled && tp->tp_size > 1; }
uint32_t ds4_tp_rank(const ds4_tp_ctx *tp) { return tp ? tp->rank : 0; }
uint32_t ds4_tp_size(const ds4_tp_ctx *tp) { return tp ? tp->tp_size : 1; }

/* ── allreduce ──────────────────────────────────────────────────────────────── */

int ds4_tp_allreduce_f32(ds4_tp_ctx *tp, float *buf_dev, size_t n,
                         char *err, size_t errlen) {
    if (!ds4_tp_enabled(tp)) return 0;
#if !DS4_TP_HAVE_RCCL
    return tp_err(err, errlen, "ds4_tp_allreduce_f32: no RCCL");
#else
    if (!tp->comm_ready) return tp_err(err, errlen, "TP: comm not ready");
    if (ncclAllReduce(buf_dev, buf_dev, n, ncclFloat, ncclSum,
                      tp->comm, tp->stream) != ncclSuccess)
        return tp_err(err, errlen, "TP: ncclAllReduce failed");
    return 0;
#endif
}

int ds4_tp_stream_sync(ds4_tp_ctx *tp, char *err, size_t errlen) {
    if (!ds4_tp_enabled(tp)) return 0;
#if !DS4_TP_HAVE_RCCL
    return tp_err(err, errlen, "ds4_tp_stream_sync: no RCCL");
#else
    if (hipStreamSynchronize(tp->stream) != hipSuccess)
        return tp_err(err, errlen, "TP: hipStreamSynchronize failed");
    return 0;
#endif
}

/* ── rank 0: control-plane broadcasts ─────────────────────────────────────── */

static int tp_broadcast(ds4_tp_ctx *tp, uint32_t type,
                        const void *payload, uint32_t payload_bytes,
                        char *err, size_t errlen) {
    if (!ds4_tp_enabled(tp) || tp->rank != 0) return 0;
    pthread_mutex_lock(&tp->mu);
    uint32_t seq = tp->seq++;
    pthread_mutex_unlock(&tp->mu);
    for (uint32_t i = 0; i < tp->n_peers; i++) {
        if (tp_send_frame(tp->peer_fds[i], type, seq, payload, payload_bytes) < 0) {
            return tp_err(err, errlen, "TP broadcast type=%u to peer %u failed", type, i);
        }
    }
    return 0;
}

int ds4_tp_broadcast_token(ds4_tp_ctx *tp, int32_t token, uint32_t pos,
                           char *err, size_t errlen) {
    ds4_tp_token_step msg = { .token = token, .pos = pos };
    return tp_broadcast(tp, DS4_TP_MSG_TOKEN, &msg, sizeof(msg), err, errlen);
}

int ds4_tp_broadcast_draft(ds4_tp_ctx *tp, const int32_t *tokens,
                           uint32_t n_draft, uint32_t pos,
                           char *err, size_t errlen) {
    ds4_tp_draft_step msg = {0};
    if (n_draft > 8) n_draft = 8;
    memcpy(msg.tokens, tokens, n_draft * sizeof(int32_t));
    msg.n_draft = n_draft;
    msg.pos     = pos;
    return tp_broadcast(tp, DS4_TP_MSG_DRAFT, &msg, sizeof(msg), err, errlen);
}

int ds4_tp_broadcast_accept(ds4_tp_ctx *tp, uint32_t n_accepted,
                            int32_t next_token, char *err, size_t errlen) {
    ds4_tp_accept msg = { .n_accepted = n_accepted, .next_token = next_token };
    return tp_broadcast(tp, DS4_TP_MSG_ACCEPT, &msg, sizeof(msg), err, errlen);
}

int ds4_tp_broadcast_reset(ds4_tp_ctx *tp, char *err, size_t errlen) {
    return tp_broadcast(tp, DS4_TP_MSG_RESET, NULL, 0, err, errlen);
}

int ds4_tp_broadcast_shutdown(ds4_tp_ctx *tp, char *err, size_t errlen) {
    return tp_broadcast(tp, DS4_TP_MSG_SHUTDOWN, NULL, 0, err, errlen);
}

/* ── worker loop (ranks 1..N-1) ──────────────────────────────────────────── */

int ds4_tp_worker_run(ds4_tp_ctx *tp, struct ds4_engine *engine,
                      int ctx_size, char *err, size_t errlen) {
    if (!tp || !tp->enabled || tp->rank == 0)
        return tp_err(err, errlen, "ds4_tp_worker_run: not a worker rank");

    int fd = tp->peer_fds[0];
    ds4_session *sess = NULL;
    uint8_t buf[256];

    if (ctx_size <= 0) ctx_size = 8192;
    if (ds4_session_create(&sess, engine, ctx_size) != 0) {
        return tp_err(err, errlen, "TP worker: session create failed");
    }

    fprintf(stderr, "ds4-tp: rank %u worker loop started\n", tp->rank);

    int pos_before_draft = 0;  /* position saved at start of DRAFT batch */

    for (;;) {
        ds4_tp_frame hdr;
        if (tp_recv_frame(fd, &hdr, buf, sizeof(buf)) < 0) {
            fprintf(stderr, "ds4-tp: rank %u lost connection to rank 0\n", tp->rank);
            break;
        }

        switch (hdr.type) {
        case DS4_TP_MSG_TOKEN: {
            ds4_tp_token_step step;
            if (hdr.bytes != sizeof(step)) break;
            memcpy(&step, buf, sizeof(step));
            /* Evaluate the token (layer eval will do RCCL allreduce internally). */
            char eval_err[256] = {0};
            if (ds4_session_eval(sess, step.token, eval_err, sizeof(eval_err)) != 0) {
                fprintf(stderr, "ds4-tp: rank %u TOKEN eval error: %s\n", tp->rank, eval_err);
#if DS4_TP_HAVE_RCCL
                ncclCommAbort(tp->comm);  /* unblock other ranks stuck in allreduce */
#endif
                goto done;
            }
            break;
        }
        case DS4_TP_MSG_DRAFT: {
            ds4_tp_draft_step step;
            if (hdr.bytes != sizeof(step)) break;
            memcpy(&step, buf, sizeof(step));
            /* Save position so ACCEPT can roll back correctly. */
            pos_before_draft = ds4_session_pos(sess);
            char eval_err[256] = {0};
            for (uint32_t i = 0; i < step.n_draft && i < 8; i++) {
                if (ds4_session_eval(sess, step.tokens[i], eval_err, sizeof(eval_err)) != 0) {
                    fprintf(stderr, "ds4-tp: rank %u DRAFT eval error: %s\n", tp->rank, eval_err);
#if DS4_TP_HAVE_RCCL
                    ncclCommAbort(tp->comm);
#endif
                    goto done;
                }
            }
            break;
        }
        case DS4_TP_MSG_ACCEPT: {
            ds4_tp_accept accept;
            if (hdr.bytes != sizeof(accept)) break;
            memcpy(&accept, buf, sizeof(accept));
            /* Rewind to pos_before_draft + n_accepted tokens of draft. */
            int target = pos_before_draft + (int)accept.n_accepted;
            if (target < ds4_session_pos(sess))
                ds4_session_rewind(sess, target);
            break;
        }
        case DS4_TP_MSG_RESET:
            ds4_session_invalidate(sess);
            break;
        case DS4_TP_MSG_SHUTDOWN:
            fprintf(stderr, "ds4-tp: rank %u shutdown\n", tp->rank);
            goto done;
        default:
            fprintf(stderr, "ds4-tp: rank %u unknown msg type %u\n", tp->rank, hdr.type);
            break;
        }
    }

done:
    ds4_session_free(sess);
    return 0;
}
