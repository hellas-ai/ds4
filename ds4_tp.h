#ifndef DS4_TP_H
#define DS4_TP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Tensor parallelism for DS4 on multi-node Strix Halo clusters.
 *
 * TP is a separate mode from the existing pipeline parallelism (PP). It is NOT
 * implemented as a ds4_distributed_role. All nodes load all layers but only a
 * 1/tp_size shard of the weight tensors (column-parallel for up/gate/Q, row-
 * parallel for down/O). After each row-parallel projection, all nodes perform
 * an RCCL all-reduce to sum their partial activations.
 *
 * All nodes run the normal ds4_session API. TP rank 0 also runs the HTTP
 * server and owns sampling / MTP speculation.
 *
 * Communication:
 *   - Data plane: RCCL ncclAllReduce (over USB4 RDMA / RoCE)
 *   - Control plane: minimal TCP protocol, rank 0 acts as server
 *
 * All-reduce points per layer (16 KiB each for 4096-float hidden state):
 *   1. After attention O_b row-parallel matmul, before hc_expand
 *   2. After MoE down + shared down partial sum, before hc_expand
 */

/* ── options (goes in ds4_engine_options) ─────────────────────────────────── */

typedef struct {
    bool    enabled;
    uint32_t rank;        /* 0 .. tp_size-1 */
    uint32_t tp_size;     /* 4 for our cluster */
    /* Rank 0 listens here; ranks 1-3 connect to rank 0 on this host:port */
    const char *bootstrap_host;
    int          bootstrap_port;
} ds4_tp_options;

/* ── opaque context (owned by ds4_engine after model load) ────────────────── */

typedef struct ds4_tp_ctx ds4_tp_ctx;

/* Create/destroy. Called from ds4_engine_open after GPU backend is ready. */
int  ds4_tp_ctx_create(ds4_tp_ctx **out, const ds4_tp_options *opt,
                       char *err, size_t errlen);
void ds4_tp_ctx_destroy(ds4_tp_ctx *tp);

bool ds4_tp_enabled(const ds4_tp_ctx *tp);
uint32_t ds4_tp_rank(const ds4_tp_ctx *tp);
uint32_t ds4_tp_size(const ds4_tp_ctx *tp);

/*
 * All-reduce a float buffer already on the GPU.  buf must be a device pointer
 * of n floats.  Blocks until the collective is complete on the RCCL stream,
 * then enqueues a stream-order event so subsequent GPU work can proceed.
 */
int ds4_tp_allreduce_f32(ds4_tp_ctx *tp, float *buf_dev, size_t n,
                         char *err, size_t errlen);

/* Synchronize the RCCL stream (call after the last allreduce of a step). */
int ds4_tp_stream_sync(ds4_tp_ctx *tp, char *err, size_t errlen);

/* ── control-plane messages ───────────────────────────────────────────────── */

#define DS4_TP_MSG_MAGIC    0x44345450u   /* "D4TP" */
#define DS4_TP_MSG_HELLO    1u
#define DS4_TP_MSG_NCCL_ID  2u
#define DS4_TP_MSG_READY    3u
#define DS4_TP_MSG_TOKEN    4u   /* rank 0 → workers: next token to eval */
#define DS4_TP_MSG_DRAFT    5u   /* rank 0 → workers: MTP draft tokens */
#define DS4_TP_MSG_ACCEPT   6u   /* rank 0 → workers: accepted draft count */
#define DS4_TP_MSG_RESET    7u   /* rank 0 → workers: reset session */
#define DS4_TP_MSG_SHUTDOWN 8u

typedef struct {
    uint32_t magic;
    uint32_t type;
    uint32_t bytes;   /* payload bytes after this header */
    uint32_t seq;
} ds4_tp_frame;

typedef struct {
    uint32_t rank;
    uint32_t tp_size;
    uint64_t model_hash;  /* ds4_engine_model_id(), used for sanity check */
} ds4_tp_hello;

typedef struct {
    int32_t  token;
    uint32_t pos;
} ds4_tp_token_step;

typedef struct {
    int32_t  tokens[8];   /* draft token ids, 0-padded */
    uint32_t n_draft;
    uint32_t pos;
} ds4_tp_draft_step;

typedef struct {
    uint32_t n_accepted;
    int32_t  next_token;  /* the first non-draft token */
} ds4_tp_accept;

/*
 * Worker loop: ranks 1..tp_size-1 call this after ds4_tp_ctx_create.
 * Blocks until DS4_TP_MSG_SHUTDOWN is received.
 */
int ds4_tp_worker_run(ds4_tp_ctx *tp, struct ds4_engine *engine,
                      int ctx_size, char *err, size_t errlen);

/*
 * Rank 0 helpers called from ds4_server / inference loop.
 * These broadcast control messages to all worker peers.
 */
int ds4_tp_broadcast_token(ds4_tp_ctx *tp, int32_t token, uint32_t pos,
                           char *err, size_t errlen);
int ds4_tp_broadcast_draft(ds4_tp_ctx *tp, const int32_t *tokens,
                           uint32_t n_draft, uint32_t pos,
                           char *err, size_t errlen);
int ds4_tp_broadcast_accept(ds4_tp_ctx *tp, uint32_t n_accepted,
                            int32_t next_token, char *err, size_t errlen);
int ds4_tp_broadcast_reset(ds4_tp_ctx *tp, char *err, size_t errlen);
int ds4_tp_broadcast_shutdown(ds4_tp_ctx *tp, char *err, size_t errlen);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DS4_TP_H */
