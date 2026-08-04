/**
 * @file zstd.c
 * @brief ZSTD compression/decompression using libzstd
 *
 * Uses streaming context for better performance on repeated decompressions.
 */

#include <carquet/error.h>
#include <stdint.h>
#include <stddef.h>
#include <zstd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ============================================================================
 * Thread-local ZSTD context management
 *
 * ZSTD contexts are expensive to create (~650KB each) so we cache them per
 * thread.  The challenge is cleanup: __thread / __declspec(thread) have no
 * destructor, so contexts allocated by OMP worker threads leak when the
 * thread pool is torn down.
 *
 * Strategy:
 *   POSIX (any)      -> pthread_key_create with destructors.  Works for both
 *                        OpenMP threads and worker pool pthreads.  The pthread
 *                        runtime calls the destructor when each thread exits.
 *   Windows (any)    -> thread-local storage with explicit
 *                        carquet_zstd_cleanup, called by each worker thread as
 *                        it exits (see reader/worker_pool.c).
 *
 * The Windows case must not depend on OpenMP.  It once fell back to plain
 * global statics when _OPENMP was undefined, on the assumption that without
 * OpenMP there was only one thread -- but carquet's own worker pool creates
 * threads on Windows whether or not OpenMP is configured, so two of them could
 * decompress through one shared ZSTD_DCtx at the same time.  That is a data
 * race, and it corrupted or crashed any multi-column zstd read.
 *
 * carquet_cleanup() (public API) calls carquet_zstd_cleanup() for the
 * calling thread.  On POSIX the worker-thread contexts are freed
 * automatically; on Windows the worker pool frees its own.
 * ============================================================================ */

#if !defined(_WIN32)
/* ---- POSIX: pthread_key with destructors (works for OMP + worker pool) ---- */
#include <pthread.h>

static pthread_key_t tls_dctx_key;
static pthread_key_t tls_cctx_key;
static pthread_once_t tls_keys_once = PTHREAD_ONCE_INIT;

static void destroy_dctx(void* ctx) {
    if (ctx) ZSTD_freeDCtx((ZSTD_DCtx*)ctx);
}

static void destroy_cctx(void* ctx) {
    if (ctx) ZSTD_freeCCtx((ZSTD_CCtx*)ctx);
}

static void init_tls_keys(void) {
    pthread_key_create(&tls_dctx_key, destroy_dctx);
    pthread_key_create(&tls_cctx_key, destroy_cctx);
}

static ZSTD_DCtx* get_dctx(void) {
    pthread_once(&tls_keys_once, init_tls_keys);
    ZSTD_DCtx* dctx = (ZSTD_DCtx*)pthread_getspecific(tls_dctx_key);
    if (!dctx) {
        dctx = ZSTD_createDCtx();
        if (dctx) pthread_setspecific(tls_dctx_key, dctx);
    }
    return dctx;
}

static ZSTD_CCtx* get_cctx(void) {
    pthread_once(&tls_keys_once, init_tls_keys);
    ZSTD_CCtx* cctx = (ZSTD_CCtx*)pthread_getspecific(tls_cctx_key);
    if (!cctx) {
        cctx = ZSTD_createCCtx();
        if (cctx) pthread_setspecific(tls_cctx_key, cctx);
    }
    return cctx;
}

void carquet_zstd_cleanup(void) {
    pthread_once(&tls_keys_once, init_tls_keys);
    ZSTD_DCtx* dctx = (ZSTD_DCtx*)pthread_getspecific(tls_dctx_key);
    if (dctx) {
        ZSTD_freeDCtx(dctx);
        pthread_setspecific(tls_dctx_key, NULL);
    }
    ZSTD_CCtx* cctx = (ZSTD_CCtx*)pthread_getspecific(tls_cctx_key);
    if (cctx) {
        ZSTD_freeCCtx(cctx);
        pthread_setspecific(tls_cctx_key, NULL);
    }
}

#else
/* ---- Windows: thread-local storage with explicit cleanup ---- */
#if defined(_MSC_VER)
#define CARQUET_ZSTD_TLS __declspec(thread)
#else
#define CARQUET_ZSTD_TLS __thread
#endif

static CARQUET_ZSTD_TLS ZSTD_DCtx* tls_dctx = NULL;
static CARQUET_ZSTD_TLS ZSTD_CCtx* tls_cctx = NULL;

static ZSTD_DCtx* get_dctx(void) {
    if (!tls_dctx) tls_dctx = ZSTD_createDCtx();
    return tls_dctx;
}

static ZSTD_CCtx* get_cctx(void) {
    if (!tls_cctx) tls_cctx = ZSTD_createCCtx();
    return tls_cctx;
}

void carquet_zstd_cleanup(void) {
    if (tls_dctx) { ZSTD_freeDCtx(tls_dctx); tls_dctx = NULL; }
    if (tls_cctx) { ZSTD_freeCCtx(tls_cctx); tls_cctx = NULL; }
}

#endif

int carquet_zstd_decompress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size) {

    if (!src || !dst || !dst_size) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    /* Use streaming context for better buffer reuse */
    ZSTD_DCtx* dctx = get_dctx();
    if (!dctx) {
        /* Fallback to simple API */
        size_t result = ZSTD_decompress(dst, dst_capacity, src, src_size);
        if (ZSTD_isError(result)) {
            return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
        }
        *dst_size = result;
        return CARQUET_OK;
    }

    size_t result = ZSTD_decompressDCtx(dctx, dst, dst_capacity, src, src_size);
    if (ZSTD_isError(result)) {
        return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
    }

    *dst_size = result;
    return CARQUET_OK;
}

int carquet_zstd_compress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size,
    int level) {

    if (!src || !dst || !dst_size) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    if (level < 1) level = 1;
    if (level > ZSTD_maxCLevel()) level = ZSTD_maxCLevel();

    /* Use cached context for repeated compressions (e.g., per-page).
     * Only enable multi-threading for large inputs (>4MB) where the
     * parallelism overhead is worthwhile. */
    ZSTD_CCtx* cctx = get_cctx();
    if (cctx) {
        ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level);

        /* Only use multi-threading for large inputs where parallelism
         * outweighs coordination overhead. For typical 1MB pages, single-
         * threaded with a cached context is faster. */
        if (src_size > 4 * 1024 * 1024) {
            ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, 4);
        } else {
            ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, 0);
        }

        size_t result = ZSTD_compress2(cctx, dst, dst_capacity, src, src_size);
        if (!ZSTD_isError(result)) {
            *dst_size = result;
            return CARQUET_OK;
        }
    }

    /* Fallback to simple API */
    size_t result = ZSTD_compress(dst, dst_capacity, src, src_size, level);
    if (ZSTD_isError(result)) {
        return CARQUET_ERROR_COMPRESSION;
    }

    *dst_size = result;
    return CARQUET_OK;
}

size_t carquet_zstd_compress_bound(size_t src_size) {
    return ZSTD_compressBound(src_size);
}

void carquet_zstd_init_tables(void) {
    /* No-op - libzstd handles initialization internally */
}
