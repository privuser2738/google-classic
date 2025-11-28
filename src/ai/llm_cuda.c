/*
 * Holo - LLM CUDA Integration
 * Full GPU-accelerated inference with model weights resident on GPU
 */

#include "llm.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#ifdef HOLO_USE_CUDA
#include "cuda_ops.h"
#endif

/* External declarations for llm.c functions and globals */
extern int g_emb_type;
extern int g_weight_type;
extern int g_v_weight_type;  /* V projection weight type (may differ from Q/K) */
extern int g_w1_weight_type; /* FFN gate weight type */
extern int g_w2_weight_type; /* FFN down weight type */
extern int g_w3_weight_type; /* FFN up weight type */
extern int g_output_type;
extern int g_emb_vocab_size;  /* Actual embedding vocab size */

extern void matmul_q(float *dst, const void *M, const float *v,
                     int out_dim, int in_dim, int quant_type);
extern void get_embedding(float *dst, const void *embeddings, int token,
                          int dim, int vocab_size, int quant_type);

/* ============================================================================
 * GPU Layer Weights Structure
 * ============================================================================ */

typedef struct {
    void *d_attn_norm;      /* Attention RMS norm [dim] */
    void *d_wq;             /* Query projection [dim, dim] */
    void *d_wk;             /* Key projection [kv_dim, dim] */
    void *d_wv;             /* Value projection [kv_dim, dim] */
    void *d_wo;             /* Output projection [dim, dim] */
    float *d_bq;            /* Query bias [dim] (Qwen) */
    float *d_bk;            /* Key bias [kv_dim] (Qwen) */
    float *d_bv;            /* Value bias [kv_dim] (Qwen) */
    float *d_attn_q_norm;   /* Query norm weight [head_dim] (Gemma3 QK-norm) */
    float *d_attn_k_norm;   /* Key norm weight [head_dim] (Gemma3 QK-norm) */
    void *d_post_attn_norm; /* Post-attention RMS norm [dim] (Gemma3) */
    void *d_ffn_norm;       /* FFN RMS norm [dim] */
    void *d_w1;             /* FFN gate [ffn_dim, dim] */
    void *d_w2;             /* FFN down [dim, ffn_dim] */
    void *d_w3;             /* FFN up [ffn_dim, dim] */
    int wv_type;            /* Per-layer wv quantization type */
    int w2_type;            /* Per-layer w2 quantization type */
    void *d_post_ffn_norm;  /* Post-FFN RMS norm [dim] (Gemma3) */
} gpu_layer_weights_t;

/* ============================================================================
 * CUDA Backend State
 * ============================================================================ */

typedef struct {
    bool initialized;
    bool available;
    bool weights_loaded;    /* True if model weights are on GPU */

    /* Activation buffers on GPU */
    float *d_x;             /* Current activation [dim] */
    float *d_xb;            /* Norm buffer [dim] */
    float *d_xb2;           /* Second buffer [dim] */
    float *d_hb;            /* FFN hidden [ffn_dim] */
    float *d_hb2;           /* FFN hidden 2 [ffn_dim] */
    float *d_q;             /* Query [dim] */
    float *d_k;             /* Key [kv_dim] */
    float *d_v;             /* Value [kv_dim] */
    float *d_att;           /* Attention scores [n_heads * max_seq] */
    float *d_logits;        /* Output logits [vocab_size] */

    /* KV cache on GPU */
    float *d_key_cache;     /* [n_layers, max_seq, kv_dim] */
    float *d_value_cache;   /* [n_layers, max_seq, kv_dim] */

    /* Model weights on GPU (all layers) */
    void *d_tok_embeddings; /* Token embeddings [vocab_size, dim] */
    gpu_layer_weights_t *d_layers;  /* Per-layer weights */
    void *d_final_norm;     /* Final RMS norm [dim] */
    void *d_output;         /* Output projection [vocab_size, dim] */

    /* Dimensions */
    int dim;
    int ffn_dim;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int max_seq;
    int vocab_size;
    int n_layers;
    int weight_type;        /* Quantization type (Q/K) */
    int v_weight_type;      /* V projection quantization (may differ) */
    int w1_weight_type;     /* FFN gate quantization */
    int w2_weight_type;     /* FFN down quantization */
    int w3_weight_type;     /* FFN up quantization */
    int emb_type;           /* Embedding quantization */
    int output_type;        /* Output quantization */
    float rms_eps;
    float rope_freq_base;
    float rope_base_global; /* 1M for Gemma3 global layers */
    float emb_scale;        /* Embedding scale factor (sqrt(dim) for Gemma) */
    bool is_gemma;          /* Is this a Gemma model */
} llm_cuda_state_t;

static llm_cuda_state_t g_cuda = {0};

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

#ifdef HOLO_USE_CUDA

/* Calculate size in bytes for quantized tensor */
static size_t quant_size(size_t n_elem, int qtype) {
    switch (qtype) {
        case 8:  /* Q8_0: 34 bytes per 32 elements */
            return ((n_elem + 31) / 32) * 34;
        case 2:  /* Q4_0: 18 bytes per 32 elements */
            return ((n_elem + 31) / 32) * 18;
        case 12: /* Q4_K: 144 bytes per 256 elements */
            return ((n_elem + 255) / 256) * 144;
        case 14: /* Q6_K: 210 bytes per 256 elements */
            return ((n_elem + 255) / 256) * 210;
        case 1:  /* F16: 2 bytes per element */
            return n_elem * 2;
        case 0:  /* F32: 4 bytes per element */
        default:
            return n_elem * 4;
    }
}

/* GPU matmul dispatcher based on quantization type */
static void gpu_matmul(float *dst, const void *M, const float *v,
                       int out_dim, int in_dim, int qtype) {
    switch (qtype) {
        case 8:
            cuda_matmul_q8_0(dst, M, v, out_dim, in_dim);
            break;
        case 2:
            cuda_matmul_q4_0(dst, M, v, out_dim, in_dim);
            break;
        case 12:
            cuda_matmul_q4_k(dst, M, v, out_dim, in_dim);
            break;
        case 14:
            cuda_matmul_q6_k(dst, M, v, out_dim, in_dim);
            break;
        case 1:
            cuda_matmul_f16(dst, M, v, out_dim, in_dim);
            break;
        case 0:
        default:
            cuda_matmul_f32(dst, (const float*)M, v, out_dim, in_dim);
            break;
    }
}

/* GPU embedding lookup dispatcher */
static void gpu_embedding(float *dst, const void *emb, int token,
                          int dim, int vocab_size, int qtype) {
    switch (qtype) {
        case 8:  /* Q8_0 */
            cuda_get_embedding_q8_0(dst, emb, token, dim, vocab_size);
            break;
        case 2:  /* Q4_0 */
            cuda_get_embedding_q4_0(dst, emb, token, dim, vocab_size);
            break;
        case 14:  /* Q6_K */
            cuda_get_embedding_q6_k(dst, emb, token, dim, vocab_size);
            break;
        case 1:  /* F16 */
            cuda_get_embedding_f16(dst, emb, token, dim, vocab_size);
            break;
        case 0:  /* F32 */
        default:
            cuda_get_embedding_f32(dst, (const float*)emb, token, dim, vocab_size);
            break;
    }
}

/* Upload tensor data from CPU to GPU */
static void* upload_tensor(const void *cpu_ptr, size_t size) {
    if (!cpu_ptr || size == 0) return NULL;
    void *d_ptr = cuda_malloc(size);
    if (d_ptr) {
        cuda_memcpy_h2d(d_ptr, cpu_ptr, size);
    }
    return d_ptr;
}

#endif /* HOLO_USE_CUDA */

/* ============================================================================
 * Initialization and Weight Upload
 * ============================================================================ */

int llm_cuda_init(llm_ctx_t *ctx) {
#ifdef HOLO_USE_CUDA
    /* Check for HOLO_DISABLE_CUDA environment variable */
    const char *disable_cuda = getenv("HOLO_DISABLE_CUDA");
    if (disable_cuda && disable_cuda[0] != '0') {
        printf("CUDA: Disabled via HOLO_DISABLE_CUDA environment variable\n");
        g_cuda.initialized = true;
        g_cuda.available = false;
        return -1;
    }

    if (g_cuda.initialized) {
        return g_cuda.available ? 0 : -1;
    }

    /* Initialize CUDA runtime */
    if (cuda_init() != 0) {
        fprintf(stderr, "CUDA: Initialization failed, using CPU backend\n");
        g_cuda.initialized = true;
        g_cuda.available = false;
        return -1;
    }

    printf("CUDA: ");
    cuda_print_info();

    /* Store model dimensions */
    g_cuda.dim = ctx->config.embedding_dim;
    g_cuda.ffn_dim = ctx->config.ffn_dim;
    g_cuda.n_heads = ctx->config.n_heads;
    g_cuda.n_kv_heads = ctx->config.n_kv_heads;
    g_cuda.head_dim = ctx->config.head_dim;
    g_cuda.max_seq = ctx->config.context_length > 4096 ? 4096 : ctx->config.context_length;
    /* Use actual embedding vocab size if available (may differ from metadata) */
    g_cuda.vocab_size = g_emb_vocab_size > 0 ? g_emb_vocab_size : ctx->config.vocab_size;
    g_cuda.n_layers = ctx->config.n_layers;
    g_cuda.weight_type = g_weight_type;
    g_cuda.v_weight_type = g_v_weight_type;  /* V projection may use different quantization */
    g_cuda.w1_weight_type = g_w1_weight_type;  /* FFN gate weight type */
    g_cuda.w2_weight_type = g_w2_weight_type;  /* FFN down weight type */
    g_cuda.w3_weight_type = g_w3_weight_type;  /* FFN up weight type */
    g_cuda.emb_type = g_emb_type;
    g_cuda.output_type = g_output_type;
    g_cuda.rms_eps = ctx->config.rms_norm_eps;
    g_cuda.rope_freq_base = ctx->config.rope_freq_base;
    g_cuda.rope_base_global = 1000000.0f;  /* 1M for Gemma3 global layers */

    /* Check if this is a Gemma model (needs embedding scaling) */
    g_cuda.is_gemma = (strstr(ctx->config.arch, "gemma") != NULL ||
                       strstr(ctx->config.arch, "Gemma") != NULL);
    g_cuda.emb_scale = g_cuda.is_gemma ? sqrtf((float)g_cuda.dim) : 1.0f;
    if (g_cuda.is_gemma) {
        printf("[CUDA] Gemma model detected - using embedding scale %.2f, global RoPE base 1M\n", g_cuda.emb_scale);
    }

    int dim = g_cuda.dim;
    int ffn_dim = g_cuda.ffn_dim;
    int n_heads = g_cuda.n_heads;
    int n_kv_heads = g_cuda.n_kv_heads;
    int head_dim = g_cuda.head_dim;
    int max_seq = g_cuda.max_seq;
    int vocab_size = g_cuda.vocab_size;
    int n_layers = g_cuda.n_layers;
    int q_dim = n_heads * head_dim;  /* Q output dimension (may differ from dim for gemma3) */
    int kv_dim = n_kv_heads * head_dim;

    /* Check available VRAM */
    size_t free_vram = cuda_get_free_memory();

    /* Calculate total memory needed */
    size_t emb_size = quant_size((size_t)vocab_size * dim, g_emb_type);
    size_t output_size = quant_size((size_t)vocab_size * dim, g_output_type);

    size_t layer_weights = 0;
    layer_weights += dim * sizeof(float);                              /* attn_norm */
    layer_weights += quant_size((size_t)q_dim * dim, g_weight_type);    /* wq [q_dim, dim] */
    layer_weights += quant_size((size_t)kv_dim * dim, g_weight_type);  /* wk */
    layer_weights += quant_size((size_t)kv_dim * dim, g_v_weight_type); /* wv (may differ) */
    layer_weights += quant_size((size_t)dim * q_dim, g_weight_type);   /* wo [dim, q_dim] */
    layer_weights += dim * sizeof(float);                              /* post_attn_norm (Gemma3) */
    layer_weights += dim * sizeof(float);                              /* ffn_norm */
    layer_weights += quant_size((size_t)ffn_dim * dim, g_w1_weight_type); /* w1 */
    layer_weights += quant_size((size_t)dim * ffn_dim, g_w2_weight_type); /* w2 */
    layer_weights += quant_size((size_t)ffn_dim * dim, g_w3_weight_type); /* w3 */
    layer_weights += dim * sizeof(float);                              /* post_ffn_norm (Gemma3) */
    /* QKV biases (optional) */
    layer_weights += (dim + kv_dim * 2) * sizeof(float);

    size_t total_weights = emb_size + output_size + dim * sizeof(float) + layer_weights * n_layers;

    size_t kv_cache_size = 2 * (size_t)n_layers * max_seq * kv_dim * sizeof(float);
    size_t activation_size = (dim * 5 + ffn_dim * 2 + kv_dim * 2 +
                              n_heads * max_seq + vocab_size) * sizeof(float);

    size_t total_needed = total_weights + kv_cache_size + activation_size;

    printf("CUDA: Memory estimate:\n");
    printf("  - Model weights: %.2f MB\n", total_weights / (1024.0 * 1024.0));
    printf("  - KV cache: %.2f MB\n", kv_cache_size / (1024.0 * 1024.0));
    printf("  - Activations: %.2f MB\n", activation_size / (1024.0 * 1024.0));
    printf("  - Total needed: %.2f MB\n", total_needed / (1024.0 * 1024.0));
    printf("  - Available VRAM: %.2f MB\n", free_vram / (1024.0 * 1024.0));

    if (total_needed > free_vram * 0.95) {
        printf("CUDA: Insufficient VRAM for full GPU residence\n");
        printf("CUDA: Falling back to CPU inference\n");
        g_cuda.initialized = true;
        g_cuda.available = false;
        return -1;
    }

    /* Allocate activation buffers - d_q and d_xb2 use q_dim (may differ from dim) */
    g_cuda.d_x = (float*)cuda_malloc(dim * sizeof(float));
    g_cuda.d_xb = (float*)cuda_malloc(dim * sizeof(float));
    g_cuda.d_xb2 = (float*)cuda_malloc(q_dim * sizeof(float));  /* MHA output is q_dim */
    g_cuda.d_hb = (float*)cuda_malloc(ffn_dim * sizeof(float));
    g_cuda.d_hb2 = (float*)cuda_malloc(ffn_dim * sizeof(float));
    g_cuda.d_q = (float*)cuda_malloc(q_dim * sizeof(float));    /* Q output is q_dim */
    g_cuda.d_k = (float*)cuda_malloc(kv_dim * sizeof(float));
    g_cuda.d_v = (float*)cuda_malloc(kv_dim * sizeof(float));
    g_cuda.d_att = (float*)cuda_malloc(n_heads * max_seq * sizeof(float));
    g_cuda.d_logits = (float*)cuda_malloc(vocab_size * sizeof(float));

    /* Allocate KV cache */
    size_t kv_size = (size_t)n_layers * max_seq * kv_dim * sizeof(float);
    g_cuda.d_key_cache = (float*)cuda_malloc(kv_size);
    g_cuda.d_value_cache = (float*)cuda_malloc(kv_size);

    if (!g_cuda.d_x || !g_cuda.d_xb || !g_cuda.d_xb2 || !g_cuda.d_hb ||
        !g_cuda.d_hb2 || !g_cuda.d_q || !g_cuda.d_k || !g_cuda.d_v ||
        !g_cuda.d_att || !g_cuda.d_logits || !g_cuda.d_key_cache ||
        !g_cuda.d_value_cache) {
        fprintf(stderr, "CUDA: Failed to allocate activation buffers\n");
        llm_cuda_cleanup();
        g_cuda.available = false;
        return -1;
    }

    cuda_memset(g_cuda.d_key_cache, 0, kv_size);
    cuda_memset(g_cuda.d_value_cache, 0, kv_size);

    /* Upload model weights to GPU */
    printf("CUDA: Uploading model weights to GPU...\n");

    llm_weights_t *w = &ctx->weights;

    /* Token embeddings */
    g_cuda.d_tok_embeddings = upload_tensor(w->tok_embeddings, emb_size);
    if (!g_cuda.d_tok_embeddings) {
        fprintf(stderr, "CUDA: Failed to upload token embeddings\n");
        llm_cuda_cleanup();
        return -1;
    }

    /* Allocate layer weight array */
    g_cuda.d_layers = (gpu_layer_weights_t*)calloc(n_layers, sizeof(gpu_layer_weights_t));
    if (!g_cuda.d_layers) {
        fprintf(stderr, "CUDA: Failed to allocate layer array\n");
        llm_cuda_cleanup();
        return -1;
    }

    /* Upload each layer's weights */
    for (int l = 0; l < n_layers; l++) {
        gpu_layer_weights_t *dl = &g_cuda.d_layers[l];

        /* Attention norm (F32) */
        dl->d_attn_norm = upload_tensor(w->layers[l].attn_norm, dim * sizeof(float));

        /* QKV weights - Wq [q_dim, dim], Wk/Wv [kv_dim, dim], Wo [dim, q_dim] */
        /* Use per-layer V weight type since it varies in mixed-quant models */
        int layer_wv_type = w->layers[l].wv_type ? w->layers[l].wv_type : g_v_weight_type;
        dl->d_wq = upload_tensor(w->layers[l].wq, quant_size((size_t)q_dim * dim, g_weight_type));
        dl->d_wk = upload_tensor(w->layers[l].wk, quant_size((size_t)kv_dim * dim, g_weight_type));
        dl->d_wv = upload_tensor(w->layers[l].wv, quant_size((size_t)kv_dim * dim, layer_wv_type));
        dl->d_wo = upload_tensor(w->layers[l].wo, quant_size((size_t)dim * q_dim, g_weight_type));
        dl->wv_type = layer_wv_type;  /* Store for inference */

        /* DEBUG disabled for production */

        /* QKV biases (Qwen) */
        if (w->layers[l].bq) {
            dl->d_bq = (float*)upload_tensor(w->layers[l].bq, dim * sizeof(float));
        }
        if (w->layers[l].bk) {
            dl->d_bk = (float*)upload_tensor(w->layers[l].bk, kv_dim * sizeof(float));
        }
        if (w->layers[l].bv) {
            dl->d_bv = (float*)upload_tensor(w->layers[l].bv, kv_dim * sizeof(float));
        }

        /* QK-norm weights (Gemma3) - F32, size is [head_dim] per head */
        if (w->layers[l].attn_q_norm) {
            dl->d_attn_q_norm = (float*)upload_tensor(w->layers[l].attn_q_norm, head_dim * sizeof(float));
        }
        if (w->layers[l].attn_k_norm) {
            dl->d_attn_k_norm = (float*)upload_tensor(w->layers[l].attn_k_norm, head_dim * sizeof(float));
        }

        /* Post-attention norm (Gemma3) - F32 */
        if (w->layers[l].post_attn_norm) {
            dl->d_post_attn_norm = upload_tensor(w->layers[l].post_attn_norm, dim * sizeof(float));
        }

        /* FFN norm (F32) */
        dl->d_ffn_norm = upload_tensor(w->layers[l].ffn_norm, dim * sizeof(float));

        /* FFN weights (may have different quantization types) */
        dl->d_w1 = upload_tensor(w->layers[l].w1, quant_size((size_t)ffn_dim * dim, g_w1_weight_type));
        /* w2 uses per-layer type since it varies between layers in mixed-quant models */
        int layer_w2_type = w->layers[l].w2_type ? w->layers[l].w2_type : g_w2_weight_type;
        dl->d_w2 = upload_tensor(w->layers[l].w2, quant_size((size_t)dim * ffn_dim, layer_w2_type));
        dl->w2_type = layer_w2_type;  /* Store for inference */
        dl->d_w3 = upload_tensor(w->layers[l].w3, quant_size((size_t)ffn_dim * dim, g_w3_weight_type));

        /* Post-FFN norm (Gemma3) - F32 */
        if (w->layers[l].post_ffn_norm) {
            dl->d_post_ffn_norm = upload_tensor(w->layers[l].post_ffn_norm, dim * sizeof(float));
        }

        if (!dl->d_wq || !dl->d_wk || !dl->d_wv || !dl->d_wo ||
            !dl->d_w1 || !dl->d_w2 || !dl->d_w3) {
            fprintf(stderr, "CUDA: Failed to upload layer %d weights\n", l);
            llm_cuda_cleanup();
            return -1;
        }
    }

    /* Final norm and output projection */
    g_cuda.d_final_norm = upload_tensor(w->norm, dim * sizeof(float));
    g_cuda.d_output = upload_tensor(w->output, output_size);

    if (!g_cuda.d_output) {
        fprintf(stderr, "CUDA: Failed to upload output weights\n");
        llm_cuda_cleanup();
        return -1;
    }

    cuda_sync();

    g_cuda.initialized = true;
    g_cuda.available = true;
    g_cuda.weights_loaded = true;

    size_t used = cuda_get_free_memory();
    printf("CUDA: Model loaded to GPU (%.2f MB used)\n",
           (free_vram - used) / (1024.0 * 1024.0));

    return 0;
#else
    (void)ctx;
    return -1;
#endif
}

void llm_cuda_cleanup(void) {
#ifdef HOLO_USE_CUDA
    if (!g_cuda.initialized) return;

    /* Free activation buffers */
    cuda_free(g_cuda.d_x);
    cuda_free(g_cuda.d_xb);
    cuda_free(g_cuda.d_xb2);
    cuda_free(g_cuda.d_hb);
    cuda_free(g_cuda.d_hb2);
    cuda_free(g_cuda.d_q);
    cuda_free(g_cuda.d_k);
    cuda_free(g_cuda.d_v);
    cuda_free(g_cuda.d_att);
    cuda_free(g_cuda.d_logits);
    cuda_free(g_cuda.d_key_cache);
    cuda_free(g_cuda.d_value_cache);

    /* Free model weights */
    cuda_free(g_cuda.d_tok_embeddings);
    cuda_free(g_cuda.d_final_norm);
    cuda_free(g_cuda.d_output);

    if (g_cuda.d_layers) {
        for (int l = 0; l < g_cuda.n_layers; l++) {
            gpu_layer_weights_t *dl = &g_cuda.d_layers[l];
            cuda_free(dl->d_attn_norm);
            cuda_free(dl->d_wq);
            cuda_free(dl->d_wk);
            cuda_free(dl->d_wv);
            cuda_free(dl->d_wo);
            cuda_free(dl->d_bq);
            cuda_free(dl->d_bk);
            cuda_free(dl->d_bv);
            cuda_free(dl->d_post_attn_norm);
            cuda_free(dl->d_ffn_norm);
            cuda_free(dl->d_w1);
            cuda_free(dl->d_w2);
            cuda_free(dl->d_w3);
            cuda_free(dl->d_post_ffn_norm);
        }
        free(g_cuda.d_layers);
    }

    cuda_cleanup();
    memset(&g_cuda, 0, sizeof(g_cuda));
#endif
}

bool llm_cuda_available(void) {
#ifdef HOLO_USE_CUDA
    return g_cuda.initialized && g_cuda.available && g_cuda.weights_loaded;
#else
    return false;
#endif
}

void llm_cuda_reset(void) {
#ifdef HOLO_USE_CUDA
    if (!llm_cuda_available()) return;

    size_t kv_size = (size_t)g_cuda.n_layers * g_cuda.max_seq *
                     g_cuda.n_kv_heads * g_cuda.head_dim * sizeof(float);
    cuda_memset(g_cuda.d_key_cache, 0, kv_size);
    cuda_memset(g_cuda.d_value_cache, 0, kv_size);
#endif
}

/* ============================================================================
 * Full GPU Forward Pass
 * ============================================================================ */

#ifdef HOLO_USE_CUDA

/* Simple timing helper */
#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
#include <sys/time.h>
static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#endif

/* Per-token timing */
static int g_profile_enabled = 0;  /* Set to 1 to enable profiling (adds debug memcpy which slows things) */
static int g_timing_enabled = 1;   /* Set to 1 to enable timing output */
static double g_total_layer_time = 0;
static double g_total_output_time = 0;
static int g_profile_tokens = 0;

int llm_forward_cuda(llm_ctx_t *ctx, const int *tokens, int n_tokens) {
    if (!llm_cuda_available()) return -1;
    if (!ctx || !ctx->loaded) return -1;

    int dim = g_cuda.dim;
    int ffn_dim = g_cuda.ffn_dim;
    int n_heads = g_cuda.n_heads;
    int n_kv_heads = g_cuda.n_kv_heads;
    int head_dim = g_cuda.head_dim;
    int n_layers = g_cuda.n_layers;
    int vocab_size = g_cuda.vocab_size;
    int q_dim = n_heads * head_dim;      /* Q output dimension (may differ from dim for gemma3) */
    int kv_dim = n_kv_heads * head_dim;
    size_t kv_layer_size = (size_t)g_cuda.max_seq * kv_dim;

    /* Process each token */
    for (int t = 0; t < n_tokens; t++) {
        int token = tokens[t];
        int pos = ctx->pos;

        if (pos >= g_cuda.max_seq) {
            fprintf(stderr, "CUDA: Position %d exceeds max sequence length %d\n",
                    pos, g_cuda.max_seq);
            return -1;
        }

        double t_start = get_time_ms();

        /* Get token embedding on GPU */
        gpu_embedding(g_cuda.d_x, g_cuda.d_tok_embeddings, token,
                      dim, vocab_size, g_cuda.emb_type);

        /* Apply embedding scaling for Gemma models (multiply by sqrt(dim)) */
        if (g_cuda.emb_scale != 1.0f) {
            cuda_vec_scale(g_cuda.d_x, g_cuda.d_x, g_cuda.emb_scale, dim);
        }

        /* DEBUG: Check embedding (only on first token) */
        if (t == 0 && g_profile_enabled) {
            float debug_buf[16];
            cuda_memcpy_d2h(debug_buf, g_cuda.d_x, 16 * sizeof(float));
            printf("[DEBUG] After embedding x[0..3]: %g %g %g %g (token=%d, type=%d)\n",
                   debug_buf[0], debug_buf[1], debug_buf[2], debug_buf[3], token, g_cuda.emb_type);
        }

        /* Process through transformer layers */
        for (int l = 0; l < n_layers; l++) {
            gpu_layer_weights_t *dl = &g_cuda.d_layers[l];

            /* ----- Attention Block ----- */

            /* RMS norm: xb = norm(x) */
            if (dl->d_attn_norm) {
                cuda_rms_norm(g_cuda.d_xb, g_cuda.d_x, (float*)dl->d_attn_norm,
                              dim, g_cuda.rms_eps);
            } else {
                cuda_memcpy_d2d(g_cuda.d_xb, g_cuda.d_x, dim * sizeof(float));
            }

            /* DEBUG: Check after RMS norm (layer 0 or 7, first token) */
            if (t == 0 && (l == 0 || l == 7) && g_profile_enabled) {
                float debug_buf[4];
                cuda_memcpy_d2h(debug_buf, g_cuda.d_xb, 4 * sizeof(float));
                printf("[DEBUG] L%d after RMS norm xb[0..3]: %g %g %g %g\n",
                       l, debug_buf[0], debug_buf[1], debug_buf[2], debug_buf[3]);
            }

            /* QKV projections - Q output is n_heads*head_dim (may differ from dim for gemma3) */
            /* Use per-layer V weight type since it varies in mixed-quant models */
            gpu_matmul(g_cuda.d_q, dl->d_wq, g_cuda.d_xb, q_dim, dim, g_cuda.weight_type);
            gpu_matmul(g_cuda.d_k, dl->d_wk, g_cuda.d_xb, kv_dim, dim, g_cuda.weight_type);
            gpu_matmul(g_cuda.d_v, dl->d_wv, g_cuda.d_xb, kv_dim, dim, dl->wv_type);

            /* QK-norm (Gemma3) - apply RMS norm to each head's Q and K vectors */
            if (dl->d_attn_q_norm) {
                for (int h = 0; h < n_heads; h++) {
                    cuda_rms_norm(g_cuda.d_q + h * head_dim, g_cuda.d_q + h * head_dim,
                                  dl->d_attn_q_norm, head_dim, g_cuda.rms_eps);
                }
            }
            if (dl->d_attn_k_norm) {
                for (int h = 0; h < n_kv_heads; h++) {
                    cuda_rms_norm(g_cuda.d_k + h * head_dim, g_cuda.d_k + h * head_dim,
                                  dl->d_attn_k_norm, head_dim, g_cuda.rms_eps);
                }
            }

            /* DEBUG: Check after QKV (layer 0 or 7, first token) */
            if (t == 0 && (l == 0 || l == 7) && g_profile_enabled) {
                printf("[PTR DEBUG] d_q=%p d_k=%p d_v=%p d_wq=%p d_wk=%p d_wv=%p\n",
                       (void*)g_cuda.d_q, (void*)g_cuda.d_k, (void*)g_cuda.d_v,
                       (void*)dl->d_wq, (void*)dl->d_wk, (void*)dl->d_wv);
                float debug_buf[4];
                cuda_memcpy_d2h(debug_buf, g_cuda.d_q, 4 * sizeof(float));
                printf("[DEBUG] L%d after Wq q[0..3]: %g %g %g %g (weight_type=%d)\n",
                       l, debug_buf[0], debug_buf[1], debug_buf[2], debug_buf[3], g_cuda.weight_type);
                /* Check head 6 and 7 of Q (uses head_dim based offsets) */
                int head6_offset = 6 * head_dim;
                int head7_offset = 7 * head_dim;
                cuda_memcpy_d2h(debug_buf, g_cuda.d_q + head6_offset, 4 * sizeof(float));
                printf("[DEBUG] L%d after Wq q[%d..%d] (head6): %g %g %g %g\n",
                       l, head6_offset, head6_offset+3, debug_buf[0], debug_buf[1], debug_buf[2], debug_buf[3]);
                cuda_memcpy_d2h(debug_buf, g_cuda.d_q + head7_offset, 4 * sizeof(float));
                printf("[DEBUG] L%d after Wq q[%d..%d] (head7): %g %g %g %g\n",
                       l, head7_offset, head7_offset+3, debug_buf[0], debug_buf[1], debug_buf[2], debug_buf[3]);
                cuda_memcpy_d2h(debug_buf, g_cuda.d_k, 4 * sizeof(float));
                printf("[DEBUG] L%d after Wk k[0..3]: %g %g %g %g\n",
                       l, debug_buf[0], debug_buf[1], debug_buf[2], debug_buf[3]);
                /* Check kv_head=3 of K (uses head_dim based offset) */
                int kv_head3_offset = 3 * head_dim;
                cuda_memcpy_d2h(debug_buf, g_cuda.d_k + kv_head3_offset, 4 * sizeof(float));
                printf("[DEBUG] L%d after Wk k[%d..%d] (kv_head3): %g %g %g %g\n",
                       l, kv_head3_offset, kv_head3_offset+3, debug_buf[0], debug_buf[1], debug_buf[2], debug_buf[3]);
                cuda_memcpy_d2h(debug_buf, g_cuda.d_v, 4 * sizeof(float));
                printf("[DEBUG] L%d after Wv v[0..3]: %g %g %g %g\n",
                       l, debug_buf[0], debug_buf[1], debug_buf[2], debug_buf[3]);
            }

            /* Add QKV biases if present (Qwen models) */
            if (dl->d_bq) {
                cuda_vec_add(g_cuda.d_q, g_cuda.d_q, dl->d_bq, q_dim);
            }
            if (dl->d_bk) {
                cuda_vec_add(g_cuda.d_k, g_cuda.d_k, dl->d_bk, kv_dim);
            }
            if (dl->d_bv) {
                cuda_vec_add(g_cuda.d_v, g_cuda.d_v, dl->d_bv, kv_dim);
            }

            /* Apply RoPE to Q and K
             * Gemma3: Global layers (L5, L11, L17...) use 1M base, local use 10K */
            bool is_global_layer = g_cuda.is_gemma && ((l + 1) % 6 == 0);
            float rope_base = is_global_layer ? g_cuda.rope_base_global : g_cuda.rope_freq_base;
            cuda_rope_apply(g_cuda.d_q, g_cuda.d_k, pos, head_dim,
                            n_heads, n_kv_heads, rope_base);

            /* DEBUG: Check after RoPE (layer 0, first token) */
            if (t == 0 && l == 0 && g_profile_enabled) {
                float debug_buf[4];
                cuda_memcpy_d2h(debug_buf, g_cuda.d_q, 4 * sizeof(float));
                printf("[DEBUG] L%d after RoPE q[0..3]: %g %g %g %g\n",
                       l, debug_buf[0], debug_buf[1], debug_buf[2], debug_buf[3]);
                cuda_memcpy_d2h(debug_buf, g_cuda.d_k, 4 * sizeof(float));
                printf("[DEBUG] L%d after RoPE k[0..3]: %g %g %g %g\n",
                       l, debug_buf[0], debug_buf[1], debug_buf[2], debug_buf[3]);
                cuda_memcpy_d2h(debug_buf, g_cuda.d_v, 4 * sizeof(float));
                printf("[DEBUG] L%d v[0..3]: %g %g %g %g\n",
                       l, debug_buf[0], debug_buf[1], debug_buf[2], debug_buf[3]);
            }

            /* Store K and V in cache (sync not needed - same stream as attention) */
            float *d_k_cache = g_cuda.d_key_cache + l * kv_layer_size + pos * kv_dim;
            float *d_v_cache = g_cuda.d_value_cache + l * kv_layer_size + pos * kv_dim;
            cuda_memcpy_d2d(d_k_cache, g_cuda.d_k, kv_dim * sizeof(float));
            cuda_memcpy_d2d(d_v_cache, g_cuda.d_v, kv_dim * sizeof(float));

            /* Multi-head attention - all heads computed in ONE kernel launch */
            float *d_k_layer = g_cuda.d_key_cache + l * kv_layer_size;
            float *d_v_layer = g_cuda.d_value_cache + l * kv_layer_size;

            cuda_multi_head_attention(g_cuda.d_xb2, g_cuda.d_q,
                                      d_k_layer, d_v_layer,
                                      pos + 1, head_dim, n_heads, n_kv_heads);

            /* DEBUG: Check after attention (layer 0 or 7, first token) */
            if (t == 0 && (l == 0 || l == 7) && g_profile_enabled) {
                float debug_buf[4];
                cuda_memcpy_d2h(debug_buf, g_cuda.d_xb2, 4 * sizeof(float));
                printf("[DEBUG] L%d after MHA xb2[0..3]: %g %g %g %g\n",
                       l, debug_buf[0], debug_buf[1], debug_buf[2], debug_buf[3]);
                /* Check if there are any NaN in MHA output (uses q_dim, not dim) */
                float *full_xb2 = (float*)malloc(q_dim * sizeof(float));
                cuda_memcpy_d2h(full_xb2, g_cuda.d_xb2, q_dim * sizeof(float));
                int nan_count = 0;
                int first_nan = -1;
                for (int i = 0; i < q_dim; i++) {
                    if (full_xb2[i] != full_xb2[i]) { /* NaN check */
                        nan_count++;
                        if (first_nan < 0) first_nan = i;
                    }
                }
                if (nan_count > 0) {
                    printf("[DEBUG] L%d MHA OUTPUT HAS %d NaN values! First NaN at idx %d\n", l, nan_count, first_nan);
                } else {
                    printf("[DEBUG] L%d MHA OUTPUT: all %d values valid (no NaN)\n", l, q_dim);
                }
                free(full_xb2);
            }

            /* Output projection - MHA output is q_dim, project back to dim */
            gpu_matmul(g_cuda.d_xb, dl->d_wo, g_cuda.d_xb2, dim, q_dim, g_cuda.weight_type);

            /* DEBUG: Check after attention output projection */
            if (t == 0 && (l == 0 || l == 4 || l == 6 || l == 7) && g_profile_enabled) {
                float debug_buf[4];
                cuda_memcpy_d2h(debug_buf, g_cuda.d_xb, 4 * sizeof(float));
                printf("[DEBUG] L%d after attn out xb[0..3]: %g %g %g %g\n",
                       l, debug_buf[0], debug_buf[1], debug_buf[2], debug_buf[3]);
            }

            /* Post-attention normalization (Gemma3) - applied BEFORE residual */
            if (dl->d_post_attn_norm) {
                cuda_rms_norm(g_cuda.d_xb, g_cuda.d_xb, (float*)dl->d_post_attn_norm,
                              dim, g_cuda.rms_eps);
                if (t == 0 && (l == 0 || l == 4 || l == 6 || l == 7) && g_profile_enabled) {
                    float debug_buf[4];
                    cuda_memcpy_d2h(debug_buf, g_cuda.d_xb, 4 * sizeof(float));
                    printf("[DEBUG] L%d after post_attn_norm xb[0..3]: %g %g %g %g\n",
                           l, debug_buf[0], debug_buf[1], debug_buf[2], debug_buf[3]);
                }
            }

            /* Residual connection */
            cuda_vec_add(g_cuda.d_x, g_cuda.d_x, g_cuda.d_xb, dim);

            /* ----- FFN Block ----- */

            /* RMS norm */
            if (dl->d_ffn_norm) {
                cuda_rms_norm(g_cuda.d_xb, g_cuda.d_x, (float*)dl->d_ffn_norm,
                              dim, g_cuda.rms_eps);
            } else {
                cuda_memcpy_d2d(g_cuda.d_xb, g_cuda.d_x, dim * sizeof(float));
            }

            /* DEBUG: Check FFN input */
            if (t == 0 && (l == 0 || l == 4 || l == 6 || l == 7) && g_profile_enabled) {
                float debug_buf[4];
                cuda_memcpy_d2h(debug_buf, g_cuda.d_xb, 4 * sizeof(float));
                printf("[DEBUG] L%d FFN input xb[0..3]: %g %g %g %g\n",
                       l, debug_buf[0], debug_buf[1], debug_buf[2], debug_buf[3]);
                if (l == 0) {
                    printf("[DEBUG] FFN weight types: w1=%d w2=%d w3=%d\n",
                           g_cuda.w1_weight_type, g_cuda.w2_weight_type, g_cuda.w3_weight_type);
                }
            }

            /* SwiGLU FFN: hb = SiLU(xb @ w1) * (xb @ w3), then xb = hb @ w2 */
            gpu_matmul(g_cuda.d_hb, dl->d_w1, g_cuda.d_xb, ffn_dim, dim, g_cuda.w1_weight_type);

            /* DEBUG: Check after w1 gate projection */
            if (t == 0 && (l == 0 || l == 4 || l == 6 || l == 7) && g_profile_enabled) {
                float debug_buf[4];
                cuda_memcpy_d2h(debug_buf, g_cuda.d_hb, 4 * sizeof(float));
                printf("[DEBUG] L%d after w1 (gate) hb[0..3]: %g %g %g %g\n",
                       l, debug_buf[0], debug_buf[1], debug_buf[2], debug_buf[3]);
            }

            gpu_matmul(g_cuda.d_hb2, dl->d_w3, g_cuda.d_xb, ffn_dim, dim, g_cuda.w3_weight_type);

            /* DEBUG: Check after w3 up projection */
            if (t == 0 && (l == 0 || l == 4 || l == 6 || l == 7) && g_profile_enabled) {
                float debug_buf[4];
                cuda_memcpy_d2h(debug_buf, g_cuda.d_hb2, 4 * sizeof(float));
                printf("[DEBUG] L%d after w3 (up) hb2[0..3]: %g %g %g %g\n",
                       l, debug_buf[0], debug_buf[1], debug_buf[2], debug_buf[3]);
            }

            cuda_silu_mul(g_cuda.d_hb, g_cuda.d_hb, g_cuda.d_hb2, ffn_dim);

            /* DEBUG: Check after SiLU*mul */
            if (t == 0 && (l == 0 || l == 4 || l == 6 || l == 7) && g_profile_enabled) {
                float debug_buf[4];
                cuda_memcpy_d2h(debug_buf, g_cuda.d_hb, 4 * sizeof(float));
                printf("[DEBUG] L%d after silu_mul hb[0..3]: %g %g %g %g\n",
                       l, debug_buf[0], debug_buf[1], debug_buf[2], debug_buf[3]);
            }

            /* Down projection - use per-layer w2 type since it varies in mixed-quant models */
            gpu_matmul(g_cuda.d_xb, dl->d_w2, g_cuda.d_hb, dim, ffn_dim, dl->w2_type);

            /* DEBUG: Check after FFN for layers with explosion issues (first token) */
            if (t == 0 && (l == 0 || l == 4 || l == 7) && g_profile_enabled) {
                float debug_buf[4];
                cuda_memcpy_d2h(debug_buf, g_cuda.d_xb, 4 * sizeof(float));
                printf("[DEBUG] L%d after FFN down xb[0..3]: %g %g %g %g\n",
                       l, debug_buf[0], debug_buf[1], debug_buf[2], debug_buf[3]);
                /* Check if values are exploding (abs > 1e6) */
                if ((debug_buf[0] > 1e6f || debug_buf[0] < -1e6f) ||
                    (debug_buf[1] > 1e6f || debug_buf[1] < -1e6f)) {
                    printf("[DEBUG] L%d FFN down EXPLOSION! Checking sum of hb...\n", l);
                    float *hb_check = (float*)malloc(ffn_dim * sizeof(float));
                    cuda_memcpy_d2h(hb_check, g_cuda.d_hb, ffn_dim * sizeof(float));
                    float sum = 0, maxv = 0;
                    for (int i = 0; i < ffn_dim; i++) {
                        float v = hb_check[i] > 0 ? hb_check[i] : -hb_check[i];
                        sum += v;
                        if (v > maxv) maxv = v;
                    }
                    printf("[DEBUG] L%d hb: sum=%.3g max=%.3g mean=%.3g\n",
                           l, sum, maxv, sum / ffn_dim);
                    free(hb_check);
                }
            }

            /* Post-FFN normalization (Gemma3) - applied BEFORE residual */
            if (dl->d_post_ffn_norm) {
                cuda_rms_norm(g_cuda.d_xb, g_cuda.d_xb, (float*)dl->d_post_ffn_norm,
                              dim, g_cuda.rms_eps);
                if (t == 0 && (l == 0 || l == 4 || l == 6 || l == 7) && g_profile_enabled) {
                    float debug_buf[4];
                    cuda_memcpy_d2h(debug_buf, g_cuda.d_xb, 4 * sizeof(float));
                    printf("[DEBUG] L%d after post_ffn_norm xb[0..3]: %g %g %g %g\n",
                           l, debug_buf[0], debug_buf[1], debug_buf[2], debug_buf[3]);
                }
            }

            /* Residual connection */
            cuda_vec_add(g_cuda.d_x, g_cuda.d_x, g_cuda.d_xb, dim);

            /* DEBUG: Check after residual (layer 0, first token) */
            if (t == 0 && l == 0 && g_profile_enabled) {
                float debug_buf[4];
                cuda_memcpy_d2h(debug_buf, g_cuda.d_x, 4 * sizeof(float));
                printf("[DEBUG] L%d after residual x[0..3]: %g %g %g %g\n",
                       l, debug_buf[0], debug_buf[1], debug_buf[2], debug_buf[3]);
            }

            /* DEBUG: Track layers and find NaN */
            if (t == 0 && g_profile_enabled) {
                float debug_val;
                cuda_memcpy_d2h(&debug_val, g_cuda.d_x, sizeof(float));
                if (l <= 5) {
                    float debug_buf[4];
                    cuda_memcpy_d2h(debug_buf, g_cuda.d_x, 4 * sizeof(float));
                    printf("[DEBUG] L%d end x[0..3]: %g %g %g %g\n",
                           l, debug_buf[0], debug_buf[1], debug_buf[2], debug_buf[3]);
                }
                /* Find which layer introduces NaN */
                if (debug_val != debug_val) {
                    printf("[DEBUG] NaN first appears at layer %d!\n", l);
                }
            }
        }

        double t_after_layers = get_time_ms();

        /* DEBUG: Check for NaN after layers (only on first token) */
        if (t == 0 && g_profile_enabled) {
            float debug_buf[16];
            cuda_memcpy_d2h(debug_buf, g_cuda.d_x, 16 * sizeof(float));
            printf("[DEBUG] After layers x[0..3]: %g %g %g %g\n",
                   debug_buf[0], debug_buf[1], debug_buf[2], debug_buf[3]);
        }

        /* Final RMS norm */
        if (g_cuda.d_final_norm) {
            cuda_rms_norm(g_cuda.d_x, g_cuda.d_x, (float*)g_cuda.d_final_norm,
                          dim, g_cuda.rms_eps);
        }

        /* DEBUG: Check for NaN after final norm (only on first token) */
        if (t == 0 && g_profile_enabled) {
            float debug_buf[16];
            cuda_memcpy_d2h(debug_buf, g_cuda.d_x, 16 * sizeof(float));
            printf("[DEBUG] After final norm x[0..3]: %g %g %g %g\n",
                   debug_buf[0], debug_buf[1], debug_buf[2], debug_buf[3]);
        }

        /* Output projection to logits */
        gpu_matmul(g_cuda.d_logits, g_cuda.d_output, g_cuda.d_x,
                   vocab_size, dim, g_cuda.output_type);

        /* Synchronize before timing D2H */
        cudaDeviceSynchronize();
        double t_after_matmul = get_time_ms();

        /* Copy logits back to CPU for sampling (this is synchronous) */
        cuda_memcpy_d2h(ctx->state.logits, g_cuda.d_logits, vocab_size * sizeof(float));

        double t_end = get_time_ms();  /* d2h is sync, so this measures actual time */

        /* Profile first few tokens - now shows matmul vs D2H separately */
        if ((g_profile_enabled || g_timing_enabled) && g_profile_tokens < 5) {
            printf("[PERF] Token %d: layers=%.1fms matmul=%.1fms d2h=%.1fms total=%.1fms\n",
                   g_profile_tokens, t_after_layers - t_start,
                   t_after_matmul - t_after_layers, t_end - t_after_matmul,
                   t_end - t_start);
            g_profile_tokens++;
            if (g_profile_tokens == 5) {
                printf("[PERF] Timing disabled after 5 tokens\n");
            }
        }

        ctx->pos++;
    }

    return 0;
}

#endif /* HOLO_USE_CUDA */

/* ============================================================================
 * Auto-dispatch wrapper
 * ============================================================================ */

int llm_forward_auto(llm_ctx_t *ctx, const int *tokens, int n_tokens) {
#ifdef HOLO_USE_CUDA
    if (llm_cuda_available()) {
        return llm_forward_cuda(ctx, tokens, n_tokens);
    }
#endif
    return llm_forward(ctx, tokens, n_tokens);
}
