/*
 * Holo - LLM CUDA Integration
 * Provides GPU-accelerated inference when CUDA is available
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
extern int g_output_type;

extern void matmul_q(float *dst, const void *M, const float *v,
                     int out_dim, int in_dim, int quant_type);
extern void get_embedding(float *dst, const void *embeddings, int token,
                          int dim, int vocab_size, int quant_type);

/* ============================================================================
 * CUDA Backend State
 * ============================================================================ */

typedef struct {
    bool initialized;
    bool available;

    /* Device pointers for frequently used buffers */
    float *d_x;         /* Current activation */
    float *d_xb;        /* Norm buffer */
    float *d_xb2;       /* Second buffer */
    float *d_hb;        /* FFN hidden */
    float *d_hb2;       /* FFN hidden 2 */
    float *d_q;         /* Query */
    float *d_k;         /* Key */
    float *d_v;         /* Value */
    float *d_att;       /* Attention scores */
    float *d_logits;    /* Output logits */

    /* KV cache on GPU */
    float *d_key_cache;
    float *d_value_cache;

    /* Layer weight buffers on GPU (reused per layer) */
    void *d_wq;         /* Query weight */
    void *d_wk;         /* Key weight */
    void *d_wv;         /* Value weight */
    void *d_wo;         /* Output weight */
    void *d_w1;         /* FFN gate */
    void *d_w2;         /* FFN down */
    void *d_w3;         /* FFN up */
    float *d_norm;      /* Norm weights (F32) */
    float *d_bq;        /* Query bias */
    float *d_bk;        /* Key bias */
    float *d_bv;        /* Value bias */

    /* Output projection (stays on GPU) */
    void *d_output;
    float *d_final_norm;

    /* Dimensions */
    int dim;
    int ffn_dim;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int max_seq;
    int vocab_size;
    int n_layers;
    int weight_type;    /* Quantization type */
} llm_cuda_state_t;

static llm_cuda_state_t g_cuda_state = {0};

/* ============================================================================
 * Initialization
 * ============================================================================ */

int llm_cuda_init(llm_ctx_t *ctx) {
#ifdef HOLO_USE_CUDA
    if (g_cuda_state.initialized) {
        return g_cuda_state.available ? 0 : -1;
    }

    /* Try to initialize CUDA */
    if (cuda_init() != 0) {
        fprintf(stderr, "CUDA: Initialization failed, using CPU backend\n");
        g_cuda_state.initialized = true;
        g_cuda_state.available = false;
        return -1;
    }

    printf("CUDA: ");
    cuda_print_info();

    /* Store dimensions */
    g_cuda_state.dim = ctx->config.embedding_dim;
    g_cuda_state.ffn_dim = ctx->config.ffn_dim;
    g_cuda_state.n_heads = ctx->config.n_heads;
    g_cuda_state.n_kv_heads = ctx->config.n_kv_heads;
    g_cuda_state.head_dim = ctx->config.head_dim;
    /* Cap max_seq to save VRAM - 4096 is plenty for interactive use */
    g_cuda_state.max_seq = ctx->config.context_length > 4096 ? 4096 : ctx->config.context_length;
    g_cuda_state.vocab_size = ctx->config.vocab_size;
    g_cuda_state.n_layers = ctx->config.n_layers;
    g_cuda_state.weight_type = g_weight_type;

    /* Allocate GPU buffers */
    size_t dim = g_cuda_state.dim;
    size_t ffn_dim = g_cuda_state.ffn_dim;
    size_t n_heads = g_cuda_state.n_heads;
    size_t max_seq = g_cuda_state.max_seq;
    size_t vocab_size = g_cuda_state.vocab_size;
    size_t n_kv_heads = g_cuda_state.n_kv_heads;
    size_t head_dim = g_cuda_state.head_dim;
    size_t n_layers = g_cuda_state.n_layers;
    size_t kv_dim = n_kv_heads * head_dim;

    /* Calculate weight sizes based on quantization */
    /* Q8_0: 34 bytes per 32 values */
    size_t bytes_per_elem = 34.0 / 32.0;  /* Approximate for Q8_0 */
    if (g_weight_type == 8) {  /* Q8_0 */
        bytes_per_elem = 34.0 / 32.0;
    } else if (g_weight_type == 12) {  /* Q4_K */
        bytes_per_elem = 144.0 / 256.0;
    } else if (g_weight_type == 14) {  /* Q6_K */
        bytes_per_elem = 210.0 / 256.0;
    } else {
        bytes_per_elem = 4;  /* F32 fallback */
    }

    /* Activation buffers */
    g_cuda_state.d_x = (float*)cuda_malloc(dim * sizeof(float));
    g_cuda_state.d_xb = (float*)cuda_malloc(dim * sizeof(float));
    g_cuda_state.d_xb2 = (float*)cuda_malloc(dim * sizeof(float));
    g_cuda_state.d_hb = (float*)cuda_malloc(ffn_dim * sizeof(float));
    g_cuda_state.d_hb2 = (float*)cuda_malloc(ffn_dim * sizeof(float));
    g_cuda_state.d_q = (float*)cuda_malloc(dim * sizeof(float));
    g_cuda_state.d_k = (float*)cuda_malloc(kv_dim * sizeof(float));
    g_cuda_state.d_v = (float*)cuda_malloc(kv_dim * sizeof(float));
    g_cuda_state.d_att = (float*)cuda_malloc(n_heads * max_seq * sizeof(float));
    g_cuda_state.d_logits = (float*)cuda_malloc(vocab_size * sizeof(float));

    /* KV cache on GPU */
    size_t kv_size = n_layers * max_seq * kv_dim * sizeof(float);
    g_cuda_state.d_key_cache = (float*)cuda_malloc(kv_size);
    g_cuda_state.d_value_cache = (float*)cuda_malloc(kv_size);

    /* Layer weight buffers (reused per layer) */
    size_t wq_size = (size_t)(dim * dim * bytes_per_elem);
    size_t wkv_size = (size_t)(kv_dim * dim * bytes_per_elem);
    size_t wo_size = (size_t)(dim * dim * bytes_per_elem);
    size_t w1_size = (size_t)(ffn_dim * dim * bytes_per_elem);
    size_t w2_size = (size_t)(dim * ffn_dim * bytes_per_elem);
    size_t w3_size = (size_t)(ffn_dim * dim * bytes_per_elem);

    g_cuda_state.d_wq = cuda_malloc(wq_size);
    g_cuda_state.d_wk = cuda_malloc(wkv_size);
    g_cuda_state.d_wv = cuda_malloc(wkv_size);
    g_cuda_state.d_wo = cuda_malloc(wo_size);
    g_cuda_state.d_w1 = cuda_malloc(w1_size);
    g_cuda_state.d_w2 = cuda_malloc(w2_size);
    g_cuda_state.d_w3 = cuda_malloc(w3_size);
    g_cuda_state.d_norm = (float*)cuda_malloc(dim * sizeof(float));
    g_cuda_state.d_bq = (float*)cuda_malloc(dim * sizeof(float));
    g_cuda_state.d_bk = (float*)cuda_malloc(kv_dim * sizeof(float));
    g_cuda_state.d_bv = (float*)cuda_malloc(kv_dim * sizeof(float));
    g_cuda_state.d_final_norm = (float*)cuda_malloc(dim * sizeof(float));

    /* Check allocations */
    if (!g_cuda_state.d_x || !g_cuda_state.d_xb || !g_cuda_state.d_xb2 ||
        !g_cuda_state.d_hb || !g_cuda_state.d_hb2 || !g_cuda_state.d_q ||
        !g_cuda_state.d_k || !g_cuda_state.d_v || !g_cuda_state.d_att ||
        !g_cuda_state.d_logits || !g_cuda_state.d_key_cache ||
        !g_cuda_state.d_value_cache || !g_cuda_state.d_wq || !g_cuda_state.d_wk ||
        !g_cuda_state.d_wv || !g_cuda_state.d_wo || !g_cuda_state.d_w1 ||
        !g_cuda_state.d_w2 || !g_cuda_state.d_w3 || !g_cuda_state.d_norm) {
        fprintf(stderr, "CUDA: Failed to allocate GPU memory\n");
        llm_cuda_cleanup();
        g_cuda_state.available = false;
        return -1;
    }

    /* Initialize KV cache to zero */
    cuda_memset(g_cuda_state.d_key_cache, 0, kv_size);
    cuda_memset(g_cuda_state.d_value_cache, 0, kv_size);

    g_cuda_state.initialized = true;
    g_cuda_state.available = true;

    size_t layer_weight_size = wq_size + wkv_size*2 + wo_size + w1_size + w2_size + w3_size;
    printf("CUDA: GPU memory allocated:\n");
    printf("  - Activations: %.2f MB\n",
           (dim*4 + ffn_dim*2 + kv_dim*2 + n_heads*max_seq + vocab_size) * sizeof(float) / (1024.0 * 1024.0));
    printf("  - KV cache: %.2f MB\n", kv_size * 2 / (1024.0 * 1024.0));
    printf("  - Layer weights buffer: %.2f MB\n", layer_weight_size / (1024.0 * 1024.0));

    return 0;
#else
    (void)ctx;
    return -1;
#endif
}

void llm_cuda_cleanup(void) {
#ifdef HOLO_USE_CUDA
    if (!g_cuda_state.initialized) return;

    cuda_free(g_cuda_state.d_x);
    cuda_free(g_cuda_state.d_xb);
    cuda_free(g_cuda_state.d_xb2);
    cuda_free(g_cuda_state.d_hb);
    cuda_free(g_cuda_state.d_hb2);
    cuda_free(g_cuda_state.d_q);
    cuda_free(g_cuda_state.d_k);
    cuda_free(g_cuda_state.d_v);
    cuda_free(g_cuda_state.d_att);
    cuda_free(g_cuda_state.d_logits);
    cuda_free(g_cuda_state.d_key_cache);
    cuda_free(g_cuda_state.d_value_cache);
    cuda_free(g_cuda_state.d_wq);
    cuda_free(g_cuda_state.d_wk);
    cuda_free(g_cuda_state.d_wv);
    cuda_free(g_cuda_state.d_wo);
    cuda_free(g_cuda_state.d_w1);
    cuda_free(g_cuda_state.d_w2);
    cuda_free(g_cuda_state.d_w3);
    cuda_free(g_cuda_state.d_norm);
    cuda_free(g_cuda_state.d_bq);
    cuda_free(g_cuda_state.d_bk);
    cuda_free(g_cuda_state.d_bv);
    cuda_free(g_cuda_state.d_final_norm);
    cuda_free(g_cuda_state.d_output);

    cuda_cleanup();
    memset(&g_cuda_state, 0, sizeof(g_cuda_state));
#endif
}

bool llm_cuda_available(void) {
#ifdef HOLO_USE_CUDA
    return g_cuda_state.initialized && g_cuda_state.available;
#else
    return false;
#endif
}

void llm_cuda_reset(void) {
#ifdef HOLO_USE_CUDA
    if (!llm_cuda_available()) return;

    size_t kv_size = (size_t)g_cuda_state.n_layers * g_cuda_state.max_seq *
                     g_cuda_state.n_kv_heads * g_cuda_state.head_dim * sizeof(float);
    cuda_memset(g_cuda_state.d_key_cache, 0, kv_size);
    cuda_memset(g_cuda_state.d_value_cache, 0, kv_size);
#endif
}

/* ============================================================================
 * GPU Forward Pass
 * ============================================================================ */

#ifdef HOLO_USE_CUDA

/* Helper to calculate quantized tensor size */
static size_t quant_tensor_size(int rows, int cols, int qtype) {
    size_t n_elem = (size_t)rows * cols;
    if (qtype == 8) {  /* Q8_0: 34 bytes per 32 elements */
        return (n_elem / 32) * 34;
    } else if (qtype == 12) {  /* Q4_K: 144 bytes per 256 elements */
        return (n_elem / 256) * 144;
    } else if (qtype == 14) {  /* Q6_K: 210 bytes per 256 elements */
        return (n_elem / 256) * 210;
    }
    return n_elem * 4;  /* F32 fallback */
}

/* Helper to do GPU matmul with correct quantization kernel */
static void gpu_matmul(float *dst, const void *M, const float *v,
                       int out_dim, int in_dim, int qtype) {
    if (qtype == 8) {
        cuda_matmul_q8_0(dst, M, v, out_dim, in_dim);
    } else if (qtype == 12) {
        cuda_matmul_q4_k(dst, M, v, out_dim, in_dim);
    } else if (qtype == 14) {
        cuda_matmul_q6_k(dst, M, v, out_dim, in_dim);
    } else {
        cuda_matmul_f32(dst, (const float*)M, v, out_dim, in_dim);
    }
}

/*
 * Hybrid GPU-accelerated forward pass
 * - Matmuls done on CPU (AVX2 optimized, no PCIe bottleneck)
 * - KV cache and attention on GPU (fused kernel)
 * - RoPE on GPU
 * This avoids the massive PCIe bandwidth needed for layer-wise weight upload
 */
int llm_forward_cuda(llm_ctx_t *ctx, const int *tokens, int n_tokens) {
    if (!llm_cuda_available()) return -1;
    if (!ctx || !ctx->loaded) return -1;

    llm_config_t *cfg = &ctx->config;
    llm_weights_t *w = &ctx->weights;
    llm_state_t *s = &ctx->state;

    int dim = cfg->embedding_dim;
    int n_heads = cfg->n_heads;
    int n_kv_heads = cfg->n_kv_heads;
    int head_dim = cfg->head_dim;
    int n_layers = cfg->n_layers;
    int ffn_dim = cfg->ffn_dim;

    int kv_dim = n_kv_heads * head_dim;
    size_t kv_layer_size = (size_t)g_cuda_state.max_seq * kv_dim;

    /* Process each token */
    for (int t = 0; t < n_tokens; t++) {
        int token = tokens[t];
        int pos = ctx->pos;

        if (pos >= g_cuda_state.max_seq) {
            fprintf(stderr, "CUDA: Position exceeds max sequence length\n");
            return -1;
        }

        /* Get token embedding on CPU */
        get_embedding(s->x, w->tok_embeddings, token, dim, cfg->vocab_size, g_emb_type);

        /* Process through transformer layers */
        for (int l = 0; l < n_layers; l++) {
            /* ----- Attention Block ----- */

            /* RMS norm on CPU */
            if (w->layers[l].attn_norm) {
                float *norm_w = (float *)w->layers[l].attn_norm;
                rms_norm(s->xb, s->x, norm_w, dim, cfg->rms_norm_eps);
            } else {
                memcpy(s->xb, s->x, dim * sizeof(float));
            }

            /* QKV projections on CPU */
            matmul_q(s->q, w->layers[l].wq, s->xb, dim, dim, g_weight_type);
            matmul_q(s->k, w->layers[l].wk, s->xb, kv_dim, dim, g_weight_type);
            matmul_q(s->v, w->layers[l].wv, s->xb, kv_dim, dim, g_weight_type);

            /* Add QKV biases if present (Qwen models) */
            if (w->layers[l].bq) {
                for (int i = 0; i < dim; i++) s->q[i] += w->layers[l].bq[i];
            }
            if (w->layers[l].bk) {
                for (int i = 0; i < kv_dim; i++) s->k[i] += w->layers[l].bk[i];
            }
            if (w->layers[l].bv) {
                for (int i = 0; i < kv_dim; i++) s->v[i] += w->layers[l].bv[i];
            }

            /* Copy Q, K, V to GPU for RoPE and attention */
            cuda_memcpy_h2d(g_cuda_state.d_q, s->q, dim * sizeof(float));
            cuda_memcpy_h2d(g_cuda_state.d_k, s->k, kv_dim * sizeof(float));
            cuda_memcpy_h2d(g_cuda_state.d_v, s->v, kv_dim * sizeof(float));

            /* Apply RoPE on GPU */
            cuda_rope_apply(g_cuda_state.d_q, g_cuda_state.d_k, pos, head_dim,
                            n_heads, n_kv_heads, cfg->rope_freq_base);

            /* Store K and V in GPU cache */
            float *d_k_cache = g_cuda_state.d_key_cache + l * kv_layer_size + pos * kv_dim;
            float *d_v_cache = g_cuda_state.d_value_cache + l * kv_layer_size + pos * kv_dim;
            cuda_memcpy_d2d(d_k_cache, g_cuda_state.d_k, kv_dim * sizeof(float));
            cuda_memcpy_d2d(d_v_cache, g_cuda_state.d_v, kv_dim * sizeof(float));

            /* Multi-head attention using fused GPU kernel */
            int kv_heads_per_group = n_kv_heads > 0 ? n_heads / n_kv_heads : 1;
            float *d_k_layer = g_cuda_state.d_key_cache + l * kv_layer_size;
            float *d_v_layer = g_cuda_state.d_value_cache + l * kv_layer_size;

            for (int h = 0; h < n_heads; h++) {
                float *d_q_h = g_cuda_state.d_q + h * head_dim;
                float *d_out_h = g_cuda_state.d_xb2 + h * head_dim;
                int kv_h = h / kv_heads_per_group;
                float *d_att_h = g_cuda_state.d_att + h * g_cuda_state.max_seq;

                cuda_attention_single(d_out_h, d_q_h, d_k_layer, d_v_layer,
                                      d_att_h, pos + 1, head_dim, kv_h, kv_dim);
            }

            /* Copy attention output back to CPU */
            cuda_memcpy_d2h(s->xb2, g_cuda_state.d_xb2, dim * sizeof(float));

            /* Output projection on CPU */
            matmul_q(s->xb, w->layers[l].wo, s->xb2, dim, dim, g_weight_type);

            /* Residual connection */
            vec_add(s->x, s->x, s->xb, dim);

            /* ----- FFN Block ----- */

            /* RMS norm */
            if (w->layers[l].ffn_norm) {
                float *norm_w = (float *)w->layers[l].ffn_norm;
                rms_norm(s->xb, s->x, norm_w, dim, cfg->rms_norm_eps);
            } else {
                memcpy(s->xb, s->x, dim * sizeof(float));
            }

            /* SwiGLU FFN on CPU */
            matmul_q(s->hb, w->layers[l].w1, s->xb, ffn_dim, dim, g_weight_type);
            matmul_q(s->hb2, w->layers[l].w3, s->xb, ffn_dim, dim, g_weight_type);

            /* SiLU and multiply */
            silu(s->hb, ffn_dim);
            vec_mul(s->hb, s->hb, s->hb2, ffn_dim);

            /* Down projection */
            matmul_q(s->xb, w->layers[l].w2, s->hb, dim, ffn_dim, g_weight_type);

            /* Residual connection */
            vec_add(s->x, s->x, s->xb, dim);
        }

        /* Final RMS norm */
        if (w->norm) {
            float *norm_w = (float *)w->norm;
            rms_norm(s->x, s->x, norm_w, dim, cfg->rms_norm_eps);
        }

        /* Output projection */
        matmul_q(s->logits, w->output, s->x, cfg->vocab_size, dim, g_output_type);

        ctx->pos++;
    }

    return 0;
}
#endif /* HOLO_USE_CUDA */

/*
 * Wrapper that dispatches to GPU or CPU based on availability
 */
int llm_forward_auto(llm_ctx_t *ctx, const int *tokens, int n_tokens) {
    /* For now, always use CPU - GPU attention kernel needs debugging */
    return llm_forward(ctx, tokens, n_tokens);
}
