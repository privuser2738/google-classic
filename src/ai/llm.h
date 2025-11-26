/*
 * Holo - LLM Inference Engine
 * Pure C implementation for transformer inference
 */

#ifndef HOLO_LLM_H
#define HOLO_LLM_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Model Configuration
 * ============================================================================ */

typedef struct {
    char arch[64];              /* Architecture name (llama, mistral, etc.) */

    int vocab_size;
    int context_length;
    int embedding_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int ffn_dim;

    float rope_freq_base;
    float rms_norm_eps;

    /* Special tokens */
    int bos_token;
    int eos_token;
    int pad_token;

    /* Quantization */
    int quant_type;             /* ggml_type_t */
} llm_config_t;

/* ============================================================================
 * KV Cache
 * ============================================================================ */

typedef struct {
    float *key_cache;           /* [n_layers, max_seq, n_kv_heads, head_dim] */
    float *value_cache;         /* [n_layers, max_seq, n_kv_heads, head_dim] */
    int max_seq_len;
    int n_layers;
    int n_kv_heads;
    int head_dim;
} llm_kv_cache_t;

/* ============================================================================
 * Tokenizer
 * ============================================================================ */

typedef struct {
    char **vocab;               /* Token strings */
    float *scores;              /* BPE merge scores */
    int vocab_size;

    int bos_token;
    int eos_token;
    int pad_token;
    int unk_token;

    /* Byte fallback for unknown chars */
    int byte_fallback_start;    /* First byte token ID */
} llm_tokenizer_t;

/* ============================================================================
 * Model Weights
 * ============================================================================ */

typedef struct {
    /* Token embedding */
    void *tok_embeddings;

    /* Per-layer weights */
    struct {
        void *attn_norm;        /* RMS norm weight */
        void *wq;               /* Query projection */
        void *wk;               /* Key projection */
        void *wv;               /* Value projection */
        void *wo;               /* Output projection */

        /* QKV biases (used by Qwen, not LLaMA) */
        float *bq;              /* Query bias [dim] */
        float *bk;              /* Key bias [kv_dim] */
        float *bv;              /* Value bias [kv_dim] */

        void *ffn_norm;         /* FFN RMS norm weight */
        void *w1;               /* FFN gate projection (or up for some models) */
        void *w2;               /* FFN down projection */
        void *w3;               /* FFN up projection (SwiGLU) */
    } *layers;

    /* Output */
    void *norm;                 /* Final RMS norm */
    void *output;               /* Output projection (often tied to embeddings) */

    int n_layers;
} llm_weights_t;

/* ============================================================================
 * Run State (activations during inference)
 * ============================================================================ */

typedef struct {
    float *x;                   /* Current activation [dim] */
    float *xb;                  /* Activation after RMS norm [dim] */
    float *xb2;                 /* Second buffer [dim] */
    float *hb;                  /* Hidden buffer for FFN [ffn_dim] */
    float *hb2;                 /* Second hidden buffer [ffn_dim] */
    float *q;                   /* Query [n_heads * head_dim] */
    float *k;                   /* Key [n_kv_heads * head_dim] */
    float *v;                   /* Value [n_kv_heads * head_dim] */
    float *att;                 /* Attention scores [n_heads * seq_len] */
    float *logits;              /* Output logits [vocab_size] */
} llm_state_t;

/* ============================================================================
 * LLM Context
 * ============================================================================ */

typedef struct {
    llm_config_t config;
    llm_weights_t weights;
    llm_state_t state;
    llm_kv_cache_t kv_cache;
    llm_tokenizer_t tokenizer;

    gguf_ctx_t *gguf;           /* GGUF file context */

    int pos;                    /* Current position in sequence */
    int *tokens;                /* Token buffer */
    int n_tokens;               /* Number of tokens in buffer */
    int max_tokens;             /* Max buffer size */

    bool loaded;
} llm_ctx_t;

/* ============================================================================
 * Sampling Parameters
 * ============================================================================ */

typedef struct {
    float temperature;          /* 0 = greedy, 1 = default */
    float top_p;                /* Nucleus sampling (0.0-1.0) */
    int top_k;                  /* Top-K (0 = disabled) */
    float repeat_penalty;       /* Repetition penalty (1.0 = none) */
    int repeat_last_n;          /* Context for repetition penalty */
    int seed;                   /* Random seed (-1 = random) */
} llm_sampler_t;

#define LLM_SAMPLER_DEFAULT { \
    .temperature = 0.7f,       \
    .top_p = 0.9f,             \
    .top_k = 40,               \
    .repeat_penalty = 1.1f,    \
    .repeat_last_n = 64,       \
    .seed = -1                 \
}

/* ============================================================================
 * API Functions
 * ============================================================================ */

/* Load model from GGUF file */
llm_ctx_t *llm_load(const char *model_path);

/* Free model and all resources */
void llm_free(llm_ctx_t *ctx);

/* Reset context (clear KV cache, reset position) */
void llm_reset(llm_ctx_t *ctx);

/* Tokenize text to token IDs */
int llm_tokenize(llm_ctx_t *ctx, const char *text, int *tokens, int max_tokens, bool add_bos);

/* Decode tokens to text */
int llm_decode(llm_ctx_t *ctx, const int *tokens, int n_tokens, char *text, int max_len);

/* Get string for a single token */
const char *llm_token_str(llm_ctx_t *ctx, int token);

/* Forward pass: process tokens through model, update KV cache */
int llm_forward(llm_ctx_t *ctx, const int *tokens, int n_tokens);

/* Sample next token from logits */
int llm_sample(llm_ctx_t *ctx, const llm_sampler_t *sampler);

/* Generate text (convenience function)
 * Returns number of tokens generated
 * Calls callback for each token (return false to stop)
 */
typedef bool (*llm_token_callback_t)(int token, const char *text, void *user_data);

int llm_generate(llm_ctx_t *ctx,
                 const char *prompt,
                 int max_new_tokens,
                 const llm_sampler_t *sampler,
                 llm_token_callback_t callback,
                 void *user_data);

/* Get model info string */
void llm_print_info(llm_ctx_t *ctx);

/* ============================================================================
 * CUDA Backend (optional GPU acceleration)
 * ============================================================================ */

/* Initialize CUDA backend for a loaded model context */
int llm_cuda_init(llm_ctx_t *ctx);

/* Cleanup CUDA resources */
void llm_cuda_cleanup(void);

/* Check if CUDA is available and initialized */
bool llm_cuda_available(void);

/* Reset CUDA state (clear GPU KV cache) */
void llm_cuda_reset(void);

/* Auto-dispatching forward that uses GPU if available, CPU otherwise */
int llm_forward_auto(llm_ctx_t *ctx, const int *tokens, int n_tokens);

/* ============================================================================
 * Chat Template Support
 * ============================================================================ */

/* Chat template types */
typedef enum {
    LLM_CHAT_TEMPLATE_NONE = 0,     /* Raw text, no template */
    LLM_CHAT_TEMPLATE_CHATML,       /* <|im_start|>role\n...<|im_end|> */
    LLM_CHAT_TEMPLATE_LLAMA2,       /* [INST] ... [/INST] */
    LLM_CHAT_TEMPLATE_LLAMA3,       /* <|start_header_id|>role<|end_header_id|> */
    LLM_CHAT_TEMPLATE_MISTRAL,      /* Same as ChatML for Mistral models */
    LLM_CHAT_TEMPLATE_MISTRAL_V1,   /* <s>role\nmessage</s> (Ministral native) */
    LLM_CHAT_TEMPLATE_GEMMA,        /* <start_of_turn>role\n...<end_of_turn> */
    LLM_CHAT_TEMPLATE_PHI,          /* <|user|>\n...<|end|>\n<|assistant|> */
    LLM_CHAT_TEMPLATE_ZEPHYR,       /* <|user|>\n...\n<|assistant|> */
} llm_chat_template_t;

/* Format a user message with chat template */
int llm_format_chat(llm_ctx_t *ctx, char *output, int max_len,
                    const char *user_message, llm_chat_template_t template_type);

/* Auto-detect template from model architecture */
llm_chat_template_t llm_detect_template(llm_ctx_t *ctx);

/* Chat generation with template (convenience function) */
int llm_chat(llm_ctx_t *ctx,
             const char *user_message,
             int max_new_tokens,
             const llm_sampler_t *sampler,
             llm_token_callback_t callback,
             void *user_data);

#ifdef __cplusplus
}
#endif

#endif /* HOLO_LLM_H */
