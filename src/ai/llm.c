/*
 * Holo - LLM Inference Engine Implementation
 * Pure C implementation for transformer inference
 */

#include "llm.h"
#include "gguf.h"
#include "tensor.h"
#include "quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* ============================================================================
 * Memory Allocation Helpers
 * ============================================================================ */

static void *alloc_aligned(size_t size) {
    /* Allocate with 64-byte alignment for SIMD */
#ifdef _WIN32
    return _aligned_malloc(size, 64);
#else
    void *ptr;
    if (posix_memalign(&ptr, 64, size) != 0) return NULL;
    return ptr;
#endif
}

static void free_aligned(void *ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/* ============================================================================
 * Tokenizer Implementation
 * ============================================================================ */

static int load_tokenizer(llm_ctx_t *ctx) {
    gguf_ctx_t *gguf = ctx->gguf;

    /* Get vocab from GGUF metadata */
    uint64_t vocab_len = 0;
    const void *vocab_arr = gguf_get_array(gguf, "tokenizer.ggml.tokens", &vocab_len);

    if (!vocab_arr || vocab_len == 0) {
        fprintf(stderr, "LLM: No vocabulary found in model\n");
        return -1;
    }

    ctx->tokenizer.vocab_size = (int)vocab_len;
    ctx->tokenizer.vocab = (char **)calloc(vocab_len, sizeof(char *));
    if (!ctx->tokenizer.vocab) return -1;

    /* Copy token strings */
    const gguf_string_t *tokens = (const gguf_string_t *)vocab_arr;
    for (uint64_t i = 0; i < vocab_len; i++) {
        ctx->tokenizer.vocab[i] = strdup(tokens[i].data);
    }

    /* Get scores if available */
    uint64_t scores_len = 0;
    const void *scores_arr = gguf_get_array(gguf, "tokenizer.ggml.scores", &scores_len);
    if (scores_arr && scores_len == vocab_len) {
        ctx->tokenizer.scores = (float *)malloc(vocab_len * sizeof(float));
        memcpy(ctx->tokenizer.scores, scores_arr, vocab_len * sizeof(float));
    }

    /* Get special tokens */
    ctx->tokenizer.bos_token = (int)gguf_get_int(gguf, "tokenizer.ggml.bos_token_id", 1);
    ctx->tokenizer.eos_token = (int)gguf_get_int(gguf, "tokenizer.ggml.eos_token_id", 2);
    ctx->tokenizer.pad_token = (int)gguf_get_int(gguf, "tokenizer.ggml.padding_token_id", 0);
    ctx->tokenizer.unk_token = (int)gguf_get_int(gguf, "tokenizer.ggml.unknown_token_id", 0);

    ctx->config.bos_token = ctx->tokenizer.bos_token;
    ctx->config.eos_token = ctx->tokenizer.eos_token;
    ctx->config.pad_token = ctx->tokenizer.pad_token;

    return 0;
}

static void free_tokenizer(llm_tokenizer_t *tok) {
    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++) {
            free(tok->vocab[i]);
        }
        free(tok->vocab);
    }
    free(tok->scores);
}

/* GPT-2/Qwen byte-level BPE: bytes are mapped to Unicode code points
 * Printable ASCII (33-126) stay the same
 * Others are shifted: byte 0x00-0x20 -> U+0100-0x0120, etc. */
static char byte_to_unicode[256][5];  /* Pre-computed byte->unicode string */
static int byte_to_unicode_initialized = 0;

static void init_byte_to_unicode(void) {
    if (byte_to_unicode_initialized) return;

    /* GPT-2 byte encoder mapping */
    int n = 0;
    for (int i = 33; i <= 126; i++) {  /* Printable ASCII */
        byte_to_unicode[i][0] = (char)i;
        byte_to_unicode[i][1] = '\0';
    }
    /* Non-printable bytes get shifted to higher Unicode */
    int shift = 256;
    for (int i = 0; i < 256; i++) {
        if (i >= 33 && i <= 126) continue;  /* Already handled */
        if (i == 161 || i == 172 || (i >= 166 && i <= 172)) continue; /* Some Latin-1 */

        /* Encode as UTF-8 the code point (256 + offset) */
        int cp = shift + n;
        if (cp < 0x80) {
            byte_to_unicode[i][0] = (char)cp;
            byte_to_unicode[i][1] = '\0';
        } else if (cp < 0x800) {
            byte_to_unicode[i][0] = (char)(0xC0 | (cp >> 6));
            byte_to_unicode[i][1] = (char)(0x80 | (cp & 0x3F));
            byte_to_unicode[i][2] = '\0';
        }
        n++;
    }
    byte_to_unicode_initialized = 1;
}

/* Find byte token for a given byte value */
static int find_byte_token(llm_ctx_t *ctx, unsigned char byte) {
    /* Try GGML format: <0xNN> */
    char byte_str[8];
    snprintf(byte_str, sizeof(byte_str), "<0x%02X>", byte);
    for (int i = 0; i < ctx->tokenizer.vocab_size; i++) {
        const char *tok = ctx->tokenizer.vocab[i];
        if (tok && strcmp(tok, byte_str) == 0) {
            return i;
        }
    }

    /* Try GPT-2/Qwen byte encoding */
    init_byte_to_unicode();
    const char *uni = byte_to_unicode[byte];
    if (uni[0]) {
        for (int i = 0; i < ctx->tokenizer.vocab_size; i++) {
            const char *tok = ctx->tokenizer.vocab[i];
            if (tok && strcmp(tok, uni) == 0) {
                return i;
            }
        }
    }

    return -1;  /* Not found */
}

/* Simple tokenizer - byte-level fallback */
int llm_tokenize(llm_ctx_t *ctx, const char *text, int *tokens, int max_tokens, bool add_bos) {
    int n = 0;

    if (add_bos && n < max_tokens) {
        tokens[n++] = ctx->tokenizer.bos_token;
    }

    /* Simple greedy tokenization - look for longest match */
    const char *p = text;
    while (*p && n < max_tokens) {
        int best_len = 0;
        int best_token = ctx->tokenizer.unk_token;

        /* Try to find longest matching token */
        for (int i = 0; i < ctx->tokenizer.vocab_size; i++) {
            const char *tok = ctx->tokenizer.vocab[i];
            if (!tok) continue;

            int len = strlen(tok);
            if (len > best_len && strncmp(p, tok, len) == 0) {
                best_len = len;
                best_token = i;
            }
        }

        if (best_len > 0) {
            tokens[n++] = best_token;
            p += best_len;
        } else {
            /* Byte fallback - look for <0xNN> byte token */
            unsigned char byte = (unsigned char)*p;
            int byte_tok = find_byte_token(ctx, byte);
            if (byte_tok >= 0) {
                tokens[n++] = byte_tok;
            } else {
                /* No byte token found, use UNK */
                tokens[n++] = ctx->tokenizer.unk_token;
            }
            p++;
        }
    }

    return n;
}

int llm_decode(llm_ctx_t *ctx, const int *tokens, int n_tokens, char *text, int max_len) {
    int pos = 0;

    for (int i = 0; i < n_tokens && pos < max_len - 1; i++) {
        const char *tok = llm_token_str(ctx, tokens[i]);
        if (tok) {
            int len = strlen(tok);
            if (pos + len < max_len) {
                strcpy(text + pos, tok);
                pos += len;
            }
        }
    }

    text[pos] = '\0';
    return pos;
}

/* Buffer for token string conversion */
static char g_token_buf[256];

/* Check if a token is a special/control token that shouldn't be displayed */
static bool is_special_token(const char *tok) {
    if (!tok || tok[0] != '<') return false;

    /* Common special tokens to hide */
    if (strcmp(tok, "<s>") == 0) return true;
    if (strcmp(tok, "</s>") == 0) return true;
    if (strcmp(tok, "<unk>") == 0) return true;
    if (strcmp(tok, "<pad>") == 0) return true;
    if (strcmp(tok, "<mask>") == 0) return true;
    if (strncmp(tok, "<|", 2) == 0) return true;  /* ChatML tokens */
    if (strncmp(tok, "<start_of_turn>", 15) == 0) return true;
    if (strncmp(tok, "<end_of_turn>", 13) == 0) return true;

    return false;
}

const char *llm_token_str(llm_ctx_t *ctx, int token) {
    if (token < 0 || token >= ctx->tokenizer.vocab_size) {
        return "";
    }
    const char *tok = ctx->tokenizer.vocab[token];
    if (!tok) return "";

    /* Skip special/control tokens */
    if (is_special_token(tok)) {
        return "";
    }

    /* Convert SentencePiece and other special characters */
    char *dst = g_token_buf;
    const unsigned char *src = (const unsigned char *)tok;
    int dst_len = 0;

    while (*src && dst_len < 250) {
        /* Check for ▁ (E2 96 81) - SentencePiece space marker */
        if (src[0] == 0xE2 && src[1] == 0x96 && src[2] == 0x81) {
            *dst++ = ' ';
            dst_len++;
            src += 3;
        }
        /* Check for byte tokens like <0x0A> (newline), <0x09> (tab), etc. */
        else if (src[0] == '<' && src[1] == '0' && src[2] == 'x' &&
                 ((src[3] >= '0' && src[3] <= '9') || (src[3] >= 'A' && src[3] <= 'F')) &&
                 ((src[4] >= '0' && src[4] <= '9') || (src[4] >= 'A' && src[4] <= 'F')) &&
                 src[5] == '>') {
            /* Parse hex byte */
            char hex[3] = { (char)src[3], (char)src[4], 0 };
            int byte = (int)strtol(hex, NULL, 16);
            /* Only output printable chars and common whitespace */
            if (byte == 0x0A || byte == 0x0D || byte == 0x09 ||
                (byte >= 0x20 && byte < 0x7F)) {
                *dst++ = (char)byte;
                dst_len++;
            }
            src += 6;
        }
        /* Check for Ġ (U+0120, bytes: C4 A0) - GPT-2 space marker */
        else if (src[0] == 0xC4 && src[1] == 0xA0) {
            *dst++ = ' ';
            dst_len++;
            src += 2;
        }
        /* Check for Ċ (U+010A, bytes: C4 8A) - GPT-2 newline marker */
        else if (src[0] == 0xC4 && src[1] == 0x8A) {
            *dst++ = '\n';
            dst_len++;
            src += 2;
        }
        /* Check for ĉ (U+0109, bytes: C4 89) - GPT-2 tab marker */
        else if (src[0] == 0xC4 && src[1] == 0x89) {
            *dst++ = '\t';
            dst_len++;
            src += 2;
        }
        /* Normal character - copy as-is */
        else {
            *dst++ = *src++;
            dst_len++;
        }
    }
    *dst = '\0';

    return g_token_buf;
}

/* ============================================================================
 * Weight Loading
 * ============================================================================ */

/* Store tensor types for proper dequantization - visible to CUDA module */
int g_emb_type = GGML_TYPE_F32;
int g_weight_type = GGML_TYPE_Q4_K;
int g_output_type = GGML_TYPE_Q4_K;

static int load_weights(llm_ctx_t *ctx) {
    gguf_ctx_t *gguf = ctx->gguf;
    int n_layers = ctx->config.n_layers;

    /* Memory-map the tensor data */
    if (gguf_mmap(gguf) != 0) {
        fprintf(stderr, "LLM: Failed to mmap model file\n");
        return -1;
    }

    /* Allocate layer weight pointers */
    ctx->weights.n_layers = n_layers;
    ctx->weights.layers = calloc(n_layers, sizeof(*ctx->weights.layers));
    if (!ctx->weights.layers) return -1;

    /* Find and assign weight tensors */
    const gguf_tensor_info_t *t;

    /* Token embeddings */
    t = gguf_find_tensor(gguf, "token_embd.weight");
    if (t) {
        ctx->weights.tok_embeddings = (void *)gguf_get_tensor_data(gguf, t);
        g_emb_type = t->type;
    }

    /* Output norm */
    t = gguf_find_tensor(gguf, "output_norm.weight");
    if (t) ctx->weights.norm = (void *)gguf_get_tensor_data(gguf, t);

    /* Output projection */
    t = gguf_find_tensor(gguf, "output.weight");
    if (t) {
        ctx->weights.output = (void *)gguf_get_tensor_data(gguf, t);
        g_output_type = t->type;
    } else {
        /* Tied embeddings */
        ctx->weights.output = ctx->weights.tok_embeddings;
        g_output_type = g_emb_type;
    }

    /* Per-layer weights */
    for (int l = 0; l < n_layers; l++) {
        char name[256];

        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);
        t = gguf_find_tensor(gguf, name);
        if (t) ctx->weights.layers[l].attn_norm = (void *)gguf_get_tensor_data(gguf, t);

        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
        t = gguf_find_tensor(gguf, name);
        if (t) {
            ctx->weights.layers[l].wq = (void *)gguf_get_tensor_data(gguf, t);
            if (l == 0) g_weight_type = t->type;
        }

        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
        t = gguf_find_tensor(gguf, name);
        if (t) ctx->weights.layers[l].wk = (void *)gguf_get_tensor_data(gguf, t);

        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
        t = gguf_find_tensor(gguf, name);
        if (t) ctx->weights.layers[l].wv = (void *)gguf_get_tensor_data(gguf, t);

        /* QKV biases (Qwen models) */
        snprintf(name, sizeof(name), "blk.%d.attn_q.bias", l);
        t = gguf_find_tensor(gguf, name);
        if (t) ctx->weights.layers[l].bq = (float *)gguf_get_tensor_data(gguf, t);

        snprintf(name, sizeof(name), "blk.%d.attn_k.bias", l);
        t = gguf_find_tensor(gguf, name);
        if (t) ctx->weights.layers[l].bk = (float *)gguf_get_tensor_data(gguf, t);

        snprintf(name, sizeof(name), "blk.%d.attn_v.bias", l);
        t = gguf_find_tensor(gguf, name);
        if (t) ctx->weights.layers[l].bv = (float *)gguf_get_tensor_data(gguf, t);

        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
        t = gguf_find_tensor(gguf, name);
        if (t) ctx->weights.layers[l].wo = (void *)gguf_get_tensor_data(gguf, t);

        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
        t = gguf_find_tensor(gguf, name);
        if (t) ctx->weights.layers[l].ffn_norm = (void *)gguf_get_tensor_data(gguf, t);

        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", l);
        t = gguf_find_tensor(gguf, name);
        if (t) ctx->weights.layers[l].w1 = (void *)gguf_get_tensor_data(gguf, t);

        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", l);
        t = gguf_find_tensor(gguf, name);
        if (t) ctx->weights.layers[l].w2 = (void *)gguf_get_tensor_data(gguf, t);

        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", l);
        t = gguf_find_tensor(gguf, name);
        if (t) ctx->weights.layers[l].w3 = (void *)gguf_get_tensor_data(gguf, t);
    }

    return 0;
}

/* ============================================================================
 * State Allocation
 * ============================================================================ */

static int alloc_state(llm_ctx_t *ctx) {
    int dim = ctx->config.embedding_dim;
    int ffn_dim = ctx->config.ffn_dim;
    int n_heads = ctx->config.n_heads;
    int vocab = ctx->config.vocab_size;
    /* Limit practical sequence length to save memory */
    int max_seq = ctx->config.context_length > 4096 ? 4096 : ctx->config.context_length;

    ctx->state.x = alloc_aligned(dim * sizeof(float));
    ctx->state.xb = alloc_aligned(dim * sizeof(float));
    ctx->state.xb2 = alloc_aligned(dim * sizeof(float));
    ctx->state.hb = alloc_aligned(ffn_dim * sizeof(float));
    ctx->state.hb2 = alloc_aligned(ffn_dim * sizeof(float));
    ctx->state.q = alloc_aligned(dim * sizeof(float));
    ctx->state.k = alloc_aligned(dim * sizeof(float));
    ctx->state.v = alloc_aligned(dim * sizeof(float));
    ctx->state.att = alloc_aligned(n_heads * max_seq * sizeof(float));
    ctx->state.logits = alloc_aligned(vocab * sizeof(float));

    if (!ctx->state.x || !ctx->state.xb || !ctx->state.xb2 ||
        !ctx->state.hb || !ctx->state.hb2 || !ctx->state.q ||
        !ctx->state.k || !ctx->state.v || !ctx->state.att ||
        !ctx->state.logits) {
        return -1;
    }

    return 0;
}

static void free_state(llm_state_t *state) {
    free_aligned(state->x);
    free_aligned(state->xb);
    free_aligned(state->xb2);
    free_aligned(state->hb);
    free_aligned(state->hb2);
    free_aligned(state->q);
    free_aligned(state->k);
    free_aligned(state->v);
    free_aligned(state->att);
    free_aligned(state->logits);
}

/* ============================================================================
 * KV Cache
 * ============================================================================ */

static int alloc_kv_cache(llm_ctx_t *ctx) {
    int n_layers = ctx->config.n_layers;
    /* Limit practical sequence length to save memory */
    int max_seq = ctx->config.context_length > 4096 ? 4096 : ctx->config.context_length;
    int n_kv_heads = ctx->config.n_kv_heads;
    int head_dim = ctx->config.head_dim;

    size_t cache_size = (size_t)n_layers * max_seq * n_kv_heads * head_dim * sizeof(float);

    ctx->kv_cache.key_cache = alloc_aligned(cache_size);
    ctx->kv_cache.value_cache = alloc_aligned(cache_size);

    if (!ctx->kv_cache.key_cache || !ctx->kv_cache.value_cache) {
        return -1;
    }

    ctx->kv_cache.max_seq_len = max_seq;
    ctx->kv_cache.n_layers = n_layers;
    ctx->kv_cache.n_kv_heads = n_kv_heads;
    ctx->kv_cache.head_dim = head_dim;

    /* Zero initialize */
    memset(ctx->kv_cache.key_cache, 0, cache_size);
    memset(ctx->kv_cache.value_cache, 0, cache_size);

    return 0;
}

static void free_kv_cache(llm_kv_cache_t *kv) {
    free_aligned(kv->key_cache);
    free_aligned(kv->value_cache);
}

/* ============================================================================
 * Model Loading
 * ============================================================================ */

llm_ctx_t *llm_load(const char *model_path) {
    /* Open GGUF file */
    gguf_ctx_t *gguf = gguf_open(model_path);
    if (!gguf) {
        fprintf(stderr, "Error: Failed to open model file\n");
        return NULL;
    }

    /* Allocate context */
    llm_ctx_t *ctx = calloc(1, sizeof(llm_ctx_t));
    if (!ctx) {
        gguf_close(gguf);
        return NULL;
    }
    ctx->gguf = gguf;

    /* Extract config from GGUF metadata */
    strncpy(ctx->config.arch, gguf->model.arch, sizeof(ctx->config.arch) - 1);
    ctx->config.vocab_size = gguf->model.vocab_size;
    ctx->config.context_length = gguf->model.context_length;
    ctx->config.embedding_dim = gguf->model.embedding_length;
    ctx->config.n_layers = gguf->model.block_count;
    ctx->config.n_heads = gguf->model.attention_head_count;
    ctx->config.n_kv_heads = gguf->model.attention_head_count_kv;
    ctx->config.head_dim = ctx->config.embedding_dim / ctx->config.n_heads;
    ctx->config.ffn_dim = gguf->model.feed_forward_length;
    ctx->config.rope_freq_base = gguf->model.rope_freq_base;
    ctx->config.rms_norm_eps = gguf->model.rms_norm_eps;

    printf("  Architecture: %s (%d layers, %d dim)\n",
           ctx->config.arch, ctx->config.n_layers, ctx->config.embedding_dim);
    printf("  Attention: %d heads, %d KV heads, head_dim=%d\n",
           ctx->config.n_heads, ctx->config.n_kv_heads, ctx->config.head_dim);
    printf("  FFN dim: %d\n", ctx->config.ffn_dim);
    printf("  Context: %d tokens\n", ctx->config.context_length);

    /* Load tokenizer */
    if (load_tokenizer(ctx) != 0) {
        fprintf(stderr, "Error: Failed to load tokenizer\n");
        llm_free(ctx);
        return NULL;
    }
    printf("  Vocabulary: %d tokens\n", ctx->tokenizer.vocab_size);

    /* Load weights */
    printf("  Loading weights...\n");
    if (load_weights(ctx) != 0) {
        fprintf(stderr, "Error: Failed to load weights\n");
        llm_free(ctx);
        return NULL;
    }

    /* Report detected quantization types */
    const char *type_names[] = {
        "F32", "F16", "Q4_0", "Q4_1", "Q4_2", "Q4_3",
        "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K",
        "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS"
    };
    printf("  Quantization: embeddings=%s, weights=%s, output=%s\n",
           g_emb_type < 18 ? type_names[g_emb_type] : "unknown",
           g_weight_type < 18 ? type_names[g_weight_type] : "unknown",
           g_output_type < 18 ? type_names[g_output_type] : "unknown");

    /* Report QKV bias (Qwen models) */
    if (ctx->weights.layers[0].bq) {
        printf("  QKV bias: enabled (Qwen-style)\n");
    }

    /* Allocate state */
    if (alloc_state(ctx) != 0) {
        fprintf(stderr, "Error: Failed to allocate state\n");
        llm_free(ctx);
        return NULL;
    }

    /* Allocate KV cache */
    if (alloc_kv_cache(ctx) != 0) {
        fprintf(stderr, "Error: Failed to allocate KV cache\n");
        llm_free(ctx);
        return NULL;
    }

    /* Allocate token buffer */
    ctx->max_tokens = ctx->config.context_length;
    ctx->tokens = malloc(ctx->max_tokens * sizeof(int));
    if (!ctx->tokens) {
        llm_free(ctx);
        return NULL;
    }

    ctx->loaded = true;

    return ctx;
}

void llm_free(llm_ctx_t *ctx) {
    if (!ctx) return;

    free_tokenizer(&ctx->tokenizer);
    free_state(&ctx->state);
    free_kv_cache(&ctx->kv_cache);
    free(ctx->weights.layers);
    free(ctx->tokens);

    if (ctx->gguf) {
        gguf_close(ctx->gguf);
    }

    free(ctx);
}

void llm_reset(llm_ctx_t *ctx) {
    if (!ctx) return;

    ctx->pos = 0;
    ctx->n_tokens = 0;

    /* Clear KV cache */
    size_t cache_size = (size_t)ctx->kv_cache.n_layers *
                        ctx->kv_cache.max_seq_len *
                        ctx->kv_cache.n_kv_heads *
                        ctx->kv_cache.head_dim * sizeof(float);
    memset(ctx->kv_cache.key_cache, 0, cache_size);
    memset(ctx->kv_cache.value_cache, 0, cache_size);
}

/* ============================================================================
 * Quantized Matrix-Vector Helper
 * Performs dst = M @ v where M is quantized
 * ============================================================================ */

/* GGUF stores weight matrices as [out_dim, in_dim] (standard row-major)
 * For y = W @ x:
 *   - x has length in_dim
 *   - y has length out_dim
 *   - W is stored as [out_dim, in_dim] in row-major
 *   - y[i] = sum_j(W[i, j] * x[j]) = sum_j(W[i * in_dim + j] * x[j])
 *   - This is standard dot product of row i with input vector x
 *
 * Parameters:
 *   dst: output vector [out_dim]
 *   M: weight matrix [out_dim, in_dim]
 *   v: input vector [in_dim]
 *   out_dim: output dimension (number of rows in M)
 *   in_dim: input dimension (number of cols in M)
 */
/* Non-static for CUDA module access */
static int g_matmul_debug = 0;

void matmul_q(float *dst, const void *M, const float *v,
              int out_dim, int in_dim, int quant_type) {
    if (!M) {
        memset(dst, 0, out_dim * sizeof(float));
        return;
    }

    /* Debug disabled */

    /* Standard row-major matmul: each output is dot product of one row with input */
    switch (quant_type) {
        case GGML_TYPE_F32: {
            const float *Mf = (const float *)M;
            for (int i = 0; i < out_dim; i++) {
                float sum = 0.0f;
                const float *row = Mf + i * in_dim;
                for (int j = 0; j < in_dim; j++) {
                    sum += row[j] * v[j];
                }
                dst[i] = sum;
            }
            break;
        }
        case GGML_TYPE_F16: {
            const uint16_t *Mf = (const uint16_t *)M;
            for (int i = 0; i < out_dim; i++) {
                float sum = 0.0f;
                const uint16_t *row = Mf + i * in_dim;
                for (int j = 0; j < in_dim; j++) {
                    sum += f16_to_f32(row[j]) * v[j];
                }
                dst[i] = sum;
            }
            break;
        }
        case GGML_TYPE_Q8_0: {
            /* Q8_0: matrix is [out_dim, in_dim], each row quantized into blocks of 32
             * blocks_per_row = in_dim / 32 */
            const block_q8_0 *blocks = (const block_q8_0 *)M;
            int blocks_per_row = in_dim / QK8_0;

            for (int i = 0; i < out_dim; i++) {
                float sum = 0.0f;
                const block_q8_0 *row = blocks + i * blocks_per_row;

                for (int b = 0; b < blocks_per_row; b++) {
                    float scale = f16_to_f32(row[b].d);
                    int base = b * QK8_0;
                    for (int k = 0; k < QK8_0; k++) {
                        sum += scale * row[b].qs[k] * v[base + k];
                    }
                }
                dst[i] = sum;
            }
            break;
        }
        case GGML_TYPE_Q4_0: {
            const block_q4_0 *blocks = (const block_q4_0 *)M;
            int blocks_per_row = in_dim / QK4_0;

            for (int i = 0; i < out_dim; i++) {
                float sum = 0.0f;
                const block_q4_0 *row = blocks + i * blocks_per_row;

                for (int b = 0; b < blocks_per_row; b++) {
                    float scale = f16_to_f32(row[b].d);
                    int base = b * QK4_0;
                    for (int k = 0; k < 16; k++) {
                        uint8_t byte = row[b].qs[k];
                        int8_t q0 = (byte & 0xF) - 8;
                        int8_t q1 = (byte >> 4) - 8;
                        sum += scale * q0 * v[base + k];
                        sum += scale * q1 * v[base + k + 16];
                    }
                }
                dst[i] = sum;
            }
            break;
        }
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
            /* TODO: Implement K-quants */
            memset(dst, 0, out_dim * sizeof(float));
            break;
        default:
            memset(dst, 0, out_dim * sizeof(float));
            break;
    }
}

/* Dequantize embedding for a token
 * GGUF tensor dims are column-major: dims[0]=cols, dims[1]=rows
 * For embeddings [4096, 32000] in GGUF means 32000 tokens x 4096 dim
 * Memory layout: each token's embedding is contiguous (standard row-major access)
 *
 * Parameters:
 *   dst: output embedding [dim]
 *   embeddings: embedding table [vocab_size, dim] in memory
 *   token: token index
 *   dim: embedding dimension
 *   vocab_size: vocabulary size (unused but for clarity)
 * Non-static for CUDA module access
 */
void get_embedding(float *dst, const void *embeddings, int token,
                   int dim, int vocab_size, int quant_type) {
    (void)vocab_size;  /* Memory layout is [vocab, dim] - token gives row offset */

    switch (quant_type) {
        case GGML_TYPE_F32: {
            /* Each token's embedding is contiguous: emb[token * dim ... token * dim + dim-1] */
            const float *emb = (const float *)embeddings + token * dim;
            memcpy(dst, emb, dim * sizeof(float));
            break;
        }
        case GGML_TYPE_F16: {
            const uint16_t *emb = (const uint16_t *)embeddings + token * dim;
            for (int i = 0; i < dim; i++) {
                dst[i] = f16_to_f32(emb[i]);
            }
            break;
        }
        case GGML_TYPE_Q8_0: {
            /* Q8_0: each token's embedding (dim values) is quantized into blocks of 32
             * blocks_per_token = dim / 32 */
            int blocks_per_token = dim / QK8_0;
            const block_q8_0 *token_blocks = (const block_q8_0 *)embeddings + token * blocks_per_token;

            for (int b = 0; b < blocks_per_token; b++) {
                float scale = f16_to_f32(token_blocks[b].d);
                int base = b * QK8_0;
                for (int k = 0; k < QK8_0; k++) {
                    dst[base + k] = scale * token_blocks[b].qs[k];
                }
            }
            break;
        }
        case GGML_TYPE_Q4_0: {
            int blocks_per_token = dim / QK4_0;
            const block_q4_0 *token_blocks = (const block_q4_0 *)embeddings + token * blocks_per_token;

            for (int b = 0; b < blocks_per_token; b++) {
                float scale = f16_to_f32(token_blocks[b].d);
                int base = b * QK4_0;
                for (int k = 0; k < 16; k++) {
                    uint8_t byte = token_blocks[b].qs[k];
                    int8_t q0 = (byte & 0xF) - 8;
                    int8_t q1 = (byte >> 4) - 8;
                    dst[base + k] = scale * q0;
                    dst[base + k + 16] = scale * q1;
                }
            }
            break;
        }
        default:
            memset(dst, 0, dim * sizeof(float));
            break;
    }
}

/* ============================================================================
 * Forward Pass
 * ============================================================================ */

int llm_forward(llm_ctx_t *ctx, const int *tokens, int n_tokens) {
    if (!ctx || !ctx->loaded) return -1;

    llm_config_t *cfg = &ctx->config;
    llm_weights_t *w = &ctx->weights;
    llm_state_t *s = &ctx->state;
    llm_kv_cache_t *kv = &ctx->kv_cache;

    int dim = cfg->embedding_dim;
    int n_heads = cfg->n_heads;
    int n_kv_heads = cfg->n_kv_heads;
    int head_dim = cfg->head_dim;
    int n_layers = cfg->n_layers;
    int ffn_dim = cfg->ffn_dim;

    /* Use detected quantization types */
    int qtype = g_weight_type;

    /* KV cache dimensions */
    int kv_dim = n_kv_heads * head_dim;
    size_t kv_layer_size = (size_t)kv->max_seq_len * kv_dim;

    /* Process each token */
    for (int t = 0; t < n_tokens; t++) {
        int token = tokens[t];
        int pos = ctx->pos;

        if (pos >= kv->max_seq_len) {
            fprintf(stderr, "LLM: Position %d exceeds max sequence length %d\n",
                    pos, kv->max_seq_len);
            return -1;
        }

        /* Get token embedding */
        if (w->tok_embeddings) {
            get_embedding(s->x, w->tok_embeddings, token, dim, cfg->vocab_size, g_emb_type);
        } else {
            memset(s->x, 0, dim * sizeof(float));
        }

        /* Debug disabled */

        /* Process through transformer layers */
        for (int l = 0; l < n_layers; l++) {
            /* ----- Attention Block ----- */

            /* RMSNorm before attention */
            if (w->layers[l].attn_norm) {
                float *norm_w = (float *)w->layers[l].attn_norm;
                rms_norm(s->xb, s->x, norm_w, dim, cfg->rms_norm_eps);
            } else {
                memcpy(s->xb, s->x, dim * sizeof(float));
            }

            /* QKV projections: q = Wq @ xb, k = Wk @ xb, v = Wv @ xb */
            matmul_q(s->q, w->layers[l].wq, s->xb, dim, dim, qtype);
            matmul_q(s->k, w->layers[l].wk, s->xb, kv_dim, dim, qtype);
            matmul_q(s->v, w->layers[l].wv, s->xb, kv_dim, dim, qtype);

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

            /* Apply RoPE to Q and K separately (K has fewer heads) */
            /* RoPE for Q (all heads) */
            for (int h = 0; h < n_heads; h++) {
                float *q_h = s->q + h * head_dim;
                int half = head_dim / 2;
                for (int i = 0; i < half; i++) {
                    float freq = 1.0f / powf(cfg->rope_freq_base, (float)(2 * i) / head_dim);
                    float theta = pos * freq;
                    float cos_t = cosf(theta);
                    float sin_t = sinf(theta);
                    float q0 = q_h[i];
                    float q1 = q_h[i + half];
                    q_h[i] = q0 * cos_t - q1 * sin_t;
                    q_h[i + half] = q0 * sin_t + q1 * cos_t;
                }
            }
            /* RoPE for K (n_kv_heads) */
            for (int h = 0; h < n_kv_heads; h++) {
                float *k_h = s->k + h * head_dim;
                int half = head_dim / 2;
                for (int i = 0; i < half; i++) {
                    float freq = 1.0f / powf(cfg->rope_freq_base, (float)(2 * i) / head_dim);
                    float theta = pos * freq;
                    float cos_t = cosf(theta);
                    float sin_t = sinf(theta);
                    float k0 = k_h[i];
                    float k1 = k_h[i + half];
                    k_h[i] = k0 * cos_t - k1 * sin_t;
                    k_h[i + half] = k0 * sin_t + k1 * cos_t;
                }
            }

            /* Store K and V in cache */
            float *k_cache = kv->key_cache + l * kv_layer_size + pos * kv_dim;
            float *v_cache = kv->value_cache + l * kv_layer_size + pos * kv_dim;
            memcpy(k_cache, s->k, kv_dim * sizeof(float));
            memcpy(v_cache, s->v, kv_dim * sizeof(float));

            /* Multi-head attention */
            /* For each query head */
            int kv_heads_per_group = n_kv_heads > 0 ? n_heads / n_kv_heads : 1;

            memset(s->xb2, 0, dim * sizeof(float));

            for (int h = 0; h < n_heads; h++) {
                float *q_h = s->q + h * head_dim;
                float *out_h = s->xb2 + h * head_dim;
                int kv_h = h / kv_heads_per_group;

                /* Compute attention scores for this head */
                /* Use fixed stride based on max_seq_len */
                float *att_h = s->att + h * kv->max_seq_len;

                for (int p = 0; p <= pos; p++) {
                    float *k_p = kv->key_cache + l * kv_layer_size + p * kv_dim + kv_h * head_dim;
                    float score = vec_dot(q_h, k_p, head_dim);
                    att_h[p] = score / sqrtf((float)head_dim);
                }

                /* Softmax attention scores */
                softmax(att_h, att_h, pos + 1);

                /* Apply attention to values */
                for (int p = 0; p <= pos; p++) {
                    float *v_p = kv->value_cache + l * kv_layer_size + p * kv_dim + kv_h * head_dim;
                    float a = att_h[p];
                    for (int d = 0; d < head_dim; d++) {
                        out_h[d] += a * v_p[d];
                    }
                }
            }

            /* Output projection: xb = Wo @ xb2 */
            matmul_q(s->xb, w->layers[l].wo, s->xb2, dim, dim, qtype);

            /* Residual connection */
            vec_add(s->x, s->x, s->xb, dim);

            /* ----- FFN Block ----- */

            /* RMSNorm before FFN */
            if (w->layers[l].ffn_norm) {
                float *norm_w = (float *)w->layers[l].ffn_norm;
                rms_norm(s->xb, s->x, norm_w, dim, cfg->rms_norm_eps);
            } else {
                memcpy(s->xb, s->x, dim * sizeof(float));
            }

            /* SwiGLU FFN:
             * hb = silu(w1 @ xb) * (w3 @ xb)
             * xb = w2 @ hb
             */
            matmul_q(s->hb, w->layers[l].w1, s->xb, ffn_dim, dim, qtype);
            matmul_q(s->hb2, w->layers[l].w3, s->xb, ffn_dim, dim, qtype);

            /* SiLU activation on gate, multiply with up projection */
            silu(s->hb, ffn_dim);
            vec_mul(s->hb, s->hb, s->hb2, ffn_dim);

            /* Down projection */
            matmul_q(s->xb, w->layers[l].w2, s->hb, dim, ffn_dim, qtype);

            /* Residual connection */
            vec_add(s->x, s->x, s->xb, dim);
        }

        /* Final RMSNorm */
        if (w->norm) {
            float *norm_w = (float *)w->norm;
            rms_norm(s->x, s->x, norm_w, dim, cfg->rms_norm_eps);
        }

        /* Output projection to vocabulary */
        if (w->output) {
            matmul_q(s->logits, w->output, s->x, cfg->vocab_size, dim, g_output_type);
        } else {
            memset(s->logits, 0, cfg->vocab_size * sizeof(float));
        }

        ctx->pos++;
    }

    return 0;
}

/* ============================================================================
 * Sampling
 * ============================================================================ */

static int g_debug_sample_count = 0;

void llm_debug_reset(void) {
    g_debug_sample_count = 0;
}

int llm_sample(llm_ctx_t *ctx, const llm_sampler_t *sampler) {
    if (!ctx || !ctx->loaded) return -1;

    float *logits = ctx->state.logits;
    int vocab_size = ctx->config.vocab_size;

    /* Debug: print first few samples (disabled) */
    if (0 && g_debug_sample_count < 5) {
        /* Find top 5 logits */
        int top_idx[5] = {0, 1, 2, 3, 4};
        float top_val[5];
        for (int i = 0; i < 5; i++) top_val[i] = logits[i];

        for (int i = 5; i < vocab_size; i++) {
            /* Find min in top 5 */
            int min_j = 0;
            for (int j = 1; j < 5; j++) {
                if (top_val[j] < top_val[min_j]) min_j = j;
            }
            if (logits[i] > top_val[min_j]) {
                top_val[min_j] = logits[i];
                top_idx[min_j] = i;
            }
        }

        /* Sort top 5 */
        for (int i = 0; i < 4; i++) {
            for (int j = i+1; j < 5; j++) {
                if (top_val[j] > top_val[i]) {
                    float tv = top_val[i]; top_val[i] = top_val[j]; top_val[j] = tv;
                    int ti = top_idx[i]; top_idx[i] = top_idx[j]; top_idx[j] = ti;
                }
            }
        }

        fprintf(stderr, "[DEBUG] Sample %d - Top logits (EOS=%d logit=%.2f):\n",
                g_debug_sample_count, ctx->config.eos_token,
                logits[ctx->config.eos_token]);
        for (int i = 0; i < 5; i++) {
            const char *tok = ctx->tokenizer.vocab[top_idx[i]];
            fprintf(stderr, "  [%d] %.3f '%s'\n", top_idx[i], top_val[i],
                    tok ? tok : "(null)");
        }
        g_debug_sample_count++;
    }

    /* Apply temperature */
    if (sampler->temperature > 0 && sampler->temperature != 1.0f) {
        apply_temperature(logits, vocab_size, sampler->temperature);
    }

    /* Apply repetition penalty */
    if (sampler->repeat_penalty != 1.0f && ctx->n_tokens > 0) {
        int start = ctx->n_tokens > sampler->repeat_last_n ?
                    ctx->n_tokens - sampler->repeat_last_n : 0;
        apply_repetition_penalty(logits, ctx->tokens + start,
                                ctx->n_tokens - start, vocab_size,
                                sampler->repeat_penalty);
    }

    /* Convert to probabilities - use dynamic allocation for large vocabularies */
    float *probs = (float *)malloc(vocab_size * sizeof(float));
    if (!probs) return -1;
    softmax(probs, logits, vocab_size);

    /* Apply top-k */
    if (sampler->top_k > 0) {
        top_k(probs, vocab_size, sampler->top_k);
    }

    /* Apply top-p */
    if (sampler->top_p > 0 && sampler->top_p < 1.0f) {
        top_p(probs, vocab_size, sampler->top_p);
    }

    /* Sample */
    int token;
    if (sampler->temperature == 0) {
        token = argmax(probs, vocab_size);
    } else {
        float r = (float)rand() / RAND_MAX;
        token = sample_prob(probs, vocab_size, r);
    }

    free(probs);
    return token;
}

/* ============================================================================
 * Generation
 * ============================================================================ */

int llm_generate(llm_ctx_t *ctx,
                 const char *prompt,
                 int max_new_tokens,
                 const llm_sampler_t *sampler,
                 llm_token_callback_t callback,
                 void *user_data) {
    if (!ctx || !ctx->loaded || !prompt) return -1;

    /* Default sampler if none provided */
    llm_sampler_t default_sampler = LLM_SAMPLER_DEFAULT;
    if (!sampler) sampler = &default_sampler;

    /* Seed RNG */
    if (sampler->seed >= 0) {
        srand((unsigned)sampler->seed);
    } else {
        srand((unsigned)time(NULL));
    }

    /* Reset context for new generation */
    llm_reset(ctx);

    /* Tokenize prompt */
    int prompt_tokens[4096];
    int n_prompt = llm_tokenize(ctx, prompt, prompt_tokens, 4096, true);

    if (n_prompt <= 0) {
        fprintf(stderr, "LLM: Failed to tokenize prompt\n");
        return -1;
    }

    /* Copy prompt tokens to context */
    if (n_prompt > ctx->max_tokens) {
        n_prompt = ctx->max_tokens;
    }
    memcpy(ctx->tokens, prompt_tokens, n_prompt * sizeof(int));
    ctx->n_tokens = n_prompt;

    /* Process prompt (prefill) */
    if (llm_forward_auto(ctx, prompt_tokens, n_prompt) != 0) {
        fprintf(stderr, "LLM: Forward pass failed on prompt\n");
        return -1;
    }

    /* Generate new tokens */
    int n_generated = 0;

    for (int i = 0; i < max_new_tokens; i++) {
        /* Sample next token */
        int token = llm_sample(ctx, sampler);

        /* Check for EOS */
        if (token == ctx->config.eos_token) {
            break;
        }

        /* Add to token buffer */
        if (ctx->n_tokens < ctx->max_tokens) {
            ctx->tokens[ctx->n_tokens++] = token;
        }

        /* Callback with token text */
        if (callback) {
            const char *text = llm_token_str(ctx, token);
            if (!callback(token, text, user_data)) {
                break;  /* User requested stop */
            }
        }

        /* Forward pass for next token */
        if (llm_forward_auto(ctx, &token, 1) != 0) {
            fprintf(stderr, "LLM: Forward pass failed at token %d\n", i);
            break;
        }

        n_generated++;
    }

    return n_generated;
}

/* ============================================================================
 * Info
 * ============================================================================ */

void llm_print_info(llm_ctx_t *ctx) {
    if (!ctx) return;

    printf("\n=== LLM Model Info ===\n");
    printf("Architecture: %s\n", ctx->config.arch);
    printf("Vocab size: %d\n", ctx->config.vocab_size);
    printf("Context length: %d\n", ctx->config.context_length);
    printf("Embedding dim: %d\n", ctx->config.embedding_dim);
    printf("Layers: %d\n", ctx->config.n_layers);
    printf("Attention heads: %d\n", ctx->config.n_heads);
    printf("KV heads: %d\n", ctx->config.n_kv_heads);
    printf("Head dim: %d\n", ctx->config.head_dim);
    printf("FFN dim: %d\n", ctx->config.ffn_dim);
    printf("RoPE base: %.1f\n", ctx->config.rope_freq_base);
    printf("RMS norm eps: %e\n", ctx->config.rms_norm_eps);
    printf("BOS token: %d\n", ctx->config.bos_token);
    printf("EOS token: %d\n", ctx->config.eos_token);
    printf("\n");
}

/* ============================================================================
 * Chat Template Support
 * ============================================================================ */

llm_chat_template_t llm_detect_template(llm_ctx_t *ctx) {
    if (!ctx) return LLM_CHAT_TEMPLATE_NONE;

    /* First, try to read the actual chat_template from GGUF metadata */
    const char *chat_template = NULL;
    if (ctx->gguf) {
        chat_template = gguf_get_string(ctx->gguf, "tokenizer.chat_template");
    }

    /* Template detection debug (disabled) */
    (void)chat_template;

    /* If we have the actual template string, analyze it */
    if (chat_template && chat_template[0]) {
        /* ChatML: uses <|im_start|> and <|im_end|> */
        if (strstr(chat_template, "im_start") || strstr(chat_template, "im_end")) {
            return LLM_CHAT_TEMPLATE_CHATML;
        }
        /* Llama 3: uses <|start_header_id|> */
        if (strstr(chat_template, "start_header_id") || strstr(chat_template, "end_header_id")) {
            return LLM_CHAT_TEMPLATE_LLAMA3;
        }
        /* Gemma: uses <start_of_turn> */
        if (strstr(chat_template, "start_of_turn") || strstr(chat_template, "end_of_turn")) {
            return LLM_CHAT_TEMPLATE_GEMMA;
        }
        /* Llama 2: uses [INST] */
        if (strstr(chat_template, "[INST]") || strstr(chat_template, "[/INST]")) {
            return LLM_CHAT_TEMPLATE_LLAMA2;
        }
        /* Mistral V1 native: simple bos_token + role + content + eos_token format */
        if (strstr(chat_template, "bos_token") && strstr(chat_template, "role") &&
            strstr(chat_template, "eos_token") && !strstr(chat_template, "[INST]") &&
            !strstr(chat_template, "im_start")) {
            return LLM_CHAT_TEMPLATE_MISTRAL_V1;
        }
        /* Phi/Zephyr style */
        if (strstr(chat_template, "<|user|>") || strstr(chat_template, "<|assistant|>")) {
            if (strstr(chat_template, "<|end|>")) {
                return LLM_CHAT_TEMPLATE_PHI;
            }
            return LLM_CHAT_TEMPLATE_ZEPHYR;
        }
    }

    /* Fall back to vocabulary-based detection */
    bool has_chatml = false;
    bool has_llama3 = false;
    bool has_gemma = false;

    /* Scan full vocabulary for special tokens (they may be at the end) */
    int scan_limit = ctx->tokenizer.vocab_size;
    for (int i = 0; i < scan_limit; i++) {
        const char *tok = ctx->tokenizer.vocab[i];
        if (!tok) continue;

        if (strstr(tok, "<|im_start|>") || strstr(tok, "<|im_end|>")) {
            has_chatml = true;
        }
        if (strstr(tok, "<|start_header_id|>") || strstr(tok, "<|end_header_id|>")) {
            has_llama3 = true;
        }
        if (strstr(tok, "<start_of_turn>") || strstr(tok, "<end_of_turn>")) {
            has_gemma = true;
        }
    }

    if (has_chatml) {
        return LLM_CHAT_TEMPLATE_CHATML;
    }
    if (has_llama3) {
        return LLM_CHAT_TEMPLATE_LLAMA3;
    }
    if (has_gemma) {
        return LLM_CHAT_TEMPLATE_GEMMA;
    }

    /* Fall back to architecture name */
    const char *arch = ctx->config.arch;

    if (strstr(arch, "mistral") || strstr(arch, "Mistral") ||
        strstr(arch, "ministral") || strstr(arch, "Ministral")) {
        return LLM_CHAT_TEMPLATE_MISTRAL_V1;
    }
    if (strstr(arch, "llama") || strstr(arch, "Llama")) {
        return LLM_CHAT_TEMPLATE_LLAMA2;
    }
    if (strstr(arch, "gemma") || strstr(arch, "Gemma")) {
        return LLM_CHAT_TEMPLATE_GEMMA;
    }
    if (strstr(arch, "phi") || strstr(arch, "Phi")) {
        return LLM_CHAT_TEMPLATE_PHI;
    }
    if (strstr(arch, "qwen") || strstr(arch, "Qwen")) {
        return LLM_CHAT_TEMPLATE_CHATML;
    }
    if (strstr(arch, "zephyr") || strstr(arch, "Zephyr")) {
        return LLM_CHAT_TEMPLATE_ZEPHYR;
    }

    /* Default to no template - use raw text */
    return LLM_CHAT_TEMPLATE_NONE;
}

int llm_format_chat(llm_ctx_t *ctx, char *output, int max_len,
                    const char *user_message, llm_chat_template_t template_type) {
    (void)ctx;  /* May use for special token lookup later */

    int len = 0;

    switch (template_type) {
        case LLM_CHAT_TEMPLATE_CHATML:
        case LLM_CHAT_TEMPLATE_MISTRAL:
            /* ChatML format: <|im_start|>role\nmessage<|im_end|>\n */
            len = snprintf(output, max_len,
                "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
                user_message);
            break;

        case LLM_CHAT_TEMPLATE_MISTRAL_V1:
            /* Mistral native format: <s>role\nmessage</s>\n
             * Note: <s> and </s> are BOS (token 1) and EOS (token 2)
             * We insert these as literal strings - the tokenizer converts them */
            len = snprintf(output, max_len,
                "<s>user\n%s</s>\n<s>assistant\n",
                user_message);
            break;

        case LLM_CHAT_TEMPLATE_LLAMA2:
            /* Llama 2 format: [INST] message [/INST] */
            len = snprintf(output, max_len,
                "[INST] %s [/INST]",
                user_message);
            break;

        case LLM_CHAT_TEMPLATE_LLAMA3:
            /* Llama 3 format */
            len = snprintf(output, max_len,
                "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n",
                user_message);
            break;

        case LLM_CHAT_TEMPLATE_GEMMA:
            /* Gemma format */
            len = snprintf(output, max_len,
                "<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n",
                user_message);
            break;

        case LLM_CHAT_TEMPLATE_PHI:
            /* Phi format */
            len = snprintf(output, max_len,
                "<|user|>\n%s<|end|>\n<|assistant|>\n",
                user_message);
            break;

        case LLM_CHAT_TEMPLATE_ZEPHYR:
            /* Zephyr format */
            len = snprintf(output, max_len,
                "<|user|>\n%s</s>\n<|assistant|>\n",
                user_message);
            break;

        case LLM_CHAT_TEMPLATE_NONE:
        default:
            /* No template, just use raw message */
            len = snprintf(output, max_len, "%s", user_message);
            break;
    }

    return len < max_len ? len : max_len - 1;
}

/* Check if token is an end-of-turn marker */
static bool is_eot_token(llm_ctx_t *ctx, int token) {
    if (token == ctx->config.eos_token) return true;

    const char *str = ctx->tokenizer.vocab[token];
    if (!str) return false;

    /* Check for common end-of-turn markers */
    if (strcmp(str, "<|im_end|>") == 0) return true;
    if (strcmp(str, "<|eot_id|>") == 0) return true;
    if (strcmp(str, "<end_of_turn>") == 0) return true;
    if (strcmp(str, "<|end|>") == 0) return true;
    if (strcmp(str, "</s>") == 0) return true;
    if (strcmp(str, "<|im_start|>") == 0) return true;  /* New turn starting */

    /* Check for start-of-new-turn tokens (tokenized ChatML markers)
     * Token "▁<|" or space followed by | often precedes "im_start" */
    if (strcmp(str, "▁<|") == 0 || strcmp(str, " <|") == 0) {
        return true;
    }

    return false;
}

int llm_chat(llm_ctx_t *ctx,
             const char *user_message,
             int max_new_tokens,
             const llm_sampler_t *sampler,
             llm_token_callback_t callback,
             void *user_data) {
    if (!ctx || !ctx->loaded || !user_message) return -1;

    /* Detect and apply chat template */
    llm_chat_template_t template_type = llm_detect_template(ctx);

    /* Override for specific architectures */
    if (strstr(ctx->config.arch, "qwen")) {
        template_type = LLM_CHAT_TEMPLATE_CHATML;
    }

    char formatted_prompt[8192];
    llm_format_chat(ctx, formatted_prompt, sizeof(formatted_prompt),
                    user_message, template_type);

    /* Default sampler if none provided */
    llm_sampler_t default_sampler = LLM_SAMPLER_DEFAULT;
    if (!sampler) sampler = &default_sampler;

    /* Seed RNG */
    if (sampler->seed >= 0) {
        srand((unsigned)sampler->seed);
    } else {
        srand((unsigned)time(NULL));
    }

    /* Reset context for new generation */
    llm_reset(ctx);
    llm_debug_reset();

    /* Tokenize formatted prompt
     * Note: Don't add BOS since the formatted prompt already contains
     * the appropriate start tokens (e.g., <s> for Mistral) */
    int prompt_tokens[8192];
    int n_prompt = llm_tokenize(ctx, formatted_prompt, prompt_tokens, 8192, false);

    if (n_prompt <= 0) {
        fprintf(stderr, "LLM: Failed to tokenize prompt\n");
        return -1;
    }

    /* Debug disabled */

    /* Copy prompt tokens to context */
    if (n_prompt > ctx->max_tokens) {
        n_prompt = ctx->max_tokens;
    }
    memcpy(ctx->tokens, prompt_tokens, n_prompt * sizeof(int));
    ctx->n_tokens = n_prompt;

    /* Process prompt (prefill) */
    if (llm_forward_auto(ctx, prompt_tokens, n_prompt) != 0) {
        fprintf(stderr, "LLM: Forward pass failed on prompt\n");
        return -1;
    }

    /* Generate new tokens */
    int n_generated = 0;

    for (int i = 0; i < max_new_tokens; i++) {
        /* Sample next token */
        int token = llm_sample(ctx, sampler);

        /* Check for EOS or end-of-turn markers */
        if (is_eot_token(ctx, token)) {
            break;
        }

        /* Add to token buffer */
        if (ctx->n_tokens < ctx->max_tokens) {
            ctx->tokens[ctx->n_tokens++] = token;
        }

        /* Callback with token text */
        if (callback) {
            const char *text = llm_token_str(ctx, token);
            if (!callback(token, text, user_data)) {
                break;  /* User requested stop */
            }
        }

        /* Forward pass for next token */
        if (llm_forward_auto(ctx, &token, 1) != 0) {
            fprintf(stderr, "LLM: Forward pass failed at token %d\n", i);
            break;
        }

        n_generated++;
    }

    return n_generated;
}
