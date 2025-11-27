/*
 * Holo - CUDA Operations Header
 * GPU-accelerated tensor operations for LLM inference
 */

#ifndef HOLO_CUDA_OPS_H
#define HOLO_CUDA_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

/* CUDA initialization and cleanup */
int cuda_init(void);
void cuda_cleanup(void);
bool cuda_available(void);
void cuda_print_info(void);

/* Get CUDA device properties */
int cuda_get_device_count(void);
size_t cuda_get_free_memory(void);
const char* cuda_get_device_name(void);

/* Memory management */
void* cuda_malloc(size_t size);
void cuda_free(void* ptr);
void cuda_memcpy_h2d(void* dst, const void* src, size_t size);  /* Host to Device */
void cuda_memcpy_d2h(void* dst, const void* src, size_t size);  /* Device to Host */
void cuda_memcpy_d2d(void* dst, const void* src, size_t size);  /* Device to Device */
void cuda_memcpy_d2d_async(void* dst, const void* src, size_t size);  /* Async D2D on copy stream */
void cuda_memcpy_d2h_async(void* dst, const void* src, size_t size);  /* Async D2H on copy stream */
void cuda_memset(void* ptr, int value, size_t size);

/* Synchronization */
void cuda_sync(void);
void cuda_sync_copy(void);  /* Sync only copy stream */

/* Vector operations (float32) */
void cuda_vec_add(float* dst, const float* a, const float* b, int n);
void cuda_vec_mul(float* dst, const float* a, const float* b, int n);
void cuda_vec_scale(float* dst, const float* a, float scale, int n);
float cuda_vec_dot(const float* a, const float* b, int n);

/* Matrix operations */
/* dst[out_dim] = M[in_dim, out_dim]^T @ v[in_dim] (GGUF transposed layout) */
void cuda_matmul_f32(float* dst, const float* M, const float* v, int out_dim, int in_dim);
void cuda_matmul_f16(float* dst, const void* M, const float* v, int out_dim, int in_dim);

/* Quantized matrix operations */
void cuda_matmul_q8_0(float* dst, const void* M, const float* v, int out_dim, int in_dim);
void cuda_matmul_q4_0(float* dst, const void* M, const float* v, int out_dim, int in_dim);
void cuda_matmul_q4_k(float* dst, const void* M, const float* v, int out_dim, int in_dim);
void cuda_matmul_q6_k(float* dst, const void* M, const float* v, int out_dim, int in_dim);

/* Batched QKV projection - computes Q, K, V in single kernel launch */
void cuda_batched_qkv_q8_0(float* q_out, float* k_out, float* v_out,
                           const void* Wq, const void* Wk, const void* Wv,
                           const float* x, int q_dim, int kv_dim, int in_dim);

/* Batched FFN gate+up with fused SiLU - computes SiLU(W1@x) * (W3@x) */
void cuda_batched_ffn_gate_up_q8_0(float* out, const void* W1, const void* W3,
                                   const float* x, int ffn_dim, int in_dim);

/* Activation functions */
void cuda_silu(float* x, int n);
void cuda_silu_mul(float* dst, const float* a, const float* b, int n);  /* Fused SiLU + multiply */
void cuda_gelu(float* x, int n);
void cuda_relu(float* x, int n);

/* Normalization */
void cuda_rms_norm(float* dst, const float* x, const float* weight, int n, float eps);
void cuda_layer_norm(float* dst, const float* x, const float* weight, const float* bias, int n, float eps);

/* Softmax */
void cuda_softmax(float* dst, const float* x, int n);
void cuda_softmax_inplace(float* x, int n);

/* Attention operations */
void cuda_attention_scores(float* dst, const float* Q, const float* K,
                           int seq_len, int kv_len, int head_dim, float scale);
void cuda_attention_apply(float* dst, const float* scores, const float* V,
                          int seq_len, int kv_len, int head_dim);

/* Single-query attention for autoregressive generation
 * Computes attention output for one query against cached KV
 * dst: output [head_dim]
 * q: query vector [head_dim]
 * k_cache: key cache [max_seq * kv_dim] for this layer
 * v_cache: value cache [max_seq * kv_dim] for this layer
 * att_buf: scratch buffer [kv_len] for attention scores
 * kv_len: number of cached positions (pos + 1)
 * head_dim: dimension per head
 * kv_head: which KV head to use
 * kv_dim: n_kv_heads * head_dim
 */
void cuda_attention_single(float* dst, const float* q,
                           const float* k_cache, const float* v_cache,
                           float* att_buf, int kv_len, int head_dim,
                           int kv_head, int kv_dim);

/* Multi-head attention - all heads computed in one kernel launch
 * Much faster than calling cuda_attention_single in a loop!
 * dst: output [n_heads * head_dim]
 * q: all query vectors [n_heads * head_dim]
 * k_cache: key cache for this layer [max_seq * n_kv_heads * head_dim]
 * v_cache: value cache for this layer [max_seq * n_kv_heads * head_dim]
 */
void cuda_multi_head_attention(float* dst, const float* q,
                               const float* k_cache, const float* v_cache,
                               int kv_len, int head_dim, int n_heads, int n_kv_heads);

/* RoPE */
void cuda_rope_apply(float* q, float* k, int pos, int head_dim, int n_heads,
                     int n_kv_heads, float freq_base);

/* Embedding lookup */
void cuda_get_embedding_f32(float* dst, const float* embeddings, int token,
                            int dim, int vocab_size);
void cuda_get_embedding_f16(float* dst, const void* embeddings, int token,
                            int dim, int vocab_size);
void cuda_get_embedding_q8_0(float* dst, const void* embeddings, int token,
                             int dim, int vocab_size);
void cuda_get_embedding_q4_0(float* dst, const void* embeddings, int token,
                             int dim, int vocab_size);
void cuda_get_embedding_q6_k(float* dst, const void* embeddings, int token,
                             int dim, int vocab_size);

/* Sampling helpers */
int cuda_argmax(const float* x, int n);

#ifdef __cplusplus
}
#endif

#endif /* HOLO_CUDA_OPS_H */
