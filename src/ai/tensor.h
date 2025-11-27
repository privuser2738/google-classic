/*
 * Holo - Tensor Operations
 * Pure C implementation for neural network computations
 */

#ifndef HOLO_TENSOR_H
#define HOLO_TENSOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Basic Tensor Operations
 * ============================================================================ */

/* Vector addition: dst = a + b */
void vec_add(float *dst, const float *a, const float *b, int n);

/* Vector scale: dst = a * scale */
void vec_scale(float *dst, const float *a, float scale, int n);

/* Element-wise multiply: dst = a * b */
void vec_mul(float *dst, const float *a, const float *b, int n);

/* Dot product: sum(a * b) */
float vec_dot(const float *a, const float *b, int n);

/* Copy: dst = src */
void vec_copy(float *dst, const float *src, int n);

/* Fill with value: dst[i] = val */
void vec_fill(float *dst, float val, int n);

/* ============================================================================
 * Matrix Operations
 * ============================================================================ */

/* Matrix-vector multiply: dst = M @ v
 * M is [rows x cols], v is [cols], dst is [rows]
 */
void mat_vec_mul(float *dst, const float *M, const float *v, int rows, int cols);

/* Matrix-vector multiply with quantized matrix (Q8_0) */
void mat_vec_mul_q8_0(float *dst, const void *M, const float *v, int rows, int cols);

/* Matrix-vector multiply with quantized matrix (Q4_K) */
void mat_vec_mul_q4_k(float *dst, const void *M, const float *v, int rows, int cols);

/* ============================================================================
 * Activation Functions
 * ============================================================================ */

/* SiLU (Swish): x * sigmoid(x) */
void silu(float *x, int n);

/* GELU approximation */
void gelu(float *x, int n);

/* ReLU */
void relu(float *x, int n);

/* ============================================================================
 * Normalization
 * ============================================================================ */

/* RMS Normalization: x = x / sqrt(mean(x^2) + eps) * weight */
void rms_norm(float *dst, const float *x, const float *weight, int n, float eps);
/* Gemma-style RMS norm: uses (1 + weight) for scale */
void rms_norm_gemma(float *dst, const float *x, const float *weight, int n, float eps);

/* Layer Normalization */
void layer_norm(float *dst, const float *x, const float *weight, const float *bias, int n, float eps);

/* ============================================================================
 * Softmax
 * ============================================================================ */

/* Softmax: dst[i] = exp(x[i]) / sum(exp(x)) */
void softmax(float *dst, const float *x, int n);

/* Softmax with temperature */
void softmax_temp(float *dst, const float *x, int n, float temperature);

/* ============================================================================
 * Rotary Position Embedding (RoPE)
 * ============================================================================ */

/* Apply RoPE to query/key vectors
 * q, k: [n_heads, head_dim]
 * pos: position in sequence
 * head_dim: dimension per head
 * n_heads: number of heads
 * freq_base: base frequency (typically 10000)
 */
void rope_apply(float *q, float *k, int pos, int head_dim, int n_heads, float freq_base);

/* ============================================================================
 * Attention
 * ============================================================================ */

/* Single-head attention scores: dst = softmax(Q @ K^T / sqrt(d))
 * Q: [seq_len, head_dim]
 * K: [kv_len, head_dim]
 * dst: [seq_len, kv_len]
 */
void attention_scores(float *dst, const float *Q, const float *K,
                      int seq_len, int kv_len, int head_dim);

/* Apply attention: dst = scores @ V
 * scores: [seq_len, kv_len]
 * V: [kv_len, head_dim]
 * dst: [seq_len, head_dim]
 */
void attention_apply(float *dst, const float *scores, const float *V,
                     int seq_len, int kv_len, int head_dim);

/* Causal mask for attention (set future positions to -inf) */
void causal_mask(float *scores, int seq_len, int kv_len, int pos);

/* ============================================================================
 * Dequantization
 * ============================================================================ */

/* Dequantize Q8_0 block to float32 */
void dequant_q8_0(float *dst, const void *src, int n);

/* Dequantize Q4_0 block to float32 */
void dequant_q4_0(float *dst, const void *src, int n);

/* Dequantize Q4_K block to float32 */
void dequant_q4_k(float *dst, const void *src, int n);

/* Dequantize Q6_K block to float32 */
void dequant_q6_k(float *dst, const void *src, int n);

/* Dequantize F16 to float32 */
void dequant_f16(float *dst, const void *src, int n);

/* ============================================================================
 * Sampling
 * ============================================================================ */

/* Argmax: return index of maximum value */
int argmax(const float *x, int n);

/* Sample from probability distribution */
int sample_prob(const float *probs, int n, float rand_val);

/* Top-K sampling: zero out all but top k values, renormalize */
void top_k(float *probs, int n, int k);

/* Top-P (nucleus) sampling: zero out values below cumulative prob p */
void top_p(float *probs, int n, float p);

/* Apply temperature to logits */
void apply_temperature(float *logits, int n, float temp);

/* Apply repetition penalty */
void apply_repetition_penalty(float *logits, const int *tokens, int n_tokens, int vocab_size, float penalty);

#ifdef __cplusplus
}
#endif

#endif /* HOLO_TENSOR_H */
