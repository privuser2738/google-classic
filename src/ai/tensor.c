/*
 * Holo - Tensor Operations Implementation
 * Pure C implementation for neural network computations
 * With SIMD optimizations for AVX2/AVX/SSE2
 */

#include "tensor.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

/* SIMD support detection */
#if defined(__AVX2__)
    #define HOLO_USE_AVX2 1
    #include <immintrin.h>
#elif defined(__AVX__)
    #define HOLO_USE_AVX 1
    #include <immintrin.h>
#elif defined(__SSE2__)
    #define HOLO_USE_SSE2 1
    #include <emmintrin.h>
#endif

/* ============================================================================
 * Q8_0 Quantization Format
 * Block of 32 values: 2-byte scale (f16) + 32 x 1-byte values
 * ============================================================================ */

typedef struct {
    uint16_t d;       /* delta (scale) as f16 */
    int8_t qs[32];    /* quantized values */
} block_q8_0;

/* ============================================================================
 * Q4_0 Quantization Format
 * Block of 32 values: 2-byte scale (f16) + 16 bytes (32 x 4-bit values)
 * ============================================================================ */

typedef struct {
    uint16_t d;       /* delta (scale) as f16 */
    uint8_t qs[16];   /* 32 x 4-bit values */
} block_q4_0;

/* ============================================================================
 * F16 Conversion
 * ============================================================================ */

/* Union for safe type punning */
typedef union {
    uint32_t u;
    float f;
} float_bits_t;

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    float_bits_t fb;

    if (exp == 0) {
        if (mant == 0) {
            fb.u = sign;
            return fb.f;
        }
        /* Denormalized */
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x400;
    } else if (exp == 31) {
        /* Inf/NaN */
        fb.u = sign | 0x7F800000 | (mant << 13);
        return fb.f;
    }

    exp += 127 - 15;
    fb.u = sign | (exp << 23) | (mant << 13);
    return fb.f;
}

/* ============================================================================
 * Basic Vector Operations
 * ============================================================================ */

#if defined(HOLO_USE_AVX2) || defined(HOLO_USE_AVX)
void vec_add(float *dst, const float *a, const float *b, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(dst + i, vc);
    }
    for (; i < n; i++) {
        dst[i] = a[i] + b[i];
    }
}
#else
void vec_add(float *dst, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = a[i] + b[i];
    }
}
#endif

void vec_scale(float *dst, const float *a, float scale, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = a[i] * scale;
    }
}

#if defined(HOLO_USE_AVX2) || defined(HOLO_USE_AVX)
void vec_mul(float *dst, const float *a, const float *b, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(dst + i, vc);
    }
    for (; i < n; i++) {
        dst[i] = a[i] * b[i];
    }
}

float vec_dot(const float *a, const float *b, int n) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;

    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        #if defined(HOLO_USE_AVX2)
        sum = _mm256_fmadd_ps(va, vb, sum);
        #else
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
        #endif
    }

    /* Horizontal sum */
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);

    /* Handle remaining elements */
    for (; i < n; i++) {
        result += a[i] * b[i];
    }

    return result;
}
#else
void vec_mul(float *dst, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = a[i] * b[i];
    }
}

float vec_dot(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
#endif

void vec_copy(float *dst, const float *src, int n) {
    memcpy(dst, src, n * sizeof(float));
}

void vec_fill(float *dst, float val, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = val;
    }
}

/* ============================================================================
 * Matrix Operations
 * ============================================================================ */

void mat_vec_mul(float *dst, const float *M, const float *v, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        dst[i] = vec_dot(M + i * cols, v, cols);
    }
}

void mat_vec_mul_q8_0(float *dst, const void *M, const float *v, int rows, int cols) {
    const block_q8_0 *blocks = (const block_q8_0 *)M;
    int blocks_per_row = cols / 32;

    for (int row = 0; row < rows; row++) {
        float sum = 0.0f;

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q8_0 *block = &blocks[row * blocks_per_row + b];
            float scale = f16_to_f32(block->d);

            for (int i = 0; i < 32; i++) {
                sum += scale * block->qs[i] * v[b * 32 + i];
            }
        }

        dst[row] = sum;
    }
}

/* ============================================================================
 * Activation Functions
 * ============================================================================ */

void silu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

void gelu(float *x, int n) {
    const float sqrt_2_pi = 0.7978845608f;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(sqrt_2_pi * (v + 0.044715f * v * v * v)));
    }
}

void relu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i] < 0) x[i] = 0;
    }
}

/* ============================================================================
 * Normalization
 * ============================================================================ */

void rms_norm(float *dst, const float *x, const float *weight, int n, float eps) {
    /* Calculate RMS */
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = sqrtf(sum_sq / n + eps);

    /* Normalize and apply weight */
    float scale = 1.0f / rms;
    for (int i = 0; i < n; i++) {
        dst[i] = x[i] * scale * weight[i];
    }
}

/* Gemma-style RMS norm: uses (1 + weight) instead of weight
 * Gemma models initialize norm weights to 0 for effective scale of 1 */
void rms_norm_gemma(float *dst, const float *x, const float *weight, int n, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = sqrtf(sum_sq / n + eps);
    float scale = 1.0f / rms;
    for (int i = 0; i < n; i++) {
        dst[i] = x[i] * scale * (1.0f + weight[i]);
    }
}

void layer_norm(float *dst, const float *x, const float *weight, const float *bias, int n, float eps) {
    /* Calculate mean */
    float mean = 0.0f;
    for (int i = 0; i < n; i++) {
        mean += x[i];
    }
    mean /= n;

    /* Calculate variance */
    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= n;

    /* Normalize */
    float scale = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; i++) {
        dst[i] = (x[i] - mean) * scale;
        if (weight) dst[i] *= weight[i];
        if (bias) dst[i] += bias[i];
    }
}

/* ============================================================================
 * Softmax
 * ============================================================================ */

void softmax(float *dst, const float *x, int n) {
    /* Find max for numerical stability */
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    /* Compute exp and sum */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        dst[i] = expf(x[i] - max_val);
        sum += dst[i];
    }

    /* Normalize */
    float scale = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        dst[i] *= scale;
    }
}

void softmax_temp(float *dst, const float *x, int n, float temperature) {
    /* Find max for numerical stability */
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    /* Compute exp with temperature and sum */
    float inv_temp = 1.0f / temperature;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        dst[i] = expf((x[i] - max_val) * inv_temp);
        sum += dst[i];
    }

    /* Normalize */
    float scale = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        dst[i] *= scale;
    }
}

/* ============================================================================
 * Rotary Position Embedding (RoPE)
 * ============================================================================ */

void rope_apply(float *q, float *k, int pos, int head_dim, int n_heads, float freq_base) {
    int half_dim = head_dim / 2;

    for (int h = 0; h < n_heads; h++) {
        float *q_head = q + h * head_dim;
        float *k_head = k + h * head_dim;

        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / powf(freq_base, (float)(2 * i) / head_dim);
            float theta = pos * freq;
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);

            /* Apply rotation to q */
            float q0 = q_head[i];
            float q1 = q_head[i + half_dim];
            q_head[i] = q0 * cos_t - q1 * sin_t;
            q_head[i + half_dim] = q0 * sin_t + q1 * cos_t;

            /* Apply rotation to k */
            float k0 = k_head[i];
            float k1 = k_head[i + half_dim];
            k_head[i] = k0 * cos_t - k1 * sin_t;
            k_head[i + half_dim] = k0 * sin_t + k1 * cos_t;
        }
    }
}

/* ============================================================================
 * Attention
 * ============================================================================ */

void attention_scores(float *dst, const float *Q, const float *K,
                      int seq_len, int kv_len, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < kv_len; j++) {
            float score = vec_dot(Q + i * head_dim, K + j * head_dim, head_dim);
            dst[i * kv_len + j] = score * scale;
        }
    }
}

void attention_apply(float *dst, const float *scores, const float *V,
                     int seq_len, int kv_len, int head_dim) {
    for (int i = 0; i < seq_len; i++) {
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int j = 0; j < kv_len; j++) {
                sum += scores[i * kv_len + j] * V[j * head_dim + d];
            }
            dst[i * head_dim + d] = sum;
        }
    }
}

void causal_mask(float *scores, int seq_len, int kv_len, int pos) {
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < kv_len; j++) {
            if (j > pos + i) {
                scores[i * kv_len + j] = -INFINITY;
            }
        }
    }
}

/* ============================================================================
 * Dequantization
 * ============================================================================ */

void dequant_q8_0(float *dst, const void *src, int n) {
    const block_q8_0 *blocks = (const block_q8_0 *)src;
    int n_blocks = n / 32;

    for (int b = 0; b < n_blocks; b++) {
        float scale = f16_to_f32(blocks[b].d);
        for (int i = 0; i < 32; i++) {
            dst[b * 32 + i] = scale * blocks[b].qs[i];
        }
    }
}

void dequant_q4_0(float *dst, const void *src, int n) {
    const block_q4_0 *blocks = (const block_q4_0 *)src;
    int n_blocks = n / 32;

    for (int b = 0; b < n_blocks; b++) {
        float scale = f16_to_f32(blocks[b].d);
        for (int i = 0; i < 16; i++) {
            uint8_t byte = blocks[b].qs[i];
            /* Low 4 bits */
            int8_t v0 = (byte & 0x0F) - 8;
            /* High 4 bits */
            int8_t v1 = (byte >> 4) - 8;

            dst[b * 32 + i] = scale * v0;
            dst[b * 32 + i + 16] = scale * v1;
        }
    }
}

void dequant_f16(float *dst, const void *src, int n) {
    const uint16_t *f16 = (const uint16_t *)src;
    for (int i = 0; i < n; i++) {
        dst[i] = f16_to_f32(f16[i]);
    }
}

/* ============================================================================
 * Sampling
 * ============================================================================ */

int argmax(const float *x, int n) {
    int max_idx = 0;
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
            max_idx = i;
        }
    }
    return max_idx;
}

int sample_prob(const float *probs, int n, float rand_val) {
    float cumsum = 0.0f;
    for (int i = 0; i < n; i++) {
        cumsum += probs[i];
        if (rand_val < cumsum) {
            return i;
        }
    }
    return n - 1;
}


/* Quickselect partition helper for top_k */
static int partition_desc(float *arr, int *indices, int left, int right, int pivot_idx) {
    float pivot_val = arr[indices[pivot_idx]];
    /* Move pivot to end */
    int tmp = indices[pivot_idx];
    indices[pivot_idx] = indices[right];
    indices[right] = tmp;

    int store_idx = left;
    for (int i = left; i < right; i++) {
        if (arr[indices[i]] > pivot_val) {  /* Descending order */
            tmp = indices[store_idx];
            indices[store_idx] = indices[i];
            indices[i] = tmp;
            store_idx++;
        }
    }
    /* Move pivot to final position */
    tmp = indices[store_idx];
    indices[store_idx] = indices[right];
    indices[right] = tmp;
    return store_idx;
}

/* Quickselect to find k-th largest element - O(n) average */
static void quickselect_k(float *arr, int *indices, int left, int right, int k) {
    while (left < right) {
        /* Choose pivot as median of left, mid, right */
        int mid = left + (right - left) / 2;
        int pivot_idx = mid;

        int pos = partition_desc(arr, indices, left, right, pivot_idx);

        if (pos == k) {
            return;
        } else if (pos < k) {
            left = pos + 1;
        } else {
            right = pos - 1;
        }
    }
}

void top_k(float *probs, int n, int k) {
    if (k >= n) return;

    /* Use quickselect to partition: indices[0..k-1] will have top k elements
     * quickselect positions the k-th element at index k-1, with all larger
     * elements before it (unsorted) and all smaller elements after it */
    int *indices = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) indices[i] = i;

    /* Partition so indices[0..k-1] contains top k (unsorted) */
    quickselect_k(probs, indices, 0, n - 1, k - 1);

    /* Find minimum of top k elements (the k-th largest value)
     * This is the threshold - anything below this gets zeroed */
    float threshold = probs[indices[0]];
    for (int i = 1; i < k; i++) {
        if (probs[indices[i]] < threshold) {
            threshold = probs[indices[i]];
        }
    }

    /* Zero out non-top-k and compute sum */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        if (probs[i] < threshold) {
            probs[i] = 0.0f;
        } else {
            sum += probs[i];
        }
    }

    /* Renormalize */
    if (sum > 0) {
        float scale = 1.0f / sum;
        for (int i = 0; i < n; i++) {
            probs[i] *= scale;
        }
    }

    free(indices);
}

/* Quicksort comparison function helper - descending order */
static int compare_probs_desc(const void *a, const void *b, void *probs_arr) {
    float *probs = (float *)probs_arr;
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    float diff = probs[ib] - probs[ia];  /* Descending */
    if (diff < 0) return -1;
    if (diff > 0) return 1;
    return 0;
}

/* In-place quicksort for indices array sorted by probs descending */
static void quicksort_indices(int *indices, float *probs, int left, int right) {
    if (left >= right) return;

    /* Choose pivot and partition */
    int mid = left + (right - left) / 2;
    float pivot_val = probs[indices[mid]];

    /* Move pivot to end */
    int tmp = indices[mid];
    indices[mid] = indices[right];
    indices[right] = tmp;

    int store = left;
    for (int i = left; i < right; i++) {
        if (probs[indices[i]] > pivot_val) {  /* Descending */
            tmp = indices[store];
            indices[store] = indices[i];
            indices[i] = tmp;
            store++;
        }
    }
    tmp = indices[store];
    indices[store] = indices[right];
    indices[right] = tmp;

    quicksort_indices(indices, probs, left, store - 1);
    quicksort_indices(indices, probs, store + 1, right);
}

void top_p(float *probs, int n, float p) {
    /* For top_p, we need sorted probabilities to accumulate to threshold p
     * Use O(n log n) quicksort instead of O(n^2) bubble sort */
    int *indices = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) indices[i] = i;

    /* Sort indices by probability descending - O(n log n) */
    quicksort_indices(indices, probs, 0, n - 1);

    /* Find cutoff where cumulative probability exceeds p */
    float cumsum = 0.0f;
    int cutoff = n;
    for (int i = 0; i < n; i++) {
        cumsum += probs[indices[i]];
        if (cumsum > p) {
            cutoff = i + 1;
            break;
        }
    }

    /* Zero out tokens below cutoff */
    for (int i = cutoff; i < n; i++) {
        probs[indices[i]] = 0.0f;
    }

    /* Renormalize */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += probs[i];
    if (sum > 0) {
        float scale = 1.0f / sum;
        for (int i = 0; i < n; i++) probs[i] *= scale;
    }

    free(indices);
}

void apply_temperature(float *logits, int n, float temp) {
    if (temp <= 0.0f) temp = 1.0f;
    float scale = 1.0f / temp;
    for (int i = 0; i < n; i++) {
        logits[i] *= scale;
    }
}

void apply_repetition_penalty(float *logits, const int *tokens, int n_tokens, int vocab_size, float penalty) {
    if (penalty == 1.0f) return;

    for (int i = 0; i < n_tokens; i++) {
        int token = tokens[i];
        if (token >= 0 && token < vocab_size) {
            if (logits[token] > 0) {
                logits[token] /= penalty;
            } else {
                logits[token] *= penalty;
            }
        }
    }
}
