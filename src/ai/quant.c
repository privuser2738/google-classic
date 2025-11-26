/*
 * Holo - Quantization Operations Implementation
 * Dequantization and quantized matrix multiplication
 *
 * Optimized for speed with lookup tables, loop unrolling, and SIMD
 */

#include "quant.h"
#include <math.h>
#include <string.h>

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

#if defined(_MSC_VER)
    #define HOLO_ALIGN(x) __declspec(align(x))
#else
    #define HOLO_ALIGN(x) __attribute__((aligned(x)))
#endif

/* ============================================================================
 * F16 Conversion with Lookup Table
 * ============================================================================ */

/* Union for safe type punning */
typedef union {
    uint32_t u;
    float f;
} float_bits_t;

/* Lookup table for f16 to f32 conversion (64KB) */
static float f16_table[65536];
static int f16_table_init = 0;

static void init_f16_table(void) {
    if (f16_table_init) return;

    for (int i = 0; i < 65536; i++) {
        uint16_t h = (uint16_t)i;
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        float_bits_t fb;

        if (exp == 0) {
            if (mant == 0) {
                fb.u = sign;
            } else {
                /* Denormalized */
                while (!(mant & 0x400)) {
                    mant <<= 1;
                    exp--;
                }
                exp++;
                mant &= ~0x400;
                exp += 127 - 15;
                fb.u = sign | (exp << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            fb.u = sign | 0x7F800000 | (mant << 13);
        } else {
            exp += 127 - 15;
            fb.u = sign | (exp << 23) | (mant << 13);
        }
        f16_table[i] = fb.f;
    }
    f16_table_init = 1;
}

float f16_to_f32(uint16_t h) {
    if (!f16_table_init) init_f16_table();
    return f16_table[h];
}

uint16_t f32_to_f16(float f) {
    float_bits_t fb;
    fb.f = f;
    uint32_t x = fb.u;

    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;

    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t)sign;
        }
        mant = (mant | 0x800000) >> (1 - exp);
        return (uint16_t)(sign | (mant >> 13));
    } else if (exp == 0xFF - 127 + 15) {
        if (mant == 0) {
            return (uint16_t)(sign | 0x7C00); /* Inf */
        }
        return (uint16_t)(sign | 0x7C00 | (mant >> 13)); /* NaN */
    }

    if (exp > 30) {
        return (uint16_t)(sign | 0x7C00); /* Overflow to Inf */
    }

    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

/* ============================================================================
 * Q8_0 Dequantization
 * 32 values per block: f16 scale + 32 x int8
 * ============================================================================ */

void dequantize_row_q8_0(const block_q8_0 *x, float *y, int k) {
    int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        float d = f16_to_f32(x[i].d);

        for (int j = 0; j < QK8_0; j++) {
            y[i * QK8_0 + j] = d * x[i].qs[j];
        }
    }
}

/* ============================================================================
 * Q4_0 Dequantization
 * 32 values per block: f16 scale + 16 bytes (32 x 4-bit values)
 * Values are unsigned 0-15, subtract 8 to get signed -8 to 7
 * ============================================================================ */

void dequantize_row_q4_0(const block_q4_0 *x, float *y, int k) {
    int nb = k / QK4_0;

    for (int i = 0; i < nb; i++) {
        float d = f16_to_f32(x[i].d);

        for (int j = 0; j < 16; j++) {
            uint8_t byte = x[i].qs[j];
            /* Low nibble */
            int8_t v0 = (byte & 0x0F) - 8;
            /* High nibble */
            int8_t v1 = (byte >> 4) - 8;

            y[i * QK4_0 + j] = d * v0;
            y[i * QK4_0 + j + 16] = d * v1;
        }
    }
}

/* ============================================================================
 * Q4_1 Dequantization
 * 32 values per block: f16 scale + f16 min + 16 bytes
 * Values are unsigned 0-15, scaled and offset
 * ============================================================================ */

void dequantize_row_q4_1(const block_q4_1 *x, float *y, int k) {
    int nb = k / QK4_1;

    for (int i = 0; i < nb; i++) {
        float d = f16_to_f32(x[i].d);
        float m = f16_to_f32(x[i].m);

        for (int j = 0; j < 16; j++) {
            uint8_t byte = x[i].qs[j];
            uint8_t v0 = byte & 0x0F;
            uint8_t v1 = byte >> 4;

            y[i * QK4_1 + j] = d * v0 + m;
            y[i * QK4_1 + j + 16] = d * v1 + m;
        }
    }
}

/* ============================================================================
 * Q5_0 Dequantization
 * 32 values: f16 scale + 4 bytes high bits + 16 bytes low 4-bit
 * ============================================================================ */

void dequantize_row_q5_0(const block_q5_0 *x, float *y, int k) {
    int nb = k / QK5_0;

    for (int i = 0; i < nb; i++) {
        float d = f16_to_f32(x[i].d);
        uint32_t qh = 0;
        memcpy(&qh, x[i].qh, 4);

        for (int j = 0; j < 16; j++) {
            uint8_t byte = x[i].qs[j];

            /* Low nibble with 5th bit */
            int8_t v0 = (byte & 0x0F) | (((qh >> j) & 1) << 4);
            v0 -= 16;

            /* High nibble with 5th bit */
            int8_t v1 = (byte >> 4) | (((qh >> (j + 16)) & 1) << 4);
            v1 -= 16;

            y[i * QK5_0 + j] = d * v0;
            y[i * QK5_0 + j + 16] = d * v1;
        }
    }
}

/* ============================================================================
 * Q5_1 Dequantization
 * 32 values: f16 scale + f16 min + 4 bytes high bits + 16 bytes low 4-bit
 * ============================================================================ */

void dequantize_row_q5_1(const block_q5_1 *x, float *y, int k) {
    int nb = k / QK5_1;

    for (int i = 0; i < nb; i++) {
        float d = f16_to_f32(x[i].d);
        float m = f16_to_f32(x[i].m);
        uint32_t qh = 0;
        memcpy(&qh, x[i].qh, 4);

        for (int j = 0; j < 16; j++) {
            uint8_t byte = x[i].qs[j];

            /* Low nibble with 5th bit */
            uint8_t v0 = (byte & 0x0F) | (((qh >> j) & 1) << 4);
            /* High nibble with 5th bit */
            uint8_t v1 = (byte >> 4) | (((qh >> (j + 16)) & 1) << 4);

            y[i * QK5_1 + j] = d * v0 + m;
            y[i * QK5_1 + j + 16] = d * v1 + m;
        }
    }
}

/* ============================================================================
 * Q6_K Dequantization (K-quant super-block: 256 values)
 * ============================================================================ */

void dequantize_row_q6_k(const block_q6_k *x, float *y, int k) {
    int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        float d = f16_to_f32(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t *scales = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = n / 16 + l / 16;
                float sc = d * scales[is];

                int idx = n + l;
                uint8_t q_lo = ql[idx / 2];
                uint8_t q_hi = qh[idx / 4];

                int shift_lo = (idx % 2) * 4;
                int shift_hi = (idx % 4) * 2;

                int8_t q = ((q_lo >> shift_lo) & 0xF) | (((q_hi >> shift_hi) & 0x3) << 4);
                q -= 32;

                y[i * QK_K + idx] = sc * q;
            }

            for (int l = 32; l < 64; l++) {
                int is = n / 16 + l / 16;
                float sc = d * scales[is];

                int idx = n + l;
                uint8_t q_lo = ql[(idx - 32) / 2 + 64];
                uint8_t q_hi = qh[(idx - 32) / 4 + 32];

                int shift_lo = ((idx - 32) % 2) * 4;
                int shift_hi = ((idx - 32) % 4) * 2;

                int8_t q = ((q_lo >> shift_lo) & 0xF) | (((q_hi >> shift_hi) & 0x3) << 4);
                q -= 32;

                y[i * QK_K + idx] = sc * q;
            }
        }
    }
}

/* ============================================================================
 * Q4_K Dequantization (K-quant super-block: 256 values)
 * ============================================================================ */

void dequantize_row_q4_k(const block_q4_k *x, float *y, int k) {
    int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        float d = f16_to_f32(x[i].d);
        float dmin = f16_to_f32(x[i].dmin);
        const uint8_t *scales_packed = x[i].scales;
        const uint8_t *qs = x[i].qs;

        /* Unpack scales and mins (6-bit each, packed into 12 bytes) */
        uint8_t scales[8];
        uint8_t mins[8];

        for (int j = 0; j < 4; j++) {
            scales[j] = scales_packed[j] & 0x3F;
            mins[j] = scales_packed[j + 4] & 0x3F;
        }
        for (int j = 0; j < 4; j++) {
            scales[j + 4] = ((scales_packed[j + 8] & 0xF) << 2) | (scales_packed[j] >> 6);
            mins[j + 4] = ((scales_packed[j + 8] >> 4) << 2) | (scales_packed[j + 4] >> 6);
        }

        /* Dequantize */
        for (int j = 0; j < QK_K / 32; j++) {
            float sc = d * scales[j];
            float m = dmin * mins[j];

            for (int l = 0; l < 16; l++) {
                uint8_t byte = qs[j * 16 + l];
                y[i * QK_K + j * 32 + l] = sc * (byte & 0xF) - m;
                y[i * QK_K + j * 32 + l + 16] = sc * (byte >> 4) - m;
            }
        }
    }
}

/* ============================================================================
 * Q5_K Dequantization
 * ============================================================================ */

void dequantize_row_q5_k(const block_q5_k *x, float *y, int k) {
    int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        float d = f16_to_f32(x[i].d);
        float dmin = f16_to_f32(x[i].dmin);
        const uint8_t *scales_packed = x[i].scales;
        const uint8_t *qh = x[i].qh;
        const uint8_t *qs = x[i].qs;

        /* Unpack scales and mins */
        uint8_t scales[8];
        uint8_t mins[8];

        for (int j = 0; j < 4; j++) {
            scales[j] = scales_packed[j] & 0x3F;
            mins[j] = scales_packed[j + 4] & 0x3F;
        }
        for (int j = 0; j < 4; j++) {
            scales[j + 4] = ((scales_packed[j + 8] & 0xF) << 2) | (scales_packed[j] >> 6);
            mins[j + 4] = ((scales_packed[j + 8] >> 4) << 2) | (scales_packed[j + 4] >> 6);
        }

        /* Dequantize with 5th bit from qh */
        for (int j = 0; j < QK_K / 32; j++) {
            float sc = d * scales[j];
            float m = dmin * mins[j];

            for (int l = 0; l < 16; l++) {
                uint8_t byte = qs[j * 16 + l];
                uint8_t h = qh[j * 4 + l / 4];

                int h_shift0 = (l % 4) * 2;
                int h_shift1 = h_shift0 + 1;

                uint8_t v0 = (byte & 0xF) | (((h >> h_shift0) & 1) << 4);
                uint8_t v1 = (byte >> 4) | (((h >> h_shift1) & 1) << 4);

                y[i * QK_K + j * 32 + l] = sc * v0 - m;
                y[i * QK_K + j * 32 + l + 16] = sc * v1 - m;
            }
        }
    }
}

/* ============================================================================
 * Q3_K Dequantization
 * ============================================================================ */

void dequantize_row_q3_k(const block_q3_k *x, float *y, int k) {
    int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        float d = f16_to_f32(x[i].d);
        const uint8_t *hmask = x[i].hmask;
        const uint8_t *qs = x[i].qs;
        const uint8_t *scales_packed = x[i].scales;

        /* Unpack scales (stored as 6-bit values) */
        int8_t scales[16];
        for (int j = 0; j < 8; j++) {
            scales[j] = (scales_packed[j] & 0xF) - 8;
            scales[j + 8] = (scales_packed[j] >> 4) - 8;
        }

        /* Dequantize */
        for (int j = 0; j < QK_K; j += 8) {
            int is = j / 16;
            float sc = d * scales[is];

            for (int l = 0; l < 8; l++) {
                int idx = j + l;
                uint8_t q2 = (qs[idx / 4] >> ((idx % 4) * 2)) & 0x3;
                uint8_t h = (hmask[idx / 8] >> (idx % 8)) & 1;
                int8_t q = (int8_t)(q2 | (h << 2)) - 4;
                y[i * QK_K + idx] = sc * q;
            }
        }
    }
}

/* ============================================================================
 * Q2_K Dequantization
 * ============================================================================ */

void dequantize_row_q2_k(const block_q2_k *x, float *y, int k) {
    int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        float d = f16_to_f32(x[i].d);
        float dmin = f16_to_f32(x[i].dmin);
        const uint8_t *scales_packed = x[i].scales;
        const uint8_t *qs = x[i].qs;

        /* Unpack scales and mins (4-bit each) */
        uint8_t scales[16];
        uint8_t mins[16];
        for (int j = 0; j < 16; j++) {
            scales[j] = scales_packed[j] & 0xF;
            mins[j] = scales_packed[j] >> 4;
        }

        /* Dequantize */
        for (int j = 0; j < QK_K / 16; j++) {
            float sc = d * scales[j];
            float m = dmin * mins[j];

            for (int l = 0; l < 16; l++) {
                int idx = j * 16 + l;
                uint8_t q = (qs[idx / 4] >> ((idx % 4) * 2)) & 0x3;
                y[i * QK_K + idx] = sc * q - m;
            }
        }
    }
}

/* ============================================================================
 * Quantized Vector Dot Products
 * Compute dot product without full dequantization for better performance
 * ============================================================================ */

#if defined(HOLO_USE_AVX2)
/* AVX2 optimized Q8_0 dot product */
float vec_dot_q8_0_f32(const block_q8_0 *x, const float *y, int k) {
    int nb = k / QK8_0;
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; i++) {
        float d = f16_to_f32(x[i].d);
        const int8_t *qs = x[i].qs;
        const float *yb = y + i * QK8_0;

        /* Load 32 int8 values and convert to floats, multiply with y */
        /* Process 8 values at a time */
        __m256 sum = _mm256_setzero_ps();

        for (int j = 0; j < 32; j += 8) {
            /* Load 8 int8 values and convert to int32 */
            __m128i q8 = _mm_loadl_epi64((const __m128i *)(qs + j));
            __m256i q32 = _mm256_cvtepi8_epi32(q8);
            __m256 qf = _mm256_cvtepi32_ps(q32);

            /* Load 8 floats from y */
            __m256 yv = _mm256_loadu_ps(yb + j);

            /* Multiply and accumulate */
            sum = _mm256_fmadd_ps(qf, yv, sum);
        }

        /* Horizontal sum and multiply by scale */
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 sum128 = _mm_add_ps(hi, lo);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float block_sum = _mm_cvtss_f32(sum128);

        acc = _mm256_add_ps(acc, _mm256_set1_ps(d * block_sum));
    }

    /* Final horizontal sum */
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}
#else
/* Scalar fallback with loop unrolling */
float vec_dot_q8_0_f32(const block_q8_0 *x, const float *y, int k) {
    int nb = k / QK8_0;
    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        float d = f16_to_f32(x[i].d);
        const int8_t *qs = x[i].qs;
        const float *yb = y + i * QK8_0;

        /* Unroll by 8 for better performance */
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;

        for (int j = 0; j < 32; j += 8) {
            s0 += qs[j+0] * yb[j+0];
            s1 += qs[j+1] * yb[j+1];
            s2 += qs[j+2] * yb[j+2];
            s3 += qs[j+3] * yb[j+3];
            s4 += qs[j+4] * yb[j+4];
            s5 += qs[j+5] * yb[j+5];
            s6 += qs[j+6] * yb[j+6];
            s7 += qs[j+7] * yb[j+7];
        }

        sum += d * (s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7);
    }

    return sum;
}
#endif

float vec_dot_q4_0_f32(const block_q4_0 *x, const float *y, int k) {
    int nb = k / QK4_0;
    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        float d = f16_to_f32(x[i].d);
        float block_sum = 0.0f;

        for (int j = 0; j < 16; j++) {
            uint8_t byte = x[i].qs[j];
            int8_t v0 = (byte & 0x0F) - 8;
            int8_t v1 = (byte >> 4) - 8;

            block_sum += v0 * y[i * QK4_0 + j];
            block_sum += v1 * y[i * QK4_0 + j + 16];
        }

        sum += d * block_sum;
    }

    return sum;
}

float vec_dot_q4_k_f32(const block_q4_k *x, const float *y, int k) {
    int nb = k / QK_K;
    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        float d = f16_to_f32(x[i].d);
        float dmin = f16_to_f32(x[i].dmin);
        const uint8_t *scales_packed = x[i].scales;
        const uint8_t *qs = x[i].qs;

        /* Unpack scales and mins */
        uint8_t scales[8];
        uint8_t mins[8];

        for (int j = 0; j < 4; j++) {
            scales[j] = scales_packed[j] & 0x3F;
            mins[j] = scales_packed[j + 4] & 0x3F;
        }
        for (int j = 0; j < 4; j++) {
            scales[j + 4] = ((scales_packed[j + 8] & 0xF) << 2) | (scales_packed[j] >> 6);
            mins[j + 4] = ((scales_packed[j + 8] >> 4) << 2) | (scales_packed[j + 4] >> 6);
        }

        /* Compute dot product */
        for (int j = 0; j < QK_K / 32; j++) {
            float sc = d * scales[j];
            float m = dmin * mins[j];

            for (int l = 0; l < 16; l++) {
                uint8_t byte = qs[j * 16 + l];
                int base = i * QK_K + j * 32;

                sum += (sc * (byte & 0xF) - m) * y[base + l];
                sum += (sc * (byte >> 4) - m) * y[base + l + 16];
            }
        }
    }

    return sum;
}

float vec_dot_q6_k_f32(const block_q6_k *x, const float *y, int k) {
    /* For Q6_K, full dequantization is complex, so we use a simpler approach */
    int nb = k / QK_K;
    float sum = 0.0f;

    /* Temporary buffer for dequantization */
    float tmp[QK_K];

    for (int i = 0; i < nb; i++) {
        /* Dequantize one block */
        dequantize_row_q6_k(&x[i], tmp, QK_K);

        /* Compute dot product */
        for (int j = 0; j < QK_K; j++) {
            sum += tmp[j] * y[i * QK_K + j];
        }
    }

    return sum;
}

/* ============================================================================
 * Quantized Matrix-Vector Multiplication
 * dst = M @ v where M is quantized
 * ============================================================================ */

void mat_vec_q8_0(float *dst, const void *M, const float *v, int rows, int cols) {
    const block_q8_0 *blocks = (const block_q8_0 *)M;
    int blocks_per_row = cols / QK8_0;

    for (int row = 0; row < rows; row++) {
        dst[row] = vec_dot_q8_0_f32(&blocks[row * blocks_per_row], v, cols);
    }
}

void mat_vec_q4_0(float *dst, const void *M, const float *v, int rows, int cols) {
    const block_q4_0 *blocks = (const block_q4_0 *)M;
    int blocks_per_row = cols / QK4_0;

    for (int row = 0; row < rows; row++) {
        dst[row] = vec_dot_q4_0_f32(&blocks[row * blocks_per_row], v, cols);
    }
}

void mat_vec_q4_k(float *dst, const void *M, const float *v, int rows, int cols) {
    const block_q4_k *blocks = (const block_q4_k *)M;
    int blocks_per_row = cols / QK_K;

    for (int row = 0; row < rows; row++) {
        dst[row] = vec_dot_q4_k_f32(&blocks[row * blocks_per_row], v, cols);
    }
}

void mat_vec_q6_k(float *dst, const void *M, const float *v, int rows, int cols) {
    const block_q6_k *blocks = (const block_q6_k *)M;
    int blocks_per_row = cols / QK_K;

    for (int row = 0; row < rows; row++) {
        dst[row] = vec_dot_q6_k_f32(&blocks[row * blocks_per_row], v, cols);
    }
}

/* ============================================================================
 * Generic Quantized MatMul Dispatcher
 * ============================================================================ */

/* Quantization type enum (matches GGML) */
#define GGML_TYPE_F32  0
#define GGML_TYPE_F16  1
#define GGML_TYPE_Q4_0 2
#define GGML_TYPE_Q4_1 3
#define GGML_TYPE_Q5_0 6
#define GGML_TYPE_Q5_1 7
#define GGML_TYPE_Q8_0 8
#define GGML_TYPE_Q2_K 10
#define GGML_TYPE_Q3_K 11
#define GGML_TYPE_Q4_K 12
#define GGML_TYPE_Q5_K 13
#define GGML_TYPE_Q6_K 14

void mat_vec_quant(float *dst, const void *M, const float *v,
                   int rows, int cols, int quant_type) {
    switch (quant_type) {
        case GGML_TYPE_Q8_0:
            mat_vec_q8_0(dst, M, v, rows, cols);
            break;
        case GGML_TYPE_Q4_0:
            mat_vec_q4_0(dst, M, v, rows, cols);
            break;
        case GGML_TYPE_Q4_K:
            mat_vec_q4_k(dst, M, v, rows, cols);
            break;
        case GGML_TYPE_Q6_K:
            mat_vec_q6_k(dst, M, v, rows, cols);
            break;
        default:
            /* Unsupported type - zero output */
            for (int i = 0; i < rows; i++) {
                dst[i] = 0.0f;
            }
            break;
    }
}

/* ============================================================================
 * Block Size Helpers
 * ============================================================================ */

int quant_block_size(int type) {
    switch (type) {
        case GGML_TYPE_Q8_0: return QK8_0;
        case GGML_TYPE_Q4_0: return QK4_0;
        case GGML_TYPE_Q4_1: return QK4_1;
        case GGML_TYPE_Q5_0: return QK5_0;
        case GGML_TYPE_Q5_1: return QK5_1;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K: return QK_K;
        default: return 1;
    }
}

size_t quant_block_bytes(int type) {
    switch (type) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_Q8_0: return sizeof(block_q8_0);
        case GGML_TYPE_Q4_0: return sizeof(block_q4_0);
        case GGML_TYPE_Q4_1: return sizeof(block_q4_1);
        case GGML_TYPE_Q5_0: return sizeof(block_q5_0);
        case GGML_TYPE_Q5_1: return sizeof(block_q5_1);
        case GGML_TYPE_Q2_K: return sizeof(block_q2_k);
        case GGML_TYPE_Q3_K: return sizeof(block_q3_k);
        case GGML_TYPE_Q4_K: return sizeof(block_q4_k);
        case GGML_TYPE_Q5_K: return sizeof(block_q5_k);
        case GGML_TYPE_Q6_K: return sizeof(block_q6_k);
        default: return 0;
    }
}

size_t quant_row_size(int type, int cols) {
    int block_size = quant_block_size(type);
    size_t block_bytes = quant_block_bytes(type);

    if (type == GGML_TYPE_F32) return cols * 4;
    if (type == GGML_TYPE_F16) return cols * 2;

    int n_blocks = (cols + block_size - 1) / block_size;
    return n_blocks * block_bytes;
}
