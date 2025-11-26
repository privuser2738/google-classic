/*
 * Holo - Quantization Operations
 * Dequantization and quantized matrix multiplication
 */

#ifndef HOLO_QUANT_H
#define HOLO_QUANT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * GGML Quantization Type Constants
 * ============================================================================ */

#define GGML_TYPE_F32     0
#define GGML_TYPE_F16     1
#define GGML_TYPE_Q4_0    2
#define GGML_TYPE_Q4_1    3
#define GGML_TYPE_Q5_0    6
#define GGML_TYPE_Q5_1    7
#define GGML_TYPE_Q8_0    8
#define GGML_TYPE_Q8_1    9
#define GGML_TYPE_Q2_K   10
#define GGML_TYPE_Q3_K   11
#define GGML_TYPE_Q4_K   12
#define GGML_TYPE_Q5_K   13
#define GGML_TYPE_Q6_K   14
#define GGML_TYPE_Q8_K   15

/* ============================================================================
 * Quantization Block Structures
 * ============================================================================ */

/* Q8_0: 32 values per block, 34 bytes total
 * - 2 bytes: f16 scale (d)
 * - 32 bytes: int8 quantized values
 */
typedef struct {
    uint16_t d;        /* f16 scale */
    int8_t qs[32];     /* quantized values */
} block_q8_0;

#define QK8_0 32

/* Q4_0: 32 values per block, 18 bytes total
 * - 2 bytes: f16 scale (d)
 * - 16 bytes: 32 x 4-bit values (packed)
 */
typedef struct {
    uint16_t d;        /* f16 scale */
    uint8_t qs[16];    /* 4-bit quantized values */
} block_q4_0;

#define QK4_0 32

/* Q4_1: 32 values per block, 20 bytes total
 * - 2 bytes: f16 scale (d)
 * - 2 bytes: f16 min (m)
 * - 16 bytes: 32 x 4-bit values (packed)
 */
typedef struct {
    uint16_t d;        /* f16 scale */
    uint16_t m;        /* f16 min */
    uint8_t qs[16];    /* 4-bit quantized values */
} block_q4_1;

#define QK4_1 32

/* Q5_0: 32 values per block
 * - 2 bytes: f16 scale (d)
 * - 4 bytes: 32 high bits
 * - 16 bytes: 32 x 4-bit low values
 */
typedef struct {
    uint16_t d;        /* f16 scale */
    uint8_t qh[4];     /* high bits */
    uint8_t qs[16];    /* low 4-bit values */
} block_q5_0;

#define QK5_0 32

/* Q5_1: 32 values per block
 * - 2 bytes: f16 scale (d)
 * - 2 bytes: f16 min (m)
 * - 4 bytes: 32 high bits
 * - 16 bytes: 32 x 4-bit low values
 */
typedef struct {
    uint16_t d;        /* f16 scale */
    uint16_t m;        /* f16 min */
    uint8_t qh[4];     /* high bits */
    uint8_t qs[16];    /* low 4-bit values */
} block_q5_1;

#define QK5_1 32

/* Q6_K: 256 values per block (super-block)
 * More complex K-quant format
 */
typedef struct {
    uint8_t ql[128];   /* low 4 bits of quantized values */
    uint8_t qh[64];    /* high 2 bits of quantized values */
    int8_t scales[16]; /* scales for sub-blocks */
    uint16_t d;        /* f16 super-block scale */
} block_q6_k;

#define QK_K 256

/* Q4_K: 256 values per block (super-block) */
typedef struct {
    uint16_t d;        /* f16 super-block scale for quantized scales */
    uint16_t dmin;     /* f16 super-block scale for quantized mins */
    uint8_t scales[12];/* scales and mins (6-bit each, packed) */
    uint8_t qs[128];   /* 4-bit quantized values */
} block_q4_k;

/* Q5_K: 256 values per block (super-block) */
typedef struct {
    uint16_t d;        /* f16 super-block scale */
    uint16_t dmin;     /* f16 super-block min */
    uint8_t scales[12];/* scales and mins */
    uint8_t qh[32];    /* high bits */
    uint8_t qs[128];   /* low 4-bit values */
} block_q5_k;

/* Q3_K: 256 values per block (super-block) */
typedef struct {
    uint8_t hmask[32]; /* high bit masks */
    uint8_t qs[64];    /* low 2 bits */
    uint8_t scales[12];/* scales */
    uint16_t d;        /* f16 super-block scale */
} block_q3_k;

/* Q2_K: 256 values per block (super-block) */
typedef struct {
    uint8_t scales[16];/* scales and mins */
    uint8_t qs[64];    /* 2-bit quantized values */
    uint16_t d;        /* f16 super-block scale */
    uint16_t dmin;     /* f16 super-block min */
} block_q2_k;

/* ============================================================================
 * F16 Conversion
 * ============================================================================ */

/* Convert f16 (half precision) to f32 */
float f16_to_f32(uint16_t h);

/* Convert f32 to f16 */
uint16_t f32_to_f16(float f);

/* ============================================================================
 * Dequantization Functions
 * Converts quantized data to float32
 * ============================================================================ */

/* Dequantize Q8_0 block */
void dequantize_row_q8_0(const block_q8_0 *x, float *y, int k);

/* Dequantize Q4_0 block */
void dequantize_row_q4_0(const block_q4_0 *x, float *y, int k);

/* Dequantize Q4_1 block */
void dequantize_row_q4_1(const block_q4_1 *x, float *y, int k);

/* Dequantize Q5_0 block */
void dequantize_row_q5_0(const block_q5_0 *x, float *y, int k);

/* Dequantize Q5_1 block */
void dequantize_row_q5_1(const block_q5_1 *x, float *y, int k);

/* Dequantize Q6_K block */
void dequantize_row_q6_k(const block_q6_k *x, float *y, int k);

/* Dequantize Q4_K block */
void dequantize_row_q4_k(const block_q4_k *x, float *y, int k);

/* Dequantize Q5_K block */
void dequantize_row_q5_k(const block_q5_k *x, float *y, int k);

/* Dequantize Q3_K block */
void dequantize_row_q3_k(const block_q3_k *x, float *y, int k);

/* Dequantize Q2_K block */
void dequantize_row_q2_k(const block_q2_k *x, float *y, int k);

/* ============================================================================
 * Quantized Vector Dot Product
 * Computes dot product without full dequantization (faster)
 * ============================================================================ */

/* Q8_0 dot F32: dot(quantized_vec, float_vec) */
float vec_dot_q8_0_f32(const block_q8_0 *x, const float *y, int k);

/* Q4_0 dot F32 */
float vec_dot_q4_0_f32(const block_q4_0 *x, const float *y, int k);

/* Q4_K dot F32 */
float vec_dot_q4_k_f32(const block_q4_k *x, const float *y, int k);

/* Q6_K dot F32 */
float vec_dot_q6_k_f32(const block_q6_k *x, const float *y, int k);

/* ============================================================================
 * Quantized Matrix-Vector Multiplication
 * dst = M @ v where M is quantized
 * ============================================================================ */

/* Q8_0 matrix @ float vector */
void mat_vec_q8_0(float *dst, const void *M, const float *v, int rows, int cols);

/* Q4_0 matrix @ float vector */
void mat_vec_q4_0(float *dst, const void *M, const float *v, int rows, int cols);

/* Q4_K matrix @ float vector */
void mat_vec_q4_k(float *dst, const void *M, const float *v, int rows, int cols);

/* Q6_K matrix @ float vector */
void mat_vec_q6_k(float *dst, const void *M, const float *v, int rows, int cols);

/* Generic quantized matmul dispatcher */
void mat_vec_quant(float *dst, const void *M, const float *v,
                   int rows, int cols, int quant_type);

/* ============================================================================
 * Block Size Helpers
 * ============================================================================ */

/* Get block size for quantization type */
int quant_block_size(int type);

/* Get bytes per block for quantization type */
size_t quant_block_bytes(int type);

/* Calculate row size in bytes */
size_t quant_row_size(int type, int cols);

#ifdef __cplusplus
}
#endif

#endif /* HOLO_QUANT_H */
