/*
 * Holo - GGUF File Format Parser
 * Pure C implementation for reading GGUF model files
 */

#ifndef HOLO_GGUF_H
#define HOLO_GGUF_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* GGUF Magic and Version */
#define GGUF_MAGIC      0x46554747  /* "GGUF" in little-endian */
#define GGUF_VERSION    3

/* GGUF Value Types */
typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12
} gguf_type_t;

/* GGML Tensor Types (quantization formats) */
typedef enum {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_BF16    = 29,
    GGML_TYPE_COUNT
} ggml_type_t;

/* Maximum dimensions for tensors */
#define GGUF_MAX_DIMS 4
#define GGUF_MAX_NAME 256

/* GGUF String */
typedef struct {
    uint64_t len;
    char *data;
} gguf_string_t;

/* GGUF Metadata Value */
typedef struct gguf_value {
    gguf_type_t type;
    union {
        uint8_t   u8;
        int8_t    i8;
        uint16_t  u16;
        int16_t   i16;
        uint32_t  u32;
        int32_t   i32;
        uint64_t  u64;
        int64_t   i64;
        float     f32;
        double    f64;
        bool      b;
        gguf_string_t str;
        struct {
            gguf_type_t type;
            uint64_t len;
            void *data;
        } arr;
    };
} gguf_value_t;

/* GGUF Metadata Key-Value Pair */
typedef struct {
    gguf_string_t key;
    gguf_value_t value;
} gguf_kv_t;

/* GGUF Tensor Info */
typedef struct {
    char name[GGUF_MAX_NAME];
    uint32_t n_dims;
    uint64_t dims[GGUF_MAX_DIMS];
    ggml_type_t type;
    uint64_t offset;  /* Offset from start of tensor data */

    /* Computed fields */
    uint64_t n_elements;
    uint64_t size_bytes;
} gguf_tensor_info_t;

/* GGUF File Context */
typedef struct {
    /* File info */
    char path[1024];
    FILE *file;

    /* Header */
    uint32_t magic;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;

    /* Metadata */
    gguf_kv_t *kv;

    /* Tensor info */
    gguf_tensor_info_t *tensors;

    /* Data offset (where tensor data starts) */
    uint64_t data_offset;

    /* Memory-mapped data (optional) */
    void *mmap_base;
    size_t mmap_size;

    /* Model metadata (extracted from KV) */
    struct {
        char arch[64];
        uint32_t vocab_size;
        uint32_t context_length;
        uint32_t embedding_length;
        uint32_t block_count;       /* num layers */
        uint32_t attention_head_count;
        uint32_t attention_head_count_kv;
        uint32_t feed_forward_length;
        float rope_freq_base;
        float rope_freq_scale;
        float rms_norm_eps;
    } model;

} gguf_ctx_t;

/* ============================================================================
 * GGUF API Functions
 * ============================================================================ */

/* Open and parse a GGUF file */
gguf_ctx_t *gguf_open(const char *path);

/* Close and free GGUF context */
void gguf_close(gguf_ctx_t *ctx);

/* Get metadata value by key */
const gguf_value_t *gguf_get_value(gguf_ctx_t *ctx, const char *key);

/* Get string metadata */
const char *gguf_get_string(gguf_ctx_t *ctx, const char *key);

/* Get integer metadata */
int64_t gguf_get_int(gguf_ctx_t *ctx, const char *key, int64_t default_val);

/* Get float metadata */
float gguf_get_float(gguf_ctx_t *ctx, const char *key, float default_val);

/* Get array metadata */
const void *gguf_get_array(gguf_ctx_t *ctx, const char *key, uint64_t *len);

/* Find tensor by name */
const gguf_tensor_info_t *gguf_find_tensor(gguf_ctx_t *ctx, const char *name);

/* Get tensor data pointer (requires mmap or load) */
const void *gguf_get_tensor_data(gguf_ctx_t *ctx, const gguf_tensor_info_t *tensor);

/* Load tensor data into buffer */
int gguf_load_tensor(gguf_ctx_t *ctx, const gguf_tensor_info_t *tensor, void *buffer, size_t size);

/* Memory-map the tensor data region */
int gguf_mmap(gguf_ctx_t *ctx);

/* Print model info */
void gguf_print_info(gguf_ctx_t *ctx);

/* ============================================================================
 * Quantization Helpers
 * ============================================================================ */

/* Get size of a single element for a type */
size_t ggml_type_size(ggml_type_t type);

/* Get block size for quantized types */
int ggml_blck_size(ggml_type_t type);

/* Calculate size in bytes for n elements */
size_t ggml_row_size(ggml_type_t type, int64_t n_elements);

/* Type name string */
const char *ggml_type_name(ggml_type_t type);

/* Dequantize a block to float32 */
void ggml_dequantize_row(ggml_type_t type, const void *src, float *dst, int64_t n);

#ifdef __cplusplus
}
#endif

#endif /* HOLO_GGUF_H */
