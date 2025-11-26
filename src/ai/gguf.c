/*
 * Holo - GGUF File Format Parser Implementation
 * Pure C implementation for reading GGUF model files
 */

#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

/* ============================================================================
 * Type Size Tables
 * ============================================================================ */

/* Size in bytes for a block of each quantized type */
static const size_t GGML_TYPE_SIZE[] = {
    [GGML_TYPE_F32]     = 4,
    [GGML_TYPE_F16]     = 2,
    [GGML_TYPE_Q4_0]    = 18,   /* 32 values in 18 bytes (0.5 bit + scale) */
    [GGML_TYPE_Q4_1]    = 20,
    [GGML_TYPE_Q5_0]    = 22,
    [GGML_TYPE_Q5_1]    = 24,
    [GGML_TYPE_Q8_0]    = 34,   /* 32 values in 34 bytes */
    [GGML_TYPE_Q8_1]    = 36,
    [GGML_TYPE_Q2_K]    = 84,
    [GGML_TYPE_Q3_K]    = 110,
    [GGML_TYPE_Q4_K]    = 144,
    [GGML_TYPE_Q5_K]    = 176,
    [GGML_TYPE_Q6_K]    = 210,
    [GGML_TYPE_Q8_K]    = 292,
    [GGML_TYPE_I8]      = 1,
    [GGML_TYPE_I16]     = 2,
    [GGML_TYPE_I32]     = 4,
    [GGML_TYPE_I64]     = 8,
    [GGML_TYPE_F64]     = 8,
    [GGML_TYPE_BF16]    = 2,
};

/* Block size (number of elements per block) */
static const int GGML_BLCK_SIZE[] = {
    [GGML_TYPE_F32]     = 1,
    [GGML_TYPE_F16]     = 1,
    [GGML_TYPE_Q4_0]    = 32,
    [GGML_TYPE_Q4_1]    = 32,
    [GGML_TYPE_Q5_0]    = 32,
    [GGML_TYPE_Q5_1]    = 32,
    [GGML_TYPE_Q8_0]    = 32,
    [GGML_TYPE_Q8_1]    = 32,
    [GGML_TYPE_Q2_K]    = 256,
    [GGML_TYPE_Q3_K]    = 256,
    [GGML_TYPE_Q4_K]    = 256,
    [GGML_TYPE_Q5_K]    = 256,
    [GGML_TYPE_Q6_K]    = 256,
    [GGML_TYPE_Q8_K]    = 256,
    [GGML_TYPE_I8]      = 1,
    [GGML_TYPE_I16]     = 1,
    [GGML_TYPE_I32]     = 1,
    [GGML_TYPE_I64]     = 1,
    [GGML_TYPE_F64]     = 1,
    [GGML_TYPE_BF16]    = 1,
};

static const char *GGML_TYPE_NAMES[] = {
    [GGML_TYPE_F32]     = "F32",
    [GGML_TYPE_F16]     = "F16",
    [GGML_TYPE_Q4_0]    = "Q4_0",
    [GGML_TYPE_Q4_1]    = "Q4_1",
    [GGML_TYPE_Q5_0]    = "Q5_0",
    [GGML_TYPE_Q5_1]    = "Q5_1",
    [GGML_TYPE_Q8_0]    = "Q8_0",
    [GGML_TYPE_Q8_1]    = "Q8_1",
    [GGML_TYPE_Q2_K]    = "Q2_K",
    [GGML_TYPE_Q3_K]    = "Q3_K",
    [GGML_TYPE_Q4_K]    = "Q4_K",
    [GGML_TYPE_Q5_K]    = "Q5_K",
    [GGML_TYPE_Q6_K]    = "Q6_K",
    [GGML_TYPE_Q8_K]    = "Q8_K",
    [GGML_TYPE_I8]      = "I8",
    [GGML_TYPE_I16]     = "I16",
    [GGML_TYPE_I32]     = "I32",
    [GGML_TYPE_I64]     = "I64",
    [GGML_TYPE_F64]     = "F64",
    [GGML_TYPE_BF16]    = "BF16",
};

size_t ggml_type_size(ggml_type_t type) {
    if (type >= GGML_TYPE_COUNT) return 0;
    return GGML_TYPE_SIZE[type];
}

int ggml_blck_size(ggml_type_t type) {
    if (type >= GGML_TYPE_COUNT) return 1;
    return GGML_BLCK_SIZE[type];
}

size_t ggml_row_size(ggml_type_t type, int64_t n_elements) {
    int blck = ggml_blck_size(type);
    size_t type_size = ggml_type_size(type);
    return (n_elements / blck) * type_size;
}

const char *ggml_type_name(ggml_type_t type) {
    if (type >= GGML_TYPE_COUNT) return "UNKNOWN";
    return GGML_TYPE_NAMES[type] ? GGML_TYPE_NAMES[type] : "UNKNOWN";
}

/* ============================================================================
 * File Reading Helpers
 * ============================================================================ */

static int read_u8(FILE *f, uint8_t *v) {
    return fread(v, 1, 1, f) == 1 ? 0 : -1;
}

static int read_u16(FILE *f, uint16_t *v) {
    return fread(v, 2, 1, f) == 1 ? 0 : -1;
}

static int read_u32(FILE *f, uint32_t *v) {
    return fread(v, 4, 1, f) == 1 ? 0 : -1;
}

static int read_u64(FILE *f, uint64_t *v) {
    return fread(v, 8, 1, f) == 1 ? 0 : -1;
}

static int read_i32(FILE *f, int32_t *v) {
    return fread(v, 4, 1, f) == 1 ? 0 : -1;
}

static int read_f32(FILE *f, float *v) {
    return fread(v, 4, 1, f) == 1 ? 0 : -1;
}

static int read_string(FILE *f, gguf_string_t *s) {
    if (read_u64(f, &s->len) != 0) return -1;
    if (s->len > 1024 * 1024) return -1;  /* Sanity check */

    s->data = (char *)malloc(s->len + 1);
    if (!s->data) return -1;

    if (s->len > 0 && fread(s->data, 1, s->len, f) != s->len) {
        free(s->data);
        s->data = NULL;
        return -1;
    }
    s->data[s->len] = '\0';
    return 0;
}

static void free_string(gguf_string_t *s) {
    free(s->data);
    s->data = NULL;
    s->len = 0;
}

/* ============================================================================
 * Value Reading
 * ============================================================================ */

static int read_value(FILE *f, gguf_value_t *val) {
    uint32_t type;
    if (read_u32(f, &type) != 0) return -1;
    val->type = (gguf_type_t)type;

    switch (val->type) {
        case GGUF_TYPE_UINT8:   return read_u8(f, &val->u8);
        case GGUF_TYPE_INT8:    return fread(&val->i8, 1, 1, f) == 1 ? 0 : -1;
        case GGUF_TYPE_UINT16:  return read_u16(f, &val->u16);
        case GGUF_TYPE_INT16:   return fread(&val->i16, 2, 1, f) == 1 ? 0 : -1;
        case GGUF_TYPE_UINT32:  return read_u32(f, &val->u32);
        case GGUF_TYPE_INT32:   return read_i32(f, &val->i32);
        case GGUF_TYPE_UINT64:  return read_u64(f, &val->u64);
        case GGUF_TYPE_INT64:   return fread(&val->i64, 8, 1, f) == 1 ? 0 : -1;
        case GGUF_TYPE_FLOAT32: return read_f32(f, &val->f32);
        case GGUF_TYPE_FLOAT64: return fread(&val->f64, 8, 1, f) == 1 ? 0 : -1;
        case GGUF_TYPE_BOOL: {
            uint8_t b;
            if (read_u8(f, &b) != 0) return -1;
            val->b = b != 0;
            return 0;
        }
        case GGUF_TYPE_STRING:
            return read_string(f, &val->str);

        case GGUF_TYPE_ARRAY: {
            uint32_t arr_type;
            if (read_u32(f, &arr_type) != 0) return -1;
            if (read_u64(f, &val->arr.len) != 0) return -1;
            val->arr.type = (gguf_type_t)arr_type;

            if (val->arr.len > 10000000) return -1;  /* Sanity */

            /* For now, just skip array data (we'll read specific arrays as needed) */
            size_t elem_size = 0;
            switch (val->arr.type) {
                case GGUF_TYPE_UINT8:
                case GGUF_TYPE_INT8:
                case GGUF_TYPE_BOOL:    elem_size = 1; break;
                case GGUF_TYPE_UINT16:
                case GGUF_TYPE_INT16:   elem_size = 2; break;
                case GGUF_TYPE_UINT32:
                case GGUF_TYPE_INT32:
                case GGUF_TYPE_FLOAT32: elem_size = 4; break;
                case GGUF_TYPE_UINT64:
                case GGUF_TYPE_INT64:
                case GGUF_TYPE_FLOAT64: elem_size = 8; break;
                case GGUF_TYPE_STRING: {
                    /* Read array of strings */
                    val->arr.data = calloc(val->arr.len, sizeof(gguf_string_t));
                    if (!val->arr.data) return -1;
                    gguf_string_t *strs = (gguf_string_t *)val->arr.data;
                    for (uint64_t i = 0; i < val->arr.len; i++) {
                        if (read_string(f, &strs[i]) != 0) return -1;
                    }
                    return 0;
                }
                default:
                    return -1;
            }

            if (elem_size > 0) {
                val->arr.data = malloc(val->arr.len * elem_size);
                if (!val->arr.data) return -1;
                if (fread(val->arr.data, elem_size, val->arr.len, f) != val->arr.len) {
                    free(val->arr.data);
                    return -1;
                }
            }
            return 0;
        }

        default:
            return -1;
    }
}

static void free_value(gguf_value_t *val) {
    if (val->type == GGUF_TYPE_STRING) {
        free_string(&val->str);
    } else if (val->type == GGUF_TYPE_ARRAY) {
        if (val->arr.type == GGUF_TYPE_STRING && val->arr.data) {
            gguf_string_t *strs = (gguf_string_t *)val->arr.data;
            for (uint64_t i = 0; i < val->arr.len; i++) {
                free_string(&strs[i]);
            }
        }
        free(val->arr.data);
    }
}

/* ============================================================================
 * GGUF File Operations
 * ============================================================================ */

gguf_ctx_t *gguf_open(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "GGUF: Cannot open file: %s\n", path);
        return NULL;
    }

    gguf_ctx_t *ctx = (gguf_ctx_t *)calloc(1, sizeof(gguf_ctx_t));
    if (!ctx) {
        fclose(f);
        return NULL;
    }

    strncpy(ctx->path, path, sizeof(ctx->path) - 1);
    ctx->file = f;

    /* Read header */
    if (read_u32(f, &ctx->magic) != 0 || ctx->magic != GGUF_MAGIC) {
        fprintf(stderr, "GGUF: Invalid magic number\n");
        goto error;
    }

    if (read_u32(f, &ctx->version) != 0) {
        fprintf(stderr, "GGUF: Cannot read version\n");
        goto error;
    }

    if (ctx->version < 2 || ctx->version > 3) {
        fprintf(stderr, "GGUF: Unsupported version %u (need 2 or 3)\n", ctx->version);
        goto error;
    }

    if (read_u64(f, &ctx->n_tensors) != 0) {
        fprintf(stderr, "GGUF: Cannot read tensor count\n");
        goto error;
    }

    if (read_u64(f, &ctx->n_kv) != 0) {
        fprintf(stderr, "GGUF: Cannot read KV count\n");
        goto error;
    }

    /* Read metadata KV pairs */
    ctx->kv = (gguf_kv_t *)calloc(ctx->n_kv, sizeof(gguf_kv_t));
    if (!ctx->kv) goto error;

    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        if (read_string(f, &ctx->kv[i].key) != 0) {
            fprintf(stderr, "GGUF: Cannot read key %llu\n", (unsigned long long)i);
            goto error;
        }
        if (read_value(f, &ctx->kv[i].value) != 0) {
            fprintf(stderr, "GGUF: Cannot read value for key '%s'\n", ctx->kv[i].key.data);
            goto error;
        }
    }

    /* Read tensor info */
    ctx->tensors = (gguf_tensor_info_t *)calloc(ctx->n_tensors, sizeof(gguf_tensor_info_t));
    if (!ctx->tensors) goto error;

    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        gguf_tensor_info_t *t = &ctx->tensors[i];

        gguf_string_t name;
        if (read_string(f, &name) != 0) goto error;
        strncpy(t->name, name.data, GGUF_MAX_NAME - 1);
        free_string(&name);

        if (read_u32(f, &t->n_dims) != 0) goto error;
        if (t->n_dims > GGUF_MAX_DIMS) goto error;

        t->n_elements = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            if (read_u64(f, &t->dims[d]) != 0) goto error;
            t->n_elements *= t->dims[d];
        }

        uint32_t type;
        if (read_u32(f, &type) != 0) goto error;
        t->type = (ggml_type_t)type;

        if (read_u64(f, &t->offset) != 0) goto error;

        /* Calculate size */
        t->size_bytes = ggml_row_size(t->type, t->n_elements);
    }

    /* Calculate data offset (aligned to 32 bytes) */
    ctx->data_offset = ftell(f);
    ctx->data_offset = (ctx->data_offset + 31) & ~31ULL;

    /* Extract model metadata */
    const char *arch = gguf_get_string(ctx, "general.architecture");
    if (arch) strncpy(ctx->model.arch, arch, sizeof(ctx->model.arch) - 1);

    char key[256];
    snprintf(key, sizeof(key), "%s.vocab_size", ctx->model.arch);
    ctx->model.vocab_size = (uint32_t)gguf_get_int(ctx, key, 32000);

    snprintf(key, sizeof(key), "%s.context_length", ctx->model.arch);
    ctx->model.context_length = (uint32_t)gguf_get_int(ctx, key, 4096);

    snprintf(key, sizeof(key), "%s.embedding_length", ctx->model.arch);
    ctx->model.embedding_length = (uint32_t)gguf_get_int(ctx, key, 4096);

    snprintf(key, sizeof(key), "%s.block_count", ctx->model.arch);
    ctx->model.block_count = (uint32_t)gguf_get_int(ctx, key, 32);

    snprintf(key, sizeof(key), "%s.attention.head_count", ctx->model.arch);
    ctx->model.attention_head_count = (uint32_t)gguf_get_int(ctx, key, 32);

    snprintf(key, sizeof(key), "%s.attention.head_count_kv", ctx->model.arch);
    ctx->model.attention_head_count_kv = (uint32_t)gguf_get_int(ctx, key, ctx->model.attention_head_count);

    snprintf(key, sizeof(key), "%s.feed_forward_length", ctx->model.arch);
    ctx->model.feed_forward_length = (uint32_t)gguf_get_int(ctx, key, 11008);

    snprintf(key, sizeof(key), "%s.rope.freq_base", ctx->model.arch);
    ctx->model.rope_freq_base = gguf_get_float(ctx, key, 10000.0f);

    snprintf(key, sizeof(key), "%s.attention.layer_norm_rms_epsilon", ctx->model.arch);
    ctx->model.rms_norm_eps = gguf_get_float(ctx, key, 1e-5f);

    return ctx;

error:
    gguf_close(ctx);
    return NULL;
}

void gguf_close(gguf_ctx_t *ctx) {
    if (!ctx) return;

    if (ctx->mmap_base) {
#ifdef _WIN32
        UnmapViewOfFile(ctx->mmap_base);
#else
        munmap(ctx->mmap_base, ctx->mmap_size);
#endif
    }

    if (ctx->kv) {
        for (uint64_t i = 0; i < ctx->n_kv; i++) {
            free_string(&ctx->kv[i].key);
            free_value(&ctx->kv[i].value);
        }
        free(ctx->kv);
    }

    free(ctx->tensors);

    if (ctx->file) fclose(ctx->file);

    free(ctx);
}

/* ============================================================================
 * Metadata Access
 * ============================================================================ */

const gguf_value_t *gguf_get_value(gguf_ctx_t *ctx, const char *key) {
    if (!ctx || !key) return NULL;

    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        if (strcmp(ctx->kv[i].key.data, key) == 0) {
            return &ctx->kv[i].value;
        }
    }
    return NULL;
}

const char *gguf_get_string(gguf_ctx_t *ctx, const char *key) {
    const gguf_value_t *v = gguf_get_value(ctx, key);
    if (v && v->type == GGUF_TYPE_STRING) {
        return v->str.data;
    }
    return NULL;
}

int64_t gguf_get_int(gguf_ctx_t *ctx, const char *key, int64_t default_val) {
    const gguf_value_t *v = gguf_get_value(ctx, key);
    if (!v) return default_val;

    switch (v->type) {
        case GGUF_TYPE_UINT8:   return v->u8;
        case GGUF_TYPE_INT8:    return v->i8;
        case GGUF_TYPE_UINT16:  return v->u16;
        case GGUF_TYPE_INT16:   return v->i16;
        case GGUF_TYPE_UINT32:  return v->u32;
        case GGUF_TYPE_INT32:   return v->i32;
        case GGUF_TYPE_UINT64:  return (int64_t)v->u64;
        case GGUF_TYPE_INT64:   return v->i64;
        default: return default_val;
    }
}

float gguf_get_float(gguf_ctx_t *ctx, const char *key, float default_val) {
    const gguf_value_t *v = gguf_get_value(ctx, key);
    if (!v) return default_val;

    switch (v->type) {
        case GGUF_TYPE_FLOAT32: return v->f32;
        case GGUF_TYPE_FLOAT64: return (float)v->f64;
        default: return default_val;
    }
}

const void *gguf_get_array(gguf_ctx_t *ctx, const char *key, uint64_t *len) {
    const gguf_value_t *v = gguf_get_value(ctx, key);
    if (v && v->type == GGUF_TYPE_ARRAY) {
        if (len) *len = v->arr.len;
        return v->arr.data;
    }
    if (len) *len = 0;
    return NULL;
}

/* ============================================================================
 * Tensor Access
 * ============================================================================ */

const gguf_tensor_info_t *gguf_find_tensor(gguf_ctx_t *ctx, const char *name) {
    if (!ctx || !name) return NULL;

    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0) {
            return &ctx->tensors[i];
        }
    }
    return NULL;
}

const void *gguf_get_tensor_data(gguf_ctx_t *ctx, const gguf_tensor_info_t *tensor) {
    if (!ctx || !tensor || !ctx->mmap_base) return NULL;
    /* mmap_base points to file start (NOT adjusted)
     * data_offset points to start of tensor data section
     * tensor->offset is relative to data section */
    return (const char *)ctx->mmap_base + ctx->data_offset + tensor->offset;
}

int gguf_load_tensor(gguf_ctx_t *ctx, const gguf_tensor_info_t *tensor, void *buffer, size_t size) {
    if (!ctx || !tensor || !buffer) return -1;
    if (size < tensor->size_bytes) return -1;

    if (fseek(ctx->file, (long)(ctx->data_offset + tensor->offset), SEEK_SET) != 0) {
        return -1;
    }

    if (fread(buffer, 1, tensor->size_bytes, ctx->file) != tensor->size_bytes) {
        return -1;
    }

    return 0;
}

int gguf_mmap(gguf_ctx_t *ctx) {
    if (!ctx) return -1;

#ifdef _WIN32
    /* Get file size properly for large files */
    HANDLE hFile = (HANDLE)_get_osfhandle(_fileno(ctx->file));
    LARGE_INTEGER li_size;
    if (!GetFileSizeEx(hFile, &li_size)) return -1;
    size_t file_size = (size_t)li_size.QuadPart;

    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) return -1;

    ctx->mmap_base = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMap);

    if (!ctx->mmap_base) return -1;
    ctx->mmap_size = file_size;

    /* Note: DON'T adjust base - keep it pointing to file start
     * tensor offsets are relative to data_offset, which we add in gguf_get_tensor_data */

#else
    struct stat st;
    int fd = fileno(ctx->file);
    if (fstat(fd, &st) != 0) return -1;

    ctx->mmap_size = st.st_size;
    void *base = mmap(NULL, ctx->mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) return -1;

    /* Don't adjust - keep pointing to file start */
    ctx->mmap_base = base;
#endif

    return 0;
}

/* ============================================================================
 * Debug / Info
 * ============================================================================ */

void gguf_print_info(gguf_ctx_t *ctx) {
    if (!ctx) return;

    printf("\n=== GGUF Model Info ===\n");
    printf("File: %s\n", ctx->path);
    printf("Version: %u\n", ctx->version);
    printf("Tensors: %llu\n", (unsigned long long)ctx->n_tensors);
    printf("Metadata: %llu entries\n", (unsigned long long)ctx->n_kv);
    printf("\n--- Model Parameters ---\n");
    printf("Architecture: %s\n", ctx->model.arch);
    printf("Vocab size: %u\n", ctx->model.vocab_size);
    printf("Context length: %u\n", ctx->model.context_length);
    printf("Embedding dim: %u\n", ctx->model.embedding_length);
    printf("Layers: %u\n", ctx->model.block_count);
    printf("Attention heads: %u\n", ctx->model.attention_head_count);
    printf("KV heads: %u\n", ctx->model.attention_head_count_kv);
    printf("FFN dim: %u\n", ctx->model.feed_forward_length);
    printf("RoPE freq base: %.1f\n", ctx->model.rope_freq_base);
    printf("RMS norm eps: %e\n", ctx->model.rms_norm_eps);
    printf("\n");
}

/* Dump all GGUF metadata keys for debugging */
void gguf_dump_keys(gguf_ctx_t *ctx) {
    if (!ctx) return;

    printf("\n=== All GGUF Metadata Keys ===\n");
    for (uint64_t i = 0; i < ctx->n_kv && i < 100; i++) {
        const char *key = ctx->kv[i].key.data ? ctx->kv[i].key.data : "(null)";
        gguf_type_t type = ctx->kv[i].value.type;

        if (type == GGUF_TYPE_UINT32) {
            printf("  [%llu] %s = %u\n", (unsigned long long)i, key, ctx->kv[i].value.u32);
        } else if (type == GGUF_TYPE_INT32) {
            printf("  [%llu] %s = %d\n", (unsigned long long)i, key, ctx->kv[i].value.i32);
        } else if (type == GGUF_TYPE_FLOAT32) {
            printf("  [%llu] %s = %f\n", (unsigned long long)i, key, ctx->kv[i].value.f32);
        } else if (type == GGUF_TYPE_STRING) {
            printf("  [%llu] %s = \"%s\"\n", (unsigned long long)i, key,
                   ctx->kv[i].value.str.data ? ctx->kv[i].value.str.data : "");
        } else {
            printf("  [%llu] %s (type=%d)\n", (unsigned long long)i, key, type);
        }
    }
    printf("\n");
}
