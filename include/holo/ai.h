/*
 * Holo - AI Inference Engine Interface
 * Pure C implementation for local AI model inference
 */

#ifndef HOLO_AI_H
#define HOLO_AI_H

#ifdef __cplusplus
extern "C" {
#endif

#include "holo/pal.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* ============================================================================
 * Constants
 * ============================================================================ */

#define HOLO_AI_MAX_CONTEXT_SIZE    131072  /* 128K tokens max */
#define HOLO_AI_MAX_BATCH_SIZE      512
#define HOLO_AI_MAX_VOCAB_SIZE      256000
#define HOLO_AI_MAX_LAYERS          128
#define HOLO_AI_MAX_MODEL_NAME      256
#define HOLO_AI_MAX_STOP_SEQUENCES  8
#define HOLO_AI_MAX_STOP_LENGTH     64

/* ============================================================================
 * Error Codes
 * ============================================================================ */

typedef enum {
    HOLO_AI_OK = 0,
    HOLO_AI_ERROR_INVALID_PARAM,
    HOLO_AI_ERROR_OUT_OF_MEMORY,
    HOLO_AI_ERROR_FILE_NOT_FOUND,
    HOLO_AI_ERROR_INVALID_MODEL,
    HOLO_AI_ERROR_UNSUPPORTED_FORMAT,
    HOLO_AI_ERROR_CONTEXT_FULL,
    HOLO_AI_ERROR_GENERATION_FAILED,
    HOLO_AI_ERROR_CANCELLED,
    HOLO_AI_ERROR_NOT_INITIALIZED,
    HOLO_AI_ERROR_ALREADY_INITIALIZED,
    HOLO_AI_ERROR_BACKEND_UNAVAILABLE
} holo_ai_error_t;

/* ============================================================================
 * Model Architecture Types
 * ============================================================================ */

typedef enum {
    HOLO_AI_ARCH_UNKNOWN = 0,
    HOLO_AI_ARCH_LLAMA,
    HOLO_AI_ARCH_LLAMA2,
    HOLO_AI_ARCH_LLAMA3,
    HOLO_AI_ARCH_MISTRAL,
    HOLO_AI_ARCH_MIXTRAL,
    HOLO_AI_ARCH_QWEN,
    HOLO_AI_ARCH_QWEN2,
    HOLO_AI_ARCH_PHI,
    HOLO_AI_ARCH_PHI2,
    HOLO_AI_ARCH_PHI3,
    HOLO_AI_ARCH_GEMMA,
    HOLO_AI_ARCH_GEMMA2,
    HOLO_AI_ARCH_STARCODER,
    HOLO_AI_ARCH_STARCODER2,
    HOLO_AI_ARCH_GPT2,
    HOLO_AI_ARCH_GPTJ,
    HOLO_AI_ARCH_GPTNEOX,
    HOLO_AI_ARCH_FALCON,
    HOLO_AI_ARCH_MPT,
    HOLO_AI_ARCH_BAICHUAN,
    HOLO_AI_ARCH_INTERNLM,
    HOLO_AI_ARCH_DEEPSEEK,
    HOLO_AI_ARCH_COMMAND_R
} holo_ai_arch_t;

/* ============================================================================
 * Quantization Types
 * ============================================================================ */

typedef enum {
    HOLO_AI_QUANT_NONE = 0,     /* F32 - no quantization */
    HOLO_AI_QUANT_F16,          /* F16 - half precision */
    HOLO_AI_QUANT_BF16,         /* BF16 - brain float */
    HOLO_AI_QUANT_Q8_0,         /* 8-bit quantization */
    HOLO_AI_QUANT_Q6_K,         /* 6-bit k-quant */
    HOLO_AI_QUANT_Q5_K_M,       /* 5-bit k-quant medium */
    HOLO_AI_QUANT_Q5_K_S,       /* 5-bit k-quant small */
    HOLO_AI_QUANT_Q4_K_M,       /* 4-bit k-quant medium */
    HOLO_AI_QUANT_Q4_K_S,       /* 4-bit k-quant small */
    HOLO_AI_QUANT_Q4_0,         /* 4-bit quantization */
    HOLO_AI_QUANT_Q3_K_M,       /* 3-bit k-quant medium */
    HOLO_AI_QUANT_Q3_K_S,       /* 3-bit k-quant small */
    HOLO_AI_QUANT_Q2_K,         /* 2-bit k-quant */
    HOLO_AI_QUANT_IQ4_XS,       /* imatrix 4-bit */
    HOLO_AI_QUANT_IQ3_XXS,      /* imatrix 3-bit */
    HOLO_AI_QUANT_IQ2_XXS,      /* imatrix 2-bit */
    HOLO_AI_QUANT_IQ1_S         /* imatrix 1-bit */
} holo_ai_quant_t;

/* ============================================================================
 * Compute Backend Types
 * ============================================================================ */

typedef enum {
    HOLO_AI_BACKEND_CPU = 0,
    HOLO_AI_BACKEND_CUDA,       /* NVIDIA GPU */
    HOLO_AI_BACKEND_VULKAN,     /* Cross-platform GPU */
    HOLO_AI_BACKEND_METAL,      /* Apple GPU */
    HOLO_AI_BACKEND_OPENCL,     /* OpenCL GPU */
    HOLO_AI_BACKEND_SYCL,       /* Intel oneAPI */
    HOLO_AI_BACKEND_ROCM,       /* AMD GPU */
    HOLO_AI_BACKEND_AUTO        /* Auto-detect best */
} holo_ai_backend_t;

/* ============================================================================
 * Message Role (for chat models)
 * ============================================================================ */

typedef enum {
    HOLO_AI_ROLE_SYSTEM = 0,
    HOLO_AI_ROLE_USER,
    HOLO_AI_ROLE_ASSISTANT,
    HOLO_AI_ROLE_TOOL
} holo_ai_role_t;

/* ============================================================================
 * Token Type
 * ============================================================================ */

typedef int32_t holo_ai_token_t;

/* ============================================================================
 * Model Information
 * ============================================================================ */

typedef struct {
    char name[HOLO_AI_MAX_MODEL_NAME];
    char path[1024];

    holo_ai_arch_t architecture;
    holo_ai_quant_t quantization;

    /* Model dimensions */
    uint32_t vocab_size;
    uint32_t context_length;
    uint32_t embedding_dim;
    uint32_t num_layers;
    uint32_t num_heads;
    uint32_t num_kv_heads;       /* For GQA models */
    uint32_t head_dim;
    uint32_t intermediate_dim;   /* FFN hidden dim */

    /* Memory requirements */
    uint64_t model_size_bytes;
    uint64_t kv_cache_per_token;

    /* Special tokens */
    holo_ai_token_t bos_token;   /* Beginning of sequence */
    holo_ai_token_t eos_token;   /* End of sequence */
    holo_ai_token_t pad_token;   /* Padding token */
    holo_ai_token_t unk_token;   /* Unknown token */

    /* Chat template info */
    bool is_chat_model;
    const char *chat_template;
} holo_ai_model_info_t;

/* ============================================================================
 * Generation Parameters
 * ============================================================================ */

typedef struct {
    /* Sampling parameters */
    float temperature;          /* 0.0 = greedy, 1.0 = default, >1.0 = creative */
    float top_p;                /* Nucleus sampling (0.0-1.0) */
    int32_t top_k;              /* Top-K sampling (0 = disabled) */
    float min_p;                /* Min-P sampling threshold */
    float typical_p;            /* Locally typical sampling */

    /* Repetition control */
    float repeat_penalty;       /* Repetition penalty (1.0 = disabled) */
    float presence_penalty;     /* Presence penalty (-2.0 to 2.0) */
    float frequency_penalty;    /* Frequency penalty (-2.0 to 2.0) */
    int32_t repeat_last_n;      /* Context for repetition penalty */

    /* Output control */
    int32_t max_tokens;         /* Max tokens to generate (0 = no limit) */
    int32_t seed;               /* Random seed (-1 = random) */

    /* Stop sequences */
    char stop_sequences[HOLO_AI_MAX_STOP_SEQUENCES][HOLO_AI_MAX_STOP_LENGTH];
    int32_t num_stop_sequences;

    /* Mirostat sampling */
    int32_t mirostat;           /* 0 = disabled, 1 = v1, 2 = v2 */
    float mirostat_tau;         /* Target entropy */
    float mirostat_eta;         /* Learning rate */

    /* Grammar constraint (GBNF format) */
    const char *grammar;

    /* JSON schema constraint */
    const char *json_schema;

    /* Streaming callback */
    bool stream;
} holo_ai_gen_params_t;

/* Default generation parameters */
#define HOLO_AI_GEN_PARAMS_DEFAULT { \
    .temperature = 0.7f,             \
    .top_p = 0.9f,                   \
    .top_k = 40,                     \
    .min_p = 0.05f,                  \
    .typical_p = 1.0f,               \
    .repeat_penalty = 1.1f,          \
    .presence_penalty = 0.0f,        \
    .frequency_penalty = 0.0f,       \
    .repeat_last_n = 64,             \
    .max_tokens = 2048,              \
    .seed = -1,                      \
    .stop_sequences = {{0}},         \
    .num_stop_sequences = 0,         \
    .mirostat = 0,                   \
    .mirostat_tau = 5.0f,            \
    .mirostat_eta = 0.1f,            \
    .grammar = NULL,                 \
    .json_schema = NULL,             \
    .stream = true                   \
}

/* ============================================================================
 * Model Loading Options
 * ============================================================================ */

typedef struct {
    holo_ai_backend_t backend;

    /* GPU settings */
    int32_t gpu_layers;         /* Layers to offload to GPU (-1 = all) */
    int32_t main_gpu;           /* Main GPU device index */
    float *tensor_split;        /* GPU memory split ratios (multi-GPU) */
    int32_t num_gpus;           /* Number of GPUs to use */

    /* Memory settings */
    uint32_t context_size;      /* Context size to allocate */
    uint32_t batch_size;        /* Batch size for prompt processing */
    bool use_mmap;              /* Memory-map the model file */
    bool use_mlock;             /* Lock model in RAM */

    /* Threading */
    int32_t num_threads;        /* CPU threads for inference */
    int32_t num_batch_threads;  /* CPU threads for batch processing */

    /* KV cache */
    holo_ai_quant_t kv_cache_type;  /* Quantization for KV cache */
    bool flash_attention;       /* Use flash attention if available */

    /* Rope settings (for extended context) */
    float rope_freq_base;       /* RoPE base frequency */
    float rope_freq_scale;      /* RoPE frequency scale */

    /* LoRA adapters */
    const char **lora_paths;    /* Paths to LoRA adapters */
    float *lora_scales;         /* Scale factors for LoRAs */
    int32_t num_loras;          /* Number of LoRA adapters */

    /* Progress callback */
    void (*progress_callback)(float progress, void *user_data);
    void *progress_user_data;
} holo_ai_load_options_t;

/* Default load options */
#define HOLO_AI_LOAD_OPTIONS_DEFAULT { \
    .backend = HOLO_AI_BACKEND_AUTO,   \
    .gpu_layers = -1,                  \
    .main_gpu = 0,                     \
    .tensor_split = NULL,              \
    .num_gpus = 1,                     \
    .context_size = 4096,              \
    .batch_size = 512,                 \
    .use_mmap = true,                  \
    .use_mlock = false,                \
    .num_threads = 0,                  \
    .num_batch_threads = 0,            \
    .kv_cache_type = HOLO_AI_QUANT_F16,\
    .flash_attention = true,           \
    .rope_freq_base = 0.0f,            \
    .rope_freq_scale = 0.0f,           \
    .lora_paths = NULL,                \
    .lora_scales = NULL,               \
    .num_loras = 0,                    \
    .progress_callback = NULL,         \
    .progress_user_data = NULL         \
}

/* ============================================================================
 * Chat Message
 * ============================================================================ */

typedef struct {
    holo_ai_role_t role;
    const char *content;
    const char *name;           /* Optional: for multi-agent scenarios */
    const char *tool_call_id;   /* Optional: for tool responses */
} holo_ai_message_t;

/* ============================================================================
 * Generation Result
 * ============================================================================ */

typedef struct {
    char *text;                 /* Generated text (caller must free) */
    size_t text_length;

    holo_ai_token_t *tokens;    /* Generated tokens (caller must free) */
    size_t num_tokens;

    /* Statistics */
    int32_t prompt_tokens;
    int32_t completion_tokens;
    int32_t total_tokens;

    /* Timing (milliseconds) */
    double prompt_eval_time;
    double generation_time;
    double tokens_per_second;

    /* Stop reason */
    enum {
        HOLO_AI_STOP_EOS,       /* Hit end of sequence token */
        HOLO_AI_STOP_LENGTH,    /* Hit max_tokens limit */
        HOLO_AI_STOP_SEQUENCE,  /* Hit stop sequence */
        HOLO_AI_STOP_CANCELLED, /* Generation cancelled */
        HOLO_AI_STOP_ERROR      /* Error occurred */
    } stop_reason;

    /* If stopped by sequence, which one */
    int32_t stop_sequence_index;
} holo_ai_result_t;

/* ============================================================================
 * Streaming Callback
 * ============================================================================ */

/* Return false to cancel generation */
typedef bool (*holo_ai_stream_callback_t)(
    const char *token_text,
    holo_ai_token_t token_id,
    void *user_data
);

/* ============================================================================
 * Opaque Types
 * ============================================================================ */

typedef struct holo_ai_engine holo_ai_engine_t;
typedef struct holo_ai_model holo_ai_model_t;
typedef struct holo_ai_context holo_ai_context_t;
typedef struct holo_ai_tokenizer holo_ai_tokenizer_t;

/* ============================================================================
 * Engine Management
 * ============================================================================ */

/* Initialize the AI engine (call once at startup) */
HOLO_API holo_ai_error_t holo_ai_init(void);

/* Shutdown the AI engine (call once at exit) */
HOLO_API void holo_ai_shutdown(void);

/* Get engine instance */
HOLO_API holo_ai_engine_t *holo_ai_get_engine(void);

/* Get error string */
HOLO_API const char *holo_ai_error_string(holo_ai_error_t error);

/* Get available backends */
HOLO_API int holo_ai_get_available_backends(holo_ai_backend_t *backends, int max_count);
HOLO_API const char *holo_ai_backend_name(holo_ai_backend_t backend);

/* ============================================================================
 * Model Management
 * ============================================================================ */

/* Load a model from file (GGUF format) */
HOLO_API holo_ai_model_t *holo_ai_model_load(
    const char *path,
    const holo_ai_load_options_t *options
);

/* Unload a model */
HOLO_API void holo_ai_model_free(holo_ai_model_t *model);

/* Get model information */
HOLO_API const holo_ai_model_info_t *holo_ai_model_info(holo_ai_model_t *model);

/* Check if model supports a feature */
HOLO_API bool holo_ai_model_supports(holo_ai_model_t *model, const char *feature);

/* ============================================================================
 * Context Management
 * ============================================================================ */

/* Create inference context */
HOLO_API holo_ai_context_t *holo_ai_context_create(
    holo_ai_model_t *model,
    uint32_t context_size
);

/* Free context */
HOLO_API void holo_ai_context_free(holo_ai_context_t *ctx);

/* Clear context (reset KV cache) */
HOLO_API void holo_ai_context_clear(holo_ai_context_t *ctx);

/* Get current context usage */
HOLO_API size_t holo_ai_context_used(holo_ai_context_t *ctx);
HOLO_API size_t holo_ai_context_max(holo_ai_context_t *ctx);

/* Save/restore context state */
HOLO_API holo_ai_error_t holo_ai_context_save(holo_ai_context_t *ctx, const char *path);
HOLO_API holo_ai_error_t holo_ai_context_load(holo_ai_context_t *ctx, const char *path);

/* ============================================================================
 * Tokenization
 * ============================================================================ */

/* Tokenize text */
HOLO_API holo_ai_error_t holo_ai_tokenize(
    holo_ai_model_t *model,
    const char *text,
    holo_ai_token_t *tokens,
    size_t *num_tokens,
    bool add_special
);

/* Detokenize tokens to text */
HOLO_API holo_ai_error_t holo_ai_detokenize(
    holo_ai_model_t *model,
    const holo_ai_token_t *tokens,
    size_t num_tokens,
    char *text,
    size_t *text_length
);

/* Get token string */
HOLO_API const char *holo_ai_token_to_string(
    holo_ai_model_t *model,
    holo_ai_token_t token
);

/* ============================================================================
 * Chat Template
 * ============================================================================ */

/* Apply chat template to messages */
HOLO_API holo_ai_error_t holo_ai_apply_chat_template(
    holo_ai_model_t *model,
    const holo_ai_message_t *messages,
    size_t num_messages,
    bool add_generation_prompt,
    char *output,
    size_t *output_length
);

/* ============================================================================
 * Text Generation
 * ============================================================================ */

/* Generate completion for a prompt */
HOLO_API holo_ai_error_t holo_ai_generate(
    holo_ai_context_t *ctx,
    const char *prompt,
    const holo_ai_gen_params_t *params,
    holo_ai_stream_callback_t callback,
    void *user_data,
    holo_ai_result_t *result
);

/* Generate chat completion */
HOLO_API holo_ai_error_t holo_ai_chat(
    holo_ai_context_t *ctx,
    const holo_ai_message_t *messages,
    size_t num_messages,
    const holo_ai_gen_params_t *params,
    holo_ai_stream_callback_t callback,
    void *user_data,
    holo_ai_result_t *result
);

/* Cancel ongoing generation */
HOLO_API void holo_ai_cancel(holo_ai_context_t *ctx);

/* Free generation result */
HOLO_API void holo_ai_result_free(holo_ai_result_t *result);

/* ============================================================================
 * Embeddings
 * ============================================================================ */

/* Generate embeddings for text */
HOLO_API holo_ai_error_t holo_ai_embed(
    holo_ai_model_t *model,
    const char *text,
    float *embedding,
    size_t *embedding_dim
);

/* Generate embeddings for multiple texts (batched) */
HOLO_API holo_ai_error_t holo_ai_embed_batch(
    holo_ai_model_t *model,
    const char **texts,
    size_t num_texts,
    float *embeddings,
    size_t *embedding_dim
);

/* ============================================================================
 * Utilities
 * ============================================================================ */

/* Estimate memory requirements for a model */
HOLO_API uint64_t holo_ai_estimate_memory(
    const char *model_path,
    uint32_t context_size,
    holo_ai_backend_t backend
);

/* Get system memory info */
HOLO_API void holo_ai_get_memory_info(
    uint64_t *total_ram,
    uint64_t *available_ram,
    uint64_t *total_vram,
    uint64_t *available_vram
);

/* Get GPU info */
typedef struct {
    char name[128];
    uint64_t total_memory;
    uint64_t available_memory;
    int compute_capability;
    holo_ai_backend_t backend;
} holo_ai_gpu_info_t;

HOLO_API int holo_ai_get_gpu_info(holo_ai_gpu_info_t *gpus, int max_count);

#ifdef __cplusplus
}
#endif

#endif /* HOLO_AI_H */
