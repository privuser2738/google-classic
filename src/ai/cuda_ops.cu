/*
 * Holo - CUDA Operations Implementation
 * GPU-accelerated tensor operations for LLM inference
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern "C" {
#include "cuda_ops.h"
}

/* ============================================================================
 * Constants and Macros
 * ============================================================================ */

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

#define CUDA_CHECK_RET(call, ret) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return ret; \
    } \
} while(0)

/* Block sizes for different kernels */
#define BLOCK_SIZE 256
#define WARP_SIZE 32

/* Quantization block sizes matching GGML */
#define QK8_0 32
#define QK4_0 32
#define QK4_K 256
#define QK6_K 256

/* ============================================================================
 * Quantization Structures (must match CPU versions)
 * ============================================================================ */

typedef struct {
    __half d;           /* delta (scale) as f16 */
    int8_t qs[QK8_0];   /* quantized values */
} block_q8_0;

typedef struct {
    __half d;           /* delta (scale) as f16 */
    uint8_t qs[QK4_0/2]; /* 32 x 4-bit values packed */
} block_q4_0;

/* Q4_K superblock: 256 values with more complex quantization */
typedef struct {
    __half d;           /* super-block scale */
    __half dmin;        /* super-block min */
    uint8_t scales[12]; /* scales and mins for sub-blocks */
    uint8_t qs[128];    /* quantized values */
} block_q4_k;

/* Q6_K: 256 values with 6-bit quantization */
typedef struct {
    uint8_t ql[128];    /* low 4 bits of quants */
    uint8_t qh[64];     /* high 2 bits of quants */
    int8_t scales[16];  /* scales */
    __half d;           /* delta */
} block_q6_k;

/* ============================================================================
 * Global State
 * ============================================================================ */

static bool g_cuda_initialized = false;
static int g_device_id = 0;
static cudaDeviceProp g_device_props;

/* ============================================================================
 * Initialization
 * ============================================================================ */

extern "C" int cuda_init(void) {
    if (g_cuda_initialized) return 0;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "CUDA: No CUDA devices found\n");
        return -1;
    }

    /* Use device 0 by default */
    g_device_id = 0;
    err = cudaSetDevice(g_device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA: Failed to set device %d\n", g_device_id);
        return -1;
    }

    err = cudaGetDeviceProperties(&g_device_props, g_device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA: Failed to get device properties\n");
        return -1;
    }

    g_cuda_initialized = true;
    return 0;
}

extern "C" void cuda_cleanup(void) {
    if (g_cuda_initialized) {
        cudaDeviceReset();
        g_cuda_initialized = false;
    }
}

extern "C" bool cuda_available(void) {
    return g_cuda_initialized;
}

extern "C" void cuda_print_info(void) {
    if (!g_cuda_initialized) {
        printf("CUDA: Not initialized\n");
        return;
    }

    printf("CUDA Device: %s\n", g_device_props.name);
    printf("  Compute capability: %d.%d\n", g_device_props.major, g_device_props.minor);
    printf("  Total memory: %.2f GB\n", g_device_props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  SM count: %d\n", g_device_props.multiProcessorCount);
    printf("  Max threads/block: %d\n", g_device_props.maxThreadsPerBlock);
    printf("  Warp size: %d\n", g_device_props.warpSize);
}

extern "C" int cuda_get_device_count(void) {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

extern "C" size_t cuda_get_free_memory(void) {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

extern "C" const char* cuda_get_device_name(void) {
    return g_cuda_initialized ? g_device_props.name : "No device";
}

/* ============================================================================
 * Memory Management
 * ============================================================================ */

extern "C" void* cuda_malloc(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    return (err == cudaSuccess) ? ptr : nullptr;
}

extern "C" void cuda_free(void* ptr) {
    if (ptr) cudaFree(ptr);
}

extern "C" void cuda_memcpy_h2d(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

extern "C" void cuda_memcpy_d2h(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

extern "C" void cuda_memcpy_d2d(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
}

extern "C" void cuda_memset(void* ptr, int value, size_t size) {
    CUDA_CHECK(cudaMemset(ptr, value, size));
}

extern "C" void cuda_sync(void) {
    cudaDeviceSynchronize();
}

/* ============================================================================
 * Vector Kernels
 * ============================================================================ */

__global__ void vec_add_kernel(float* dst, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = a[i] + b[i];
    }
}

__global__ void vec_mul_kernel(float* dst, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = a[i] * b[i];
    }
}

__global__ void vec_scale_kernel(float* dst, const float* a, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = a[i] * scale;
    }
}

/* Parallel reduction for dot product */
__global__ void vec_dot_kernel(float* result, const float* a, const float* b, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    /* Load and multiply */
    sdata[tid] = (i < n) ? a[i] * b[i] : 0.0f;
    __syncthreads();

    /* Reduce within block */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    /* Write block result */
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

extern "C" void cuda_vec_add(float* dst, const float* a, const float* b, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vec_add_kernel<<<blocks, BLOCK_SIZE>>>(dst, a, b, n);
}

extern "C" void cuda_vec_mul(float* dst, const float* a, const float* b, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vec_mul_kernel<<<blocks, BLOCK_SIZE>>>(dst, a, b, n);
}

extern "C" void cuda_vec_scale(float* dst, const float* a, float scale, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vec_scale_kernel<<<blocks, BLOCK_SIZE>>>(dst, a, scale, n);
}

extern "C" float cuda_vec_dot(const float* a, const float* b, int n) {
    float* d_result;
    float h_result = 0.0f;

    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vec_dot_kernel<<<blocks, BLOCK_SIZE>>>(d_result, a, b, n);

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return h_result;
}

/* ============================================================================
 * Matrix-Vector Multiplication Kernels
 * GGUF layout: M[in_dim, out_dim], compute dst[j] = sum_i(M[i,j] * v[i])
 * ============================================================================ */

/* F32 matmul - one thread per output element */
__global__ void matmul_f32_kernel(float* dst, const float* M, const float* v,
                                   int out_dim, int in_dim) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out_dim) return;

    float sum = 0.0f;
    for (int i = 0; i < in_dim; i++) {
        sum += M[i * out_dim + j] * v[i];
    }
    dst[j] = sum;
}

/* F16 matmul */
__global__ void matmul_f16_kernel(float* dst, const __half* M, const float* v,
                                   int out_dim, int in_dim) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out_dim) return;

    float sum = 0.0f;
    for (int i = 0; i < in_dim; i++) {
        sum += __half2float(M[i * out_dim + j]) * v[i];
    }
    dst[j] = sum;
}

/* Q8_0 matmul - optimized version */
__global__ void matmul_q8_0_kernel(float* dst, const block_q8_0* M, const float* v,
                                    int out_dim, int in_dim) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out_dim) return;

    int blocks_per_row = out_dim / QK8_0;
    int block_idx = j / QK8_0;
    int within_block = j % QK8_0;

    float sum = 0.0f;
    for (int i = 0; i < in_dim; i++) {
        const block_q8_0* row = M + i * blocks_per_row;
        float scale = __half2float(row[block_idx].d);
        sum += scale * row[block_idx].qs[within_block] * v[i];
    }
    dst[j] = sum;
}

/* Q4_0 matmul */
__global__ void matmul_q4_0_kernel(float* dst, const block_q4_0* M, const float* v,
                                    int out_dim, int in_dim) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out_dim) return;

    int blocks_per_row = out_dim / QK4_0;
    int block_idx = j / QK4_0;
    int within_block = j % QK4_0;

    float sum = 0.0f;
    for (int i = 0; i < in_dim; i++) {
        const block_q4_0* row = M + i * blocks_per_row;
        float scale = __half2float(row[block_idx].d);

        int byte_idx = within_block < 16 ? within_block : within_block - 16;
        uint8_t byte = row[block_idx].qs[byte_idx];
        int8_t val = within_block < 16 ? (byte & 0xF) - 8 : (byte >> 4) - 8;

        sum += scale * val * v[i];
    }
    dst[j] = sum;
}

/* Q4_K matmul (superblock quantization) */
__global__ void matmul_q4_k_kernel(float* dst, const block_q4_k* M, const float* v,
                                    int out_dim, int in_dim) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out_dim) return;

    int blocks_per_row = out_dim / QK4_K;
    int block_idx = j / QK4_K;
    int within_block = j % QK4_K;

    float sum = 0.0f;
    for (int i = 0; i < in_dim; i++) {
        const block_q4_k* blk = M + i * blocks_per_row + block_idx;

        /* Decode Q4_K format */
        float d = __half2float(blk->d);
        float dmin = __half2float(blk->dmin);

        /* Each superblock has 8 sub-blocks of 32 values */
        int sub_block = within_block / 32;
        int within_sub = within_block % 32;

        /* Get scale and min for this sub-block */
        uint8_t sc_byte = blk->scales[sub_block];
        float scale = d * (sc_byte & 0x3F);
        float min_val = dmin * (sc_byte >> 6);

        /* Get quantized value */
        int q_idx = within_block / 2;
        uint8_t qbyte = blk->qs[q_idx];
        int8_t qval = (within_block & 1) ? (qbyte >> 4) : (qbyte & 0xF);

        float val = scale * qval - min_val;
        sum += val * v[i];
    }
    dst[j] = sum;
}

/* Q6_K matmul */
__global__ void matmul_q6_k_kernel(float* dst, const block_q6_k* M, const float* v,
                                    int out_dim, int in_dim) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out_dim) return;

    int blocks_per_row = out_dim / QK6_K;
    int block_idx = j / QK6_K;
    int within_block = j % QK6_K;

    float sum = 0.0f;
    for (int i = 0; i < in_dim; i++) {
        const block_q6_k* blk = M + i * blocks_per_row + block_idx;
        float d = __half2float(blk->d);

        /* Decode Q6_K: 4 bits from ql + 2 bits from qh */
        int sub_block = within_block / 16;
        int within_sub = within_block % 16;

        int8_t scale = blk->scales[sub_block];
        uint8_t ql = blk->ql[within_block / 2];
        uint8_t qh = blk->qh[within_block / 4];

        int ql_val = (within_block & 1) ? (ql >> 4) : (ql & 0xF);
        int qh_shift = (within_block % 4) * 2;
        int qh_val = (qh >> qh_shift) & 0x3;

        int q = ql_val | (qh_val << 4);
        q -= 32;  /* Q6_K uses offset of 32 */

        float val = d * scale * q;
        sum += val * v[i];
    }
    dst[j] = sum;
}

extern "C" void cuda_matmul_f32(float* dst, const float* M, const float* v,
                                 int out_dim, int in_dim) {
    int blocks = (out_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    matmul_f32_kernel<<<blocks, BLOCK_SIZE>>>(dst, M, v, out_dim, in_dim);
}

extern "C" void cuda_matmul_f16(float* dst, const void* M, const float* v,
                                 int out_dim, int in_dim) {
    int blocks = (out_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    matmul_f16_kernel<<<blocks, BLOCK_SIZE>>>(dst, (const __half*)M, v, out_dim, in_dim);
}

extern "C" void cuda_matmul_q8_0(float* dst, const void* M, const float* v,
                                  int out_dim, int in_dim) {
    int blocks = (out_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    matmul_q8_0_kernel<<<blocks, BLOCK_SIZE>>>(dst, (const block_q8_0*)M, v, out_dim, in_dim);
}

extern "C" void cuda_matmul_q4_0(float* dst, const void* M, const float* v,
                                  int out_dim, int in_dim) {
    int blocks = (out_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    matmul_q4_0_kernel<<<blocks, BLOCK_SIZE>>>(dst, (const block_q4_0*)M, v, out_dim, in_dim);
}

extern "C" void cuda_matmul_q4_k(float* dst, const void* M, const float* v,
                                  int out_dim, int in_dim) {
    int blocks = (out_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    matmul_q4_k_kernel<<<blocks, BLOCK_SIZE>>>(dst, (const block_q4_k*)M, v, out_dim, in_dim);
}

extern "C" void cuda_matmul_q6_k(float* dst, const void* M, const float* v,
                                  int out_dim, int in_dim) {
    int blocks = (out_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    matmul_q6_k_kernel<<<blocks, BLOCK_SIZE>>>(dst, (const block_q6_k*)M, v, out_dim, in_dim);
}

/* ============================================================================
 * Activation Kernels
 * ============================================================================ */

__global__ void silu_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

__global__ void gelu_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        const float sqrt_2_pi = 0.7978845608f;
        x[i] = 0.5f * v * (1.0f + tanhf(sqrt_2_pi * (v + 0.044715f * v * v * v)));
    }
}

__global__ void relu_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (x[i] < 0) x[i] = 0;
    }
}

extern "C" void cuda_silu(float* x, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    silu_kernel<<<blocks, BLOCK_SIZE>>>(x, n);
}

extern "C" void cuda_gelu(float* x, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gelu_kernel<<<blocks, BLOCK_SIZE>>>(x, n);
}

extern "C" void cuda_relu(float* x, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_kernel<<<blocks, BLOCK_SIZE>>>(x, n);
}

/* ============================================================================
 * Normalization Kernels
 * ============================================================================ */

__global__ void rms_norm_kernel(float* dst, const float* x, const float* weight,
                                 int n, float eps, float* sum_sq) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    /* Phase 1: Compute sum of squares */
    sdata[tid] = (i < n) ? x[i] * x[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(sum_sq, sdata[0]);
}

__global__ void rms_norm_apply_kernel(float* dst, const float* x, const float* weight,
                                       int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = x[i] * scale * weight[i];
    }
}

extern "C" void cuda_rms_norm(float* dst, const float* x, const float* weight, int n, float eps) {
    float* d_sum_sq;
    float h_sum_sq = 0.0f;

    cudaMalloc(&d_sum_sq, sizeof(float));
    cudaMemset(d_sum_sq, 0, sizeof(float));

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rms_norm_kernel<<<blocks, BLOCK_SIZE>>>(dst, x, weight, n, eps, d_sum_sq);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_sum_sq, d_sum_sq, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_sum_sq);

    float rms = sqrtf(h_sum_sq / n + eps);
    float scale = 1.0f / rms;

    rms_norm_apply_kernel<<<blocks, BLOCK_SIZE>>>(dst, x, weight, n, scale);
}

extern "C" void cuda_layer_norm(float* dst, const float* x, const float* weight,
                                 const float* bias, int n, float eps) {
    /* TODO: Implement full layer norm with mean subtraction */
    /* For now, use RMS norm as approximation */
    cuda_rms_norm(dst, x, weight, n, eps);
}

/* ============================================================================
 * Softmax Kernel
 * ============================================================================ */

__global__ void softmax_find_max_kernel(const float* x, float* max_val, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? x[i] : -INFINITY;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid] < sdata[tid + s]) {
            sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax((int*)max_val, __float_as_int(sdata[0]));
    }
}

__global__ void softmax_exp_sum_kernel(const float* x, float* dst, float max_val,
                                        float* sum, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (i < n) {
        val = expf(x[i] - max_val);
        dst[i] = val;
    }
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(sum, sdata[0]);
}

__global__ void softmax_normalize_kernel(float* dst, float inv_sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] *= inv_sum;
    }
}

extern "C" void cuda_softmax(float* dst, const float* x, int n) {
    float *d_max, *d_sum;
    float h_max = -INFINITY, h_sum = 0.0f;

    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemcpy(d_max, &h_max, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(float));

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* Find max */
    softmax_find_max_kernel<<<blocks, BLOCK_SIZE>>>(x, d_max, n);
    cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);

    /* Compute exp and sum */
    softmax_exp_sum_kernel<<<blocks, BLOCK_SIZE>>>(x, dst, h_max, d_sum, n);
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    /* Normalize */
    float inv_sum = 1.0f / h_sum;
    softmax_normalize_kernel<<<blocks, BLOCK_SIZE>>>(dst, inv_sum, n);

    cudaFree(d_max);
    cudaFree(d_sum);
}

extern "C" void cuda_softmax_inplace(float* x, int n) {
    cuda_softmax(x, x, n);
}

/* ============================================================================
 * Attention Kernels
 * ============================================================================ */

__global__ void attention_scores_kernel(float* dst, const float* Q, const float* K,
                                         int seq_len, int kv_len, int head_dim, float scale) {
    int i = blockIdx.x;  /* query position */
    int j = threadIdx.x; /* key position */

    if (i >= seq_len || j >= kv_len) return;

    float score = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        score += Q[i * head_dim + d] * K[j * head_dim + d];
    }
    dst[i * kv_len + j] = score * scale;
}

__global__ void attention_apply_kernel(float* dst, const float* scores, const float* V,
                                        int seq_len, int kv_len, int head_dim) {
    int i = blockIdx.x;  /* query position */
    int d = threadIdx.x; /* head dimension */

    if (i >= seq_len || d >= head_dim) return;

    float sum = 0.0f;
    for (int j = 0; j < kv_len; j++) {
        sum += scores[i * kv_len + j] * V[j * head_dim + d];
    }
    dst[i * head_dim + d] = sum;
}

extern "C" void cuda_attention_scores(float* dst, const float* Q, const float* K,
                                       int seq_len, int kv_len, int head_dim, float scale) {
    dim3 grid(seq_len);
    dim3 block(min(kv_len, 1024));
    attention_scores_kernel<<<grid, block>>>(dst, Q, K, seq_len, kv_len, head_dim, scale);
}

extern "C" void cuda_attention_apply(float* dst, const float* scores, const float* V,
                                      int seq_len, int kv_len, int head_dim) {
    dim3 grid(seq_len);
    dim3 block(min(head_dim, 1024));
    attention_apply_kernel<<<grid, block>>>(dst, scores, V, seq_len, kv_len, head_dim);
}

/* Single-query attention kernel for autoregressive generation
 * One block computes the full attention for one query against all cached KV
 * Much more efficient than calling multiple kernels per position
 *
 * Uses warp-level reductions for correctness
 */
__global__ void attention_single_kernel(float* dst, const float* q,
                                         const float* k_cache, const float* v_cache,
                                         float* att_buf, int kv_len, int head_dim,
                                         int kv_head, int kv_dim) {
    extern __shared__ float sdata[];
    float* scores = sdata;  /* [kv_len] */
    /* Use last 32 floats for reduction scratch */
    float* reduce_scratch = sdata + kv_len;

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Step 1: Compute attention scores Q · K for all positions */
    for (int p = tid; p < kv_len; p += block_size) {
        const float* k_p = k_cache + p * kv_dim + kv_head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q[d] * k_p[d];
        }
        scores[p] = score * scale;
    }
    __syncthreads();

    /* Step 2: Softmax - find max using parallel reduction */
    float my_max = -1e30f;
    for (int p = tid; p < kv_len; p += block_size) {
        my_max = fmaxf(my_max, scores[p]);
    }

    /* Warp-level reduction for max */
    for (int offset = 16; offset > 0; offset /= 2) {
        my_max = fmaxf(my_max, __shfl_down_sync(0xffffffff, my_max, offset));
    }

    /* First thread of each warp writes to shared memory */
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = (block_size + 31) / 32;

    if (lane_id == 0 && warp_id < 8) {
        reduce_scratch[warp_id] = my_max;
    }
    __syncthreads();

    /* First warp reduces across warps */
    if (tid < num_warps && tid < 8) {
        my_max = reduce_scratch[tid];
    } else if (tid < 32) {
        my_max = -1e30f;
    }
    if (tid < 32) {
        for (int offset = 16; offset > 0; offset /= 2) {
            my_max = fmaxf(my_max, __shfl_down_sync(0xffffffff, my_max, offset));
        }
    }
    if (tid == 0) {
        reduce_scratch[0] = my_max;
    }
    __syncthreads();
    float max_val = reduce_scratch[0];

    /* Step 3: Softmax - exp and sum */
    float my_sum = 0.0f;
    for (int p = tid; p < kv_len; p += block_size) {
        scores[p] = expf(scores[p] - max_val);
        my_sum += scores[p];
    }

    /* Warp-level reduction for sum */
    for (int offset = 16; offset > 0; offset /= 2) {
        my_sum += __shfl_down_sync(0xffffffff, my_sum, offset);
    }

    if (lane_id == 0 && warp_id < 8) {
        reduce_scratch[warp_id] = my_sum;
    }
    __syncthreads();

    /* First warp reduces across warps */
    if (tid < num_warps && tid < 8) {
        my_sum = reduce_scratch[tid];
    } else if (tid < 32) {
        my_sum = 0.0f;
    }
    if (tid < 32) {
        for (int offset = 16; offset > 0; offset /= 2) {
            my_sum += __shfl_down_sync(0xffffffff, my_sum, offset);
        }
    }
    if (tid == 0) {
        reduce_scratch[0] = my_sum;
    }
    __syncthreads();
    float sum_val = reduce_scratch[0];

    /* Step 4: Normalize scores */
    float inv_sum = 1.0f / sum_val;
    for (int p = tid; p < kv_len; p += block_size) {
        scores[p] *= inv_sum;
    }
    __syncthreads();

    /* Step 5: Weighted sum of values: out = sum(scores[p] * V[p]) */
    for (int d = tid; d < head_dim; d += block_size) {
        float out_d = 0.0f;
        for (int p = 0; p < kv_len; p++) {
            const float* v_p = v_cache + p * kv_dim + kv_head * head_dim;
            out_d += scores[p] * v_p[d];
        }
        dst[d] = out_d;
    }
}

extern "C" void cuda_attention_single(float* dst, const float* q,
                                       const float* k_cache, const float* v_cache,
                                       float* att_buf, int kv_len, int head_dim,
                                       int kv_head, int kv_dim) {
    /* Use enough threads to cover both kv_len (for scores) and head_dim (for output) */
    int threads = min(max(kv_len, head_dim), 256);
    /* Shared memory for attention scores + 8 floats for reduction scratch */
    size_t smem = (kv_len + 8) * sizeof(float);
    attention_single_kernel<<<1, threads, smem>>>(dst, q, k_cache, v_cache,
                                                   att_buf, kv_len, head_dim,
                                                   kv_head, kv_dim);
}

/* ============================================================================
 * RoPE Kernel
 * ============================================================================ */

__global__ void rope_apply_kernel(float* q, float* k, int pos, int head_dim,
                                   int n_heads, int n_kv_heads, float freq_base) {
    int h = blockIdx.x;  /* head index */
    int i = threadIdx.x; /* dimension within half head_dim */

    int half_dim = head_dim / 2;
    if (i >= half_dim) return;

    float freq = 1.0f / powf(freq_base, (float)(2 * i) / head_dim);
    float theta = pos * freq;
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);

    /* Apply to Q (all heads) */
    if (h < n_heads) {
        float* q_h = q + h * head_dim;
        float q0 = q_h[i];
        float q1 = q_h[i + half_dim];
        q_h[i] = q0 * cos_t - q1 * sin_t;
        q_h[i + half_dim] = q0 * sin_t + q1 * cos_t;
    }

    /* Apply to K (n_kv_heads) */
    if (h < n_kv_heads) {
        float* k_h = k + h * head_dim;
        float k0 = k_h[i];
        float k1 = k_h[i + half_dim];
        k_h[i] = k0 * cos_t - k1 * sin_t;
        k_h[i + half_dim] = k0 * sin_t + k1 * cos_t;
    }
}

extern "C" void cuda_rope_apply(float* q, float* k, int pos, int head_dim,
                                 int n_heads, int n_kv_heads, float freq_base) {
    int max_heads = max(n_heads, n_kv_heads);
    dim3 grid(max_heads);
    dim3 block(head_dim / 2);
    rope_apply_kernel<<<grid, block>>>(q, k, pos, head_dim, n_heads, n_kv_heads, freq_base);
}

/* ============================================================================
 * Embedding Lookup Kernels
 * ============================================================================ */

__global__ void get_embedding_f32_kernel(float* dst, const float* embeddings,
                                          int token, int dim, int vocab_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        /* GGUF layout: [dim, vocab] */
        dst[i] = embeddings[i * vocab_size + token];
    }
}

__global__ void get_embedding_f16_kernel(float* dst, const __half* embeddings,
                                          int token, int dim, int vocab_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        dst[i] = __half2float(embeddings[i * vocab_size + token]);
    }
}

__global__ void get_embedding_q8_0_kernel(float* dst, const block_q8_0* embeddings,
                                           int token, int dim, int vocab_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        int blocks_per_row = vocab_size / QK8_0;
        int block_idx = token / QK8_0;
        int within_block = token % QK8_0;

        const block_q8_0* row = embeddings + i * blocks_per_row;
        float scale = __half2float(row[block_idx].d);
        dst[i] = scale * row[block_idx].qs[within_block];
    }
}

extern "C" void cuda_get_embedding_f32(float* dst, const float* embeddings,
                                        int token, int dim, int vocab_size) {
    int blocks = (dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    get_embedding_f32_kernel<<<blocks, BLOCK_SIZE>>>(dst, embeddings, token, dim, vocab_size);
}

extern "C" void cuda_get_embedding_f16(float* dst, const void* embeddings,
                                        int token, int dim, int vocab_size) {
    int blocks = (dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    get_embedding_f16_kernel<<<blocks, BLOCK_SIZE>>>(dst, (const __half*)embeddings,
                                                       token, dim, vocab_size);
}

extern "C" void cuda_get_embedding_q8_0(float* dst, const void* embeddings,
                                         int token, int dim, int vocab_size) {
    int blocks = (dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    get_embedding_q8_0_kernel<<<blocks, BLOCK_SIZE>>>(dst, (const block_q8_0*)embeddings,
                                                        token, dim, vocab_size);
}

/* ============================================================================
 * Sampling Helper
 * ============================================================================ */

__global__ void argmax_kernel(const float* x, int* result, int n) {
    __shared__ float max_vals[BLOCK_SIZE];
    __shared__ int max_idxs[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    max_vals[tid] = (i < n) ? x[i] : -INFINITY;
    max_idxs[tid] = i;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && max_vals[tid] < max_vals[tid + s]) {
            max_vals[tid] = max_vals[tid + s];
            max_idxs[tid] = max_idxs[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        /* Atomic compare and swap for global max */
        /* Simplified: just write first block's result */
        if (blockIdx.x == 0) {
            *result = max_idxs[0];
        }
    }
}

extern "C" int cuda_argmax(const float* x, int n) {
    int* d_result;
    int h_result = 0;

    cudaMalloc(&d_result, sizeof(int));

    /* For simplicity, use single block for small vocab or do proper reduction */
    int blocks = 1;
    int threads = min(n, BLOCK_SIZE);
    argmax_kernel<<<blocks, threads>>>(x, d_result, min(n, BLOCK_SIZE));

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return h_result;
}
