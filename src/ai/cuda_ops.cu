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

/* CUDA streams for async operations */
static cudaStream_t g_compute_stream = nullptr;
static cudaStream_t g_copy_stream = nullptr;

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

    /* Create CUDA streams for async operations */
    err = cudaStreamCreate(&g_compute_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA: Failed to create compute stream\n");
        return -1;
    }
    err = cudaStreamCreate(&g_copy_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA: Failed to create copy stream\n");
        cudaStreamDestroy(g_compute_stream);
        return -1;
    }

    g_cuda_initialized = true;
    return 0;
}

extern "C" void cuda_cleanup(void) {
    if (g_cuda_initialized) {
        if (g_compute_stream) {
            cudaStreamDestroy(g_compute_stream);
            g_compute_stream = nullptr;
        }
        if (g_copy_stream) {
            cudaStreamDestroy(g_copy_stream);
            g_copy_stream = nullptr;
        }
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

/* Async memory copy on copy stream - allows overlap with compute */
extern "C" void cuda_memcpy_d2d_async(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, g_copy_stream));
}

extern "C" void cuda_memcpy_d2h_async(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, g_copy_stream));
}

extern "C" void cuda_memset(void* ptr, int value, size_t size) {
    CUDA_CHECK(cudaMemset(ptr, value, size));
}

extern "C" void cuda_sync(void) {
    cudaDeviceSynchronize();
}

/* Synchronize just the copy stream */
extern "C" void cuda_sync_copy(void) {
    if (g_copy_stream) cudaStreamSynchronize(g_copy_stream);
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

/* Fused residual add using vectorized float4 loads/stores
 * Much faster for large dimensions due to better memory coalescing
 */
__global__ void vec_add_f4_kernel(float* dst, const float* a, const float* b, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 av = *reinterpret_cast<const float4*>(a + i);
        float4 bv = *reinterpret_cast<const float4*>(b + i);
        float4 result;
        result.x = av.x + bv.x;
        result.y = av.y + bv.y;
        result.z = av.z + bv.z;
        result.w = av.w + bv.w;
        *reinterpret_cast<float4*>(dst + i) = result;
    } else if (i < n) {
        /* Handle remaining elements */
        for (int j = i; j < n; j++) {
            dst[j] = a[j] + b[j];
        }
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
    /* Use vectorized kernel for large, aligned arrays (typical LLM dims are mult of 128) */
    if (n >= 256 && (n % 4) == 0) {
        int blocks = (n / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        vec_add_f4_kernel<<<blocks, BLOCK_SIZE>>>(dst, a, b, n);
    } else {
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        vec_add_kernel<<<blocks, BLOCK_SIZE>>>(dst, a, b, n);
    }
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
 * Optimized Matrix-Vector Multiplication Kernels
 *
 * Key optimizations:
 * 1. Tiled computation with shared memory for vector caching
 * 2. Warp-level reductions using __shfl_down_sync
 * 3. Multiple rows per thread block for better occupancy
 * 4. Vectorized loads where possible
 * ============================================================================ */

/* Tile size for shared memory - must be multiple of 32 for Q8_0 blocks */
#define TILE_SIZE 128
#define ROWS_PER_BLOCK 4

/* Optimized Q8_0 matmul using shared memory and warp reductions
 * Each thread block computes ROWS_PER_BLOCK output values
 * Threads within a block cooperatively process input tiles
 */
__global__ void matmul_q8_0_kernel(float* dst, const block_q8_0* M, const float* v,
                                    int out_dim, int in_dim) {
    /* Shared memory for input vector tiles */
    __shared__ float sv[TILE_SIZE];

    int row_base = blockIdx.x * ROWS_PER_BLOCK;
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;

    int blocks_per_row = in_dim / QK8_0;

    /* Each warp accumulates for one output row */
    float sums[ROWS_PER_BLOCK];
    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        sums[r] = 0.0f;
    }

    /* Process input in tiles */
    for (int tile_start = 0; tile_start < in_dim; tile_start += TILE_SIZE) {
        /* Cooperatively load vector tile into shared memory */
        int tile_end = min(tile_start + TILE_SIZE, in_dim);
        for (int i = tid; i < TILE_SIZE && (tile_start + i) < in_dim; i += blockDim.x) {
            sv[i] = v[tile_start + i];
        }
        __syncthreads();

        /* Each thread processes part of the tile for its assigned rows */
        for (int r = 0; r < ROWS_PER_BLOCK; r++) {
            int out_idx = row_base + r;
            if (out_idx >= out_dim) continue;

            const block_q8_0* row = M + out_idx * blocks_per_row;

            /* Process Q8_0 blocks that fall within this tile */
            int block_start = tile_start / QK8_0;
            int block_end = (tile_end + QK8_0 - 1) / QK8_0;

            for (int b = block_start + (tid / QK8_0); b < block_end && b < blocks_per_row;
                 b += (blockDim.x / QK8_0)) {
                int k = tid % QK8_0;
                int global_k = b * QK8_0 + k;
                if (global_k >= tile_start && global_k < tile_end) {
                    float scale = __half2float(row[b].d);
                    int local_k = global_k - tile_start;
                    sums[r] += scale * row[b].qs[k] * sv[local_k];
                }
            }
        }
        __syncthreads();
    }

    /* Warp-level reduction for each row */
    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sums[r] += __shfl_down_sync(0xffffffff, sums[r], offset);
        }
    }

    /* First thread of each warp writes partial result to shared memory */
    __shared__ float partial_sums[ROWS_PER_BLOCK][8];  /* Max 8 warps */
    if (lane == 0 && warp_id < 8) {
        for (int r = 0; r < ROWS_PER_BLOCK; r++) {
            partial_sums[r][warp_id] = sums[r];
        }
    }
    __syncthreads();

    /* First warp reduces across all warps */
    if (warp_id == 0) {
        for (int r = 0; r < ROWS_PER_BLOCK; r++) {
            float val = (lane < warps_per_block) ? partial_sums[r][lane] : 0.0f;
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
            if (lane == 0 && row_base + r < out_dim) {
                dst[row_base + r] = val;
            }
        }
    }
}

/* Simpler but still optimized Q8_0 matmul - one block per output row
 * Uses vectorized float4 loads for better memory bandwidth
 */
__global__ void matmul_q8_0_simple_kernel(float* dst, const block_q8_0* M, const float* v,
                                           int out_dim, int in_dim) {
    int out_idx = blockIdx.x;
    if (out_idx >= out_dim) return;

    int tid = threadIdx.x;
    int blocks_per_row = in_dim / QK8_0;
    const block_q8_0* row = M + out_idx * blocks_per_row;

    /* Each thread processes multiple Q8_0 blocks */
    float sum = 0.0f;

    /* Main loop with vectorized loads where possible */
    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        float scale = __half2float(row[b].d);
        int base = b * QK8_0;

        /* Process 32 int8 values per block */
        float block_sum = 0.0f;

        /* Load vector values as float4 for better bandwidth (if aligned) */
        const float4* v4 = reinterpret_cast<const float4*>(v + base);

        /* Unroll by 8: process 8 float4s = 32 floats per iteration */
        #pragma unroll 8
        for (int j = 0; j < 8; j++) {
            float4 vv = v4[j];
            int k = j * 4;
            block_sum += row[b].qs[k] * vv.x;
            block_sum += row[b].qs[k+1] * vv.y;
            block_sum += row[b].qs[k+2] * vv.z;
            block_sum += row[b].qs[k+3] * vv.w;
        }
        sum += scale * block_sum;
    }

    /* Warp reduction using shuffle */
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    /* Cross-warp reduction using shared memory */
    __shared__ float warp_sums[8];
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (lane == 0 && warp_id < 8) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    /* Final reduction by first warp */
    if (tid < 32) {
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        sum = (tid < num_warps) ? warp_sums[tid] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            dst[out_idx] = sum;
        }
    }
}

/* F32 matmul with shared memory tiling */
__global__ void matmul_f32_kernel(float* dst, const float* M, const float* v,
                                   int out_dim, int in_dim) {
    __shared__ float sv[TILE_SIZE];

    int out_idx = blockIdx.x;
    if (out_idx >= out_dim) return;

    int tid = threadIdx.x;
    const float* row = M + out_idx * in_dim;

    float sum = 0.0f;

    /* Process in tiles */
    for (int tile = 0; tile < in_dim; tile += TILE_SIZE) {
        /* Load tile into shared memory */
        if (tile + tid < in_dim && tid < TILE_SIZE) {
            sv[tid] = v[tile + tid];
        }
        __syncthreads();

        /* Compute partial dot product */
        int tile_end = min(tile + TILE_SIZE, in_dim);
        for (int j = tile + tid; j < tile_end; j += blockDim.x) {
            sum += row[j] * sv[j - tile];
        }
        __syncthreads();
    }

    /* Warp reduction */
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float warp_sums[8];
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (lane == 0 && warp_id < 8) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    if (tid < 32) {
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        sum = (tid < num_warps) ? warp_sums[tid] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            dst[out_idx] = sum;
        }
    }
}

/* F16 matmul with shared memory */
__global__ void matmul_f16_kernel(float* dst, const __half* M, const float* v,
                                   int out_dim, int in_dim) {
    __shared__ float sv[TILE_SIZE];

    int out_idx = blockIdx.x;
    if (out_idx >= out_dim) return;

    int tid = threadIdx.x;
    const __half* row = M + out_idx * in_dim;

    float sum = 0.0f;

    for (int tile = 0; tile < in_dim; tile += TILE_SIZE) {
        if (tile + tid < in_dim && tid < TILE_SIZE) {
            sv[tid] = v[tile + tid];
        }
        __syncthreads();

        int tile_end = min(tile + TILE_SIZE, in_dim);
        for (int j = tile + tid; j < tile_end; j += blockDim.x) {
            sum += __half2float(row[j]) * sv[j - tile];
        }
        __syncthreads();
    }

    /* Warp reduction */
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float warp_sums[8];
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (lane == 0 && warp_id < 8) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    if (tid < 32) {
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        sum = (tid < num_warps) ? warp_sums[tid] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            dst[out_idx] = sum;
        }
    }
}

/* Q4_0 matmul with optimizations */
__global__ void matmul_q4_0_kernel(float* dst, const block_q4_0* M, const float* v,
                                    int out_dim, int in_dim) {
    int out_idx = blockIdx.x;
    if (out_idx >= out_dim) return;

    int tid = threadIdx.x;
    int blocks_per_row = in_dim / QK4_0;
    const block_q4_0* row = M + out_idx * blocks_per_row;

    float sum = 0.0f;
    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        float scale = __half2float(row[b].d);
        int base = b * QK4_0;

        float block_sum = 0.0f;
        #pragma unroll 4
        for (int k = 0; k < 16; k += 4) {
            uint8_t byte0 = row[b].qs[k];
            uint8_t byte1 = row[b].qs[k+1];
            uint8_t byte2 = row[b].qs[k+2];
            uint8_t byte3 = row[b].qs[k+3];

            block_sum += ((byte0 & 0xF) - 8) * v[base + k];
            block_sum += ((byte0 >> 4) - 8) * v[base + k + 16];
            block_sum += ((byte1 & 0xF) - 8) * v[base + k + 1];
            block_sum += ((byte1 >> 4) - 8) * v[base + k + 17];
            block_sum += ((byte2 & 0xF) - 8) * v[base + k + 2];
            block_sum += ((byte2 >> 4) - 8) * v[base + k + 18];
            block_sum += ((byte3 & 0xF) - 8) * v[base + k + 3];
            block_sum += ((byte3 >> 4) - 8) * v[base + k + 19];
        }
        sum += scale * block_sum;
    }

    /* Warp reduction */
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float warp_sums[8];
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (lane == 0 && warp_id < 8) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    if (tid < 32) {
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        sum = (tid < num_warps) ? warp_sums[tid] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            dst[out_idx] = sum;
        }
    }
}

/* Q4_K matmul - row-major layout: M[out_dim, in_dim]
 * Super-block quantization with 256 values per block
 * Matches llama.cpp dequantize_row_q4_K exactly
 */

/* Helper function matching llama.cpp get_scale_min_k4 */
__device__ inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

__global__ void matmul_q4_k_kernel(float* dst, const block_q4_k* M, const float* v,
                                    int out_dim, int in_dim) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_dim) return;

    int blocks_per_row = in_dim / QK4_K;
    const block_q4_k* row = M + out_idx * blocks_per_row;

    float sum = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        const block_q4_k* blk = &row[b];
        float d = __half2float(blk->d);
        float dmin = __half2float(blk->dmin);
        const uint8_t* q = blk->qs;
        int base = b * QK4_K;

        /* Process 256 values in 4 groups of 64 */
        int is = 0;
        for (int j = 0; j < QK4_K; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, blk->scales, &sc1, &m1);
            get_scale_min_k4(is + 1, blk->scales, &sc2, &m2);
            float d1 = d * sc1, min1 = dmin * m1;
            float d2 = d * sc2, min2 = dmin * m2;

            /* First 32 values: lower 4 bits */
            for (int l = 0; l < 32; l++) {
                sum += (d1 * (q[l] & 0xF) - min1) * v[base + j + l];
            }
            /* Next 32 values: upper 4 bits */
            for (int l = 0; l < 32; l++) {
                sum += (d2 * (q[l] >> 4) - min2) * v[base + j + 32 + l];
            }
            q += 32;
            is += 2;
        }
    }
    dst[out_idx] = sum;
}

/* Q6_K matmul - row-major layout: M[out_dim, in_dim]
 * 256 values per block, 6-bit quantization
 * Q6_K layout from llama.cpp dequantize_row_q6_K:
 * - Two 128-value chunks per block
 * - Each l from 0..31 produces 4 values at positions l, l+32, l+64, l+96
 */
__global__ void matmul_q6_k_kernel(float* dst, const block_q6_k* M, const float* v,
                                    int out_dim, int in_dim) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_dim) return;

    int blocks_per_row = in_dim / QK6_K;
    const block_q6_k* row = M + out_idx * blocks_per_row;

    float sum = 0.0f;
    for (int blk_idx = 0; blk_idx < blocks_per_row; blk_idx++) {
        const block_q6_k* blk = &row[blk_idx];
        float d = __half2float(blk->d);
        int base = blk_idx * QK6_K;

        /* Process two 128-value chunks per block */
        for (int chunk = 0; chunk < 2; chunk++) {
            const uint8_t *ql = blk->ql + chunk * 64;
            const uint8_t *qh = blk->qh + chunk * 32;
            const int8_t *sc = blk->scales + chunk * 8;
            int chunk_base = base + chunk * 128;

            /* Each l produces 4 output values */
            for (int l = 0; l < 32; l++) {
                int is = l / 16;

                /* q1: position l */
                int8_t q1 = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                /* q2: position l+32 */
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                /* q3: position l+64 */
                int8_t q3 = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                /* q4: position l+96 */
                int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                sum += d * sc[is + 0] * q1 * v[chunk_base + l];
                sum += d * sc[is + 2] * q2 * v[chunk_base + l + 32];
                sum += d * sc[is + 4] * q3 * v[chunk_base + l + 64];
                sum += d * sc[is + 6] * q4 * v[chunk_base + l + 96];
            }
        }
    }
    dst[out_idx] = sum;
}

extern "C" void cuda_matmul_f32(float* dst, const float* M, const float* v,
                                 int out_dim, int in_dim) {
    /* One block per output row for shared memory tiling */
    matmul_f32_kernel<<<out_dim, BLOCK_SIZE>>>(dst, M, v, out_dim, in_dim);
}

extern "C" void cuda_matmul_f16(float* dst, const void* M, const float* v,
                                 int out_dim, int in_dim) {
    matmul_f16_kernel<<<out_dim, BLOCK_SIZE>>>(dst, (const __half*)M, v, out_dim, in_dim);
}

/* Warp-per-row Q8_0 matmul - each warp computes one output row
 * Multiple warps per block, good for large outputs with moderate in_dim
 * Uses shared memory to cache input vector
 */
#define WARPS_PER_BLOCK 16  /* 16 warps = 512 threads */

__global__ void matmul_q8_0_warp_per_row_kernel(float* dst, const block_q8_0* M, const float* v,
                                                  int out_dim, int in_dim) {
    /* Shared memory for input vector */
    extern __shared__ float sv[];

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int out_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    /* Cooperatively load input vector into shared memory */
    for (int i = tid; i < in_dim; i += blockDim.x) {
        sv[i] = v[i];
    }
    __syncthreads();

    if (out_idx >= out_dim) return;

    int blocks_per_row = in_dim / QK8_0;
    const block_q8_0* row = M + out_idx * blocks_per_row;

    /* Each lane in warp processes different Q8_0 blocks */
    float sum = 0.0f;

    for (int b = lane; b < blocks_per_row; b += WARP_SIZE) {
        float scale = __half2float(row[b].d);
        int base = b * QK8_0;

        float block_sum = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < QK8_0; k += 4) {
            block_sum += row[b].qs[k]   * sv[base + k];
            block_sum += row[b].qs[k+1] * sv[base + k + 1];
            block_sum += row[b].qs[k+2] * sv[base + k + 2];
            block_sum += row[b].qs[k+3] * sv[base + k + 3];
        }
        sum += scale * block_sum;
    }

    /* Warp reduction */
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    /* Lane 0 of each warp writes result */
    if (lane == 0) {
        dst[out_idx] = sum;
    }
}

/* Warp-per-row Q4_K matmul - each warp computes one output row
 * Q4_K has 256 values per block, complex sub-block scales
 * This version parallelizes WITHIN blocks when blocks_per_row is small
 */
__global__ void matmul_q4_k_warp_per_row_kernel(float* dst, const block_q4_k* M, const float* v,
                                                  int out_dim, int in_dim) {
    extern __shared__ float sv[];

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int out_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    /* Cooperatively load input vector into shared memory */
    for (int i = tid; i < in_dim; i += blockDim.x) {
        sv[i] = v[i];
    }
    __syncthreads();

    if (out_idx >= out_dim) return;

    int blocks_per_row = in_dim / QK4_K;
    const block_q4_k* row = M + out_idx * blocks_per_row;

    float sum = 0.0f;

    /* Each lane processes ALL blocks but different packed bytes within each block
     * 128 packed bytes = 256 values, so each lane handles 4 packed bytes (8 values)
     */
    for (int b = 0; b < blocks_per_row; b++) {
        const block_q4_k* blk = &row[b];
        float d = __half2float(blk->d);
        float dmin = __half2float(blk->dmin);
        int base = b * QK4_K;

        /* Decode scales (packed 6-bit format) */
        uint8_t sc[8], m_[8];
        for (int i = 0; i < 4; i++) {
            sc[i] = blk->scales[i] & 0x3F;
            m_[i] = blk->scales[i] >> 6;
        }
        for (int i = 4; i < 8; i++) {
            sc[i] = blk->scales[i] & 0x3F;
            m_[i] = blk->scales[i] >> 6;
        }

        float block_sum = 0.0f;
        /* Each lane processes different packed bytes (j) within the block */
        for (int j = lane; j < QK4_K / 2; j += WARP_SIZE) {
            uint8_t qbyte = blk->qs[j];
            int8_t q0 = qbyte & 0xF;
            int8_t q1 = qbyte >> 4;

            int sub0 = (j * 2) / 32;
            int sub1 = (j * 2 + 1) / 32;

            float scale0 = d * sc[sub0];
            float min0 = dmin * m_[sub0];
            float scale1 = d * sc[sub1];
            float min1 = dmin * m_[sub1];

            block_sum += (scale0 * q0 - min0) * sv[base + j * 2];
            block_sum += (scale1 * q1 - min1) * sv[base + j * 2 + 1];
        }
        sum += block_sum;
    }

    /* Warp reduction */
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        dst[out_idx] = sum;
    }
}

/* Warp-per-row Q6_K matmul - each warp computes one output row
 * Q6_K has 256 values per block, 6-bit quantization
 * This version parallelizes WITHIN blocks when blocks_per_row is small
 */
__global__ void matmul_q6_k_warp_per_row_kernel(float* dst, const block_q6_k* M, const float* v,
                                                  int out_dim, int in_dim) {
    extern __shared__ float sv[];

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int out_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    /* Cooperatively load input vector into shared memory */
    for (int i = tid; i < in_dim; i += blockDim.x) {
        sv[i] = v[i];
    }
    __syncthreads();

    if (out_idx >= out_dim) return;

    int blocks_per_row = in_dim / QK6_K;
    const block_q6_k* row = M + out_idx * blocks_per_row;

    float sum = 0.0f;

    /* Each lane processes ALL blocks but different elements within each block
     * Lane i processes elements i, i+32, i+64, ... within each block
     * This ensures all 32 lanes are active even with few blocks
     */
    for (int b = 0; b < blocks_per_row; b++) {
        const block_q6_k* blk = &row[b];
        float d = __half2float(blk->d);
        int base = b * QK6_K;

        float block_sum = 0.0f;

        /* Each lane handles different elements within the 256-value block */
        for (int j = lane; j < QK6_K; j += WARP_SIZE) {
            int sub_block = j / 16;
            int8_t scale = blk->scales[sub_block];

            /* Decode 6-bit value: 4 bits from ql + 2 bits from qh */
            uint8_t ql = blk->ql[j / 2];
            uint8_t qh = blk->qh[j / 4];

            int ql_val = (j & 1) ? (ql >> 4) : (ql & 0xF);
            int qh_shift = (j % 4) * 2;
            int qh_val = (qh >> qh_shift) & 0x3;

            int q = ql_val | (qh_val << 4);
            q -= 32;

            block_sum += d * scale * q * sv[base + j];
        }
        sum += block_sum;
    }

    /* Warp reduction */
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        dst[out_idx] = sum;
    }
}

extern "C" void cuda_matmul_q8_0(float* dst, const void* M, const float* v,
                                  int out_dim, int in_dim) {
    /* Use warp-per-row kernel for large outputs (like vocab projection)
     * This reduces kernel launches from out_dim to out_dim/8
     */
    if (out_dim > 4096) {
        int num_blocks = (out_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        int threads = WARPS_PER_BLOCK * WARP_SIZE;  /* 256 threads */
        size_t smem_size = in_dim * sizeof(float);
        matmul_q8_0_warp_per_row_kernel<<<num_blocks, threads, smem_size>>>(
            dst, (const block_q8_0*)M, v, out_dim, in_dim);
    } else {
        /* Use simpler kernel for smaller outputs */
        matmul_q8_0_simple_kernel<<<out_dim, BLOCK_SIZE>>>(dst, (const block_q8_0*)M, v, out_dim, in_dim);
    }
}

extern "C" void cuda_matmul_q4_0(float* dst, const void* M, const float* v,
                                  int out_dim, int in_dim) {
    matmul_q4_0_kernel<<<out_dim, BLOCK_SIZE>>>(dst, (const block_q4_0*)M, v, out_dim, in_dim);
}

extern "C" void cuda_matmul_q4_k(float* dst, const void* M, const float* v,
                                  int out_dim, int in_dim) {
    /* Use warp-per-row kernel for large outputs (like vocab projection) */
    if (out_dim > 4096) {
        int num_blocks = (out_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        int threads = WARPS_PER_BLOCK * WARP_SIZE;
        size_t smem_size = in_dim * sizeof(float);
        matmul_q4_k_warp_per_row_kernel<<<num_blocks, threads, smem_size>>>(
            dst, (const block_q4_k*)M, v, out_dim, in_dim);
    } else {
        int blocks = (out_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
        matmul_q4_k_kernel<<<blocks, BLOCK_SIZE>>>(dst, (const block_q4_k*)M, v, out_dim, in_dim);
    }
}

extern "C" void cuda_matmul_q6_k(float* dst, const void* M, const float* v,
                                  int out_dim, int in_dim) {
    /* Always use simple kernel for now - warp kernel has NaN issues */
    int blocks = (out_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    matmul_q6_k_kernel<<<blocks, BLOCK_SIZE>>>(dst, (const block_q6_k*)M, v, out_dim, in_dim);
}

/* ============================================================================
 * Batched QKV Projection
 *
 * Instead of 3 separate matmuls for Q, K, V projections, we compute them
 * together. This improves GPU utilization and reduces kernel launch overhead.
 *
 * For models with dim=4096, this launches 4096+1024+1024 = 6144 blocks once
 * instead of 3 kernel launches with 4096, 1024, 1024 blocks each.
 * ============================================================================ */

/* Batched QKV matmul for Q8_0 quantization
 * Computes Q = Wq @ x, K = Wk @ x, V = Wv @ x in one kernel
 * Each block handles one output row from one of Q, K, or V
 */
__global__ void batched_qkv_q8_0_kernel(
    float* q_out, float* k_out, float* v_out,
    const block_q8_0* Wq, const block_q8_0* Wk, const block_q8_0* Wv,
    const float* x,
    int q_dim, int kv_dim, int in_dim
) {
    /* Determine which output (Q, K, or V) this block handles */
    int total_q = q_dim;
    int total_k = kv_dim;

    int block_id = blockIdx.x;
    float* dst;
    const block_q8_0* weights;
    int out_idx;

    if (block_id < total_q) {
        /* This block computes one Q output */
        dst = q_out;
        weights = Wq;
        out_idx = block_id;
    } else if (block_id < total_q + total_k) {
        /* This block computes one K output */
        dst = k_out;
        weights = Wk;
        out_idx = block_id - total_q;
    } else {
        /* This block computes one V output */
        dst = v_out;
        weights = Wv;
        out_idx = block_id - total_q - total_k;
    }

    int tid = threadIdx.x;
    int blocks_per_row = in_dim / QK8_0;
    const block_q8_0* row = weights + out_idx * blocks_per_row;

    float sum = 0.0f;

    /* Process Q8_0 blocks with vectorized loads */
    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        float scale = __half2float(row[b].d);
        int base = b * QK8_0;

        float block_sum = 0.0f;
        const float4* v4 = reinterpret_cast<const float4*>(x + base);

        #pragma unroll 8
        for (int j = 0; j < 8; j++) {
            float4 vv = v4[j];
            int k = j * 4;
            block_sum += row[b].qs[k] * vv.x;
            block_sum += row[b].qs[k+1] * vv.y;
            block_sum += row[b].qs[k+2] * vv.z;
            block_sum += row[b].qs[k+3] * vv.w;
        }
        sum += scale * block_sum;
    }

    /* Warp reduction */
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float warp_sums[8];
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (lane == 0 && warp_id < 8) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    if (tid < 32) {
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        sum = (tid < num_warps) ? warp_sums[tid] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            dst[out_idx] = sum;
        }
    }
}

extern "C" void cuda_batched_qkv_q8_0(
    float* q_out, float* k_out, float* v_out,
    const void* Wq, const void* Wk, const void* Wv,
    const float* x,
    int q_dim, int kv_dim, int in_dim
) {
    int total_blocks = q_dim + kv_dim + kv_dim;  /* Q + K + V rows */
    batched_qkv_q8_0_kernel<<<total_blocks, BLOCK_SIZE>>>(
        q_out, k_out, v_out,
        (const block_q8_0*)Wq, (const block_q8_0*)Wk, (const block_q8_0*)Wv,
        x, q_dim, kv_dim, in_dim
    );
}

/* ============================================================================
 * Batched FFN Gate+Up Projection with Fused SiLU
 *
 * Computes: gate = W1 @ x, up = W3 @ x, out = SiLU(gate) * up
 * All in one kernel launch, combining what would be 3 kernels into 1
 * ============================================================================ */

__global__ void batched_ffn_gate_up_q8_0_kernel(
    float* out,           /* Output: SiLU(gate) * up [ffn_dim] */
    const block_q8_0* W1, /* Gate weights [ffn_dim, dim] */
    const block_q8_0* W3, /* Up weights [ffn_dim, dim] */
    const float* x,       /* Input [dim] */
    int ffn_dim, int in_dim
) {
    int out_idx = blockIdx.x;
    if (out_idx >= ffn_dim) return;

    int tid = threadIdx.x;
    int blocks_per_row = in_dim / QK8_0;

    /* Compute gate and up projections in parallel within this block */
    const block_q8_0* gate_row = W1 + out_idx * blocks_per_row;
    const block_q8_0* up_row = W3 + out_idx * blocks_per_row;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        float gate_scale = __half2float(gate_row[b].d);
        float up_scale = __half2float(up_row[b].d);
        int base = b * QK8_0;

        float gate_block = 0.0f;
        float up_block = 0.0f;

        const float4* v4 = reinterpret_cast<const float4*>(x + base);

        #pragma unroll 8
        for (int j = 0; j < 8; j++) {
            float4 vv = v4[j];
            int k = j * 4;
            gate_block += gate_row[b].qs[k] * vv.x;
            gate_block += gate_row[b].qs[k+1] * vv.y;
            gate_block += gate_row[b].qs[k+2] * vv.z;
            gate_block += gate_row[b].qs[k+3] * vv.w;

            up_block += up_row[b].qs[k] * vv.x;
            up_block += up_row[b].qs[k+1] * vv.y;
            up_block += up_row[b].qs[k+2] * vv.z;
            up_block += up_row[b].qs[k+3] * vv.w;
        }
        gate_sum += gate_scale * gate_block;
        up_sum += up_scale * up_block;
    }

    /* Warp reduction for both sums */
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        gate_sum += __shfl_down_sync(0xffffffff, gate_sum, offset);
        up_sum += __shfl_down_sync(0xffffffff, up_sum, offset);
    }

    __shared__ float gate_warp_sums[8];
    __shared__ float up_warp_sums[8];
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (lane == 0 && warp_id < 8) {
        gate_warp_sums[warp_id] = gate_sum;
        up_warp_sums[warp_id] = up_sum;
    }
    __syncthreads();

    if (tid < 32) {
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        gate_sum = (tid < num_warps) ? gate_warp_sums[tid] : 0.0f;
        up_sum = (tid < num_warps) ? up_warp_sums[tid] : 0.0f;

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            gate_sum += __shfl_down_sync(0xffffffff, gate_sum, offset);
            up_sum += __shfl_down_sync(0xffffffff, up_sum, offset);
        }

        if (tid == 0) {
            /* Fused SiLU and multiply */
            float silu_gate = gate_sum / (1.0f + expf(-gate_sum));
            out[out_idx] = silu_gate * up_sum;
        }
    }
}

extern "C" void cuda_batched_ffn_gate_up_q8_0(
    float* out,
    const void* W1, const void* W3,
    const float* x,
    int ffn_dim, int in_dim
) {
    batched_ffn_gate_up_q8_0_kernel<<<ffn_dim, BLOCK_SIZE>>>(
        out, (const block_q8_0*)W1, (const block_q8_0*)W3, x, ffn_dim, in_dim
    );
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

/* Fused SiLU + elementwise multiply: dst = SiLU(a) * b
 * Saves one kernel launch and one memory round-trip
 */
__global__ void silu_mul_kernel(float* dst, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = a[i];
        float silu_v = v / (1.0f + expf(-v));
        dst[i] = silu_v * b[i];
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

extern "C" void cuda_silu_mul(float* dst, const float* a, const float* b, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    silu_mul_kernel<<<blocks, BLOCK_SIZE>>>(dst, a, b, n);
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

/* Fused RMS norm - single kernel with simple tree reduction */
__global__ void rms_norm_fused_kernel(float* dst, const float* x, const float* weight,
                                       int n, float eps) {
    __shared__ float s_reduce[256];

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    /* Phase 1: Compute sum of squares */
    float my_sum_sq = 0.0f;
    for (int i = tid; i < n; i += block_size) {
        float val = x[i];
        my_sum_sq += val * val;
    }
    s_reduce[tid] = my_sum_sq;
    __syncthreads();

    /* Tree reduction */
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_reduce[tid] += s_reduce[tid + s];
        }
        __syncthreads();
    }

    /* Compute scale */
    __shared__ float scale;
    if (tid == 0) {
        float sum_sq = s_reduce[0];
        float rms = sqrtf(sum_sq / n + eps);
        scale = 1.0f / rms;
    }
    __syncthreads();

    /* Phase 2: Apply normalization */
    for (int i = tid; i < n; i += block_size) {
        dst[i] = x[i] * scale * weight[i];
    }
}

extern "C" void cuda_rms_norm(float* dst, const float* x, const float* weight, int n, float eps) {
    /* Single fused kernel - much faster than two-pass approach */
    rms_norm_fused_kernel<<<1, 256>>>(dst, x, weight, n, eps);
}

extern "C" void cuda_layer_norm(float* dst, const float* x, const float* weight,
                                 const float* bias, int n, float eps) {
    /* TODO: Implement full layer norm with mean subtraction */
    /* For now, use RMS norm as approximation */
    cuda_rms_norm(dst, x, weight, n, eps);
}

/* ============================================================================
 * Softmax Kernel - Fused single-pass implementation
 * ============================================================================ */

/* Fused softmax kernel - one kernel does everything
 * Uses online softmax algorithm to compute in a single pass
 */
__global__ void softmax_fused_kernel(float* dst, const float* x, int n) {
    __shared__ float warp_max[8];
    __shared__ float warp_sum[8];

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    /* Phase 1: Find max */
    float my_max = -1e30f;
    for (int i = tid; i < n; i += blockDim.x) {
        my_max = fmaxf(my_max, x[i]);
    }

    /* Warp reduction for max */
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        my_max = fmaxf(my_max, __shfl_down_sync(0xffffffff, my_max, offset));
    }
    if (lane == 0 && warp_id < 8) warp_max[warp_id] = my_max;
    __syncthreads();

    if (tid < num_warps) my_max = warp_max[tid];
    else if (tid < WARP_SIZE) my_max = -1e30f;
    if (tid < WARP_SIZE) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            my_max = fmaxf(my_max, __shfl_down_sync(0xffffffff, my_max, offset));
        }
    }

    __shared__ float max_val;
    if (tid == 0) max_val = my_max;
    __syncthreads();

    /* Phase 2: Compute exp and sum */
    float my_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float e = expf(x[i] - max_val);
        dst[i] = e;
        my_sum += e;
    }

    /* Warp reduction for sum */
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        my_sum += __shfl_down_sync(0xffffffff, my_sum, offset);
    }
    if (lane == 0 && warp_id < 8) warp_sum[warp_id] = my_sum;
    __syncthreads();

    if (tid < num_warps) my_sum = warp_sum[tid];
    else if (tid < WARP_SIZE) my_sum = 0.0f;
    if (tid < WARP_SIZE) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            my_sum += __shfl_down_sync(0xffffffff, my_sum, offset);
        }
    }

    __shared__ float inv_sum;
    if (tid == 0) inv_sum = 1.0f / my_sum;
    __syncthreads();

    /* Phase 3: Normalize */
    for (int i = tid; i < n; i += blockDim.x) {
        dst[i] *= inv_sum;
    }
}

extern "C" void cuda_softmax(float* dst, const float* x, int n) {
    /* Single fused kernel - avoids multiple kernel launches and memory copies */
    softmax_fused_kernel<<<1, 256>>>(dst, x, n);
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

/* ============================================================================
 * Flash Attention Implementation
 *
 * Key insight: Instead of materializing the full attention matrix (O(n^2) memory),
 * we compute attention in tiles, maintaining running softmax statistics.
 *
 * For each tile of keys:
 *   1. Compute QK^T for that tile
 *   2. Update running max and sum for numerical stability
 *   3. Rescale previous output and accumulate new contribution
 *
 * Memory: O(tile_size) instead of O(seq_len)
 * Compute: Same O(n^2 * d) but with better cache locality
 * ============================================================================ */

#define FLASH_TILE_SIZE 64  /* Keys per tile - tuned for shared memory */

/* Flash Attention kernel for single-query autoregressive generation
 * Processes keys/values in tiles to minimize memory and maximize cache reuse
 *
 * Each thread processes multiple output dimensions (head_dim / blockDim.x)
 * We store per-thread accumulators and running softmax stats
 */
__global__ void flash_attention_kernel(float* dst, const float* q,
                                        const float* k_cache, const float* v_cache,
                                        int kv_len, int head_dim,
                                        int kv_head, int kv_dim) {
    /* Shared memory layout:
     * - Key tile: [FLASH_TILE_SIZE, head_dim]
     * - Value tile: [FLASH_TILE_SIZE, head_dim]
     * - Scores: [FLASH_TILE_SIZE]
     * - Max scratch: [8] for cross-warp reduction
     * - Sum scratch: [8] for cross-warp reduction
     */
    extern __shared__ float smem[];
    float* s_keys = smem;
    float* s_vals = smem + FLASH_TILE_SIZE * head_dim;
    float* s_scores = smem + 2 * FLASH_TILE_SIZE * head_dim;
    float* s_max = smem + 2 * FLASH_TILE_SIZE * head_dim + FLASH_TILE_SIZE;
    float* s_sum = s_max + 8;

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    float scale = rsqrtf((float)head_dim);

    /* Per-thread running statistics and output accumulator */
    float m_running = -1e30f;
    float l_running = 0.0f;

    /* Per-thread output: each thread handles dims [tid, tid+block_size, tid+2*block_size, ...] */
    float out_running[8] = {0};  /* Support head_dim up to 8 * block_size */

    /* Process KV cache in tiles */
    for (int tile_start = 0; tile_start < kv_len; tile_start += FLASH_TILE_SIZE) {
        int tile_end = min(tile_start + FLASH_TILE_SIZE, kv_len);
        int tile_len = tile_end - tile_start;

        /* Cooperatively load key/value tile into shared memory */
        for (int idx = tid; idx < tile_len * head_dim; idx += block_size) {
            int pos_in_tile = idx / head_dim;
            int dim_idx = idx % head_dim;
            int global_pos = tile_start + pos_in_tile;

            const float* k_p = k_cache + global_pos * kv_dim + kv_head * head_dim;
            const float* v_p = v_cache + global_pos * kv_dim + kv_head * head_dim;

            s_keys[pos_in_tile * head_dim + dim_idx] = k_p[dim_idx];
            s_vals[pos_in_tile * head_dim + dim_idx] = v_p[dim_idx];
        }
        __syncthreads();

        /* Compute attention scores: score[i] = Q . K[i] */
        for (int i = tid; i < tile_len; i += block_size) {
            float score = 0.0f;
            #pragma unroll 4
            for (int d = 0; d < head_dim; d++) {
                score += q[d] * s_keys[i * head_dim + d];
            }
            s_scores[i] = score * scale;
        }
        __syncthreads();

        /* Find tile max with proper multi-warp reduction */
        float m_tile = -1e30f;
        for (int i = tid; i < tile_len; i += block_size) {
            m_tile = fmaxf(m_tile, s_scores[i]);
        }
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            m_tile = fmaxf(m_tile, __shfl_down_sync(0xffffffff, m_tile, offset));
        }
        if (lane == 0 && warp_id < 8) s_max[warp_id] = m_tile;
        __syncthreads();
        if (tid < num_warps) m_tile = s_max[tid];
        else if (tid < WARP_SIZE) m_tile = -1e30f;
        if (tid < WARP_SIZE) {
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                m_tile = fmaxf(m_tile, __shfl_down_sync(0xffffffff, m_tile, offset));
            }
        }
        if (tid == 0) s_max[0] = m_tile;
        __syncthreads();
        m_tile = s_max[0];

        /* Compute new running max */
        float m_new = fmaxf(m_running, m_tile);

        /* Compute exp(score - m_new) and tile sum */
        float l_tile = 0.0f;
        for (int i = tid; i < tile_len; i += block_size) {
            s_scores[i] = expf(s_scores[i] - m_new);
            l_tile += s_scores[i];
        }
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            l_tile += __shfl_down_sync(0xffffffff, l_tile, offset);
        }
        if (lane == 0 && warp_id < 8) s_sum[warp_id] = l_tile;
        __syncthreads();
        if (tid < num_warps) l_tile = s_sum[tid];
        else if (tid < WARP_SIZE) l_tile = 0.0f;
        if (tid < WARP_SIZE) {
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                l_tile += __shfl_down_sync(0xffffffff, l_tile, offset);
            }
        }
        if (tid == 0) s_sum[0] = l_tile;
        __syncthreads();
        l_tile = s_sum[0];

        /* Correction factor for previously accumulated values */
        float alpha = expf(m_running - m_new) * l_running;
        float l_new = alpha + l_tile;

        /* Update output accumulator for dimensions this thread handles */
        int slot = 0;
        for (int d = tid; d < head_dim; d += block_size, slot++) {
            /* Rescale previous accumulation and add new contribution */
            float new_val = out_running[slot] * alpha;
            for (int i = 0; i < tile_len; i++) {
                new_val += s_scores[i] * s_vals[i * head_dim + d];
            }
            out_running[slot] = new_val / l_new;
        }

        m_running = m_new;
        l_running = l_new;
        __syncthreads();
    }

    /* Write final output */
    int slot = 0;
    for (int d = tid; d < head_dim; d += block_size, slot++) {
        dst[d] = out_running[slot];
    }
}

/* ============================================================================
 * Multi-Head Attention - All heads in parallel
 *
 * This is the KEY optimization - instead of launching n_heads separate kernels,
 * we process all heads in one kernel launch with one block per head.
 * ============================================================================ */

/* Optimized multi-head attention kernel
 * Key optimizations:
 * 1. Vectorized Q·K computation
 * 2. Online softmax (fused max/exp/sum in one pass where possible)
 * 3. Better memory access patterns for values
 */
__global__ void multi_head_attention_kernel(
    float* dst,           /* Output: [n_heads * head_dim] */
    const float* q,       /* Query: [n_heads * head_dim] */
    const float* k_cache, /* Key cache: [max_seq * n_kv_heads * head_dim] */
    const float* v_cache, /* Value cache: [max_seq * n_kv_heads * head_dim] */
    int kv_len,           /* Number of cached positions (pos + 1) */
    int head_dim,
    int n_heads,
    int n_kv_heads,
    int kv_dim            /* n_kv_heads * head_dim */
) {
    /* One block per query head */
    int head = blockIdx.x;
    if (head >= n_heads) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int kv_head = head / (n_heads / n_kv_heads);

    float scale = rsqrtf((float)head_dim);

    /* Shared memory layout:
     * [0, kv_len): attention scores
     * [kv_len, kv_len+block_size): reduction buffer
     * Total: (kv_len + block_size) * sizeof(float)
     */
    extern __shared__ float smem[];
    float* scores = smem;
    float* s_reduce = smem + kv_len;  /* Reduction buffer after scores */

    /* Query pointer for this head */
    const float* q_h = q + head * head_dim;
    float* out_h = dst + head * head_dim;

    /* Step 1: Compute Q·K scores for all positions
     * Each thread handles multiple positions for better ILP */
    for (int p = tid; p < kv_len; p += block_size) {
        const float* k_p = k_cache + p * kv_dim + kv_head * head_dim;
        float score = 0.0f;

        /* Unrolled dot product for common head_dim values */
        int d = 0;
        #pragma unroll 4
        for (; d + 3 < head_dim; d += 4) {
            score += q_h[d] * k_p[d];
            score += q_h[d+1] * k_p[d+1];
            score += q_h[d+2] * k_p[d+2];
            score += q_h[d+3] * k_p[d+3];
        }
        for (; d < head_dim; d++) {
            score += q_h[d] * k_p[d];
        }
        scores[p] = score * scale;

        /* DEBUG: Print first score for all heads at p=0 - only when called from first token */
        if (tid == 0 && p == 0 && kv_len == 1) {
            const float* v_test = v_cache + kv_head * head_dim;
            printf("[ATTN DEBUG L0] head=%d kv_head=%d q[0..3]=%g,%g,%g,%g k[0..3]=%g,%g,%g,%g score=%g\n",
                   head, kv_head,
                   q_h[0], q_h[1], q_h[2], q_h[3],
                   k_p[0], k_p[1], k_p[2], k_p[3],
                   score * scale);
        }
    }
    __syncthreads();

    /* Step 2: Softmax - find max using simple shared memory reduction */
    float my_max = -1e30f;
    for (int p = tid; p < kv_len; p += block_size) {
        my_max = fmaxf(my_max, scores[p]);
    }
    s_reduce[tid] = my_max;
    __syncthreads();

    /* Tree reduction for max */
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < block_size) {
            s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + s]);
        }
        __syncthreads();
    }
    float max_val = s_reduce[0];
    __syncthreads();

    /* Step 3: Softmax - exp and sum */
    float my_sum = 0.0f;
    for (int p = tid; p < kv_len; p += block_size) {
        float e = expf(scores[p] - max_val);
        scores[p] = e;
        my_sum += e;
    }
    s_reduce[tid] = my_sum;
    __syncthreads();

    /* Tree reduction for sum */
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < block_size) {
            s_reduce[tid] = s_reduce[tid] + s_reduce[tid + s];
        }
        __syncthreads();
    }
    float inv_sum = 1.0f / s_reduce[0];
    __syncthreads();

    /* Step 4: Normalize scores in place */
    for (int p = tid; p < kv_len; p += block_size) {
        scores[p] *= inv_sum;
    }
    __syncthreads();

    /* Step 5: Weighted sum of values
     * Each thread computes one or more output dimensions */
    for (int d = tid; d < head_dim; d += block_size) {
        float out_d = 0.0f;
        const float* v_base = v_cache + kv_head * head_dim + d;

        /* Process 4 positions at a time for better ILP */
        int p = 0;
        #pragma unroll 4
        for (; p + 3 < kv_len; p += 4) {
            out_d += scores[p] * v_base[p * kv_dim];
            out_d += scores[p+1] * v_base[(p+1) * kv_dim];
            out_d += scores[p+2] * v_base[(p+2) * kv_dim];
            out_d += scores[p+3] * v_base[(p+3) * kv_dim];
        }
        for (; p < kv_len; p++) {
            out_d += scores[p] * v_base[p * kv_dim];
        }
        out_h[d] = out_d;
    }
}

/* Launch multi-head attention - all heads in parallel */
extern "C" void cuda_multi_head_attention(
    float* dst, const float* q,
    const float* k_cache, const float* v_cache,
    int kv_len, int head_dim, int n_heads, int n_kv_heads
) {
    int kv_dim = n_kv_heads * head_dim;
    /* Use enough threads for both score computation and value accumulation */
    /* For long sequences, we want more threads for parallel softmax */
    int threads = min(max(max(kv_len, head_dim), 128), 256);
    /* Shared memory: scores[kv_len] + reduction_buffer[threads]
     * Need space for kv_len scores plus reduction array of size threads */
    size_t smem = (kv_len + threads) * sizeof(float);
    multi_head_attention_kernel<<<n_heads, threads, smem>>>(
        dst, q, k_cache, v_cache, kv_len, head_dim, n_heads, n_kv_heads, kv_dim);
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
    /* Use Flash Attention for longer sequences (better memory efficiency)
     * Fall back to simple kernel for very short sequences (lower overhead) */
    if (kv_len > 128) {
        /* Flash Attention: tiled computation, O(1) memory per head */
        int threads = 128;  /* Good balance for shared memory usage */
        /* Shared memory: 2 * FLASH_TILE_SIZE * head_dim (K,V) + FLASH_TILE_SIZE (scores) + 16 (scratch) */
        size_t smem = (2 * FLASH_TILE_SIZE * head_dim + FLASH_TILE_SIZE + 16) * sizeof(float);
        flash_attention_kernel<<<1, threads, smem>>>(dst, q, k_cache, v_cache,
                                                      kv_len, head_dim,
                                                      kv_head, kv_dim);
    } else {
        /* Simple kernel for short sequences */
        int threads = min(max(kv_len, head_dim), 256);
        size_t smem = (kv_len + 8) * sizeof(float);
        attention_single_kernel<<<1, threads, smem>>>(dst, q, k_cache, v_cache,
                                                       att_buf, kv_len, head_dim,
                                                       kv_head, kv_dim);
    }
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
 * Layout: [vocab_size, dim] - each token's embedding is contiguous
 * ============================================================================ */

__global__ void get_embedding_f32_kernel(float* dst, const float* embeddings,
                                          int token, int dim, int vocab_size) {
    (void)vocab_size;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        /* Row-major: token's embedding starts at token * dim */
        dst[i] = embeddings[token * dim + i];
    }
}

__global__ void get_embedding_f16_kernel(float* dst, const __half* embeddings,
                                          int token, int dim, int vocab_size) {
    (void)vocab_size;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        dst[i] = __half2float(embeddings[token * dim + i]);
    }
}

__global__ void get_embedding_q8_0_kernel(float* dst, const block_q8_0* embeddings,
                                           int token, int dim, int vocab_size) {
    (void)vocab_size;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        /* Each token's embedding is dim values, quantized in blocks of 32 */
        int blocks_per_token = dim / QK8_0;
        const block_q8_0* token_emb = embeddings + token * blocks_per_token;

        int block_idx = i / QK8_0;
        int within_block = i % QK8_0;

        float scale = __half2float(token_emb[block_idx].d);
        float val = scale * (float)token_emb[block_idx].qs[within_block];
        dst[i] = val;
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

__global__ void get_embedding_q4_0_kernel(float* dst, const block_q4_0* embeddings,
                                           int token, int dim, int vocab_size) {
    (void)vocab_size;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        /* Q4_0: 32 values per block, packed as 4-bit (2 per byte in qs[16])
         * Values are signed: subtract 8 to center around 0 */
        int blocks_per_token = dim / QK4_0;
        const block_q4_0* token_emb = embeddings + token * blocks_per_token;

        int block_idx = i / QK4_0;
        int within_block = i % QK4_0;  /* 0-31 */

        const block_q4_0* blk = &token_emb[block_idx];
        float d = __half2float(blk->d);

        /* Each byte in qs contains 2 values:
         * - lower nibble: value at position j (0-15)
         * - upper nibble: value at position j+16 (16-31) */
        int byte_idx = within_block % 16;  /* Which byte (0-15) */
        int nibble;
        if (within_block < 16) {
            /* Lower 16 values: use lower nibble */
            nibble = blk->qs[byte_idx] & 0xF;
        } else {
            /* Upper 16 values: use upper nibble */
            nibble = (blk->qs[byte_idx] >> 4) & 0xF;
        }

        /* Q4_0 values are unsigned 0-15, subtract 8 to get signed -8 to +7 */
        dst[i] = d * (float)(nibble - 8);
    }
}

extern "C" void cuda_get_embedding_q4_0(float* dst, const void* embeddings,
                                         int token, int dim, int vocab_size) {
    int blocks = (dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    get_embedding_q4_0_kernel<<<blocks, BLOCK_SIZE>>>(dst, (const block_q4_0*)embeddings,
                                                        token, dim, vocab_size);
}

__global__ void get_embedding_q6_k_kernel(float* dst, const block_q6_k* embeddings,
                                           int token, int dim, int vocab_size) {
    (void)vocab_size;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        /* Each token's embedding is dim values, quantized in superblocks of 256 */
        int blocks_per_token = dim / QK6_K;
        const block_q6_k* token_emb = embeddings + token * blocks_per_token;

        int block_idx = i / QK6_K;
        int j = i % QK6_K;  /* Position within the 256-value superblock (0-255) */

        const block_q6_k* blk = &token_emb[block_idx];
        float d = __half2float(blk->d);

        /* Q6_K layout (from llama.cpp dequantize_row_q6_K):
         * - Two 128-value chunks per block
         * - For each chunk: ql[0..63], qh[0..31], sc[0..7]
         * - Each l from 0..31 produces 4 values at positions l, l+32, l+64, l+96
         *   q1: (ql[l] & 0xF) | ((qh[l] >> 0 & 3) << 4)  -> position l
         *   q2: (ql[l+32] & 0xF) | ((qh[l] >> 2 & 3) << 4) -> position l+32
         *   q3: (ql[l] >> 4) | ((qh[l] >> 4 & 3) << 4)   -> position l+64
         *   q4: (ql[l+32] >> 4) | ((qh[l] >> 6 & 3) << 4) -> position l+96
         */
        int chunk = j / 128;           /* Which 128-value chunk: 0 or 1 */
        int pos_in_chunk = j % 128;    /* Position within chunk: 0-127 */
        int l = pos_in_chunk % 32;     /* Base position: 0-31 */
        int which = pos_in_chunk / 32; /* Which of 4 values: 0,1,2,3 */

        /* Pointers offset by chunk */
        const uint8_t *ql = blk->ql + chunk * 64;
        const uint8_t *qh = blk->qh + chunk * 32;
        const int8_t *sc = blk->scales + chunk * 8;

        /* Scale index: values 0-15 use sc[0], 16-31 use sc[1], etc. */
        int is = l / 16 + which * 2;

        int8_t q_val;
        switch (which) {
            case 0:  /* Position l: lower 4 bits of ql[l], bits 0-1 of qh[l] */
                q_val = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                break;
            case 1:  /* Position l+32: lower 4 bits of ql[l+32], bits 2-3 of qh[l] */
                q_val = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                break;
            case 2:  /* Position l+64: upper 4 bits of ql[l], bits 4-5 of qh[l] */
                q_val = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                break;
            default: /* Position l+96: upper 4 bits of ql[l+32], bits 6-7 of qh[l] */
                q_val = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                break;
        }

        dst[i] = d * sc[is] * q_val;
    }
}

extern "C" void cuda_get_embedding_q6_k(float* dst, const void* embeddings,
                                         int token, int dim, int vocab_size) {
    int blocks = (dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    get_embedding_q6_k_kernel<<<blocks, BLOCK_SIZE>>>(dst, (const block_q6_k*)embeddings,
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

    max_vals[tid] = (i < n) ? x[i] : -1e30f;
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
