// #define __CUDA_ARCH__ 860

#include<iostream>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

using namespace nvcuda;

template<typename T>
void template_init_matrix(T *A, int size) {
    for (int i = 0; i < size; i++) {
        A[i] = 1.0;
    }
}

void initMatrix(float *A, int size) {
    for (int i = 0; i < size; i++) {
        A[i] = 1.0;
    }
}

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void matmul_tensor_core(const half *A, const half *B, float *C, int M, int N, int K, float alpha, float beta) {

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fragment_matrix_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fragment_matrix_b;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fragment_accumulator;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fragment_result;
    
    wmma::fill_fragment(fragment_accumulator, 0.0f);
    
    // __shared__ half A_tile[];

    int warpM = (blockDim.x * blockIdx.x + threadIdx.x) / warpSize;
    int warpN = (blockDim.y * blockIdx.y + threadIdx.y);

    for (int t = 0; t < K; t += WMMA_K) {

        int aRow = warpM * WMMA_M;
        int aCol = t;

        int bRow = t;
        int bCol = warpN * WMMA_N;

        wmma::load_matrix_sync(fragment_matrix_a, &A[aRow * M + aCol], WMMA_N);
        wmma::load_matrix_sync(fragment_matrix_b, &B[bRow * K + bCol], WMMA_K);

        wmma::mma_sync(fragment_accumulator, fragment_matrix_a, fragment_matrix_b, fragment_accumulator);

    }
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < M && cCol < N) {
        wmma::load_matrix_sync(fragment_result, &C[cRow * M + cCol], M, wmma::mem_row_major);

        for (int i = 0; i < fragment_result.num_elements; i++) {
            fragment_result.x[i] = alpha * fragment_accumulator.x[i] + beta * fragment_result.x[i];
        }

        wmma::store_matrix_sync(&C[cRow * M + cCol], fragment_result,  M, wmma::mem_row_major);
    }
}

constexpr int M_tile = 128;
constexpr int N_tile = 256;
constexpr int K_tile = 32;

constexpr int M_wrap_tile = M_tile / 2;
constexpr int N_wrap_tile = N_tile / 4;

constexpr int tensor_core_fragment_size = 16;
constexpr int threads_per_block = 32 * 4 * 2; 


__global__ void matmul_tensor_core_v2(half *A, half *B, float *C, int M, int N, int K, float alpha, float beta) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tid = threadIdx.x;
    int wid = tid >> 5;     // 32 thread per block; wrap id = thread id / 32, 0 \leq wid \leq 8

    constexpr int PAD = 8;

    __shared__ half shared_A[M_tile][K_tile + PAD];
    __shared__ half shared_B[K_tile][N_tile + PAD];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fragment_matrix_A[M_wrap_tile / tensor_core_fragment_size][K_tile / tensor_core_fragment_size];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fragment_matrix_B[K_tile / tensor_core_fragment_size][M_wrap_tile / tensor_core_fragment_size];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fragment_accumulator[M_wrap_tile / tensor_core_fragment_size][N_wrap_tile / tensor_core_fragment_size];

    for (int i = 0; i < M_wrap_tile / tensor_core_fragment_size; i++) 
        for (int j = 0; j < N_wrap_tile / tensor_core_fragment_size; j++)
            wmma::fill_fragment(fragment_accumulator[i][j], 0.0);

    // a thread read 2 * 8 from A to shared, 
    int load_A_shared_m = (tid >> 2) << 1;
    int load_A_shared_k = (tid & 3) << 3; 
    
    // a thread read 4 * 8 from B to shared, 256 / 8 = 32
    int load_B_shared_k = (tid >> 5) << 2;
    int load_B_shared_n = (tid & 31) << 3;

    int load_A_global_m = by * M_tile + load_A_shared_m;
    int load_B_global_n = bx * N_tile + load_B_shared_n;

    int load_a_global_addr = load_A_global_m * K + load_A_shared_k;
    int load_b_global_addr = load_B_shared_k * N + load_B_global_n;

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    for (int k = 0; k < K; k += K_tile) {
        FLOAT4(shared_A[load_A_shared_m    ][load_A_shared_k]) = FLOAT4(A[load_a_global_addr        ]);
        FLOAT4(shared_A[load_A_shared_m + 1][load_A_shared_k]) = FLOAT4(A[load_a_global_addr +     K]);
        FLOAT4(shared_B[load_B_shared_k    ][load_B_shared_n]) = FLOAT4(B[load_b_global_addr        ]);
        FLOAT4(shared_B[load_B_shared_k + 1][load_B_shared_n]) = FLOAT4(B[load_b_global_addr +     N]);
        FLOAT4(shared_B[load_B_shared_k + 2][load_B_shared_n]) = FLOAT4(B[load_b_global_addr + 2 * N]);
        FLOAT4(shared_B[load_B_shared_k + 3][load_B_shared_n]) = FLOAT4(B[load_b_global_addr + 3 * N]);

        load_a_global_addr += K_tile;
        load_b_global_addr += K_tile * N;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < M_wrap_tile / tensor_core_fragment_size; i++) {
            #pragma unroll
            for (int kk = 0; kk < K_tile / tensor_core_fragment_size; kk++) {
                wmma::load_matrix_sync(fragment_matrix_A[i][kk], &shared_A[i * tensor_core_fragment_size + comp_c_frag_m * M_wrap_tile][kk * tensor_core_fragment_size], K_tile + PAD);
            }
        }

        #pragma unroll
        for (int kk = 0; kk < K_tile / tensor_core_fragment_size; kk++) {
            #pragma unroll
            for (int j = 0; j < N_wrap_tile / tensor_core_fragment_size; j++) {
                wmma::load_matrix_sync(fragment_matrix_B[kk][j], &shared_B[kk * tensor_core_fragment_size][comp_c_frag_n * N_wrap_tile + j * tensor_core_fragment_size], N_tile + PAD);
            }
        }

        #pragma unroll
        for (int i = 0; i < M_wrap_tile / tensor_core_fragment_size; i++) {
            #pragma unroll
            for (int j = 0; j < N_wrap_tile / tensor_core_fragment_size; j++) {
                #pragma unroll
                for (int kk = 0; kk < K_tile / tensor_core_fragment_size; kk++) {
                    wmma::mma_sync(fragment_accumulator[i][j], fragment_matrix_A[i][kk], fragment_matrix_B[kk][j], fragment_accumulator[i][j]);
                }
            }
        }

        __syncthreads();
    }

    int store_c_global_m = M_tile * by + comp_c_frag_m * M_wrap_tile;
    int store_c_global_n = N_tile * bx + comp_c_frag_n * N_wrap_tile;
    int store_c_global_addr = store_c_global_m * N + store_c_global_n;

    #pragma unroll
    for (int i = 0; i < M_wrap_tile / tensor_core_fragment_size; i++)
        #pragma unroll
        for (int j = 0; j < N_wrap_tile / tensor_core_fragment_size; j++)
            wmma::store_matrix_sync(&C[store_c_global_addr + N * i * tensor_core_fragment_size + j * tensor_core_fragment_size], fragment_accumulator[i][j], N, wmma::mem_row_major);
}

int main() {
    long long M = 8192;
    long long K = 8192;
    long long N = 8192;

    half *A, *B;
    A = (half *) malloc(M * K * sizeof(half));
    B = (half *) malloc(K * N * sizeof(half));

    float *C;
    C = (float *) malloc(M * N * sizeof(float));


    template_init_matrix(A, M * K);    
    template_init_matrix(B, K * N);

    long startClock = clock();

    half *dev_A, *dev_B;
    float *dev_C;

    cudaMalloc(&dev_A, M * K * sizeof(half));
    cudaMalloc(&dev_B, K * N * sizeof(half));
    cudaMalloc(&dev_C, M * N * sizeof(float));

    cudaMemcpy(dev_A, A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, K * N * sizeof(half), cudaMemcpyHostToDevice);


   // First: using WMMA
    dim3 gridDim;
    dim3 blockDim;
    
    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    // blockDim.x = 128;
    // blockDim.y = 4;

    // gridDim.x = (M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    // gridDim.y = (N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);


    // V2

    blockDim.x = threads_per_block;
    gridDim.x = N / N_tile;
    gridDim.y = M / M_tile;

    long kernel_call_start_clock = clock();
    matmul_tensor_core_v2<<<gridDim, blockDim>>>(dev_A, dev_B, dev_C, M, N, K, 1, 0);

    cudaDeviceSynchronize();
    long kernel_call_end_clock = clock();

    cudaMemcpy(C, dev_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    long endClock = clock();

    printf("Kernel Time Cost: %f s\n", (kernel_call_end_clock - kernel_call_start_clock) * 1.0 / CLOCKS_PER_SEC);
    printf("Time Cost: %f s\n", (endClock - startClock) * 1.0 / CLOCKS_PER_SEC);

    double workload = 2 * M * K * N;
    float kernel_time_second = (kernel_call_end_clock - kernel_call_start_clock) * 1.0 / CLOCKS_PER_SEC;
    printf("Performance: %lf TFLOPS \n", workload / (kernel_time_second * 1e12));

    float delta = 0;
    for (int i = 0; i < M * N; i++) {
        delta = max(delta, abs(C[i] - 8192));
        if (abs(C[i] - 8192) > 0.5) {
            printf("error at %lld %lld %f\n", i / N, i % N, C[i]);
            break;
        }
    }
    
    printf("max delta %f \n", delta);
}
