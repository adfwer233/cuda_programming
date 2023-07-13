// #define __CUDA_ARCH__ 860

#include<iostream>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>


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

int main() {
    int M = 8192;
    int K = 8192;
    int N = 8192;

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
    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = (M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

    long kernel_call_start_clock = clock();
    matmul_tensor_core<<<gridDim, blockDim>>>(dev_A, dev_B, dev_C, M, N, K, 1, 0);

    cudaDeviceSynchronize();
    long kernel_call_end_clock = clock();

    cudaMemcpy(C, dev_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    long endClock = clock();

    printf("Kernel Time Cost: %f s\n", (kernel_call_end_clock - kernel_call_start_clock) * 1.0 / CLOCKS_PER_SEC);
    printf("Time Cost: %f s\n", (endClock - startClock) * 1.0 / CLOCKS_PER_SEC);

    float delta = 0;
    for (int i = 0; i < M * N; i++) {
        delta = max(delta, abs(C[i] - 8192));
        if (abs(C[i] - 8192) > 0.5) {
            printf("error at %d %d \n", i / N, i % N);
            break;
        }
    }
    
    printf("max delta %f \n", delta);
}