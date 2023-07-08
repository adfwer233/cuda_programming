#include<stdio.h>

void initMatrix(float *A, int size) {
    for (int i = 0; i < size; i++) {
        A[i] = 1.0;
    }
}

__global__ void matmul(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;

    if (tx < M && ty < N) {
        float c = 0;
        for (int i = 0; i < K; i++) {
            c += A[tx * M + i] * B[i * K + ty];
        }
        C[tx * M + ty] = beta * C[tx * M + ty] + alpha * c;
    }
}

constexpr int M_block = 128;
constexpr int N_block = 128;
constexpr int K_block = 8;
constexpr int M_thread = 8;
constexpr int N_thread = 8;

__global__ void matmul_tile(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int tx = blockDim.x * blockIdx.x * M_block + M_thread * threadIdx.x;
    int ty = blockDim.y * blockIdx.y * N_block + N_thread * threadIdx.y;

    // for (int i = 0; i < M_thread; i++) {
    //     for (int j = 0; j < N_thread; j++) {
    //         int c = 0;
    //         for (int k = 0; k < K; k++) {
    //             c += A[(tx + i) * M + k] * B[k * K + ty + j];
    //         }
    //         C[(tx + i) * M + ty + j] = alpha * c + beta * C[(tx + i) * M + ty + j];
    //     }
    // }

    float c[M_thread * N_thread] = {};

    for (int block = 0; block < K; block += K_block) {
        #pragma unroll
        for (int k = 0; k < K_block; k++) {
            for (int i = 0 ; i < M_thread; i++) {
                for (int j = 0; j < N_thread; j++) {
                    c[i * M_thread + j] += A[(tx + i) * M + block + k] * B[(block + k) * K + ty + j];
                }
            }
        }
    }

    for (int i = 0; i < M_thread; i++) {
        for (int j = 0; j < N_thread; j++) {
            C[(tx + i) * M + ty + j] = alpha * c[i * M_thread + j] + beta * C[(tx + i) * M + ty + j];
        }
    }
}

int main() {
    int M = 8192;
    int K = 8192;
    int N = 8192;

    float *A, *B, *C;
    A = (float *)malloc(M * K * sizeof(float));
    B = (float *)malloc(K * N * sizeof(float));
    C = (float *)malloc(M * N * sizeof(float));

    float *dev_A, *dev_B, *dev_C;

    long startClock = clock();

    cudaMalloc(&dev_A, M * K * sizeof(float));
    cudaMalloc(&dev_B, K * N * sizeof(float));
    cudaMalloc(&dev_C, M * N * sizeof(float));

    initMatrix(A, M * K);
    initMatrix(B, K * N);

    cudaMemcpy(dev_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    long kernel_call_start_clock = clock();

    // const int thread_num = 32;
    // dim3 grid(M / thread_num, N / thread_num);
    // dim3 block(thread_num, thread_num);
    // matmul<<<grid, block>>>(dev_A, dev_B, dev_C, M, N, K, 1, 0);

    const int thread_num = M_block / M_thread;
    dim3 grid(M / thread_num, N / thread_num);
    dim3 block(thread_num, thread_num);
    matmul<<<grid, block>>>(dev_A, dev_B, dev_C, M, N, K, 1, 0);

    cudaDeviceSynchronize();
    long kernel_call_end_clock = clock();

    cudaMemcpy(C, dev_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    long endClock = clock();

    printf("Kernel Time Cost: %f s\n", (kernel_call_end_clock - kernel_call_start_clock) * 1.0 / CLOCKS_PER_SEC);
    printf("Time Cost: %f s\n", (endClock - startClock) * 1.0 / CLOCKS_PER_SEC);

    float delta = 0;
    for (int i = 0; i < M * N; i++) {
        delta = max(delta, abs(C[i] - 8192));
    }
    
    printf("max delta %f \n", delta);
}