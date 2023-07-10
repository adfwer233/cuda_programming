#include<stdio.h>

void genRandomMatrix(float *A, int M, int N)
{
    srand(time(NULL)); // Initialization, should only be called once.
    float a = 5.0;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = (float)rand() / ((float)RAND_MAX / a);
        }
    }
}

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

constexpr int K_block = 16;

__global__ void matmul_tile(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {
    __shared__ int tile_A[K_block][K_block];
    __shared__ int tile_B[K_block][K_block];

    float c = 0;

    for (int t = 0; t < K; t += K_block) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = t + threadIdx.y;

        tile_A[threadIdx.x][threadIdx.y] = A[i * M + j];
        tile_B[threadIdx.y][threadIdx.x] = B[j * K + i];

        __syncthreads();

        for (int k = 0; k < K_block; k++)
            // c += 1;
            c += tile_A[threadIdx.x][k] * tile_B[k][threadIdx.y];
    
        __syncthreads();
    }

    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    C[tx * M + ty] = c;
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

    const int thread_num = 16;
    dim3 grid(M / thread_num, N / thread_num);
    dim3 block(thread_num, thread_num);
    matmul_tile<<<grid, block>>>(dev_A, dev_B, dev_C, M, N, K, 1, 0);

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
        }
    }
    
    printf("max delta %f \n", delta);
}