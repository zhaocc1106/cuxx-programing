//
// 实现简单的矩阵乘法sgemm
//

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdlib>
#include <iostream>

#include "cuda_common.h"

#define OFFSET(row, col, ld) ((col) + (row) * (ld))
#define BLOCK_ROWS 32
#define BLOCK_COLS 32

void CpuSgemm(float* a, float* b, float* c, const int M, const int N, const int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0;
      for (int k = 0; k < K; k++) {
        sum += a[OFFSET(i, k, K)] * b[OFFSET(k, j, N)];
      }
      c[OFFSET(i, j, N)] = sum;
    }
  }
}

// 简单的矩阵乘法实现
__global__ void NaiveSgemm(
  float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, const int M, const int N, const int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float sum = 0;
#pragma unroll
    for (int k = 0; k < K; k++) {
      sum += a[OFFSET(row, k, K)] * b[OFFSET(k, col, N)];
    }
    c[OFFSET(row, col, N)] = sum;
  }
}

// 同cpu版本作比较，测试准确性
void TestError(void (*GpuSgemm)(float* a, float* b, float* c, const int M, const int N, const int K),
               const dim3& grid_dim,
               const dim3& block_dim,
               const int M,
               const int N,
               const int K) {
  // 生成随机值矩阵
  float* a = (float*)malloc(M * K * sizeof(float));
  float* b = (float*)malloc(K * N * sizeof(float));
  float* c_cpu = (float*)malloc(M * N * sizeof(float));
  for (int i = 0; i < M * K; i++) {
    a[i] = rand() / static_cast<float>(RAND_MAX);
  }
  // PrintMatrix(a, M, K, "a");
  for (int i = 0; i < K * N; i++) {
    b[i] = rand() / static_cast<float>(RAND_MAX);
  }
  // PrintMatrix(b, K, N, "b");
  // cpu版本计算
  CpuSgemm(a, b, c_cpu, M, N, K);
  // PrintMatrix(c_cpu, M, N, "c_cpu");

  // gpu版本计算
  float* d_a;
  float* d_b;
  float* d_c;
  CHECK(cudaMalloc(&d_a, M * K * sizeof(float)));
  CHECK(cudaMalloc(&d_b, K * N * sizeof(float)));
  CHECK(cudaMalloc(&d_c, M * N * sizeof(float)));
  CHECK(cudaMemcpy(d_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice));
  GpuSgemm<<<grid_dim, block_dim>>>(d_a, d_b, d_c, M, N, K);
  float* c_gpu = (float*)malloc(M * N * sizeof(float));
  CHECK(cudaMemcpy(c_gpu, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaDeviceSynchronize());
  // PrintMatrix(c_gpu, M, N, "c_gpu");

  // 比较cpu和gpu版本计算结果
  for (int i = 0; i < M * N; i++) {
    if (std::abs(c_cpu[i] - c_gpu[i]) > 1e-3) {
      printf("Error: c_cpu[%d] = %f, c_gpu[%d] = %f\n", i, c_cpu[i], i, c_gpu[i]);
      exit(1);
    }
  }

  free(a);
  free(b);
  free(c_cpu);
  free(c_gpu);
  CHECK(cudaFree(d_a));
  CHECK(cudaFree(d_b));
  CHECK(cudaFree(d_c));
}

double TestPerformance(void (*GpuSgemm)(float* a, float* b, float* c, const int M, const int N, const int K),
                       const dim3& grid_dim,
                       const dim3& block_dim,
                       const int M,
                       const int N,
                       const int K) {
  // 生成随机值矩阵
  float* a = (float*)malloc(M * K * sizeof(float));
  float* b = (float*)malloc(K * N * sizeof(float));
  for (int i = 0; i < M * K; i++) {
    a[i] = rand() / static_cast<float>(RAND_MAX);
  }
  for (int i = 0; i < K * N; i++) {
    b[i] = rand() / static_cast<float>(RAND_MAX);
  }

  // gpu版本计算
  float* d_a;
  float* d_b;
  float* d_c;
  CHECK(cudaMalloc(&d_a, M * K * sizeof(float)));
  CHECK(cudaMalloc(&d_b, K * N * sizeof(float)));
  CHECK(cudaMalloc(&d_c, M * N * sizeof(float)));
  CHECK(cudaMemcpy(d_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice));

  cudaEvent_t start, end;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&end));
  CHECK(cudaEventRecord(start));

  GpuSgemm<<<grid_dim, block_dim>>>(d_a, d_b, d_c, M, N, K);

  CHECK(cudaEventRecord(end));
  CHECK(cudaEventSynchronize(end));

  float msec = 0;
  CHECK(cudaEventElapsedTime(&msec, start, end));
  // printf("msec: %f\n", msec);
  double sec = msec / 1000.0;

  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(end));

  CHECK(cudaDeviceSynchronize());

  free(a);
  free(b);
  CHECK(cudaFree(d_a));
  CHECK(cudaFree(d_b));
  CHECK(cudaFree(d_c));

  return sec;
}

int main() {
  // 初始化cuda设备
  InitDevice(0);

  const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
  const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
  const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};

  // 测试准确性
  for (int i = 0; i < 6; i++) {
    dim3 block_dim(BLOCK_COLS, BLOCK_ROWS);
    dim3 grid_dim((N_list[i] + block_dim.x - 1) / block_dim.x, (M_list[i] + block_dim.y - 1) / block_dim.y);
    TestError(NaiveSgemm, grid_dim, block_dim, M_list[i], N_list[i], K_list[i]);
    printf("NaiveSgemm Accuracy Test %d, M = %d, N = %d, K = %d\n", i, M_list[i], N_list[i], K_list[i]);
  }

  // 测试性能和算力
  for (int i = 0; i < 15; i++) {
    dim3 block_dim(BLOCK_COLS, BLOCK_ROWS);
    dim3 grid_dim((N_list[i] + block_dim.x - 1) / block_dim.x, (M_list[i] + block_dim.y - 1) / block_dim.y);
    double total_sec = 0.0;
    double max_sec = 0;
    double min_sec = DBL_MAX;
    int repeat = 5;
    for (int j = 0; j < repeat; j++) {
      double sec = TestPerformance(NaiveSgemm, grid_dim, block_dim, M_list[i], N_list[i], K_list[i]);
      total_sec += sec;
      max_sec = std::max(max_sec, sec);
      min_sec = std::min(min_sec, sec);
    }
    double avg_sec = total_sec / repeat;
    double gflops = double(2) * M_list[i] * N_list[i] * K_list[i] / avg_sec / 1024 / 1024 / 1024;
    printf("NaiveSgemm Performance Test %d, M = %d, N = %d, K = %d, time = %.6f %.6f %.6f sec, gflops = %.6f GFLOPS\n",
           i, M_list[i], N_list[i], K_list[i], min_sec, avg_sec, max_sec, gflops);
  }

  return 0;
}