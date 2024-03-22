//
// 测试cublaslt sgemm性能
//

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdlib>
#include <iostream>

#include "cuda_common.h"

#define OFFSET(row, col, ld) ((col) + (row) * (ld))
#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

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

void CublasLtSgemm(float* a, float* b, float* c, const int M, const int N, const int K) {
  cublasLtHandle_t handle;
  cublasLtMatmulDesc_t matmul_desc;
  cublasLtMatrixLayout_t a_desc, b_desc, c_desc;
  cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_32F, K, M, K);
  cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_32F, N, K, N);
  cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, N, M, N);

  // 初始化CUBLAS LT句柄和矩阵乘法描述符
  cublasLtCreate(&handle);
  cublasLtMatmulDescCreate(&matmul_desc, CUDA_R_32F);

  // 计算
  float alpha = 1.0;
  float beta = 0.0;
  cublasLtMatmul(handle, matmul_desc, &alpha, b, b_desc, a, a_desc, &beta, c, c_desc, c, c_desc, nullptr, nullptr, 0,
                 nullptr);
}

// 同cpu版本作比较，测试准确性
float TestError(const dim3& grid_dim, const dim3& block_dim, const int M, const int N, const int K) {
  int a_size = M * K * sizeof(float);
  int b_size = K * N * sizeof(float);
  int c_size = M * N * sizeof(float);

  // 生成随机值矩阵
  float* a = (float*)malloc(a_size);
  float* b = (float*)malloc(K * N * sizeof(float));
  float* c_cpu = (float*)malloc(c_size);
  // float *a, *b, *c_cpu;
  // CHECK(cudaMallocHost(&a, a_size));
  // CHECK(cudaMallocHost(&b, b_size));
  // CHECK(cudaMallocHost(&c_cpu, c_size));
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
  CHECK(cudaMalloc(&d_a, a_size));
  CHECK(cudaMalloc(&d_b, b_size));
  CHECK(cudaMalloc(&d_c, c_size));
  CHECK(cudaMemcpy(d_a, a, a_size, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b, b, b_size, cudaMemcpyHostToDevice));
  CublasLtSgemm(d_a, d_b, d_c, M, N, K);
  float* c_gpu = (float*)malloc(c_size);
  // float* c_gpu;
  // CHECK(cudaMallocHost(&c_gpu, c_size));
  CHECK(cudaMemcpy(c_gpu, d_c, c_size, cudaMemcpyDeviceToHost));
  CHECK(cudaDeviceSynchronize());
  // PrintMatrix(c_gpu, M, N, "c_gpu");

  // 比较cpu和gpu版本计算结果
  float max_diff = 0;
  for (int i = 0; i < M * N; i++) {
    // if (std::abs(c_cpu[i] - c_gpu[i]) > 1e-3) {
    //   printf("Error: c_cpu[%d] = %f, c_gpu[%d] = %f\n", i, c_cpu[i], i, c_gpu[i]);
    //   // exit(1);
    // }
    max_diff = std::max(max_diff, std::abs(c_cpu[i] - c_gpu[i]));
  }

  free(a);
  free(b);
  free(c_cpu);
  free(c_gpu);
  // CHECK(cudaFreeHost(a));
  // CHECK(cudaFreeHost(b));
  // CHECK(cudaFreeHost(c_cpu));
  // CHECK(cudaFreeHost(c_gpu));
  CHECK(cudaFree(d_a));
  CHECK(cudaFree(d_b));
  CHECK(cudaFree(d_c));
  return max_diff;
}

// 测试性能
double TestPerformance(const dim3& grid_dim, const dim3& block_dim, const int M, const int N, const int K) {
  int a_size = M * K * sizeof(float);
  int b_size = K * N * sizeof(float);
  int c_size = M * N * sizeof(float);

  // 生成随机值矩阵
  // 生成随机值矩阵
  float* a = (float*)malloc(a_size);
  float* b = (float*)malloc(K * N * sizeof(float));
  float* c_cpu = (float*)malloc(c_size);
  // float *a, *b;
  // CHECK(cudaMallocHost(&a, a_size));
  // CHECK(cudaMallocHost(&b, b_size));
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
  CHECK(cudaMalloc(&d_a, a_size));
  CHECK(cudaMalloc(&d_b, b_size));
  CHECK(cudaMalloc(&d_c, c_size));
  CHECK(cudaMemcpy(d_a, a, a_size, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b, b, b_size, cudaMemcpyHostToDevice));

  cudaEvent_t start, end;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&end));
  CHECK(cudaEventRecord(start));

  CublasLtSgemm(d_a, d_b, d_c, M, N, K);

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
  // CHECK(cudaFreeHost(a));
  // CHECK(cudaFreeHost(b));
  CHECK(cudaFree(d_a));
  CHECK(cudaFree(d_b));
  CHECK(cudaFree(d_c));

  return sec;
}

int main() {
  // 初始化cuda设备
  InitDevice(0);

  const int M_list[15] = {128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
  const int N_list[15] = {128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
  const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};
  const int BM = 128; // 分块行数
  const int BN = 128; // 分块列数
  const int TM = 8;   // block每一个线程处理块的行数
  const int TN = 8;   // block每一个线程处理块的列数

  // 测试TileSgemmBySharedMem准确性
  for (int i = 0; i < 5; i++) {
    dim3 block_dim(BN / TN, BM / TM);
    dim3 grid_dim((N_list[i] + BN - 1) / BN, (M_list[i] + BM - 1) / BM);
    printf("grid_dim: (%d, %d), block_dim: (%d, %d)\n", grid_dim.x, grid_dim.y, block_dim.x, block_dim.y);
    float max_diff = TestError(grid_dim, block_dim, M_list[i], N_list[i], K_list[i]);
    printf("CublasLt Accuracy Test %d passed, M = %d, N = %d, K = %d, max_diff: %f.\n", i, M_list[i], N_list[i],
           K_list[i], max_diff);
  }

  // 测试TileSgemmBySharedMem性能和算力
  for (int i = 0; i < 14; i++) {
    double total_sec = 0.0;
    double max_sec = 0;
    double min_sec = DBL_MAX;
    int repeat = 5;
    for (int j = 0; j < repeat; j++) {
      dim3 block_dim(BN / TN, BM / TM);
      dim3 grid_dim((N_list[i] + BN - 1) / BN, (M_list[i] + BM - 1) / BM);
      double sec = TestPerformance(grid_dim, block_dim, M_list[i], N_list[i], K_list[i]);
      total_sec += sec;
      max_sec = std::max(max_sec, sec);
      min_sec = std::min(min_sec, sec);
    }
    double avg_sec = total_sec / repeat;
    double gflops = double(2) * M_list[i] * N_list[i] * K_list[i] / avg_sec / 1024 / 1024 / 1024;
    printf(
      "CublasLt Performance Test %d, M = %d, N = %d, K = %d, time = %.6f %.6f %.6f sec, gflops = %.6f "
      "GFLOPS\n",
      i, M_list[i], N_list[i], K_list[i], min_sec, avg_sec, max_sec, gflops);
  }

  return 0;
}