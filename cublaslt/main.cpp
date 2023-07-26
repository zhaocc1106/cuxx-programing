#include <cublasLt.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <thread>

int main() {
  cublasLtHandle_t handle;
  cublasLtMatmulDesc_t matmul_desc;

  // 初始化CUBLAS LT句柄和矩阵乘法描述符
  cublasLtCreate(&handle);
  cublasLtMatmulDescCreate(&matmul_desc, CUDA_R_32F);

  // 设置矩阵A、B和C的维度
  int m = 5, n = 3, k = 4;

  // 分配设备内存并填充矩阵A、B
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, m * k * sizeof(float));
  cudaMalloc(&d_B, k * n * sizeof(float));
  cudaMalloc(&d_C, m * n * sizeof(float));
  // 列式优先填充矩阵A、B
  float h_A[m * k], h_B[k * n];
  for (int j = 0; j < k; j++) {
    for (int i = 0; i < m; i++) {
      h_A[j * m + i] = (float)(i + j * m);
    }
  }
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < k; i++) {
      h_B[j * k + i] = (float)(i + j * k);
    }
  }
  cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);

  // 设置算法搜索的属性
  cublasLtMatmulPreference_t preference;
  cublasLtMatmulPreferenceCreate(&preference);
  // malloc workspace
  void* workspace;
  size_t workspace_size = 1024 * 1024; // 1MB
  cudaMalloc(&workspace, workspace_size);
  cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace,
                                       workspace_size);

  // 搜索最佳的矩阵乘法算法，它使用启发式算法来确定最适合给定矩阵维度和硬件配置的CUBLAS
  // LT算法
  cublasLtMatrixLayout_t a_desc, b_desc, c_desc;
  int lda = m, ldb = k, ldc = m;
  cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_32F, m, k, lda);
  cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_32F, k, n, ldb);
  cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, m, n, ldc);
  int requested_algo_count = 10;
  cublasLtMatmulHeuristicResult_t heuristic_result[requested_algo_count];
  cublasLtMatmulAlgoGetHeuristic(handle, matmul_desc, a_desc, b_desc, c_desc, c_desc, preference, requested_algo_count,
                                 heuristic_result, &requested_algo_count);
  printf("requested_algo_count = %d\n", requested_algo_count);
  float alpha = 1.0f, beta = 0.0f;
  for (int i = 0; i < requested_algo_count; i++) {
    printf("heuristic_result[%d].state = %d\n", i, heuristic_result[i].state);
    printf("heuristic_result[%d].wavesCount = %f\n", i, heuristic_result[i].wavesCount);
    printf("heuristic_result[%d].workspaceSize = %ld\n", i, heuristic_result[i].workspaceSize);
    auto begin_us =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
    cublasLtMatmul(handle, matmul_desc, &alpha, d_A, a_desc, d_B, b_desc, &beta, d_C, c_desc, d_C, c_desc,
                   &heuristic_result[i].algo, workspace, workspace_size, nullptr);
    auto end_us =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
    printf("used time = %ld us\n", end_us - begin_us);
    printf("-------------------------------------------\n");
  }

  // 拷贝结果到主机内存
  float h_C[m * n];
  cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

  // 行优先打印矩阵
  printf("A =\n");
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      printf("%7.0f", h_A[j * m + i]);
    }
    printf("\n");
  }
  printf("B =\n");
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      printf("%7.0f", h_B[j * k + i]);
    }
    printf("\n");
  }
  printf("C =\n");
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%7.0f", h_C[j * m + i]);
    }
    printf("\n");
  }

  // 释放资源
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cublasLtDestroy(handle);
  cublasLtMatmulDescDestroy(matmul_desc);

  return 0;
}