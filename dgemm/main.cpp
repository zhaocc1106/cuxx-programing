// Example: DGEMM usage.
//-----------------------------------------------------------
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <iostream>

int main() {
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  stat = cublasCreate(&handle);

  // Matrix A
  float* a = 0;
  float* devPtrA;
  const int m = 6;
  const int n = 5;
  a = (float*)malloc(m * n * sizeof(*a));
  if (!a) {
    printf("host memory allocation failed");
    return EXIT_FAILURE;
  }
  // cublas api数据布局是列优先，所以这里按列优先初始化
  for (int j = 1; j <= n; j++) {
    for (int i = 1; i <= m; i++) {
      a[(j - 1) * m + (i - 1)] = (float)((i - 1) * n + j);
    }
  }
  // 按行优先打印矩阵a
  printf("a =\n");
  for (int i = 1; i <= m; i++) {
    for (int j = 1; j <= n; j++) {
      printf("%7.0f", a[(j - 1) * m + (i - 1)]);
    }
    printf("\n");
  }
  cudaStat = cudaMalloc((void**)&devPtrA, m * n * sizeof(*a));
  if (cudaStat != cudaSuccess) {
    printf("device memory allocation failed");
    return EXIT_FAILURE;
  }
  stat = cublasSetMatrix(m, n, sizeof(*a), a, m, devPtrA, m);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data download failed");
    cudaFree(devPtrA);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }

  // Matrix B
  float* b = 0;
  float* devPtrB;
  const int m2 = 5;
  const int n2 = 3;
  b = (float*)malloc(m2 * n2 * sizeof(*b));
  if (!b) {
    printf("host memory allocation failed");
    return EXIT_FAILURE;
  }
  // cublas api数据布局是列优先，所以这里按列优先初始化
  for (int j = 1; j <= n2; j++) {
    for (int i = 1; i <= m2; i++) {
      b[(j - 1) * m2 + (i - 1)] = (float)((i - 1) * n2 + j);
    }
  }
  // print b
  printf("b =\n");
  for (int i = 1; i <= m2; i++) {
    for (int j = 1; j <= n2; j++) {
      printf("%7.0f", b[(j - 1) * m2 + (i - 1)]);
    }
    printf("\n");
  }
  cudaStat = cudaMalloc((void**)&devPtrB, m2 * n2 * sizeof(*b));
  if (cudaStat != cudaSuccess) {
    printf("device memory allocation failed");
    cudaFree(devPtrA);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  stat = cublasSetMatrix(m2, n2, sizeof(*b), b, m2, devPtrB, m2);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data download failed");
    cudaFree(devPtrB);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }

  // dgemm
  const int m3 = 6;
  const int n3 = 3;
  const int k3 = 5;
  const float alf = 1;
  const float bet = 1;
  const float* alpha = &alf;
  const float* beta = &bet;
  float* c = 0;
  float* devPtrC;
  c = (float*)malloc(m3 * n3 * sizeof(*c));
  if (!c) {
    printf("host memory allocation failed");
    return EXIT_FAILURE;
  }
  for (int j = 1; j <= n3; j++) {
    for (int i = 1; i <= m3; i++) {
      c[(j - 1) * m3 + (i - 1)] = 0;
    }
  }
  cudaStat = cudaMalloc((void**)&devPtrC, m3 * n3 * sizeof(*c));
  if (cudaStat != cudaSuccess) {
    printf("device memory allocation failed");
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  stat = cublasSetMatrix(m3, n3, sizeof(*c), c, m3, devPtrC, m3);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data download failed");
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }

  // C = A*B
  stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m3, n3, k3, alpha, devPtrA, m, devPtrB, m2, beta, devPtrC, m3);

  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("cublasSgemm failed");
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }

  stat = cublasGetMatrix(m3, n3, sizeof(*c), devPtrC, m3, c, m3);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data upload failed");
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }

  // print c after Sgemm
  printf("c final =\n");
  for (int i = 1; i <= m3; i++) {
    for (int j = 1; j <= n3; j++) {
      printf("%7.0f", c[(j - 1) * m3 + (i - 1)]);
    }
    printf("\n");
  }

  cudaFree(devPtrA);
  cudaFree(devPtrB);
  cudaFree(devPtrC);
  cublasDestroy(handle);
  free(a);
  free(b);
  free(c);
  return EXIT_SUCCESS;
}
