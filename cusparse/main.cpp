#include <cuda_runtime.h>
#include <cusparse.h>

#include <chrono>
#include <cstdio>

#define GET_TIME_US() \
  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count()

void printf_csr_mat(int row, int col, int* csrRowPtr, int* csrColInd, float* csrVal) {
  for (int i = 0; i < row; i++) {
    int begin = csrRowPtr[i];
    int end = csrRowPtr[i + 1];
    for (int j = 0; j < col; j++) {
      if (j < end - begin && csrColInd[begin + j] == j) {
        printf("%7.1f", csrVal[begin + j]);
      } else {
        printf("%7.1f", 0.0);
      }
    }
    printf("\n");
  }
}

int main(void) {
  cusparseHandle_t handle = NULL;
  cusparseCreate(&handle);

  const int rowA = 3, colA = 3, nnzA = 6;
  const int rowB = 3, colB = 2, nnzB = 4;

  // csr(compressed sparse row)格式存储稀疏矩阵
  // 代表有3行，第一行有2(2-0)个元素，第二行有1(3-2)个元素，第三行有3(6-3)个元素，6代表非零元素个数
  int csrRowPtrA[] = {0, 2, 3, 6};
  int csrColIndA[] = {0, 2, 1, 0, 1, 2};            // 代表非零元素的列索引
  float csrValA[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // 代表非零元素的值
  // 以上稀疏矩阵：
  // [1.0, 0.0, 2.0]
  // [0.0, 3.0, 0.0]
  // [4.0, 5.0, 6.0]

  int csrRowPtrB[] = {0, 2, 4, 4};
  int csrColIndB[] = {0, 1, 0, 1};
  float csrValB[] = {7.0, 8.0, 9.0, 10.0};
  // 以上稀疏矩阵：
  // [7.0, 8.0]
  // [9.0, 10.0]
  // [0.0, 0.0]

  int *d_csrRowPtrA, *d_csrColIndA, *d_csrRowPtrB, *d_csrColIndB;
  float *d_csrValA, *d_csrValB;

  cudaMalloc((void**)&d_csrRowPtrA, (rowA + 1) * sizeof(int));
  cudaMalloc((void**)&d_csrColIndA, nnzA * sizeof(int));
  cudaMalloc((void**)&d_csrValA, nnzA * sizeof(float));

  cudaMemcpy(d_csrRowPtrA, csrRowPtrA, (rowA + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csrColIndA, csrColIndA, nnzA * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csrValA, csrValA, nnzA * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_csrRowPtrB, (rowB + 1) * sizeof(int));
  cudaMalloc((void**)&d_csrColIndB, nnzB * sizeof(int));
  cudaMalloc((void**)&d_csrValB, nnzB * sizeof(float));

  cudaMemcpy(d_csrRowPtrB, csrRowPtrB, (rowB + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csrColIndB, csrColIndB, nnzB * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csrValB, csrValB, nnzB * sizeof(float), cudaMemcpyHostToDevice);

  const int rowC = rowA, colC = colB;
  int nnzC = 0;

  int *d_csrRowPtrC, *d_csrColIndC;
  float* d_csrValC;

  auto begin_us = GET_TIME_US();
  // 计算稀疏矩阵乘法结果的非零元素个数
  cudaMalloc((void**)&d_csrRowPtrC, (rowC + 1) * sizeof(int));
  cusparseMatDescr_t descr_a, descr_b, descr_c;
  cusparseCreateMatDescr(&descr_a);
  cusparseCreateMatDescr(&descr_b);
  cusparseCreateMatDescr(&descr_c);
  cusparseSetMatType(descr_a, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatType(descr_b, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatType(descr_c, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr_a, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatIndexBase(descr_b, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatIndexBase(descr_c, CUSPARSE_INDEX_BASE_ZERO);
  cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, rowA, colB, colA,
                      descr_a, nnzA, d_csrRowPtrA, d_csrColIndA, descr_b, nnzB, d_csrRowPtrB, d_csrColIndB, descr_c,
                      d_csrRowPtrC, &nnzC);

  cudaMalloc((void**)&d_csrColIndC, nnzC * sizeof(int));
  cudaMalloc((void**)&d_csrValC, nnzC * sizeof(float));

  cusparseScsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, rowA, colB, colA,
                   descr_a, nnzA, d_csrValA, d_csrRowPtrA, d_csrColIndA, descr_b, nnzB, d_csrValB, d_csrRowPtrB,
                   d_csrColIndB, descr_c, d_csrValC, d_csrRowPtrC, d_csrColIndC);
  auto end_us = GET_TIME_US();
  printf("cusparseScsrgemm used time = %ld us\n", end_us - begin_us);

  // 从设备端拷贝结果到主机端
  int* csrRowPtrC = (int*)malloc((rowC + 1) * sizeof(int));
  int* csrColIndC = (int*)malloc(nnzC * sizeof(int));
  float* csrValC = (float*)malloc(nnzC * sizeof(float));

  cudaMemcpy(csrRowPtrC, d_csrRowPtrC, (rowC + 1) * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(csrColIndC, d_csrColIndC, nnzC * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(csrValC, d_csrValC, nnzC * sizeof(float), cudaMemcpyDeviceToHost);

  // 打印系数矩阵A, B, C
  printf("A:\n");
  printf_csr_mat(rowA, colA, csrRowPtrA, csrColIndA, csrValA);
  printf("B:\n");
  printf_csr_mat(rowB, colB, csrRowPtrB, csrColIndB, csrValB);
  printf("C:\n");
  printf_csr_mat(rowC, colC, csrRowPtrC, csrColIndC, csrValC);

  cudaFree(d_csrRowPtrA);
  cudaFree(d_csrColIndA);
  cudaFree(d_csrValA);
  cudaFree(d_csrRowPtrB);
  cudaFree(d_csrColIndB);
  cudaFree(d_csrValB);
  cudaFree(d_csrRowPtrC);
  cudaFree(d_csrColIndC);
  cudaFree(d_csrValC);

  cusparseDestroy(handle);

  return 0;
}