// 通过warp分块实现sgemv

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <iostream>

#include "cuda_common.h"

#define GET_TIME_US() \
  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count()
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

template <typename T>
__inline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset, warpSize);
  }
  return val;
}

// 矩阵A * 向量x = 向量y
// 矩阵A的大小为m * n, 向量x的大小为n, 向量y的大小为m
// 假设n为32的倍数，通过warp分块实现sgemv， 每一个warp处理矩阵一行
// blockDim.x = 32, blockDim.y = 4
__global__ void SgemvN32(
  const float* __restrict__ A, const float* __restrict__ x, float* __restrict__ y, int M, int N) {
  auto tx = threadIdx.x;    // 0 ~ 31
  auto ty = threadIdx.y;    // 0 ~ 4
  auto bx = blockIdx.x;     // 0 ~ m / 32
  int lane = tx % warpSize; // 0 ~ 31

  int m = bx * blockDim.y + ty; // 当前处理的行号
  // printf("m: %d, bx: %d, blockDim.y: %d, ty: %d\n", m, bx, blockDim.y, ty);
  if (m >= M) {
    return;
  }

  float sum = 0;
  auto n_warp = (N + warpSize - 1) / warpSize;
  for (int i = 0; i < n_warp; ++i) {
    auto idx = i * warpSize + lane;
    if (idx < N) {
      sum += A[m * N + idx] * x[idx];
    }
  }
  sum = WarpReduceSum(sum);
  if (lane == 0) {
    y[m] = sum;
  }
}

// 矩阵A * 向量x = 向量y
// 矩阵A的大小为m * n, 向量x的大小为n, 向量y的大小为m
// 假设n为128的倍数，通过warp分块实现sgemv， 每一个warp处理矩阵一行，可以通过FLOAT4优化数据加载
// blockDim.x = 32, blockDim.y = 4
__global__ void SgemvN128(float* __restrict__ A, float* __restrict__ x, float* __restrict__ y, int M, int N) {
  auto tx = threadIdx.x;    // 0 ~ 31
  auto ty = threadIdx.y;    // 0 ~ 4
  auto bx = blockIdx.x;     // 0 ~ m / 32
  int lane = tx % warpSize; // 0 ~ 31

  int m = bx * blockDim.y + ty; // 当前处理的行号
  if (m >= M) {
    return;
  }
  float sum = 0;
  auto n_warp = (N + warpSize - 1) / warpSize / 4;
  for (int i = 0; i < n_warp; ++i) {
    auto idx = (i * warpSize + lane) * 4;
    float r[4], r1[4];
    FLOAT4(r[0]) = FLOAT4(A[m * N + idx]);
    FLOAT4(r1[0]) = FLOAT4(x[idx]);
    if (idx < N) {
      sum += r[0] * r1[0] + r[1] * r1[1] + r[2] * r1[2] + r[3] * r1[3];
    }
  }
  sum = WarpReduceSum(sum);
  if (lane == 0) {
    y[m] = sum;
  }
}

void SgemvCpu(const float* A, const float* x, float* y, int M, int N) {
  for (int i = 0; i < M; ++i) {
    y[i] = 0;
    for (int j = 0; j < N; ++j) {
      y[i] += A[i * N + j] * x[j];
    }
  }
}

void Test() {
  int M = 1024;
  int N = 1024;
  float* A = new float[M * N];
  float* A_lead_col = new float[M * N]; // 列式存储
  float* x = new float[N];
  float* y = new float[M];
  float* y_cpu = new float[M];
  float* A_gpu = nullptr;
  float* A_gpu_lead_col = nullptr; // 列式存储
  float* x_gpu = nullptr;
  float* y_gpu = nullptr;
  CUDA_CHECK(cudaMalloc(&A_gpu, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&A_gpu_lead_col, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&x_gpu, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&y_gpu, M * sizeof(float)));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  for (int repeat = 0; repeat < 10; repeat++) {
    for (int i = 0; i < M * N; ++i) {
      A[i] = rand() % 1000 / 1000.0f;
    }
    // 转换为列式存储
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        A_lead_col[j * M + i] = A[i * N + j];
      }
    }
    for (int i = 0; i < N; ++i) {
      x[i] = rand() % 1000 / 1000.0f;
    }

    auto begin = GET_TIME_US();
    SgemvCpu(A, x, y_cpu, M, N);
    auto end = GET_TIME_US();
    std::cout << "cpu time: " << end - begin << " us" << std::endl;

    memset(y, 0, M * sizeof(float));
    begin = GET_TIME_US();
    CUDA_CHECK(cudaMemcpy(A_gpu, A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(x_gpu, x, N * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(32, 4);
    dim3 grid((M + 4 - 1) / 4, 1);
    CUDA_CHECK(cudaMemset(y_gpu, 0, M * sizeof(float)));
    SgemvN32<<<grid, block>>>(A_gpu, x_gpu, y_gpu, M, N);
    CUDA_CHECK(cudaMemcpy(y, y_gpu, M * sizeof(float), cudaMemcpyDeviceToHost));
    end = GET_TIME_US();
    for (int i = 0; i < M; ++i) {
      if (fabs(y[i] - y_cpu[i]) > 1e-3) {
        std::cout << "SgemvN32 error at " << i << " " << y[i] << " " << y_cpu[i] << std::endl;
      }
    }
    std::cout << "SgemvN32 time: " << end - begin << " us" << std::endl;

    memset(y, 0, M * sizeof(float));
    begin = GET_TIME_US();
    CUDA_CHECK(cudaMemcpy(A_gpu, A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(x_gpu, x, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(y_gpu, 0, M * sizeof(float)));
    SgemvN128<<<grid, block>>>(A_gpu, x_gpu, y_gpu, M, N);
    CUDA_CHECK(cudaMemcpy(y, y_gpu, M * sizeof(float), cudaMemcpyDeviceToHost));
    end = GET_TIME_US();
    for (int i = 0; i < M; ++i) {
      if (fabs(y[i] - y_cpu[i]) > 1e-3) {
        std::cout << "SgemvN128 error at " << i << " " << y[i] << " " << y_cpu[i] << std::endl;
      }
    }
    std::cout << "SgemvN128 time: " << end - begin << " us" << std::endl;

    memset(y, 0, M * sizeof(float));
    begin = GET_TIME_US();
    CUBLAS_CHECK(cublasSetMatrix(M, N, sizeof(float), A_lead_col, M, A_gpu_lead_col, M));
    CUBLAS_CHECK(cublasSetVector(N, sizeof(float), x, 1, x_gpu, 1));
    CUBLAS_CHECK(cublasSetVector(M, sizeof(float), y, 1, y_gpu, 1));
    float alpha = 1.0f;
    float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N, M, N, &alpha, A_gpu_lead_col, M, x_gpu, 1, &beta, y_gpu, 1));
    CUBLAS_CHECK(cublasGetVector(M, sizeof(float), y_gpu, 1, y, 1));
    end = GET_TIME_US();
    for (int i = 0; i < M; ++i) {
      if (fabs(y[i] - y_cpu[i]) > 1e-3) {
        std::cout << "SgemvCublas error at " << i << " " << y[i] << " " << y_cpu[i] << std::endl;
      }
    }
    std::cout << "SgemvCublas time: " << end - begin << " us" << std::endl;

    std::cout << "--------" << std::endl;
  }
  CUBLAS_CHECK(cublasDestroy(handle));

  CUDA_CHECK(cudaFree(A_gpu));
  CUDA_CHECK(cudaFree(x_gpu));
  CUDA_CHECK(cudaFree(y_gpu));
  delete[] A;
  delete[] x;
  delete[] y;
  delete[] y_cpu;
}

int main() {
  InitDevice(0);
  Test();
  return 0;
}
