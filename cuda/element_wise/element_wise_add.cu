#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <iostream>

#include "cuda_common.h"

#define GET_US() \
  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

__global__ void ElementWiseAdd(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ c,
                               const int N) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void ElementWiseAddFloat4(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, const int N) {
  auto idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float r_a[4];
    FLOAT4(r_a) = FLOAT4(a[idx]);
    float r_b[4];
    FLOAT4(r_b) = FLOAT4(b[idx]);
    c[idx] = r_a[0] + r_b[0];
    c[idx + 1] = r_a[1] + r_b[1];
    c[idx + 2] = r_a[2] + r_b[2];
    c[idx + 3] = r_a[3] + r_b[3];
  }
}

void ElementWiseAddCpu(const float* a, const float* b, float* c, const int N) {
  for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}

void Test(int N) {
  auto* a = new float[N];
  auto* b = new float[N];
  auto* c = new float[N];

  float* a_d;
  float* b_d;
  float* c_d;
  auto* c_h = new float[N];
  CUDA_CHECK(cudaMalloc(&a_d, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b_d, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&c_d, N * sizeof(float)));

  int repeat = 10;
  for (int i = 0; i < repeat; i++) {
    for (int j = 0; j < N; j++) {
      a[j] = rand() % 10;
      b[j] = rand() % 10;
    }
    std::cout << "N: " << N << std::endl;

    auto begin = GET_US();
    ElementWiseAddCpu(a, b, c, N);
    auto end = GET_US();
    std::cout << "ElementWiseAddCpu, time: " << end - begin << " us" << std::endl;

    begin = GET_US();
    CUDA_CHECK(cudaMemcpy(a_d, a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, b, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(c_d, 0, N * sizeof(float)));
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    ElementWiseAdd<<<grid, block>>>(a_d, b_d, c_d, N);
    CUDA_CHECK(cudaMemcpy(c_h, c_d, N * sizeof(float), cudaMemcpyDeviceToHost));
    end = GET_US();
    std::cout << "ElementWiseAdd, time: " << end - begin << " us" << std::endl;
    for (int j = 0; j < N; j++) {
      if (c[j] != c_h[j]) {
        std::cout << "Error: " << c[j] << " != " << c_h[j] << std::endl;
        break;
      }
    }

    begin = GET_US();
    CUDA_CHECK(cudaMemcpy(a_d, a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, b, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(c_d, 0, N * sizeof(float)));
    ElementWiseAddFloat4<<<grid, block>>>(a_d, b_d, c_d, N);
    CUDA_CHECK(cudaMemcpy(c_h, c_d, N * sizeof(float), cudaMemcpyDeviceToHost));
    end = GET_US();
    std::cout << "ElementWiseAddFloat4, time: " << end - begin << " us" << std::endl;
    for (int j = 0; j < N; j++) {
      if (c[j] != c_h[j]) {
        std::cout << "Error: " << c[j] << " != " << c_h[j] << std::endl;
        break;
      }
    }

    // cublas
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    begin = GET_US();
    float alpha = 1.0;
    CUDA_CHECK(cudaMemcpy(a_d, a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, b, N * sizeof(float), cudaMemcpyHostToDevice));
    CUBLAS_CHECK(cublasSaxpy(handle, N, &alpha, a_d, 1, b_d, 1));
    end = GET_US();
    CUDA_CHECK(cudaMemcpy(c_h, b_d, N * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "cublasSaxpy, time: " << end - begin << " us" << std::endl;
    for (int j = 0; j < N; j++) {
      if (c[j] != c_h[j]) {
        std::cout << "Error: " << c[j] << " != " << c_h[j] << std::endl;
        break;
      }
    }
    CUBLAS_CHECK(cublasDestroy(handle));

    std::cout << "--------" << std::endl;
  }

  delete[] a;
  delete[] b;
  delete[] c;
  CUDA_CHECK(cudaFree(a_d));
  CUDA_CHECK(cudaFree(b_d));
  CUDA_CHECK(cudaFree(c_d));
  delete[] c_h;
}

int main() {
  InitDevice(0);
  Test(1 << 20);
  return 0;
}