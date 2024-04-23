// CUDA实现sigmoid算子

#include <cuda_runtime.h>

#include <chrono>
#include <iostream>

#include "cuda_common.h"

#define FLOAT4(val) (reinterpret_cast<float4*>(&(val))[0])
#define GET_US() \
  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count()

__global__ void Sigmoid(float* __restrict__ a, float* __restrict__ b, int N) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }
  b[idx] = 1.0f / (1.0f + expf(-a[idx]));
}

__global__ void SigmoidFloat4(float* __restrict__ a, float* __restrict__ b, int N) {
  auto idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx >= N) {
    return;
  }
  float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  val = FLOAT4(a[idx]);
  val.x = 1.0f / (1.0f + expf(-val.x));
  if (idx + 1 < N) {
    val.y = 1.0f / (1.0f + expf(-val.y));
  }
  if (idx + 2 < N) {
    val.z = 1.0f / (1.0f + expf(-val.z));
  }
  if (idx + 3 < N) {
    val.w = 1.0f / (1.0f + expf(-val.w));
  }
  FLOAT4(b[idx]) = val;
}

void SigmoidCpu(float* a, float* b, int N) {
  for (int i = 0; i < N; i++) {
    b[i] = 1.0f / (1.0f + expf(-a[i]));
  }
}

void Test(int N) {
  auto* a = new float[N];
  auto* b = new float[N];
  float* a_d;
  float* b_d;
  float* b_h;
  CUDA_CHECK(cudaMalloc(&a_d, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b_d, N * sizeof(float)));
  b_h = new float[N];

  int repeat = 10;
  for (int i = 0; i < repeat; i++) {
    std::cout << "N: " << N << std::endl;
    for (int j = 0; j < N; j++) {
      a[j] = rand() / static_cast<float>(RAND_MAX);
    }
    int block_size = 512;
    int grid_size = (N + block_size - 1) / block_size;
    auto start = GET_US();
    SigmoidCpu(a, b, N);
    auto end = GET_US();
    std::cout << "SigmoidCpu time: " << end - start << " us" << std::endl;

    start = GET_US();
    CUDA_CHECK(cudaMemcpy(a_d, a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(b_d, 0, N * sizeof(float)));
    Sigmoid<<<grid_size, block_size>>>(a_d, b_d, N);
    CUDA_CHECK(cudaMemcpy(b_h, b_d, N * sizeof(float), cudaMemcpyDeviceToHost));
    end = GET_US();
    std::cout << "Sigmoid time: " << end - start << " us" << std::endl;
    for (int j = 0; j < N; j++) {
      if (fabs(b[j] - b_h[j]) > 1e-5) {
        std::cout << "Error: " << b[j] << " " << b_h[j] << std::endl;
        break;
      }
    }

    start = GET_US();
    CUDA_CHECK(cudaMemcpy(a_d, a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(b_d, 0, N * sizeof(float)));
    block_size /= 4;
    grid_size = (N + block_size - 1) / block_size;
    SigmoidFloat4<<<grid_size, block_size>>>(a_d, b_d, N);
    CUDA_CHECK(cudaMemcpy(b_h, b_d, N * sizeof(float), cudaMemcpyDeviceToHost));
    end = GET_US();
    std::cout << "SigmoidFloat4 time: " << end - start << " us" << std::endl;
    for (int j = 0; j < N; j++) {
      if (fabs(b[j] - b_h[j]) > 1e-5) {
        std::cout << "Error: " << b[j] << " " << b_h[j] << std::endl;
        break;
      }
    }

    std::cout << "--------" << std::endl;
  }

  delete b_h;
  CUDA_CHECK(cudaFree(a_d));
  CUDA_CHECK(cudaFree(b_d));
  delete[] b;
  delete[] a;
}

int main() {
  InitDevice(0);
  Test(1 << 20);
  return 0;
}
