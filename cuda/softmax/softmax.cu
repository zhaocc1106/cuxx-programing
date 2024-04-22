// CUDA实现softmax算子
#include <cuda_runtime.h>

#include <chrono>
#include <iostream>

#include "cuda_common.h"

#define GET_TIME_US() \
  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count()

#define WARP_SIZE 32
#define BLOCK_DIM_X 512

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

template <typename T>
__inline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset, warpSize);
    // __shfl_xor_sync也可以完成相同动作
    // val += __shfl_xor_sync(0xffffffff, val, offset, warpSize);
  }
  return val;
}

template <typename T>
__inline__ __device__ T BlockReduceSum(T val, T* shared) {
  const int laneid = threadIdx.x % warpSize;
  const int warpid = threadIdx.x / warpSize;
  val = WarpReduceSum(val);
  // __syncthreads();
  if (laneid == 0) {
    shared[warpid] = val;
  }
  __syncthreads();
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[laneid] : T(0);
  if (warpid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}

__global__ void Softmax(float* __restrict__ a, float* __restrict__ b, float* __restrict__ total, int N) {
  __shared__ float shared[BLOCK_DIM_X / WARP_SIZE];
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }

  float exp_val = 0.0f;
  exp_val = expf(a[idx]);
  float sum = BlockReduceSum(exp_val, shared);
  if (threadIdx.x == 0) {
    atomicAdd(total, sum);
  }

  // 同步所有线程内存
  __threadfence();
  // printf("idx: %d, total: %f\n", idx, *total);
  if (idx < N) {
    b[idx] = exp_val / (*total);
  }
}

__global__ void SoftmaxFloat4(float* __restrict__ a, float* __restrict__ b, float* __restrict__ total, int N) {
  __shared__ float shared[BLOCK_DIM_X / WARP_SIZE];
  auto idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx >= N) {
    return;
  }

  float4 exp_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  if (idx < N) {
    exp_val.x = expf(a[idx]);
  }
  if (idx + 1 < N) {
    exp_val.y = expf(a[idx + 1]);
  }
  if (idx + 2 < N) {
    exp_val.z = expf(a[idx + 2]);
  }
  if (idx + 3 < N) {
    exp_val.w = expf(a[idx + 3]);
  }
  float sum = BlockReduceSum(exp_val.x + exp_val.y + exp_val.z + exp_val.w, shared);
  if (threadIdx.x == 0) {
    atomicAdd(total, sum);
  }

  // 同步所有线程内存
  __threadfence();
  if (idx < N) {
    b[idx] = exp_val.x / (*total);
  }
  if (idx + 1 < N) {
    b[idx + 1] = exp_val.y / (*total);
  }
  if (idx + 2 < N) {
    b[idx + 2] = exp_val.z / (*total);
  }
  if (idx + 3 < N) {
    b[idx + 3] = exp_val.w / (*total);
  }
}

void SoftmaxCPU(float* a, float* b, float* total, int N) {
  float sum = 0.0f;
  for (int i = 0; i < N; i++) {
    sum += expf(a[i]);
  }
  *total = sum;
  for (int i = 0; i < N; i++) {
    b[i] = expf(a[i]) / sum;
  }
}

void Test(int N) {
  auto* a = new float[N];
  auto* b = new float[N];
  auto* total = new float[1];

  float* a_d;
  float* b_d;
  float* total_d;
  auto* b_h = new float[N];
  float* total_h = new float[1];
  CUDA_CHECK(cudaMalloc(&a_d, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b_d, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&total_d, sizeof(float)));

  int repeat = 10;
  for (int i = 0; i < repeat; i++) {
    for (int j = 0; j < N; j++) {
      a[j] = rand();
    }
    std::cout << "N: " << N << std::endl;

    auto begin = GET_TIME_US();
    SoftmaxCPU(a, b, total, N);
    auto end = GET_TIME_US();
    std::cout << "SoftmaxCPU, time: " << end - begin << " us" << std::endl;

    begin = GET_TIME_US();
    CUDA_CHECK(cudaMemcpy(a_d, a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(b_d, 0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(total_d, 0, sizeof(float)));
    Softmax<<<(N + BLOCK_DIM_X - 1) / BLOCK_DIM_X, BLOCK_DIM_X>>>(a_d, b_d, total_d, N);
    CUDA_CHECK(cudaMemcpy(b_h, b_d, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(total_h, total_d, sizeof(float), cudaMemcpyDeviceToHost));
    end = GET_TIME_US();
    std::cout << "SoftmaxGPU, time: " << end - begin << " us" << std::endl;
    for (int j = 0; j < N; j++) {
      if (fabs(b[j] - b_h[j]) > 1e-5) {
        std::cout << "Error b[" << j << "]: " << b[j] << " " << b_h[j] << std::endl;
        break;
      }
    }
    // std::cout << "total: " << *total << " " << *total_h << std::endl;
    if (fabs(*total - *total_h) > 1e-3) {
      std::cout << "Error total: " << *total << " " << *total_h << std::endl;
      break;
    }

    begin = GET_TIME_US();
    CUDA_CHECK(cudaMemcpy(a_d, a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(b_d, 0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(total_d, 0, sizeof(float)));
    auto block_dim_x = BLOCK_DIM_X / 4;
    SoftmaxFloat4<<<(N + block_dim_x - 1) / block_dim_x, block_dim_x>>>(a_d, b_d, total_d, N);
    CUDA_CHECK(cudaMemcpy(b_h, b_d, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(total_h, total_d, sizeof(float), cudaMemcpyDeviceToHost));
    end = GET_TIME_US();
    std::cout << "SoftmaxFloat4GPU, time: " << end - begin << " us" << std::endl;
    for (int j = 0; j < N; j++) {
      if (fabs(b[j] - b_h[j]) > 1e-5) {
        std::cout << "Error b[" << j << "]: " << b[j] << " " << b_h[j] << std::endl;
        break;
      }
    }
    if (fabs(*total - *total_h) > 1e-3) {
      std::cout << "Error total: " << *total << " " << *total_h << std::endl;
      break;
    }
  }

  delete[] a;
  delete[] b;
  delete[] total;
  delete[] b_h;
  delete[] total_h;
  CUDA_CHECK(cudaFree(a_d));
  CUDA_CHECK(cudaFree(b_d));
  CUDA_CHECK(cudaFree(total_d));
}

int main() {
  Test(1 << 20);
  return 0;
}
