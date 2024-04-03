// 通过warp tile实现reduce max.

#include <cuda_runtime.h>

#include <cassert>
#include <cfloat>
#include <chrono>
#include <iostream>

#include "cuda_common.h"

#define GET_TIME_US() \
  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count()

#define WARP_SIZE 32
#define BLOCK_DIM_X 1024

template <typename T>
__inline__ __device__ T WarpReduceMax(T val) {
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val = fmax(val, __shfl_down_sync(0xffffffff, val, offset, warpSize));
  }
  return val;
}

template <typename T>
__inline__ __device__ T BlockReduceMax(T val, T* shared) {
  const int laneid = threadIdx.x % warpSize;
  const int warpid = threadIdx.x / warpSize;
  val = WarpReduceMax(val);
  // __syncthreads();
  if (laneid == 0) {
    shared[warpid] = val;
  }
  __syncthreads();
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[laneid] : T(-DBL_MAX);
  if (warpid == 0) {
    val = WarpReduceMax(val);
  }
  return val;
}

template <typename T>
__global__ void ReduceMaxV1(const T* __restrict__ input, T* __restrict__ output, const int N) {
  __shared__ T shared[BLOCK_DIM_X / WARP_SIZE];
  T max = T(-DBL_MAX);
  // printf("blockDim.x: %d, gridDim.x: %d, i: %d, N: %d\n", blockDim.x, gridDim.x, blockIdx.x * blockDim.x +
  // threadIdx.x, N);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    max = fmax(max, input[i]);
  }
  max = BlockReduceMax(max, shared);
  if (threadIdx.x == 0) {
    output[blockIdx.x] = max;
  }
}

template <typename T>
__global__ void ReduceMaxV2(const T* __restrict__ input, T* __restrict__ output, const int N) {
  auto bucket_size = (N + blockDim.x - 1) / blockDim.x;
  __shared__ T shared[BLOCK_DIM_X];
  auto begin = threadIdx.x * bucket_size;
  auto end = min(begin + bucket_size, N);

  T max = T(-DBL_MAX);
  for (int i = begin; i < end; i++) {
    max = fmax(max, input[i]);
  }
  shared[threadIdx.x] = max;
  __syncthreads();

  if (threadIdx.x == 0) {
    T block_max = T(-DBL_MAX);
    for (int i = 0; i < blockDim.x; i++) {
      block_max = fmax(block_max, shared[i]);
    }
    output[0] = block_max;
  }
}

template <typename T>
void TestReduceMax(const long long N) {
  T* input = (T*)malloc(N * sizeof(T));
  T* output = (T*)malloc(1 * sizeof(T));
  T* d_input;
  T* d_output;
  CHECK(cudaMalloc(&d_input, N * sizeof(T)));
  CHECK(cudaMalloc(&d_output, 1 * sizeof(T)));

  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < N; j++) {
      input[j] = T(rand());
    }

    auto cpu_begin_us = GET_TIME_US();
    T max = 0;
    for (int j = 0; j < N; j++) {
      max = std::max(max, input[j]);
    }
    auto cpu_end_us = GET_TIME_US();
    std::cout << "cpu ReduceMax: " << max << ", time: " << cpu_end_us - cpu_begin_us << " us" << std::endl;

    auto gpu_begin_us = GET_TIME_US();
    CHECK(cudaMemcpy(d_input, input, N * sizeof(T), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_output, 0, 1 * sizeof(T)));
    ReduceMaxV1<<<1, BLOCK_DIM_X>>>(d_input, d_output, N);
    CHECK(cudaMemcpy(output, d_output, 1 * sizeof(T), cudaMemcpyDeviceToHost));
    auto gpu_end_us = GET_TIME_US();
    if (abs(output[0] - max) > 1e-3) {
      std::cout << "ReduceMaxV1 error, " << output[0] << ", " << max << std::endl;
      exit(1);
    }
    std::cout << "gpu ReduceMaxV1: " << output[0] << ", time: " << gpu_end_us - gpu_begin_us << " us" << std::endl;

    gpu_begin_us = GET_TIME_US();
    CHECK(cudaMemcpy(d_input, input, N * sizeof(T), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_output, 0, 1 * sizeof(T)));
    ReduceMaxV2<<<1, 256>>>(d_input, d_output, N);
    CHECK(cudaMemcpy(output, d_output, 1 * sizeof(T), cudaMemcpyDeviceToHost));
    gpu_end_us = GET_TIME_US();
    if (abs(output[0] - max) > 1e-3) {
      std::cout << "ReduceMaxV2 error" << std::endl;
      exit(1);
    }
    std::cout << "gpu ReduceMaxV2: " << output[0] << ", time: " << gpu_end_us - gpu_begin_us << " us" << std::endl;

    std::cout << "--------" << std::endl;
  }
  CHECK(cudaFree(d_input));
  CHECK(cudaFree(d_output));
  free(input);
  free(output);
}

int main() {
  InitDevice(0);

  TestReduceMax<double>(1 << 20);
  return 0;
}