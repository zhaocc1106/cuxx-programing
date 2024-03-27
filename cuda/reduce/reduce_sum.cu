// 通过warp tile实现reduce sum.

#include <cuda_runtime.h>

#include <cassert>
#include <chrono>
#include <iostream>

#include "cuda_common.h"

#define GET_TIME_US() \
  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count()

#define WARP_SIZE 32
#define BLOCK_DIM_X 1024

template <typename T>
__inline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset, warpSize);
  }
  return val;
}

template <typename T>
__inline__ __device__ T BlockReduceSum(T val, T* shared) {
  const int laneid = threadIdx.x % warpSize;
  const int warpid = threadIdx.x / warpSize;
  val = WarpReduceSum(val);
  __syncthreads();
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

template <typename T>
__global__ void ReduceSumV1(const T* __restrict__ input, T* __restrict__ output, const int N) {
  __shared__ T shared[BLOCK_DIM_X / WARP_SIZE];
  T sum = T(0);
  // printf("blockDim.x: %d, gridDim.x: %d, i: %d, N: %d\n", blockDim.x, gridDim.x, blockIdx.x * blockDim.x +
  // threadIdx.x, N);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    sum += input[i];
  }
  sum = BlockReduceSum(sum, shared);
  if (threadIdx.x == 0) {
    output[blockIdx.x] = sum;
  }
}

template <typename T>
__global__ void ReduceSumV2(const T* __restrict__ input, T* __restrict__ output, const int N) {
  auto bucket_size = (N + blockDim.x - 1) / blockDim.x;
  __shared__ T shared[BLOCK_DIM_X];
  auto begin = threadIdx.x * bucket_size;
  auto end = min(begin + bucket_size, N);

  T sum = T(0);
  for (int i = begin; i < end; i++) {
    sum += input[i];
  }
  shared[threadIdx.x] = sum;
  __syncthreads();

  if (threadIdx.x == 0) {
    T block_sum = T(0);
    for (int i = 0; i < blockDim.x; i++) {
      block_sum += shared[i];
    }
    output[0] = block_sum;
  }
}

template <typename T>
void TestReduceSum(const long long N) {
  T* input = (T*)malloc(N * sizeof(T));
  T* output = (T*)malloc(1 * sizeof(T));
  T* d_input;
  T* d_output;
  CHECK(cudaMalloc(&d_input, N * sizeof(T)));
  CHECK(cudaMalloc(&d_output, N * sizeof(T)));

  for (int j = 0; j < N; j++) {
    input[j] = rand() / static_cast<T>(RAND_MAX);
  }
  for (int i = 0; i < 10; i++) {
    auto cpu_begin_us = GET_TIME_US();
    T sum = 0;
    for (int j = 0; j < N; j++) {
      sum += input[j];
    }
    auto cpu_end_us = GET_TIME_US();
    std::cout << "cpu ReduceSum: " << sum << ", time: " << cpu_end_us - cpu_begin_us << " us" << std::endl;

    auto gpu_begin_us = GET_TIME_US();
    CHECK(cudaMemcpy(d_input, input, N * sizeof(T), cudaMemcpyHostToDevice));
    ReduceSumV1<<<1, BLOCK_DIM_X>>>(d_input, d_output, N);
    CHECK(cudaMemcpy(output, d_output, 1 * sizeof(T), cudaMemcpyDeviceToHost));
    auto gpu_end_us = GET_TIME_US();
    if (abs(output[0] - sum) > 1e-3) {
      std::cout << "ReduceSumV1 error, " << output[0] << ", " << sum << std::endl;
      exit(1);
    }
    std::cout << "gpu ReduceSumV1: " << output[0] << ", time: " << gpu_end_us - gpu_begin_us << " us" << std::endl;

    gpu_begin_us = GET_TIME_US();
    CHECK(cudaMemcpy(d_input, input, N * sizeof(T), cudaMemcpyHostToDevice));
    ReduceSumV2<<<1, 256>>>(d_input, d_output, N);
    CHECK(cudaMemcpy(output, d_output, 1 * sizeof(T), cudaMemcpyDeviceToHost));
    gpu_end_us = GET_TIME_US();
    if (abs(output[0] - sum) > 1e-3) {
      std::cout << "ReduceSumV2 error" << std::endl;
      exit(1);
    }
    std::cout << "gpu ReduceSumV2: " << output[0] << ", time: " << gpu_end_us - gpu_begin_us << " us" << std::endl;

    std::cout << std::endl;
  }
  CHECK(cudaFree(d_input));
  CHECK(cudaFree(d_output));
  free(input);
  free(output);
}

int main() {
  InitDevice(0);

  TestReduceSum<double>(1 << 20);
  return 0;
}