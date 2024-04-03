// 通过warp tile实现reduce sum.

#include <cuda_runtime.h>

#include <cassert>
#include <chrono>
#include <iostream>

#include "cuda_common.h"

#define GET_TIME_US() \
  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count()

#define WARP_SIZE 32
#define BLOCK_DIM_X 512
#define GRID_DIM_X 512

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

template <typename T>
__global__ void ReduceSumByWarpTile(const T* __restrict__ input, T* __restrict__ output, const int N) {
  __shared__ T shared[BLOCK_DIM_X / WARP_SIZE];
  T sum = T(0);
  // printf("blockDim.x: %d, gridDim.x: %d, i: %d, N: %d\n", blockDim.x, gridDim.x, blockIdx.x * blockDim.x +
  // threadIdx.x, N);
  sum = input[blockIdx.x * blockDim.x + threadIdx.x];
  sum = BlockReduceSum(sum, shared);

  // Merge all block results
  if (threadIdx.x == 0) {
    // printf("blockIdx.x: %d, sum: %f\n", blockIdx.x, sum);
    atomicAdd(output, sum);
  }
}

// 仅能用于float
template <typename T = float>
__global__ void ReduceSumByWarpTileFloat4(T* __restrict__ input, T* __restrict__ output, const int N) {
  __shared__ T shared[BLOCK_DIM_X / WARP_SIZE];
  auto idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  float r[4];
  FLOAT4(r[0]) = FLOAT4(input[idx]);
  T sum = T(0);
  sum = idx < N ? (r[0] + r[1] + r[2] + r[3]) : T(0);
  sum = BlockReduceSum(sum, shared);

  // Merge all block results
  if (threadIdx.x == 0) {
    // printf("blockIdx.x: %d, sum: %f\n", blockIdx.x, sum);
    atomicAdd(output, sum);
  }
}

template <typename T>
__global__ void ReduceSumByBucket(const T* __restrict__ input, T* __restrict__ output, const int N) {
  auto thread_num = gridDim.x * blockDim.x;
  auto bucket_size = (N + thread_num - 1) / thread_num;
  auto begin = (blockIdx.x * blockDim.x + threadIdx.x) * bucket_size;
  auto end = min(begin + bucket_size, N);
  // printf("blockIdx.x: %d, threadIdx.x: %d, begin: %d, end: %d\n", blockIdx.x, threadIdx.x, begin, end);

  T sum = T(0);
  for (int i = begin; i < end; i++) {
    sum += input[i];
  }

  // Merge all buckets results
  atomicAdd(output, sum);
}

template <typename T>
__global__ void ReduceSumByAtomic(const T* __restrict__ input, T* __restrict__ output, const int N) {
  auto thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_idx < N) {
    atomicAdd(output, input[thread_idx]);
  }
}

template <typename T>
void TestReduceSum(const long long N) {
  T* input = (T*)malloc(N * sizeof(T));
  T* output = (T*)malloc(1 * sizeof(T));
  T* d_input;
  T* d_output;
  CHECK(cudaMalloc(&d_input, N * sizeof(T)));
  CHECK(cudaMalloc(&d_output, 1 * sizeof(T)));

  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < N; j++) {
      input[j] = rand() % 10;
      // input[j] = T(1);
    }

    auto cpu_begin_us = GET_TIME_US();
    T sum = 0;
    for (int j = 0; j < N; j++) {
      sum += input[j];
    }
    auto cpu_end_us = GET_TIME_US();
    std::cout << "cpu ReduceSum: " << sum << ", time: " << cpu_end_us - cpu_begin_us << " us" << std::endl;

    auto gpu_begin_us = GET_TIME_US();
    CHECK(cudaMemcpy(d_input, input, N * sizeof(T), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_output, 0, 1 * sizeof(T)));
    dim3 block(BLOCK_DIM_X);
    dim3 grid((N + block.x - 1) / block.x);
    ReduceSumByWarpTile<<<grid, block>>>(d_input, d_output, N);
    CHECK(cudaMemcpy(output, d_output, 1 * sizeof(T), cudaMemcpyDeviceToHost));
    auto gpu_end_us = GET_TIME_US();
    if (abs(output[0] - sum) > 1e-3) {
      std::cout << "ReduceSumByWarpTile error, " << output[0] << ", " << sum << std::endl;
      exit(1);
    }
    std::cout << "gpu ReduceSumByWarpTile: " << output[0] << ", time: " << gpu_end_us - gpu_begin_us << " us"
              << std::endl;

    gpu_begin_us = GET_TIME_US();
    CHECK(cudaMemcpy(d_input, input, N * sizeof(T), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_output, 0, 1 * sizeof(T)));
    block = dim3(BLOCK_DIM_X / 4);
    grid = dim3((N + block.x - 1) / block.x);
    ReduceSumByWarpTileFloat4<<<grid, block>>>(d_input, d_output, N);
    CHECK(cudaMemcpy(output, d_output, 1 * sizeof(T), cudaMemcpyDeviceToHost));
    gpu_end_us = GET_TIME_US();
    if (abs(output[0] - sum) > 1e-3) {
      std::cout << "ReduceSumByWarpTileFloat4 error, " << output[0] << ", " << sum << std::endl;
      exit(1);
    }
    std::cout << "gpu ReduceSumByWarpTileFloat4: " << output[0] << ", time: " << gpu_end_us - gpu_begin_us << " us"
              << std::endl;

    gpu_begin_us = GET_TIME_US();
    CHECK(cudaMemcpy(d_input, input, N * sizeof(T), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_output, 0, 1 * sizeof(T)));
    ReduceSumByBucket<<<GRID_DIM_X, BLOCK_DIM_X>>>(d_input, d_output, N);
    CHECK(cudaMemcpy(output, d_output, 1 * sizeof(T), cudaMemcpyDeviceToHost));
    gpu_end_us = GET_TIME_US();
    if (abs(output[0] - sum) > 1e-3) {
      std::cout << "ReduceSumByBucket error, " << output[0] << ", " << sum << std::endl;
      exit(1);
    }
    std::cout << "gpu ReduceSumByBucket: " << output[0] << ", time: " << gpu_end_us - gpu_begin_us << " us"
              << std::endl;

    gpu_begin_us = GET_TIME_US();
    CHECK(cudaMemcpy(d_input, input, N * sizeof(T), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_output, 0, 1 * sizeof(T)));
    ReduceSumByAtomic<<<grid, block>>>(d_input, d_output, N);
    CHECK(cudaMemcpy(output, d_output, 1 * sizeof(T), cudaMemcpyDeviceToHost));
    gpu_end_us = GET_TIME_US();
    if (abs(output[0] - sum) > 1e-3) {
      std::cout << "ReduceSumByAtomic error, " << output[0] << ", " << sum << std::endl;
      exit(1);
    }
    std::cout << "gpu ReduceSumByAtomic: " << output[0] << ", time: " << gpu_end_us - gpu_begin_us << " us"
              << std::endl;

    std::cout << "--------" << std::endl;
  }
  CHECK(cudaFree(d_input));
  CHECK(cudaFree(d_output));
  free(input);
  free(output);
}

int main() {
  InitDevice(0);

  TestReduceSum<float>(1 << 20);
  return 0;
}