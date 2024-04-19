// 通过warp tile实现dot product并比较与cublas的性能
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <iostream>

#include "cuda_common.h"

#define GET_US() \
  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
#define BLOCK_DIM_X 512
#define WARP_SIZE 32

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

template <typename T>
__device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset, warpSize);
  }
  return val;
}

template <typename T>
__device__ T BlockReduceSum(T val, T* shared) {
  const int laneid = threadIdx.x % warpSize;
  const int warpid = threadIdx.x / warpSize;
  val = WarpReduceSum(val);
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
__global__ void DotProductWarpTile(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c, const int N) {
  __shared__ T shared[BLOCK_DIM_X / WARP_SIZE];
  T sum = T(0);
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    sum = a[idx] * b[idx];
  }

  sum = BlockReduceSum(sum, shared);

  if (threadIdx.x == 0) {
    atomicAdd(c, sum);
  }
}

__global__ void DotProductWarpTileFloat4(float* __restrict__ a,
                                         float* __restrict__ b,
                                         float* __restrict__ c,
                                         const int N) {
  __shared__ float shared[BLOCK_DIM_X / WARP_SIZE];
  float sum = 0;
  auto idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

  if (idx < N) {
    float a4[4];
    float b4[4];
    FLOAT4(a4) = FLOAT4(a[idx]);
    FLOAT4(b4) = FLOAT4(b[idx]);
    sum = a4[0] * b4[0] + a4[1] * b4[1] + a4[2] * b4[2] + a4[3] * b4[3];
  }

  sum = BlockReduceSum(sum, shared);

  if (threadIdx.x == 0) {
    atomicAdd(c, sum);
  }
}

template <typename T>
void DotProductCpu(const T* a, const T* b, T* c, const int N) {
  T sum = T(0);
  for (int i = 0; i < N; i++) {
    sum += a[i] * b[i];
  }
  *c = sum;
}

template <typename T, int N = 1024>
void Test() {
  T* a = new T[N];
  T* b = new T[N];
  T* c = new T[1];
  T* d_a;
  T* d_b;
  T* d_c;
  CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_c, sizeof(T)));

  const int repeat = 11;
  for (int i = 0; i < repeat; i++) {
    std::cout << "N=" << N << std::endl;

    for (int j = 0; j < N; j++) {
      a[j] = rand() % 10;
      b[j] = rand() % 10;
    }

    auto begin = GET_US();
    DotProductCpu(a, b, c, N);
    auto end = GET_US();
    std::cout << "DotProductCpu: " << *c << ", time: " << end - begin << " us" << std::endl;

    begin = GET_US();
    CUDA_CHECK(cudaMemcpy(d_a, a, N * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, N * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, sizeof(T)));
    DotProductWarpTile<<<N / BLOCK_DIM_X, BLOCK_DIM_X>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaMemcpy(c, d_c, sizeof(T), cudaMemcpyDeviceToHost));
    end = GET_US();
    std::cout << "DotProductWarpTile: " << *c << ", time: " << end - begin << " us" << std::endl;

    begin = GET_US();
    CUDA_CHECK(cudaMemcpy(d_a, a, N * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, N * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, sizeof(T)));
    DotProductWarpTileFloat4<<<N / BLOCK_DIM_X / 4, BLOCK_DIM_X>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaMemcpy(c, d_c, sizeof(T), cudaMemcpyDeviceToHost));
    end = GET_US();
    std::cout << "DotProductWarpTileFloat4: " << *c << ", time: " << end - begin << " us" << std::endl;

    // cublas
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    begin = GET_US();
    CUDA_CHECK(cudaMemcpy(d_a, a, N * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, N * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, sizeof(T)));
    CUBLAS_CHECK(cublasDotEx(handle, N, d_a, CUDA_R_32F, 1, d_b, CUDA_R_32F, 1, d_c, CUDA_R_32F, CUDA_R_32F));
    CUDA_CHECK(cudaMemcpy(c, d_c, sizeof(T), cudaMemcpyDeviceToHost));
    end = GET_US();
    std::cout << "DotProductCublas: " << *c << ", time: " << end - begin << " us" << std::endl;
    CUBLAS_CHECK(cublasDestroy(handle));

    std::cout << "--------" << std::endl;
  }

  delete[] a;
  delete[] b;
  delete[] c;
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
}

int main() {
  InitDevice(0);
  Test<float, 1 << 20>();
  return 0;
}
