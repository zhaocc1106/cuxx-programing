//
// Created by zcc-mac on 2024/3/6.
//

#ifndef GEMM__CUDA_COMMON_H_
#define GEMM__CUDA_COMMON_H_

#include <sys/time.h>

#include <ctime>

#define CUDA_CHECK(call)                                               \
  {                                                                    \
    const cudaError_t error = call;                                    \
    if (error != cudaSuccess) {                                        \
      printf("ERROR: %s:%d,", __FILE__, __LINE__);                     \
      printf("code:%d,reason:%s\n", error, cudaGetErrorString(error)); \
      exit(1);                                                         \
    }                                                                  \
  }

#define CUBLAS_CHECK(call)                                            \
  {                                                                   \
    cublasStatus_t error = call;                                      \
    if (error != CUBLAS_STATUS_SUCCESS) {                             \
      printf("ERROR: %s:%d, status:%d\n", __FILE__, __LINE__, error); \
      exit(1);                                                        \
    }                                                                 \
  }

void PrintMatrix(float* matrix, int row, int col, const std::string& name) {
  printf("%s:\n", name.c_str());
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      printf("%f ", matrix[i * col + j]);
    }
    printf("\n");
  }
}

void InitDevice(int devNum) {
  int dev = devNum;
  cudaDeviceProp deviceProp{};
  CUDA_CHECK((cudaGetDeviceProperties(&deviceProp, dev)));
  printf(
    "Using device %d, name: %s, warpSize: %d, concurrentKernels: %d, "
    "totalConstMem: %zu, totalGlobalMem:"
    " %zu, maxThreadsPerMultiProcessor: "
    "%d, maxThreadsPerBlock %d, "
    "globalL1CacheSupported: %d, localL1CacheSupported: %d,\nl2CacheSize: "
    "%d, asyncEngineCount: %d, "
    "unifiedAddressing: %d.\n"
    "multiProcessorCount: %d, compute capability: %d.%d, "
    "regsPerMultiprocessor: %d, regsPerBlock:%d"
    ", sharedMemPerBlock: %zu, sharedMemPerMultiprocessor: %zu, "
    "computeMode: %d.\n",
    dev, deviceProp.name, deviceProp.warpSize, deviceProp.concurrentKernels, deviceProp.totalConstMem,
    deviceProp.totalGlobalMem, deviceProp.maxThreadsPerMultiProcessor, deviceProp.maxThreadsPerBlock,
    deviceProp.globalL1CacheSupported, deviceProp.localL1CacheSupported, deviceProp.l2CacheSize,
    deviceProp.asyncEngineCount, deviceProp.unifiedAddressing, deviceProp.multiProcessorCount, deviceProp.major,
    deviceProp.minor, deviceProp.regsPerMultiprocessor, deviceProp.regsPerBlock, deviceProp.sharedMemPerBlock,
    deviceProp.sharedMemPerMultiprocessor, deviceProp.computeMode);
  CUDA_CHECK(cudaSetDevice(dev));
}

#endif // GEMM__CUDA_COMMON_H_
