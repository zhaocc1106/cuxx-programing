//
// 通过shared mem + 分块矩阵乘法实现更高性能的sgemm
//

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdlib>
#include <iostream>

#include "cuda_common.h"

#define OFFSET(row, col, ld) ((col) + (row) * (ld))
#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

void CpuSgemm(float* a, float* b, float* c, const int M, const int N, const int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0;
      for (int k = 0; k < K; k++) {
        sum += a[OFFSET(i, k, K)] * b[OFFSET(k, j, N)];
      }
      c[OFFSET(i, j, N)] = sum;
    }
  }
}

// 通过shared mem实现分块矩阵乘法，每次核函数调用计算目标矩阵的一个分块
__global__ void TileSgemmBySharedMem(
  float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, const int M, const int N, const int K) {
  const int BM = 128; // 分块行数
  const int BN = 128; // 分块列数
  const int BK = 8; // 考虑shared mem容量受限，K维上每次读取BK个元素，则外部循环需要进行K/BK次

  const int TM = 8; // block每一个线程处理块的行数
  const int TN = 8; // block每一个线程处理块的列数
  // 则block的线程数dim3(BM/TM, BN/TN)

  const int bx = blockIdx.x;            // block的列索引
  const int by = blockIdx.y;            // block的行索引
  const int tx = threadIdx.x;           // block内的线程列索引
  const int ty = threadIdx.y;           // block内的线程行索引
  const int tid = ty * blockDim.x + tx; // block内的线程索引

  __shared__ float s_a[BM][BK]; // 使用shared mem存储分块a[BM, BK]
  __shared__ float s_b[BK][BN]; // 使用shared mem存储分块b[BK, BN]

  float r_c[TM][TN] = {0}; // 使用寄存器存储分块c[TM, TN]，每个线程负责计算分块c[TM, TN]的元素

  // block内的线程数 = (BM * BN) / (TM * TN) = 256
  // 对于每一个block，需要进行K/BK次循环，每次循环读取一个分块a和分块b
  // 分块a、b总数据量为BM * BK == BK * BN = 128 * 8 = 1024，不超过shared mem容量
  // 则每个分块a[BM, BK]对于每个thread需要读取BM * BK / block内线程数 = 1024 / 256 = 4个元素，即每个thread读取4个float从
  // global mem到shared mem，使用FLOAT4*进行搬运，每次搬运4个float
  int load_a_smem_m = tid >> 1;        // 该线程搬运的分块a[BM, BK]的行索引
  int load_a_smem_k = (tid & 1) << 2;  // 该线程搬运的分块a[BM, BK]的列索引，(tid % 2) == 0 ? 0 : 4
  int load_b_smem_k = tid >> 5;        // 该线程搬运的分块b[BK, BN]的行索引， tid / 32
  int load_b_smem_n = (tid & 31) << 2; // 该线程搬运的分块b[BK, BN]的列索引，(tid % 32) * 4

  int load_a_gmem_m = by * BM + load_a_smem_m; // 该线程搬运的分块a[BM, BK]的行索引对应到global mem的行索引
  int load_b_gmem_n = bx * BN + load_b_smem_n; // 该线程搬运的分块b[BK, BN]的列索引对应到global mem的列索引

#pragma unroll
  for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
    // 搬运分块a[BM, BK]到shared mem
    int load_a_gmem_k = bk * BK + load_a_smem_k; // 该线程搬运的分块a[BM, BK]的列索引对应到global mem的列索引
    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);
    // 搬运分块b[BK, BN]到shared mem
    int load_b_gmem_k = bk * BK + load_b_smem_k; // 该线程搬运的分块b[BK, BN]的行索引对应到global mem的行索引
    int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
    FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);

    // 等待block内所有线程搬运完毕
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; k++) {
#pragma unroll
      for (int m = 0; m < TM; m++) { // 对于分块c[TM, TN]的每一行
#pragma unroll
        for (int n = 0; n < TN; n++) { // 对于分块c[TM, TN]的每一列
          // 计算c[TM, TN]的[m][n]元素
          int s_a_m = ty * TM + m; // 用于计算[m][n]元素对应到shared mem的s_a的行索引
          int s_b_n = tx * TN + n; // 用于计算[m][n]元素对应到shared mem的s_b的列索引
          r_c[m][n] += s_a[s_a_m][k] * s_b[k][s_b_n];
        }
      }
    }

    // 等待block内所有线程计算完毕
    __syncthreads();
  }

// 将分块c[TM, TN]的结果写回到global mem
#pragma unroll
  for (int m = 0; m < TM; m++) {
    int store_c_gmem_m = by * BM + ty * TM + m; // 该线程计算的分块c[TM, TN]的行索引对应到global mem的行索引
                                                // #pragma unroll
    for (int n = 0; n < TN; n += 4) {
      int store_c_gmem_n = bx * BN + tx * TN + n; // 该线程计算的分块c[TM, TN]的列索引对应到global mem的列索引
      int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
      FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[m][n]);
    }
  }
}

// 同cpu版本作比较，测试准确性
float TestError(void (*GpuSgemm)(float* a, float* b, float* c, const int M, const int N, const int K),
                const dim3& grid_dim,
                const dim3& block_dim,
                const int M,
                const int N,
                const int K) {
  int a_size = M * K * sizeof(float);
  int b_size = K * N * sizeof(float);
  int c_size = M * N * sizeof(float);

  // 生成随机值矩阵
  float* a = (float*)malloc(a_size);
  float* b = (float*)malloc(K * N * sizeof(float));
  float* c_cpu = (float*)malloc(c_size);
  // float *a, *b, *c_cpu;
  // CHECK(cudaMallocHost(&a, a_size));
  // CHECK(cudaMallocHost(&b, b_size));
  // CHECK(cudaMallocHost(&c_cpu, c_size));
  for (int i = 0; i < M * K; i++) {
    a[i] = rand() / static_cast<float>(RAND_MAX);
  }
  // PrintMatrix(a, M, K, "a");
  for (int i = 0; i < K * N; i++) {
    b[i] = rand() / static_cast<float>(RAND_MAX);
  }
  // PrintMatrix(b, K, N, "b");
  // cpu版本计算
  CpuSgemm(a, b, c_cpu, M, N, K);
  // PrintMatrix(c_cpu, M, N, "c_cpu");

  // gpu版本计算
  float* d_a;
  float* d_b;
  float* d_c;
  CHECK(cudaMalloc(&d_a, a_size));
  CHECK(cudaMalloc(&d_b, b_size));
  CHECK(cudaMalloc(&d_c, c_size));
  CHECK(cudaMemcpy(d_a, a, a_size, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b, b, b_size, cudaMemcpyHostToDevice));
  GpuSgemm<<<grid_dim, block_dim>>>(d_a, d_b, d_c, M, N, K);
  float* c_gpu = (float*)malloc(c_size);
  // float* c_gpu;
  // CHECK(cudaMallocHost(&c_gpu, c_size));
  CHECK(cudaMemcpy(c_gpu, d_c, c_size, cudaMemcpyDeviceToHost));
  CHECK(cudaDeviceSynchronize());
  // PrintMatrix(c_gpu, M, N, "c_gpu");

  // 比较cpu和gpu版本计算结果
  float max_diff = 0;
  for (int i = 0; i < M * N; i++) {
    // if (std::abs(c_cpu[i] - c_gpu[i]) > 1e-3) {
    //   printf("Error: c_cpu[%d] = %f, c_gpu[%d] = %f\n", i, c_cpu[i], i, c_gpu[i]);
    //   // exit(1);
    // }
    max_diff = std::max(max_diff, std::abs(c_cpu[i] - c_gpu[i]));
  }

  free(a);
  free(b);
  free(c_cpu);
  free(c_gpu);
  // CHECK(cudaFreeHost(a));
  // CHECK(cudaFreeHost(b));
  // CHECK(cudaFreeHost(c_cpu));
  // CHECK(cudaFreeHost(c_gpu));
  CHECK(cudaFree(d_a));
  CHECK(cudaFree(d_b));
  CHECK(cudaFree(d_c));
  return max_diff;
}

// 测试性能
double TestPerformance(void (*GpuSgemm)(float* a, float* b, float* c, const int M, const int N, const int K),
                       const dim3& grid_dim,
                       const dim3& block_dim,
                       const int M,
                       const int N,
                       const int K) {
  int a_size = M * K * sizeof(float);
  int b_size = K * N * sizeof(float);
  int c_size = M * N * sizeof(float);

  // 生成随机值矩阵
  // 生成随机值矩阵
  float* a = (float*)malloc(a_size);
  float* b = (float*)malloc(K * N * sizeof(float));
  float* c_cpu = (float*)malloc(c_size);
  // float *a, *b;
  // CHECK(cudaMallocHost(&a, a_size));
  // CHECK(cudaMallocHost(&b, b_size));
  for (int i = 0; i < M * K; i++) {
    a[i] = rand() / static_cast<float>(RAND_MAX);
  }
  for (int i = 0; i < K * N; i++) {
    b[i] = rand() / static_cast<float>(RAND_MAX);
  }

  // gpu版本计算
  float* d_a;
  float* d_b;
  float* d_c;
  CHECK(cudaMalloc(&d_a, a_size));
  CHECK(cudaMalloc(&d_b, b_size));
  CHECK(cudaMalloc(&d_c, c_size));
  CHECK(cudaMemcpy(d_a, a, a_size, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b, b, b_size, cudaMemcpyHostToDevice));

  cudaEvent_t start, end;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&end));
  CHECK(cudaEventRecord(start));

  GpuSgemm<<<grid_dim, block_dim>>>(d_a, d_b, d_c, M, N, K);

  CHECK(cudaEventRecord(end));
  CHECK(cudaEventSynchronize(end));

  float msec = 0;
  CHECK(cudaEventElapsedTime(&msec, start, end));
  // printf("msec: %f\n", msec);
  double sec = msec / 1000.0;

  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(end));

  CHECK(cudaDeviceSynchronize());

  free(a);
  free(b);
  // CHECK(cudaFreeHost(a));
  // CHECK(cudaFreeHost(b));
  CHECK(cudaFree(d_a));
  CHECK(cudaFree(d_b));
  CHECK(cudaFree(d_c));

  return sec;
}

int main() {
  // 初始化cuda设备
  InitDevice(0);

  const int M_list[15] = {128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
  const int N_list[15] = {128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
  const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};
  const int BM = 128; // 分块行数
  const int BN = 128; // 分块列数
  const int TM = 8;   // block每一个线程处理块的行数
  const int TN = 8;   // block每一个线程处理块的列数

  // 测试TileSgemmBySharedMem准确性
  for (int i = 0; i < 5; i++) {
    dim3 block_dim(BN / TN, BM / TM);
    dim3 grid_dim((N_list[i] + BN - 1) / BN, (M_list[i] + BM - 1) / BM);
    printf("grid_dim: (%d, %d), block_dim: (%d, %d)\n", grid_dim.x, grid_dim.y, block_dim.x, block_dim.y);
    float max_diff = TestError(TileSgemmBySharedMem, grid_dim, block_dim, M_list[i], N_list[i], K_list[i]);
    printf("TileSgemmBySharedMem Accuracy Test %d passed, M = %d, N = %d, K = %d, max_diff: %f.\n", i, M_list[i],
           N_list[i], K_list[i], max_diff);
  }

  // 测试TileSgemmBySharedMem性能和算力
  for (int i = 0; i < 14; i++) {
    double total_sec = 0.0;
    double max_sec = 0;
    double min_sec = DBL_MAX;
    int repeat = 5;
    for (int j = 0; j < repeat; j++) {
      dim3 block_dim(BN / TN, BM / TM);
      dim3 grid_dim((N_list[i] + BN - 1) / BN, (M_list[i] + BM - 1) / BM);
      double sec = TestPerformance(TileSgemmBySharedMem, grid_dim, block_dim, M_list[i], N_list[i], K_list[i]);
      total_sec += sec;
      max_sec = std::max(max_sec, sec);
      min_sec = std::min(min_sec, sec);
    }
    double avg_sec = total_sec / repeat;
    double gflops = double(2) * M_list[i] * N_list[i] * K_list[i] / avg_sec / 1024 / 1024 / 1024;
    printf(
      "TileSgemmBySharedMem Performance Test %d, M = %d, N = %d, K = %d, time = %.6f %.6f %.6f sec, gflops = %.6f "
      "GFLOPS\n",
      i, M_list[i], N_list[i], K_list[i], min_sec, avg_sec, max_sec, gflops);
  }

  return 0;
}