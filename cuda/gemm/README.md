# 实现不同版本的gemm并比较性能

## 环境

1080Ti + CUDA10

## sgemm_naive

简单版本的sgemm。

```
NaiveSgemm Performance Test 10, M = 4096, N = 4096, K = 1024, time = 0.057050 0.057068 0.057085 sec, gflops = 560.731668 GFLOPS
NaiveSgemm Performance Test 11, M = 6144, N = 6144, K = 1024, time = 0.107072 0.114973 0.127611 sec, gflops = 626.235908 GFLOPS
NaiveSgemm Performance Test 12, M = 8192, N = 8192, K = 1024, time = 0.190706 0.190917 0.191142 sec, gflops = 670.448292 GFLOPS
NaiveSgemm Performance Test 13, M = 12288, N = 12288, K = 1024, time = 0.429226 0.429767 0.430233 sec, gflops = 670.130314 GFLOPS
NaiveSgemm Performance Test 14, M = 16384, N = 16384, K = 1024, time = 0.764678 0.766153 0.767095 sec, gflops = 668.274232 GFLOPS
```

## sgemm_tile_shm

通过shared mem + 分块矩阵乘法实现更高性能的sgemm。

```
TileSgemmBySharedMem Performance Test 9, M = 4096, N = 4096, K = 1024, time = 0.011735 0.011795 0.011845 sec, gflops = 2712.934083 GFLOPS
TileSgemmBySharedMem Performance Test 10, M = 6144, N = 6144, K = 1024, time = 0.026214 0.026220 0.026226 sec, gflops = 2745.957263 GFLOPS
TileSgemmBySharedMem Performance Test 11, M = 8192, N = 8192, K = 1024, time = 0.046110 0.046144 0.046162 sec, gflops = 2773.929722 GFLOPS
TileSgemmBySharedMem Performance Test 12, M = 12288, N = 12288, K = 1024, time = 0.080718 0.087581 0.102973 sec, gflops = 3288.367231 GFLOPS
TileSgemmBySharedMem Performance Test 13, M = 16384, N = 16384, K = 1024, time = 0.143302 0.143312 0.143319 sec, gflops = 3572.620439 GFLOPS
```