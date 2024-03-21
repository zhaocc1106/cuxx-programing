# 实现不同版本的gemm并比较性能

参考 https://zhuanlan.zhihu.com/p/657632577

## 环境

1080Ti + CUDA10

## sgemm_naive

简单版本的sgemm。

```
  NaiveSgemm Performance Test 0, M = 128, N = 128, K = 1024, time = 0.000097 0.000099 0.000100 sec, gflops = 316.962838 GFLOPS
  NaiveSgemm Performance Test 1, M = 192, N = 192, K = 1024, time = 0.000189 0.000191 0.000192 sec, gflops = 368.668733 GFLOPS
  NaiveSgemm Performance Test 2, M = 256, N = 256, K = 1024, time = 0.000274 0.000329 0.000367 sec, gflops = 379.365428 GFLOPS
  NaiveSgemm Performance Test 3, M = 384, N = 384, K = 1024, time = 0.000559 0.000561 0.000566 sec, gflops = 501.687449 GFLOPS
  NaiveSgemm Performance Test 4, M = 512, N = 512, K = 1024, time = 0.000925 0.000928 0.000930 sec, gflops = 538.993834 GFLOPS
  NaiveSgemm Performance Test 5, M = 768, N = 768, K = 1024, time = 0.002080 0.002084 0.002088 sec, gflops = 539.944989 GFLOPS
  NaiveSgemm Performance Test 6, M = 1024, N = 1024, K = 1024, time = 0.003517 0.003550 0.003601 sec, gflops = 563.334079 GFLOPS
  NaiveSgemm Performance Test 7, M = 1536, N = 1536, K = 1024, time = 0.007983 0.008003 0.008048 sec, gflops = 562.309715 GFLOPS
  NaiveSgemm Performance Test 8, M = 2048, N = 2048, K = 1024, time = 0.014242 0.014262 0.014309 sec, gflops = 560.935051 GFLOPS
  NaiveSgemm Performance Test 9, M = 3072, N = 3072, K = 1024, time = 0.031776 0.031797 0.031842 sec, gflops = 566.098011 GFLOPS
  NaiveSgemm Performance Test 10, M = 4096, N = 4096, K = 1024, time = 0.056582 0.056600 0.056623 sec, gflops = 565.374995 GFLOPS
  NaiveSgemm Performance Test 11, M = 6144, N = 6144, K = 1024, time = 0.105375 0.112704 0.126589 sec, gflops = 638.843171 GFLOPS
  NaiveSgemm Performance Test 12, M = 8192, N = 8192, K = 1024, time = 0.187867 0.188162 0.188857 sec, gflops = 680.265895 GFLOPS
  NaiveSgemm Performance Test 13, M = 12288, N = 12288, K = 1024, time = 0.423600 0.424451 0.424865 sec, gflops = 678.524038 GFLOPS
  NaiveSgemm Performance Test 14, M = 16384, N = 16384, K = 1024, time = 0.754873 0.758256 0.760623 sec, gflops = 675.233319 GFLOPS
```

## sgemm_tile_shm

通过shared mem + 分块矩阵乘法实现更高性能的sgemm。

```
  TileSgemmBySharedMem Performance Test 0, M = 128, N = 128, K = 1024, time = 0.000344 0.000345 0.000348 sec, gflops = 90.608705 GFLOPS
  TileSgemmBySharedMem Performance Test 1, M = 256, N = 256, K = 1024, time = 0.000344 0.000346 0.000348 sec, gflops = 360.974562 GFLOPS
  TileSgemmBySharedMem Performance Test 2, M = 384, N = 384, K = 1024, time = 0.000363 0.000364 0.000365 sec, gflops = 773.059007 GFLOPS
  TileSgemmBySharedMem Performance Test 3, M = 512, N = 512, K = 1024, time = 0.000364 0.000366 0.000370 sec, gflops = 1364.628853 GFLOPS
  TileSgemmBySharedMem Performance Test 4, M = 768, N = 768, K = 1024, time = 0.000602 0.000610 0.000617 sec, gflops = 1845.530559 GFLOPS
  TileSgemmBySharedMem Performance Test 5, M = 1024, N = 1024, K = 1024, time = 0.001212 0.001218 0.001226 sec, gflops = 1641.522901 GFLOPS
  TileSgemmBySharedMem Performance Test 6, M = 1536, N = 1536, K = 1024, time = 0.001809 0.001816 0.001825 sec, gflops = 2478.139550 GFLOPS
  TileSgemmBySharedMem Performance Test 7, M = 2048, N = 2048, K = 1024, time = 0.003004 0.003020 0.003035 sec, gflops = 2648.619345 GFLOPS
  TileSgemmBySharedMem Performance Test 8, M = 3072, N = 3072, K = 1024, time = 0.006596 0.006615 0.006626 sec, gflops = 2721.161498 GFLOPS
  TileSgemmBySharedMem Performance Test 9, M = 4096, N = 4096, K = 1024, time = 0.011462 0.011478 0.011494 sec, gflops = 2787.894787 GFLOPS
  TileSgemmBySharedMem Performance Test 10, M = 6144, N = 6144, K = 1024, time = 0.025515 0.025547 0.025570 sec, gflops = 2818.331159 GFLOPS
  TileSgemmBySharedMem Performance Test 11, M = 8192, N = 8192, K = 1024, time = 0.044801 0.044999 0.045075 sec, gflops = 2844.488909 GFLOPS
  TileSgemmBySharedMem Performance Test 12, M = 12288, N = 12288, K = 1024, time = 0.078227 0.086165 0.100518 sec, gflops = 3342.415267 GFLOPS
  TileSgemmBySharedMem Performance Test 13, M = 16384, N = 16384, K = 1024, time = 0.138997 0.139031 0.139070 sec, gflops = 3682.628138 GFLOPS
```

## sgemm_tile_shm_v2

在sgemm_tile_shm基础上进行bank conflict缓解优化

```
  TileSgemmBySharedMemV2 Accuracy Test 4 passed, M = 768, N = 768, K = 1024, max_diff: 0.000092.
  TileSgemmBySharedMemV2 Performance Test 0, M = 128, N = 128, K = 1024, time = 0.000212 0.000214 0.000216 sec, gflops = 146.358508 GFLOPS
  TileSgemmBySharedMemV2 Performance Test 1, M = 256, N = 256, K = 1024, time = 0.000215 0.000216 0.000219 sec, gflops = 579.149860 GFLOPS
  TileSgemmBySharedMemV2 Performance Test 2, M = 384, N = 384, K = 1024, time = 0.000233 0.000235 0.000236 sec, gflops = 1196.441962 GFLOPS
  TileSgemmBySharedMemV2 Performance Test 3, M = 512, N = 512, K = 1024, time = 0.000234 0.000236 0.000238 sec, gflops = 2114.458153 GFLOPS
  TileSgemmBySharedMemV2 Performance Test 4, M = 768, N = 768, K = 1024, time = 0.000283 0.000287 0.000291 sec, gflops = 3920.450757 GFLOPS
  TileSgemmBySharedMemV2 Performance Test 5, M = 1024, N = 1024, K = 1024, time = 0.000505 0.000509 0.000512 sec, gflops = 3930.619842 GFLOPS
  TileSgemmBySharedMemV2 Performance Test 6, M = 1536, N = 1536, K = 1024, time = 0.000822 0.000826 0.000832 sec, gflops = 5449.271931 GFLOPS
  TileSgemmBySharedMemV2 Performance Test 7, M = 2048, N = 2048, K = 1024, time = 0.001363 0.001368 0.001372 sec, gflops = 5849.677361 GFLOPS
  TileSgemmBySharedMemV2 Performance Test 8, M = 3072, N = 3072, K = 1024, time = 0.002949 0.002981 0.002992 sec, gflops = 6037.441805 GFLOPS
  TileSgemmBySharedMemV2 Performance Test 9, M = 4096, N = 4096, K = 1024, time = 0.005153 0.005157 0.005160 sec, gflops = 6205.737505 GFLOPS
  TileSgemmBySharedMemV2 Performance Test 10, M = 6144, N = 6144, K = 1024, time = 0.011372 0.011380 0.011385 sec, gflops = 6327.077976 GFLOPS
  TileSgemmBySharedMemV2 Performance Test 11, M = 8192, N = 8192, K = 1024, time = 0.020036 0.020044 0.020051 sec, gflops = 6385.899907 GFLOPS
  TileSgemmBySharedMemV2 Performance Test 12, M = 12288, N = 12288, K = 1024, time = 0.038527 0.041305 0.044925 sec, gflops = 6972.580034 GFLOPS
  TileSgemmBySharedMemV2 Performance Test 13, M = 16384, N = 16384, K = 1024, time = 0.063240 0.065728 0.068515 sec, gflops = 7789.674893 GFLOPS
```