# sgemv

cuda实现sgemv。

## 环境

1080Ti + CUDA10

## sgemv.cu

通过warp tile实现sgem，并比较和cublas的性能。

```
N = 16
cpu avg time: 102.5 us, sgemv_fn avg time: 41.5 us, sgemv_cublas avg time: 45.7 us
N = 32
cpu avg time: 201.7 us, sgemv_fn avg time: 53.8 us, sgemv_cublas avg time: 60.2 us
N = 128
cpu avg time: 795.7 us, sgemv_fn avg time: 118.5 us, sgemv_cublas avg time: 122.5 us
```