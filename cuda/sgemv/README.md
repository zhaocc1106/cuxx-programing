# sgemv

cuda实现sgemv。

## 环境

1080Ti + CUDA10

## sgemv.cu

通过warp tile实现sgem，并比较和cublas的性能。

```
cpu time: 6263 us
SgemvN32 time: 490 us
SgemvN128 time: 485 us
SgemvCublas time: 495 us
```