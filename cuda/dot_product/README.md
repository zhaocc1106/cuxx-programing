# dot_product

通过cuda实现dot_product算法，并比较和cublas的性能。

## 环境

K80 + CUDA10

## dot_product.cu

```
N=1048576
DotProductCpu: 2.12579e+07, time: 3778 us
DotProductWarpTile: 2.12587e+07, time: 1493 us
DotProductWarpTileFloat4: 2.12587e+07, time: 1413 us
DotProductCublas: 2.12587e+07, time: 1452 us
```