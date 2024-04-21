# element_wise

CUDA实现element wise算子。

## 环境

K80 + CUDA10

## element_wise_add

```
N: 1048576
ElementWiseAddCpu, time: 6640 us
ElementWiseAdd, time: 2091 us
ElementWiseAddFloat4, time: 2087 us
cublasSaxpy, time: 1095 us
```