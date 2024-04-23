# activation

CUDA实现激活函数sigmoid、relu等。

## 环境

K80 + CUDA10

## sigmoid.cu

```
N: 1048576
SigmoidCpu time: 18169 us
Sigmoid time: 1503 us
SigmoidFloat4 time: 1496 us
```

## relu.cu

```
N: 1048576
ReluCpu time: 13549 us
Relu time: 1496 us
ReluFloat4 time: 1537 us
```