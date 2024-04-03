# reduce

通过cuda实现的reduce算法，包括sum、max、min、mean等。
参考[CUDA编程入门之Warp-Level Primitives](https://zhuanlan.zhihu.com/p/572820783)。

## reduce_sum.cu

通过warp tile实现的block reduce sum算法。
```
cpu ReduceSum: 4.71767e+06, time: 3355 us
gpu ReduceSumByWarpTile: 4.71767e+06, time: 501 us
gpu ReduceSumByWarpTileFloat4: 4.71767e+06, time: 498 us
gpu ReduceSumByBucket: 4.71767e+06, time: 1010 us
gpu ReduceSumByAtomic: 4.71767e+06, time: 2659 us
```

## reduce_max.cu

通过warp tile实现的block reduce max算法。
```
cpu ReduceMax: 2.14748e+09, time: 3291 us
gpu ReduceMaxV1: 2.14748e+09, time: 1141 us
gpu ReduceMaxV2: 2.14748e+09, time: 1195 us
```