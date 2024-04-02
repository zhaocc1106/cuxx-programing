# reduce

通过cuda实现的reduce算法，包括sum、max、min、mean等。
参考[CUDA编程入门之Warp-Level Primitives](https://zhuanlan.zhihu.com/p/572820783)。

## reduce_sum.cu

通过warp tile实现的block reduce sum算法。

## reduce_max.cu

通过warp tile实现的block reduce max算法。