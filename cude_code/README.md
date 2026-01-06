# CUDA GEMM 优化实现

本项目实现了多种 CUDA 版本的通用矩阵乘法（GEMM），展示了不同的优化技术及其性能提升。

## 文件说明

### `gemm.cu`
CPU 和 GPU 基础版本的性能对比。
- **CPU GEMM**: 标准三重循环实现
- **GPU GEMM**: 基础 CUDA kernel，每个线程计算一个元素
- **矩阵大小**: (M, N, K) = (20480, 2048, 8192)
- **数据类型**: FP32 (float)

### `gemm_shared_memory.cu`
共享内存优化的 GEMM 实现。
- **Baseline**: 基础 CUDA kernel（全局内存访问）
- **优化版本**: 使用共享内存进行分块（tiling）优化
- **功能**: 自动搜索最优 tile size (BM, BN, BK)，并与 baseline 进行性能对比
- **矩阵大小**: (M, N, K) = (20480, 2048, 8192)
- **数据类型**: FP32 (float)

### `gemm_prefetch.cu`
双缓冲数据预取优化的 GEMM 实现。
- **Baseline**: 基础 CUDA kernel（全局内存访问）
- **优化版本**: 使用双缓冲（double buffering）技术进行数据预取，实现计算与内存访问的重叠
- **功能**: 自动搜索最优 tile size，并与 baseline 进行性能对比
- **矩阵大小**: (M, N, K) = (20480, 2048, 8192)
- **数据类型**: FP32 (float)

### `gemm_tensor.cu`
Tensor Core 优化的 GEMM 实现。
- **CUDA Core 版本**: 使用 FP32 精度计算的 baseline（输入输出为 FP16）
- **Tensor Core 版本**: 使用 WMMA API 调用 Tensor Core 进行加速
- **功能**: 对比 Tensor Core 与 CUDA Core 的性能和精度差异
- **矩阵大小**: (M, N, K) = (20480, 2048, 8192)
- **数据类型**: FP16 (half precision)

## 编译方法

### 前置要求
- NVIDIA GPU（支持 CUDA）
- CUDA Toolkit（建议 11.0+）
- 对于 `gemm_tensor.cu`，需要支持 Tensor Core 的 GPU（如 Volta 架构及以上）

### 编译所有文件
```bash
make
```

### 编译单个文件
```bash
make gemm              # 编译 gemm.cu
make gemm_shared_memory # 编译 gemm_shared_memory.cu
make gemm_prefetch     # 编译 gemm_prefetch.cu
make gemm_tensor       # 编译 gemm_tensor.cu
```

### 检查 GPU 信息
```bash
make info
```

## 运行方法

```bash
make run              # 运行 gemm
make run-shared-mem   # 运行 gemm_shared_memory
make run-prefetch     # 运行 gemm_prefetch
make run-tensor       # 运行 gemm_tensor
```

或者直接运行：
```bash
./gemm
./gemm_shared_memory
./gemm_prefetch
./gemm_tensor
```

## 清理

```bash
make clean
```

## 优化技术说明

1. **共享内存优化**: 将数据块加载到共享内存中，减少全局内存访问次数
2. **数据预取**: 使用双缓冲技术，在计算当前数据块的同时预取下一个数据块，实现计算与内存访问的重叠
3. **Tensor Core**: 利用 GPU 的专用 Tensor Core 单元，大幅提升 FP16 矩阵运算性能
