# CUDA GEMM 性能测试

本项目实现了 CUDA 版本的通用矩阵乘法（GEMM），用于测量和比较 CPU 和 GPU 的计算性能。

## 项目要求

- **GEMM 参数**: (M, N, K) = (20480, 2048, 8192)
- **数据类型**: FP32 (float)
- **任务**: 实现 CUDA GEMM，测量 CPU 和 GPU 上的计算时间和实际算力，进行性能分析和评估

## 文件说明

- `gemm.cu`: 主程序文件，包含：
  - CPU GEMM 参考实现（用于验证正确性）
  - CUDA GEMM 基础 Kernel（每个线程计算一个元素）
  - CUDA GEMM 优化 Kernel（使用共享内存的 tile-based 方法）
  - cuBLAS 参考实现（用于性能对比）
  - 性能测量和分析代码

- `Makefile`: 编译脚本

- `README.md`: 本文件

## 编译方法

### 前置要求

- NVIDIA GPU（支持 CUDA）
- CUDA Toolkit（建议 11.0+）
- cuBLAS 库（通常随 CUDA Toolkit 安装）

### 编译步骤

```bash
cd /home/huangl/workspace/hw_work/cude_code
make
```

### 检查 GPU 信息

```bash
make info
# 或直接运行
nvidia-smi
```

## 运行方法

```bash
./gemm
```

或者使用 make：

```bash
make run
```

## 输出说明

程序会输出以下内容：

1. **CPU GEMM 性能分析**
   - 计算时间（毫秒）
   - FLOPS（浮点运算次数）
   - GFLOPS（每秒十亿次浮点运算）
   - 内存带宽（GB/s）

2. **GPU GEMM (基础 Kernel) 性能分析**
   - 使用全局内存的基础实现
   - 性能指标同上

3. **GPU GEMM (优化 Kernel) 性能分析**
   - 使用共享内存的 tile-based 优化实现
   - 性能指标同上

4. **cuBLAS 参考性能**
   - NVIDIA 官方优化的 GEMM 库性能
   - 作为性能上限参考

5. **性能对比总结**
   - 各实现的执行时间对比
   - 加速比（相对于 CPU）
   - 算力对比（GFLOPS）

## 实现细节

### CPU GEMM

使用三重循环的标准实现：
```cpp
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[i*K + k] * B[k*N + j];
        }
        C[i*N + j] = sum;
    }
}
```

### GPU GEMM (基础 Kernel)

- 每个线程计算 C 矩阵的一个元素
- 使用全局内存访问
- Block 大小：16x16 = 256 个线程

### GPU GEMM (优化 Kernel)

- 使用共享内存缓存数据块（tile）
- Tile 大小：16x16
- 减少全局内存访问次数
- 提高内存访问效率

### 性能测量

- 多次运行取平均值（默认 10 次）
- 使用高精度时钟测量时间
- 计算 FLOPS 和 GFLOPS
- 计算内存带宽

## 预期结果

根据硬件配置不同，预期结果可能包括：

- **CPU**: 通常在 1-10 GFLOPS 范围内
- **GPU (基础)**: 通常在 100-500 GFLOPS 范围内
- **GPU (优化)**: 通常在 500-2000 GFLOPS 范围内
- **cuBLAS**: 通常在 1000-10000+ GFLOPS 范围内（取决于 GPU 型号）

## 注意事项

1. 确保有足够的 GPU 内存（至少 2GB 可用内存）
2. 矩阵大小较大，CPU 计算可能需要较长时间
3. 首次运行可能较慢（GPU 初始化）
4. 结果验证使用相对误差容差（默认 1e-3）

## 故障排除

### 编译错误

- 检查 CUDA Toolkit 是否正确安装：`nvcc --version`
- 检查 GPU 架构是否匹配：`nvidia-smi` 查看 GPU 型号，然后修改 Makefile 中的 `-arch=sm_XX`

### 运行时错误

- 检查 GPU 是否可用：`nvidia-smi`
- 检查内存是否足够：矩阵总大小约为 1.5GB
- 检查 CUDA 驱动版本是否兼容

### 性能异常

- 确保 GPU 没有被其他程序占用
- 尝试关闭其他 GPU 应用程序
- 检查 GPU 是否处于性能模式

## 进一步优化方向

1. 使用 Tensor Core（如果 GPU 支持）
2. 使用更复杂的优化技术（如 register blocking）
3. 使用 CUDA Streams 实现异步执行
4. 使用混合精度（FP16）计算
5. 实现多 GPU 版本

## 参考资料

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [GEMM Optimization Techniques](https://github.com/NVIDIA/cutlass)

