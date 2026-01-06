# Makefile for CUDA GEMM

# 编译器
NVCC = nvcc

# 编译选项
NVCC_FLAGS = -O3 -arch=sm_75 -std=c++14
NVCC_FLAGS += -Xcompiler -fopenmp  # 如果需要 OpenMP

# 链接库
LIBS = -lcublas

# 目标文件
TARGET = gemm
SOURCE = gemm.cu

TARGET_SHARED_MEM = gemm_shared_memory
SOURCE_SHARED_MEM = gemm_shared_memory.cu

TARGET_PREFETCH = gemm_prefetch
SOURCE_PREFETCH = gemm_prefetch.cu

TARGET_TENSOR = gemm_tensor
SOURCE_TENSOR = gemm_tensor.cu

# 默认目标：编译所有
all: $(TARGET) $(TARGET_SHARED_MEM) $(TARGET_PREFETCH) $(TARGET_TENSOR)

# 编译基础 GEMM
$(TARGET): $(SOURCE)
	$(NVCC) $(NVCC_FLAGS) $(SOURCE) -o $(TARGET) $(LIBS)

# 编译 Shared Memory 优化版本
$(TARGET_SHARED_MEM): $(SOURCE_SHARED_MEM)
	$(NVCC) $(NVCC_FLAGS) $(SOURCE_SHARED_MEM) -o $(TARGET_SHARED_MEM) $(LIBS)

# 编译 Prefetch 优化版本
$(TARGET_PREFETCH): $(SOURCE_PREFETCH)
	$(NVCC) $(NVCC_FLAGS) $(SOURCE_PREFETCH) -o $(TARGET_PREFETCH) $(LIBS)

# 编译 Tensor Core 版本
$(TARGET_TENSOR): $(SOURCE_TENSOR)
	$(NVCC) $(NVCC_FLAGS) $(SOURCE_TENSOR) -o $(TARGET_TENSOR) $(LIBS)

# 清理
clean:
	rm -f $(TARGET) $(TARGET_SHARED_MEM) $(TARGET_PREFETCH) $(TARGET_TENSOR)

# 运行基础版本（需要确保有 GPU）
run: $(TARGET)
	./$(TARGET)

# 运行 Shared Memory 优化版本
run-shared-mem: $(TARGET_SHARED_MEM)
	./$(TARGET_SHARED_MEM)

# 运行 Prefetch 优化版本
run-prefetch: $(TARGET_PREFETCH)
	./$(TARGET_PREFETCH)

# 运行 Tensor Core 版本
run-tensor: $(TARGET_TENSOR)
	./$(TARGET_TENSOR)

# 显示 GPU 信息
info:
	nvidia-smi

.PHONY: all clean run run-shared-mem run-prefetch run-tensor info


