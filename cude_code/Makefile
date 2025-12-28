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

TARGET_SHARED = share_mem_gemm
SOURCE_SHARED = share_mem_gemm.cu

TARGET_PREFETCH = prefatch_gemm
SOURCE_PREFETCH = prefatch_gemm.cu

# 默认目标：编译所有
all: $(TARGET) $(TARGET_SHARED) $(TARGET_PREFETCH)

# 编译基础 GEMM
$(TARGET): $(SOURCE)
	$(NVCC) $(NVCC_FLAGS) $(SOURCE) -o $(TARGET) $(LIBS)

# 编译共享内存优化 GEMM
$(TARGET_SHARED): $(SOURCE_SHARED)
	$(NVCC) $(NVCC_FLAGS) $(SOURCE_SHARED) -o $(TARGET_SHARED) $(LIBS)

# 编译数据预取优化 GEMM
$(TARGET_PREFETCH): $(SOURCE_PREFETCH)
	$(NVCC) $(NVCC_FLAGS) $(SOURCE_PREFETCH) -o $(TARGET_PREFETCH) $(LIBS)

# 清理
clean:
	rm -f $(TARGET) $(TARGET_SHARED) $(TARGET_PREFETCH)

# 运行基础版本（需要确保有 GPU）
run: $(TARGET)
	./$(TARGET)

# 运行共享内存优化版本
run-shared: $(TARGET_SHARED)
	./$(TARGET_SHARED)

# 运行数据预取优化版本
run-prefetch: $(TARGET_PREFETCH)
	./$(TARGET_PREFETCH)

# 显示 GPU 信息
info:
	nvidia-smi

.PHONY: all clean run run-shared run-prefetch info


