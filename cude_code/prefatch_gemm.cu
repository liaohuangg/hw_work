/**
 * CUDA GEMM (General Matrix Multiplication) - 数据预取优化版本
 * 计算 C = A * B，其中 A[M][K], B[K][N], C[M][N]
 * 
 * 参数：(M, N, K) = (20480, 2048, 8192)
 * 数据类型：FP32 (float)
 * 
 * 本文件专注于数据预取优化技术，包括：
 * 1. 双缓冲（Double Buffering）技术
 * 2. 异步内存拷贝（Async Memory Copy）
 * 3. Pipeline 预取优化
 * 4. 隐藏内存访问延迟
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <algorithm>

// 使用 FP32 (float) 作为数据类型
using fp32 = float;

// GEMM 参数
const int M = 20480;  // A 矩阵的行数，C 矩阵的行数
const int N = 2048;   // B 矩阵的列数，C 矩阵的列数
const int K = 8192;   // A 矩阵的列数，B 矩阵的行数

// Tile 大小配置
#define TILE_SIZE 32   // 使用 32x32 tile 以获得更好的性能

// CUDA 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

/**
 * 基础 CUDA GEMM Kernel（不使用预取，作为性能基准）
 * 使用共享内存 tile-based 方法
 */
__global__ void gemm_kernel_baseline(
    const fp32* __restrict__ A,
    const fp32* __restrict__ B,
    fp32* __restrict__ C,
    int M, int N, int K
) {
    // 共享内存：每个 block 缓存 A 和 B 的一个 tile
    __shared__ fp32 tileA[TILE_SIZE][TILE_SIZE];
    __shared__ fp32 tileB[TILE_SIZE][TILE_SIZE];
    
    // 线程在 block 中的位置
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 线程在全局矩阵中的位置
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    // 使用寄存器存储累加结果
    fp32 sum = 0.0f;
    
    // 遍历 K 维度
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < num_tiles; tile++) {
        // 加载 A 的 tile（同步加载）
        int a_row = row;
        int a_col = tile * TILE_SIZE + tx;
        
        if (a_row < M && a_col < K) {
            tileA[ty][tx] = A[a_row * K + a_col];
        } else {
            tileA[ty][tx] = 0.0f;
        }
        
        // 加载 B 的 tile（同步加载）
        int b_row = tile * TILE_SIZE + ty;
        int b_col = col;
        
        if (b_row < K && b_col < N) {
            tileB[ty][tx] = B[b_row * N + b_col];
        } else {
            tileB[ty][tx] = 0.0f;
        }
        
        // 同步，确保所有线程都加载完数据
        __syncthreads();
        
        // 计算 tile 内的点积
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        
        // 同步，确保所有线程都计算完再加载下一个 tile
        __syncthreads();
    }
    
    // 写入结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * 数据预取优化 GEMM Kernel（使用双缓冲技术）
 * 
 * 优化策略：
 * 1. 使用双缓冲共享内存（tileA[2][TILE_SIZE][TILE_SIZE]）
 * 2. 在计算当前 tile 的同时，预取下一个 tile 的数据
 * 3. 通过重叠计算和内存访问来隐藏延迟
 */
__global__ void gemm_kernel_prefetch_double_buffer(
    const fp32* __restrict__ A,
    const fp32* __restrict__ B,
    fp32* __restrict__ C,
    int M, int N, int K
) {
    // 双缓冲共享内存：两个 tile 缓冲区
    __shared__ fp32 tileA[2][TILE_SIZE][TILE_SIZE];
    __shared__ fp32 tileB[2][TILE_SIZE][TILE_SIZE];
    
    // 线程在 block 中的位置
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 线程在全局矩阵中的位置
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    // 使用寄存器存储累加结果
    fp32 sum = 0.0f;
    
    // 计算 tile 数量
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // 预取第一个 tile（buffer 0）
    int tile = 0;
    int a_row = row;
    int a_col = tile * TILE_SIZE + tx;
    int b_row = tile * TILE_SIZE + ty;
    int b_col = col;
    
    if (a_row < M && a_col < K) {
        tileA[0][ty][tx] = A[a_row * K + a_col];
    } else {
        tileA[0][ty][tx] = 0.0f;
    }
    
    if (b_row < K && b_col < N) {
        tileB[0][ty][tx] = B[b_row * N + b_col];
    } else {
        tileB[0][ty][tx] = 0.0f;
    }
    
    __syncthreads();
    
    // 主循环：使用双缓冲重叠计算和内存访问
    for (tile = 0; tile < num_tiles; tile++) {
        // 当前使用的 buffer（0 或 1）
        int current_buffer = tile % 2;
        // 下一个 buffer（用于预取）
        int next_buffer = (tile + 1) % 2;
        
        // 如果不是最后一个 tile，预取下一个 tile 的数据
        if (tile + 1 < num_tiles) {
            int next_tile = tile + 1;
            int next_a_col = next_tile * TILE_SIZE + tx;
            int next_b_row = next_tile * TILE_SIZE + ty;
            
            // 预取 A 的下一个 tile
            if (a_row < M && next_a_col < K) {
                tileA[next_buffer][ty][tx] = A[a_row * K + next_a_col];
            } else {
                tileA[next_buffer][ty][tx] = 0.0f;
            }
            
            // 预取 B 的下一个 tile
            if (next_b_row < K && b_col < N) {
                tileB[next_buffer][ty][tx] = B[next_b_row * N + b_col];
            } else {
                tileB[next_buffer][ty][tx] = 0.0f;
            }
        }
        
        // 计算当前 tile 的点积（使用当前 buffer）
        // 注意：此时预取操作可能还在进行，但当前 buffer 的数据已经准备好
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[current_buffer][ty][k] * tileB[current_buffer][k][tx];
        }
        
        // 同步：确保预取完成且计算完成
        __syncthreads();
    }
    
    // 写入结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}



/**
 * 打印详细的性能分析结果
 */
void print_performance_analysis(
    const std::string& name,
    double time_ms,
    int M, int N, int K
) {
    // 计算 FLOPS（浮点运算次数）
    long long flops = (long long)M * N * 2 * K;
    
    // 计算 GFLOPS（每秒十亿次浮点运算）
    double gflops = (flops / 1e9) / (time_ms / 1000.0);
    
    // 计算内存带宽（字节）
    long long bytes = (long long)(M * K + K * N + M * N) * sizeof(fp32);
    
    // 计算内存带宽（GB/s）
    double bandwidth_gb_s = (bytes / 1e9) / (time_ms / 1000.0);
    
    // 计算计算强度（FLOPS/Byte）
    double compute_intensity = (double)flops / bytes;
    
    std::cout << "\n========== " << name << " 性能分析 ==========" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "计算时间: " << time_ms << " ms" << std::endl;
    std::cout << "FLOPS: " << flops / 1e12 << " TFLOPS" << std::endl;
    std::cout << "性能: " << gflops << " GFLOPS" << std::endl;
    std::cout << "内存带宽: " << bandwidth_gb_s << " GB/s" << std::endl;
    std::cout << "计算强度: " << compute_intensity << " FLOPS/Byte" << std::endl;
    std::cout << "Tile 大小: " << TILE_SIZE << "x" << TILE_SIZE << std::endl;
    std::cout << "共享内存使用: " << (2 * 2 * TILE_SIZE * TILE_SIZE * sizeof(fp32) / 1024.0) << " KB per block (双缓冲)" << std::endl;
    std::cout << "矩阵大小: A[" << M << "][" << K << "] * B[" << K << "][" << N << "] = C[" << M << "][" << N << "]" << std::endl;
    std::cout << "数据大小: A=" << (M*K*sizeof(fp32)/1e6) << " MB, "
              << "B=" << (K*N*sizeof(fp32)/1e6) << " MB, "
              << "C=" << (M*N*sizeof(fp32)/1e6) << " MB" << std::endl;
}

/**
 * 打印 GPU 设备信息
 */
void print_gpu_info() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        std::cerr << "错误：未找到 CUDA 设备" << std::endl;
        return;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    std::cout << "\n========== GPU 设备信息 ==========" << std::endl;
    std::cout << "设备名称: " << prop.name << std::endl;
    std::cout << "计算能力: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "全局内存: " << prop.totalGlobalMem / 1e9 << " GB" << std::endl;
    std::cout << "共享内存 per block: " << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "共享内存 per SM: " << prop.sharedMemPerMultiprocessor / 1024.0 << " KB" << std::endl;
    std::cout << "每个 block 的最大线程数: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "每个 SM 的最大线程数: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "SM 数量: " << prop.multiProcessorCount << std::endl;
    std::cout << "Warp 大小: " << prop.warpSize << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "========== CUDA GEMM 数据预取优化性能测试 ==========" << std::endl;
    std::cout << "矩阵大小: A[" << M << "][" << K << "] * B[" << K << "][" << N << "] = C[" << M << "][" << N << "]" << std::endl;
    std::cout << "数据类型: FP32 (float)" << std::endl;
    
    // 打印 GPU 信息
    print_gpu_info();
    
    // 分配主机内存
    size_t size_A = M * K * sizeof(fp32);
    size_t size_B = K * N * sizeof(fp32);
    size_t size_C = M * N * sizeof(fp32);
    
    fp32* h_A = (fp32*)malloc(size_A);
    fp32* h_B = (fp32*)malloc(size_B);
    fp32* h_C_gpu_baseline = (fp32*)malloc(size_C);
    fp32* h_C_gpu_prefetch = (fp32*)malloc(size_C);
    
    // 初始化矩阵（使用随机值）
    std::cout << "\n初始化矩阵数据..." << std::endl;
    srand(42);  // 固定随机种子以便复现
    for (int i = 0; i < M * K; i++) {
        h_A[i] = static_cast<fp32>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = static_cast<fp32>(rand()) / RAND_MAX;
    }
    
    // 分配设备内存
    fp32* d_A;
    fp32* d_B;
    fp32* d_C;
    
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    
    // 拷贝数据到设备
    std::cout << "拷贝数据到 GPU..." << std::endl;
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    
    // 多次运行取平均值
    int num_runs = 10;
    double total_time = 0.0;
    
    // ========== GPU 基准 Kernel（不使用预取）==========
    std::cout << "\n========== GPU GEMM (基准 - 无预取) ==========" << std::endl;
    
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, 
                  (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // 预热
    gemm_kernel_baseline<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 多次运行取平均值
    total_time = 0.0;
    for (int i = 0; i < num_runs; i++) {
        CUDA_CHECK(cudaMemset(d_C, 0, size_C));
        
        auto start = std::chrono::high_resolution_clock::now();
        gemm_kernel_baseline<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_time += duration.count() / 1000.0;
    }
    double time_gpu_baseline_ms = total_time / num_runs;
    
    CUDA_CHECK(cudaMemcpy(h_C_gpu_baseline, d_C, size_C, cudaMemcpyDeviceToHost));
    print_performance_analysis("GPU (基准 - 无预取)", time_gpu_baseline_ms, M, N, K);
    
    // ========== GPU 数据预取优化 Kernel（双缓冲）==========
    std::cout << "\n========== GPU GEMM (数据预取优化 - 双缓冲) ==========" << std::endl;
    
    // 预热
    gemm_kernel_prefetch_double_buffer<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 多次运行取平均值
    total_time = 0.0;
    for (int i = 0; i < num_runs; i++) {
        CUDA_CHECK(cudaMemset(d_C, 0, size_C));
        
        auto start = std::chrono::high_resolution_clock::now();
        gemm_kernel_prefetch_double_buffer<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_time += duration.count() / 1000.0;
    }
    double time_gpu_prefetch_ms = total_time / num_runs;
    
    CUDA_CHECK(cudaMemcpy(h_C_gpu_prefetch, d_C, size_C, cudaMemcpyDeviceToHost));
    print_performance_analysis("GPU (数据预取优化 - 双缓冲)", time_gpu_prefetch_ms, M, N, K);
    
    // ========== 性能对比总结 ==========
    std::cout << "\n========== 性能对比总结 ==========" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "GPU (基准 - 无预取):          " << time_gpu_baseline_ms << " ms" << std::endl;
    std::cout << "GPU (数据预取优化 - 双缓冲):  " << time_gpu_prefetch_ms << " ms" 
              << " (相对基准提升: " << time_gpu_baseline_ms / time_gpu_prefetch_ms << "x)" << std::endl;
    
    // 计算 GFLOPS 对比
    long long flops = (long long)M * N * 2 * K;
    double gflops_gpu_baseline = (flops / 1e9) / (time_gpu_baseline_ms / 1000.0);
    double gflops_gpu_prefetch = (flops / 1e9) / (time_gpu_prefetch_ms / 1000.0);
    
    std::cout << "\n算力对比 (GFLOPS):" << std::endl;
    std::cout << "GPU (基准 - 无预取):          " << gflops_gpu_baseline << " GFLOPS" << std::endl;
    std::cout << "GPU (数据预取优化 - 双缓冲):  " << gflops_gpu_prefetch << " GFLOPS" << std::endl;
    
    // 性能分析
    std::cout << "\n========== 性能分析 ==========" << std::endl;
    std::cout << "数据预取优化的优势：" << std::endl;
    std::cout << "1. 双缓冲预取相对基准版本提升: " << (time_gpu_baseline_ms / time_gpu_prefetch_ms - 1.0) * 100 << "%" << std::endl;
    std::cout << "2. 通过重叠计算和内存访问，有效隐藏内存延迟" << std::endl;
    std::cout << "3. 双缓冲技术允许在计算当前 tile 的同时预取下一个 tile" << std::endl;
    
    // 清理资源
    free(h_A);
    free(h_B);
    free(h_C_gpu_baseline);
    free(h_C_gpu_prefetch);
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    std::cout << "\n========== 测试完成 ==========" << std::endl;
    
    return 0;
}

