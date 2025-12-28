/**
 * CUDA GEMM (General Matrix Multiplication) 实现
 * 计算 C = A * B，其中 A[M][K], B[K][N], C[M][N]
 * 
 * 参数：(M, N, K) = (20480, 2048, 8192)
 * 数据类型：FP32 (float)
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cstring>

// 使用 FP32 (float) 作为数据类型
using fp32 = float;

// GEMM 参数
const int M = 20480;  // A 矩阵的行数，C 矩阵的行数
const int N = 2048;   // B 矩阵的列数，C 矩阵的列数
const int K = 8192;   // A 矩阵的列数，B 矩阵的行数

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
 * CPU 端 GEMM 参考实现（用于验证正确性）
 * C = A * B
 */
void cpu_gemm(const fp32* A, const fp32* B, fp32* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            fp32 sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * 基础 CUDA GEMM Kernel（每个线程计算 C 矩阵的一个元素）
 * 使用全局内存，无优化
 */
__global__ void gemm_kernel_basic(
    const fp32* A,  // [M][K]
    const fp32* B,  // [K][N]
    fp32* C,        // [M][N]
    int M, int N, int K
) {
    // 计算当前线程负责的 C 矩阵元素位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查
    if (row < M && col < N) {
        fp32 sum = 0.0f;
        // 累加 A[row][k] * B[k][col]
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/**
 * 优化的 CUDA GEMM Kernel（使用共享内存）
 * 使用 tile-based 方法，每个 block 处理一个 tile
 */
#define TILE_SIZE 16

__global__ void gemm_kernel_tiled(
    const fp32* A,
    const fp32* B,
    fp32* C,
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
    
    fp32 sum = 0.0f;
    
    // 遍历 K 维度，每次处理一个 tile
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // 加载 A 的 tile 到共享内存
        if (row < M && (tile * TILE_SIZE + tx) < K) {
            tileA[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        } else {
            tileA[ty][tx] = 0.0f;
        }
        
        // 加载 B 的 tile 到共享内存
        if ((tile * TILE_SIZE + ty) < K && col < N) {
            tileB[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
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
 * 验证两个矩阵是否相等（考虑浮点数误差）
 */
bool verify_result(
    const fp32* C_gpu, 
    const fp32* C_cpu, 
    int size,
    fp32 tolerance = 1e-3f
) {
    for (int i = 0; i < size; i++) {
        fp32 diff = std::abs(C_gpu[i] - C_cpu[i]);
        fp32 max_val = std::max(std::abs(C_gpu[i]), std::abs(C_cpu[i]));
        if (max_val > 1e-6f && diff / max_val > tolerance) {
            std::cout << "验证失败：位置 " << i 
                      << " GPU=" << C_gpu[i] 
                      << " CPU=" << C_cpu[i] 
                      << " 相对误差=" << diff / max_val << std::endl;
            return false;
        }
    }
    return true;
}

/**
 * 打印性能分析结果
 */
void print_performance_analysis(
    const std::string& name,
    double time_ms,
    int M, int N, int K
) {
    // 计算 FLOPS（浮点运算次数）
    // GEMM: C[M][N] = A[M][K] * B[K][N]
    // 每个 C[i][j] 需要 K 次乘法和 K 次加法，共 2*K 次浮点运算
    // 总 FLOPS = M * N * (2 * K)
    long long flops = (long long)M * N * 2 * K;
    
    // 计算 GFLOPS（每秒十亿次浮点运算）
    double gflops = (flops / 1e9) / (time_ms / 1000.0);
    
    // 计算内存带宽（字节）
    // A: M*K*sizeof(fp32), B: K*N*sizeof(fp32), C: M*N*sizeof(fp32)
    long long bytes = (long long)(M * K + K * N + M * N) * sizeof(fp32);
    
    // 计算内存带宽（GB/s）
    double bandwidth_gb_s = (bytes / 1e9) / (time_ms / 1000.0);
    
    std::cout << "\n========== " << name << " 性能分析 ==========" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "计算时间: " << time_ms << " ms" << std::endl;
    std::cout << "FLOPS: " << flops / 1e12 << " TFLOPS" << std::endl;
    std::cout << "性能: " << gflops << " GFLOPS" << std::endl;
    std::cout << "内存带宽: " << bandwidth_gb_s << " GB/s" << std::endl;
    std::cout << "矩阵大小: A[" << M << "][" << K << "] * B[" << K << "][" << N << "] = C[" << M << "][" << N << "]" << std::endl;
    std::cout << "数据大小: A=" << (M*K*sizeof(fp32)/1e6) << " MB, "
              << "B=" << (K*N*sizeof(fp32)/1e6) << " MB, "
              << "C=" << (M*N*sizeof(fp32)/1e6) << " MB" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "========== CUDA GEMM 性能测试 ==========" << std::endl;
    std::cout << "矩阵大小: A[" << M << "][" << K << "] * B[" << K << "][" << N << "] = C[" << M << "][" << N << "]" << std::endl;
    std::cout << "数据类型: FP32 (float)" << std::endl;
    
    // 分配主机内存
    size_t size_A = M * K * sizeof(fp32);
    size_t size_B = K * N * sizeof(fp32);
    size_t size_C = M * N * sizeof(fp32);
    
    fp32* h_A = (fp32*)malloc(size_A);
    fp32* h_B = (fp32*)malloc(size_B);
    fp32* h_C_cpu = (fp32*)malloc(size_C);
    fp32* h_C_gpu_basic = (fp32*)malloc(size_C);
    fp32* h_C_gpu_tiled = (fp32*)malloc(size_C);
    
    // 初始化矩阵（使用随机值）
    std::cout << "\n初始化矩阵数据..." << std::endl;
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
    
    // ========== CPU 计算 ==========
    // std::cout << "\n========== CPU GEMM 计算 ==========" << std::endl;
    // auto start_cpu = std::chrono::high_resolution_clock::now();
    // cpu_gemm(h_A, h_B, h_C_cpu, M, N, K);
    // auto end_cpu = std::chrono::high_resolution_clock::now();
    // auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);
    // double time_cpu_ms = duration_cpu.count();
    // print_performance_analysis("CPU", time_cpu_ms, M, N, K);
    
    // ========== GPU 基础 Kernel ==========
    std::cout << "\n========== GPU GEMM (基础 Kernel) ==========" << std::endl;
    
    // 配置线程块和网格大小
    dim3 blockSize(16, 16);  // 每个 block 16x16 = 256 个线程
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                  (M + blockSize.y - 1) / blockSize.y);
    
    // 预热
    gemm_kernel_basic<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 多次运行取平均值
    int num_runs = 10;
    double total_time = 0.0;
    for (int i = 0; i < num_runs; i++) {
        CUDA_CHECK(cudaMemset(d_C, 0, size_C));
        
        auto start = std::chrono::high_resolution_clock::now();
        gemm_kernel_basic<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_time += duration.count() / 1000.0;  // 转换为毫秒
    }
    double time_gpu_basic_ms = total_time / num_runs;
    
    // 拷贝结果回主机
    CUDA_CHECK(cudaMemcpy(h_C_gpu_basic, d_C, size_C, cudaMemcpyDeviceToHost));
    
    print_performance_analysis("GPU (基础 Kernel)", time_gpu_basic_ms, M, N, K);
    
    // 验证基础 kernel 结果
    std::cout << "\n验证 GPU (基础 Kernel) 结果..." << std::endl;
    if (verify_result(h_C_gpu_basic, h_C_cpu, M * N)) {
        std::cout << "✓ GPU (基础 Kernel) 结果正确！" << std::endl;
    } else {
        std::cout << "✗ GPU (基础 Kernel) 结果错误！" << std::endl;
    }
    
    // ========== GPU 优化 Kernel (Tiled) ==========
    std::cout << "\n========== GPU GEMM (优化 Kernel - Tiled) ==========" << std::endl;
    
    dim3 blockSize_tiled(TILE_SIZE, TILE_SIZE);
    dim3 gridSize_tiled((N + TILE_SIZE - 1) / TILE_SIZE, 
                        (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // 预热
    gemm_kernel_tiled<<<gridSize_tiled, blockSize_tiled>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 多次运行取平均值
    total_time = 0.0;
    for (int i = 0; i < num_runs; i++) {
        CUDA_CHECK(cudaMemset(d_C, 0, size_C));
        
        auto start = std::chrono::high_resolution_clock::now();
        gemm_kernel_tiled<<<gridSize_tiled, blockSize_tiled>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_time += duration.count() / 1000.0;
    }
    double time_gpu_tiled_ms = total_time / num_runs;
    
    // 拷贝结果回主机
    CUDA_CHECK(cudaMemcpy(h_C_gpu_tiled, d_C, size_C, cudaMemcpyDeviceToHost));
    
    print_performance_analysis("GPU (优化 Kernel - Tiled)", time_gpu_tiled_ms, M, N, K);
    
    // 验证优化 kernel 结果
    std::cout << "\n验证 GPU (优化 Kernel) 结果..." << std::endl;
    if (verify_result(h_C_gpu_tiled, h_C_cpu, M * N)) {
        std::cout << "✓ GPU (优化 Kernel) 结果正确！" << std::endl;
    } else {
        std::cout << "✗ GPU (优化 Kernel) 结果错误！" << std::endl;
    }
    
    // ========== cuBLAS 参考性能 ==========
    std::cout << "\n========== cuBLAS GEMM (参考性能) ==========" << std::endl;
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const fp32 alpha = 1.0f;
    const fp32 beta = 0.0f;
    
    // 预热
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 多次运行取平均值
    total_time = 0.0;
    for (int i = 0; i < num_runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_time += duration.count() / 1000.0;
    }
    double time_cublas_ms = total_time / num_runs;
    
    print_performance_analysis("cuBLAS (参考)", time_cublas_ms, M, N, K);
    
    cublasDestroy(handle);
    
    // ========== 性能对比总结 ==========
    std::cout << "\n========== 性能对比总结 ==========" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CPU:              " << time_cpu_ms << " ms" << std::endl;
    std::cout << "GPU (基础):       " << time_gpu_basic_ms << " ms" 
              << " (加速比: " << time_cpu_ms / time_gpu_basic_ms << "x)" << std::endl;
    std::cout << "GPU (优化):       " << time_gpu_tiled_ms << " ms" 
              << " (加速比: " << time_cpu_ms / time_gpu_tiled_ms << "x)" << std::endl;
    std::cout << "cuBLAS (参考):    " << time_cublas_ms << " ms" 
              << " (加速比: " << time_cpu_ms / time_cublas_ms << "x)" << std::endl;
    
    // 计算 GFLOPS 对比
    long long flops = (long long)M * N * 2 * K;
    double gflops_cpu = (flops / 1e9) / (time_cpu_ms / 1000.0);
    double gflops_gpu_basic = (flops / 1e9) / (time_gpu_basic_ms / 1000.0);
    double gflops_gpu_tiled = (flops / 1e9) / (time_gpu_tiled_ms / 1000.0);
    double gflops_cublas = (flops / 1e9) / (time_cublas_ms / 1000.0);
    
    std::cout << "\n算力对比 (GFLOPS):" << std::endl;
    std::cout << "CPU:              " << gflops_cpu << " GFLOPS" << std::endl;
    std::cout << "GPU (基础):       " << gflops_gpu_basic << " GFLOPS" << std::endl;
    std::cout << "GPU (优化):       " << gflops_gpu_tiled << " GFLOPS" << std::endl;
    std::cout << "cuBLAS (参考):    " << gflops_cublas << " GFLOPS" << std::endl;
    
    // 清理资源
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu_basic);
    free(h_C_gpu_tiled);
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    std::cout << "\n========== 测试完成 ==========" << std::endl;
    
    return 0;
}

