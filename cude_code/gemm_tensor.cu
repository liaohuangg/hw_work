#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cmath>

using namespace nvcuda;

// 矩阵大小
const int M = 20480;
const int N = 2048;
const int K = 8192;

// CUDA 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Tensor Core tile size (WMMA API)
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

/**
 * CUDA Core GEMM Kernel (FP32 精度计算) - 参考 baseline 实现
 * 每个线程计算 C 矩阵的一个元素
 * 输入数据为 FP16，但计算使用 FP32 精度以提高准确性
 */
__global__ void gemm_kernel_cuda_core_fp32(
    const __half* __restrict__ A,  // [M][K] - FP16 输入
    const __half* __restrict__ B,  // [K][N] - FP16 输入
    __half* __restrict__ C,        // [M][N] - FP16 输出
    int M, int N, int K
) {
    // 计算当前线程负责的 C 矩阵元素位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查
    if (row < M && col < N) {
        // 使用 FP32 精度进行累加计算
        float sum = 0.0f;
        // 累加 A[row][k] * B[k][col]
        // 将 FP16 输入转换为 FP32 进行计算，提高精度
        for (int k = 0; k < K; k++) {
            float a_val = __half2float(A[row * K + k]);
            float b_val = __half2float(B[k * N + col]);
            sum += a_val * b_val;  // FP32 精度计算
        }
        // 将 FP32 结果转换为 FP16 存储
        C[row * N + col] = __float2half(sum);
    }
}

/**
 * Tensor Core GEMM Kernel (FP16)
 * 使用 WMMA API 调用 Tensor Core
 * 每个 warp 处理一个 16x16 的 tile
 */
__global__ void gemm_kernel_tensor_core_fp16(
    const __half* __restrict__ A,  // [M][K]
    const __half* __restrict__ B,  // [K][N]
    __half* __restrict__ C,        // [M][N]
    int M, int N, int K
) {
    // 计算当前 warp 在 grid 中的位置
    // 每个 warp 处理一个 16x16 的 tile
    int warpM = blockIdx.y;
    int warpN = blockIdx.x;
    
    // 计算当前 warp 负责的 C 矩阵 tile 位置
    int tileM = warpM * WMMA_M;
    int tileN = warpN * WMMA_N;
    
    // 声明 WMMA fragments（每个 warp 共享）
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // 初始化 accumulator fragment
    wmma::fill_fragment(c_frag, 0.0f);
    
    // K 维度分块迭代
    int num_tiles_k = (K + WMMA_K - 1) / WMMA_K;
    
    for (int tile_k = 0; tile_k < num_tiles_k; tile_k++) {
        int tileK = tile_k * WMMA_K;
        
        // 加载 A 的 tile (row-major)
        // 边界检查在 load_matrix_sync 内部处理
        if (tileM < M && tileK < K) {
            wmma::load_matrix_sync(a_frag, A + tileM * K + tileK, K);
        } else {
            wmma::fill_fragment(a_frag, __float2half(0.0f));
        }
        
        // 加载 B 的 tile (col-major)
        if (tileK < K && tileN < N) {
            wmma::load_matrix_sync(b_frag, B + tileK * N + tileN, N);
        } else {
            wmma::fill_fragment(b_frag, __float2half(0.0f));
        }
        
        // Tensor Core 矩阵乘法
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // 存储结果（需要将 float accumulator 转换为 __half）
    if (tileM < M && tileN < N) {
        // 先存储到 shared memory（float 类型）
        __shared__ float temp_C[WMMA_M * WMMA_N];
        wmma::store_matrix_sync(temp_C, c_frag, WMMA_N, wmma::mem_row_major);
        __syncthreads();
        
        // 将 float 转换为 __half 并存储到 global memory
        // 每个线程处理多个元素
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        int num_threads = blockDim.x * blockDim.y;
        
        for (int i = tid; i < WMMA_M * WMMA_N; i += num_threads) {
            int row_in_tile = i / WMMA_N;
            int col_in_tile = i % WMMA_N;
            int global_row = tileM + row_in_tile;
            int global_col = tileN + col_in_tile;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = __float2half(temp_C[i]);
            }
        }
    }
}

/**
 * 验证两个结果矩阵的精度差异
 */
void verify_precision(
    const __half* C_tensor,
    const __half* C_cuda_core,
    int M, int N,
    const std::string& name
) {
    std::cout << "\n========== " << name << " ==========" << std::endl;
    
    double max_diff = 0.0;
    double max_rel_diff = 0.0;
    double sum_diff = 0.0;
    double sum_sq_diff = 0.0;
    int count = 0;
    int large_diff_count = 0;
    
    for (int i = 0; i < M * N; i++) {
        float val_tensor = __half2float(C_tensor[i]);
        float val_cuda = __half2float(C_cuda_core[i]);
        
        float diff = std::abs(val_tensor - val_cuda);
        float rel_diff = (val_cuda != 0.0f) ? std::abs(diff / val_cuda) : diff;
        
        max_diff = std::max(max_diff, (double)diff);
        max_rel_diff = std::max(max_rel_diff, (double)rel_diff);
        sum_diff += diff;
        sum_sq_diff += diff * diff;
        count++;
        
        if (diff > 0.1f) {
            large_diff_count++;
        }
    }
    
    double mean_diff = sum_diff / count;
    double rms_diff = std::sqrt(sum_sq_diff / count);
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "最大绝对误差: " << max_diff << std::endl;
    std::cout << "最大相对误差: " << max_rel_diff << std::endl;
    std::cout << "平均绝对误差: " << mean_diff << std::endl;
    std::cout << "RMS 误差: " << rms_diff << std::endl;
    std::cout << "误差 > 0.1 的元素数: " << large_diff_count << " / " << count 
              << " (" << (100.0 * large_diff_count / count) << "%)" << std::endl;
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
    
    // 计算内存带宽（字节）- FP16 是 2 字节
    long long bytes = (long long)(M * K + K * N + M * N) * sizeof(__half);
    
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
    std::cout << "矩阵大小: A[" << M << "][" << K << "] * B[" << K << "][" << N << "] = C[" << M << "][" << N << "]" << std::endl;
    std::cout << "数据大小: A=" << (M*K*sizeof(__half)/1e6) << " MB, "
              << "B=" << (K*N*sizeof(__half)/1e6) << " MB, "
              << "C=" << (M*N*sizeof(__half)/1e6) << " MB" << std::endl;
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
    
    // 检查是否支持 Tensor Core
    bool supports_tensor_core = (prop.major >= 7) || (prop.major == 6 && prop.minor == 0);
    std::cout << "Tensor Core 支持: " << (supports_tensor_core ? "是" : "否") << std::endl;
    
    std::cout << "全局内存: " << prop.totalGlobalMem / 1e9 << " GB" << std::endl;
    std::cout << "共享内存 per block: " << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个 block 的最大线程数: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "SM 数量: " << prop.multiProcessorCount << std::endl;
    std::cout << "Warp 大小: " << prop.warpSize << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "========== CUDA Tensor Core vs CUDA Core GEMM 性能对比 ==========" << std::endl;
    std::cout << "矩阵大小: A[" << M << "][" << K << "] * B[" << K << "][" << N << "] = C[" << M << "][" << N << "]" << std::endl;
    std::cout << "数据类型: FP16 (half precision)" << std::endl;
    
    // 打印 GPU 信息
    print_gpu_info();
    
    // 分配主机内存
    size_t size_A = M * K * sizeof(__half);
    size_t size_B = K * N * sizeof(__half);
    size_t size_C = M * N * sizeof(__half);
    
    __half* h_A = (__half*)malloc(size_A);
    __half* h_B = (__half*)malloc(size_B);
    __half* h_C_tensor = (__half*)malloc(size_C);
    __half* h_C_cuda_core = (__half*)malloc(size_C);
    
    // 初始化矩阵（使用随机值，转换为 FP16）
    std::cout << "\n初始化矩阵数据..." << std::endl;
    srand(42);  // 固定随机种子以便复现
    for (int i = 0; i < M * K; i++) {
        float val = static_cast<float>(rand()) / RAND_MAX;
        h_A[i] = __float2half(val);
    }
    for (int i = 0; i < K * N; i++) {
        float val = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = __float2half(val);
    }
    
    // 分配设备内存
    __half* d_A;
    __half* d_B;
    __half* d_C_tensor;
    __half* d_C_cuda_core;
    
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C_tensor, size_C));
    CUDA_CHECK(cudaMalloc(&d_C_cuda_core, size_C));
    
    // 拷贝数据到设备
    std::cout << "拷贝数据到 GPU..." << std::endl;
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    
    // ========== CUDA Core 版本 ==========
    std::cout << "\n========== CUDA Core GEMM (FP32 精度计算) ==========" << std::endl;
    
    dim3 blockSize_cuda_core(16, 16);
    dim3 gridSize_cuda_core((N + blockSize_cuda_core.x - 1) / blockSize_cuda_core.x, 
                            (M + blockSize_cuda_core.y - 1) / blockSize_cuda_core.y);
    
    // 预热
    gemm_kernel_cuda_core_fp32<<<gridSize_cuda_core, blockSize_cuda_core>>>(d_A, d_B, d_C_cuda_core, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 单次运行测量时间
    CUDA_CHECK(cudaMemset(d_C_cuda_core, 0, size_C));
    
    auto start_cuda_core = std::chrono::high_resolution_clock::now();
    gemm_kernel_cuda_core_fp32<<<gridSize_cuda_core, blockSize_cuda_core>>>(d_A, d_B, d_C_cuda_core, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end_cuda_core = std::chrono::high_resolution_clock::now();
    
    auto duration_cuda_core = std::chrono::duration_cast<std::chrono::microseconds>(end_cuda_core - start_cuda_core);
    double time_cuda_core_ms = duration_cuda_core.count() / 1000.0;
    
    CUDA_CHECK(cudaMemcpy(h_C_cuda_core, d_C_cuda_core, size_C, cudaMemcpyDeviceToHost));
    print_performance_analysis("CUDA Core (FP32 精度)", time_cuda_core_ms, M, N, K);
    
    // ========== Tensor Core 版本 ==========
    std::cout << "\n========== Tensor Core GEMM (FP16) ==========" << std::endl;
    
    // Tensor Core 使用 warp-level 编程
    // 每个 warp 处理一个 16x16 的 tile
    dim3 blockSize_tensor(32, 4);  // 128 threads = 4 warps
    dim3 gridSize_tensor((N + WMMA_N - 1) / WMMA_N, 
                         (M + WMMA_M - 1) / WMMA_M);
    
    // 预热
    gemm_kernel_tensor_core_fp16<<<gridSize_tensor, blockSize_tensor>>>(d_A, d_B, d_C_tensor, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 单次运行测量时间
    CUDA_CHECK(cudaMemset(d_C_tensor, 0, size_C));
    
    auto start_tensor = std::chrono::high_resolution_clock::now();
    gemm_kernel_tensor_core_fp16<<<gridSize_tensor, blockSize_tensor>>>(d_A, d_B, d_C_tensor, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end_tensor = std::chrono::high_resolution_clock::now();
    
    auto duration_tensor = std::chrono::duration_cast<std::chrono::microseconds>(end_tensor - start_tensor);
    double time_tensor_ms = duration_tensor.count() / 1000.0;
    
    CUDA_CHECK(cudaMemcpy(h_C_tensor, d_C_tensor, size_C, cudaMemcpyDeviceToHost));
    print_performance_analysis("Tensor Core (FP16)", time_tensor_ms, M, N, K);
    
    // ========== 性能对比 ==========
    std::cout << "\n========== 性能对比总结 ==========" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CUDA Core (FP32 精度) 时间: " << time_cuda_core_ms << " ms" << std::endl;
    std::cout << "Tensor Core (FP16) 时间: " << time_tensor_ms << " ms" << std::endl;
    std::cout << "加速比: " << (time_cuda_core_ms / time_tensor_ms) << "x" << std::endl;
    
    // ========== 精度验证 ==========
    verify_precision(h_C_tensor, h_C_cuda_core, M, N, "精度对比 (Tensor Core FP16 vs CUDA Core FP32)");
    
    // 清理资源
    free(h_A);
    free(h_B);
    free(h_C_tensor);
    free(h_C_cuda_core);
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_tensor));
    CUDA_CHECK(cudaFree(d_C_cuda_core));
    
    std::cout << "\n========== 测试完成 ==========" << std::endl;
    
    return 0;
}

