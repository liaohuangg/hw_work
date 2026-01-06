#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>

// 数据类型定义
typedef float fp32;

// 矩阵大小（可以根据需要调整）
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

/**
 * 基础 CUDA GEMM Kernel（不使用共享内存，作为性能基准）
 * 每个线程计算 C 矩阵的一个元素
 */
__global__ void gemm_kernel_basic(
    const fp32* __restrict__ A,  // [M][K]
    const fp32* __restrict__ B,  // [K][N]
    fp32* __restrict__ C,        // [M][N]
    int M, int N, int K
) {
    // 计算当前线程负责的 C 矩阵元素位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查
    if (row < M && col < N) {
        fp32 sum = 0.0f;
        // 累加 A[row][k] * B[k][col]
        // 注意：A 按行访问（coalesced），B 按列访问（非 coalesced）
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/**
 * GEMM Kernel：Global Memory 分块放入 Shared Memory（参数化版本）
 * 
 * 优化技术：
 * 1. 将 global memory 中的数据分块（tile）加载到 shared memory
 * 2. 利用 shared memory 的低延迟特性
 * 3. 数据重用：shared memory 中的数据被多个线程复用
 * 
 * 内存访问优化：
 * - 原始 naive 算法：每个线程直接从 global memory 读取，访问量为 M*N*K*2
 * - 分块算法：访问量减少为 M*N*K*(1/bm + 1/bn)，其中 bm 和 bn 是 tile 大小
 * - 内存访问减少比例：1/2 * (1/bm + 1/bn)
 * 
 * 模板参数：
 * - BM: block 在 M 维度的大小（tile size，对应 BLOCK_SIZE）
 * - BN: block 在 N 维度的大小（tile size，对应 BLOCK_SIZE）
 * - BK: block 在 K 维度的大小（tile size，对应 BLOCK_SIZE）
 * 
 * 注意：此实现按照标准分块矩阵乘法模式，使用 row = threadIdx.x + blockIdx.x * blockDim.x
 */
template<int BM, int BN, int BK>
__global__ void gemm_kernel_global_blocking_shared(
    const fp32* __restrict__ A,
    const fp32* __restrict__ B,
    fp32* __restrict__ C,
    int M, int N, int K
) {
    // Shared memory：缓存 global memory 的分块数据
    // BM * BK 的 A tile 和 BK * BN 的 B tile
    __shared__ fp32 Mds[BM][BK];  // 对应图片中的 Mds[BLOCK_SIZE][BLOCK_SIZE]
    __shared__ fp32 Nds[BK][BN];  // 对应图片中的 Nds[BLOCK_SIZE][BLOCK_SIZE]
    
    // 线程在 block 中的位置
    int tx = threadIdx.x;  // 范围: [0, BN)
    int ty = threadIdx.y;  // 范围: [0, BM)
    
    // 计算全局行列索引（每个线程负责计算 C 矩阵的一个元素）
    int row = blockIdx.y * BM + ty;  // 全局行索引
    int col = blockIdx.x * BN + tx;  // 全局列索引
    
    // 累加结果
    fp32 P = 0.0f;
    
    // 外层循环：遍历 K 维度的分块（对应图片中的 for(int i = 0; i < N/BLOCK_SIZE; i++)）
    // 这里 N 对应 K 维度，BK 对应 BLOCK_SIZE
    int num_tiles_k = (K + BK - 1) / BK;
    
    for (int i = 0; i < num_tiles_k; i++) {
        // 加载 A 的 tile 到 shared memory
        // Mds 的大小是 [BM][BK]，需要 BM * BK 个元素
        // 当 BM = BN = BK 时，每个线程正好加载一个元素
        int a_row = blockIdx.y * BM + ty;
        int a_col = i * BK + tx;
        if (a_row < M && a_col < K) {
            Mds[ty][tx] = A[a_row * K + a_col];
        } else {
            Mds[ty][tx] = 0.0f;
        }
        
        // 加载 B 的 tile 到 shared memory
        // Nds 的大小是 [BK][BN]，需要 BK * BN 个元素
        // 当 BM = BN = BK 时，每个线程正好加载一个元素
        int b_row = i * BK + ty;
        int b_col = blockIdx.x * BN + tx;
        if (b_row < K && b_col < N) {
            Nds[ty][tx] = B[b_row * N + b_col];
        } else {
            Nds[ty][tx] = 0.0f;
        }
        
        // 同步：确保所有线程都完成数据加载
        __syncthreads();
        
        // 内层循环：计算点积
        for (int j = 0; j < BK; j++) {
            P += Mds[ty][j] * Nds[j][tx];
        }
        
        // 同步：确保所有线程都完成计算，再加载下一个 tile
        __syncthreads();
    }
    
    // 写入结果到 global memory（对应图片中的 d_C[row * K + col] = P）
    // 注意：图片中使用的是 row * K，但 C 是 M×N，所以应该是 row * N + col
    if (row < M && col < N) {
        C[row * N + col] = P;
    }
}

/**
 * 验证优化版本与baseline版本的计算结果是否一致
 */
void verify_results(
    const fp32* C_baseline,
    const fp32* C_optimized,
    int size,
    const std::string& optimization_name,
    fp32 tolerance = 1e-3f
) {
    fp32 max_abs_error = 0.0f;
    fp32 max_rel_error = 0.0f;
    fp32 sum_abs_error = 0.0f;
    fp32 sum_rel_error = 0.0f;
    int error_count = 0;
    int total_count = 0;
    
    for (int i = 0; i < size; i++) {
        fp32 diff = std::abs(C_baseline[i] - C_optimized[i]);
        fp32 max_val = std::max(std::abs(C_baseline[i]), std::abs(C_optimized[i]));
        
        max_abs_error = std::max(max_abs_error, diff);
        sum_abs_error += diff;
        
        if (max_val > 1e-6f) {
            fp32 rel_error = diff / max_val;
            max_rel_error = std::max(max_rel_error, rel_error);
            sum_rel_error += rel_error;
            total_count++;
            
            if (rel_error > tolerance) {
                error_count++;
            }
        }
    }
    
    fp32 avg_abs_error = sum_abs_error / size;
    fp32 avg_rel_error = (total_count > 0) ? (sum_rel_error / total_count) : 0.0f;
    
    std::cout << "\n========== " << optimization_name << " 数据验证结果 ==========" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "最大绝对误差: " << max_abs_error << std::endl;
    std::cout << "平均绝对误差: " << avg_abs_error << std::endl;
    std::cout << "最大相对误差: " << max_rel_error << std::endl;
    std::cout << "平均相对误差: " << avg_rel_error << std::endl;
    std::cout << "误差超过阈值 (" << tolerance << ") 的元素数量: " << error_count << " / " << size << std::endl;
    
    if (error_count == 0) {
        std::cout << "✓ 验证通过：优化版本与baseline版本计算结果一致！" << std::endl;
    } else {
        std::cout << "✗ 验证失败：有 " << error_count << " 个元素的相对误差超过阈值" << std::endl;
        
        // 打印前几个错误示例
        int sample_count = 0;
        for (int i = 0; i < size && sample_count < 5; i++) {
            fp32 diff = std::abs(C_baseline[i] - C_optimized[i]);
            fp32 max_val = std::max(std::abs(C_baseline[i]), std::abs(C_optimized[i]));
            if (max_val > 1e-6f && diff / max_val > tolerance) {
                std::cout << "  示例 " << (sample_count + 1) << ": 位置[" << i << "] "
                          << "Baseline=" << C_baseline[i] << ", "
                          << "Optimized=" << C_optimized[i] << ", "
                          << "相对误差=" << (diff / max_val) << std::endl;
                sample_count++;
            }
        }
    }
    std::cout << std::defaultfloat;
}

/**
 * 打印详细的性能分析结果
 */
void print_performance_analysis(
    const std::string& name,
    double time_ms,
    int M, int N, int K,
    int tile_size = 0
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
    
    if (tile_size > 0) {
        std::cout << "Tile 大小: " << tile_size << "x" << tile_size << std::endl;
        std::cout << "共享内存使用: " << (2 * tile_size * tile_size * sizeof(fp32) / 1024.0) << " KB per block" << std::endl;
    }
    
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
    std::cout << "========== CUDA GEMM Shared Memory 优化性能测试 ==========" << std::endl;
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
    fp32* h_C_baseline = (fp32*)malloc(size_C);
    fp32* h_C_optimized = (fp32*)malloc(size_C);
    
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
    
    // ========== Baseline 版本 ==========
    std::cout << "\n========== GPU GEMM (Baseline - 无任何优化) ==========" << std::endl;
    
    dim3 blockSize_basic(16, 16);
    dim3 gridSize_basic((N + blockSize_basic.x - 1) / blockSize_basic.x, 
                        (M + blockSize_basic.y - 1) / blockSize_basic.y);
    
    // 预热
    gemm_kernel_basic<<<gridSize_basic, blockSize_basic>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 单次运行测量时间
    CUDA_CHECK(cudaMemset(d_C, 0, size_C));
    
    auto start = std::chrono::high_resolution_clock::now();
    gemm_kernel_basic<<<gridSize_basic, blockSize_basic>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double time_baseline_ms = duration.count() / 1000.0;
    
    CUDA_CHECK(cudaMemcpy(h_C_baseline, d_C, size_C, cudaMemcpyDeviceToHost));
    print_performance_analysis("Baseline (基础 Kernel - 无任何优化)", time_baseline_ms, M, N, K);
    
    // ========== Shared Memory 优化版本（参数搜索）==========
    std::cout << "\n========== GPU GEMM (Global Memory 分块 -> Shared Memory) ==========" << std::endl;
    std::cout << "测试不同的 Tile Size (BM×BN×BK) 参数..." << std::endl;
    
    // 测试不同的 tile size 组合
    struct TileConfig {
        int bm, bn, bk;
    };
    TileConfig tile_configs[] = {
        {8, 8, 8},
        {16, 16, 16},
        {24, 24, 24},
        {32, 32, 32}
    };
    int num_tile_configs = sizeof(tile_configs) / sizeof(tile_configs[0]);
    double best_time_optimized = 1e10;
    TileConfig best_tile_config = {32, 32, 32};
    std::vector<double> times_optimized(num_tile_configs);
    std::vector<double> gflops_optimized(num_tile_configs);
    
    for (int tile_idx = 0; tile_idx < num_tile_configs; tile_idx++) {
        int bm = tile_configs[tile_idx].bm;
        int bn = tile_configs[tile_idx].bn;
        int bk = tile_configs[tile_idx].bk;
        
        // 检查 shared memory 限制和 block size 限制
        size_t shared_mem_size = (bm * bk + bk * bn) * sizeof(fp32);
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        
        // 检查每个block的线程数
        int threads_per_block = bm * bn;
        if (threads_per_block > prop.maxThreadsPerBlock) {
            std::cout << "跳过 BM=" << bm << ", BN=" << bn << ", BK=" << bk 
                      << " (每个block线程数: " << threads_per_block 
                      << ", 超过限制: " << prop.maxThreadsPerBlock << ")" << std::endl;
            times_optimized[tile_idx] = -1;
            continue;
        }
        
        if (shared_mem_size > prop.sharedMemPerBlock) {
            std::cout << "跳过 BM=" << bm << ", BN=" << bn << ", BK=" << bk 
                      << " (共享内存需求: " << (shared_mem_size / 1024.0) 
                      << " KB, 超过限制: " << (prop.sharedMemPerBlock / 1024.0) << " KB)" << std::endl;
            times_optimized[tile_idx] = -1;
            continue;
        }
        
        dim3 blockSize_tile(bn, bm);
        dim3 gridSize_tile((N + bn - 1) / bn, (M + bm - 1) / bm);
        
        std::cout << "\n--- 测试 BM=" << bm << ", BN=" << bn << ", BK=" << bk << " ---" << std::endl;
        std::cout << "共享内存使用: " << (shared_mem_size / 1024.0) << " KB per block" << std::endl;
        
        // 预热
        if (bm == 8 && bn == 8 && bk == 8) {
            gemm_kernel_global_blocking_shared<8, 8, 8><<<gridSize_tile, blockSize_tile>>>(d_A, d_B, d_C, M, N, K);
        } else if (bm == 16 && bn == 16 && bk == 16) {
            gemm_kernel_global_blocking_shared<16, 16, 16><<<gridSize_tile, blockSize_tile>>>(d_A, d_B, d_C, M, N, K);
        } else if (bm == 24 && bn == 24 && bk == 24) {
            gemm_kernel_global_blocking_shared<24, 24, 24><<<gridSize_tile, blockSize_tile>>>(d_A, d_B, d_C, M, N, K);
        } else if (bm == 32 && bn == 32 && bk == 32) {
            gemm_kernel_global_blocking_shared<32, 32, 32><<<gridSize_tile, blockSize_tile>>>(d_A, d_B, d_C, M, N, K);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 单次运行测量时间
        CUDA_CHECK(cudaMemset(d_C, 0, size_C));
        
        auto start_tile = std::chrono::high_resolution_clock::now();
        
        if (bm == 8 && bn == 8 && bk == 8) {
            gemm_kernel_global_blocking_shared<8, 8, 8><<<gridSize_tile, blockSize_tile>>>(d_A, d_B, d_C, M, N, K);
        } else if (bm == 16 && bn == 16 && bk == 16) {
            gemm_kernel_global_blocking_shared<16, 16, 16><<<gridSize_tile, blockSize_tile>>>(d_A, d_B, d_C, M, N, K);
        } else if (bm == 24 && bn == 24 && bk == 24) {
            gemm_kernel_global_blocking_shared<24, 24, 24><<<gridSize_tile, blockSize_tile>>>(d_A, d_B, d_C, M, N, K);
        } else if (bm == 32 && bn == 32 && bk == 32) {
            gemm_kernel_global_blocking_shared<32, 32, 32><<<gridSize_tile, blockSize_tile>>>(d_A, d_B, d_C, M, N, K);
        }
        
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end_tile = std::chrono::high_resolution_clock::now();
        
        auto duration_tile = std::chrono::duration_cast<std::chrono::microseconds>(end_tile - start_tile);
        double time_ms = duration_tile.count() / 1000.0;
        times_optimized[tile_idx] = time_ms;
        
        long long flops = (long long)M * N * 2 * K;
        double gflops = (flops / 1e9) / (time_ms / 1000.0);
        gflops_optimized[tile_idx] = gflops;
        
        std::cout << "BM=" << bm << ", BN=" << bn << ", BK=" << bk << ": " << time_ms << " ms, " << gflops << " GFLOPS";
        if (time_ms < best_time_optimized) {
            best_time_optimized = time_ms;
            best_tile_config = tile_configs[tile_idx];
            std::cout << " (当前最佳)";
        }
        std::cout << std::endl;
    }
    
    // 使用最佳参数再次运行并保存结果
    std::cout << "\n使用最佳 Tile Size BM=" << best_tile_config.bm << ", BN=" << best_tile_config.bn 
              << ", BK=" << best_tile_config.bk << " 运行..." << std::endl;
    
    dim3 blockSize_optimized(best_tile_config.bn, best_tile_config.bm);
    dim3 gridSize_optimized((N + best_tile_config.bn - 1) / best_tile_config.bn, 
                            (M + best_tile_config.bm - 1) / best_tile_config.bm);
    
    // 预热
    if (best_tile_config.bm == 8 && best_tile_config.bn == 8 && best_tile_config.bk == 8) {
        gemm_kernel_global_blocking_shared<8, 8, 8><<<gridSize_optimized, blockSize_optimized>>>(d_A, d_B, d_C, M, N, K);
    } else if (best_tile_config.bm == 16 && best_tile_config.bn == 16 && best_tile_config.bk == 16) {
        gemm_kernel_global_blocking_shared<16, 16, 16><<<gridSize_optimized, blockSize_optimized>>>(d_A, d_B, d_C, M, N, K);
    } else if (best_tile_config.bm == 24 && best_tile_config.bn == 24 && best_tile_config.bk == 24) {
        gemm_kernel_global_blocking_shared<24, 24, 24><<<gridSize_optimized, blockSize_optimized>>>(d_A, d_B, d_C, M, N, K);
    } else if (best_tile_config.bm == 32 && best_tile_config.bn == 32 && best_tile_config.bk == 32) {
        gemm_kernel_global_blocking_shared<32, 32, 32><<<gridSize_optimized, blockSize_optimized>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 单次运行测量时间
    CUDA_CHECK(cudaMemset(d_C, 0, size_C));
    
    auto start_best = std::chrono::high_resolution_clock::now();
    
    if (best_tile_config.bm == 8 && best_tile_config.bn == 8 && best_tile_config.bk == 8) {
        gemm_kernel_global_blocking_shared<8, 8, 8><<<gridSize_optimized, blockSize_optimized>>>(d_A, d_B, d_C, M, N, K);
    } else if (best_tile_config.bm == 16 && best_tile_config.bn == 16 && best_tile_config.bk == 16) {
        gemm_kernel_global_blocking_shared<16, 16, 16><<<gridSize_optimized, blockSize_optimized>>>(d_A, d_B, d_C, M, N, K);
    } else if (best_tile_config.bm == 24 && best_tile_config.bn == 24 && best_tile_config.bk == 24) {
        gemm_kernel_global_blocking_shared<24, 24, 24><<<gridSize_optimized, blockSize_optimized>>>(d_A, d_B, d_C, M, N, K);
    } else if (best_tile_config.bm == 32 && best_tile_config.bn == 32 && best_tile_config.bk == 32) {
        gemm_kernel_global_blocking_shared<32, 32, 32><<<gridSize_optimized, blockSize_optimized>>>(d_A, d_B, d_C, M, N, K);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end_best = std::chrono::high_resolution_clock::now();
    
    auto duration_best = std::chrono::duration_cast<std::chrono::microseconds>(end_best - start_best);
    double time_optimized_ms = duration_best.count() / 1000.0;
    
    CUDA_CHECK(cudaMemcpy(h_C_optimized, d_C, size_C, cudaMemcpyDeviceToHost));
    print_performance_analysis("Shared Memory 优化 (BM=" + std::to_string(best_tile_config.bm) + 
                               ", BN=" + std::to_string(best_tile_config.bn) + ", BK=" + std::to_string(best_tile_config.bk) + ")", 
                               time_optimized_ms, M, N, K);
    
    // ========== 数据验证 ==========
    verify_results(h_C_baseline, h_C_optimized, M * N, "Shared Memory 优化");
    
    // ========== 性能对比总结 ==========
    std::cout << "\n========== 性能对比总结 ==========" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    
    long long flops = (long long)M * N * 2 * K;
    double gflops_baseline = (flops / 1e9) / (time_baseline_ms / 1000.0);
    double gflops_optimized_final = (flops / 1e9) / (time_optimized_ms / 1000.0);
    
    std::cout << "\n========== Baseline vs Shared Memory 优化对比 ==========" << std::endl;
    std::cout << "1. Baseline (基础 - 无任何优化):       " << time_baseline_ms << " ms, " << gflops_baseline << " GFLOPS" << std::endl;
    std::cout << "2. Shared Memory 优化: " << time_optimized_ms << " ms, " << gflops_optimized_final << " GFLOPS" << std::endl;
    std::cout << "\n性能提升:" << std::endl;
    std::cout << "  - 时间减少: " << ((time_baseline_ms - time_optimized_ms) / time_baseline_ms * 100) << "%" << std::endl;
    std::cout << "  - 速度提升: " << (time_baseline_ms / time_optimized_ms) << "x" << std::endl;
    std::cout << "  - GFLOPS 提升: " << ((gflops_optimized_final - gflops_baseline) / gflops_baseline * 100) << "%" << std::endl;
    
    std::cout << "\n========== 优化分析 ==========" << std::endl;
    std::cout << "Shared Memory 优化相对Baseline提升: " 
              << (time_baseline_ms / time_optimized_ms - 1.0) * 100 << "%" << std::endl;
    std::cout << "   - 内存访问减少比例：1/2 * (1/bm + 1/bn) = 1/2 * (1/" << best_tile_config.bm 
              << " + 1/" << best_tile_config.bn << ") = " 
              << (0.5 * (1.0/best_tile_config.bm + 1.0/best_tile_config.bn)) << std::endl;
    std::cout << "   - 利用 shared memory 低延迟特性" << std::endl;
    std::cout << "   - 数据重用：shared memory 中的数据被多个线程复用" << std::endl;
    
    // 打印 Tile Size 参数搜索总结
    std::cout << "\n========== Tile Size 参数搜索总结 ==========" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    for (int tile_idx = 0; tile_idx < num_tile_configs; tile_idx++) {
        if (times_optimized[tile_idx] > 0) {
            std::cout << "BM=" << tile_configs[tile_idx].bm << ", BN=" << tile_configs[tile_idx].bn 
                      << ", BK=" << tile_configs[tile_idx].bk << ": " << times_optimized[tile_idx] << " ms, " 
                      << gflops_optimized[tile_idx] << " GFLOPS";
            if (tile_configs[tile_idx].bm == best_tile_config.bm && 
                tile_configs[tile_idx].bn == best_tile_config.bn && 
                tile_configs[tile_idx].bk == best_tile_config.bk) {
                std::cout << " (最佳)";
            }
            std::cout << std::endl;
        }
    }
    
    // 清理资源
    free(h_A);
    free(h_B);
    free(h_C_baseline);
    free(h_C_optimized);
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    std::cout << "\n========== 测试完成 ==========" << std::endl;
    
    return 0;
}

