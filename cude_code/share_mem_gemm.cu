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
const int M = 2048;
const int N = 2048;
const int K = 2048;

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
 * - BM: block 在 M 维度的大小（tile size）
 * - BN: block 在 N 维度的大小（tile size）
 * - BK: block 在 K 维度的大小（tile size）
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
    __shared__ fp32 tileA[BM][BK];
    __shared__ fp32 tileB[BK][BN];
    
    // 线程在 block 中的位置
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 当前 block 负责计算的 C 矩阵块的位置
    int block_row = blockIdx.y;  // M 维度上的 block 索引
    int block_col = blockIdx.x;  // N 维度上的 block 索引
    
    // 当前线程负责计算的 C 矩阵元素位置
    int row = block_row * BM + ty;
    int col = block_col * BN + tx;
    
    // 累加结果
    fp32 sum = 0.0f;
    
    // K 维度分块迭代
    // 总共需要 K/BK 次迭代
    int num_tiles_k = (K + BK - 1) / BK;
    
    for (int tile_k = 0; tile_k < num_tiles_k; tile_k++) {
        // 加载 A 的 tile (BM * BK) 到 shared memory
        int a_row = row;
        int a_col = tile_k * BK + tx;
        
        if (a_row < M && a_col < K) {
            tileA[ty][tx] = A[a_row * K + a_col];
        } else {
            tileA[ty][tx] = 0.0f;
        }
        
        // 加载 B 的 tile (BK * BN) 到 shared memory
        int b_row = tile_k * BK + ty;
        int b_col = col;
        
        if (b_row < K && b_col < N) {
            tileB[ty][tx] = B[b_row * N + b_col];
        } else {
            tileB[ty][tx] = 0.0f;
        }
        
        // 同步：确保所有线程都完成数据加载
        __syncthreads();
        
        // 计算：从 shared memory 读取数据进行计算
        for (int k = 0; k < BK; k++) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        
        // 同步：确保所有线程都完成计算，再加载下一个 tile
        __syncthreads();
    }
    
    // 写入结果到 global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * GEMM Kernel：Global Memory 分块放入 Shared Memory（数据预取版本）
 * 
 * 优化技术：
 * 1. Global Memory -> Shared Memory：分块加载，减少 global memory 访问
 * 2. 数据预取（Prefetch）：使用双缓冲（Double Buffering）掩盖访存延迟
 *    - Shared Memory Prefetch：在计算当前tile的同时，预取下一个tile到shared memory
 *    - Register Prefetch：在计算当前register数据的同时，预取下一个register数据
 * 
 * 预取原理：
 * - 问题：从global memory加载数据到shared memory需要很长时间，导致计算单元等待
 * - 解决方案：使用双缓冲，读写分离
 *   - Buffer 0：当前用于计算的tile
 *   - Buffer 1：预取的下一个tile
 *   - 在计算Buffer 0的同时，将下一个tile加载到Buffer 1
 *   - 下一轮迭代时，直接使用Buffer 1，同时预取到Buffer 0
 * 
 * 预取流程：
 * 1. 预取第一个tile到Buffer 0
 * 2. 对于每个后续tile：
 *    - 如果还有下一个tile，预取到另一个buffer
 *    - 计算当前buffer的数据
 *    - 切换buffer
 */
/**
 * 模板参数：
 * - BM: block 在 M 维度的大小（tile size）
 * - BN: block 在 N 维度的大小（tile size）
 * - BK: block 在 K 维度的大小（tile size）
 */
template<int BM, int BN, int BK>
__global__ void gemm_kernel_global_blocking_shared_prefetch(
    const fp32* __restrict__ A,
    const fp32* __restrict__ B,
    fp32* __restrict__ C,
    int M, int N, int K
) {
    // Shared memory：使用双缓冲进行prefetch
    // 双缓冲：tileA[0]和tileA[1]，tileB[0]和tileB[1]
    __shared__ fp32 tileA[2][BM][BK];
    __shared__ fp32 tileB[2][BK][BN];
    
    // 线程在 block 中的位置
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 当前 block 负责计算的 C 矩阵块的位置
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    
    // 当前线程负责计算的 C 矩阵元素位置
    int row = block_row * BM + ty;
    int col = block_col * BN + tx;
    
    // 累加结果
    fp32 sum = 0.0f;
    
    // K 维度分块迭代
    int num_tiles_k = (K + BK - 1) / BK;
    
    // 预取第一个tile到Buffer 0
    int a_row = row;
    int a_col = 0 * BK + tx;
    if (a_row < M && a_col < K) {
        tileA[0][ty][tx] = A[a_row * K + a_col];
    } else {
        tileA[0][ty][tx] = 0.0f;
    }
    
    int b_row = 0 * BK + ty;
    int b_col = col;
    if (b_row < K && b_col < N) {
        tileB[0][ty][tx] = B[b_row * N + b_col];
    } else {
        tileB[0][ty][tx] = 0.0f;
    }
    
    // 同步：确保第一个tile加载完成
    __syncthreads();
    
    // 主循环：使用双缓冲进行prefetch
    for (int tile_k = 0; tile_k < num_tiles_k; tile_k++) {
        int current_buffer = tile_k % 2;  // 当前用于计算的buffer
        int next_buffer = (tile_k + 1) % 2;  // 用于prefetch的buffer
        
        // 如果还有下一个tile，预取到next_buffer
        if (tile_k + 1 < num_tiles_k) {
            // 预取A的下一个tile
            int a_row_next = row;
            int a_col_next = (tile_k + 1) * BK + tx;
            if (a_row_next < M && a_col_next < K) {
                tileA[next_buffer][ty][tx] = A[a_row_next * K + a_col_next];
            } else {
                tileA[next_buffer][ty][tx] = 0.0f;
            }
            
            // 预取B的下一个tile
            int b_row_next = (tile_k + 1) * BK + ty;
            int b_col_next = col;
            if (b_row_next < K && b_col_next < N) {
                tileB[next_buffer][ty][tx] = B[b_row_next * N + b_col_next];
            } else {
                tileB[next_buffer][ty][tx] = 0.0f;
            }
        }
        
        // 计算当前buffer的数据（与prefetch并行进行）
        // 使用register prefetch：在计算当前k值的同时，预取下一个k值
        fp32 regA_curr, regA_next;
        fp32 regB_curr, regB_next;
        
        // 预取第一个k值
        regA_curr = tileA[current_buffer][ty][0];
        regB_curr = tileB[current_buffer][0][tx];
        
        // 如果BK > 1，预取第二个k值
        if (BK > 1) {
            regA_next = tileA[current_buffer][ty][1];
            regB_next = tileB[current_buffer][1][tx];
        }
        
        // 计算循环：使用register prefetch
        for (int k = 0; k < BK; k++) {
            // 计算当前k值（使用预取的register数据）
            sum += regA_curr * regB_curr;
            
            // 如果还有下一个k值，预取并更新register
            if (k + 1 < BK) {
                regA_curr = regA_next;
                regB_curr = regB_next;
                // 预取下一个k值（在计算当前k值的同时进行）
                if (k + 2 < BK) {
                    regA_next = tileA[current_buffer][ty][k + 2];
                    regB_next = tileB[current_buffer][k + 2][tx];
                }
            }
        }
        
        // 同步：确保所有线程完成计算和prefetch，再进入下一轮
        __syncthreads();
    }
    
    // 写入结果到 global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * GEMM Kernel：Global Memory 分块放入 Shared Memory + Shared Memory 分块放入 Register
 * （无寄存器bank conflict优化版本）
 * 
 * 三级优化：
 * 1. Global Memory -> Shared Memory：分块加载，减少 global memory 访问
 * 2. Shared Memory -> Register：将 shared memory 数据缓存到 register，减少 shared memory 访问延迟
 * 3. Register 计算：直接在 register 中进行计算，最高效
 * 
 * 优化原理：
 * - 第一级：Global Memory 分块，访问量减少为 1/2 * (1/bm + 1/bn)
 * - 第二级：Shared Memory 分块到 Register，进一步减少 shared memory 访问次数
 * - Register 访问延迟最低，可以最大化计算效率
 * 
 * 参数说明：
 * - BM, BN, BK: Global Memory 分块大小（通常为 32）
 * - REG_SIZE: Register 分块大小（K 维度），用于将 Shared Memory 数据分块加载到寄存器
 */
template<int BM, int BN, int BK, int REG_SIZE>
__global__ void gemm_kernel_global_blocking_shared_register(
    const fp32* __restrict__ A,
    const fp32* __restrict__ B,
    fp32* __restrict__ C,
    int M, int N, int K
) {
    // 1. 分块参数（通过模板参数传入）
    // BM, BN, BK, REG_SIZE 已通过模板参数定义

    // 2. Shared Memory：缓存Global分块
    __shared__ fp32 tileA[BM][BK];
    __shared__ fp32 tileB[BK][BN];

    // 3. 寄存器：缓存Shared分块数据（只缓存 tileA，tileB 直接从 shared memory 读取）
    fp32 regA[REG_SIZE];

    // 4. 线程/Block位置计算
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    // 5. 当前Block负责的C矩阵大区块（BM×BN）
    int c_block_row_start = block_row * BM;
    int c_block_col_start = block_col * BN;

    // 6. 每个线程负责计算C矩阵的一个元素
    int c_row = c_block_row_start + ty;
    int c_col = c_block_col_start + tx;

    // 7. 累加结果（寄存器存储）
    fp32 sum = 0.0f;

    // 8. K维度分块迭代
    int num_k_tiles = (K + BK - 1) / BK;
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        // ========== 第一级：Global -> Shared 分块加载 ==========
        // 加载A的BM×BK块到Shared Memory
        int a_row = c_block_row_start + ty;
        int a_col = k_tile * BK + tx;
        tileA[ty][tx] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;

        // 加载B的BK×BN块到Shared Memory
        int b_row = k_tile * BK + ty;
        int b_col = c_block_col_start + tx;
        tileB[ty][tx] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;

        __syncthreads(); // 确保Shared Memory加载完成

        // ========== 第二级：Shared -> Register 分块 ==========
        // 将 Shared Memory 中的数据分块加载到 register
        // 每个线程处理 BK 个元素，分多次加载到 register
        int num_reg_tiles = (BK + REG_SIZE - 1) / REG_SIZE;
        for (int reg_tile = 0; reg_tile < num_reg_tiles; reg_tile++) {
            // 从 shared memory 加载 tileA 的一行片段到 register
            #pragma unroll
            for (int i = 0; i < REG_SIZE; i++) {
                int k_idx = reg_tile * REG_SIZE + i;
                if (k_idx < BK) {
                    regA[i] = tileA[ty][k_idx];
                } else {
                    regA[i] = 0.0f;
                }
            }

            // ========== 第三级：Register 计算 ==========
            // 使用 register 中的 tileA 数据和 shared memory 中的 tileB 数据计算
            // 这样可以减少对 tileB 的重复访问
            #pragma unroll
            for (int i = 0; i < REG_SIZE; i++) {
                int k_idx = reg_tile * REG_SIZE + i;
                if (k_idx < BK) {
                    sum += regA[i] * tileB[k_idx][tx];
                }
            }
        }

        __syncthreads(); // 确保当前K块计算完成
    }

    // 写入结果到Global Memory
    if (c_row < M && c_col < N) {
        C[c_row * N + c_col] = sum;
    }
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
    std::cout << "========== CUDA GEMM 共享内存优化性能测试 ==========" << std::endl;
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
    fp32* h_C_gpu_basic = (fp32*)malloc(size_C);
    fp32* h_C_gpu_global_blocking_shared = (fp32*)malloc(size_C);
    fp32* h_C_gpu_global_blocking_shared_register = (fp32*)malloc(size_C);
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
    
    // ========== GPU 基础 Kernel（不使用共享内存）==========
    std::cout << "\n========== GPU GEMM (基础 Kernel - 无共享内存) ==========" << std::endl;
    
    dim3 blockSize_basic(16, 16);
    dim3 gridSize_basic((N + blockSize_basic.x - 1) / blockSize_basic.x, 
                        (M + blockSize_basic.y - 1) / blockSize_basic.y);
    
    // 预热
    gemm_kernel_basic<<<gridSize_basic, blockSize_basic>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 多次运行取平均值
    total_time = 0.0;
    for (int i = 0; i < num_runs; i++) {
        CUDA_CHECK(cudaMemset(d_C, 0, size_C));
        
        auto start = std::chrono::high_resolution_clock::now();
        gemm_kernel_basic<<<gridSize_basic, blockSize_basic>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_time += duration.count() / 1000.0;
    }
    double time_gpu_basic_ms = total_time / num_runs;
    
    CUDA_CHECK(cudaMemcpy(h_C_gpu_basic, d_C, size_C, cudaMemcpyDeviceToHost));
    print_performance_analysis("GPU (基础 Kernel - 无任何优化)", time_gpu_basic_ms, M, N, K);
    
    // ========== GPU Global Memory 分块放入 Shared Memory（参数搜索）==========
    std::cout << "\n========== GPU GEMM (Global Memory 分块 -> Shared Memory) ==========" << std::endl;
    std::cout << "测试不同的 Tile Size (BM×BN×BK) 参数..." << std::endl;
    
    // 测试不同的 tile size 组合
    struct TileConfig {
        int bm, bn, bk;
    };
    TileConfig tile_configs[] = {
        {16, 16, 16},
        {32, 32, 32},
        {64, 64, 32},
    };
    int num_tile_configs = sizeof(tile_configs) / sizeof(tile_configs[0]);
    double best_time_tile = 1e10;
    TileConfig best_tile_config = {32, 32, 32};
    std::vector<double> times_tile(num_tile_configs);
    std::vector<double> gflops_tile(num_tile_configs);
    
    for (int tile_idx = 0; tile_idx < num_tile_configs; tile_idx++) {
        int bm = tile_configs[tile_idx].bm;
        int bn = tile_configs[tile_idx].bn;
        int bk = tile_configs[tile_idx].bk;
        
        // 检查 shared memory 限制
        size_t shared_mem_size = (bm * bk + bk * bn) * sizeof(fp32);
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        if (shared_mem_size > prop.sharedMemPerBlock) {
            std::cout << "跳过 BM=" << bm << ", BN=" << bn << ", BK=" << bk 
                      << " (共享内存需求: " << (shared_mem_size / 1024.0) 
                      << " KB, 超过限制: " << (prop.sharedMemPerBlock / 1024.0) << " KB)" << std::endl;
            times_tile[tile_idx] = -1;
            continue;
        }
        
        dim3 blockSize_tile(bn, bm);
        dim3 gridSize_tile((N + bn - 1) / bn, (M + bm - 1) / bm);
        
        std::cout << "\n--- 测试 BM=" << bm << ", BN=" << bn << ", BK=" << bk << " ---" << std::endl;
        std::cout << "共享内存使用: " << (shared_mem_size / 1024.0) << " KB per block" << std::endl;
        
        // 预热
        if (bm == 16 && bn == 16 && bk == 16) {
            gemm_kernel_global_blocking_shared<16, 16, 16><<<gridSize_tile, blockSize_tile>>>(d_A, d_B, d_C, M, N, K);
        } else if (bm == 32 && bn == 32 && bk == 32) {
            gemm_kernel_global_blocking_shared<32, 32, 32><<<gridSize_tile, blockSize_tile>>>(d_A, d_B, d_C, M, N, K);
        } else if (bm == 64 && bn == 64 && bk == 32) {
            gemm_kernel_global_blocking_shared<64, 64, 32><<<gridSize_tile, blockSize_tile>>>(d_A, d_B, d_C, M, N, K);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 多次运行取平均值
        total_time = 0.0;
        for (int i = 0; i < num_runs; i++) {
            CUDA_CHECK(cudaMemset(d_C, 0, size_C));
            
            auto start = std::chrono::high_resolution_clock::now();
            
            if (bm == 16 && bn == 16 && bk == 16) {
                gemm_kernel_global_blocking_shared<16, 16, 16><<<gridSize_tile, blockSize_tile>>>(d_A, d_B, d_C, M, N, K);
            } else if (bm == 32 && bn == 32 && bk == 32) {
                gemm_kernel_global_blocking_shared<32, 32, 32><<<gridSize_tile, blockSize_tile>>>(d_A, d_B, d_C, M, N, K);
            } else if (bm == 64 && bn == 64 && bk == 32) {
                gemm_kernel_global_blocking_shared<64, 64, 32><<<gridSize_tile, blockSize_tile>>>(d_A, d_B, d_C, M, N, K);
            }
            
            CUDA_CHECK(cudaDeviceSynchronize());
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            total_time += duration.count() / 1000.0;
        }
        
        double time_ms = total_time / num_runs;
        times_tile[tile_idx] = time_ms;
        
        long long flops = (long long)M * N * 2 * K;
        double gflops = (flops / 1e9) / (time_ms / 1000.0);
        gflops_tile[tile_idx] = gflops;
        
        std::cout << "BM=" << bm << ", BN=" << bn << ", BK=" << bk << ": " << time_ms << " ms, " << gflops << " GFLOPS";
        if (time_ms < best_time_tile) {
            best_time_tile = time_ms;
            best_tile_config = tile_configs[tile_idx];
            std::cout << " (当前最佳)";
        }
        std::cout << std::endl;
    }
    
    // 使用最佳参数再次运行并保存结果
    std::cout << "\n使用最佳 Tile Size BM=" << best_tile_config.bm << ", BN=" << best_tile_config.bn 
              << ", BK=" << best_tile_config.bk << " 运行..." << std::endl;
    
    dim3 blockSize_global_blocking(best_tile_config.bn, best_tile_config.bm);
    dim3 gridSize_global_blocking((N + best_tile_config.bn - 1) / best_tile_config.bn, 
                                  (M + best_tile_config.bm - 1) / best_tile_config.bm);
    
    // 预热
    if (best_tile_config.bm == 16 && best_tile_config.bn == 16 && best_tile_config.bk == 16) {
        gemm_kernel_global_blocking_shared<16, 16, 16><<<gridSize_global_blocking, blockSize_global_blocking>>>(d_A, d_B, d_C, M, N, K);
    } else if (best_tile_config.bm == 32 && best_tile_config.bn == 32 && best_tile_config.bk == 32) {
        gemm_kernel_global_blocking_shared<32, 32, 32><<<gridSize_global_blocking, blockSize_global_blocking>>>(d_A, d_B, d_C, M, N, K);
    } else if (best_tile_config.bm == 64 && best_tile_config.bn == 64 && best_tile_config.bk == 32) {
        gemm_kernel_global_blocking_shared<64, 64, 32><<<gridSize_global_blocking, blockSize_global_blocking>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 多次运行取平均值
    total_time = 0.0;
    for (int i = 0; i < num_runs; i++) {
        CUDA_CHECK(cudaMemset(d_C, 0, size_C));
        
        auto start = std::chrono::high_resolution_clock::now();
        
        if (best_tile_config.bm == 16 && best_tile_config.bn == 16 && best_tile_config.bk == 16) {
            gemm_kernel_global_blocking_shared<16, 16, 16><<<gridSize_global_blocking, blockSize_global_blocking>>>(d_A, d_B, d_C, M, N, K);
        } else if (best_tile_config.bm == 32 && best_tile_config.bn == 32 && best_tile_config.bk == 32) {
            gemm_kernel_global_blocking_shared<32, 32, 32><<<gridSize_global_blocking, blockSize_global_blocking>>>(d_A, d_B, d_C, M, N, K);
        } else if (best_tile_config.bm == 64 && best_tile_config.bn == 64 && best_tile_config.bk == 32) {
            gemm_kernel_global_blocking_shared<64, 64, 32><<<gridSize_global_blocking, blockSize_global_blocking>>>(d_A, d_B, d_C, M, N, K);
        }
        
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_time += duration.count() / 1000.0;
    }
    double time_gpu_global_blocking_shared_ms = total_time / num_runs;
    
    CUDA_CHECK(cudaMemcpy(h_C_gpu_global_blocking_shared, d_C, size_C, cudaMemcpyDeviceToHost));
    print_performance_analysis("GPU (Global Memory 分块 -> Shared Memory, BM=" + std::to_string(best_tile_config.bm) + 
                               ", BN=" + std::to_string(best_tile_config.bn) + ", BK=" + std::to_string(best_tile_config.bk) + ")", 
                               time_gpu_global_blocking_shared_ms, M, N, K);
    
    // 打印 Tile Size 参数搜索总结
    std::cout << "\n========== Tile Size 参数搜索总结 ==========" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    for (int tile_idx = 0; tile_idx < num_tile_configs; tile_idx++) {
        if (times_tile[tile_idx] > 0) {
            std::cout << "BM=" << tile_configs[tile_idx].bm << ", BN=" << tile_configs[tile_idx].bn 
                      << ", BK=" << tile_configs[tile_idx].bk << ": " << times_tile[tile_idx] << " ms, " 
                      << gflops_tile[tile_idx] << " GFLOPS";
            if (tile_configs[tile_idx].bm == best_tile_config.bm && 
                tile_configs[tile_idx].bn == best_tile_config.bn && 
                tile_configs[tile_idx].bk == best_tile_config.bk) {
                std::cout << " (最佳)";
            }
            std::cout << std::endl;
        }
    }
    
    // 打印 Tile Size 参数搜索总结
    std::cout << "\n========== Tile Size 参数搜索总结 ==========" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    for (int tile_idx = 0; tile_idx < num_tile_configs; tile_idx++) {
        if (times_tile[tile_idx] > 0) {
            std::cout << "BM=" << tile_configs[tile_idx].bm << ", BN=" << tile_configs[tile_idx].bn 
                      << ", BK=" << tile_configs[tile_idx].bk << ": " << times_tile[tile_idx] << " ms, " 
                      << gflops_tile[tile_idx] << " GFLOPS";
            if (tile_configs[tile_idx].bm == best_tile_config.bm && 
                tile_configs[tile_idx].bn == best_tile_config.bn && 
                tile_configs[tile_idx].bk == best_tile_config.bk) {
                std::cout << " (最佳)";
            }
            std::cout << std::endl;
        }
    }
    
    // ========== GPU Global Memory 分块 -> Shared Memory -> Register（参数搜索）==========
    std::cout << "\n========== GPU GEMM (Global Memory 分块 -> Shared Memory -> Register) ==========" << std::endl;
    std::cout << "测试不同的 REG_SIZE 参数..." << std::endl;
    
    // 测试不同的 REG_SIZE 值
    int reg_sizes[] = {2, 4, 8, 16, 32};
    int num_reg_sizes = sizeof(reg_sizes) / sizeof(reg_sizes[0]);
    double best_time_reg = 1e10;
    int best_reg_size = 0;
    std::vector<double> times_reg(num_reg_sizes);
    std::vector<double> gflops_reg(num_reg_sizes);
    
    for (int reg_idx = 0; reg_idx < num_reg_sizes; reg_idx++) {
        int reg_size = reg_sizes[reg_idx];
        
        // 检查 REG_SIZE 是否超过 BK
        if (reg_size > 32) {
            std::cout << "跳过 REG_SIZE=" << reg_size << " (超过 BK=32)" << std::endl;
            times_reg[reg_idx] = -1;
            continue;
        }
        
        std::cout << "\n--- 测试 REG_SIZE=" << reg_size << " ---" << std::endl;
        
        // 根据 REG_SIZE 调用不同的模板实例
        total_time = 0.0;
        for (int i = 0; i < num_runs; i++) {
            CUDA_CHECK(cudaMemset(d_C, 0, size_C));
            
            auto start = std::chrono::high_resolution_clock::now();
            
            // 使用模板参数调用不同的 kernel 实例
            switch (reg_size) {
                case 1:
                    gemm_kernel_global_blocking_shared_register<32, 32, 32, 1><<<gridSize_global_blocking, blockSize_global_blocking>>>(d_A, d_B, d_C, M, N, K);
                    break;
                case 2:
                    gemm_kernel_global_blocking_shared_register<32, 32, 32, 2><<<gridSize_global_blocking, blockSize_global_blocking>>>(d_A, d_B, d_C, M, N, K);
                    break;
                case 4:
                    gemm_kernel_global_blocking_shared_register<32, 32, 32, 4><<<gridSize_global_blocking, blockSize_global_blocking>>>(d_A, d_B, d_C, M, N, K);
                    break;
                case 8:
                    gemm_kernel_global_blocking_shared_register<32, 32, 32, 8><<<gridSize_global_blocking, blockSize_global_blocking>>>(d_A, d_B, d_C, M, N, K);
                    break;
                case 16:
                    gemm_kernel_global_blocking_shared_register<32, 32, 32, 16><<<gridSize_global_blocking, blockSize_global_blocking>>>(d_A, d_B, d_C, M, N, K);
                    break;
                case 32:
                    gemm_kernel_global_blocking_shared_register<32, 32, 32, 32><<<gridSize_global_blocking, blockSize_global_blocking>>>(d_A, d_B, d_C, M, N, K);
                    break;
            }
            
            CUDA_CHECK(cudaDeviceSynchronize());
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            total_time += duration.count() / 1000.0;
        }
        
        double time_ms = total_time / num_runs;
        times_reg[reg_idx] = time_ms;
        
        long long flops = (long long)M * N * 2 * K;
        double gflops = (flops / 1e9) / (time_ms / 1000.0);
        gflops_reg[reg_idx] = gflops;
        
        std::cout << "REG_SIZE=" << reg_size << ": " << time_ms << " ms, " << gflops << " GFLOPS";
        if (time_ms < best_time_reg) {
            best_time_reg = time_ms;
            best_reg_size = reg_size;
            std::cout << " (当前最佳)";
        }
        std::cout << std::endl;
    }
    
    // 使用最佳参数再次运行并保存结果
    std::cout << "\n使用最佳 REG_SIZE=" << best_reg_size << " 运行..." << std::endl;
    total_time = 0.0;
    for (int i = 0; i < num_runs; i++) {
        CUDA_CHECK(cudaMemset(d_C, 0, size_C));
        
        auto start = std::chrono::high_resolution_clock::now();
        
        switch (best_reg_size) {
            case 1:
                gemm_kernel_global_blocking_shared_register<32, 32, 32, 1><<<gridSize_global_blocking, blockSize_global_blocking>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 2:
                gemm_kernel_global_blocking_shared_register<32, 32, 32, 2><<<gridSize_global_blocking, blockSize_global_blocking>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 4:
                gemm_kernel_global_blocking_shared_register<32, 32, 32, 4><<<gridSize_global_blocking, blockSize_global_blocking>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 8:
                gemm_kernel_global_blocking_shared_register<32, 32, 32, 8><<<gridSize_global_blocking, blockSize_global_blocking>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 16:
                gemm_kernel_global_blocking_shared_register<32, 32, 32, 16><<<gridSize_global_blocking, blockSize_global_blocking>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 32:
                gemm_kernel_global_blocking_shared_register<32, 32, 32, 32><<<gridSize_global_blocking, blockSize_global_blocking>>>(d_A, d_B, d_C, M, N, K);
                break;
        }
        
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_time += duration.count() / 1000.0;
    }
    double time_gpu_global_blocking_shared_register_ms = total_time / num_runs;
    
    CUDA_CHECK(cudaMemcpy(h_C_gpu_global_blocking_shared_register, d_C, size_C, cudaMemcpyDeviceToHost));
    print_performance_analysis("GPU (Global Memory 分块 -> Shared Memory -> Register, REG_SIZE=" + std::to_string(best_reg_size) + ")", 
                               time_gpu_global_blocking_shared_register_ms, M, N, K);
    
    // ========== GPU Global Memory 分块 -> Shared Memory（数据预取版本，参数搜索）==========
    std::cout << "\n========== GPU GEMM (Global Memory 分块 -> Shared Memory, 数据预取) ==========" << std::endl;
    std::cout << "测试不同的 Tile Size (BM×BN×BK) 参数..." << std::endl;
    
    // 测试不同的 tile size 组合（使用与 global_blocking_shared 相同的配置）
    TileConfig prefetch_tile_configs[] = {
        {16, 16, 16},
        {32, 32, 32},
        {64, 64, 32},
    };
    int num_prefetch_tile_configs = sizeof(prefetch_tile_configs) / sizeof(prefetch_tile_configs[0]);
    double best_time_prefetch = 1e10;
    TileConfig best_prefetch_tile_config = {32, 32, 32};
    std::vector<double> times_prefetch(num_prefetch_tile_configs);
    std::vector<double> gflops_prefetch(num_prefetch_tile_configs);
    
    for (int tile_idx = 0; tile_idx < num_prefetch_tile_configs; tile_idx++) {
        int bm = prefetch_tile_configs[tile_idx].bm;
        int bn = prefetch_tile_configs[tile_idx].bn;
        int bk = prefetch_tile_configs[tile_idx].bk;
        
        // 检查 shared memory 限制（双缓冲需要2倍内存）
        size_t shared_mem_size = 2 * (bm * bk + bk * bn) * sizeof(fp32);
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        if (shared_mem_size > prop.sharedMemPerBlock) {
            std::cout << "跳过 BM=" << bm << ", BN=" << bn << ", BK=" << bk 
                      << " (共享内存需求: " << (shared_mem_size / 1024.0) 
                      << " KB, 超过限制: " << (prop.sharedMemPerBlock / 1024.0) << " KB)" << std::endl;
            times_prefetch[tile_idx] = -1;
            continue;
        }
        
        dim3 blockSize_prefetch(bn, bm);
        dim3 gridSize_prefetch((N + bn - 1) / bn, (M + bm - 1) / bm);
        
        std::cout << "\n--- 测试 BM=" << bm << ", BN=" << bn << ", BK=" << bk << " ---" << std::endl;
        std::cout << "共享内存使用: " << (shared_mem_size / 1024.0) << " KB per block (双缓冲)" << std::endl;
        
        // 预热
        if (bm == 16 && bn == 16 && bk == 16) {
            gemm_kernel_global_blocking_shared_prefetch<16, 16, 16><<<gridSize_prefetch, blockSize_prefetch>>>(d_A, d_B, d_C, M, N, K);
        } else if (bm == 32 && bn == 32 && bk == 32) {
            gemm_kernel_global_blocking_shared_prefetch<32, 32, 32><<<gridSize_prefetch, blockSize_prefetch>>>(d_A, d_B, d_C, M, N, K);
        } else if (bm == 64 && bn == 64 && bk == 32) {
            gemm_kernel_global_blocking_shared_prefetch<64, 64, 32><<<gridSize_prefetch, blockSize_prefetch>>>(d_A, d_B, d_C, M, N, K);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 多次运行取平均值
        total_time = 0.0;
        for (int i = 0; i < num_runs; i++) {
            CUDA_CHECK(cudaMemset(d_C, 0, size_C));
            
            auto start = std::chrono::high_resolution_clock::now();
            
            if (bm == 16 && bn == 16 && bk == 16) {
                gemm_kernel_global_blocking_shared_prefetch<16, 16, 16><<<gridSize_prefetch, blockSize_prefetch>>>(d_A, d_B, d_C, M, N, K);
            } else if (bm == 32 && bn == 32 && bk == 32) {
                gemm_kernel_global_blocking_shared_prefetch<32, 32, 32><<<gridSize_prefetch, blockSize_prefetch>>>(d_A, d_B, d_C, M, N, K);
            } else if (bm == 64 && bn == 64 && bk == 32) {
                gemm_kernel_global_blocking_shared_prefetch<64, 64, 32><<<gridSize_prefetch, blockSize_prefetch>>>(d_A, d_B, d_C, M, N, K);
            }
            
            CUDA_CHECK(cudaDeviceSynchronize());
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            total_time += duration.count() / 1000.0;
        }
        
        double time_ms = total_time / num_runs;
        times_prefetch[tile_idx] = time_ms;
        
        long long flops = (long long)M * N * 2 * K;
        double gflops = (flops / 1e9) / (time_ms / 1000.0);
        gflops_prefetch[tile_idx] = gflops;
        
        std::cout << "BM=" << bm << ", BN=" << bn << ", BK=" << bk << ": " << time_ms << " ms, " << gflops << " GFLOPS";
        if (time_ms < best_time_prefetch) {
            best_time_prefetch = time_ms;
            best_prefetch_tile_config = prefetch_tile_configs[tile_idx];
            std::cout << " (当前最佳)";
        }
        std::cout << std::endl;
    }
    
    // 使用最佳参数再次运行并保存结果
    std::cout << "\n使用最佳 Tile Size BM=" << best_prefetch_tile_config.bm << ", BN=" << best_prefetch_tile_config.bn 
              << ", BK=" << best_prefetch_tile_config.bk << " 运行..." << std::endl;
    
    dim3 blockSize_prefetch_best(best_prefetch_tile_config.bn, best_prefetch_tile_config.bm);
    dim3 gridSize_prefetch_best((N + best_prefetch_tile_config.bn - 1) / best_prefetch_tile_config.bn, 
                                (M + best_prefetch_tile_config.bm - 1) / best_prefetch_tile_config.bm);
    
    // 预热
    if (best_prefetch_tile_config.bm == 16 && best_prefetch_tile_config.bn == 16 && best_prefetch_tile_config.bk == 16) {
        gemm_kernel_global_blocking_shared_prefetch<16, 16, 16><<<gridSize_prefetch_best, blockSize_prefetch_best>>>(d_A, d_B, d_C, M, N, K);
    } else if (best_prefetch_tile_config.bm == 32 && best_prefetch_tile_config.bn == 32 && best_prefetch_tile_config.bk == 32) {
        gemm_kernel_global_blocking_shared_prefetch<32, 32, 32><<<gridSize_prefetch_best, blockSize_prefetch_best>>>(d_A, d_B, d_C, M, N, K);
    } else if (best_prefetch_tile_config.bm == 64 && best_prefetch_tile_config.bn == 64 && best_prefetch_tile_config.bk == 32) {
        gemm_kernel_global_blocking_shared_prefetch<64, 64, 32><<<gridSize_prefetch_best, blockSize_prefetch_best>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 多次运行取平均值
    total_time = 0.0;
    for (int i = 0; i < num_runs; i++) {
        CUDA_CHECK(cudaMemset(d_C, 0, size_C));
        
        auto start = std::chrono::high_resolution_clock::now();
        
        if (best_prefetch_tile_config.bm == 16 && best_prefetch_tile_config.bn == 16 && best_prefetch_tile_config.bk == 16) {
            gemm_kernel_global_blocking_shared_prefetch<16, 16, 16><<<gridSize_prefetch_best, blockSize_prefetch_best>>>(d_A, d_B, d_C, M, N, K);
        } else if (best_prefetch_tile_config.bm == 32 && best_prefetch_tile_config.bn == 32 && best_prefetch_tile_config.bk == 32) {
            gemm_kernel_global_blocking_shared_prefetch<32, 32, 32><<<gridSize_prefetch_best, blockSize_prefetch_best>>>(d_A, d_B, d_C, M, N, K);
        } else if (best_prefetch_tile_config.bm == 64 && best_prefetch_tile_config.bn == 64 && best_prefetch_tile_config.bk == 32) {
            gemm_kernel_global_blocking_shared_prefetch<64, 64, 32><<<gridSize_prefetch_best, blockSize_prefetch_best>>>(d_A, d_B, d_C, M, N, K);
        }
        
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_time += duration.count() / 1000.0;
    }
    double time_gpu_prefetch_ms = total_time / num_runs;
    
    CUDA_CHECK(cudaMemcpy(h_C_gpu_prefetch, d_C, size_C, cudaMemcpyDeviceToHost));
    print_performance_analysis("GPU (Global Memory 分块 -> Shared Memory, 数据预取, BM=" + std::to_string(best_prefetch_tile_config.bm) + 
                               ", BN=" + std::to_string(best_prefetch_tile_config.bn) + ", BK=" + std::to_string(best_prefetch_tile_config.bk) + ")", 
                               time_gpu_prefetch_ms, M, N, K);
    
    // 打印 Prefetch Tile Size 参数搜索总结
    std::cout << "\n========== Prefetch Tile Size 参数搜索总结 ==========" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    for (int tile_idx = 0; tile_idx < num_prefetch_tile_configs; tile_idx++) {
        if (times_prefetch[tile_idx] > 0) {
            std::cout << "BM=" << prefetch_tile_configs[tile_idx].bm << ", BN=" << prefetch_tile_configs[tile_idx].bn 
                      << ", BK=" << prefetch_tile_configs[tile_idx].bk << ": " << times_prefetch[tile_idx] << " ms, " 
                      << gflops_prefetch[tile_idx] << " GFLOPS";
            if (prefetch_tile_configs[tile_idx].bm == best_prefetch_tile_config.bm && 
                prefetch_tile_configs[tile_idx].bn == best_prefetch_tile_config.bn && 
                prefetch_tile_configs[tile_idx].bk == best_prefetch_tile_config.bk) {
                std::cout << " (最佳)";
            }
            std::cout << std::endl;
        }
    }
    
    // 打印参数搜索总结
    std::cout << "\n========== REG_SIZE 参数搜索总结 ==========" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    for (int reg_idx = 0; reg_idx < num_reg_sizes; reg_idx++) {
        if (times_reg[reg_idx] > 0) {
            std::cout << "REG_SIZE=" << reg_sizes[reg_idx] << ": " << times_reg[reg_idx] << " ms, " 
                      << gflops_reg[reg_idx] << " GFLOPS";
            if (reg_sizes[reg_idx] == best_reg_size) {
                std::cout << " (最佳)";
            }
            std::cout << std::endl;
        }
    }
    
    // ========== 性能对比总结 ==========
    std::cout << "\n========== 性能对比总结 ==========" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "========== 四个版本对比 ==========" << std::endl;
    
    // 1. Baseline版本
    std::cout << "1. Baseline (基础 - 无任何优化):       " << time_gpu_basic_ms << " ms" << std::endl;
    
    // 2. Global Memory分块版本
    std::cout << "2. Global Memory 分块 -> Shared Memory: " << time_gpu_global_blocking_shared_ms << " ms" 
              << " (相对Baseline提升: " << time_gpu_basic_ms / time_gpu_global_blocking_shared_ms << "x)" << std::endl;
    
    // 3. Shared Memory分块版本
    std::cout << "3. Shared Memory 分块 (Register缓存): " << time_gpu_global_blocking_shared_register_ms << " ms" 
              << " (相对Baseline提升: " << time_gpu_basic_ms / time_gpu_global_blocking_shared_register_ms << "x, "
              << "相对Global分块版本提升: " << time_gpu_global_blocking_shared_ms / time_gpu_global_blocking_shared_register_ms << "x)" << std::endl;
    
    // 4. 数据预取版本
    std::cout << "4. Global Memory 分块 -> Shared Memory (数据预取): " << time_gpu_prefetch_ms << " ms" 
              << " (相对Baseline提升: " << time_gpu_basic_ms / time_gpu_prefetch_ms << "x, "
              << "相对Global分块版本提升: " << time_gpu_global_blocking_shared_ms / time_gpu_prefetch_ms << "x, "
              << "相对Shared分块版本提升: " << time_gpu_global_blocking_shared_register_ms / time_gpu_prefetch_ms << "x)" << std::endl;
    
    // 计算 GFLOPS 对比
    long long flops = (long long)M * N * 2 * K;
    double gflops_gpu_basic = (flops / 1e9) / (time_gpu_basic_ms / 1000.0);
    double gflops_gpu_global_blocking_shared = (flops / 1e9) / (time_gpu_global_blocking_shared_ms / 1000.0);
    double gflops_gpu_global_blocking_shared_register = (flops / 1e9) / (time_gpu_global_blocking_shared_register_ms / 1000.0);
    
    std::cout << "\n算力对比 (GFLOPS):" << std::endl;
    std::cout << "1. Baseline:       " << gflops_gpu_basic << " GFLOPS" << std::endl;
    std::cout << "2. Global Memory 分块 -> Shared Memory: " << gflops_gpu_global_blocking_shared << " GFLOPS" << std::endl;
    std::cout << "3. Shared Memory 分块 (Register缓存): " << gflops_gpu_global_blocking_shared_register << " GFLOPS" << std::endl;
    double gflops_gpu_prefetch = (flops / 1e9) / (time_gpu_prefetch_ms / 1000.0);
    std::cout << "4. Global Memory 分块 -> Shared Memory (数据预取): " << gflops_gpu_prefetch << " GFLOPS" << std::endl;
    
    // 性能分析
    std::cout << "\n========== 性能分析 ==========" << std::endl;
    std::cout << "========== 四个版本优化分析 ==========" << std::endl;
    std::cout << "1. Global Memory 分块 -> Shared Memory 相对Baseline提升: " 
              << (time_gpu_basic_ms / time_gpu_global_blocking_shared_ms - 1.0) * 100 << "%" << std::endl;
    std::cout << "   - 内存访问减少比例：1/2 * (1/bm + 1/bn) = 1/2 * (1/32 + 1/32) = 1/32" << std::endl;
    std::cout << "   - 利用 shared memory 低延迟特性" << std::endl;
    
    std::cout << "2. Shared Memory 分块 (Register缓存) 相对Baseline提升: " 
              << (time_gpu_basic_ms / time_gpu_global_blocking_shared_register_ms - 1.0) * 100 << "%" << std::endl;
    std::cout << "   - 相对Global分块版本提升: " 
              << (time_gpu_global_blocking_shared_ms / time_gpu_global_blocking_shared_register_ms - 1.0) * 100 << "%" << std::endl;
    std::cout << "   - 进一步减少 shared memory 访问延迟" << std::endl;
    std::cout << "   - Register 访问延迟最低，最大化计算效率" << std::endl;
    
    std::cout << "3. 数据预取版本 相对Baseline提升: " 
              << (time_gpu_basic_ms / time_gpu_prefetch_ms - 1.0) * 100 << "%" << std::endl;
    std::cout << "   - 相对Global分块版本提升: " 
              << (time_gpu_global_blocking_shared_ms / time_gpu_prefetch_ms - 1.0) * 100 << "%" << std::endl;
    std::cout << "   - 相对Shared分块版本提升: " 
              << (time_gpu_global_blocking_shared_register_ms / time_gpu_prefetch_ms - 1.0) * 100 << "%" << std::endl;
    std::cout << "   - 使用双缓冲（Double Buffering）掩盖访存延迟" << std::endl;
    std::cout << "   - Shared Memory Prefetch：在计算当前tile的同时，预取下一个tile" << std::endl;
    std::cout << "   - Register Prefetch：在计算当前register数据的同时，预取下一个register数据" << std::endl;
    std::cout << "   - 通过重叠计算和访存，最大化计算单元利用率" << std::endl;
    
    // 分析为什么 Register 版本可能更慢
    std::cout << "\n========== Register 版本性能分析 ==========" << std::endl;
    if (time_gpu_global_blocking_shared_register_ms > time_gpu_global_blocking_shared_ms) {
        std::cout << "⚠️  Shared Memory 分块版本比 Global 分块版本慢 " 
                  << ((time_gpu_global_blocking_shared_register_ms / time_gpu_global_blocking_shared_ms - 1.0) * 100) << "%" << std::endl;
        std::cout << "\n可能的原因：" << std::endl;
        std::cout << "1. 额外的循环开销：Register 版本需要额外的循环来分块加载数据" << std::endl;
        std::cout << "   - Global分块版本：直接访问 shared memory，编译器可以优化" << std::endl;
        std::cout << "   - Register 版本：需要额外的循环和寄存器加载操作" << std::endl;
        std::cout << "2. 寄存器压力：使用过多寄存器可能导致寄存器溢出到 local memory" << std::endl;
        std::cout << "   - REG_SIZE=" << best_reg_size << " 可能不是最优值" << std::endl;
        std::cout << "   - 寄存器数量有限，过多使用可能导致性能下降" << std::endl;
        std::cout << "3. 指令开销：加载到寄存器的指令可能比直接访问 shared memory 更慢" << std::endl;
        std::cout << "   - Shared Memory 有缓存，访问延迟已经很低" << std::endl;
        std::cout << "   - 额外的寄存器操作可能无法带来足够的收益" << std::endl;
        std::cout << "4. 编译器优化：编译器可能已经很好地优化了 shared memory 访问" << std::endl;
        std::cout << "   - 现代 GPU 的 shared memory 访问已经高度优化" << std::endl;
        std::cout << "   - 手动寄存器优化可能不如编译器自动优化效果好" << std::endl;
    } else {
        std::cout << "✓ Shared Memory 分块版本比 Global 分块版本快 " 
                  << ((time_gpu_global_blocking_shared_ms / time_gpu_global_blocking_shared_register_ms - 1.0) * 100) << "%" << std::endl;
        std::cout << "最佳 REG_SIZE=" << best_reg_size << " 带来了性能提升" << std::endl;
    }
    
    // 清理资源
    free(h_A);
    free(h_B);
    free(h_C_gpu_basic);
    free(h_C_gpu_global_blocking_shared);
    free(h_C_gpu_global_blocking_shared_register);
    free(h_C_gpu_prefetch);
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    std::cout << "\n========== 测试完成 ==========" << std::endl;
    
    return 0;
}
