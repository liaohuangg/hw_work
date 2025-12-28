/**
 * CUDA GEMM (General Matrix Multiplication) - 共享内存优化版本
 * 计算 C = A * B，其中 A[M][K], B[K][N], C[M][N]
 * 
 * 参数：(M, N, K) = (20480, 2048, 8192)
 * 数据类型：FP32 (float)
 * 
 * 本文件专注于共享内存优化技术，包括：
 * 1. Tile-based 共享内存缓存
 * 2. 内存访问模式优化
 * 3. 寄存器优化
 * 4. 详细的性能分析
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

// Tile 大小配置（可以根据 GPU 架构调整）
#define TILE_SIZE_16 16   // 较小的 tile，适合较小的共享内存
#define TILE_SIZE_32 32   // 较大的 tile，需要更多共享内存但性能更好

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
 * 共享内存优化 GEMM Kernel (Tile Size = 16)
 * 使用 tile-based 方法，每个 block 处理一个 tile
 * 
 * 优化技术：
 * 1. 共享内存缓存：减少全局内存访问
 * 2. 合并内存访问：每个 warp 访问连续内存
 * 3. 数据重用：共享内存中的数据被多个线程复用
 */
__global__ void gemm_kernel_shared_mem_16(
    const fp32* __restrict__ A,
    const fp32* __restrict__ B,
    fp32* __restrict__ C,
    int M, int N, int K
) {
    // 共享内存：每个 block 缓存 A 和 B 的一个 tile
    // 使用 __shared__ 关键字声明共享内存
    __shared__ fp32 tileA[TILE_SIZE_16][TILE_SIZE_16];
    __shared__ fp32 tileB[TILE_SIZE_16][TILE_SIZE_16];
    
    // 线程在 block 中的位置
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 线程在全局矩阵中的位置
    int row = blockIdx.y * TILE_SIZE_16 + ty;
    int col = blockIdx.x * TILE_SIZE_16 + tx;
    
    // 使用寄存器存储累加结果（避免重复访问共享内存）
    fp32 sum = 0.0f;
    
    // 遍历 K 维度，每次处理一个 tile
    // 外层循环：将 K 维度分成多个 tile
    for (int tile = 0; tile < (K + TILE_SIZE_16 - 1) / TILE_SIZE_16; tile++) {
        // 加载 A 的 tile 到共享内存
        // 每个线程负责加载一个元素
        // 注意：row 和 col 是全局坐标，需要转换为 tile 内的局部坐标
        int a_row = row;
        int a_col = tile * TILE_SIZE_16 + tx;
        
        if (a_row < M && a_col < K) {
            tileA[ty][tx] = A[a_row * K + a_col];
        } else {
            tileA[ty][tx] = 0.0f;  // 边界填充
        }
        
        // 加载 B 的 tile 到共享内存
        // B 矩阵需要转置访问以实现合并内存访问
        int b_row = tile * TILE_SIZE_16 + ty;
        int b_col = col;
        
        if (b_row < K && b_col < N) {
            tileB[ty][tx] = B[b_row * N + b_col];
        } else {
            tileB[ty][tx] = 0.0f;  // 边界填充
        }
        
        // 同步，确保所有线程都加载完数据后再进行计算
        // 这是共享内存编程的关键：必须同步才能保证数据一致性
        __syncthreads();
        
        // 计算 tile 内的点积
        // 现在数据在共享内存中，访问速度比全局内存快得多
        for (int k = 0; k < TILE_SIZE_16; k++) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        
        // 同步，确保所有线程都计算完再加载下一个 tile
        // 避免数据竞争
        __syncthreads();
    }
    
    // 写入结果到全局内存
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * 共享内存优化 GEMM Kernel (Tile Size = 32)
 * 使用更大的 tile 大小，进一步提高数据重用率
 * 
 * 优势：
 * 1. 更大的 tile 意味着更多的数据重用
 * 2. 减少外层循环次数
 * 3. 更好的计算/内存访问比
 * 
 * 限制：
 * 1. 需要更多共享内存（32x32x2x4 = 8KB per block）
 * 2. 需要确保 GPU 有足够的共享内存
 */
__global__ void gemm_kernel_shared_mem_32(
    const fp32* __restrict__ A,
    const fp32* __restrict__ B,
    fp32* __restrict__ C,
    int M, int N, int K
) {
    // 共享内存：使用更大的 tile
    __shared__ fp32 tileA[TILE_SIZE_32][TILE_SIZE_32];
    __shared__ fp32 tileB[TILE_SIZE_32][TILE_SIZE_32];
    
    // 线程在 block 中的位置
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 线程在全局矩阵中的位置
    int row = blockIdx.y * TILE_SIZE_32 + ty;
    int col = blockIdx.x * TILE_SIZE_32 + tx;
    
    // 使用寄存器存储累加结果
    fp32 sum = 0.0f;
    
    // 遍历 K 维度
    for (int tile = 0; tile < (K + TILE_SIZE_32 - 1) / TILE_SIZE_32; tile++) {
        // 加载 A 的 tile
        int a_row = row;
        int a_col = tile * TILE_SIZE_32 + tx;
        
        if (a_row < M && a_col < K) {
            tileA[ty][tx] = A[a_row * K + a_col];
        } else {
            tileA[ty][tx] = 0.0f;
        }
        
        // 加载 B 的 tile
        int b_row = tile * TILE_SIZE_32 + ty;
        int b_col = col;
        
        if (b_row < K && b_col < N) {
            tileB[ty][tx] = B[b_row * N + b_col];
        } else {
            tileB[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // 计算点积
        for (int k = 0; k < TILE_SIZE_32; k++) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        
        __syncthreads();
    }
    
    // 写入结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * 共享内存缓存优化 GEMM Kernel（不使用 K 维度分块）
 * 
 * 核心思路：
 * 1. 每个线程块处理 C 矩阵的一行（或一列）
 * 2. 将矩阵 A 的整行和矩阵 B 的整列缓存到共享内存（不是分块）
 * 3. K 维度不分块，直接遍历整个 K 维度
 * 4. 通过共享内存减少全局内存访问次数，利用共享内存低延迟特性加速计算
 * 
 * 注意：这种实现本质上是"共享内存缓存 + 全局 K 循环"
 * 虽然能提升性能，但效率不如分块版本（因为某些数据仍会被重复读取）
 * 
 * 实现方式：
 * - 每个 block 处理 C 矩阵的一行（blockDim.x 个元素）
 * - 将 A 的这一整行（K 个元素）加载到共享内存
 * - 将 B 的每一列（K 个元素）加载到共享内存，然后计算
 */
#define SHARED_CACHE_SIZE 1024  // 共享内存缓存大小（根据 GPU 限制调整）

__global__ void gemm_kernel_shared_cache_no_tiling(
    const fp32* __restrict__ A,
    const fp32* __restrict__ B,
    fp32* __restrict__ C,
    int M, int N, int K
) {
    // 共享内存：缓存 A 的一整行和 B 的一整列
    // 注意：由于 K 可能很大（8192），我们使用循环分段加载
    __shared__ fp32 shared_A[SHARED_CACHE_SIZE];  // 缓存 A 的一行的一部分
    __shared__ fp32 shared_B[SHARED_CACHE_SIZE];  // 缓存 B 的一列的一部分
    
    // 线程在 block 中的位置
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;  // 线程在 block 中的线性索引
    
    // 当前 block 处理的 C 矩阵的行
    int row = blockIdx.y;
    
    // 当前线程处理的 C 矩阵的列
    int col = blockIdx.x * blockDim.x + tx;
    
    // 边界检查
    if (row >= M || col >= N) {
        return;
    }
    
    // 累加结果
    fp32 sum = 0.0f;
    
    // 遍历整个 K 维度（不分块）
    // 由于 K 可能很大，我们分段加载到共享内存
    int num_segments = (K + SHARED_CACHE_SIZE - 1) / SHARED_CACHE_SIZE;
    
    for (int seg = 0; seg < num_segments; seg++) {
        int k_start = seg * SHARED_CACHE_SIZE;
        int k_end = (k_start + SHARED_CACHE_SIZE < K) ? (k_start + SHARED_CACHE_SIZE) : K;
        int seg_size = k_end - k_start;
        
        // 协作加载 A 的这一行的片段到共享内存
        // 每个线程加载一个元素
        if (tid < seg_size) {
            int k_idx = k_start + tid;
            if (k_idx < K) {
                shared_A[tid] = A[row * K + k_idx];
            } else {
                shared_A[tid] = 0.0f;
            }
        }
        
        // 协作加载 B 的这一列的片段到共享内存
        // 每个线程加载一个元素
        if (tid < seg_size) {
            int k_idx = k_start + tid;
            if (k_idx < K) {
                shared_B[tid] = B[k_idx * N + col];
            } else {
                shared_B[tid] = 0.0f;
            }
        }
        
        // 同步，确保所有数据都加载完成
        __syncthreads();
        
        // 计算这个片段内的点积
        for (int k = 0; k < seg_size; k++) {
            sum += shared_A[k] * shared_B[k];
        }
        
        // 同步，确保所有线程都计算完再加载下一个片段
        __syncthreads();
    }
    
    // 写入结果
    C[row * N + col] = sum;
}

/**
 * 共享内存缓存优化 GEMM Kernel（不使用 K 维度分块）- 简化版本
 * 
 * 这个版本假设 K 不是特别大，可以直接将整行/整列加载到共享内存
 * 如果 K 很大，可以使用上面的分段版本
 */
__global__ void gemm_kernel_shared_cache_simple(
    const fp32* __restrict__ A,
    const fp32* __restrict__ B,
    fp32* __restrict__ C,
    int M, int N, int K
) {
    // 共享内存：缓存 A 的一整行和 B 的一整列
    // 注意：这个版本假设 K 不会超过共享内存限制
    extern __shared__ fp32 shared_mem[];
    fp32* shared_A = shared_mem;
    fp32* shared_B = shared_mem + K;
    
    // 线程在 block 中的位置
    int tx = threadIdx.x;
    int tid = threadIdx.x;  // 假设 blockDim.y = 1
    
    // 当前 block 处理的 C 矩阵的行
    int row = blockIdx.y;
    
    // 当前线程处理的 C 矩阵的列
    int col = blockIdx.x * blockDim.x + tx;
    
    // 边界检查
    if (row >= M || col >= N) {
        return;
    }
    
    // 协作加载 A 的这一整行到共享内存
    // 每个线程加载 K/blockDim.x 个元素
    int elements_per_thread = (K + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < elements_per_thread; i++) {
        int k_idx = tid + i * blockDim.x;
        if (k_idx < K) {
            shared_A[k_idx] = A[row * K + k_idx];
        }
    }
    
    // 协作加载 B 的这一整列到共享内存
    for (int i = 0; i < elements_per_thread; i++) {
        int k_idx = tid + i * blockDim.x;
        if (k_idx < K) {
            shared_B[k_idx] = B[k_idx * N + col];
        }
    }
    
    // 同步，确保所有数据都加载完成
    __syncthreads();
    
    // 计算点积（遍历整个 K 维度，不分块）
    fp32 sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += shared_A[k] * shared_B[k];
    }
    
    // 写入结果
    C[row * N + col] = sum;
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
    fp32* h_C_gpu_shared_16 = (fp32*)malloc(size_C);
    fp32* h_C_gpu_shared_32 = (fp32*)malloc(size_C);
    fp32* h_C_gpu_cache_no_tiling = (fp32*)malloc(size_C);
    
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
    print_performance_analysis("GPU (基础 Kernel)", time_gpu_basic_ms, M, N, K);
    
    // ========== GPU 共享内存优化 Kernel (Tile Size = 16) ==========
    std::cout << "\n========== GPU GEMM (共享内存优化 - Tile 16x16) ==========" << std::endl;
    
    dim3 blockSize_16(TILE_SIZE_16, TILE_SIZE_16);
    dim3 gridSize_16((N + TILE_SIZE_16 - 1) / TILE_SIZE_16, 
                      (M + TILE_SIZE_16 - 1) / TILE_SIZE_16);
    
    // 预热
    gemm_kernel_shared_mem_16<<<gridSize_16, blockSize_16>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 多次运行取平均值
    total_time = 0.0;
    for (int i = 0; i < num_runs; i++) {
        CUDA_CHECK(cudaMemset(d_C, 0, size_C));
        
        auto start = std::chrono::high_resolution_clock::now();
        gemm_kernel_shared_mem_16<<<gridSize_16, blockSize_16>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_time += duration.count() / 1000.0;
    }
    double time_gpu_shared_16_ms = total_time / num_runs;
    
    CUDA_CHECK(cudaMemcpy(h_C_gpu_shared_16, d_C, size_C, cudaMemcpyDeviceToHost));
    print_performance_analysis("GPU (共享内存优化 - Tile 16)", time_gpu_shared_16_ms, M, N, K, TILE_SIZE_16);
    
    // ========== GPU 共享内存优化 Kernel (Tile Size = 32) ==========
    std::cout << "\n========== GPU GEMM (共享内存优化 - Tile 32x32) ==========" << std::endl;
    
    dim3 blockSize_32(TILE_SIZE_32, TILE_SIZE_32);
    dim3 gridSize_32((N + TILE_SIZE_32 - 1) / TILE_SIZE_32, 
                      (M + TILE_SIZE_32 - 1) / TILE_SIZE_32);
    
    // 预热
    gemm_kernel_shared_mem_32<<<gridSize_32, blockSize_32>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 多次运行取平均值
    total_time = 0.0;
    for (int i = 0; i < num_runs; i++) {
        CUDA_CHECK(cudaMemset(d_C, 0, size_C));
        
        auto start = std::chrono::high_resolution_clock::now();
        gemm_kernel_shared_mem_32<<<gridSize_32, blockSize_32>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_time += duration.count() / 1000.0;
    }
    double time_gpu_shared_32_ms = total_time / num_runs;
    
    CUDA_CHECK(cudaMemcpy(h_C_gpu_shared_32, d_C, size_C, cudaMemcpyDeviceToHost));
    print_performance_analysis("GPU (共享内存优化 - Tile 32)", time_gpu_shared_32_ms, M, N, K, TILE_SIZE_32);
    
    // ========== GPU 共享内存缓存优化 Kernel（不使用 K 维度分块）==========
    std::cout << "\n========== GPU GEMM (共享内存缓存 - 无 K 维度分块) ==========" << std::endl;
    
    // 配置：每个 block 处理一行，每个线程处理一列
    dim3 blockSize_cache(32, 1);  // 32 个线程，每个线程处理一列
    dim3 gridSize_cache((N + blockSize_cache.x - 1) / blockSize_cache.x, M);
    
    // 预热
    gemm_kernel_shared_cache_no_tiling<<<gridSize_cache, blockSize_cache>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 多次运行取平均值
    total_time = 0.0;
    for (int i = 0; i < num_runs; i++) {
        CUDA_CHECK(cudaMemset(d_C, 0, size_C));
        
        auto start = std::chrono::high_resolution_clock::now();
        gemm_kernel_shared_cache_no_tiling<<<gridSize_cache, blockSize_cache>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_time += duration.count() / 1000.0;
    }
    double time_gpu_cache_no_tiling_ms = total_time / num_runs;
    
    CUDA_CHECK(cudaMemcpy(h_C_gpu_cache_no_tiling, d_C, size_C, cudaMemcpyDeviceToHost));
    print_performance_analysis("GPU (共享内存缓存 - 无 K 维度分块)", time_gpu_cache_no_tiling_ms, M, N, K);
    
    // ========== 性能对比总结 ==========
    std::cout << "\n========== 性能对比总结 ==========" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "GPU (基础 - 无共享内存):       " << time_gpu_basic_ms << " ms" << std::endl;
    std::cout << "GPU (共享内存优化 - Tile 16):  " << time_gpu_shared_16_ms << " ms" 
              << " (相对基础提升: " << time_gpu_basic_ms / time_gpu_shared_16_ms << "x)" << std::endl;
    std::cout << "GPU (共享内存优化 - Tile 32):  " << time_gpu_shared_32_ms << " ms" 
              << " (相对基础提升: " << time_gpu_basic_ms / time_gpu_shared_32_ms << "x)" << std::endl;
    std::cout << "GPU (共享内存缓存 - 无 K 分块): " << time_gpu_cache_no_tiling_ms << " ms" 
              << " (相对基础提升: " << time_gpu_basic_ms / time_gpu_cache_no_tiling_ms << "x)" << std::endl;
    
    // 计算 GFLOPS 对比
    long long flops = (long long)M * N * 2 * K;
    double gflops_gpu_basic = (flops / 1e9) / (time_gpu_basic_ms / 1000.0);
    double gflops_gpu_shared_16 = (flops / 1e9) / (time_gpu_shared_16_ms / 1000.0);
    double gflops_gpu_shared_32 = (flops / 1e9) / (time_gpu_shared_32_ms / 1000.0);
    double gflops_gpu_cache_no_tiling = (flops / 1e9) / (time_gpu_cache_no_tiling_ms / 1000.0);
    
    std::cout << "\n算力对比 (GFLOPS):" << std::endl;
    std::cout << "GPU (基础 - 无共享内存):       " << gflops_gpu_basic << " GFLOPS" << std::endl;
    std::cout << "GPU (共享内存优化 - Tile 16):  " << gflops_gpu_shared_16 << " GFLOPS" << std::endl;
    std::cout << "GPU (共享内存优化 - Tile 32):  " << gflops_gpu_shared_32 << " GFLOPS" << std::endl;
    std::cout << "GPU (共享内存缓存 - 无 K 分块): " << gflops_gpu_cache_no_tiling << " GFLOPS" << std::endl;
    
    // 性能分析
    std::cout << "\n========== 性能分析 ==========" << std::endl;
    std::cout << "共享内存优化的优势：" << std::endl;
    std::cout << "1. Tile 16 相对基础版本提升: " << (time_gpu_basic_ms / time_gpu_shared_16_ms - 1.0) * 100 << "%" << std::endl;
    std::cout << "2. Tile 32 相对基础版本提升: " << (time_gpu_basic_ms / time_gpu_shared_32_ms - 1.0) * 100 << "%" << std::endl;
    std::cout << "3. Tile 32 相对 Tile 16 提升: " << (time_gpu_shared_16_ms / time_gpu_shared_32_ms - 1.0) * 100 << "%" << std::endl;
    std::cout << "4. 共享内存缓存（无 K 分块）相对基础版本提升: " << (time_gpu_basic_ms / time_gpu_cache_no_tiling_ms - 1.0) * 100 << "%" << std::endl;
    std::cout << "5. Tile 32 相对共享内存缓存（无 K 分块）提升: " << (time_gpu_cache_no_tiling_ms / time_gpu_shared_32_ms - 1.0) * 100 << "%" << std::endl;
    std::cout << "\n注意：共享内存缓存（无 K 分块）版本虽然使用共享内存，但效率不如分块版本，" << std::endl;
    std::cout << "因为某些数据仍会被重复读取。分块版本通过更好的数据重用获得更高性能。" << std::endl;
    
    // 清理资源
    free(h_A);
    free(h_B);
    free(h_C_gpu_basic);
    free(h_C_gpu_shared_16);
    free(h_C_gpu_shared_32);
    free(h_C_gpu_cache_no_tiling);
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    std::cout << "\n========== 测试完成 ==========" << std::endl;
    
    return 0;
}

