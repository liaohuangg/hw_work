#include <iostream>
#include <vector>
#include <random>
#include <cmath>

// 定义浮点数类型（fp32）
using fp32 = float;

/**
 * @brief 2D卷积运算（CONV2D）实现
 * @param X 输入特征图，维度：[N, C_in, H_in, W_in]
 * @param K 卷积核，维度：[C_out, C_in, K_h, K_w]
 * @param stride 步长（仅支持stride_h=stride_w=stride）
 * @param padding 填充（仅支持padding_h=padding_w=padding）
 * @return 输出特征图，维度：[N, C_out, H_out, W_out]
 */
std::vector<fp32> conv2d(
    const std::vector<fp32>& X,
    const std::vector<fp32>& K,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride = 1,
    int padding = 0
) {
    // 计算输出特征图的高和宽
    int H_out = (H_in - K_h + 2 * padding) / stride + 1;
    int W_out = (W_in - K_w + 2 * padding) / stride + 1;

    // 初始化输出特征图（全0）
    std::vector<fp32> Y(N * C_out * H_out * W_out, 0.0f);

    // 遍历批量（这里N=1）
    for (int n = 0; n < N; ++n) {
        // 遍历输出通道
        for (int c_out = 0; c_out < C_out; ++c_out) {
            // 遍历输出特征图的高
            for (int h_out = 0; h_out < H_out; ++h_out) {
                // 遍历输出特征图的宽
                for (int w_out = 0; w_out < W_out; ++w_out) {
                    // 计算当前输出像素对应的输入窗口起始位置
                    int h_in_start = h_out * stride - padding;
                    int w_in_start = w_out * stride - padding;

                    fp32 sum = 0.0f;
                    // 遍历输入通道
                    for (int c_in = 0; c_in < C_in; ++c_in) {
                        // 遍历卷积核的高
                        for (int k_h = 0; k_h < K_h; ++k_h) {
                            // 遍历卷积核的宽
                            for (int k_w = 0; k_w < K_w; ++k_w) {
                                // 计算输入特征图的当前坐标
                                int h_in = h_in_start + k_h;
                                int w_in = w_in_start + k_w;

                                // 边界检查：超出输入范围则跳过（padding=0时可省略，但保留鲁棒性）
                                if (h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in) {
                                    continue;
                                }

                                // 计算输入特征图的索引（NCHW布局）
                                size_t x_idx = n * C_in * H_in * W_in 
                                             + c_in * H_in * W_in 
                                             + h_in * W_in 
                                             + w_in;

                                // 计算卷积核的索引（C_out, C_in, K_h, K_w）
                                size_t k_idx = c_out * C_in * K_h * K_w 
                                             + c_in * K_h * K_w 
                                             + k_h * K_w 
                                             + k_w;

                                // 点积累加
                                sum += X[x_idx] * K[k_idx];
                            }
                        }
                    }

                    // 赋值给输出特征图（NCHW布局）
                    size_t y_idx = n * C_out * H_out * W_out 
                                 + c_out * H_out * W_out 
                                 + h_out * W_out 
                                 + w_out;
                    Y[y_idx] = sum;
                }
            }
        }
    }

    return Y;
}

// 辅助函数：生成随机浮点数数组
std::vector<fp32> generate_random_data(size_t size, fp32 min = 0.0f, fp32 max = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<fp32> dist(min, max);

    std::vector<fp32> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
    return data;
}

int main() {
    // ========== 1. 定义卷积参数 ==========
    const int N = 1;         // 批量大小
    const int C_in = 3;      // 输入通道数
    const int H_in = 112;    // 输入高度
    const int W_in = 112;    // 输入宽度
    const int C_out = 4;     // 输出通道数
    const int K_h = 3;       // 卷积核高度
    const int K_w = 3;       // 卷积核宽度
    const int stride = 1;    // 步长
    const int padding = 0;   // 填充

    // ========== 2. 计算输出维度 ==========
    const int H_out = (H_in - K_h + 2 * padding) / stride + 1;
    const int W_out = (W_in - K_w + 2 * padding) / stride + 1;
    std::cout << "输出特征图维度：N=" << N << ", C_out=" << C_out 
              << ", H_out=" << H_out << ", W_out=" << W_out << std::endl;
    // 验证输出维度是否为110×110
    if (H_out != 110 || W_out != 110) {
        std::cerr << "错误：输出维度计算错误！" << std::endl;
        return -1;
    }

    // ========== 3. 生成输入数据和卷积核 ==========
    const size_t X_size = N * C_in * H_in * W_in;  // 1*3*112*112 = 37632
    const size_t K_size = C_out * C_in * K_h * K_w;// 4*3*3*3 = 108
    std::vector<fp32> X = generate_random_data(X_size);
    std::vector<fp32> K = generate_random_data(K_size);

    // ========== 4. 执行卷积运算 ==========
    std::vector<fp32> Y = conv2d(X, K, N, C_in, H_in, W_in, C_out, K_h, K_w, stride, padding);

    // ========== 5. 验证输出大小 ==========
    const size_t Y_size = N * C_out * H_out * W_out;  // 1*4*110*110 = 48400
    if (Y.size() != Y_size) {
        std::cerr << "错误：输出特征图大小错误！预期：" << Y_size << "，实际：" << Y.size() << std::endl;
        return -1;
    }

    // ========== 6. 输出示例结果（验证） ==========
    std::cout << "卷积运算完成！" << std::endl;
    std::cout << "输入大小：" << X_size << " 个fp32" << std::endl;
    std::cout << "卷积核大小：" << K_size << " 个fp32" << std::endl;
    std::cout << "输出大小：" << Y_size << " 个fp32" << std::endl;
    std::cout << "输出特征图第一个元素值：" << Y[0] << std::endl;

    return 0;
}