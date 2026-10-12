// kernels_gpu.h —— 显存上的算子（host 侧的启动入口）
//
// 与 CPU 版 src/kernels.h + matmul.h 是**同一个抽象层级**：一条向量进、一条
// 向量出，不暴露布局细节。区别是这里的实现在 .cu 里、跑在显存上。
//
// ## 精度口径（与 CPU 版对齐，但有一个例外）
//
// CPU 版的约定是"归一化的平方和与 softmax 的求和在 double 里累加"，理由是
// 单串行 float 累加器的相对误差随长度**线性**增长 —— 长度到 4096 时
// n·eps ≈ 2.4e-4，已经贴着 p99 ≤ 1e-3 的判据了。所以这里同样用 double 做
// 两个归约（RMSNorm 的平方和、softmax 的和）。
//
// **例外**：融合点积（GpuMatMulQuant）里的累加是 float。CPU 版那边也刻意用
// float —— 它的注释写着"attention 的加权求和是 fma 累加的浮点乘加，必须和
// torch 一样用 float 累加，否则反而引入额外舍入"。但 GPU 侧 128 个线程分段
// 累加再树形归约，求和顺序与 CPU 的 4 累加器不同，所以这一步**不可能逐位
// 相同**，判据只能是相对误差（见 docs/DESIGN.md §7）。
//
// ## 一律用默认 stream
//
// P1/P2 不做多流（DESIGN §4.3）：decode 每层十几个小 kernel，多流的
// cross-stream 事件同步比省下的并行更贵。等到 P4 用 CUDA Graph 把整段录成图。
#ifndef RPGBAR_KERNELS_GPU_H_
#define RPGBAR_KERNELS_GPU_H_

#include "cuda_common.h"

#ifdef RPG_HAS_CUDA

#include <cstdint>

#include "gpu_model.h"
#include "model_meta.h"

namespace rpg {

// 反量化一整行（n_elems 个元素）到 f32。用作逐位对拍的"设备侧答案"，
// 也用于把任意张量的原始值拿回 host 看。
void GpuDequantRow(uint32_t type, const uint8_t* src_dev, int64_t n_elems,
                   float* dst_dev);

// OUT[t, r] = dot(W 的第 r 行, X 的第 t 行)，即 OUT = X · Wᵀ。
//   W   : [rows, cols] 的量化矩阵（显存，原始字节）
//   X   : [n_tokens, cols] 行主（显存）
//   OUT : [n_tokens, rows] 行主（显存）
// 权重只在 kernel 里即时反量化一次，**同一份权重服务本 tile 的所有 token** ——
// 这是"量化直传显存"能划算的关键（否则每个 token 都要重扫一遍 1 GiB 权重）。
void GpuMatMulQuant(const GpuTensor& w, const float* X_dev, int64_t n_tokens,
                    float* OUT_dev);

// 逐行 RMSNorm：out[row] = x[row] * rsqrt(mean(x[row]²) + eps) * w
//   n_rows 行、每行 n 个元素、行间距 stride（out 可以与 x 同一块内存）。
// 传 stride 是为了让它同时覆盖两种用法：
//   * 整个残差流：n = hidden, stride = hidden, n_rows = seq
//   * QK-norm  ：n = head_dim, stride = head_dim, n_rows = seq * n_heads
void GpuRmsNorm(const float* x, const float* w, int64_t n_rows, int64_t n,
                int64_t stride, float eps, float* out);

// RoPE（**半维配对**，不是相邻配对 —— 写错不崩、只是胡说）。
// x 是 [n_rows, n_heads, head_dim]；rope_tab 是 [max_pos, head_dim]，一行内
// 前半 cos、后半 sin。位置取 pos0 + row。
void GpuRope(float* x, int32_t n_heads, int32_t head_dim, const float* rope_tab,
             int64_t pos0, int64_t n_rows);

// 全因果注意力（GQA，无滑动窗口）：out = softmax(QKᵀ/√d)V。
// q: [seq, n_heads*head_dim]   k/v: [seq, n_kv_heads*head_dim]   out: 同 q。
// 位置 t 只看 [0, t] —— 掩码靠"只遍历到 t"实现，不写 -inf 的格子。
void GpuCausalAttention(const ModelMeta& c, const float* q_dev,
                        const float* k_dev, const float* v_dev, int64_t seq,
                        float* out_dev);

// gate[i] = silu(gate[i]) * up[i]，就地。
void GpuSiluMulInPlace(float* gate_dev, const float* up_dev, int64_t n);

// a[i] += b[i]，就地。
void GpuVecAddInPlace(float* a_dev, const float* b_dev, int64_t n);

// embedding 查表：x[t] = 第 toks[t] 行的反量化值。
void GpuEmbeddingLookup(const GpuTensor& emb, const int32_t* toks_dev, int64_t seq,
                        float* x_dev);

}  // namespace rpg

#endif  // RPG_HAS_CUDA
#endif  // RPGBAR_KERNELS_GPU_H_
