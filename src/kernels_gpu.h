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

// **仅供判据工具**：强制 GpuMatMulQuant 走通用（P4 之前那版）kernel。
// 默认关闭 —— 默认走 P4 的 Q4_K / Q6_K 专用 kernel。
//
// 存在的理由与"P1 故意留着无 cache 的 attention 路径"是同一个：新旧两份实现
// 都在，才能把差异**归因**到"只是换了写法"。`tools/matmul_bench --verify` 用它
// 做同一份权重/输入下两条路径的**逐位**对拍。生产路径不会调它。
void GpuSetForceGenericMatMul(bool force);
bool GpuForceGenericMatMul();

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
//
// **这是 P1 的无 cache 路径**（q/k/v 全是本段算出来的）。带前缀的版本是
// GpuAttentionFromCache —— 两条路径刻意分开、也刻意都要留着：它能一次性证明
// "cache 的写入/读取没把布局搞错"（两条路径在同一段 token 上的差异只该来自
// KV 的 fp16 舍入，不该是别的）。
void GpuCausalAttention(const ModelMeta& c, const float* q_dev,
                        const float* k_dev, const float* v_dev, int64_t seq,
                        float* out_dev);

// ---------------------------------------------------------------- P2：KV Cache

// 把本段算出的 K/V 写进 cache（fp32 → fp16，round-to-nearest）。
//   k_dev / v_dev : [seq, n_kv_heads * head_dim] 行主，本段的 K/V
//   k_layer/v_cache_layer : **第 layer 层的**基址，[n_kv_heads, max_seq, head_dim]
//   pos0          : 本段第一个位置在 cache 里的下标
//
// ⚠️ 源布局是 [t][head][dim]（一个 token 的多头排在一起），目标是 [head][pos][dim]
// （一个头的连续位置排在一起）。**这两者不同**，所以不是整段 memcpy，而是一次
// 带下标重排的遍历。写反了的症状：不崩，注意力读到别的头/别的位置的历史，
// 生成结果从某个位置起慢慢跑偏。
void GpuKvWrite(const float* k_dev, const float* v_dev, int32_t n_kv_heads,
                int32_t head_dim, int64_t pos0, int64_t seq,
                uint16_t* k_cache_layer, uint16_t* v_cache_layer, int64_t max_seq);

// 从 KV Cache 读历史的因果注意力（GQA）。
//   q_dev      : [seq, n_heads * head_dim]
//   k/v_layer  : **第 layer 层的**基址，fp16，[n_kv_heads, max_seq, head_dim]
//   max_seq    : 一个头占的位置数（= 头与头之间的步长）
// 位置 pos = pos0 + t 能看到 cache 里 [0, pos] 的全部历史 —— 包括本段里排在它
// 前面的（它们上一步已经写进去了），因果性自然成立，不需要额外的掩码。
//
// ⚠️ `sh_positions` 决定 kernel 里的 scores 共享内存按**多少个位置**备。
//   * 省略（< 0）⇒ 按 `pos0 + seq`。这是 P2 起就在用的口径，逐字不变。
//   * 显式给值 ⇒ 按它备。**CUDA Graph 路径必须给** `kv->max_seq()`：shared memory
//     的大小是启动参数、会被录进图，而回放时 pos0 每步递增 —— 照捕获那一刻的
//     pos0 备，第 2 步起就越界写（症状同样是"第一个 token 对、之后慢慢跑偏"）。
void GpuAttentionFromCache(const ModelMeta& c, int64_t pos0, const float* q_dev,
                           int64_t seq, const uint16_t* k_layer,
                           const uint16_t* v_layer, int64_t max_seq,
                           float* out_dev, int64_t sh_positions = -1);

// 把 AttentionFromCacheKernel 的动态共享内存上限**在捕获之前**抬到位。
//
// 为什么必须单独有这么个入口：`cudaFuncSetAttribute` 是**非法**的捕获期 API
// （它改的是函数属性，图里没有这种节点），而 scores 的共享内存大小是启动参数、
// 会被录进图 ⇒ 图路径只能按 `max_seq` 一次备够，也就只能在 BeginCapture 之前
// 把这个上限抬好。CaptureDecodeStep 就是先调它、再 BeginCapture。
//
// 参数就是它将要用的位置数（图路径传 `kv->max_seq()`）。**幂等**：不超过
// 48 KiB 时什么都不做 —— 1.7B/4B 在 RPGBar 的上下文上限（6144）下正是这种情况，
// 于是这条在默认配置里是个空操作，不会悄悄动全局函数属性。
void GpuAttentionWarmupShmem(int64_t sh_positions);

// 设备端 argmax：把**最大值的下标**写到 out_idx_dev（1 个 int32）。
// **并列时取最小下标**，与 CPU 版 kernels.h 的 ArgMax 同语义（`x[i] > bv`
// 严格大于）—— 差一个并列规则就会让"逐 token 一致"的判据红，而那种红会被
// 误读成"数值精度不够"。
//
// 存在的理由：贪心解码每步都要 argmax，而 logits 是 [vocab] = 151936 个 float
// （607 KB）。为了取一个下标回拷 607 KB 是纯浪费 —— 归约在设备端做完，调用方
// 只回拷 4 个字节。
//
// ⚠️ 结果写在调用方给的显存里、**不自动回拷**，也不用函数内的静态缓冲：
// 每步 cudaMalloc 一次 4 字节是同步操作（几十微秒），而命名空间级的静态
// DeviceBuffer 要到进程退出才析构 —— 那时 CUDA 上下文可能已经销毁
// （RPGBarEngine 的"退出挂起"就是这么来的）。落点用 GpuScratch::argmax_idx。
void GpuArgMax(const float* x_dev, int64_t n, int32_t* out_idx_dev);


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
