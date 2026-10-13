// sampler_gpu.h —— 采样的设备端实现（P3）
//
// ## 与 CPU 版的关系
//
// CPU 版的 `sampler.{h,cc}` 把采样拆成两半，各自可单独验证：
//
//   1. **确定性的一半** `FilterToProbs`：temperature / top_k / top_p 把 logits
//      变成"筛选后的概率分布"。同样的输入必得同样的分布。
//   2. **随机性的一半** `CounterUniform`：把宿主的 seed 与一个**绝对位置**
//      揉成 [0,1) 的均匀数 `u = H(seed, counter)`，纯函数。
//
// 本文件是这两半在 GPU 上的实现。`sampler.h` 在这里是**同源副本**（DESIGN §3），
// 文件名、字段名、默认值、注释全部逐字来自上游 —— 因为"top_k 的并列一起保留"、
// "top_p 至少留一个 token"、"贪心不消耗随机数"这些语义是**契约**，抄成两份
// 迟早会漂。`tests/check_sync.py` 盯着它。
//
// ⚠️ **只有 `sampler.h` 是副本，`sampler.cc` 不是**：那边的 `FilterToProbs` /
// `SampleToken` 依赖 CPU 侧的 `kernels.h`（ArgMax / SoftmaxInPlace），搬过来就
// 等于把 CPU 前向也搬进来 —— 那是 DESIGN §3 明确要重写的一栏。所以设备端是
// **重写**，而 CPU 的实现留在原处，由 `tools/build_cpu_ref.sh` 现编成参考侧。
//
// ## 判据：为什么这一块能"逐 token 对拍"
//
// DESIGN §6 第 10 条写过"采样不是能对拍的那一半"。那句话针对的是**接在模型后面**
// 的采样 —— 两边 logits 本来就差 fp16 的舍入，采样把它放大成"整段序列不同"。
// 但**把模型摘掉**之后，判据立刻回来了：
//
//     同一份 logits（从文件读，逐位相同）+ 同一个 seed ⇒ 两边必须给出同一个 token
//
// 于是采样的每一步都能独立钉死，与模型的数值精度完全解耦 —— 和 P2 里
// `kv_map_check` 把"布局"与"精度"分开是同一个套路。`tools/sample_check.cu`
// 就是按这条做的，**不需要模型、不需要权重**。
//
// ## 两条与 CPU 版对齐时最容易写错的地方
//
//   * **并存取的语义**：top_k 的阈值取"第 k 大的值"，`< 阈值` 才剔除（并列一起
//     保留）；top_p 的切点由**累积概率**决定，不是"取前几个"。差一个 `<` 还是
//     `<=`，判据会红，而那种红看起来像"浮点精度不够"。
//   * **贪心不消耗随机数**：`temperature <= 0` 时 CPU 的 `SampleToken` 直接
//     `ArgMax` 返回，不碰 `u`。设备端必须走同一个分支 —— 否则"T=0 与 M2 逐位
//     一致"这条向后兼容的承诺就断了。
//
// ⚠️ 有一处**刻意不与 CPU 逐位对齐**：GPU 的 `expf` 与 glibc 的 `expf` 不是
// 同一个实现（末位可能差 1 ULP）。于是概率会有 ~1e-7 的相对差，累积概率过
// `u` 的**位置**在极少数情况下会被推过一格。判据按 P2 的纪律处理：token 层
// 要求逐位相同，**若分歧，必须打印 `u` 与两侧的累积概率**，只有 `u` 距边界
// < 1e-6 时才允许存在（那正是 1 ULP 足以推过去的位置）。
#ifndef RPGBAR_SAMPLER_GPU_H_
#define RPGBAR_SAMPLER_GPU_H_

#include "cuda_common.h"

#ifdef RPG_HAS_CUDA

#include <cstdint>

// 采样参数的**唯一来源**（同源副本）。SampleOptions 里的 temperature / top_k /
// top_p 就是宿主在请求里能调的那三个。
#include "sampler.h"

namespace rpg {

// ---------------------------------------------------------------- 随机源
//
// counter-based：`u = H(seed, 绝对位置)`。**没有内部状态**，所以引擎整体仍是
// 纯函数（seed 与 prompt / temperature 一样都是宿主传进来的输入）。
//
// 三个都是**宿主函数**（生成循环里每步算一个 u）。之所以不标 `__host__ __device__`：
// 这个头会被 host 侧的 .cc 引用（engine.cc 要解析采样参数），而 `__host__` 这类
// 限定符在 nvcc 之外编不过。将来若要把采样整个塞进 kernel，再在 .cu 里另写一份
// device 版本 —— 那时判据会同时盯住两份（见 .cu 里对"重写 vs 复制"的说明）。
// ⚠️ 三行全是**整数运算 + 乘一个 2 的幂**，所以与 CPU 版的对拍判据是
// **逐位相同**，不是"误差在容差内"。
uint64_t DevSplitMix64(uint64_t x);
uint64_t DevCounterRng(uint64_t seed, uint64_t counter);
float DevCounterUniform(uint64_t seed, uint64_t counter);

// ---------------------------------------------------------------- 筛选
//
// 把 logits 处理成筛选后的概率分布（长度 n，和为 1，已 softmax），写到
// `probs_dev`。与 CPU 版 `FilterToProbs` 同语义：
//   * `temperature <= 0` ⇒ one-hot（argmax 位置为 1，并列取**最小下标**）；
//   * `top_k > 1 && top_k < n` ⇒ 阈值取第 k 大的值，`< 阈值` 的置 -inf；
//   * `top_p ∈ (0,1)` ⇒ 按累积概率切尾巴，**至少保留 1 个**；
//   * 切完尾巴后重新归一化（CPU 版也是这么做的）。
//
// `probs_dev` 的长度必须 ≥ n。不修改 `logits_dev`。
void GpuFilterToProbs(const float* logits_dev, int64_t n, const SampleOptions& opt,
                      float* probs_dev);

// 逆变换：累积概率第一次跨过 u 的那个 token 就是它（`cum > u`，与 CPU 版同一个
// 比较方向）。一个 block、一个线程、按**下标顺序**串行累加 —— 这是刻意的，
// 见 .cu 里的长注释（并行扫描会把求和顺序改掉，于是 u 落在边界上时选出的 token
// 可能与 CPU 不同）。
// 永远给出一个合法的 token：没有跨过（只可能被浮点误差推到这条分支）时返回
// probs 里概率最大的那个，与 CPU 版的兜底一致。
void GpuInverseSample(const float* probs_dev, int64_t n, float u,
                      int32_t* out_idx_dev);

// ---------------------------------------------------------------- 采样
//
// 采样一个 token。语义 = CPU 版 `SampleToken(logits, n, opt, u)`，其中
//     u = CounterUniform(seed, counter)
// `counter` 是**绝对位置**（这个 token 将要落进 cache 的下标）—— 用绝对下标
// 而不是"本轮第几个"：宿主每轮重发完整 messages 时同一段前缀的位置不变，
// 于是暖缓存（续写）与冷缓存（从头）会拿到同一个 u（与 CPU 版逐字同源的理由）。
//
// 贪心（`temperature <= 0`）时退化为 argmax，**不消耗随机数**。
// `probs_dev` 是中间缓冲（长度 ≥ n），由调用方给（GpuScratch::probs）。
void GpuSampleToken(const float* logits_dev, int64_t n, const SampleOptions& opt,
                    uint64_t seed, uint64_t counter, float* probs_dev,
                    int32_t* out_idx_dev);

}  // namespace rpg

#endif  // RPG_HAS_CUDA
#endif  // RPGBAR_SAMPLER_GPU_H_
