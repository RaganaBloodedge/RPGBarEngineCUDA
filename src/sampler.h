// RPGBarEngine —— 采样：确定性筛选 + counter-based 随机源
//
// 引擎里**没有有状态的随机数生成器**（DESIGN §12 #6）。这个文件把采样拆成
// 两半，各自都能单独验证：
//
//   1. 确定性的一半 —— `FilterToProbs`：按 temperature / top_k / top_p 把
//      logits 变成"筛选后的概率分布"。同样的输入必得同样的分布，可以逐元素
//      和参考实现对拍。温度就活在这一半（它缩放 logits，不碰随机数）。
//
//   2. 随机性的一半 —— `CounterUniform`：把**宿主给的 seed** 与一个**绝对
//      位置**揉成一个 [0,1) 的均匀数。这是纯函数：同一个 (seed, 位置) 永远
//      得到同一个 u，所以引擎整体仍是纯函数（seed 与 prompt / temperature
//      一样，都是宿主传进来的输入）。
//
// 为什么用 counter-based 而不是"宿主递一串随机数"：生成 30 个 token 就要 30
// 个随机数，让宿主替引擎攒这些数既别扭又容易错位。宿主只递一个 seed，引擎按
// `u_i = H(seed, i)` 展开 —— 位置取 KV Cache 的**绝对下标**，于是：
//   * 同一个 seed + 同一段 prompt → 同一串 u → 整局可复现；
//   * 暖缓存（续写）与冷缓存（从头）逐位一致 —— 位置相同，u 就相同；
//   * 多会话（DM / 小助手）各拿各的 seed，随机流互不扰动。
//
// 温度不会改变这里的任何随机数 —— 它只走 `FilterToProbs` 那条链。两者在
// `SampleToken` 最后一步才汇合：用 u 在筛选后的分布上做逆变换。

#ifndef RPGBAR_SAMPLER_H_
#define RPGBAR_SAMPLER_H_

#include <cstdint>
#include <vector>

namespace rpg {

// 采样的确定性参数。引擎侧有一组固定的兜底默认（宿主通常只调 temperature，
// top_k / top_p 由引擎钉住，见 DESIGN §12 #6 与 README「关于随机性」）。
struct SampleOptions {
  // <= 0 视为贪心（等价于 argmax，不消耗随机数）。1.0 = 原始分布。
  float temperature = 1.0f;
  // 只保留概率最高的前 k 个（0 = 关闭）。与 HF TopKLogitsWarper 同语义：
  // 阈值取第 k 大的值，`< 阈值` 的都被剔除（并列的一起保留）。
  int32_t top_k = 0;
  // 核采样：只保留累积概率刚跨过 top_p 的那一小撮（1.0 = 关闭）。
  // 与 HF TopPLogitsWarper 同语义，且**至少保留 1 个** token。
  float top_p = 1.0f;
};

// SplitMix64 —— 一个把 64 位整数彻底打散的混合函数（无状态、可复现）。
uint64_t SplitMix64(uint64_t x);

// counter-based 随机源：u = H(seed, counter)。纯函数，无内部状态。
//   * seed    —— 宿主的整局种子（同一局用同一个）；
//   * counter —— 这个随机数对应的**绝对位置**（KV Cache 下标）。
// 取结果的高 24 位映射到 [0,1)，避免用到质量较差的低位。
uint64_t CounterRng(uint64_t seed, uint64_t counter);
float CounterUniform(uint64_t seed, uint64_t counter);

// 把 logits 处理成筛选后的概率分布（长度 n，和为 1，已 softmax）。
// 不修改传入的 logits。temperature <= 0 时退化为 one-hot（argmax 位置为 1）。
// 这条是采样的**确定性一半** —— 对拍就比它。
void FilterToProbs(const float* logits, int64_t n, const SampleOptions& opt,
                   std::vector<float>* probs);

// 用宿主给的 u ∈ [0,1) 在筛选后的分布上做逆变换，返回选中的 token。
// 这是确定性筛选与随机性的汇合点：温度走 FilterToProbs，seed 走调用方算出的 u。
int32_t SampleToken(const float* logits, int64_t n, const SampleOptions& opt,
                    float u);

}  // namespace rpg

#endif  // RPGBAR_SAMPLER_H_
