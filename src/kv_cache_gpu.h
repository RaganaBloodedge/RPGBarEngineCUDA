// kv_cache_gpu.h —— 显存里的 KV Cache（fp16）
//
// 与 CPU 版 src/kv_cache.h 的对应关系：**布局照搬，元素类型与存放位置不同**。
//   CPU 版：[n_layers, n_kv_heads, max_seq, head_dim]，host 内存，fp32
//   这里  ：同一个维度顺序，**显存**，**fp16**
//
// ## 为什么维度顺序一个字不改
//
// 那个顺序（层 → KV 头 → 位置 → 维度）在 CPU 上是为访存局部性选的（见 CPU 版
// 的注释：同一层的 K/V 是连续的一大块、一个头的连续位置连续、一条 head 向量
// 自身连续）。GPU 上这三条理由**一条都没变**，而且多出一条：kernel 里
// "第 layer 层的 K 基址" = `k + layer * n_kv_heads * max_seq * head_dim` ——
// 层号是纯乘法，没有除法与取模。换个布局（比如把层放到最内）会让每个 kernel
// 多算两次整除，还会把 cache 的写入打散成 max_seq 次不连续的小事务。
// ⇒ **两边布局一致不是"顺手"，是刻意**：对拍时"差异只来自算力层"这句话才成立，
//   显存里取错一格与算错一个数长得一模一样，布局一致能少掉一整类怀疑。
//
// ## 为什么是 fp16
//
// @4096 上下文、1.7B（28 层 / 8 KV 头 / head_dim 128）：
//   fp32 = 2 × 28 × 8 × 4096 × 128 × 4 字节 = **896 MiB**
//   fp16 = 同式 × 2 字节             = **448 MiB**
// 一行之差省掉 448 MiB。代价写进 DESIGN §4.2：**"两边输出逐位相同"这个判据从此
// 不存在**，对拍全部换成容差判据（§7）。这是有意的取舍，不是妥协。
//
// ## 只支持顺序追加
//
// 与 CPU 版同一条：写入的 pos 必须从当前长度开始连续。DM 与小助手各自的会话都是
// 一条直线，没有需要"插空"的场景；而支持任意写入会让 len 的语义变模糊（"已写入
// 数量"还是"最大位置"？），还要多维护一张空洞表。
#ifndef RPGBAR_KV_CACHE_GPU_H_
#define RPGBAR_KV_CACHE_GPU_H_

#include "cuda_common.h"

#ifdef RPG_HAS_CUDA

#include <cstdint>
#include <string>

#include "model_meta.h"

namespace rpg {

// 元素类型用 uint16_t 而不是 __half：这个头文件会被 host 侧的 .cc 引用
// （engine.cc 将来要建 cache），而 cuda_fp16.h 不该进 host TU —— 它带进整套
// __host__/__device__ 宏，对 host 编译没有收益、只有风险（cuda_common.h 的
// 头注释记着这个坑）。kernel 侧 reinterpret_cast<__half*> 即可，
// 两种写法都是 2 字节、位模式相同。
class GpuKvCache {
 public:
  GpuKvCache() = default;
  // 立刻分配 max_seq 全长 —— 不做"按需增长"。理由与 CPU 版同：显存分配的失败
  // 要在**建会话时**报出来（那时还能选小一点的 max_seq），而不是在生成到第
  // 3000 个 token 时炸在某个 kernel 里。
  GpuKvCache(const ModelMeta& c, int64_t max_seq);

  int64_t len() const { return len_; }
  int64_t max_seq() const { return max_seq_; }
  int32_t n_layers() const { return n_layers_; }
  int32_t n_kv_heads() const { return n_kv_heads_; }
  int32_t head_dim() const { return head_dim_; }
  bool valid() const { return max_seq_ > 0; }

  // 第 layer 层的 K/V 基址（fp16，[n_kv_heads, max_seq, head_dim] 展开）。
  // 越界检查在这里做一次 —— attention 与写入 kernel 每步都要拿这个基址，
  // 而层号来自循环、不来自数据，检查一次就够。
  uint16_t* k_layer(int32_t layer);
  uint16_t* v_layer(int32_t layer);
  const uint16_t* k_layer(int32_t layer) const;

  // 长度归零但**保留显存**（同一房间开新的一局时不必重分配那 448 MiB）。
  // 与 CPU 版同一条：old 值还在，只是再也读不到（attention 只遍历 [0, len)）。
  void Reset() { len_ = 0; }

  void SetLen(int64_t n);
  void AddLen(int64_t n) { SetLen(len_ + n); }

  // K + V 两块的总字节数（fp16）。
  int64_t bytes() const {
    return static_cast<int64_t>(k_.size()) + static_cast<int64_t>(v_.size());
  }

  // 一行摘要，给工具与日志用。
  // 例：KV Cache(fp16) 28 层 × 8 KV 头 × 4096 × 128 = 448.0 MiB（当前 21）
  std::string Describe() const;

 private:
  int64_t LayerElems() const {
    return static_cast<int64_t>(n_kv_heads_) * max_seq_ * head_dim_;
  }

  int32_t n_layers_ = 0;
  int32_t n_kv_heads_ = 0;
  int32_t head_dim_ = 0;
  int64_t max_seq_ = 0;
  int64_t len_ = 0;
  DeviceBuffer k_;  // [n_layers * n_kv_heads * max_seq * head_dim] fp16
  DeviceBuffer v_;
};

}  // namespace rpg

#endif  // RPG_HAS_CUDA
#endif  // RPGBAR_KV_CACHE_GPU_H_
