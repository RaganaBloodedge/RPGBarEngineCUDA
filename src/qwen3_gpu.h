// qwen3_gpu.h —— Qwen3 前向（显存版）
//
// 与 CPU 版 src/qwen3.h 的对应关系：**结构照搬，实现重写**。
//   * ForwardFull（整段前向，无 cache）→ GpuForwardFull
//   * ForwardLayer                      → GpuForwardLayer（内部函数）
//   * KV Cache 的增量路径              → **P2**，这里还没有
//
// ## 为什么 P1 先做"无 cache 的全序列前向"
//
// 单层对拍要的是"这一层的算术对不对"。增量解码会把 KV Cache 的精度（fp16）、
// 前缀复用的位置偏移、cache 布局三件事一起搅进来 —— 那三件事任何一件错了，
// 症状都是"某层的输出偏了一点"，与"某个 kernel 错了"长得一模一样。
// 所以先在没有 cache 的路径上把算术逐层钉死（判据 P1），再让 cache 进来（P2）。
// 顺序反了的话，P1 那条 p99 ≤ 1e-3 会变成一个永远查不清的混合指标。
#ifndef RPGBAR_QWEN3_GPU_H_
#define RPGBAR_QWEN3_GPU_H_

#include "cuda_common.h"

#ifdef RPG_HAS_CUDA

#include <cstdint>
#include <vector>

#include "gpu_model.h"
#include "model_meta.h"

namespace rpg {

// 前向的临时缓冲（显存）。与 CPU 版 ForwardScratch 一一对应。
//
// ⚠️ **只增不减**。CPU 版那边踩过一次"按本段长度缩容、但索引是绝对位置"的
// 堆越界（症状是"单轮正常、第二轮进程无声消失"）。这里的索引全是段内局部
// 下标，理论上按 seq 精确重分配也不会错 —— 但"只增不减"这个策略把整类
// 问题从"要靠推理确认安全"变成"不可能发生"，代价只是多占几百 KB。
class GpuScratch {
 public:
  void Ensure(const ModelMeta& c, int64_t seq);

  int64_t bytes() const;

  DeviceBuffer x;     // [seq, hidden]  残差流
  DeviceBuffer h;     // [seq, hidden]  归一化输出
  DeviceBuffer q;     // [seq, q_dim]
  DeviceBuffer k;     // [seq, kv_dim]
  DeviceBuffer v;     // [seq, kv_dim]
  DeviceBuffer attn;  // [seq, q_dim]
  DeviceBuffer proj;  // [seq, hidden]
  DeviceBuffer gate;  // [seq, intermediate]
  DeviceBuffer up;    // [seq, intermediate]
  DeviceBuffer toks;  // [seq] int32
  DeviceBuffer logits;  // [seq, vocab]
};

struct GpuForwardOptions {
  // 只跑前 N 层（0 = 全部）。截断时不计算末端归一化与 lm_head，logits 留空
  // —— 与 CPU 版 ForwardOptions 同语义，用于把"误差从第几层开始"定位到层。
  int32_t layer_limit = 0;
  // 非空时逐层收集残差流 [n_use+1, seq, hidden]（会做 n_use+1 次 显存→内存 拷贝）。
  std::vector<float>* capture_hidden = nullptr;
};

// 整段前向（**不使用 KV Cache**）。tokens 是 host 上的 token id 数组。
// logits 输出 [seq, vocab] 行主（仅在 layer_limit 足额时非空）。
void GpuForwardFull(const GpuModel& m, const int32_t* tokens, int64_t seq,
                    GpuScratch* s, std::vector<float>* logits,
                    const GpuForwardOptions& opt = GpuForwardOptions());

}  // namespace rpg

#endif  // RPG_HAS_CUDA
#endif  // RPGBAR_QWEN3_GPU_H_
