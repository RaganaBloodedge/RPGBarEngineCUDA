// qwen3_gpu.h —— Qwen3 前向（显存版）
//
// 与 CPU 版 src/qwen3.h 的对应关系：**结构照搬，实现重写**。
//   * ForwardFull（整段前向，可带 cache）→ GpuForwardFull
//   * ForwardStep（单 token 增量）        → GpuForwardStep
//   * ForwardLayer                        → GpuForwardLayer（内部函数）
//   * Generate / GenerateGreedy           → GpuGenerate / GpuGenerateGreedy
//
// ## 两条路径共用同一份层实现（这是刻意的）
//
// CPU 版的 `ForwardStep` 只有三行 —— 它就是 `ForwardFull(seq = 1, kv != nullptr)`。
// 于是"填 cache 的全序列"与"不填 cache 的全序列"在数值上必然**由构造相同**，
// 区别只剩 attention 的数据来源。这条在 GPU 侧一样成立，而且是 P2 唯一能用的
// 布局自检：cache 布局写错不会崩，只会让生成结果慢慢跑偏，那种 bug 只能靠
// "两条路径互相印证"抓出来。
//
// ⚠️ 但请注意**判据的口径变了**：CPU 侧那条自检是"逐位相同"（K/V 都是 fp32，
// 两边算的是同一串数）；GPU 侧 KV 是 fp16（DESIGN §4.2），写进 cache 的那一刻
// 就有舍入 ⇒ 只能要求"相对误差 p99 ≤ 1e-3 + argmax 一致"。**这个降级是有意的**，
// 它换来的是每会话省 448 MiB 显存。如果哪天这条自检红得很大（不是 1e-3 量级），
// 那不是精度问题，是布局错了 —— 这正是留着它的意义。
//
// ## P1 为什么先做"无 cache 的全序列前向"
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
#include "kv_cache_gpu.h"
#include "model_meta.h"

namespace rpg {

// 前向的临时缓冲（显存）。与 CPU 版 ForwardScratch 一一对应。
//
// ⚠️ **只增不减**。CPU 版那边踩过一次"按本段长度缩容、但索引是绝对位置"的
// 堆越界（症状是"单轮正常、第二轮进程无声消失"）。这里的索引全是段内局部
// 下标，理论上按 seq 精确重分配也不会错 —— 但"只增不减"这个策略把整类
// 问题从"要靠推理确认安全"变成"不可能发生"，代价只是多占几百 KB。
//
// ⚠️ **CPU 版的 `scores` 缓冲在这里不存在**，原因值得记一笔：那边的 scores 在
// host 内存里、按绝对位置索引，所以必须按 `pos0 + seq` 备、且不能与"按 seq 备"
// 的缓冲共用一个 Ensure()。显存版把 scores 放在 kernel 的**共享内存**里、
// 每个 block 按需分配（`AttentionFromCacheKernel` 的 n_scores），生命周期止于
// 这一次 launch ⇒ 那一整类越界在这里不可达。
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
  // argmax 的结果落点（1 个 int32）。做成一个具名槽位而不是就地 cudaMalloc：
  // 每步分配一次 4 字节是同步操作（几十微秒），而且**不许用命名空间级静态**
  // 的 DeviceBuffer 来缓存它 —— 那种静态对象要到进程退出才析构，那时 CUDA
  // 上下文可能已经销毁（RPGBarEngine 的"退出挂起"就是这么来的）。
  DeviceBuffer argmax_idx;
};

struct GpuForwardOptions {
  // 只跑前 N 层（0 = 全部）。截断时不计算末端归一化与 lm_head，logits 留空
  // —— 与 CPU 版 ForwardOptions 同语义，用于把"误差从第几层开始"定位到层。
  int32_t layer_limit = 0;
  // 非空时逐层收集残差流 [n_use+1, seq, hidden]（会做 n_use+1 次 显存→内存 拷贝）。
  std::vector<float>* capture_hidden = nullptr;
};

// 整段前向。tokens 是 host 上的 token id 数组。
//
// kv != nullptr 时：
//   * 把这一段的 K/V 追加写入 cache（位置从 cache 当前长度开始）；
//   * 注意力用 cache 里的**全部历史**（前缀 + 本段）；
//   * 写完后 cache 长度 += seq。
// 这既是 prefill，也是"复用上一轮相同前缀"的手段 —— 与 CPU 版同语义。
//
// logits 为 **nullptr** 时只把结果算到显存（`s->logits`），不下载到 host。
// 增量解码每步都要 logits，但只需要它的 argmax 一个下标 —— 把 607 KB 拷回
// host 再丢掉，是 decode 里唯一一处可以整块省掉的搬运（见 GpuGenerate）。
//
// ⚠️ 用 kv 时 seq 可以 > 1（prefill），但 cache 只支持**顺序追加**：本段的
// pos0 就是 cache 当前长度。
void GpuForwardFull(const GpuModel& m, const int32_t* tokens, int64_t seq,
                    GpuScratch* s, std::vector<float>* logits,
                    const GpuForwardOptions& opt = GpuForwardOptions(),
                    GpuKvCache* kv = nullptr);

// 单 token 增量前向。pos 必须等于 cache 的当前长度（只支持顺序追加）。
// logits 可为 nullptr（只算到显存）。语义与 CPU 版 ForwardStep 一致：
// 它就是 ForwardFull(seq = 1, kv) —— 两条路径不是"两份实现"，是同一个函数
// 在不同的 (seq, pos0) 上跑。
void GpuForwardStep(const GpuModel& m, int32_t token, int64_t pos, GpuKvCache* kv,
                    GpuScratch* s, std::vector<float>* logits);

// ---------------------------------------------------------------- 生成

struct GpuGenerateOptions {
  int32_t max_tokens = 64;
  // 命中任一 id 即停。停止 token 本身**计入**返回序列（与 CPU 版
  // GenerateOptions 一致 —— 它也会把 eos 放进输出）。
  std::vector<int32_t> stop_ids;
  // ⚠️ **没有 temperature / top_k / top_p**：P2 只做贪心（DESIGN §5 的阶段表里
  // 采样参数是 P3）。这不是"还没写"，是刻意不写 —— 贪心是可对拍的（T=0 下
  // 与 CPU 版逐 token 比），采样不是。先把确定性的那条路径钉死。
  //
  // 非空时逐步收集每一步的 logits（形状 [返回的 token 数, vocab]）。**默认
  // nullptr**，此时 decode 循环里一次 607 KB 的搬运都没有；对拍工具才需要它
  // （用来在分歧处打印两边的 top-5 与两个 token 的 logit 差，DESIGN §7.3）。
  std::vector<float>* step_logits = nullptr;
};

// 贪心解码。first_logits_dev 是"下一个 token"的分布所在的**显存**地址
// （[vocab]，即 prefill 最后一个位置那行 = `s->logits` 的最后一行）。
// 生成的 token 追加进 out_ids（不含 prompt）。
//
// 与 CPU 版 Generate 的两条不变量一致：
//   * **每个**返回的 token 都写进 cache（包括最后那个与停止 token）——
//     代价是末尾多算一步前向，换来 `cache.len() == prompt 长度 + 返回 token 数`，
//     调用方能直接在 cache 上续下一轮而不必 reset；
//   * 上下文满了明确报错，不静默截断（静默停下会让人以为"模型自己讲完了"）。
void GpuGenerateGreedy(const GpuModel& m, GpuKvCache* kv, GpuScratch* s,
                       const float* first_logits_dev,
                       const GpuGenerateOptions& opt,
                       std::vector<int32_t>* out_ids);

}  // namespace rpg

#endif  // RPG_HAS_CUDA
#endif  // RPGBAR_QWEN3_GPU_H_
