// qwen3_gpu.cu —— Qwen3 前向的实现（见 qwen3_gpu.h 的说明）

#include "qwen3_gpu.h"

#ifdef RPG_HAS_CUDA

#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "common.h"
#include "kernels_gpu.h"
#include "cuda_common.h"

namespace rpg {

namespace {

float* F(DeviceBuffer& b) { return static_cast<float*>(b.data()); }
const float* CF(const DeviceBuffer& b) {
  return static_cast<const float*>(b.data());
}

// 只增不减（理由见 GpuScratch 的注释）。
void Need(DeviceBuffer& b, int64_t n_floats) {
  RPG_CHECK(n_floats > 0, "缓冲尺寸必须为正");
  const size_t bytes = static_cast<size_t>(n_floats) * sizeof(float);
  if (b.size() < bytes) b.Alloc(bytes);
}

// 一层前向。x 就地更新（残差加在 x 上）—— 与 CPU 版 ForwardLayer 同序：
// 顺序错（比如 QK-norm 放到 RoPE 之后）不会崩，只会让注意力整体偏掉。
//
// pos0 / kv 的用法与 CPU 版完全对应：
//   * pos0 是**本段第一个位置在 cache 里的绝对下标**。RoPE 必须用它 ——
//     增量解码时位置要接续下去，每步都从 0 开始会让模型以为每次都回到句首
//     （症状：续写出来的东西像在开新话题）；
//   * kv != nullptr 时：先把本段的 K/V 写进 cache，再从 cache 读全部历史。
//     **写入必须在注意力之前** —— 位置 pos0+t 要能看到同段里排在它前面的
//     那些位置（s <= pos0 + t），而那些位置刚刚才写进去。
void GpuForwardLayer(const GpuModel& m, int32_t layer, int64_t pos0,
                     GpuScratch* s, int64_t seq, GpuKvCache* kv) {
  const ModelMeta& c = m.config();
  const GpuLayer& w = m.layers()[static_cast<size_t>(layer)];
  const int64_t hidden = c.hidden;
  const int64_t inter = c.intermediate;
  const int32_t hd = c.head_dim;
  const float* rope = m.rope_dev();

  float* x = F(s->x);
  float* h = F(s->h);
  float* q = F(s->q);
  float* k = F(s->k);
  float* v = F(s->v);
  float* attn = F(s->attn);
  float* proj = F(s->proj);
  float* gate = F(s->gate);
  float* up = F(s->up);

  // ---- 注意力 ----
  // pre-norm：归一化在子层**输入**上做，残差加在原始 x 上。
  GpuRmsNorm(x, CF(w.attn_norm.buf), seq, hidden, hidden, c.rms_eps, h);
  GpuMatMulQuant(w.attn_q, h, seq, q);
  GpuMatMulQuant(w.attn_k, h, seq, k);
  GpuMatMulQuant(w.attn_v, h, seq, v);

  // QK-norm：**在 RoPE 之前**，对每个头在 head_dim 上各做一次 RMSNorm。
  // 这里直接就地（out == x）—— 归一化先在 double 里把整行的平方和归约完，
  // 之后每个线程只写自己那几个元素，所以就地是安全的。
  GpuRmsNorm(q, CF(w.q_norm.buf), seq * c.n_heads, hd, hd, c.rms_eps, q);
  GpuRmsNorm(k, CF(w.k_norm.buf), seq * c.n_kv_heads, hd, hd, c.rms_eps, k);

  // RoPE。K 旋转、**V 不旋转** —— V 不参与位置编码。
  // pos0 传进去：本段第 t 个 token 的位置是 pos0 + t，不是 t。
  GpuRope(q, c.n_heads, hd, rope, pos0, seq);
  GpuRope(k, c.n_kv_heads, hd, rope, pos0, seq);

  if (kv != nullptr) {
    // 写 cache（fp32 → fp16），再从 cache 读历史。两条路径的差别只在这里：
    // 不带 cache 时 KV 直接来自本段的 s->k / s->v，算的是同一串数；带 cache 时
    // K/V 多过一次 fp16 舍入 ⇒ 这就是"带 cache 与不带 cache 不再逐位相同"的
    // 全部原因，也是 P2 自检的红线所在（相对误差该在 1e-3 量级，不该更大）。
    GpuKvWrite(k, v, c.n_kv_heads, hd, pos0, seq, kv->k_layer(layer),
               kv->v_layer(layer), kv->max_seq());
    GpuAttentionFromCache(c, pos0, q, seq, kv->k_layer(layer),
                          kv->v_layer(layer), kv->max_seq(), attn);
  } else {
    GpuCausalAttention(c, q, k, v, seq, attn);
  }

  GpuMatMulQuant(w.attn_o, attn, seq, proj);
  GpuVecAddInPlace(x, proj, seq * hidden);

  // ---- 前馈（SwiGLU）----
  GpuRmsNorm(x, CF(w.ffn_norm.buf), seq, hidden, hidden, c.rms_eps, h);
  GpuMatMulQuant(w.ffn_gate, h, seq, gate);
  GpuMatMulQuant(w.ffn_up, h, seq, up);
  GpuSiluMulInPlace(gate, up, seq * inter);
  GpuMatMulQuant(w.ffn_down, gate, seq, proj);
  GpuVecAddInPlace(x, proj, seq * hidden);
}

}  // namespace

// ---------------------------------------------------------------- GpuScratch

void GpuScratch::Ensure(const ModelMeta& c, int64_t seq) {
  Need(x, seq * c.hidden);
  Need(h, seq * c.hidden);
  Need(q, seq * c.q_dim());
  Need(k, seq * c.kv_dim());
  Need(v, seq * c.kv_dim());
  Need(attn, seq * c.q_dim());
  Need(proj, seq * c.hidden);
  Need(gate, seq * c.intermediate);
  Need(up, seq * c.intermediate);
  Need(logits, seq * c.vocab);
  // 采样的概率缓冲：**按 vocab 备，不乘 seq** —— 它一步只要一行（P3）。
  // ⚠️ 与上面那批的区别值得记一笔：那些缓冲的下标是"段内局部位置"，与 seq 同
  // 尺度；这块是"词表下标"，与 seq 无关。CPU 引擎当初那个"第二轮进程无声消失"
  // 的堆越界，根子就是"按一个尺度备、按另一个尺度索引"。这里两者都不缩容
  // （Need 只增不减），所以整类问题不可达。
  Need(probs, c.vocab);
  const size_t tok_bytes = static_cast<size_t>(seq) * sizeof(int32_t);
  if (toks.size() < tok_bytes) toks.Alloc(tok_bytes);
  // argmax 的落点。放在这里分配而不是在 Generate 里 —— 每步 cudaMalloc 一次
  // 4 字节是同步操作，而这块缓冲一次分配就永远够用。
  if (argmax_idx.size() < sizeof(int32_t)) argmax_idx.Alloc(sizeof(int32_t));
}

int64_t GpuScratch::bytes() const {
  int64_t n = 0;
  for (const DeviceBuffer* b : {&x, &h, &q, &k, &v, &attn, &proj, &gate, &up,
                                &toks, &logits, &probs, &argmax_idx}) {
    n += static_cast<int64_t>(b->size());
  }
  return n;
}

// ---------------------------------------------------------------- 前向

void GpuForwardFull(const GpuModel& m, const int32_t* tokens, int64_t seq,
                    GpuScratch* s, std::vector<float>* logits,
                    const GpuForwardOptions& opt, GpuKvCache* kv) {
  const ModelMeta& c = m.config();
  const int64_t hidden = c.hidden;

  RPG_CHECK(seq > 0, "prompt 为空");

  // 这一段在 cache 里的起始位置（不传 cache 就是 0）。与 CPU 版逐字对应。
  const int64_t pos0 = (kv != nullptr) ? kv->len() : 0;
  RPG_CHECK(pos0 + seq <= m.rope_positions(),
            "位置 %lld 超出 RoPE 表覆盖的 %lld（构造 GpuModel 时传更大的 max_pos，"
            "或先 session_reset）",
            static_cast<long long>(pos0 + seq),
            static_cast<long long>(m.rope_positions()));
  if (kv != nullptr) {
    RPG_CHECK(pos0 + seq <= kv->max_seq(), "位置 %lld 超出 KV Cache 容量 %lld",
              static_cast<long long>(pos0 + seq),
              static_cast<long long>(kv->max_seq()));
  }

  // token 越界必须在 **host 侧**挡：设备端读越界的 embedding 行不会崩，
  // 只会读到别人张量的字节 —— 然后"能出词但胡说"。
  for (int64_t t = 0; t < seq; ++t) {
    RPG_CHECK(tokens[t] >= 0 && tokens[t] < c.vocab,
              "第 %lld 个 token id = %d 越界（词表 %d）", static_cast<long long>(t),
              tokens[t], c.vocab);
  }

  const int32_t n_use = (opt.layer_limit > 0 && opt.layer_limit < c.n_layers)
                            ? opt.layer_limit
                            : c.n_layers;
  if (opt.capture_hidden) {
    opt.capture_hidden->assign(static_cast<size_t>(n_use + 1) * seq * hidden, 0.0f);
  }

  s->Ensure(c, seq);
  s->toks.Upload(tokens, static_cast<size_t>(seq) * sizeof(int32_t));

  // ---- embedding ----
  // Qwen3 **不**乘 sqrt(hidden)（Gemma 2 才乘）。抄错的表现是整体 logits 缩放、
  // softmax 后温度不对 —— 能出词，但不对。
  GpuEmbeddingLookup(m.token_embd(), static_cast<const int32_t*>(s->toks.data()),
                     seq, F(s->x));
  if (opt.capture_hidden) {
    s->x.Download(opt.capture_hidden->data(),
                  static_cast<size_t>(seq) * hidden * sizeof(float));
  }

  // ---- 逐层 ----
  for (int32_t l = 0; l < n_use; ++l) {
    GpuForwardLayer(m, l, pos0, s, seq, kv);
    if (opt.capture_hidden) {
      s->x.Download(opt.capture_hidden->data() +
                        static_cast<size_t>(l + 1) * seq * hidden,
                    static_cast<size_t>(seq) * hidden * sizeof(float));
    }
  }

  // cache 的长度在**所有层都写完**之后才推进：中途任何一层抛错（越界、未知
  // 量化类型……）都保留原来的 len，调用方重试时不会多算一段空洞。
  // 与 CPU 版同一条，位置也一样（在逐层循环之后）。
  if (kv != nullptr) kv->SetLen(pos0 + seq);

  if (n_use < c.n_layers) {
    // 截断模式：末端归一化与 lm_head 依赖更深的层，跳过并返回空 logits，
    // 而不是给一个"看起来能用"的错值（与 CPU 版同约定）。
    if (logits != nullptr) logits->clear();
    return;
  }

  // ---- 末端归一化 + 输出头 ----
  // ⚠️ 无论 logits 是不是 nullptr 都**必须**算到这里：增量解码要的就是这段
  // 结果留在显存里。把 nullptr 的早退写在前面（让 lm_head 也一起跳过）会让
  // decode 循环里 `s->logits` 一直是上一轮的垃圾 —— 而且不会崩，只会让生成
  // 的 token 从第二个开始就是错的。**这个顺序是刻意的，别动。**
  GpuRmsNorm(F(s->x), CF(m.output_norm().buf), seq, hidden, hidden, c.rms_eps,
             F(s->h));
  GpuMatMulQuant(m.lm_head(), CF(s->h), seq, F(s->logits));

  if (logits == nullptr) {
    // 调用方只要显存里的结果（增量解码走这里：它只需要下一步的 argmax，
    // 把 607 KB 的 logits 拷回 host 再丢掉是纯浪费）。
    return;
  }

  logits->assign(static_cast<size_t>(seq) * c.vocab, 0.0f);
  s->logits.Download(logits->data(), logits->size() * sizeof(float));
}

void GpuForwardStep(const GpuModel& m, int32_t token, int64_t pos, GpuKvCache* kv,
                    GpuScratch* s, std::vector<float>* logits) {
  RPG_CHECK(kv != nullptr, "GpuForwardStep 需要 KV Cache");
  RPG_CHECK(pos == kv->len(),
            "增量解码只支持顺序追加：pos = %lld，而 cache 长度 = %lld",
            static_cast<long long>(pos), static_cast<long long>(kv->len()));
  // 与 CPU 版一样只有这一行 —— 增量解码不是"另一份实现"，就是整段前向在
  // seq = 1、pos0 = cache.len() 上的特例。
  GpuForwardFull(m, &token, 1, s, logits, GpuForwardOptions(), kv);
}

// ---------------------------------------------------------------- 生成

void GpuGenerateGreedy(const GpuModel& m, GpuKvCache* kv, GpuScratch* s,
                       const float* first_logits_dev,
                       const GpuGenerateOptions& opt,
                       std::vector<int32_t>* out_ids) {
  const ModelMeta& c = m.config();
  RPG_CHECK(kv != nullptr, "GpuGenerateGreedy 需要 KV Cache");
  RPG_CHECK(first_logits_dev != nullptr, "first_logits 为空（prefill 之后才有）");
  RPG_CHECK(opt.max_tokens > 0, "max_tokens 必须为正，给了 %d", opt.max_tokens);
  RPG_CHECK(!s->argmax_idx.empty(),
            "GpuScratch 还没 Ensure 过（先跑一次 prefill）");
  int32_t* idx_dev = static_cast<int32_t*>(s->argmax_idx.data());

  // 每一步的 logits 都在同一个缓冲里演进：prefill 的最后一行进来，之后每次
  // GpuForwardStep 覆盖 s->logits 的第 0 行。于是**整个 decode 循环里没有一次
  // 607 KB 的显存→内存搬运**，只有 argmax 的 4 个字节。
  const float* cur = first_logits_dev;

  for (int32_t i = 0; i < opt.max_tokens; ++i) {
    // 贪心 = argmax。并列取最小下标（与 CPU 版 ArgMax 同语义）——
    // 差一个并列规则就会让"逐 token 一致"的判据红。
    GpuArgMax(cur, c.vocab, idx_dev);
    int32_t next = 0;
    RPG_CUDA_CHECK(
        cudaMemcpy(&next, idx_dev, sizeof(int32_t), cudaMemcpyDeviceToHost));
    out_ids->push_back(next);

    // 对拍工具要的中间量（默认不做）。收集的是**这一步做决策时看到的分布**，
    // 也就是"第一个返回的 token 来自 prefill 的最后一个位置"这个语义 ——
    // 顺序上必须在 ForwardStep 之前，否则收到的是下一步的。
    if (opt.step_logits != nullptr) {
      const size_t base = opt.step_logits->size();
      opt.step_logits->resize(base + static_cast<size_t>(c.vocab));
      RPG_CUDA_CHECK(cudaMemcpy(opt.step_logits->data() + base, cur,
                                static_cast<size_t>(c.vocab) * sizeof(float),
                                cudaMemcpyDeviceToHost));
    }

    bool stop = false;
    for (const int32_t sid : opt.stop_ids) {
      if (sid == next) {
        stop = true;
        break;
      }
    }

    // 上下文满了就明确报错，不静默截断 —— 静默停下会让人以为"模型自己
    // 讲完了"，而实际是宿主把 max_seq 配小了。
    RPG_CHECK(kv->len() < kv->max_seq(),
              "上下文已满（%lld 个位置，已生成 %d 个 token）—— 需要更大的 "
              "max_seq，或先 session_reset",
              static_cast<long long>(kv->max_seq()), i + 1);

    // **每个**返回的 token 都写进 cache —— 包括最后那个与停止 token。
    // 代价是末尾多算一步前向，换来的是这条不变量：
    //   cache.len() == prompt 长度 + 返回的 token 数
    // 它让调用方可以直接在 cache 上续下一轮（不必 reset 重来），并且可以被
    // 外部一句话验证。少了这一步，cache 会永远差最后一个 token —— 那种
    // "短一个"的错误不会崩，只会让下一轮的第一个 token 悄悄变味。
    GpuForwardStep(m, next, kv->len(), kv, s, /*logits=*/nullptr);
    cur = CF(s->logits);
    if (stop) break;
  }
}

}  // namespace rpg

#endif  // RPG_HAS_CUDA
