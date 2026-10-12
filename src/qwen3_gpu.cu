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
void GpuForwardLayer(const GpuModel& m, int32_t layer, GpuScratch* s,
                     int64_t seq) {
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
  GpuRope(q, c.n_heads, hd, rope, 0, seq);
  GpuRope(k, c.n_kv_heads, hd, rope, 0, seq);

  GpuCausalAttention(c, q, k, v, seq, attn);
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
  const size_t tok_bytes = static_cast<size_t>(seq) * sizeof(int32_t);
  if (toks.size() < tok_bytes) toks.Alloc(tok_bytes);
}

int64_t GpuScratch::bytes() const {
  int64_t n = 0;
  for (const DeviceBuffer* b : {&x, &h, &q, &k, &v, &attn, &proj, &gate, &up,
                                &toks, &logits}) {
    n += static_cast<int64_t>(b->size());
  }
  return n;
}

// ---------------------------------------------------------------- 前向

void GpuForwardFull(const GpuModel& m, const int32_t* tokens, int64_t seq,
                    GpuScratch* s, std::vector<float>* logits,
                    const GpuForwardOptions& opt) {
  const ModelMeta& c = m.config();
  const int64_t hidden = c.hidden;

  RPG_CHECK(seq > 0, "prompt 为空");
  RPG_CHECK(seq <= m.rope_positions(),
            "位置 %lld 超出 RoPE 表覆盖的 %lld（构造 GpuModel 时传更大的 max_pos）",
            static_cast<long long>(seq),
            static_cast<long long>(m.rope_positions()));

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
    GpuForwardLayer(m, l, s, seq);
    if (opt.capture_hidden) {
      s->x.Download(opt.capture_hidden->data() +
                        static_cast<size_t>(l + 1) * seq * hidden,
                    static_cast<size_t>(seq) * hidden * sizeof(float));
    }
  }

  if (n_use < c.n_layers) {
    // 截断模式：末端归一化与 lm_head 依赖更深的层，跳过并返回空 logits，
    // 而不是给一个"看起来能用"的错值（与 CPU 版同约定）。
    logits->clear();
    return;
  }

  // ---- 末端归一化 + 输出头 ----
  GpuRmsNorm(F(s->x), CF(m.output_norm().buf), seq, hidden, hidden, c.rms_eps,
             F(s->h));
  GpuMatMulQuant(m.lm_head(), CF(s->h), seq, F(s->logits));
  logits->assign(static_cast<size_t>(seq) * c.vocab, 0.0f);
  s->logits.Download(logits->data(), logits->size() * sizeof(float));
}

}  // namespace rpg

#endif  // RPG_HAS_CUDA
