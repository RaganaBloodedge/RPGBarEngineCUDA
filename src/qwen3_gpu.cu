// qwen3_gpu.cu —— Qwen3 前向的实现（见 qwen3_gpu.h 的说明）

#include "qwen3_gpu.h"

#ifdef RPG_HAS_CUDA

#include <bit>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "common.h"
#include "kernels_gpu.h"
#include "cuda_common.h"
#include "sampler_gpu.h"

namespace rpg {

namespace {

float* F(DeviceBuffer& b) { return static_cast<float*>(b.data()); }
const float* CF(const DeviceBuffer& b) {
  return static_cast<const float*>(b.data());
}

// 只增不减（理由见 GpuScratch 的注释）。
void Need(DeviceBuffer& b, int64_t n_floats) {
  RPG_CHECK(n_floats > 0, "buffer size must be positive");
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
// `attn_sh_positions` 透传给 GpuAttentionFromCache：它决定 scores 的共享内存按
// 多少个位置备。**图路径必须给 kv->max_seq()** —— 那个大小是启动参数、会被录进
// 图，而回放时 pos0 每步递增（照捕获那一刻的 pos0 备，第 2 步起就写越界了）。
void GpuForwardLayer(const GpuModel& m, int32_t layer, int64_t pos0,
                     GpuScratch* s, int64_t seq, GpuKvCache* kv,
                     int64_t attn_sh_positions) {
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
                          kv->v_layer(layer), kv->max_seq(), attn,
                          attn_sh_positions);
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

// 前向的主体：**token 已经在显存里**（`s->toks` 的前 seq 个），既没有 host 侧的
// 校验、也没有 H2D 上传、也不推进 cache 的 len。
//
// 为什么拆出来：CUDA Graph 路径要求"从 token 到 logits"这一段里**一次 host 调用
// 都没有**，而 GpuForwardFull 里正好有三件进不了图的事（校验 host 数组、Upload、
// kv->SetLen）。拆开之后两条路径共用同一段前向代码 —— 与"一个 GpuGenerate + 一个
// 薄包装"是同一种做法，防的是两份实现漂移（今天只差 Upload 一行，明天就是
// "哪个分支忘了处理 layer_limit"）。
void ForwardCore(const GpuModel& m, int64_t seq, GpuScratch* s,
                 std::vector<float>* logits, const GpuForwardOptions& opt,
                 GpuKvCache* kv, int64_t attn_sh_positions) {
  const ModelMeta& c = m.config();
  const int64_t hidden = c.hidden;
  // 这一段在 cache 里的起始位置（不传 cache 就是 0）。与 CPU 版逐字对应。
  const int64_t pos0 = (kv != nullptr) ? kv->len() : 0;

  const int32_t n_use = (opt.layer_limit > 0 && opt.layer_limit < c.n_layers)
                            ? opt.layer_limit
                            : c.n_layers;
  if (opt.capture_hidden) {
    opt.capture_hidden->assign(static_cast<size_t>(n_use + 1) * seq * hidden, 0.0f);
  }

  s->Ensure(c, seq);

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
    GpuForwardLayer(m, l, pos0, s, seq, kv, attn_sh_positions);
    if (opt.capture_hidden) {
      s->x.Download(opt.capture_hidden->data() +
                        static_cast<size_t>(l + 1) * seq * hidden,
                    static_cast<size_t>(seq) * hidden * sizeof(float));
    }
  }

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

GpuScratch::~GpuScratch() {
  // 图只在**会话释放**时销毁，这是对的：图里含这个会话的 KV 层基址与 scratch
  // 地址，换一个会话就是另一张图。失败只记日志 —— 与 DeviceBuffer::Free 同一条
  // 理由（析构不能抛；实践中只在上下文已销毁时失败，DESIGN §9.6）。
  if (graph_exec != nullptr) {
    const cudaError_t e = cudaGraphExecDestroy(graph_exec);
    if (e != cudaSuccess) {
      RPG_LOG_WARN(
          "cudaGraphExecDestroy failed (the context may already be destroyed): %s",
          cudaGetErrorString(e));
    }
    graph_exec = nullptr;
  }
}

// ---------------------------------------------------------------- 前向

void GpuForwardFull(const GpuModel& m, const int32_t* tokens, int64_t seq,
                    GpuScratch* s, std::vector<float>* logits,
                    const GpuForwardOptions& opt, GpuKvCache* kv) {
  const ModelMeta& c = m.config();

  RPG_CHECK(seq > 0, "prompt is empty");

  // 这一段在 cache 里的起始位置（不传 cache 就是 0）。与 CPU 版逐字对应。
  const int64_t pos0 = (kv != nullptr) ? kv->len() : 0;
  RPG_CHECK(pos0 + seq <= m.rope_positions(),
            "position %lld is beyond the %lld positions the RoPE table covers (pass a larger max_pos when constructing GpuModel, "
            "or call session_reset first)",
            static_cast<long long>(pos0 + seq),
            static_cast<long long>(m.rope_positions()));
  if (kv != nullptr) {
    RPG_CHECK(pos0 + seq <= kv->max_seq(), "position %lld exceeds the KV cache capacity %lld",
              static_cast<long long>(pos0 + seq),
              static_cast<long long>(kv->max_seq()));
  }

  // token 越界必须在 **host 侧**挡：设备端读越界的 embedding 行不会崩，
  // 只会读到别人张量的字节 —— 然后"能出词但胡说"。
  for (int64_t t = 0; t < seq; ++t) {
    RPG_CHECK(tokens[t] >= 0 && tokens[t] < c.vocab,
              "token #%lld has id = %d, out of range (vocab %d)", static_cast<long long>(t),
              tokens[t], c.vocab);
  }

  // ⚠️ 上传之前必须先 Ensure —— Upload 要求缓冲区已经分配。
  s->Ensure(c, seq);
  s->toks.Upload(tokens, static_cast<size_t>(seq) * sizeof(int32_t));

  // attn_sh_positions = -1 ⇒ scores 按 pos0 + seq 备（P2 起的口径，逐字不变）。
  // 图路径不能走这里 —— 它的共享内存预算要按 max_seq 给，见 ForwardCore 的注释。
  ForwardCore(m, seq, s, logits, opt, kv, /*attn_sh_positions=*/-1);

  // cache 的长度在**所有层都写完**之后才推进：中途任何一层抛错（越界、未知
  // 量化类型……）都保留原来的 len，调用方重试时不会多算一段空洞。
  // 与 CPU 版同一条，位置也一样（在逐层循环之后）。
  if (kv != nullptr) kv->SetLen(pos0 + seq);
}

void GpuForwardStep(const GpuModel& m, int32_t token, int64_t pos, GpuKvCache* kv,
                    GpuScratch* s, std::vector<float>* logits) {
  RPG_CHECK(kv != nullptr, "GpuForwardStep requires a KV cache");
  RPG_CHECK(pos == kv->len(),
            "incremental decoding only appends in order: pos = %lld but the cache length is %lld",
            static_cast<long long>(pos), static_cast<long long>(kv->len()));
  // 与 CPU 版一样只有这一行 —— 增量解码不是"另一份实现"，就是整段前向在
  // seq = 1、pos0 = cache.len() 上的特例。
  GpuForwardFull(m, &token, 1, s, logits, GpuForwardOptions(), kv);
}

void GpuForwardStepDev(const GpuModel& m, const int32_t* token_dev, int64_t pos,
                       GpuKvCache* kv, GpuScratch* s) {
  RPG_CHECK(kv != nullptr, "GpuForwardStepDev requires a KV cache");
  RPG_CHECK(token_dev != nullptr, "GpuForwardStepDev requires a device-side token");
  RPG_CHECK(pos == kv->len(),
            "incremental decoding only appends in order: pos = %lld but the cache length is %lld",
            static_cast<long long>(pos), static_cast<long long>(kv->len()));
  // 与 GpuForwardFull 里同一组检查。区别是 token 的**值**校验不了 —— 它在显存里
  // （那正是它能进 CUDA Graph 的原因），而设备端读越界的 embedding 行不会崩、
  // 只会读到别人张量的字节。位置与容量这两条仍然要在 host 挡。
  RPG_CHECK(pos + 1 <= m.rope_positions(),
            "position %lld is beyond the %lld positions the RoPE table covers",
            static_cast<long long>(pos + 1),
            static_cast<long long>(m.rope_positions()));
  RPG_CHECK(pos + 1 <= kv->max_seq(), "position %lld exceeds the KV cache capacity %lld",
            static_cast<long long>(pos + 1),
            static_cast<long long>(kv->max_seq()));

  const ModelMeta& c = m.config();
  s->Ensure(c, 1);
  // token 进 s->toks：**设备到设备**的 4 字节拷贝，可以被 CUDA Graph 捕获
  //（H2D 不行 —— 那需要一个 host 源地址，回放时早就失效了）。
  RPG_CUDA_CHECK(cudaMemcpyAsync(s->toks.data(), token_dev, sizeof(int32_t),
                                 cudaMemcpyDeviceToDevice, EngineStream()));
  // attn_sh_positions 给 max_seq：这条路径是**为图准备的**（见 ForwardCore 的注释），
  // 共享内存要一次备够。不这么做就只能按捕获那一刻的 pos0 备，回放时写越界。
  ForwardCore(m, 1, s, /*logits=*/nullptr, GpuForwardOptions(), kv,
              /*attn_sh_positions=*/kv->max_seq());
  kv->SetLen(pos + 1);
}

// ---------------------------------------------------------------- 生成

namespace {

// 图的"形状指纹"。图里录进了这些量：模型的各个指针（权重、RoPE 表）、scratch 与
// KV Cache 的地址与容量、以及采样参数和 seed（它们是采样 kernel 的参数）。
// 任何一个变了，旧图就是错的 —— 而**回放不会报错**，它只会拿旧地址/旧参数算出
// 一段看着合理的结果。所以宁可多录一张图，也不要复用一张对不上的。
//
// 判据里"换 seed 会变"那条靠的就是 seed 进了 key。
uint64_t MakeGraphKey(const GpuModel& m, GpuKvCache* kv, GpuScratch* s,
                      const GpuGenerateOptions& opt) {
  const ModelMeta& c = m.config();
  uint64_t h = 1469598103934665603ull;  // FNV-1a 偏移基数
  auto mix = [&h](uint64_t v) {
    h ^= v;
    h *= 1099511628211ull;
  };
  auto mixf = [&mix](float v) { mix(static_cast<uint64_t>(std::bit_cast<uint32_t>(v))); };

  mix(reinterpret_cast<uintptr_t>(&m));
  mix(reinterpret_cast<uintptr_t>(kv));
  mix(reinterpret_cast<uintptr_t>(s));
  mix(static_cast<uint64_t>(kv->max_seq()));
  mix(static_cast<uint64_t>(c.n_layers));
  mix(static_cast<uint64_t>(c.hidden));
  mix(static_cast<uint64_t>(c.vocab));
  mix(static_cast<uint64_t>(c.n_heads));
  mix(static_cast<uint64_t>(c.n_kv_heads));
  mix(static_cast<uint64_t>(c.head_dim));
  mixf(opt.sample.temperature);
  mix(static_cast<uint64_t>(opt.sample.top_k));
  mixf(opt.sample.top_p);
  mix(opt.seed);
  return h;
}

// 录一张"单步 decode"的图：
//    采样当前 logits → 选中的 token 写回 toks → 前向（含 embedding）→ 新 logits
//
// 与老循环的一轮**完全同序**（老循环也是"先采样 cur、再前向 next"）。图不是
// "另一份实现"，而是同一段计算被录下来 —— 这正是"图的输出必须与老路径逐位相同"
// 这条判据能成立的原因：两边跑的是同一批 kernel、同一组参数。
//
// 进得了图的：所有 kernel、D2D 拷贝、地址固定的显存。
// 进不了的：H2D/D2H（源或目标是 host 地址）、cudaMalloc、任何同步调用。
//
// 返回 nullptr 表示"录不出来"（专用流建不出来、或实例化失败）—— 调用方据此
// **退化成老路径**，而不是崩。
cudaGraphExec_t CaptureDecodeStep(const GpuModel& m, GpuKvCache* kv,
                                  GpuScratch* s, const GpuGenerateOptions& opt,
                                  const float* cur_logits, float* probs,
                                  int32_t* idx_dev) {
  const cudaStream_t cap = CaptureStream();
  if (cap == nullptr) return nullptr;
  const ModelMeta& c = m.config();

  // ---- 捕获之前先把"惰性分配"做完 ----
  // 捕获期间 cudaMalloc 是非法操作，而这些缓冲区可能到"第一次真正用到"才分配。
  // 少了这一步，症状是 cudaStreamEndCapture 返回一个看不出根因的
  // cudaErrorStreamCaptureUnsupported。
  GpuSamplerWarmup();        // u 的落点
  (void)PosStagingPtr();     // pos0 / counter 的落点
  s->Ensure(c, /*seq=*/1);   // 所有前向缓冲
  // attention 的 scores 共享内存要按 max_seq 备（见 ForwardCore 的注释），而
  // "抬 48 KiB 上限"只能在这一刻做 —— 捕获期间 cudaFuncSetAttribute 非法。
  // 不超过上限时它是个空操作，所以默认配置下不多付任何代价。
  GpuAttentionWarmupShmem(kv->max_seq());

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  {
    ScopedEngineStream guard(cap);

    // 「这台机器 / 这个驱动不支持流捕获」⇒ 返回 nullptr，调用方退回老路径
    //（图是优化，不是正确性的前提）。
    //
    // ⚠️ **只有这一步的失败可以被这样吞掉**。捕获窗口**里面**的失败一定是有人
    // 往里放了非法操作（cudaMalloc / H2D / 同步调用），那是 bug —— 必须带着
    // 清楚的信息抛出去。静默退回老路径的后果是"图从没跑过"变成一条看不出来的
    // 绿：判据会比较"老路径 vs 老路径"然后全过。所以 tools/graph_check 还要
    // 额外断言 graph_exec != nullptr，两道防线都要有。
    const cudaError_t be =
        cudaStreamBeginCapture(cap, cudaStreamCaptureModeThreadLocal);
    if (be != cudaSuccess) {
      RPG_LOG_WARN(
          "cudaStreamBeginCapture is not usable here (%s); the graph path will "
          "fall back to per-step launches",
          cudaGetErrorString(be));
      return nullptr;
    }

    try {
      // (a) 采样：cur logits → idx。counter 从 pos0 暂存区读 —— 它与这一步
      //     前向的 pos0 是同一个数（这个 token 将要落进 cache 的下标）；
      //     u 由设备端从 counter 算出，全程不碰 host。
      GpuSampleTokenDev(cur_logits, c.vocab, opt.sample, opt.seed,
                        reinterpret_cast<const uint64_t*>(PosStagingPtr()), probs,
                        idx_dev);
      // (b) 把选中的 token 写回 toks。**D2D** 4 字节 —— 可捕获；换成 H2D 就
      //     得在回放时读一个早已失效的 host 地址。
      RPG_CUDA_CHECK(cudaMemcpyAsync(s->toks.data(), idx_dev, sizeof(int32_t),
                                     cudaMemcpyDeviceToDevice, EngineStream()));
      // (c) 前向。attn_sh_positions 给 **max_seq**：共享内存大小是启动参数、
      //     会被录进图，而回放时 pos0 每步递增（见 ForwardCore 的注释）。
      ForwardCore(m, 1, s, /*logits=*/nullptr, GpuForwardOptions(), kv,
                  /*attn_sh_positions=*/kv->max_seq());

      RPG_CUDA_CHECK(cudaStreamEndCapture(cap, &graph));
    } catch (...) {
      // 捕获中途失败必须**把这次捕获作废**，否则引擎流会一直停在捕获状态 ——
      // 之后每一次 cudaMemcpy / 同步都报 cudaErrorStreamCaptureUnsupported，
      // 而那个症状离这里很远。
      cudaStreamEndCapture(cap, &graph);
      if (graph != nullptr) cudaGraphDestroy(graph);
      throw;
    }
  }

  if (graph == nullptr) return nullptr;
  const cudaError_t ie = cudaGraphInstantiateWithFlags(&exec, graph, 0);
  cudaGraphDestroy(graph);
  if (ie != cudaSuccess) return nullptr;
  return exec;
}

// 图路径的生成循环。与老循环的差别**只在每步的 host 调用次数**：
//     老：~400 次 kernel 启动（每次 launch 都是一次驱动往返）+ 一次 D2H 同步
//     新：1 次 cudaGraphLaunch + 一次 D2H 同步
// 计算内容一字不差 —— 同一批 kernel、同一组参数、同一个顺序。
// 返回 false 表示"图没跑成"（录不出来），调用方应退回老路径。
bool GenerateGraph(const GpuModel& m, GpuKvCache* kv, GpuScratch* s,
                   const GpuGenerateOptions& opt, const float* first_logits_dev,
                   float* probs, int32_t* idx_dev,
                   std::vector<int32_t>* out_ids) {
  const ModelMeta& c = m.config();
  const cudaStream_t cap = CaptureStream();
  RPG_CHECK(cap != nullptr,
            "the CUDA graph path was requested but no capture stream is available");

  // 图里每一步都采 `s->logits` 的第 0 行，所以先把 prefill 的**最后一行**搬过来
  // ——老循环的第一步采样的正是那一行。搬完就统一了："每步都采 s->logits 的第 0 行"。
  // 只在捕获之前做一次：之后每次前向自己会覆盖第 0 行。
  {
    ScopedEngineStream guard(cap);
    if (first_logits_dev != CF(s->logits)) {
      RPG_CUDA_CHECK(cudaMemcpyAsync(s->logits.data(), first_logits_dev,
                                     static_cast<size_t>(c.vocab) * sizeof(float),
                                     cudaMemcpyDeviceToDevice, cap));
    }
  }

  // 需要（或需要重录）就录一张。key 不符 = 形状/参数变了。
  const uint64_t key = MakeGraphKey(m, kv, s, opt);
  if (s->graph_exec == nullptr || s->graph_key != key) {
    if (s->graph_exec != nullptr) {
      cudaGraphExecDestroy(s->graph_exec);
      s->graph_exec = nullptr;
      s->graph_key = 0;
    }
    s->graph_exec =
        CaptureDecodeStep(m, kv, s, opt, CF(s->logits), probs, idx_dev);
    if (s->graph_exec != nullptr) s->graph_key = key;
  }
  if (s->graph_exec == nullptr) {
    // 录不出来（比如驱动不支持流捕获）⇒ **退回老路径**，而不是崩。
    // 刻意不在这种时候抛异常：图是优化，不是正确性的前提。
    RPG_LOG_WARN(
        "CUDA graph capture is unavailable here; falling back to per-step launches");
    return false;
  }

  ScopedEngineStream guard(cap);
  for (int32_t i = 0; i < opt.max_tokens; ++i) {
    const int64_t pos = kv->len();
    // 位置的两条上界都要在 host 挡：图里读不了这些条件（也读不了 rope 表的
    // 长度），而越界的 RoPE 读不会崩，只会悄悄算错。
    RPG_CHECK(pos < kv->max_seq(),
              "context is full (%lld positions, %d tokens generated); a larger "
              "max_seq is needed, or call session_reset first",
              static_cast<long long>(kv->max_seq()), i + 1);
    RPG_CHECK(pos + 1 <= m.rope_positions(),
              "position %lld is beyond the %lld positions the RoPE table covers",
              static_cast<long long>(pos + 1),
              static_cast<long long>(m.rope_positions()));

    // 每步只有这一次 host 写入（8 字节），图的 kernel 从这块显存里读位置。
    StagePos0(pos);
    RPG_CUDA_CHECK(cudaGraphLaunch(s->graph_exec, cap));
    int32_t next = 0;
    RPG_CUDA_CHECK(cudaMemcpyAsync(&next, idx_dev, sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, cap));
    // 同步一次：拿 token + 让下一步能安全地覆盖 pos0 暂存区。
    RPG_CUDA_CHECK(cudaStreamSynchronize(cap));
    out_ids->push_back(next);
    // 与老路径同一条不变量：**每个**返回的 token 都写进 cache（含停止 token），
    // 于是 cache.len() == prompt 长度 + 返回的 token 数。
    kv->SetLen(pos + 1);

    bool stop = false;
    for (const int32_t sid : opt.stop_ids) {
      if (sid == next) {
        stop = true;
        break;
      }
    }
    if (stop) break;
  }
  return true;
}

// 老路径：每步一次 launch。P0–P3 的行为**逐字不变**，也是图路径录不出来时的退路。
void GenerateNoGraph(const GpuModel& m, GpuKvCache* kv, GpuScratch* s,
                     const GpuGenerateOptions& opt, const float* first_logits_dev,
                     float* probs, int32_t* idx_dev,
                     std::vector<int32_t>* out_ids) {
  const ModelMeta& c = m.config();

  // 每一步的 logits 都在同一个缓冲里演进：prefill 的最后一行进来，之后每次
  // GpuForwardStep 覆盖 s->logits 的第 0 行。于是**整个 decode 循环里没有一次
  // 607 KB 的显存→内存搬运**，只有采样结果的 4 个字节。
  const float* cur = first_logits_dev;

  for (int32_t i = 0; i < opt.max_tokens; ++i) {
    // 绝对位置 = 这个 token 将要落进 cache 的下标（**此刻还没写进去**），
    // 与 CPU 版 Generate 逐字一致。用绝对下标而不是"本轮第几个"：宿主每轮
    // 重发完整 messages 时同一段前缀的位置不变，于是暖缓存（续写）与冷缓存
    // （从头）会拿到同一个 u —— 那正是 counter-based 随机源的用处。
    //
    // ⚠️ 只有一个调用点：`temperature <= 0` 时 GpuSampleToken 内部直接走
    // `GpuArgMax`（并列取最小下标）—— 与 P2 的贪心是**同一个 kernel、
    // 同一条路径**，所以"贪心的结果逐位不变"是由构造保证的，不是靠测出来的。
    const uint64_t counter = static_cast<uint64_t>(kv->len());
    GpuSampleToken(cur, c.vocab, opt.sample, opt.seed, counter, probs, idx_dev);
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
              "context is full (%lld positions, %d tokens generated); a larger "
              "max_seq is needed, or call session_reset first",
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

}  // namespace

void GpuGenerate(const GpuModel& m, GpuKvCache* kv, GpuScratch* s,
                 const float* first_logits_dev, const GpuGenerateOptions& opt,
                 std::vector<int32_t>* out_ids) {
  const ModelMeta& c = m.config();
  RPG_CHECK(kv != nullptr, "GpuGenerate requires a KV cache");
  RPG_CHECK(first_logits_dev != nullptr, "first_logits is null (only available after a prefill)");
  RPG_CHECK(opt.max_tokens > 0, "max_tokens must be positive, got %d", opt.max_tokens);
  RPG_CHECK(!s->argmax_idx.empty(),
            "GpuScratch has not been Ensure()d yet (run a prefill first)");
  // 贪心不消耗随机数、也不需要概率缓冲（GpuSampleToken 的那条分支只调
  // GpuArgMax）。于是"老路径"连 probs 都不必存在 —— 这一点是刻意的：
  // 让贪心路径对 scratch 的要求与 P2 完全一致，就少一个可能红的地方。
  const bool greedy = !(opt.sample.temperature > 0.0f);
  if (!greedy) {
    RPG_CHECK(s->probs.size() >= static_cast<size_t>(c.vocab) * sizeof(float),
              "the sampling probability buffer is not ready ([%d] floats); run a prefill first",
              c.vocab);
  }
  float* probs = static_cast<float*>(s->probs.data());
  int32_t* idx_dev = static_cast<int32_t*>(s->argmax_idx.data());

  // 图路径与 step_logits 互斥：后者要逐步把 607 KB 的 logits 拷回 host，而图里
  // 既不能有 D2H、也不能按步改图。同时打开时**按老路径走并说清楚** —— 而不是
  // 悄悄给出一份没有 step_logits 的结果（对拍工具会以为自己在对拍）。
  if (opt.use_graph && opt.step_logits != nullptr) {
    RPG_LOG_WARN(
        "use_graph is ignored: step_logits needs a per-step download, which a CUDA "
        "graph cannot do");
  }
  const bool use_graph = opt.use_graph && opt.step_logits == nullptr;

  if (use_graph) {
    // 录不出图就**退回老路径**（图是优化，不是正确性的前提）。判据里那条
    // "两条路径逐位相同"因此永远不会因为"图没开成"而变成假绿 —— 工具会看到
    // out_ids 与老路径一致，但 graph_exec 是空的。
    if (!GenerateGraph(m, kv, s, opt, first_logits_dev, probs, idx_dev, out_ids)) {
      GenerateNoGraph(m, kv, s, opt, first_logits_dev, probs, idx_dev, out_ids);
    }
  } else {
    GenerateNoGraph(m, kv, s, opt, first_logits_dev, probs, idx_dev, out_ids);
  }
}

void GpuGenerateGreedy(const GpuModel& m, GpuKvCache* kv, GpuScratch* s,
                       const float* first_logits_dev,
                       const GpuGenerateOptions& opt,
                       std::vector<int32_t>* out_ids) {
  // 与 CPU 版 GenerateGreedy 同构：**拷一份再改**，不动调用方传进来的 opt。
  // （那边拷的理由是"参数里还有 stop_ids 之类要原样带过去"；这里一样。）
  GpuGenerateOptions g = opt;
  g.sample.temperature = 0.0f;  // 贪心：与 P2 的行为逐位一致
  // top_k / top_p 留着不管 —— 贪心分支不看它们（SampleToken 直接 ArgMax）。
  // 刻意不在这里"顺手清干净"：清掉会让人以为"top_k 在贪心时也要被尊重"，
  // 而 CPU 版同样不清。
  GpuGenerate(m, kv, s, first_logits_dev, g, out_ids);
}

}  // namespace rpg

#endif  // RPG_HAS_CUDA
