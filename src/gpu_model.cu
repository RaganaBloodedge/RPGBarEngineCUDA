// gpu_model.cu —— 把 GGUF 的权重搬到显存（见 gpu_model.h 里的设计说明）。

#include "gpu_model.h"

#ifdef RPG_HAS_CUDA

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "common.h"
#include "quant.h"

namespace rpg {

namespace {

// GPU 侧实现不了的类型在**加载期**就拒。
//
// 为什么不"遇到不认识的类型就退回标量"：融合点积里没有"标量兜底"这条路
// （kernel 里只有一套展开），退化的结果会是拿 0 当权重 —— 能出词，但胡说。
// 一份 Q4_K_M 权重里只有 F32 / Q4_K / Q6_K 三种，别的类型出现说明换权重了，
// 那就该停在这里，而不是在 28 层之后算出 NaN。
void CheckGpuType(const GgufTensor& t) {
  switch (t.type) {
    case kGgmlF32:
    case kGgmlQ4_K:
    case kGgmlQ6_K:
      return;
    default:
      Throw("tensor %s has type %s; the CUDA backend implements only F32 / Q4_K / Q6_K so far. "
            "Use a quantization it does support (the Q4_K+Q6_K pair used by Q4_K_M / Q5_K_M), "
            "or wait for this type to be implemented in a later stage.",
            t.name.c_str(), TypeName(t.type));
  }
}

}  // namespace

// ---------------------------------------------------------------- 上传

GpuTensor GpuModel::UploadMatrix(const GgufTensor& t) {
  CheckGpuType(t);
  GpuTensor g;
  g.type = t.type;
  g.matrix = true;
  g.rows = t.rows();
  g.cols = t.cols();
  g.n_elements = t.n_elements;
  g.buf.Alloc(static_cast<size_t>(t.n_bytes));
  // **原始字节直传**，不反量化（见 gpu_model.h）。量化块是自包含的：
  // 每个块自带 scale/min，所以按字节搬过去之后 kernel 能独立解任意一个块。
  g.buf.Upload(gguf_.TensorBytes(t), static_cast<size_t>(t.n_bytes));
  return g;
}

GpuTensor GpuModel::ImportNorm(const GgufTensor& t) {
  CheckGpuType(t);
  RPG_CHECK(t.dims.size() == 1, "tensor %s should be a 1-D norm weight, got %lld dims",
            t.name.c_str(), static_cast<long long>(t.dims.size()));
  GpuTensor g;
  g.type = kGgmlF32;  // 上传后统一按 f32 用
  g.matrix = false;
  g.n_elements = t.n_elements;
  g.rows = t.n_elements;
  g.cols = 1;
  g.buf.Alloc(static_cast<size_t>(t.n_elements) * sizeof(float));

  if (t.type == kGgmlF32) {
    // norm 在 GGUF 里就是 F32（量化工具会刻意留成 F32 —— 它参与每一次归一化，
    // 量化误差会被逐层放大）。这里不做逐元素转换，直接把字节搬上去。
    RPG_CHECK(t.n_bytes == t.n_elements * 4, "norm %s: byte count does not match element count",
              t.name.c_str());
    g.buf.Upload(gguf_.TensorBytes(t), static_cast<size_t>(t.n_bytes));
  } else {
    // 少数 GGUF 会把 norm 也量化。支持它，但留日志 —— 精度有损，出现即异常。
    RPG_LOG_WARN("norm weight %s is %s; dequantized (slight precision loss)", t.name.c_str(),
                 TypeName(t.type));
    std::vector<float> tmp(static_cast<size_t>(t.n_elements));
    gguf_.DequantizeTo(t, tmp.data());
    g.buf.Upload(tmp.data(), tmp.size() * sizeof(float));
  }
  return g;
}

// ---------------------------------------------------------------- RoPE 表

void GpuModel::BuildRopeTable(int32_t max_pos) {
  const int32_t half = cfg_.head_dim / 2;
  n_rope_pos_ = max_pos > cfg_.context_length ? cfg_.context_length : max_pos;
  if (n_rope_pos_ <= 0) n_rope_pos_ = 1;

  // 与 CPU 版的 BuildRopeTable 同一个算法、同一个精度取舍：**用 double 算角度
  // 再降成 float**。rope_theta = 1e6 时 pos * inv_freq 在大位置上能到几百弧度，
  // 在 float 里算角度会明显放大误差。两边算法一致 ⇒ 表逐个 bit 相同 ⇒ RoPE
  // 这一步在层对拍里不引入任何额外误差（它也是最容易被"算错但不崩"的一步）。
  std::vector<float> rope(static_cast<size_t>(n_rope_pos_) * half * 2, 0.0f);
  std::vector<double> inv(static_cast<size_t>(half));
  for (int32_t i = 0; i < half; ++i) {
    const double exp =
        static_cast<double>(2 * i) / static_cast<double>(cfg_.head_dim);
    inv[i] = 1.0 / std::pow(static_cast<double>(cfg_.rope_theta), exp);
  }
  for (int64_t p = 0; p < n_rope_pos_; ++p) {
    float* row = rope.data() + p * half * 2;
    for (int32_t i = 0; i < half; ++i) {
      const double ang = static_cast<double>(p) * inv[i];
      row[i] = static_cast<float>(std::cos(ang));
      row[half + i] = static_cast<float>(std::sin(ang));
    }
  }

  rope_.Alloc(rope.size() * sizeof(float));
  rope_.Upload(rope.data(), rope.size() * sizeof(float));
  RPG_LOG_INFO("RoPE table uploaded to device: %lld positions x head_dim %d (%.1f MiB)",
               static_cast<long long>(n_rope_pos_), cfg_.head_dim,
               static_cast<double>(rope_.size()) / (1024.0 * 1024.0));
}

// ---------------------------------------------------------------- 加载

const GgufTensor& GpuModel::Require(const std::string& name) const {
  const GgufTensor* t = gguf_.Find(name);
  if (!t) {
    Throw("tensor %s not found in the GGUF (arch %s, %lld tensors in total)", name.c_str(),
          cfg_.arch.c_str(), static_cast<long long>(gguf_.tensors().size()));
  }
  return *t;
}

void GpuModel::CheckShape(const GgufTensor& t, int64_t rows, int64_t cols) const {
  if (t.dims.size() != 2 || t.rows() != rows || t.cols() != cols) {
    char got[64];
    std::snprintf(got, sizeof(got), "[%lld,%lld]", static_cast<long long>(t.rows()),
                  static_cast<long long>(t.cols()));
    if (!t.is_matrix()) {
      std::snprintf(got, sizeof(got), "%lld dims, not a matrix",
                    static_cast<long long>(t.dims.size()));
    }
    Throw("tensor %s has the wrong shape: expected [%lld,%lld] (rows x cols), got %s", t.name.c_str(),
          static_cast<long long>(rows), static_cast<long long>(cols), got);
  }
  CheckGpuType(t);
}

GpuModel GpuModel::Load(const std::string& path, int32_t max_pos) {
  GpuModel m;
  const DeviceInfo dev = QueryDevice(0);
  if (!dev.available) {
    Throw("no usable CUDA device, cannot upload the weights: %s", dev.reason.c_str());
  }

  m.gguf_ = GgufFile::Open(path);
  RPG_CHECK(m.gguf_.valid(), "cannot open the GGUF: %s", path.c_str());

  // 形状/超参只推一次，且与 engine.cc / model_info 用的是同一处（model_meta.h）。
  m.cfg_ = ReadModelMeta(m.gguf_);
  const ModelMeta& c = m.cfg_;

  m.tokenizer_ = Tokenizer::FromGguf(m.gguf_);
  // 词表大小有两个来源：token_embd 的行数（元数据侧）与分词器的词表大小。
  // 两者必须相等 —— 不等就是权重和分词器不是一套，而不等的结果是"能出 id
  // 但查不到词"，很难归因。这里一次性钉死。
  RPG_CHECK(static_cast<int32_t>(m.tokenizer_.vocab_size()) == c.vocab,
            "tokenizer vocab %lld does not match the %d rows of token_embd (weights and tokenizer are not a matched pair)",
            static_cast<long long>(m.tokenizer_.vocab_size()), c.vocab);

  const auto t_begin = std::chrono::steady_clock::now();

  // ---- embedding 与输出头 ----
  const GgufTensor& emb = m.Require("token_embd.weight");
  m.CheckShape(emb, c.vocab, c.hidden);
  m.token_embd_ = m.UploadMatrix(emb);

  if (!c.tied_embeddings) {
    const GgufTensor& out = m.Require("output.weight");
    m.CheckShape(out, c.vocab, c.hidden);
    m.lm_head_ = m.UploadMatrix(out);
  }
  // tied 时不传第二份：1.7B 的 embedding 是 151936×2048，Q4 下也有 150 MiB，
  // 白扔一份没有意义。lm_head() 在 tied 时直接返回 token_embd_。

  const GgufTensor& on = m.Require("output_norm.weight");
  RPG_CHECK(on.n_elements == c.hidden, "output_norm length %lld != hidden %d",
            static_cast<long long>(on.n_elements), c.hidden);
  m.output_norm_ = m.ImportNorm(on);

  // ---- 逐层 ----
  m.layers_.resize(static_cast<size_t>(c.n_layers));
  for (int32_t l = 0; l < c.n_layers; ++l) {
    GpuLayer& w = m.layers_[static_cast<size_t>(l)];
    const int32_t layer = l;
    // 注意：Load 是 static，没有 this —— 捕获取到的是局部对象 m。
    auto tensor = [&m, layer](const char* suffix) -> const GgufTensor& {
      char nm[64];
      std::snprintf(nm, sizeof(nm), "blk.%d.%s", layer, suffix);
      return m.Require(nm);
    };

    w.attn_norm = m.ImportNorm(tensor("attn_norm.weight"));
    w.ffn_norm = m.ImportNorm(tensor("ffn_norm.weight"));
    w.q_norm = m.ImportNorm(tensor("attn_q_norm.weight"));
    w.k_norm = m.ImportNorm(tensor("attn_k_norm.weight"));

    const GgufTensor& q = tensor("attn_q.weight");
    const GgufTensor& k = tensor("attn_k.weight");
    const GgufTensor& v = tensor("attn_v.weight");
    const GgufTensor& o = tensor("attn_output.weight");
    const GgufTensor& gate = tensor("ffn_gate.weight");
    const GgufTensor& up = tensor("ffn_up.weight");
    const GgufTensor& down = tensor("ffn_down.weight");

    m.CheckShape(q, c.q_dim(), c.hidden);
    m.CheckShape(k, c.kv_dim(), c.hidden);
    m.CheckShape(v, c.kv_dim(), c.hidden);
    m.CheckShape(o, c.hidden, c.q_dim());
    m.CheckShape(gate, c.intermediate, c.hidden);
    m.CheckShape(up, c.intermediate, c.hidden);
    m.CheckShape(down, c.hidden, c.intermediate);

    w.attn_q = m.UploadMatrix(q);
    w.attn_k = m.UploadMatrix(k);
    w.attn_v = m.UploadMatrix(v);
    w.attn_o = m.UploadMatrix(o);
    w.ffn_gate = m.UploadMatrix(gate);
    w.ffn_up = m.UploadMatrix(up);
    w.ffn_down = m.UploadMatrix(down);
  }

  m.BuildRopeTable(max_pos);

  const auto t_end = std::chrono::steady_clock::now();
  const double sec = std::chrono::duration<double>(t_end - t_begin).count();
  const DeviceInfo after = QueryDevice(0);
  RPG_LOG_INFO("weights on device: %s (%d layers / hidden %d / vocab %d / tied=%s)",
               c.name.c_str(), c.n_layers, c.hidden, c.vocab,
               c.tied_embeddings ? "yes" : "no");
  RPG_LOG_INFO("  device memory %.2f GiB (quantized bytes %.2f GiB) / upload took %.1f s"
               " / %.2f GiB free on device",
               static_cast<double>(m.device_bytes()) / (1024.0 * 1024.0 * 1024.0),
               static_cast<double>(c.weight_bytes) / (1024.0 * 1024.0 * 1024.0),
               sec, static_cast<double>(after.free_mem) / (1024.0 * 1024.0 * 1024.0));
  return m;
}

int64_t GpuModel::device_bytes() const {
  int64_t n = static_cast<int64_t>(rope_.size());
  auto add = [&n](const GpuTensor& t) { n += t.bytes(); };
  add(token_embd_);
  add(lm_head_);
  add(output_norm_);
  for (const GpuLayer& l : layers_) {
    add(l.attn_norm);
    add(l.ffn_norm);
    add(l.q_norm);
    add(l.k_norm);
    add(l.attn_q);
    add(l.attn_k);
    add(l.attn_v);
    add(l.attn_o);
    add(l.ffn_gate);
    add(l.ffn_up);
    add(l.ffn_down);
  }
  return n;
}

std::string GpuModel::Describe() const {
  const DeviceInfo d = QueryDevice(0);
  // 注意别把临时 std::string 的 c_str() 塞进三元里 —— 那种写法在多数编译器上
  // 恰好能活到函数调用结束，但看着就不安全。
  const std::string dev = d.available ? d.name : ("no device (" + d.reason + ")");
  char buf[768];
  std::snprintf(
      buf, sizeof(buf),
      "%s (%s)\n"
      "  size       : %d layers / hidden %d / ffn %d\n"
      "  attention  : %d Q heads / %d KV heads / head_dim %d (%d Q heads per group)\n"
      "  vocab      : %d\n"
      "  position   : RoPE theta=%.6g, table covers %lld positions (max %d)\n"
      "  norm       : RMSNorm eps=%.3g\n"
      "  output     : %s\n"
      "  quantized  : %lld bytes (%.2f GiB)\n"
      "  device mem : %lld bytes (%.2f GiB)\n"
      "  device     : %s\n",
      cfg_.name.c_str(), cfg_.arch.c_str(), cfg_.n_layers, cfg_.hidden,
      cfg_.intermediate, cfg_.n_heads, cfg_.n_kv_heads, cfg_.head_dim,
      cfg_.heads_per_kv(), cfg_.vocab, cfg_.rope_theta,
      static_cast<long long>(n_rope_pos_), cfg_.context_length,
      static_cast<double>(cfg_.rms_eps),
      cfg_.tied_embeddings ? "shared with token_embd (tied)" : "separate output.weight",
      static_cast<long long>(cfg_.weight_bytes),
      static_cast<double>(cfg_.weight_bytes) / (1024.0 * 1024.0 * 1024.0),
      static_cast<long long>(device_bytes()),
      static_cast<double>(device_bytes()) / (1024.0 * 1024.0 * 1024.0),
      dev.c_str());
  return buf;
}

}  // namespace rpg

#endif  // RPG_HAS_CUDA
