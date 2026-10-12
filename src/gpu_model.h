// RPGBarEngineCUDA —— 显存里的模型（权重 + 配置 + RoPE 表）
//
// 职责边界与 CPU 版的 src/model.cc 一样："把 GGUF 变成一堆可用的权重 + 一份
// 配置"，不含任何前向逻辑（前向在 qwen3_gpu.cu）。**但实现是重写的**，原因不是
// 抄不动，而是两边的"权重"压根不是一种东西：
//
//   CPU 版：QuantMatrix 是**指向 mmap 的视图**（零拷贝，权重要多少读多少）
//   这里  ：GpuTensor  是**显存里的字节**（量化块原样上传，kernel 里即时反量化）
//
// ## 量化权重直传显存，不解量化
//
// Qwen3-1.7B-Q4_K_M 的量化字节是 1.03 GiB。如果在上传前解成 fp16 是 3.5 GiB、
// 解成 fp32 是 7 GiB —— 8 GB 的卡上 fp32 直接出局，fp16 也没给 KV 留余地。
// 所以走"原样上传 + kernel 里 on-the-fly 反量化"：**读显存的量等于量化的量**，
// 这也是 llama.cpp 在显存上的做法（见 docs/DESIGN.md §4.1）。
//
// ## 只认三种类型，且在加载期就拒
//
// F32（norm）、Q4_K、Q6_K —— 这份 Q4_K_M 权重里只有这三种。别的类型不是
// "以后再支持"，而是**加载期抛错**：让一个不认识的量化块进到 kernel 里，
// 症状会是"能出词但胡说"，那种 bug 比一条明确的错误信息贵得多。
#ifndef RPGBAR_GPU_MODEL_H_
#define RPGBAR_GPU_MODEL_H_

#include "cuda_common.h"

#ifdef RPG_HAS_CUDA

#include <cstdint>
#include <string>
#include <vector>

#include "gguf.h"
#include "model_meta.h"  // 形状与超参（与 engine.cc 共用同一份推导）
#include "quant.h"       // kGgmlF32 / TypeName / TypeRowBytes
#include "tokenizer.h"

namespace rpg {

// 显存里的一个权重张量。**量化张量按原始字节上传**，所以 buf 的大小是
// 量化后的大小（不是 rows*cols*4）。
struct GpuTensor {
  DeviceBuffer buf;
  uint32_t type = kGgmlF32;
  bool matrix = false;   // 2 维（权重矩阵）还是 1 维（norm）
  int64_t rows = 0;      // 矩阵：行数（= dims[1]，每行是一个输出神经元）
  int64_t cols = 0;      // 矩阵：列数（= dims[0]，也是点积长度）
  int64_t n_elements = 0;

  bool valid() const { return !buf.empty(); }
  const char* type_name() const { return TypeName(type); }
  int64_t bytes() const { return static_cast<int64_t>(buf.size()); }
  // 一行的字节数（量化块对齐）。F32 的 norm 没有"行"的概念，别对它调。
  int64_t row_bytes() const { return TypeRowBytes(type, cols); }
};

// 一层的全部权重。与 CPU 版的 LayerWeights 一一对应，只是每一项都在显存。
struct GpuLayer {
  GpuTensor attn_norm;  // [hidden]      f32
  GpuTensor ffn_norm;   // [hidden]      f32
  GpuTensor q_norm;     // [head_dim]    f32，RoPE 之前
  GpuTensor k_norm;     // [head_dim]    f32，RoPE 之前

  GpuTensor attn_q;  // [n_heads*head_dim, hidden]
  GpuTensor attn_k;  // [n_kv_heads*head_dim, hidden]
  GpuTensor attn_v;  // [n_kv_heads*head_dim, hidden]
  GpuTensor attn_o;  // [hidden, n_heads*head_dim]

  GpuTensor ffn_gate;  // [intermediate, hidden]
  GpuTensor ffn_up;    // [intermediate, hidden]
  GpuTensor ffn_down;  // [hidden, intermediate]
};

class GpuModel {
 public:
  // max_pos：RoPE 表覆盖的位置数（超过 context_length 会被截到 context_length）。
  // 与 CPU 版同语义 —— 表在 host 上用 double 算好再上传，这样两边**逐个 bit
  // 相同**（RoPE 表算错是"注意力整体偏掉"，很难从事后现象反推）。
  static GpuModel Load(const std::string& path, int32_t max_pos = 4096);

  const ModelMeta& config() const { return cfg_; }
  const GgufFile& gguf() const { return gguf_; }
  const Tokenizer& tokenizer() const { return tokenizer_; }
  const std::string& path() const { return gguf_.path(); }

  const GpuTensor& token_embd() const { return token_embd_; }
  // tied 时**不重复上传第二份**，这里直接返回 embedding（1.7B 的 embedding
  // 是 151936×2048，Q4 下也有 150 MiB）。
  const GpuTensor& lm_head() const {
    return cfg_.tied_embeddings ? token_embd_ : lm_head_;
  }
  const GpuTensor& output_norm() const { return output_norm_; }
  const std::vector<GpuLayer>& layers() const { return layers_; }

  // RoPE 表：显存里 [n_rope_pos, head_dim]，一行内前半是 cos、后半是 sin。
  // 行内布局与 CPU 版的 rope_cos/rope_sin 完全一致 —— kernel 里就是
  //   cos = tab + pos * head_dim, sin = cos + head_dim/2
  const float* rope_dev() const { return static_cast<const float*>(rope_.data()); }
  int64_t rope_positions() const { return n_rope_pos_; }

  int64_t device_bytes() const;
  std::string Describe() const;

 private:
  const GgufTensor& Require(const std::string& name) const;
  void CheckShape(const GgufTensor& t, int64_t rows, int64_t cols) const;
  // 1 维 norm 权重 → 显存（F32 原样上传；量化的先解成 f32 再上）。
  GpuTensor ImportNorm(const GgufTensor& t);
  // 量化矩阵 → 显存（原始字节）。
  GpuTensor UploadMatrix(const GgufTensor& t);
  void BuildRopeTable(int32_t max_pos);

  GgufFile gguf_;
  ModelMeta cfg_;
  Tokenizer tokenizer_;

  GpuTensor token_embd_;
  GpuTensor lm_head_;
  GpuTensor output_norm_;
  std::vector<GpuLayer> layers_;
  DeviceBuffer rope_;
  int64_t n_rope_pos_ = 0;

  // 上传完不需要再碰 mmap，但 Tokenizer 可能引用它 —— 留着（代价是虚拟地址，
  // 不是常驻内存）。
  int64_t weight_bytes_ = 0;
  int64_t device_bytes_ = 0;
};

}  // namespace rpg

#endif  // RPG_HAS_CUDA
#endif  // RPGBAR_GPU_MODEL_H_
