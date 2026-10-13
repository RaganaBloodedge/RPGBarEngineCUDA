// RPGBarEngine —— GGUF 读取
//
// 按 ggml 的 docs/gguf.md（v3）实现，兼容 v1/v2（唯一差别：v1 的头里两个计数
// 是 uint32，v2/v3 是 uint64）。
//
// 文件结构（顺序）：
//   magic u32 = 'GGUF' | version u32 | tensor_count uN | metadata_kv_count uN
//   metadata_kv[]        —— 每条：key(gguf_string) | value_type u32 | value
//   tensor_info[]        —— 每条：name | n_dims u32 | dims u64[] | type u32 | offset u64
//   对齐填充（到 general.alignment，默认 32）
//   tensor_data[]        —— 各张量的裸字节；tensor_info.offset 相对本段起点
//
// 维度约定（GGUF 与 ggml 一致）：dims[0] 是**最快变化**的维度。所以内存里
// 一个权重矩阵是行主的 [dims[1], dims[0]] —— 即"每行是一个输出神经元的权重"，
// 正好对上引擎内部 `out = x · Wᵀ` 的约定（见 ops.h）。

#ifndef RPGBAR_GGUF_H_
#define RPGBAR_GGUF_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "json.h"
#include "mmap_file.h"

namespace rpg {

struct GgufTensor {
  std::string name;
  std::vector<int64_t> dims;  // dims[0] 是最快维
  uint32_t type = 0;          // GgmlType id
  int64_t offset = 0;         // 相对 tensor_data 起点
  int64_t n_elements = 0;
  int64_t n_bytes = 0;

  // 行主视角：权重是 [rows, cols]，cols 是最快维。
  int64_t cols() const { return dims.empty() ? 0 : dims[0]; }
  int64_t rows() const { return dims.size() >= 2 ? dims[1] : 1; }
  bool is_matrix() const { return dims.size() == 2; }
};

class GgufFile {
 public:
  GgufFile() = default;

  // 打开 + 解析元数据与张量表（不触碰张量数据本身，靠 mmap 惰性换页）。
  static GgufFile Open(const std::string& path);

  bool valid() const { return file_.valid(); }

  // ---- 元数据 ----
  const Json& meta() const { return meta_; }
  const Json& operator[](const std::string& key) const { return meta_[key]; }
  bool HasMeta(const std::string& key) const { return meta_.Has(key); }

  int64_t MetaInt(const std::string& key, int64_t def) const;
  float MetaFloat(const std::string& key, float def) const;
  bool MetaBool(const std::string& key, bool def) const;
  std::string MetaStr(const std::string& key, const std::string& def = "") const;

  // 架构（general.architecture），如 "qwen3"。
  const std::string& architecture() const { return arch_; }
  // 按架构前缀取键：ArchInt("block_count") ←→ 读 "qwen3.block_count"。
  int64_t ArchInt(const std::string& suffix, int64_t def) const;
  float ArchFloat(const std::string& suffix, float def) const;

  // ---- 张量 ----
  const std::vector<GgufTensor>& tensors() const { return tensors_; }
  const GgufTensor* Find(const std::string& name) const;

  // 张量数据的裸指针（指向 mmap 区，生命周期同本对象）。
  const uint8_t* TensorBytes(const GgufTensor& t) const;

  // 把某个张量反量化成 f32 写入 dst（dst 需容纳 n_elements 个 float）。
  void DequantizeTo(const GgufTensor& t, float* dst) const;

  uint32_t alignment() const { return alignment_; }
  const std::string& path() const { return file_.path(); }

  // 人类可读摘要（给 dump 工具与日志用）。
  std::string Describe() const;

  // 统计本文件出现了哪些 ggml 类型及各自的张量数。
  std::vector<std::pair<std::string, int>> TypeHistogram() const;

 private:
  MappedFile file_;
  Json meta_;
  std::vector<GgufTensor> tensors_;
  std::unordered_map<std::string, int> index_;
  std::string arch_;
  uint32_t alignment_ = 32;
  int64_t data_start_ = 0;  // tensor_data 在文件中的绝对偏移
};

}  // namespace rpg

#endif  // RPGBAR_GGUF_H_
