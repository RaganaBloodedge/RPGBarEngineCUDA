// RPGBarEngineCUDA —— GGUF 元数据 → 形状与超参。**整个仓库只有这一份推导。**
//
// 为什么专门开一个文件：这份 Q4_K_M 权重里，形状错了不会崩，只会**静默算错**。
// 典型的一条是 head_dim —— Qwen3-1.7B 上 `hidden/n_heads = 2048/16 = 128` 恰好
// 等于真的 `attention.key_length`，所以"用除法推导"这条错路在 1.7B 上能跑通，
// 换成 Qwen2.5-1.5B（hidden 1536 / 16 头 = 96，而 key_length 是 128）就错。
// 这类"两处推导、其中一处是错的"的坑，唯一的防法就是**只留一处推导**。
//
// 此前这份逻辑存在两份：从 CPU 版复制来的 src/model.cc（但 model.cc 属于
// "重写层"、没有复制进本仓库，所以那份不在），以及 engine.cc 里 load_model 内联
// 的一段。P1 要把权重搬上显存，会再需要一次 —— 那就变成第三份。于是抽到这里：
//   * engine.cc 的 load_model / model_info 用 ModelMeta
//   * gpu_model.cu 的 GpuModel::Load 用 ModelMeta
//   * 将来 CPU 版也该收到这里（它现在还有自己的 ModelConfig）
//
// 字段名与 CPU 版 model.h 的 ModelConfig **逐字对齐**，方便两边对着读。
#ifndef RPGBAR_MODEL_META_H_
#define RPGBAR_MODEL_META_H_

#include <cstdint>
#include <string>

#include "gguf.h"

namespace rpg {

struct ModelMeta {
  std::string arch;  // "qwen3" —— 别的架构这里就抛错，不做"尽力而为"
  std::string name;

  int32_t n_layers = 0;
  int32_t hidden = 0;
  int32_t n_heads = 0;     // Q 头数
  int32_t n_kv_heads = 0;  // KV 头数（GQA）
  int32_t head_dim = 0;    // 只读 qwen3.attention.key_length，不许用 hidden/heads 推
  int32_t value_dim = 0;   // 通常 == head_dim
  int32_t intermediate = 0;
  int32_t vocab = 0;       // 取 token_embd 的行数（不去解析 tokens 数组）
  int32_t context_length = 0;

  float rope_theta = 10000.0f;
  float rms_eps = 1e-6f;

  // 没有 output.weight（也没有 lm_head.weight）时为真 —— lm_head 复用 token_embd。
  // 两种命名都查：转换成 GGUF 的工具有时写 output.weight、有时写 lm_head.weight。
  bool tied_embeddings = false;

  // 分词器的特殊 token 与"要不要补 BOS"。Qwen3 的 add_bos 是 false，但转换版本
  // 不保证 —— 对拍时两边都得照这个键来，否则 token 序列都不一样，谈不到一起。
  int32_t bos_id = -1;
  int32_t eos_id = -1;
  int32_t pad_id = -1;
  bool add_bos = false;

  // 全部张量的裸字节数之和（量化后的实际大小，不含任何 f32 副本）。
  int64_t weight_bytes = 0;

  int32_t q_dim() const { return n_heads * head_dim; }
  int32_t kv_dim() const { return n_kv_heads * head_dim; }
  int32_t heads_per_kv() const { return n_kv_heads > 0 ? n_heads / n_kv_heads : 0; }
};

// 读元数据 + 做完所有一致性校验（缺项、不整除、value_dim != head_dim 都抛）。
// 调用方拿到的是一个**已经验证过、可以直接拿去分配缓冲**的结构。
ModelMeta ReadModelMeta(const GgufFile& g);

}  // namespace rpg

#endif  // RPGBAR_MODEL_META_H_
