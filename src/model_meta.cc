// model_meta.cc —— 元数据解析的唯一实现（见 model_meta.h 里"为什么只留一处"）。

#include "model_meta.h"

#include <string>

#include "common.h"

namespace rpg {

ModelMeta ReadModelMeta(const GgufFile& g) {
  ModelMeta m;

  m.arch = g.architecture();
  // 只实现 qwen3。写"尽力而为"的分支在这里没有意义：形状字段的名字按架构不同，
  // 猜错的后果是另一种静默算错。
  RPG_CHECK(m.arch == "qwen3", "模型架构是 %s，本引擎（当前）只实现 qwen3",
            m.arch.c_str());
  m.name = g.MetaStr("general.name", g.MetaStr("general.basename", "(无名)"));

  m.n_layers = static_cast<int32_t>(g.ArchInt("block_count", 0));
  m.hidden = static_cast<int32_t>(g.ArchInt("embedding_length", 0));
  m.n_heads = static_cast<int32_t>(g.ArchInt("attention.head_count", 0));
  m.n_kv_heads =
      static_cast<int32_t>(g.ArchInt("attention.head_count_kv", m.n_heads));
  m.intermediate = static_cast<int32_t>(g.ArchInt("feed_forward_length", 0));
  m.context_length = static_cast<int32_t>(g.ArchInt("context_length", 4096));

  // head_dim 只能读 key_length。**不可以**退化成 hidden / n_heads —— Qwen3 两者
  // 恰好相等，会把这条错误掩掉，换个模型就错（见文件头）。
  const int64_t key_len = g.ArchInt("attention.key_length", 0);
  if (key_len > 0) {
    m.head_dim = static_cast<int32_t>(key_len);
  } else {
    RPG_CHECK(m.n_heads > 0 && m.hidden % m.n_heads == 0,
              "缺 attention.key_length 且 %d 不被 %d 整除，无法推 head_dim",
              m.hidden, m.n_heads);
    RPG_LOG_WARN("GGUF 缺 attention.key_length，按 hidden/heads 推得 %d",
                 m.hidden / m.n_heads);
    m.head_dim = m.hidden / m.n_heads;
  }
  m.value_dim = static_cast<int32_t>(g.ArchInt("attention.value_length", m.head_dim));

  m.rope_theta = g.ArchFloat("rope.freq_base", 10000.0f);
  m.rms_eps = g.ArchFloat("attention.layer_norm_rms_epsilon", 1e-6f);

  m.bos_id = static_cast<int32_t>(g.MetaInt("tokenizer.ggml.bos_token_id", -1));
  m.eos_id = static_cast<int32_t>(g.MetaInt("tokenizer.ggml.eos_token_id", -1));
  m.pad_id =
      static_cast<int32_t>(g.MetaInt("tokenizer.ggml.padding_token_id", -1));
  m.add_bos = g.MetaBool("tokenizer.ggml.add_bos_token", false);

  // 词表大小取 token_embd 的行数 —— 不去解析 tokenizer.ggml.tokens 那个大数组，
  // 那一份留给分词器自己读。两者必须一致，由下面的形状校验保证。
  const GgufTensor* emb = g.Find("token_embd.weight");
  RPG_CHECK(emb != nullptr, "找不到 token_embd.weight");
  m.vocab = static_cast<int32_t>(emb->rows());

  RPG_CHECK(m.n_layers > 0 && m.hidden > 0 && m.n_heads > 0 && m.n_kv_heads > 0 &&
                m.head_dim > 0 && m.intermediate > 0 && m.vocab > 0,
            "GGUF 元数据不全：layers=%d hidden=%d heads=%d kv=%d head_dim=%d "
            "ffn=%d vocab=%d",
            m.n_layers, m.hidden, m.n_heads, m.n_kv_heads, m.head_dim,
            m.intermediate, m.vocab);
  RPG_CHECK(m.n_heads % m.n_kv_heads == 0,
            "Q 头数 %d 不是 KV 头数 %d 的整数倍，GQA 无法分组", m.n_heads,
            m.n_kv_heads);
  RPG_CHECK(m.value_dim == m.head_dim,
            "value_length=%d 与 key_length=%d 不同，当前实现假设两者相等",
            m.value_dim, m.head_dim);

  // tied：两种命名都查（不同的转换工具写法不同）。
  m.tied_embeddings =
      g.Find("output.weight") == nullptr && g.Find("lm_head.weight") == nullptr;

  for (const GgufTensor& t : g.tensors()) m.weight_bytes += t.n_bytes;
  return m;
}

}  // namespace rpg
