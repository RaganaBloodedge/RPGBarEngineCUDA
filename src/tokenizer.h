// RPGBarEngine —— 分词器（Qwen 的 BBPE）
//
// Qwen3 用的是 GPT-2 风格的 **byte-level BPE**，词表与 merges 都从 GGUF 里读
// （tokenizer.ggml.tokens / tokenizer.ggml.merges / tokenizer.ggml.token_type）。
//
// 完整流程（与 HF tokenizers 的 tokenizer.json 定义一致）：
//
//   1. 特殊 token 切分 —— 26 个 added token（<|im_start|> / <|im_end|> /
//      <tool_call> / <think> …）不参与 BPE，直接以整串命中固定 id。
//      GGUF 里用 token_type 标出：3=CONTROL、4=USER_DEFINED。
//   2. 预分词 —— 按 qwen2 正则把普通文本切成小片：
//        (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//        |[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//   3. 字节编码 —— 每个字节按 GPT-2 的 bytes_to_unicode 映到 1 个码点
//      （空格 0x20 → 'Ġ'），这样"字符"粒度的 BPE 等价于"字节"粒度。
//   4. BPE 合并 —— 反复取"rank 最小的相邻对"合并一个，直到无可合并。
//      这个"每次只合一个、并列取最左"的次序与 HF 的实现一致；换成
//      "同 rank 全合"在边界情形会给出不同结果。
//   5. 查表 —— 合并后的每个片段查 tokens → id。
//
// 已知差异（刻意保留）：
//   * **不做 NFC 归一化**。Qwen3 的 tokenizer.json 里写了 normalizer=NFC，
//     但 llama.cpp 对 BPE 不做任何 Unicode 归一化。我们的端到端基准是
//     llama.cpp，所以选择与它一致 —— 输入已是 NFC（绝大多数中英文本如此）时
//     两者与 HF 完全一致；输入是分解形式（如 e + U+0301）时会与 HF 不同。
//     要补的话需要引入完整的 NFC 表和"重排 + 组合"算法，属于独立工作。

#ifndef RPGBAR_TOKENIZER_H_
#define RPGBAR_TOKENIZER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rpg {

class GgufFile;

class Tokenizer {
 public:
  Tokenizer();

  // 从已打开的 GGUF 里读词表与 merges。缺 merges 或 token_type 会抛错。
  static Tokenizer FromGguf(const GgufFile& g);

  // 编码。parse_special=true 时文本里的特殊 token 串会被切成单个 id。
  std::vector<int32_t> Encode(const std::string& text,
                              bool parse_special = true) const;

  // 解码回文本。
  std::string Decode(const std::vector<int32_t>& ids,
                     bool skip_special = false) const;

  // 只跑 BPE（不做特殊 token 切分）。给对拍脚本定位问题用。
  std::vector<int32_t> EncodePlain(const std::string& text) const;

  int32_t IdOf(const std::string& token) const;
  const std::string& TokenOf(int32_t id) const;
  bool IsSpecial(int32_t id) const;

  int32_t bos_id() const { return bos_; }
  int32_t eos_id() const { return eos_; }
  int32_t pad_id() const { return pad_; }
  int32_t unk_id() const { return unk_; }

  size_t vocab_size() const { return tokens_.size(); }
  size_t merge_count() const { return merges_.size(); }

  // 预分词结果（给测试看正则切得对不对）。
  std::vector<std::string> PreTokenize(const std::string& text) const;

 private:
  std::vector<int32_t> EncodePiece(const std::string& piece) const;

  std::vector<std::string> tokens_;
  std::vector<uint8_t> types_;
  std::unordered_map<std::string, int32_t> token_to_id_;
  // key = left + '\x1F' + right。'\x1F' 不会出现在字节编码后的 token 串里
  // （0x1F 会被映到 U+011F），所以这个分隔符是安全的。
  std::unordered_map<std::string, int32_t> merges_;
  // 特殊 token 按长度降序，保证贪婪匹配到最长的那一个。
  std::vector<std::pair<std::string, int32_t>> specials_;

  int32_t bos_ = -1;
  int32_t eos_ = -1;
  int32_t pad_ = -1;
  int32_t unk_ = -1;
};

}  // namespace rpg

#endif  // RPGBAR_TOKENIZER_H_
