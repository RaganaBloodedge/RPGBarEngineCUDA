#include "tokenizer.h"

#include <algorithm>
#include <climits>
#include <cstring>

#include "common.h"
#include "gguf.h"
#include "unicode.h"

namespace rpg {
namespace {

// 码点 + 它在原串里的字节偏移。预分词在"码点"上做判断，最后按字节切片。
struct CpEntry {
  uint32_t cp;
  uint32_t off;
};

std::vector<CpEntry> DecodeCps(const std::string& s) {
  std::vector<CpEntry> out;
  out.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) {
    const Utf8Char c = Utf8Decode(s.data() + i, s.size() - i);
    out.push_back(CpEntry{c.cp, static_cast<uint32_t>(i)});
    i += (c.len > 0) ? static_cast<size_t>(c.len) : 1;
  }
  return out;
}

// ---------------------------------------------------------------------------
// qwen2 预分词正则的手写实现
//
// 正则原文（tokenizer.json / llama.cpp 的 LLAMA_VOCAB_PRE_TYPE_QWEN2）：
//   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//   |[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// 为什么手写而不引正则库：这个模式只用到一个分支、几个字符类和一个负向前瞻，
// 但引入一个完整的正则引擎（要支持 Unicode 类别、回溯语义）代价远大于收益，
// 而且引擎要编进游戏本体。手写版本的分支顺序、贪婪/回溯行为与 Rust regex 的
// "最左优先"语义逐条对齐，并用上万条文本与官方对拍验证（tools/tok_check）。

constexpr uint32_t kSpace = 0x20;
constexpr uint32_t kApostrophe = 0x27;

inline bool IsNewline(uint32_t c) { return c == '\r' || c == '\n'; }

// 返回从 cps[p] 起的匹配长度（码点数）；0 表示不匹配。
int MatchAt(const std::vector<CpEntry>& cps, size_t p) {
  const size_t m = cps.size();
  const uint32_t c = cps[p].cp;

  // 1. 英文缩写：'s 't 're 've 'm 'll 'd
  if (c == kApostrophe && p + 1 < m) {
    const uint32_t c1 = cps[p + 1].cp;
    const bool is_lower = c1 >= 'a' && c1 <= 'z';
    const bool is_upper = c1 >= 'A' && c1 <= 'Z';
    if (is_lower || is_upper) {
      const uint32_t l = (c1 | 0x20);  // 归一到小写
      if (l == 's' || l == 't' || l == 'm' || l == 'd') return 2;
      if ((l == 'r' || l == 'v') && p + 2 < m) {
        const uint32_t c2 = cps[p + 2].cp | 0x20;
        if (c2 == 'e') return 3;
      }
      if (l == 'l' && p + 2 < m) {
        const uint32_t c2 = cps[p + 2].cp | 0x20;
        if (c2 == 'l') return 3;
      }
    }
  }

  // 2. [^\r\n\p{L}\p{N}]?\p{L}+
  {
    size_t q = p;
    if (!IsNewline(c) && !IsUnicodeLetter(c) && !IsUnicodeNumber(c)) q = p + 1;
    size_t r = q;
    while (r < m && IsUnicodeLetter(cps[r].cp)) ++r;
    if (r > q) return static_cast<int>(r - p);
    // 走到这里说明可选前缀之后没有字母。回溯去掉前缀也无济于事：
    // 能进到 q=p+1 就说明 c 本身不是字母或数字，\p{L}+ 在 p 处同样匹配不了。
  }

  // 3. \p{N}
  if (IsUnicodeNumber(c)) return 1;

  // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
  {
    size_t q = (c == kSpace) ? p + 1 : p;
    size_t r = q;
    while (r < m && !IsUnicodeSpace(cps[r].cp) && !IsUnicodeLetter(cps[r].cp) &&
           !IsUnicodeNumber(cps[r].cp)) {
      ++r;
    }
    if (r > q) {
      size_t t = r;
      while (t < m && IsNewline(cps[t].cp)) ++t;
      return static_cast<int>(t - p);
    }
  }

  // 5/6/7. 空白（三个分支只在"有没有换行""是否到结尾"上有区别）
  if (IsUnicodeSpace(c)) {
    size_t e = p;
    while (e < m && IsUnicodeSpace(cps[e].cp)) ++e;

    // 5. \s*[\r\n]+ —— 最长匹配：找到 [p,e) 里最靠后的换行，匹配到它为止。
    //    （因为它右边的字符必须属于 [\r\n]，而 \s* 可以吃掉前面的任意空白。）
    size_t last_nl = SIZE_MAX;
    for (size_t i = p; i < e; ++i) {
      if (IsNewline(cps[i].cp)) last_nl = i;
    }
    if (last_nl != SIZE_MAX) return static_cast<int>(last_nl - p + 1);

    // 6. \s+(?!\S)
    //    空白跑到输入末尾 → 整个吃下；否则回溯一个字符，让后面的空白位
    //    满足"后面不是非空白"。
    if (e == m) return static_cast<int>(e - p);
    if (e - p >= 2) return static_cast<int>(e - 1 - p);

    // 7. \s+
    return static_cast<int>(e - p);
  }

  return 0;  // 理论到不了：正则的最后一个分支覆盖一切空白，其它字符由 2/3/4 覆盖
}

}  // namespace

Tokenizer::Tokenizer() = default;

Tokenizer Tokenizer::FromGguf(const GgufFile& g) {
  Tokenizer t;

  const Json& toks = g["tokenizer.ggml.tokens"];
  RPG_CHECK(toks.IsArray() && toks.size() > 0,
            "GGUF 里没有 tokenizer.ggml.tokens，无法建分词器");

  const size_t n = toks.size();
  t.tokens_.reserve(n);
  t.token_to_id_.reserve(n * 2);
  for (size_t i = 0; i < n; ++i) {
    // 词表里理论上可能出现重复串（例如量化工具写入的占位 token），
    // 保持"先出现的 id 获胜"，与 llama.cpp 的建表方式一致。
    t.tokens_.push_back(toks.At(i).AsString());
    t.token_to_id_.emplace(t.tokens_.back(), static_cast<int32_t>(i));
  }

  t.types_.assign(n, 1);
  const Json& types = g["tokenizer.ggml.token_type"];
  if (types.IsArray()) {
    RPG_CHECK(types.size() == n, "token_type 数量（" RPG_SZ "）与 tokens（" RPG_SZ
                                 "）不一致",
              RPG_SZ_ARG(types.size()), RPG_SZ_ARG(n));
    for (size_t i = 0; i < n; ++i) {
      t.types_[i] = static_cast<uint8_t>(types.At(i).AsInt(1));
    }
  }

  const Json& merges = g["tokenizer.ggml.merges"];
  RPG_CHECK(merges.IsArray() && merges.size() > 0,
            "GGUF 里没有 tokenizer.ggml.merges —— Qwen 的 BBPE 必须有 merges");
  t.merges_.reserve(merges.size() * 2);
  for (size_t i = 0; i < merges.size(); ++i) {
    const std::string& s = merges.At(i).AsString();
    const size_t sp = s.find(' ');
    RPG_CHECK(sp != std::string::npos,
              "merge 第 " RPG_SZ " 项没有空格分隔：%s", RPG_SZ_ARG(i), s.c_str());
    std::string key = s.substr(0, sp);
    key.push_back('\x1F');
    key.append(s, sp + 1, std::string::npos);
    // rank 小的先合并；并列时取最左（在合并循环里用严格小于实现）。
    t.merges_.emplace(std::move(key), static_cast<int32_t>(i));
  }

  // 特殊 token：CONTROL(3) 与 USER_DEFINED(4)。UNUSED(5) 不算 ——
  // 那些是 [PAD…] 占位，不该在正文里被贪婪匹配掉。
  for (size_t i = 0; i < n; ++i) {
    if (t.types_[i] == 3 || t.types_[i] == 4) {
      t.specials_.emplace_back(t.tokens_[i], static_cast<int32_t>(i));
    }
  }
  std::sort(t.specials_.begin(), t.specials_.end(),
            [](const std::pair<std::string, int32_t>& a,
               const std::pair<std::string, int32_t>& b) {
              return a.first.size() > b.first.size();
            });

  t.bos_ = static_cast<int32_t>(g.MetaInt("tokenizer.ggml.bos_token_id", -1));
  t.eos_ = static_cast<int32_t>(g.MetaInt("tokenizer.ggml.eos_token_id", -1));
  t.pad_ =
      static_cast<int32_t>(g.MetaInt("tokenizer.ggml.padding_token_id", -1));
  t.unk_ =
      static_cast<int32_t>(g.MetaInt("tokenizer.ggml.unknown_token_id", -1));

  RPG_LOG_INFO("分词器就绪：" RPG_SZ " 个 token，" RPG_SZ " 条 merges，" RPG_SZ
               " 个特殊 token",
               RPG_SZ_ARG(t.tokens_.size()), RPG_SZ_ARG(t.merges_.size()),
               RPG_SZ_ARG(t.specials_.size()));
  return t;
}

int32_t Tokenizer::IdOf(const std::string& token) const {
  const auto it = token_to_id_.find(token);
  return it == token_to_id_.end() ? -1 : it->second;
}

const std::string& Tokenizer::TokenOf(int32_t id) const {
  static const std::string kEmpty;
  if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) return kEmpty;
  return tokens_[static_cast<size_t>(id)];
}

bool Tokenizer::IsSpecial(int32_t id) const {
  if (id < 0 || static_cast<size_t>(id) >= types_.size()) return false;
  const uint8_t t = types_[static_cast<size_t>(id)];
  return t == 3 || t == 4;
}

std::vector<std::string> Tokenizer::PreTokenize(const std::string& text) const {
  std::vector<std::string> pieces;
  const std::vector<CpEntry> cps = DecodeCps(text);
  const size_t m = cps.size();
  size_t p = 0;
  while (p < m) {
    int len = MatchAt(cps, p);
    if (len <= 0) len = 1;  // 防御：单字符成片
    const size_t end = p + static_cast<size_t>(len);
    const size_t b0 = cps[p].off;
    const size_t b1 = (end < m) ? cps[end].off : text.size();
    pieces.push_back(text.substr(b0, b1 - b0));
    p = end;
  }
  return pieces;
}

std::vector<int32_t> Tokenizer::EncodePiece(const std::string& piece) const {
  std::vector<int32_t> ids;
  if (piece.empty()) return ids;

  // 字节 → 字节编码串
  std::string enc;
  enc.reserve(piece.size() * 2);
  for (const unsigned char b : piece) AppendUtf8(&enc, ByteToCodepoint(b));

  // 拆成符号（每个符号 = 原串的一个字节）
  const std::vector<CpEntry> cps = DecodeCps(enc);
  std::vector<std::string> syms;
  syms.reserve(cps.size());
  for (size_t i = 0; i < cps.size(); ++i) {
    const size_t b0 = cps[i].off;
    const size_t b1 = (i + 1 < cps.size()) ? cps[i + 1].off : enc.size();
    syms.emplace_back(enc, b0, b1 - b0);
  }

  // BPE：每轮合并"rank 最小"的一对（并列取最左），一次只合一个。
  std::string key;
  const auto rank_of = [&](const std::string& l, const std::string& r) -> int {
    key.clear();
    key.reserve(l.size() + r.size() + 1);
    key += l;
    key.push_back('\x1F');
    key += r;
    const auto it = merges_.find(key);
    return it == merges_.end() ? -1 : it->second;
  };

  while (syms.size() > 1) {
    int best = -1;
    int best_rank = INT_MAX;
    for (size_t i = 0; i + 1 < syms.size(); ++i) {
      const int r = rank_of(syms[i], syms[i + 1]);
      if (r >= 0 && r < best_rank) {
        best_rank = r;
        best = static_cast<int>(i);
      }
    }
    if (best < 0) break;
    syms[static_cast<size_t>(best)] += syms[static_cast<size_t>(best) + 1];
    syms.erase(syms.begin() + best + 1);
  }

  for (const std::string& s : syms) {
    const auto it = token_to_id_.find(s);
    if (it != token_to_id_.end()) {
      ids.push_back(it->second);
      continue;
    }
    // 兜底：整体查不到就按单个字节查（byte-level BPE 里每个字节都该有 token）。
    const std::vector<CpEntry> sc = DecodeCps(s);
    for (const CpEntry& e : sc) {
      const int b = CodepointToByte(e.cp);
      RPG_CHECK(b >= 0, "符号 %s 既不在词表里，也不是字节编码字符", s.c_str());
      std::string bs;
      AppendUtf8(&bs, ByteToCodepoint(static_cast<uint8_t>(b)));
      const auto bit = token_to_id_.find(bs);
      RPG_CHECK(bit != token_to_id_.end(), "字节 0x%02X 在词表里没有对应 token",
                b);
      ids.push_back(bit->second);
    }
  }
  return ids;
}

std::vector<int32_t> Tokenizer::EncodePlain(const std::string& text) const {
  std::vector<int32_t> out;
  for (const std::string& piece : PreTokenize(text)) {
    const std::vector<int32_t> part = EncodePiece(piece);
    out.insert(out.end(), part.begin(), part.end());
  }
  return out;
}

std::vector<int32_t> Tokenizer::Encode(const std::string& text,
                                       bool parse_special) const {
  std::vector<int32_t> out;
  const size_t n = text.size();
  size_t i = 0;
  size_t plain_start = 0;
  while (i < n) {
    int32_t hit_id = -1;
    size_t hit_len = 0;
    if (parse_special) {
      for (const auto& sp : specials_) {
        if (sp.first.size() <= n - i &&
            text.compare(i, sp.first.size(), sp.first) == 0) {
          hit_id = sp.second;
          hit_len = sp.first.size();
          break;
        }
      }
    }
    if (hit_id >= 0) {
      if (i > plain_start) {
        const std::vector<int32_t> part =
            EncodePlain(text.substr(plain_start, i - plain_start));
        out.insert(out.end(), part.begin(), part.end());
      }
      out.push_back(hit_id);
      i += hit_len;
      plain_start = i;
    } else {
      ++i;
    }
  }
  if (n > plain_start) {
    const std::vector<int32_t> part = EncodePlain(text.substr(plain_start));
    out.insert(out.end(), part.begin(), part.end());
  }
  return out;
}

std::string Tokenizer::Decode(const std::vector<int32_t>& ids,
                              bool skip_special) const {
  std::string out;
  for (const int32_t id : ids) {
    if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) continue;
    if (skip_special && IsSpecial(id)) continue;
    const std::vector<CpEntry> cps = DecodeCps(tokens_[static_cast<size_t>(id)]);
    for (const CpEntry& e : cps) {
      const int b = CodepointToByte(e.cp);
      if (b >= 0) {
        out.push_back(static_cast<char>(b));
      } else {
        // 不是字节编码字符：原样输出（例如特殊 token 里的 '|' 是直接映射，
        // 不会走到这里；真走到这里说明词表里有非字节编码的自定义 token）。
        AppendUtf8(&out, e.cp);
      }
    }
  }
  return out;
}

}  // namespace rpg
