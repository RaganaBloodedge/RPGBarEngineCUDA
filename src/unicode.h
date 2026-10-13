// RPGBarEngine —— Unicode 小工具
//
// 只做三件事，但每件都必须精确：
//   1. UTF-8 编解码（不依赖 locale、不用 std::wstring_convert）；
//   2. 通用类别判断：\\p{L} / \\p{N} / \\s —— 范围表由 tools/gen_unicode_tables.py
//      从 UCD 生成（见该脚本注释里"为什么不手写近似"）；
//   3. GPT-2 的字节 ↔ 码点映射（byte-level BPE 的地基）。
//
// 关于 GPT-2 的 bytes_to_unicode：它把 256 个字节双射到 256 个"可见"码点，
// 让 控制字符（0x00-0x20）、空格（0x20→U+0120 'Ġ'）、以及 0x7F-0xA0/0xAD
// 这些在文本里难以表达或会被编辑器吃掉的字节，都有稳定的字符表示。
// 词表里存的正是这些映射后的字符，所以编解码两侧必须严格互逆。

#ifndef RPGBAR_UNICODE_H_
#define RPGBAR_UNICODE_H_

#include <cstdint>
#include <string>

#include "unicode_tables.h"

namespace rpg {

// ---- 范围表二分查找 ----
inline bool InRanges(const CodeRange* rs, size_t n, uint32_t cp) {
  size_t lo = 0;
  size_t hi = n;
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    if (cp < rs[mid].lo) {
      hi = mid;
    } else if (cp > rs[mid].hi) {
      lo = mid + 1;
    } else {
      return true;
    }
  }
  return false;
}

inline bool IsUnicodeLetter(uint32_t cp) {
  return InRanges(kLetterRanges, sizeof(kLetterRanges) / sizeof(CodeRange), cp);
}

inline bool IsUnicodeNumber(uint32_t cp) {
  return InRanges(kNumberRanges, sizeof(kNumberRanges) / sizeof(CodeRange), cp);
}

inline bool IsUnicodeSpace(uint32_t cp) {
  return InRanges(kSpaceRanges, sizeof(kSpaceRanges) / sizeof(CodeRange), cp);
}

// ---- UTF-8 ----

// 解一个码点。返回消耗的字节数（1..4）；输入非法时返回 1 并把 cp 设为 0xFFFD。
struct Utf8Char {
  uint32_t cp;
  int len;
};

inline Utf8Char Utf8Decode(const char* s, size_t remaining) {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(s);
  if (remaining == 0) return Utf8Char{0, 0};
  const uint8_t b0 = p[0];
  if (b0 < 0x80) return Utf8Char{b0, 1};

  int len = 0;
  uint32_t cp = 0;
  if ((b0 & 0xE0) == 0xC0) {
    len = 2;
    cp = b0 & 0x1Fu;
  } else if ((b0 & 0xF0) == 0xE0) {
    len = 3;
    cp = b0 & 0x0Fu;
  } else if ((b0 & 0xF8) == 0xF0) {
    len = 4;
    cp = b0 & 0x07u;
  } else {
    return Utf8Char{0xFFFD, 1};  // 非法起始字节
  }
  if (remaining < static_cast<size_t>(len)) return Utf8Char{0xFFFD, 1};

  for (int i = 1; i < len; ++i) {
    // 续字节必须是 10xxxxxx。
    if ((p[i] & 0xC0) != 0x80) return Utf8Char{0xFFFD, 1};
    cp = (cp << 6) | (p[i] & 0x3Fu);
  }
  // 拒绝过长编码（安全考虑，也避免同一个码点有两种字节表示）。
  if ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) ||
      (len == 4 && cp < 0x10000)) {
    return Utf8Char{0xFFFD, 1};
  }
  if (cp >= 0xD800 && cp <= 0xDFFF) return Utf8Char{0xFFFD, 1};
  return Utf8Char{cp, len};
}

// 把一个码点写成 UTF-8，返回写入字节数（1..4）。
inline int Utf8Encode(uint32_t cp, char* out) {
  if (cp < 0x80) {
    out[0] = static_cast<char>(cp);
    return 1;
  }
  if (cp < 0x800) {
    out[0] = static_cast<char>(0xC0 | (cp >> 6));
    out[1] = static_cast<char>(0x80 | (cp & 0x3F));
    return 2;
  }
  if (cp < 0x10000) {
    out[0] = static_cast<char>(0xE0 | (cp >> 12));
    out[1] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out[2] = static_cast<char>(0x80 | (cp & 0x3F));
    return 3;
  }
  out[0] = static_cast<char>(0xF0 | (cp >> 18));
  out[1] = static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
  out[2] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
  out[3] = static_cast<char>(0x80 | (cp & 0x3F));
  return 4;
}

inline void AppendUtf8(std::string* out, uint32_t cp) {
  char buf[4];
  const int n = Utf8Encode(cp, buf);
  out->append(buf, static_cast<size_t>(n));
}

// ---- GPT-2 字节 ↔ 码点 ----

// 该字节是否"直接映射"（字节值 == 码点值）。
inline bool ByteIsDirect(uint8_t b) {
  // 三段：'!'..'~'、'¡'..'¬'、'®'..'ÿ'。写成 b >= 174 而不是 b <= 255，
  // 否则编译器会提醒"uint8_t 的 <= 255 恒真"。
  return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
}

inline uint32_t ByteToCodepoint(uint8_t b) {
  if (ByteIsDirect(b)) return b;
  // 剩下的 68 个字节按原顺序接在 256 之后。
  uint32_t n = 0;
  if (b <= 32) {
    n = b;  // 0..32 -> 0..32
  } else if (b <= 160) {
    n = 33 + static_cast<uint32_t>(b - 127);  // 127..160 -> 33..66
  } else {
    n = 67;  // b == 173
  }
  return 256 + n;
}

// 反向：码点 -> 字节。不是本映射表里的码点返回 -1。
inline int CodepointToByte(uint32_t cp) {
  if (cp <= 255) {
    return ByteIsDirect(static_cast<uint8_t>(cp)) ? static_cast<int>(cp) : -1;
  }
  if (cp > 323) return -1;  // 映射表最大码点 = 256 + 67
  const uint32_t n = cp - 256;
  if (n < 33) return static_cast<int>(n);          // 原字节 0..32
  if (n < 67) return static_cast<int>(127 + (n - 33));  // 原字节 127..160
  return 173;                                      // 原字节 173
}

}  // namespace rpg

#endif  // RPGBAR_UNICODE_H_
