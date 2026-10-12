#include "json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace rpg {
namespace {

const Json& NullJson() {
  static const Json kNull;
  return kNull;
}

void AppendUtf8(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// 递归下降解析器。就地为每个位置判断，不做 token 化（JSON 语法够简单）。
class Parser {
 public:
  Parser(const std::string& s) : s_(s) {}

  bool Parse(Json* out) {
    SkipWs();
    if (!ParseValue(out)) return false;
    SkipWs();
    if (i_ != s_.size()) return Fail("尾随字符");
    return true;
  }

  const std::string& error() const { return err_; }

 private:
  bool Fail(const char* what) {
    if (err_.empty()) {
      char buf[128];
      std::snprintf(buf, sizeof(buf), "%s（位置 %llu）", what,
                    static_cast<unsigned long long>(i_));
      err_ = buf;
    }
    return false;
  }

  void SkipWs() {
    while (i_ < s_.size()) {
      const char c = s_[i_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++i_;
      } else {
        break;
      }
    }
  }

  bool Literal(const char* lit) {
    const size_t n = std::strlen(lit);
    if (s_.compare(i_, n, lit) != 0) return false;
    i_ += n;
    return true;
  }

  bool ParseValue(Json* out) {
    if (i_ >= s_.size()) return Fail("预期一个值，却到末尾");
    switch (s_[i_]) {
      case '{': return ParseObject(out);
      case '[': return ParseArray(out);
      case '"': {
        std::string str;
        if (!ParseString(&str)) return false;
        *out = Json(std::move(str));
        return true;
      }
      case 't':
        if (!Literal("true")) return Fail("非法字面量");
        *out = Json(true);
        return true;
      case 'f':
        if (!Literal("false")) return Fail("非法字面量");
        *out = Json(false);
        return true;
      case 'n':
        if (!Literal("null")) return Fail("非法字面量");
        *out = Json();
        return true;
      default: return ParseNumber(out);
    }
  }

  bool ParseObject(Json* out) {
    ++i_;  // '{'
    Json obj = Json::Object();
    SkipWs();
    if (i_ < s_.size() && s_[i_] == '}') {
      ++i_;
      *out = std::move(obj);
      return true;
    }
    for (;;) {
      SkipWs();
      std::string key;
      if (i_ >= s_.size() || s_[i_] != '"') return Fail("对象键必须是字符串");
      if (!ParseString(&key)) return false;
      SkipWs();
      if (i_ >= s_.size() || s_[i_] != ':') return Fail("键后缺冒号");
      ++i_;
      SkipWs();
      Json val;
      if (!ParseValue(&val)) return false;
      obj.Set(key, std::move(val));
      SkipWs();
      if (i_ >= s_.size()) return Fail("对象未闭合");
      if (s_[i_] == ',') {
        ++i_;
        continue;
      }
      if (s_[i_] == '}') {
        ++i_;
        break;
      }
      return Fail("对象里预期 ',' 或 '}'");
    }
    *out = std::move(obj);
    return true;
  }

  bool ParseArray(Json* out) {
    ++i_;  // '['
    Json arr = Json::Array();
    SkipWs();
    if (i_ < s_.size() && s_[i_] == ']') {
      ++i_;
      *out = std::move(arr);
      return true;
    }
    for (;;) {
      SkipWs();
      Json val;
      if (!ParseValue(&val)) return false;
      arr.PushBack(std::move(val));
      SkipWs();
      if (i_ >= s_.size()) return Fail("数组未闭合");
      if (s_[i_] == ',') {
        ++i_;
        continue;
      }
      if (s_[i_] == ']') {
        ++i_;
        break;
      }
      return Fail("数组里预期 ',' 或 ']'");
    }
    *out = std::move(arr);
    return true;
  }

  bool ParseHex4(uint32_t* v) {
    if (i_ + 4 > s_.size()) return Fail("\\u 转义不完整");
    uint32_t r = 0;
    for (int k = 0; k < 4; ++k) {
      const char c = s_[i_ + static_cast<size_t>(k)];
      r <<= 4;
      if (c >= '0' && c <= '9') {
        r |= static_cast<uint32_t>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        r |= static_cast<uint32_t>(c - 'a' + 10);
      } else if (c >= 'A' && c <= 'F') {
        r |= static_cast<uint32_t>(c - 'A' + 10);
      } else {
        return Fail("非法十六进制");
      }
    }
    i_ += 4;
    *v = r;
    return true;
  }

  bool ParseString(std::string* out) {
    ++i_;  // 开引号
    out->clear();
    while (i_ < s_.size()) {
      const unsigned char c = static_cast<unsigned char>(s_[i_]);
      if (c == '"') {
        ++i_;
        return true;
      }
      if (c == '\\') {
        ++i_;
        if (i_ >= s_.size()) return Fail("转义不完整");
        const char e = s_[i_++];
        switch (e) {
          case '"': out->push_back('"'); break;
          case '\\': out->push_back('\\'); break;
          case '/': out->push_back('/'); break;
          case 'b': out->push_back('\b'); break;
          case 'f': out->push_back('\f'); break;
          case 'n': out->push_back('\n'); break;
          case 'r': out->push_back('\r'); break;
          case 't': out->push_back('\t'); break;
          case 'u': {
            uint32_t cp = 0;
            if (!ParseHex4(&cp)) return false;
            // 代理对：高位 D800-DBFF + 低位 DC00-DFFF 合成一个码点。
            if (cp >= 0xD800 && cp <= 0xDBFF) {
              if (i_ + 2 <= s_.size() && s_[i_] == '\\' && s_[i_ + 1] == 'u') {
                i_ += 2;
                uint32_t lo = 0;
                if (!ParseHex4(&lo)) return false;
                if (lo >= 0xDC00 && lo <= 0xDFFF) {
                  cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                } else {
                  // 孤立的高位代理：按 UTF-8 替换字符处理，不中断解析。
                  cp = 0xFFFD;
                }
              } else {
                cp = 0xFFFD;
              }
            } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
              cp = 0xFFFD;
            }
            AppendUtf8(*out, cp);
            break;
          }
          default: return Fail("未知转义");
        }
        continue;
      }
      if (c < 0x20) return Fail("字符串里出现未转义控制字符");
      out->push_back(static_cast<char>(c));
      ++i_;
    }
    return Fail("字符串未闭合");
  }

  bool ParseNumber(Json* out) {
    const size_t start = i_;
    if (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+')) ++i_;
    bool any = false;
    while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') {
      ++i_;
      any = true;
    }
    if (i_ < s_.size() && s_[i_] == '.') {
      ++i_;
      while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') {
        ++i_;
        any = true;
      }
    }
    if (any && i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
      ++i_;
      if (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+')) ++i_;
      while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
    }
    if (!any) return Fail("不是合法的数字");
    const std::string num = s_.substr(start, i_ - start);
    *out = Json(std::strtod(num.c_str(), nullptr));
    // 记住字面写法：`1` 与 `1.0` 值相同但字节不同，工具定义要原样回写。
    out->SetIntLiteral(num.find('.') == std::string::npos &&
                       num.find('e') == std::string::npos &&
                       num.find('E') == std::string::npos);
    return true;
  }

  const std::string& s_;
  size_t i_ = 0;
  std::string err_;
};

void DumpString(std::string& out, const std::string& s) {
  out.push_back('"');
  for (const unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  out.push_back('"');
}

// 最短的、能往返回来的小数表示 —— 就是 Python 的 repr(float) 那一套。
//
// 为什么不能用 "%.17g"：它给出的是"精确到 double 位数"的写法，
// 0.7 会变成 0.69999999999999996。合法，但没法与官方实现逐字节对齐，
// 而工具定义是要原样渲染进 prompt 的。
//
// 做法：从 1 位有效数字开始试，谁先能原样解析回来就用谁（double 最多 17 位
// 有效数字一定能往返，所以循环必然命中）。
std::string ShortestFloat(double v) {
  char buf[64] = {0};
  for (int p = 1; p <= 17; ++p) {
    std::snprintf(buf, sizeof(buf), "%.*g", p, v);
    if (std::strtod(buf, nullptr) == v) break;
  }
  std::string s = buf;
  // Python 里 float 一定有 '.' 或 'e'（1.0 而非 1），补上以免与整数混淆。
  const bool looks_float = s.find('.') != std::string::npos ||
                           s.find('e') != std::string::npos ||
                           s.find('E') != std::string::npos ||
                           s.find("inf") != std::string::npos ||
                           s.find("nan") != std::string::npos;
  if (!looks_float) s += ".0";
  return s;
}

void DumpNumber(std::string& out, const Json& j) {
  const double v = j.AsDouble();
  if (!std::isfinite(v)) {
    out += "null";  // JSON 没有 NaN / Inf
    return;
  }
  // 整数字面量就按整数写（"max_tokens": 256 而不是 256.0 或 2.56e2）。
  // 超出 int64 能表示的范围时才退回浮点写法。
  if (j.IsIntLiteral() && std::fabs(v) < 9.007199254740992e15) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(v));
    out += buf;
    return;
  }
  out += ShortestFloat(v);
}

void DumpValue(std::string& out, const Json& j, bool pretty, int indent,
               bool python_sep) {
  const auto newline_indent = [&](int n) {
    if (!pretty) return;
    out.push_back('\n');
    out.append(static_cast<size_t>(n) * 2, ' ');
  };

  switch (j.type()) {
    case Json::Type::kNull: out += "null"; break;
    case Json::Type::kBool: out += j.AsBool() ? "true" : "false"; break;
    case Json::Type::kNumber: DumpNumber(out, j); break;
    case Json::Type::kString: DumpString(out, j.AsString()); break;
    case Json::Type::kArray: {
      if (j.elements().empty()) {
        out += "[]";
        break;
      }
      out.push_back('[');
      bool first = true;
      for (const auto& e : j.elements()) {
        if (!first) out += python_sep ? ", " : ",";
        first = false;
        newline_indent(indent + 1);
        DumpValue(out, e, pretty, indent + 1, python_sep);
      }
      newline_indent(indent);
      out.push_back(']');
      break;
    }
    case Json::Type::kObject: {
      if (j.items().empty()) {
        out += "{}";
        break;
      }
      out.push_back('{');
      bool first = true;
      for (const auto& kv : j.items()) {
        if (!first) out += python_sep ? ", " : ",";
        first = false;
        newline_indent(indent + 1);
        DumpString(out, kv.first);
        out.push_back(':');
        if (pretty || python_sep) out.push_back(' ');
        DumpValue(out, kv.second, pretty, indent + 1, python_sep);
      }
      newline_indent(indent);
      out.push_back('}');
      break;
    }
  }
}

}  // namespace

int Json::Find(const std::string& key) const {
  for (size_t i = 0; i < obj_.size(); ++i) {
    if (obj_[i].first == key) return static_cast<int>(i);
  }
  return -1;
}

const Json& Json::operator[](const std::string& key) const {
  const int i = Find(key);
  return i >= 0 ? obj_[static_cast<size_t>(i)].second : NullJson();
}

Json& Json::operator[](const std::string& key) {
  const int i = Find(key);
  if (i >= 0) return obj_[static_cast<size_t>(i)].second;
  if (type_ != Type::kObject) {
    type_ = Type::kObject;
    obj_.clear();
  }
  obj_.emplace_back(key, Json());
  return obj_.back().second;
}

void Json::Set(const std::string& key, Json v) {
  const int i = Find(key);
  if (i >= 0) {
    obj_[static_cast<size_t>(i)].second = std::move(v);
    return;
  }
  if (type_ != Type::kObject) {
    type_ = Type::kObject;
    obj_.clear();
  }
  obj_.emplace_back(key, std::move(v));
}

std::string Json::Dump(bool pretty, int indent) const {
  std::string out;
  out.reserve(256);
  DumpValue(out, *this, pretty, indent, /*python_sep=*/false);
  return out;
}

std::string Json::DumpPython() const {
  std::string out;
  out.reserve(256);
  DumpValue(out, *this, /*pretty=*/false, /*indent=*/0, /*python_sep=*/true);
  return out;
}

std::string Json::EscapeString(const std::string& s) {
  std::string out;
  DumpString(out, s);
  return out;
}

Json Json::Parse(const std::string& text, std::string* err) {
  Json out;
  std::string local;
  if (!ParseInto(text, &out, &local)) {
    if (err != nullptr) *err = local;
    return Json();
  }
  return out;
}

bool Json::ParseInto(const std::string& text, Json* out, std::string* err) {
  Parser p(text);
  Json v;
  if (!p.Parse(&v)) {
    if (err != nullptr) *err = p.error();
    return false;
  }
  *out = std::move(v);
  return true;
}

}  // namespace rpg
