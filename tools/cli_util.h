// tools/cli_util.h —— 命令行工具共用的小件。
//
// 只有四件事，但每一件都踩过坑，所以集中放一处：
//   * ParseIds —— 逗号/空格分隔的 id 列表；
//   * ReadFile —— 按**字节**读。Windows 下 `cmd` / Git Bash 传给 argv 的
//     字符串会按控制台代码页转码，中文提示词当命令行参数传进去已经是乱码
//     （实测 "你好" 变一坨），所以中文一律走文件；
//   * WriteText / WriteIds —— 按字节写；
//   * SetupBinaryStdio —— Windows 上 stdin/stdout 默认文本模式，会把 \n
//     换成 \r\n，还会把 0x1A 当 EOF。走二进制协议的工具必须关掉。
#ifndef RPGBAR_TOOLS_CLI_UTIL_H_
#define RPGBAR_TOOLS_CLI_UTIL_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common.h"

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

namespace cli {

// 解析 "151936,1234 5678" 这类列表。空串 → 空表。
inline std::vector<int32_t> ParseIds(const std::string& s) {
  std::vector<int32_t> out;
  size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && (s[i] == ',' || s[i] == ' ' || s[i] == '\n' ||
                            s[i] == '\r' || s[i] == '\t')) {
      ++i;
    }
    if (i >= s.size()) break;
    char* end = nullptr;
    const long v = std::strtol(s.c_str() + i, &end, 10);
    if (end == s.c_str() + i) {
      rpg::Throw("id 列表里有解析不了的内容：%s", s.c_str() + i);
    }
    out.push_back(static_cast<int32_t>(v));
    i = static_cast<size_t>(end - s.c_str());
  }
  return out;
}

inline std::string FormatIds(const std::vector<int32_t>& ids) {
  std::string body;
  for (size_t i = 0; i < ids.size(); ++i) {
    body += std::to_string(ids[i]);
    body += (i + 1 == ids.size()) ? "\n" : ",";
  }
  return body;
}

inline bool WriteText(const std::string& path, const std::string& text) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) return false;
  const bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size();
  std::fclose(f);
  return ok;
}

inline void WriteIds(const std::string& path, const std::vector<int32_t>& ids) {
  RPG_CHECK(WriteText(path, FormatIds(ids)), "写不了 %s", path.c_str());
}

inline std::string ReadFile(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) rpg::Throw("读不了文件 %s", path.c_str());
  std::string s;
  char buf[4096];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
  std::fclose(f);
  return s;
}

// 从文件读 id 列表（对拍参考物常用这个格式）。
inline std::vector<int32_t> ReadIds(const std::string& path) {
  return ParseIds(ReadFile(path));
}

inline void SetupBinaryStdio() {
#if defined(_WIN32)
  _setmode(_fileno(stdin), _O_BINARY);
  _setmode(_fileno(stdout), _O_BINARY);
#endif
}

}  // namespace cli

#endif  // RPGBAR_TOOLS_CLI_UTIL_H_
