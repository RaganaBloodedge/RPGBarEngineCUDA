// cuda_common.cc —— 与 CUDA 运行时**无关**的那部分：设备信息的格式化，以及
// host-only 构建下的兜底 QueryDevice。
//
// 为什么拆成 .cc + .cu 两个文件：`.cu` 只有装了 nvcc 才编得动。而 QueryDevice 的
// 兜底实现（"为什么没有 GPU"）恰恰是 host-only 构建里**最需要**的那条信息 ——
// 宿主宁愿看到"本产物未编译 CUDA 后端"，也不愿看到一个空的设备名。
// 所以把"不需要 CUDA 类型"的部分放这里，两条构建路径都编它。
#include "cuda_common.h"

#include <cstdio>
#include <string>

namespace rpg {

namespace {

// 设备名 / 原因文本要进 JSON，双引号与反斜杠先转义。
std::string JsonEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n";  break;
      case '\r': out += "\\r";  break;
      case '\t': out += "\\t";  break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

}  // namespace

#ifndef RPG_HAS_CUDA

// host-only 构建（没装 CUDA Toolkit）。存在的意义只有一条：让宿主拿到一句
// **能看懂**的话。真正的设备查询在 cuda_common.cu 里。
DeviceInfo QueryDevice(int) {
  DeviceInfo d;
  d.reason = "本产物未编译 CUDA 后端（构建时未检测到 nvcc）";
  return d;
}

#endif  // !RPG_HAS_CUDA

std::string DescribeDevice(const DeviceInfo& d) {
  if (!d.available) return "无可用 CUDA 设备：" + d.reason;
  char buf[256];
  std::snprintf(buf, sizeof(buf),
                "%s | 算力 %d.%d | %d SM | 显存 %.1f / %.1f GiB | CUDA 驱动 %s / 运行时 %s",
                d.name.c_str(), d.cc_major, d.cc_minor, d.sm_count,
                static_cast<double>(d.free_mem) / 1073741824.0,
                static_cast<double>(d.total_mem) / 1073741824.0,
                d.driver.empty() ? "?" : d.driver.c_str(),
                d.runtime.empty() ? "?" : d.runtime.c_str());
  return buf;
}

std::string DeviceJson(const DeviceInfo& d) {
  std::string out = "{\"available\":";
  out += d.available ? "true" : "false";
  out += ",\"name\":\"" + JsonEscape(d.name) + "\"";
  out += ",\"cc\":\"" + std::to_string(d.cc_major) + "." + std::to_string(d.cc_minor) + "\"";
  out += ",\"sm_count\":" + std::to_string(d.sm_count);
  out += ",\"total_mem\":" + std::to_string(d.total_mem);
  out += ",\"free_mem\":" + std::to_string(d.free_mem);
  out += ",\"driver\":\"" + JsonEscape(d.driver) + "\"";
  out += ",\"runtime\":\"" + JsonEscape(d.runtime) + "\"";
  if (!d.reason.empty()) out += ",\"reason\":\"" + JsonEscape(d.reason) + "\"";
  out += "}";
  return out;
}

}  // namespace rpg
