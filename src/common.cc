#include "common.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace rpg {
namespace {

thread_local std::string g_last_error;

const char* LevelName(LogLevel lv) {
  switch (lv) {
    case LogLevel::kDebug: return "debug";
    case LogLevel::kInfo: return "info";
    case LogLevel::kWarn: return "warn";
    case LogLevel::kError: return "error";
  }
  return "?";
}

bool EnvIsOff(const char* name) {
  const char* v = std::getenv(name);
  if (v == nullptr) return false;
  return std::strcmp(v, "off") == 0 || std::strcmp(v, "0") == 0 ||
         std::strcmp(v, "false") == 0;
}

LogLevel MinLevel() {
  static const LogLevel lv = [] {
    const char* v = std::getenv("RPGBAR_ENGINE_LOG");
    if (v == nullptr) return LogLevel::kInfo;
    std::string s(v);
    for (char& c : s) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (s == "debug") return LogLevel::kDebug;
    if (s == "warn") return LogLevel::kWarn;
    if (s == "error") return LogLevel::kError;
    if (s == "off") return LogLevel::kError;  // 由 LogEnabled 单独拦
    return LogLevel::kInfo;
  }();
  return lv;
}

std::mutex& LogMutex() {
  static std::mutex m;
  return m;
}

}  // namespace

void SetLastError(const std::string& msg) { g_last_error = msg; }

const char* LastError() { return g_last_error.c_str(); }

void Throw(const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  SetLastError(buf);
  throw Error(std::string(buf));
}

bool LogEnabled(LogLevel lv) {
  if (EnvIsOff("RPGBAR_ENGINE_LOG")) return false;
  return static_cast<int>(lv) >= static_cast<int>(MinLevel());
}

void Log(LogLevel lv, const char* fmt, ...) {
  if (!LogEnabled(lv)) return;
  char buf[2048];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  // 日志本身要线程安全：多个 session 可能并行生成。
  std::lock_guard<std::mutex> lk(LogMutex());
  std::fprintf(stderr, "[rpg_engine %s] %s\n", LevelName(lv), buf);
  std::fflush(stderr);
}

}  // namespace rpg
