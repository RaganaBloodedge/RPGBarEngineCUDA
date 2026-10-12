// RPGBarEngine —— 基础设施：错误、日志、断言
//
// 错误策略：
//   * 引擎内部用异常传错误（RPG_THROW / RPG_CHECK），因为调用链深、需要中途返回。
//   * 异常**不得**穿出 C ABI 边界。engine.cc 在最外层 try/catch，把异常翻成
//     rpg_engine_last_error()，ABI 上只返回 NULL / 负数（见 include/rpg_engine.h）。
//
// 这不是"能省就省"的取舍：C ABI 一旦让 C++ 异常穿过 extern "C"，
// 宿主（Unity IL2CPP / CPython）那边是未定义行为。

#ifndef RPGBAR_COMMON_H_
#define RPGBAR_COMMON_H_

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace rpg {

class Error : public std::runtime_error {
 public:
  explicit Error(const std::string& msg) : std::runtime_error(msg) {}
};

// 线程局部的最近错误文本，供 rpg_engine_last_error() 读取。
void SetLastError(const std::string& msg);
const char* LastError();

// 格式化后抛出。[[noreturn]] 是给编译器看的：让它在"函数末尾抛异常"的分支
// 不再报 "control reaches end of non-void function"。__attribute__((format))
// 让 GCC/Clang 顺手检查占位符与实参是否匹配。
[[noreturn]]
#if defined(__GNUC__)
__attribute__((format(printf, 1, 2)))
#endif
void Throw(const char* fmt, ...);

// 日志：默认走 stderr，前缀固定，方便宿主把引擎日志和游戏日志分开过滤。
// RPGBAR_ENGINE_LOG=off 可整体静音。
enum class LogLevel { kDebug, kInfo, kWarn, kError };
bool LogEnabled(LogLevel lv);
void Log(LogLevel lv, const char* fmt, ...);

}  // namespace rpg

// size_t 的可移植打印：MinGW 的 printf 走 MSVCRT，**不支持 %zu**（会把它当成
// 非法占位符，参数错位后输出乱码，而且不报错）。所以统一走 %llu + 显式转换。
#define RPG_SZ "%llu"
#define RPG_SZ_ARG(x) static_cast<unsigned long long>(x)

// 断言式检查：失败即抛 Error（不区分 debug/release —— 引擎收到的输入来自文件与
// 宿主，任何一条不成立都意味着"再往下算就是错的"，早失败比静默算出垃圾好）。
#define RPG_CHECK(cond, ...)                              \
  do {                                                    \
    if (!(cond)) ::rpg::Throw(__VA_ARGS__);               \
  } while (0)

#define RPG_LOG_INFO(...) ::rpg::Log(::rpg::LogLevel::kInfo, __VA_ARGS__)
#define RPG_LOG_WARN(...) ::rpg::Log(::rpg::LogLevel::kWarn, __VA_ARGS__)
#define RPG_LOG_ERR(...) ::rpg::Log(::rpg::LogLevel::kError, __VA_ARGS__)

#endif  // RPGBAR_COMMON_H_
