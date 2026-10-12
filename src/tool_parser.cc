// RPGBarEngine —— 输出解析的实现。算法见头文件顶部的说明。

#include "tool_parser.h"

#include <cstddef>
#include <string>
#include <vector>

#include "common.h"

namespace rpg {
namespace {

constexpr const char* kThinkOpen = "<think>";
constexpr const char* kThinkClose = "</think>";
constexpr const char* kCallOpen = "<tool_call>";
constexpr const char* kCallClose = "</tool_call>";

// 去掉两端的 ASCII 空白。只用于"该说的话"与工具调用块的边界，
// 不用来动内容本身。
std::string TrimAscii(const std::string& s) {
  auto is_space = [](char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v';
  };
  size_t b = 0;
  size_t e = s.size();
  while (b < e && is_space(s[b])) ++b;
  while (e > b && is_space(s[e - 1])) --e;
  return s.substr(b, e - b);
}

// 解析出**不可用**的工具调用时记一条日志。
//
// 为什么不把原文（arguments_raw）放进响应里：**响应是给游戏的，日志是给排错的**。
// 宿主拿到 `arguments: null` 已经够它决定"这次调用不可用"；而 arguments_raw 是
// 模型吐坏 JSON 时的原始片段，只在排错时有用，且内容完全由模型自由发挥 —— 塞进
// 响应会把"引擎只解析、不判断合法性"的契约搅浑，也让每个宿主都得自己决定要不要
// 忽略它。钉在日志里，两边都干净。
//
// 默认级别（info）就能看到：定位"模型偶尔不按格式出牌"这类问题，靠的就是这条。
void LogInvalidCall(const ParsedToolCall& call) {
  if (call.name.empty() && call.arguments_raw.empty()) {
    RPG_LOG_WARN("工具调用不可用：<tool_call> 里没有可解析的 name / arguments");
    return;
  }
  if (call.name.empty()) {
    RPG_LOG_WARN("工具调用不可用：name 为空（参数原文 arguments_raw=%s）",
                 call.arguments_raw.c_str());
    return;
  }
  RPG_LOG_WARN(
      "工具调用不可用：name=\"%s\" 的 arguments 不是合法 JSON 对象"
      "（参数原文 arguments_raw=%s）",
      call.name.c_str(), call.arguments_raw.c_str());
}

// 从 `<tool_call>` 的内容里取 name / arguments。
// 体内容形如 {"name": "read_scene", "arguments": {"scene": "temple"}}。
ParsedToolCall ParseOneCall(const std::string& body) {
  ParsedToolCall call;
  std::string err;
  const Json j = Json::Parse(TrimAscii(body), &err);
  if (!j.IsObject()) {
    call.arguments_raw = TrimAscii(body);
    LogInvalidCall(call);
    return call;
  }
  call.name = j.Get("name").AsString("");

  const Json a = j.Get("arguments");
  if (a.IsString()) {
    call.arguments_raw = a.AsString();
    // 有的模型会把参数再包一层字符串（"{\"a\":1}"），这里顺手拆开。
    std::string inner_err;
    const Json inner = Json::Parse(call.arguments_raw, &inner_err);
    if (inner.IsObject()) call.arguments = inner;
  } else if (!a.IsNull()) {
    call.arguments_raw = a.Dump();
    call.arguments = a;
  }

  call.ok = !call.name.empty() && call.arguments.IsObject();
  if (!call.ok) LogInvalidCall(call);
  return call;
}

// 去掉控制类特殊 token。
//
// ⚠️ 这条是 M3 踩出来的：**`<tool_call>` / `</tool_call>` 本身就是特殊 token**
// （id 151657/151658），所以解码时若开了 skip_special，结构标记会跟着一起消失 ——
// 解析器就只能看到裸 JSON，还会把 JSON 当成旁白文本交给玩家。
// 因此契约是：**给解析器的是"保留特殊 token 的解码文本"**，由解析器在这里
// 把不该给玩家看的控制标记清掉。
//
// 名单里刻意**不含** `<tool_call>` / `<think>`：那两个是结构标记，要先被下面的
// 结构扫描消费掉，提前删掉就分不出"这段是工具调用"了。
void StripControlMarkers(std::string* s) {
  // <|im_start|> 后面跟的 role 词也一起吃掉：模型偶尔会把对话头照抄出来，
  // 只删标记会在文本里留下一个孤零零的 "assistant"。
  static const char* kRoles[] = {"system", "user", "assistant", "tool"};
  const std::string start = "<|im_start|>";
  const size_t start_len = start.size();
  for (size_t p = s->find(start); p != std::string::npos; p = s->find(start, p)) {
    size_t end = p + start_len;
    for (const char* role : kRoles) {
      const size_t rl = std::char_traits<char>::length(role);
      if (s->compare(end, rl, role) == 0 && (*s)[end + rl] == '\n') {
        end += rl + 1;
        break;
      }
    }
    s->erase(p, end - p);
  }
  for (const char* marker : {"<|im_end|>", "<|endoftext|>"}) {
    const size_t len = std::char_traits<char>::length(marker);
    for (size_t p = s->find(marker); p != std::string::npos;
         p = s->find(marker, p)) {
      s->erase(p, len);
    }
  }
}

}  // namespace

ParsedOutput ParseModelOutput(const std::string& raw) {
  ParsedOutput out;

  // ---- 第零步：清控制标记（结构标记留着，下一步要用）----
  std::string work = raw;
  StripControlMarkers(&work);

  // ---- 第一步：剥思考块 ----
  // 循环处理，因为模型可能吐出多段（极少数情况，但不值得为它写特例）。
  for (;;) {
    const size_t b = work.find(kThinkOpen);
    if (b == std::string::npos) break;
    const size_t body = b + std::char_traits<char>::length(kThinkOpen);
    const size_t e = work.find(kThinkClose, body);
    if (e == std::string::npos) {
      // 没有闭合：从 `<think>` 到结尾都算思考内容，文本到此为止。
      out.reasoning += work.substr(body);
      work.erase(b);
      out.unterminated_think = true;
      break;
    }
    out.reasoning += work.substr(body, e - body);
    work.erase(b, e + std::char_traits<char>::length(kThinkClose) - b);
  }

  // ---- 第二步：抽工具调用块，剩下的就是文本 ----
  std::string text;
  size_t pos = 0;
  for (;;) {
    const size_t b = work.find(kCallOpen, pos);
    if (b == std::string::npos) break;
    text.append(work, pos, b - pos);

    const size_t body = b + std::char_traits<char>::length(kCallOpen);
    const size_t e = work.find(kCallClose, body);
    if (e == std::string::npos) {
      // 截断在工具调用中间：仍然把能解析的部分交出去（宿主可以据此重试），
      // 但要打上标记。
      out.tool_calls.push_back(ParseOneCall(work.substr(body)));
      out.unterminated_tool_call = true;
      pos = work.size();
      break;
    }
    out.tool_calls.push_back(
        ParseOneCall(work.substr(body, e - body)));
    pos = e + std::char_traits<char>::length(kCallClose);
  }
  text.append(work, pos, std::string::npos);

  out.text = TrimAscii(text);
  return out;
}

}  // namespace rpg
