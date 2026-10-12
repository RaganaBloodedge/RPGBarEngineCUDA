// RPGBarEngine —— Qwen3 ChatML 渲染的实现
//
// 这个文件的每一行都对着官方 tokenizer_config.json 的 chat_template 写。
// 改这里的任何分支前，先跑 tools/check_chatml.py（它拿 HF 的官方渲染器逐字节比）。

#include "chat_template.h"

#include <cstddef>
#include <string>
#include <vector>

namespace rpg {
namespace {

// ---- Python 的 str.strip('\n') 系列：**只**处理 '\n'，不是所有空白 ----
// 官方模板里用的是 rstrip('\n') / lstrip('\n') / strip('\n')，用"去掉所有空白"
// 会连空格和制表符一起吃掉，结果与原版不同。

std::string RStripNewline(const std::string& s) {
  size_t n = s.size();
  while (n > 0 && s[n - 1] == '\n') --n;
  return s.substr(0, n);
}

std::string LStripNewline(const std::string& s) {
  size_t i = 0;
  while (i < s.size() && s[i] == '\n') ++i;
  return s.substr(i);
}

std::string StripNewline(const std::string& s) {
  return LStripNewline(RStripNewline(s));
}

bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Python 的 s.split(sep)[0]：第一个分隔符之前的部分；没有分隔符就是整串。
std::string BeforeFirst(const std::string& s, const std::string& sep) {
  const size_t p = s.find(sep);
  return p == std::string::npos ? s : s.substr(0, p);
}

// Python 的 s.split(sep)[-1]：最后一个分隔符之后的部分；没有分隔符就是整串。
std::string AfterLast(const std::string& s, const std::string& sep) {
  const size_t p = s.rfind(sep);
  return p == std::string::npos ? s : s.substr(p + sep.size());
}

const char kToolPreamble[] =
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>";

const char kToolEpilogue[] =
    "\n</tools>\n\n"
    "For each function call, return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
    "</tool_call><|im_end|>\n";

// Jinja 的"真值"判断：空数组 / null 为假。
bool JinjaTruthy(const Json& j) {
  switch (j.type()) {
    case Json::Type::kNull: return false;
    case Json::Type::kBool: return j.AsBool();
    case Json::Type::kNumber: return j.AsDouble() != 0.0;
    case Json::Type::kString: return !j.AsString().empty();
    case Json::Type::kArray: return !j.elements().empty();
    case Json::Type::kObject: return !j.items().empty();
  }
  return false;
}

}  // namespace

std::string RenderChatml(const std::vector<ChatMessage>& messages,
                         const Json& tools, const ChatTemplateOptions& opt) {
  const bool has_tools = JinjaTruthy(tools);
  const bool first_is_system = !messages.empty() && messages[0].role == "system";

  std::string out;
  out.reserve(512);

  // ---- 头部：system 消息（有工具时会把工具定义拼进同一条 system）----
  if (has_tools) {
    out += "<|im_start|>system\n";
    if (first_is_system) {
      out += messages[0].content;
      out += "\n\n";
    }
    out += kToolPreamble;
    for (const Json& t : tools.elements()) {
      out += "\n";
      // 这里必须是 Python 的 json.dumps 样式（", " 与 ": "），且中文不转义 ——
      // 官方模板用的是 Jinja 的 tojson 过滤器。
      out += t.DumpPython();
    }
    out += kToolEpilogue;
  } else if (first_is_system) {
    out += "<|im_start|>system\n";
    out += messages[0].content;
    out += "<|im_end|>\n";
  }

  // ---- last_query_index：最后一条"真实用户提问"的下标 ----
  // 从后往前找第一条 role=user 且不是整条 <tool_response> 包裹的消息。
  // 默认值是 messages 长度 - 1（也就是"没有任何真实提问"时没人满足条件）。
  int64_t last_query_index = static_cast<int64_t>(messages.size()) - 1;
  for (int64_t i = static_cast<int64_t>(messages.size()) - 1; i >= 0; --i) {
    const std::string& c = messages[static_cast<size_t>(i)].content;
    if (messages[static_cast<size_t>(i)].role == "user" &&
        !(StartsWith(c, "<tool_response>") && EndsWith(c, "</tool_response>"))) {
      last_query_index = i;
      break;
    }
  }

  const int64_t n = static_cast<int64_t>(messages.size());
  for (int64_t i = 0; i < n; ++i) {
    const ChatMessage& msg = messages[static_cast<size_t>(i)];
    std::string content = msg.content;

    if (msg.role == "user" || (msg.role == "system" && i != 0)) {
      out += "<|im_start|>";
      out += msg.role;
      out += "\n";
      out += content;
      out += "<|im_end|>\n";
      continue;
    }

    if (msg.role == "assistant") {
      // 思考内容：优先用显式给的 reasoning_content；否则从 content 里的
      // `</think>` 反拆（模型把整段思考+回答一起吐出来时的形态）。
      // 下面统一用 reasoning 的"真值"（非空）来做判断，这与官方模板对
      // reasoning_content 变量的判断等价。
      std::string reasoning;
      if (msg.has_reasoning) {
        reasoning = msg.reasoning_content;
      } else if (content.find("</think>") != std::string::npos) {
        reasoning = LStripNewline(
            AfterLast(RStripNewline(BeforeFirst(content, "</think>")), "<think>"));
        content = LStripNewline(AfterLast(content, "</think>"));
      }

      out += "<|im_start|>assistant\n";
      // 只有"最后一次真实提问之后"的 assistant 才渲染思考块；且只有当它是
      // 最后一条、或者它确实带了思考内容时 —— 与官方模板的两个条件一致。
      if (i > last_query_index &&
          (i == n - 1 || (i != n - 1 && !reasoning.empty()))) {
        out += "<think>\n";
        out += StripNewline(reasoning);
        out += "\n</think>\n\n";
        out += LStripNewline(content);
      } else {
        out += content;
      }

      if (!msg.tool_calls.empty()) {
        for (size_t k = 0; k < msg.tool_calls.size(); ++k) {
          if ((k == 0 && !content.empty()) || k != 0) out += "\n";
          out += "<tool_call>\n{\"name\": \"";
          out += msg.tool_calls[k].name;
          out += "\", \"arguments\": ";
          out += msg.tool_calls[k].arguments;
          out += "}\n</tool_call>";
        }
      }
      out += "<|im_end|>\n";
      continue;
    }

    if (msg.role == "tool") {
      // 连续的 tool 消息共用一条 `<|im_start|>user`，各自包一层 tool_response。
      const bool starts_group =
          i == 0 || messages[static_cast<size_t>(i - 1)].role != "tool";
      const bool ends_group =
          i == n - 1 || messages[static_cast<size_t>(i + 1)].role != "tool";
      if (starts_group) out += "<|im_start|>user";
      out += "\n<tool_response>\n";
      out += content;
      out += "\n</tool_response>";
      if (ends_group) out += "<|im_end|>\n";
      continue;
    }

    // 未知 role：官方模板静默跳过，我们保持一致。
  }

  if (opt.add_generation_prompt) {
    out += "<|im_start|>assistant\n";
    if (!opt.enable_thinking) {
      out += "<think>\n\n</think>\n\n";
    }
  }
  return out;
}

std::vector<ChatMessage> ParseMessages(const Json& arr) {
  std::vector<ChatMessage> out;
  if (!arr.IsArray()) return out;
  out.reserve(arr.elements().size());

  for (const Json& m : arr.elements()) {
    ChatMessage cm;
    cm.role = m.Get("role").AsString("");

    const Json c = m.Get("content");
    if (c.IsString()) cm.content = c.AsString();

    const Json rc = m.Get("reasoning_content");
    if (rc.IsString()) {
      cm.has_reasoning = true;
      cm.reasoning_content = rc.AsString();
    }

    const Json tcs = m.Get("tool_calls");
    if (tcs.IsArray()) {
      for (const Json& tc : tcs.elements()) {
        // 允许两种写法：{"function":{"name":…}} 与直接 {"name":…}。
        const Json fn = tc.Get("function");
        const Json& src = fn.IsObject() ? fn : tc;
        ToolCall call;
        call.name = src.Get("name").AsString("");
        const Json a = src.Get("arguments");
        // arguments 是字符串就原样用（模型输出的形态）；是对象就序列化。
        call.arguments = a.IsString() ? a.AsString() : a.DumpPython();
        cm.tool_calls.push_back(std::move(call));
      }
    }
    out.push_back(std::move(cm));
  }
  return out;
}

Json ParseTools(const Json& v) {
  return v.IsArray() ? v : Json::Array();
}

}  // namespace rpg
