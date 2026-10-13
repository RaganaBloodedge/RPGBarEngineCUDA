// RPGBarEngine —— Qwen3 的 ChatML 模板（渲染方向）
//
// 只做 Qwen3 这一个模板，**不引入 Jinja2**：宿主的工具定义、消息历史要渲染成
// 一段确定的字符串，模板本身是固定的，引一个通用模板引擎只会多一份依赖与
// 一类"模板写错"的运行时错误。
//
// 权威依据是官方 tokenizer_config.json 里的 chat_template 字段（纯 Jinja 字符串），
// 我们逐条复刻它的分支。对拍手段：用 HF 的 tokenizer.apply_chat_template 渲染
// 同一份 messages/tools，**逐字节**比 —— 见 tools/check_chatml.py。
//
// 三个容易写错、且"错了不崩只是结果不同"的地方，都在这里点明：
//
//   1. **空思考块是 `<think>\n\n</think>\n\n`**（不是 ` thinking`）。Qwen3 的特殊
//      token 就叫 `<think>` / `</think>`，没有前导空格。enable_thinking=false 时
//      它被放在 assistant 头部，让模型直接开始回答。
//   2. **`last_query_index`**：官方模板只给"最后一次真实用户提问**之后**"的
//      assistant 消息渲染 `<think>` 块 —— 这是给"带工具结果的多轮"设计的。
//      判断"真实提问"的标准：role=user 且内容不是 `<tool_response>…</tool_response>`
//      整个包起来的。
//   3. **连续的 tool 消息合并成一条 user 消息**，每条内容各包一层
//      `<tool_response>…</tool_response>`。
//
// 已知差异：官方模板里的 `message.content is string` 判断我们用 C++ 类型天然满足
// （content 一律是字符串）；非字符串 content（多模态数组）在引擎里不支持 —— 本
// 引擎只处理文本，传进来也会被当成空串，这与官方模板对非字符串的处理一致。

#ifndef RPGBAR_CHAT_TEMPLATE_H_
#define RPGBAR_CHAT_TEMPLATE_H_

#include <string>
#include <vector>

#include "json.h"

namespace rpg {

// 一次工具调用。arguments 用 **JSON 文本**保存而不是 Json 对象：
// 回写进 prompt 时要与模型当时的输出逐字节一致，重新序列化会改数字写法。
struct ToolCall {
  std::string name;
  std::string arguments = "{}";
};

// 一条消息。字段按官方模板真正用到的来，不多不少。
struct ChatMessage {
  std::string role;  // system / user / assistant / tool
  std::string content;

  // 「给了 reasoning_content」与「给了空串」在官方模板里是两条分支
  // （它用 `message.reasoning_content is string` 判断），所以要分开记。
  bool has_reasoning = false;
  std::string reasoning_content;

  std::vector<ToolCall> tool_calls;
};

struct ChatTemplateOptions {
  // 末尾补 `<|im_start|>assistant\n`，让模型从这里开始续写。
  bool add_generation_prompt = true;

  // 只有显式设为 false 才补空思考块（`<think>\n\n</think>\n\n`）。
  // 之所以默认 true：官方模板写的是 `enable_thinking is defined and enable_thinking is false`，
  // 不传这个变量时不补 —— 也就是"默认思考模式"，与 Qwen3 的默认行为一致。
  bool enable_thinking = true;
};

// 渲染 ChatML。tools 传 null 或空数组表示"这次不给工具"（走无工具分支）。
std::string RenderChatml(const std::vector<ChatMessage>& messages,
                         const Json& tools,
                         const ChatTemplateOptions& opt = ChatTemplateOptions());

// 把 C ABI 收到的 JSON 解析成消息列表 / 工具列表。
// 容错取向：字段缺失或类型不对时按官方模板的语义落默认值（content 缺省空串、
// tool_calls 缺省空数组），不抛异常 —— 宿主传一份"有点小毛病"的请求时，
// 我们希望它降级成"少带了点东西"，而不是整轮失败。
std::vector<ChatMessage> ParseMessages(const Json& arr);
Json ParseTools(const Json& v);

}  // namespace rpg

#endif  // RPGBAR_CHAT_TEMPLATE_H_
