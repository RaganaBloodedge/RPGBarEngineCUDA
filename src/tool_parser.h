// RPGBarEngine —— 模型输出解析（`<tool_call>` / `<think>`）
//
// 引擎**只解析、不执行**：工具是否允许、参数是否合法、状态怎么改，全在宿主。
// 这里只负责把模型吐出来的那段文本切成三样东西：
//   * text       —— 该说的话（思考块与工具调用块都剥掉）
//   * reasoning  —— 思考块内容（引擎内部保留，用于把思考块从 text 里剥干净）
//   * tool_calls —— 结构化的调用（可能 0 个或多个）
//
// ⚠️ reasoning **不下发到 C ABI 响应**（响应只有 text / tool_calls /
// finish_reason / usage 四键）。理由是防剧透：DM 的思考里常常写着它对剧本
// 下一步的打算，而响应最终会流到玩家侧 UI。小助手更是本来就该与玩家同视角。
// 想调 DM 时想看它在想什么 —— 那是**调试需求，走日志**，不走响应。
//
// 参数非法的工具调用（模型吐了坏 JSON、漏了 name）同样**不把原文塞进响应**：
// 响应里那一条给 `arguments: null`，原文（arguments_raw）记进日志供排错。
//
// 为什么不用"找第一个 `</think>` 就切"这种简写：模型在思考模式下的输出形态很活，
// 可能完全没有思考块（non-thinking），可能有；工具调用块可能一个、可能多个，
// 也可能因为 max_tokens 用尽而被截断在中间。这些都要能降级处理而不是抛异常 ——
// 一次生成失败不该让宿主整轮崩掉。
//
// 调用方约定：传进来的应是**保留特殊 token 的解码文本**，也就是解码时
// `skip_special = false`。
//
// ⚠️ 这条不是随便定的，是踩出来的：**`<tool_call>` / `</tool_call>` 自己就是
// 特殊 token**（id 151657/151658）。若解码时开了 skip_special，这两个标记会
// 跟着一起被删掉 —— 解析器就只能看到裸 JSON，还会把 JSON 当成旁白念给玩家听。
// 所以"清掉不该给玩家看的标记"这件事放在这里做（见 .cc 的 StripControlMarkers），
// 而不是靠解码开关。

#ifndef RPGBAR_TOOL_PARSER_H_
#define RPGBAR_TOOL_PARSER_H_

#include <string>
#include <vector>

#include "json.h"

namespace rpg {

struct ParsedToolCall {
  std::string name;
  // 参数的 JSON 文本。arguments 字段在模型输出里是字符串时原样保留；
  // 是对象时按紧凑样式重新序列化。
  std::string arguments_raw;
  // 解析出来的参数对象。解析失败或不是对象时是 null。
  Json arguments;
  // name 非空且 arguments 是合法对象 —— 宿主可以直接拿去执行。
  bool ok = false;
};

struct ParsedOutput {
  // 剥离思考块与工具调用块之后的文本，两端空白已去掉。
  std::string text;
  // 所有思考块的内容拼接（没有则空串）。
  std::string reasoning;
  std::vector<ParsedToolCall> tool_calls;

  // 降级标记：说明这次输出被截断了。宿主据此可以决定"重试"或"当作失败"，
  // 而不是拿着半截结果当正常输出用。
  bool unterminated_think = false;
  bool unterminated_tool_call = false;
};

ParsedOutput ParseModelOutput(const std::string& raw);

}  // namespace rpg

#endif  // RPGBAR_TOOL_PARSER_H_
