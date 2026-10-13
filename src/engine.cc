/* engine.cc —— C ABI 的实现（P0 骨架 → P1/P2 前向与 KV → **P3 接入**）。
 *
 * 这里是引擎与宿主的唯一交界面。三条贯穿全文件的约定（与 CPU 版逐条对齐）：
 *
 * 1. **异常绝不穿过 extern "C"**。引擎内部用异常传错误（见 src/common.h），
 *    但宿主是 Unity IL2CPP 与 CPython —— 让 C++ 异常穿出 C ABI 到那边是未定义
 *    行为。每个入口都包 try/catch，翻成 "返回负数 / NULL + rpg_engine_last_error()"。
 *    ⚠️ CUDA 的错误码走同一条路：RPG_CUDA_CHECK 抛 rpg::Error，不在这里出现
 *    cudaError_t。
 *
 * 2. **句柄只增不复用**。句柄是自增整数，销毁后永不回收 —— 于是宿主拿旧句柄
 *    再调用时一定失败（而不是像复用的 fd 那样打到另一个对象上，那种错最难查）。
 *
 * 3. **本层不做业务判断**。工具允不允许、参数合不合法、状态怎么改，全在宿主
 *    （对齐项目里"护栏做在工具体里"那条纪律）。这里只做：解析请求 → 渲染 →
 *    分词 → 前向/生成 → 解析输出 → 拼响应。
 *
 * ## P3：generate 真的实现了
 *
 * P0 时代这个文件里那句"**`generate` 还没有实现** —— 它会明确报错，而不是返回
 * 一段假文本"**已经不成立**：既然整网前向（P1）、fp16 KV Cache 与增量解码（P2）、
 * 设备端采样（P3 第一/二步）都各自有判据且全绿，接进 ABI 就是把它们缝起来。
 *
 * 现在的分工是：
 *   load_model     —— 读元数据 + **权重上卡**（1.03 GiB，量化字节原样上传）
 *   session_create —— 分配该会话的 KV Cache（4096 长度 @1.7B 是 448 MiB）
 *   generate       —— 渲染 ChatML → 分词 → **与 cached_ids 对最长公共前缀** →
 *                     只对新增部分做前向 → 采样生成 → 解码 → 解析 → 拼响应
 *
 * 与 CPU 版 `src/engine.cc` 是**逐条对齐**的：同样的九步、同样的不变量
 * （cache 长度 == prompt + 生成数；失败检查全部前置；logits 不做 host 往返）。
 * 差异只在"算在哪"：CPU 版把 logits 下载回 host 再取最后一行，这里**整个 decode
 * 循环一次 607 KB 的搬运都没有** —— `GpuGenerate` 直接吃显存里的那一行。
 *
 * 解码**不加 skip_special**：`<tool_call>` / `</tool_call>` / `<think>` 自己就是
 * 特殊 token，跳过它们会把结构标记一起删掉（CPU 版 DESIGN §16.4）。
 */

#include "rpg_engine.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "chat_template.h"
#include "common.h"
#include "cuda_common.h"
#include "gguf.h"
#include "json.h"
#include "model_meta.h"
#include "tool_parser.h"
#include "tokenizer.h"

// 显存侧的组件只在这三种构建里有定义（它们的头文件整体被 #ifdef 包住）。
#ifdef RPG_HAS_CUDA
#include "gpu_model.h"
#include "kv_cache_gpu.h"
#include "qwen3_gpu.h"
// SampleOptions 的语义是**契约**（同源副本 sampler.h）—— qwen3_gpu.h 已经把它
// 带进来了，这里再写一次是为了让"engine.cc 直接用到 SampleOptions"在 include
// 列表里看得见。⚠️ 必须在 namespace 之外：sampler.h 自带 `namespace rpg`。
#include "sampler.h"
#endif

namespace {

// 引擎版本。与 CMakeLists.txt 的 project(VERSION) 保持一致。
constexpr const char* kEngineVersion = "0.1.0";

// 请求/会话配置里没给 max_tokens 时的默认值（与 CPU 版一致）。
constexpr int32_t kDefaultMaxTokens = 256;
// KV Cache 里给停止标记与端点检查留的余量（与 CPU 版同值）。
constexpr int64_t kSeqMargin = 4;

// 采样默认（与 CPU 版一致：top_k 关、top_p 0.95）。
constexpr int32_t kDefaultTopK = 0;
constexpr float kDefaultTopP = 0.95f;
constexpr float kMaxTemperature = 2.0f;

// ---------------------------------------------------------------- 注册表

struct Session {
  int32_t model = -1;

#ifdef RPG_HAS_CUDA
  // 会话的显存状态。KV Cache 是一份会话最大的显存占用（4096 长度 @1.7B 是
  // 448 MiB），所以它跟着会话走、会话销毁即释放；`scratch` 只增不减，同一
  // 会话反复生成时白捡（见 GpuScratch 的头注释）。
  //
  // 两者都**只在 CUDA 构建里存在**：没有 nvcc 的构建要能照常编译、照常读
  // 元数据（P0 就定下的"CI 里跑得起来"），那种构建下 generate 会明确报错。
  std::unique_ptr<rpg::GpuKvCache> kv;
  rpg::GpuScratch scratch;
#endif

  // **cache 里当前实际存着的 token 序列**。generate 靠它做最长公共前缀匹配：
  // 只有长度与内容都逐一对上才能复用前缀。这里和 `kv->len()` 必须始终相等，
  // 不一致意味着"复用了错误的 K/V"—— 那种错不崩，只让输出悄悄变味。
  std::vector<int32_t> cached_ids;

  int32_t max_tokens = kDefaultMaxTokens;
  std::vector<std::string> stop;
  float temperature = 0.0f;
  uint64_t seed = 0;
  int32_t max_seq = 0;
};

struct ModelEntry {
  rpg::GgufFile gguf;

  // 形状与超参：**由 ReadModelMeta 统一推导**（src/model_meta.h 里写了为什么
  // 只留一处 —— 两处推导 head_dim 这类字段会静默算错）。
  rpg::ModelMeta meta;

  // RoPE 表覆盖的位置数（已按 context_length 截过）、新建会话的 KV 容量。
  int32_t max_pos = 0;
  int32_t default_max_seq = 0;

#ifdef RPG_HAS_CUDA
  // 权重上卡（P3）。与 CPU 版最大的差别在这一行：那边 `Model` 里的
  // QuantMatrix 是**指向 mmap 的视图**（零拷贝），所以 load 只要几毫秒；
  // 这里必须把 1.03 GiB 量化字节真的搬进显存，load 变成"几秒"量级。
  //
  // 设备不可用时它是 nullptr（只读到元数据），此时建会话会明确报错而不是
  // 悄悄退化 —— 见 session_create。
  std::unique_ptr<rpg::GpuModel> gpu;
#endif

  std::vector<int32_t> sessions;  // 便于 unload 时级联销毁
};

std::mutex g_mutex;
std::unordered_map<int32_t, ModelEntry> g_models;
std::unordered_map<int32_t, std::unique_ptr<Session>> g_sessions;
int32_t g_next_model = 1;
int32_t g_next_session = 1;

// ------- 下面两个 *Locked 版本要求调用方已持 g_mutex（g_mutex 不是递归锁）
ModelEntry* ModelLocked(int32_t h) {
  const auto it = g_models.find(h);
  RPG_CHECK(it != g_models.end(), "无效的模型句柄 %d（可能已被 unload）", h);
  return &it->second;
}

Session* SessionLocked(int32_t h) {
  const auto it = g_sessions.find(h);
  RPG_CHECK(it != g_sessions.end(), "无效的会话句柄 %d（可能已被 destroy）", h);
  return it->second.get();
}

ModelEntry* RequireModel(int32_t h) {
  std::lock_guard<std::mutex> lk(g_mutex);
  return ModelLocked(h);
}

Session* RequireSession(int32_t h) {
  std::lock_guard<std::mutex> lk(g_mutex);
  return SessionLocked(h);
}

// ---------------------------------------------------------------- 小工具

// 引擎内部 new[] / delete[] 成对出现（rpg_engine_free 也在本文件），跨 dll
// 边界传出去的字符串不会落到另一个 CRT 的堆上。
char* DupForAbi(const std::string& s) {
  char* buf = new char[s.size() + 1];
  std::memcpy(buf, s.data(), s.size());
  buf[s.size()] = '\0';
  return buf;
}

std::string JsonToAbiString(const rpg::Json& j) { return j.Dump(/*pretty=*/false); }

// 把一份配置 JSON（可为空）解析成对象。空串/空指针 = 全用默认值。
rpg::Json ParseConfigOrEmpty(const char* text, const char* what) {
  if (text == nullptr || *text == '\0') return rpg::Json::Object();
  std::string err;
  rpg::Json j = rpg::Json::Parse(text, &err);
  RPG_CHECK(err.empty(), "%s 不是合法 JSON：%s", what, err.c_str());
  RPG_CHECK(j.IsObject(), "%s 必须是 JSON 对象", what);
  return j;
}

// 停止 token 的来源有三处：模型自带 eos、`<|im_end|>`、以及请求/会话里给的
// 停止串。后两者里的字符串必须先变成 token id —— 引擎按 token 粒度停，
// 不做「回看最近 N 个 token 的文本匹配」，那是宿主的事（与 CPU 版逐字同义）。
void AppendStopString(const rpg::Tokenizer& tok, const std::string& s,
                      std::vector<int32_t>* out) {
  if (s.empty()) return;
  const int32_t direct = tok.IdOf(s);
  if (direct >= 0) {
    out->push_back(direct);
    return;
  }
  const std::vector<int32_t> ids = tok.Encode(s, /*parse_special=*/false);
  if (ids.size() == 1) {
    out->push_back(ids[0]);
    return;
  }
  RPG_LOG_WARN("停止串「%s」不是单个 token（编成 " RPG_SZ
               " 个），已忽略 —— 多 token 的停止串需要文本级回看，属宿主职责",
               s.c_str(), RPG_SZ_ARG(ids.size()));
}

void AppendStopStrings(const rpg::Tokenizer& tok, const rpg::Json& arr,
                       std::vector<int32_t>* out) {
  if (!arr.IsArray()) return;
  for (const rpg::Json& e : arr.elements()) {
    if (e.IsString()) AppendStopString(tok, e.AsString(), out);
  }
}

bool ContainsId(const std::vector<int32_t>& v, int32_t id) {
  return std::find(v.begin(), v.end(), id) != v.end();
}

}  // namespace

// ABI 边界的统一收口：把异常翻成"调用方看得懂的返回码"。
//
// 写法说明：宏**从 `catch` 开始**，不含闭合 try 块的那个 `}` —— 于是调用处是
//     try { ...; return x; }
//     RPG_ABI_CATCH(-1)
//   }
// 展开后正好是 `try {...} catch {...} catch {...}`。如果让宏自带那个 `}`，
// 调用处就得省掉它，读起来会像"少了个括号"。
#define RPG_ABI_CATCH(ret)                                 \
  catch (const std::exception& e) {                        \
    rpg::SetLastError(e.what());                           \
    return ret;                                            \
  }                                                        \
  catch (...) {                                            \
    rpg::SetLastError("未知异常（不是 std::exception）");    \
    return ret;                                            \
  }

extern "C" {

const char* rpg_engine_version(void) { return kEngineVersion; }

const char* rpg_engine_last_error(void) { return rpg::LastError(); }

int32_t rpg_engine_load_model(const char* model_path, const char* config_json) {
  try {
    rpg::SetLastError("");
    RPG_CHECK(model_path != nullptr && *model_path != '\0', "model_path 为空");
    const rpg::Json cfg = ParseConfigOrEmpty(config_json, "模型配置");

    auto entry = std::make_unique<ModelEntry>();
    entry->gguf = rpg::GgufFile::Open(model_path);
    RPG_CHECK(entry->gguf.valid(), "GGUF 打不开：%s", model_path);

    const rpg::GgufFile& g = entry->gguf;
    // 形状/超参只在这里推一次（见 model_meta.h）。
    entry->meta = rpg::ReadModelMeta(g);
    const rpg::ModelMeta& mm = entry->meta;

    // 上下文长度：配置可覆盖，且不能超过 GGUF 里声明的 context_length。
    int32_t ctx = mm.context_length;
    if (cfg.Has("context")) ctx = static_cast<int32_t>(cfg["context"].AsInt(ctx));
    if (ctx > mm.context_length) ctx = mm.context_length;
    if (ctx <= 0) ctx = mm.context_length;
    entry->max_pos = ctx;
    entry->default_max_seq = ctx;

    // 顺序是有意的：先把形状打出来（快），再上卡（几秒），日志里能看出卡在哪一步。
    RPG_LOG_INFO("已加载 %s（%d 层 / hidden %d / %d 头 / %d KV 头 / 词表 %d / 位置 %d）",
                 mm.name.c_str(), mm.n_layers, mm.hidden, mm.n_heads,
                 mm.n_kv_heads, mm.vocab, entry->max_pos);

#ifdef RPG_HAS_CUDA
    // 权重上卡（P3）。设备**完全不可用**时降级成"只读元数据" —— P0 就定下的那条
    // （CI 与没 GPU 的机器要能 load 起来、能读 model_info）。但设备可用而上传失败
    // （显存不足、未知量化类型……）**必须抛**：那是真错误，静默降级会让宿主以为
    // "加载成功了"，然后在 generate 时收到一条完全看不懂的报错。
    const rpg::DeviceInfo dev = rpg::QueryDevice(0);
    if (!dev.available) {
      RPG_LOG_WARN("没有可用的 CUDA 设备：%s —— 只读到元数据，建会话与生成会失败",
                   dev.reason.c_str());
    } else {
      entry->gpu = std::make_unique<rpg::GpuModel>(
          rpg::GpuModel::Load(model_path, entry->max_pos));
      RPG_LOG_INFO("权重已上卡：%s", entry->gpu->Describe().c_str());
    }
#else
    RPG_LOG_INFO("本次构建没有 CUDA 后端：只读元数据，generate 会明确报错。");
#endif

    std::lock_guard<std::mutex> lk(g_mutex);
    const int32_t h = g_next_model++;
    g_models.emplace(h, std::move(*entry));
    return h;
  }
  RPG_ABI_CATCH(-1)
}

void rpg_engine_unload_model(int32_t model) {
  try {
    std::lock_guard<std::mutex> lk(g_mutex);
    const auto it = g_models.find(model);
    if (it == g_models.end()) return;  // 无效句柄 = 空操作（约定见头文件）
    for (int32_t s : it->second.sessions) g_sessions.erase(s);
    g_models.erase(it);
  } catch (const std::exception& e) {
    RPG_LOG_WARN("unload_model 时出错（已忽略）：%s", e.what());
  } catch (...) {
    RPG_LOG_WARN("unload_model 时遇到未知异常（已忽略）");
  }
}

int32_t rpg_engine_session_create(int32_t model, const char* session_json) {
  try {
    rpg::SetLastError("");
    ModelEntry* m = RequireModel(model);
    const rpg::Json cfg = ParseConfigOrEmpty(session_json, "会话配置");

    auto s = std::make_unique<Session>();
    s->model = model;
    s->max_seq = m->default_max_seq;
    if (cfg.Has("max_seq")) {
      const int64_t v = cfg["max_seq"].AsInt(m->default_max_seq);
      RPG_CHECK(v > 0 && v <= m->max_pos, "会话 max_seq=%lld 超出模型位置范围 (0, %d]",
                static_cast<long long>(v), m->max_pos);
      s->max_seq = static_cast<int32_t>(v);
    }
    if (cfg.Has("max_tokens")) {
      const int64_t v = cfg["max_tokens"].AsInt(kDefaultMaxTokens);
      s->max_tokens = static_cast<int32_t>(v > 0 ? v : kDefaultMaxTokens);
    }
    // 缓存里至少得留一个位置给生成结果，否则这个会话永远产不出东西。
    // **收窄而不是报错**：max_tokens 偏小是配置问题，不该让会话建不出来
    // （与 CPU 版同一条，连收窄到什么值都一样）。
    if (s->max_tokens >= s->max_seq) {
      RPG_LOG_WARN("会话 max_tokens=%d 相对 max_seq=%d 过大，已收到 %d",
                   s->max_tokens, s->max_seq, s->max_seq - 1);
      s->max_tokens = s->max_seq - 1;
    }

#ifdef RPG_HAS_CUDA
    // KV Cache 在这里就分配、**不**延迟到第一次 generate：显存不够要在建会话时
    // 报出来（那时宿主还能选一个小一点的 max_seq），而不是生成到第 3000 个 token
    // 时炸在某个 kernel 里（GpuKvCache 头注释里那条理由）。
    RPG_CHECK(m->gpu != nullptr,
              "这个模型加载时没有可用的 CUDA 设备（当时只读了元数据），无法建会话 —— "
              "确认机器上有能用的 NVIDIA 卡，且产物是 CUDA 版");
    s->kv = std::make_unique<rpg::GpuKvCache>(m->meta, s->max_seq);
#endif
    if (cfg.Has("temperature")) {
      float t = static_cast<float>(cfg["temperature"].AsDouble(0.0));
      if (t > kMaxTemperature) {
        RPG_LOG_WARN("temperature %.2f 超过上限 %.1f，已钳住", t, kMaxTemperature);
        t = kMaxTemperature;
      }
      s->temperature = t < 0.0f ? 0.0f : t;
    }
    // seed 落在**会话**上（整局一次），由宿主按"本局种子 + 会话标签"派生 —— 引擎
    // 不做随机数，只把 seed 当纯函数的输入（CPU 版 DESIGN 里那条"随机性归属"）。
    if (cfg.Has("seed")) s->seed = static_cast<uint64_t>(cfg["seed"].AsInt(0));
    if (cfg.Has("stop")) {
      const rpg::Json& arr = cfg["stop"];
      if (arr.IsArray()) {
        const int64_t n = static_cast<int64_t>(arr.size());
        for (int64_t i = 0; i < n; ++i) {
          const rpg::Json item = arr.GetAt(static_cast<size_t>(i));
          if (item.IsString()) s->stop.push_back(item.AsString(""));
        }
      }
    }

#ifdef RPG_HAS_CUDA
    // K+V 各 fp16 ⇒ 每个元素 4 字节（CPU 版这里写 8，因为那边是 fp32）。
    RPG_LOG_INFO("已创建会话（模型 #%d，缓存容量 %d，KV 约 %.0f MiB）", model,
                 s->max_seq,
                 static_cast<double>(s->max_seq) * m->meta.n_layers *
                     m->meta.n_kv_heads * m->meta.head_dim * 4 / 1048576.0);
#else
    RPG_LOG_INFO("已创建会话（模型 #%d，缓存容量 %d）", model, s->max_seq);
#endif

    std::lock_guard<std::mutex> lk(g_mutex);
    const int32_t h = g_next_session++;
    g_sessions.emplace(h, std::move(s));
    g_models[model].sessions.push_back(h);
    return h;
  }
  RPG_ABI_CATCH(-1)
}

void rpg_engine_session_destroy(int32_t session) {
  try {
    std::lock_guard<std::mutex> lk(g_mutex);
    const auto it = g_sessions.find(session);
    if (it == g_sessions.end()) return;
    const int32_t owner = it->second->model;
    const auto mit = g_models.find(owner);
    if (mit != g_models.end()) {
      auto& v = mit->second.sessions;
      v.erase(std::remove(v.begin(), v.end(), session), v.end());
    }
    g_sessions.erase(it);
  } catch (const std::exception& e) {
    RPG_LOG_WARN("session_destroy 时出错（已忽略）：%s", e.what());
  } catch (...) {
    RPG_LOG_WARN("session_destroy 时遇到未知异常（已忽略）");
  }
}

int32_t rpg_engine_session_reset(int32_t session) {
  try {
    rpg::SetLastError("");
    Session* s = RequireSession(session);  // 不存在即抛 → 返回负值
#ifdef RPG_HAS_CUDA
    // Reset 只把长度归零、**显存留着** —— 同一房间开新一局时不必重新分配那几百
    // MiB（与 CPU 版同一条）。cached_ids 必须跟着清：它是"cache 里存着什么"的
    // 唯一记录，漏清会让下一次 generate 拿一个与 cache 对不上的前缀去做匹配
    // —— 那种错不崩，只让输出悄悄变味。
    if (s->kv != nullptr) s->kv->Reset();
#endif
    s->cached_ids.clear();
    return RPG_ENGINE_OK;
  }
  RPG_ABI_CATCH(-1)
}

char* rpg_engine_generate(int32_t session, const char* request_json) {
  try {
    rpg::SetLastError("");
    Session* s = RequireSession(session);
    ModelEntry* m = RequireModel(s->model);

#ifdef RPG_HAS_CUDA
    // 两个"不该发生但要说清"的前置检查：设备不可用时 load_model 只读了元数据
    // （见那边），这种模型不该走到生成。
    RPG_CHECK(m->gpu != nullptr,
              "模型加载时没有可用的 CUDA 设备（当时只读了元数据），无法生成");
    RPG_CHECK(s->kv != nullptr,
              "会话没有 KV Cache —— 不该发生（建会话时会分配），请报 bug");
    const rpg::GpuModel& model = *m->gpu;
    const rpg::ModelMeta& c = model.config();
    const rpg::Tokenizer& tok = model.tokenizer();

    // ---------------------------------------------------------- 1. 解析请求
    RPG_CHECK(request_json != nullptr && *request_json != '\0', "request_json 为空");
    std::string perr;
    const rpg::Json req = rpg::Json::Parse(request_json, &perr);
    RPG_CHECK(perr.empty(), "request_json 解析失败：%s", perr.c_str());
    RPG_CHECK(req.IsObject(), "request_json 必须是 JSON 对象");

    const std::vector<rpg::ChatMessage> msgs =
        rpg::ParseMessages(req.Get("messages"));
    RPG_CHECK(!msgs.empty(), "messages 为空：至少需要一条消息");
    // 工具列表为空数组时，渲染走的是"无工具"分支（与官方模板一致）。
    const rpg::Json tools = rpg::ParseTools(req.Get("tools"));

    // ---------------------------------------------------------- 2. 渲染 + 分词
    rpg::ChatTemplateOptions topt;
    topt.add_generation_prompt = true;
    // 缺省 = 思考模式（官方模板只在显式 false 时补空思考块）。
    topt.enable_thinking = req.Get("enable_thinking").AsBool(true);

    const std::string prompt = rpg::RenderChatml(msgs, tools, topt);
    std::vector<int32_t> ids = tok.Encode(prompt, /*parse_special=*/true);
    if (c.add_bos && c.bos_id >= 0) ids.insert(ids.begin(), c.bos_id);
    const int64_t total = static_cast<int64_t>(ids.size());
    RPG_CHECK(total > 0, "渲染后的 prompt 编码为空");

    // ---------------------------------------------------------- 3. 与缓存对前缀
    // 宿主每轮把**完整 messages** 传进来，所以本轮 prompt 通常是"上轮 prompt
    // + 新回复 + 新消息"。找出公共前缀，只对新增的部分做前向。
    int64_t lcp = 0;
    const int64_t cached = static_cast<int64_t>(s->cached_ids.size());
    RPG_CHECK(cached == s->kv->len(),
              "内部状态不一致：cached_ids=" RPG_SZ " 而 kv.len=" RPG_SZ,
              RPG_SZ_ARG(cached), RPG_SZ_ARG(s->kv->len()));
    while (lcp < cached && lcp < total && s->cached_ids[lcp] == ids[lcp]) ++lcp;

    // 至少要前向一个位置，才有"下一个 token 的分布"可用 —— 全前缀命中（宿主
    // 原样重发同一份请求）时回退一个位置重算那一个 token。这里刻意不缓存上次
    // 的 logits 行：多存一份状态就多一处要同步的地方，而代价只是重算一个位置。
    if (lcp == total) lcp = total - 1;
    const int64_t rows = total - lcp;  // 本次要前向的位置数（≥ 1）

    // ---------------------------------------------------------- 4. 生成长度与容量
    int32_t max_tokens = req.Get("max_tokens").AsInt32(s->max_tokens);
    if (max_tokens <= 0) max_tokens = s->max_tokens;

    // 采样参数：请求优先，其次会话默认（temperature <= 0 = 贪心）。温度钳到
    // [0, 上限] —— 旋钮拧到底也不会把模型推进"胡言乱语区"。
    float temperature = req.Get("temperature").AsFloat(s->temperature);
    if (temperature < 0.0f) temperature = 0.0f;
    if (temperature > kMaxTemperature) {
      RPG_LOG_WARN("temperature=%.3f 超过上限 %.1f，已钳制", temperature,
                   kMaxTemperature);
      temperature = kMaxTemperature;
    }
    const uint64_t seed = static_cast<uint64_t>(
        req.Get("seed").AsInt(static_cast<int64_t>(s->seed)));

    // **所有会失败的条件检查都在这里做完**，之后才动 cache —— 否则一次失败会
    // 把会话留在"cached_ids 与 kv.len 对不上"的状态里，下一次调用直接报内部
    // 状态不一致（这个顺序是 CPU 版 M4 第一次跑红时定下来的）。
    RPG_CHECK(total + max_tokens + kSeqMargin <= s->kv->max_seq(),
              "序列超长：prompt " RPG_SZ " + 生成 %d + 余量 " RPG_SZ
              " 超过会话容量 " RPG_SZ "（调大 session_json.max_seq，或先 session_reset）",
              RPG_SZ_ARG(total), max_tokens, RPG_SZ_ARG(kSeqMargin),
              RPG_SZ_ARG(s->kv->max_seq()));

    // ---------------------------------------------------------- 5. prefill
    if (lcp < s->kv->len()) {
      // 回退到 lcp：kv 里 [0, lcp) 的 K/V 内容仍然有效（同一位置的 K/V 只由它
      // 自己的前缀决定），所以归零长度再设回 lcp 即可，不必重算前缀。
      s->kv->Reset();
    }
    s->kv->SetLen(lcp);

    try {
      // ⚠️ logits 传 **nullptr**：这一段前向的结果只要留在显存里 ——
      // `s->scratch.logits` 的最后一行就是"下一个 token 的分布"。CPU 版必须把
      // rows × vocab 的 logits 下载回 host 再取那一行；在 CUDA 侧那一整块搬运
      // 根本不需要（GpuGenerate 直接吃显存地址）。这是两边**唯一**的实质差异。
      rpg::GpuForwardFull(model, ids.data() + lcp, rows, &s->scratch,
                          /*logits=*/nullptr, rpg::GpuForwardOptions(), s->kv.get());
      RPG_CHECK(s->kv->len() == total,
                "prefill 后 kv.len=" RPG_SZ "，应为 " RPG_SZ,
                RPG_SZ_ARG(s->kv->len()), RPG_SZ_ARG(total));
      s->cached_ids.assign(ids.begin(), ids.end());
    } catch (...) {
      // 前向中途失败（OOM、越界……）会把 cache 留在一个"说不清长度"的状态。
      // 宁可从零重来，也不能让下一次调用拿着错的 cached_ids 去复用前缀。
      s->kv->Reset();
      s->cached_ids.clear();
      throw;
    }

    // ---------------------------------------------------------- 6. 停止 token
    std::vector<int32_t> stop_ids;
    if (c.eos_id >= 0) stop_ids.push_back(c.eos_id);
    const int32_t im_end = tok.IdOf("<|im_end|>");
    if (im_end >= 0 && !ContainsId(stop_ids, im_end)) stop_ids.push_back(im_end);
    AppendStopStrings(tok, req.Get("stop"), &stop_ids);
    for (const std::string& sp : s->stop) AppendStopString(tok, sp, &stop_ids);

    // ---------------------------------------------------------- 7. 生成
    rpg::GpuGenerateOptions gopt;
    gopt.max_tokens = max_tokens;
    gopt.stop_ids = stop_ids;
    gopt.sample.temperature = temperature;
    gopt.sample.top_k = kDefaultTopK;
    gopt.sample.top_p = kDefaultTopP;
    gopt.seed = seed;

    std::vector<int32_t> gen;
    // prefill 的最后一行的**显存地址**。offset 用本次的 rows（GpuForwardFull 内部
    // 按本次 seq 写 logits），不是 scratch 的容量。
    const float* last =
        static_cast<const float*>(s->scratch.logits.data()) + (rows - 1) * c.vocab;
    rpg::GpuGenerate(model, s->kv.get(), &s->scratch, last, gopt, &gen);
    s->cached_ids.insert(s->cached_ids.end(), gen.begin(), gen.end());

    // ---------------------------------------------------------- 8. 解码 + 解析
    // **不加 skip_special**：`<tool_call>` / `</tool_call>` 自己就是特殊 token，
    // 跳过它们会把结构标记一起删掉（见文件头）。控制标记由解析器清。
    const std::string raw = tok.Decode(gen, /*skip_special=*/false);
    const rpg::ParsedOutput parsed = rpg::ParseModelOutput(raw);

    // ---------------------------------------------------------- 9. 拼响应
    std::string finish;
    if (!parsed.tool_calls.empty()) {
      finish = "tool_calls";
    } else if (parsed.unterminated_tool_call || parsed.unterminated_think) {
      // 被 max_tokens 截断在半截结构里：宿主据此决定重试还是当失败。
      finish = "length";
    } else if (static_cast<int32_t>(gen.size()) >= max_tokens &&
               (gen.empty() || !ContainsId(stop_ids, gen.back()))) {
      finish = "length";
    } else {
      finish = "stop";
    }

    rpg::Json resp = rpg::Json::Object();
    resp.Set("text", parsed.text);

    rpg::Json calls = rpg::Json::Array();
    for (const rpg::ParsedToolCall& pc : parsed.tool_calls) {
      rpg::Json one = rpg::Json::Object();
      one.Set("name", pc.name);
      // 参数不是合法对象时留 null —— 引擎只解析、不判断合法性（parsed.ok 的语义）。
      one.Set("arguments", pc.ok ? pc.arguments : rpg::Json());
      calls.PushBack(std::move(one));
    }
    resp.Set("tool_calls", std::move(calls));
    resp.Set("finish_reason", finish);

    rpg::Json usage = rpg::Json::Object();
    usage.Set("prompt_tokens", total);
    usage.Set("completion_tokens", static_cast<int64_t>(gen.size()));
    resp.Set("usage", std::move(usage));

    return DupForAbi(JsonToAbiString(resp));
#else
    (void)s;
    (void)m;
    // 没编 CUDA 的产物：明确报错，不给假文本。
    rpg::Throw(
        "本产物**未编译 CUDA 后端**（构建时没检测到 nvcc），因此没有前向实现。"
        "请先装 CUDA Toolkit（≥12.8，sm_120 需要），或改用 RPGBarEngine 的 CPU 版。");
    return nullptr;  // 不会到达（rpg::Throw 是 noreturn）
#endif
  }
  RPG_ABI_CATCH(nullptr)
}

void rpg_engine_free(char* buf) { delete[] buf; }

char* rpg_engine_model_info(int32_t model) {
  try {
    rpg::SetLastError("");
    ModelEntry* m = RequireModel(model);

    rpg::Json o = rpg::Json::Object();
    o.Set("engine", "rpg_engine_cuda");
    o.Set("engine_version", kEngineVersion);
#ifdef RPG_HAS_CUDA
    o.Set("backend", "cuda");
#else
    o.Set("backend", "host-only");   // 没编 CUDA：元数据能读，生成不能跑
#endif
    // 下面这些键名与 CPU 版**逐字对齐** —— 宿主（engine_probe.py / Unity 侧）
    // 已经按它们取值，换实现不该让宿主改代码。
    o.Set("arch", m->meta.arch);
    o.Set("name", m->meta.name);
    o.Set("n_layers", static_cast<int64_t>(m->meta.n_layers));
    o.Set("hidden", static_cast<int64_t>(m->meta.hidden));
    o.Set("n_heads", static_cast<int64_t>(m->meta.n_heads));
    o.Set("n_kv_heads", static_cast<int64_t>(m->meta.n_kv_heads));
    o.Set("head_dim", static_cast<int64_t>(m->meta.head_dim));
    o.Set("intermediate", static_cast<int64_t>(m->meta.intermediate));
    o.Set("vocab", static_cast<int64_t>(m->meta.vocab));
    o.Set("context_length", static_cast<int64_t>(m->meta.context_length));
    o.Set("rope_theta", static_cast<double>(m->meta.rope_theta));
    o.Set("tied_embeddings", m->meta.tied_embeddings);
    o.Set("path", m->gguf.path());
    o.Set("max_pos", static_cast<int64_t>(m->max_pos));
    o.Set("default_max_seq", static_cast<int64_t>(m->default_max_seq));
    o.Set("weight_bytes", m->meta.weight_bytes);
    o.Set("sessions", static_cast<int64_t>(m->sessions.size()));
    // CUDA 版**多出来**的两项：设备信息与权重在显存里的实际占用。宿主可以据此
    // 显示"跑在哪块卡上"与"还剩多少显存能开会话"。
    o.Set("device_available", rpg::QueryDevice(0).available);
#ifdef RPG_HAS_CUDA
    // 设备不可用（只读了元数据）时这一项**不出现**，而不是报 0 —— 0 会被读成
    // "权重不占显存"，那是另一回事。
    if (m->gpu != nullptr) {
      o.Set("device_bytes", m->gpu->device_bytes());
    }
#endif

    return DupForAbi(JsonToAbiString(o));
  }
  RPG_ABI_CATCH(nullptr)
}

}  // extern "C"
