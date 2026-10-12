/* engine.cc —— C ABI 的实现（P0：骨架 + 模型元数据）。
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
 *    （对齐项目里"护栏做在工具体里"那条纪律）。
 *
 * 当前阶段（P0）能做到的事：加载 GGUF、读元数据、开关会话、报出设备信息。
 * **`generate` 还没有实现** —— 它会明确报错，而不是返回一段假文本。
 * 这是刻意的：一个"看起来能用但输出是垃圾"的引擎，比一个"明确说没实现"的
 * 引擎危险得多。阶段划分见 docs/DESIGN.md §5。
 */

#include "rpg_engine.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "cuda_common.h"
#include "gguf.h"
#include "json.h"
#include "model_meta.h"

namespace {

// 引擎版本。与 CMakeLists.txt 的 project(VERSION) 保持一致。
constexpr const char* kEngineVersion = "0.1.0";

// 请求/会话配置里没给 max_tokens 时的默认值（与 CPU 版一致）。
constexpr int32_t kDefaultMaxTokens = 256;

// 采样默认（与 CPU 版一致：top_k 关、top_p 0.95）。
constexpr int32_t kDefaultTopK = 0;
constexpr float kDefaultTopP = 0.95f;
constexpr float kMaxTemperature = 2.0f;

// ---------------------------------------------------------------- 注册表

struct Session {
  int32_t model = -1;

  // P1：前向本身已经实现并逐层对拍过，但**验证走 tools/layer_check**，
  // ABI 侧还没有挂上任何显存状态。P2 把 KV Cache 与上卡接进来之后，这里才会有
  // 显存句柄；在那之前 reset 保持空操作（不假装清了个东西）。
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

    RPG_LOG_INFO("已加载 %s（%d 层 / hidden %d / %d 头 / %d KV 头 / 词表 %d / "
                 "位置 %d）—— 只读元数据，权重未上传显存（上卡在 P2 接进 ABI）",
                 mm.name.c_str(), mm.n_layers, mm.hidden, mm.n_heads,
                 mm.n_kv_heads, mm.vocab, entry->max_pos);

    const rpg::DeviceInfo dev = rpg::QueryDevice(0);
    if (!dev.available) {
      // 这一条是**警告不是错误**：P0 允许在没 GPU 的机器上加载元数据（CI 里跑
      // 得起来），只是不能生成。真正的失败发生在 generate。
      RPG_LOG_WARN("没有可用的 CUDA 设备：%s", dev.reason.c_str());
    }

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

    RPG_LOG_INFO("已创建会话（模型 #%d，缓存容量 %d）", model, s->max_seq);

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
    (void)s;
    // P0：会话还没有 KV 状态，reset 是空操作。P1 起这里清显存里的 KV 与 token 历史。
    return RPG_ENGINE_OK;
  }
  RPG_ABI_CATCH(-1)
}

char* rpg_engine_generate(int32_t session, const char* request_json) {
  try {
    rpg::SetLastError("");
    Session* s = RequireSession(session);
    (void)s;
    (void)request_json;
    // ABI 到这里为止。**明确报错，不返回假文本。**
#ifdef RPG_HAS_CUDA
    rpg::Throw(
        "CUDA 后端的整网生成还没实现（P1 只做到「无 KV Cache 的整网前向」，"
        "逐层对拍已过 —— 见 docs/DESIGN.md §5 的阶段表）。"
        "当前产物编入了 CUDA 支持，但 rpg_engine_generate 还是空的，"
        "请暂时用 RPGBarEngine 的 CPU 版。");
#else
    rpg::Throw(
        "本产物**未编译 CUDA 后端**（构建时没检测到 nvcc），因此没有前向实现。"
        "请先装 CUDA Toolkit（≥12.8，sm_120 需要），或改用 RPGBarEngine 的 CPU 版。");
#endif
    return nullptr;  // 不会到达（rpg::Throw 是 noreturn）
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
    // CUDA 版**多出来**的一项：设备信息。宿主可以据此在界面上显示"跑在哪块卡上"。
    o.Set("device_available", rpg::QueryDevice(0).available);

    return DupForAbi(JsonToAbiString(o));
  }
  RPG_ABI_CATCH(nullptr)
}

}  // extern "C"
