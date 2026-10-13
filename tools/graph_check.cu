// graph_check —— P4 判据：**CUDA Graph 路径**。
//
// P4 的目标（DESIGN §5）是"从架构上把 decode 的启动开销摊掉"：
//
//   指标      CPU 版实测    GPU 目标
//   prefill   ~145 tok/s    >= 1000 tok/s
//   decode    10-14 tok/s   >= 60 tok/s
//
// 而 P2 实测 GPU 的 decode 只比 CPU 快 **1.5x**（prefill 快 14.1x）。原因不是
// 算力：decode 每步只算 1 个 token，却要发 28 层 x 十几个 kernel ≈ 400 次 kernel
// 启动 + 一次 argmax 同步。CUDA Graph 把这几百次启动录成一张图，回放时一次提交。
//
// 于是这个工具要证明两件事，缺一不可：
//
//   (**1) 图路径的输出与老路径逐位相同。**
//       这不是"顺手测一下" —— 图之所以能这么要求，是因为它**不是另一份实现**，
//       而是同一段计算（同一批 kernel、同一组参数、同一个顺序）被录下来。所以
//       "逐位相同"是可以严格断言的，不是"差不多"。
//
//   (**2) 图**真的**跑起来了。**
//       这条比第一条更容易被忽略，而且更致命：`GenerateGraph` 在录不出图时会
//       **静默退回老路径**（图是优化，不是正确性的前提）。于是"老路径 vs 老路径"
//       当然逐位相同 —— 判据全绿，一行代码都没跑到。所以每一组比较都要额外断言
//       `GpuScratch::graph_exec != nullptr`。**这是本文件最重要的一行防御。**
//
// 覆盖的用例（同一份 prompt、同一个 seed，两条路径各跑一遍）：
//   A. 贪心（T = 0）：走 `GpuArgMax` 那条分支 —— 图里也要逐位等于 P2 的贪心
//   B. 采样（T = 1，top_k / top_p 关）：整个采样链都在图里（counter 从设备端读、
//      u 由 kernel 算、逆变换读设备指针）—— 这正是 P4 最容易被录错的一环
//   C. 采样 + 筛选（T = 1，top_k = 40，top_p = 0.9）：筛选参数是**启动参数**、
//      会被录进图，一起再验一遍
//   D. **两段连续生成**（同一会话，不 reset）：第一段跑完接着跑第二段。
//      这是宿主多轮工具循环的形态（前缀复用 + 位置接续），验的是"图跨调用复用
//      且 pos0 继续递增" —— 单段那条测不到第二次调用之后的行为。
//      顺带断言第二段**没有重录图**（graph_exec 指针不变）。
//   E. 性能：prefill / decode tok/s，两条路径对照，并给出与目标的差距。
//
// ⚠️ 与 step_logits 互斥：图里不能有 D2H，所以 `use_graph && step_logits` 时
// 引擎按老路径走（并打一条警告）。本工具不做逐 token 的 logits 对拍 ——
// 那是 tools/gen_check（与 CPU 引擎对拍）的活。
//
// 用法：
//   graph_check model.gguf [--prompt-file p.txt] [--max-tokens 32]
//               [--max-seq 512] [--max-pos 4096] [--seed 12345] [--bench-tokens 128]
//   （⚠️ 注释里不写 shell 的续行反斜杠 —— 行尾的 `\` 会让 GCC 认为注释续行）

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cli_util.h"
#include "common.h"
#include "cuda_common.h"
#include "gpu_model.h"
#include "kv_cache_gpu.h"
#include "qwen3_gpu.h"
#include "sampler.h"
#include "sampler_gpu.h"

namespace {

struct RunResult {
  std::vector<std::vector<int32_t>> chunks;  // 每段生成的 token（不含 prompt）
  int64_t kv_len = 0;
  double prefill_ms = 0;   // prefill（整段 prompt）
  double decode_ms = 0;    // 所有段加起来（不含 prefill）
  int64_t generated = 0;   // 生成的总 token 数
  bool graph_used = false;          // 至少一段里 graph_exec != nullptr
  bool graph_reused = false;        // 第二段复用了第一段录的图（没重录）
};

// 一个"会话"：prefill 一次，然后**不 reset** 连着跑 `chunks` 段生成。
//
// 段与段之间不复位，正是宿主多轮工具循环的形态 —— 也正因为不复位，第二段的位置
// 从 `seq + max_tokens` 起，能把"图里录死的 pos0"这类错误逼出来。
//
// 每段的第一行 logits 来源：第一段用 prefill 的**最后一行**；之后用上一步前向
// 写在 `s->logits` 第 0 行的那一份（老路径与图路径的约定一致）。
RunResult RunSession(const rpg::GpuModel& m, const std::vector<int32_t>& prompt,
                     const rpg::SampleOptions& sample, uint64_t seed,
                     int32_t max_tokens, int32_t chunks, int64_t cap,
                     bool use_graph) {
  const rpg::ModelMeta& c = m.config();
  const int64_t seq = static_cast<int64_t>(prompt.size());

  rpg::GpuKvCache kv(c, cap);
  rpg::GpuScratch s;

  RunResult r;
  const auto t0 = std::chrono::steady_clock::now();
  // logits 传 nullptr：整段只算到显存（prefill 那 607 KB 一次都不用搬）。
  rpg::GpuForwardFull(m, prompt.data(), seq, &s, /*logits=*/nullptr,
                      rpg::GpuForwardOptions(), &kv);
  RPG_CUDA_CHECK(cudaDeviceSynchronize());
  r.prefill_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t0).count();
  RPG_CHECK(kv.len() == seq, "after the prefill kv.len=%lld, expected %lld",
            static_cast<long long>(kv.len()), static_cast<long long>(seq));

  rpg::GpuGenerateOptions g;
  g.max_tokens = max_tokens;
  g.sample = sample;
  g.seed = seed;
  g.use_graph = use_graph;

  const float* first = static_cast<const float*>(s.logits.data()) +
                       (seq - 1) * static_cast<int64_t>(c.vocab);

  cudaGraphExec_t first_graph = nullptr;
  for (int32_t k = 0; k < chunks; ++k) {
    std::vector<int32_t> ids;
    const auto t1 = std::chrono::steady_clock::now();
    rpg::GpuGenerate(m, &kv, &s, first, g, &ids);
    RPG_CUDA_CHECK(cudaDeviceSynchronize());
    r.decode_ms += std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t1).count();

    r.chunks.push_back(ids);
    r.generated += static_cast<int64_t>(ids.size());

    if (use_graph) {
      if (s.graph_exec != nullptr) r.graph_used = true;
      if (k == 0) {
        first_graph = s.graph_exec;
      } else if (first_graph != nullptr && s.graph_exec == first_graph) {
        r.graph_reused = true;
      }
    }
    // 下一段的起点：上一步前向写下的 logits 第 0 行。
    first = static_cast<const float*>(s.logits.data());
  }

  r.kv_len = kv.len();
  return r;
}

void PrintIds(const char* tag, const std::vector<int32_t>& v) {
  std::printf("    %s [", tag);
  for (size_t i = 0; i < v.size(); ++i) std::printf(i ? ",%d" : "%d", v[i]);
  std::printf("]\n");
}

// 两条路径的每一段是否逐位相同；不同则打印前几处位置。
bool SameChunks(const RunResult& a, const RunResult& b, const char* what) {
  if (a.chunks.size() != b.chunks.size()) {
    std::printf("  **FAIL**: chunk count differs (%lld vs %lld)\n",
                static_cast<long long>(a.chunks.size()),
                static_cast<long long>(b.chunks.size()));
    return false;
  }
  bool ok = true;
  for (size_t k = 0; k < a.chunks.size(); ++k) {
    const std::vector<int32_t>& x = a.chunks[k];
    const std::vector<int32_t>& y = b.chunks[k];
    if (x.size() != y.size()) {
      std::printf("  **FAIL** [%s] chunk %lld: length differs (%lld vs %lld)\n",
                  what, static_cast<long long>(k),
                  static_cast<long long>(x.size()),
                  static_cast<long long>(y.size()));
      ok = false;
      continue;
    }
    size_t diff = 0;
    size_t first_bad = 0;
    bool have_bad = false;
    for (size_t i = 0; i < x.size(); ++i) {
      if (x[i] != y[i]) {
        ++diff;
        if (!have_bad) {
          have_bad = true;
          first_bad = i;
        }
      }
    }
    if (diff == 0) {
      std::printf("  [%s] chunk %lld: %lld/%lld tokens bit-identical\n", what,
                  static_cast<long long>(k), static_cast<long long>(x.size()),
                  static_cast<long long>(x.size()));
    } else {
      std::printf(
          "  **FAIL** [%s] chunk %lld: %lld/%lld tokens differ (first at %lld: "
          "%d != %d)\n",
          what, static_cast<long long>(k), static_cast<long long>(diff),
          static_cast<long long>(x.size()), static_cast<long long>(first_bad),
          x[first_bad], y[first_bad]);
      ok = false;
    }
  }
  return ok;
}

// ---------------------------------------------------------------- 设备端 counter
//
// 与图**无关**的一条定点判据，插在这里是因为它把"图录得对不对"与"新的采样入口
// 对不对"这两件事分开。
//
// P4 为图路径加了 `GpuSampleTokenDev`：counter 从**设备内存**读、`u` 由 kernel 算。
// 这是新代码，而 P3 的 tools/sample_check 盯的是 `GpuSampleToken`（u 在 host 算）
// —— 于是这条新路径在出问题之前没有任何判据。它一旦错了，症状是"图路径的采样
// 与老路径完全不同"（看起来像图录错了），而真正的根因在采样入口。
//
// 判据：同一份 logits + 同一个 seed + **同一个 counter**，两个入口必须选出
// **同一个 token**。逐位，不是"差不多"。
int CheckDevCounter(const rpg::GpuModel& m, const std::vector<int32_t>& ids,
                    const rpg::SampleOptions& sample, uint64_t seed, int64_t cap,
                    int32_t n_cases) {
  const rpg::ModelMeta& c = m.config();
  const int64_t seq = static_cast<int64_t>(ids.size());
  std::printf("\n========== D device-side counter sampling (not graph specific) ==========\n");

  rpg::GpuKvCache kv(c, cap);
  rpg::GpuScratch s;
  std::vector<float> lg;
  rpg::GpuForwardFull(m, ids.data(), seq, &s, &lg, rpg::GpuForwardOptions(), &kv);
  const float* row = static_cast<const float*>(s.logits.data()) +
                     (seq - 1) * static_cast<int64_t>(c.vocab);
  float* probs = static_cast<float*>(s.probs.data());
  int32_t* idx = static_cast<int32_t*>(s.argmax_idx.data());
  // counter 用**自己的一块**显存，不去碰引擎的 pos0 暂存区 —— 往那里写而不经过
  // StagePos0 会让它的值缓存变成谎话（见 cuda_common.h）。判据工具不该去动
  // 引擎的全局状态。
  rpg::DeviceBuffer counter_dev(sizeof(uint64_t));

  int fail = 0;
  for (int32_t k = 0; k < n_cases; ++k) {
    const uint64_t counter = static_cast<uint64_t>(seq) + static_cast<uint64_t>(k);

    // (1) host 算 u 的老入口。
    rpg::GpuSampleToken(row, c.vocab, sample, seed, counter, probs, idx);
    RPG_CUDA_CHECK(cudaDeviceSynchronize());
    int32_t from_host = 0;
    RPG_CUDA_CHECK(cudaMemcpy(&from_host, idx, sizeof(int32_t), cudaMemcpyDeviceToHost));

    // (2) 设备端算 u 的新入口（counter 从显存读）。
    counter_dev.Upload(&counter, sizeof(uint64_t));
    rpg::GpuSampleTokenDev(row, c.vocab, sample, seed,
                           reinterpret_cast<const uint64_t*>(counter_dev.data()),
                           probs, idx);
    RPG_CUDA_CHECK(cudaDeviceSynchronize());
    int32_t from_dev = 0;
    RPG_CUDA_CHECK(cudaMemcpy(&from_dev, idx, sizeof(int32_t), cudaMemcpyDeviceToHost));

    const float u_host = rpg::DevCounterUniform(seed, counter);
    if (from_host == from_dev) {
      std::printf("  counter %-6llu u_host = %.9f : host -> %d, dev -> %d  OK\n",
                  static_cast<unsigned long long>(counter),
                  static_cast<double>(u_host), from_host, from_dev);
    } else {
      std::printf("  counter %-6llu u_host = %.9f : host -> %d, dev -> %d  **FAIL**\n",
                  static_cast<unsigned long long>(counter),
                  static_cast<double>(u_host), from_host, from_dev);
      ++fail;
    }
  }
  if (fail == 0) {
    std::printf("  both sampling entry points agree on %d case(s): OK\n", n_cases);
  } else {
    std::printf("  **FAIL**: the device-side counter path disagrees with the host "
                "path in %d/%d case(s) => the graph path's sampling is the broken "
                "one here, not the graph.\n", fail, n_cases);
  }
  return fail;
}

// 一组对照：老路径 vs 图路径。
//
// 除了"逐位相同"，还**必须**断言图真的跑起来了 —— 见文件头的第 (2) 条。
int CompareGraph(const rpg::GpuModel& m, const std::vector<int32_t>& ids,
                 const char* tag, const rpg::SampleOptions& sample, uint64_t seed,
                 int32_t max_tokens, int32_t chunks, int64_t cap) {
  std::printf("\n========== %s ==========\n", tag);
  int fail = 0;

  const RunResult off =
      RunSession(m, ids, sample, seed, max_tokens, chunks, cap, /*use_graph=*/false);
  const RunResult on =
      RunSession(m, ids, sample, seed, max_tokens, chunks, cap, /*use_graph=*/true);

  for (size_t k = 0; k < off.chunks.size(); ++k) {
    PrintIds("no-graph", off.chunks[k]);
    PrintIds("graph   ", on.chunks[k]);
  }
  std::printf("  decoded: \"%s\"\n", m.tokenizer().Decode(off.chunks[0]).c_str());

  // (A) 图真的录成并被用到了 —— 少了这条，"老路径 vs 老路径"也会全绿。
  if (!on.graph_used) {
    std::printf(
        "  **FAIL**: the graph path never produced a graph (graph_exec == null) "
        "=> the run silently fell back to per-step launches, and the comparison "
        "below is meaningless.\n");
    ++fail;
  } else {
    std::printf("  graph was captured and used: OK\n");
  }
  if (chunks > 1) {
    if (on.graph_reused) {
      std::printf("  the 2nd chunk reused the same graph (no re-capture): OK\n");
    } else if (on.graph_used) {
      std::printf("  **FAIL**: the 2nd chunk did not reuse the graph (re-captured "
                  "or lost it) => graph_key is unstable across calls\n");
      ++fail;
    }
  }

  // (B) 逐位相同。
  if (!SameChunks(off, on, "bit-exact")) ++fail;

  // (C) 不变量：cache.len() == prompt + 生成数（两段合计）。
  for (int side = 0; side < 2; ++side) {
    const RunResult& r = side ? on : off;
    const int64_t want = static_cast<int64_t>(ids.size()) + r.generated;
    if (r.kv_len != want) {
      std::printf("  **FAIL**: %s kv.len=%lld, expected %lld\n",
                  side ? "graph" : "no-graph", static_cast<long long>(r.kv_len),
                  static_cast<long long>(want));
      ++fail;
    }
  }
  if (fail == 0) {
    std::printf("  OK\n");
  }
  return fail;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf(
        "usage: graph_check <model.gguf> [options]\n"
        "  --prompt-file <f>   read the prompt from a file (avoids argv transcoding)\n"
        "  --prompt <text>     encode this text instead (ASCII only)\n"
        "  --max-tokens <N>    tokens per chunk (default 32)\n"
        "  --chunks <N>        chunks per session (default 2; 1 disables the\n"
        "                      'graph reused across calls' check)\n"
        "  --max-seq <N>       KV cache capacity (default: rounded up as needed)\n"
        "  --max-pos <N>       positions covered by the RoPE table (default 4096)\n"
        "  --seed <N>          session seed (default 12345)\n"
        "  --bench-tokens <N>  tokens for the throughput section (default 128; 0 skips)\n");
    return 2;
  }

  const std::string model_path = argv[1];
  std::string prompt = "You are the dungeon master. Describe what the adventurer sees.";
  int32_t max_pos = 4096;
  int32_t max_tokens = 32;
  int32_t chunks = 2;
  int32_t bench_tokens = 128;
  int64_t max_seq = 0;
  uint64_t seed = 12345;

  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) rpg::Throw("option %s needs a value", a.c_str());
      return argv[++i];
    };
    if (a == "--prompt") prompt = next();
    else if (a == "--prompt-file") prompt = cli::ReadFile(next());
    else if (a == "--max-tokens") max_tokens = std::atoi(next());
    else if (a == "--chunks") chunks = std::atoi(next());
    else if (a == "--max-seq") max_seq = std::atoll(next());
    else if (a == "--max-pos") max_pos = std::atoi(next());
    else if (a == "--seed") seed = static_cast<uint64_t>(std::strtoull(next(), nullptr, 10));
    else if (a == "--bench-tokens") bench_tokens = std::atoi(next());
    else rpg::Throw("unknown option %s", a.c_str());
  }

  try {
    rpg::GpuModel model = rpg::GpuModel::Load(model_path, max_pos);
    const rpg::ModelMeta& c = model.config();
    std::printf("%s", model.Describe().c_str());

    cudaDeviceProp prop{};
    RPG_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s (sm_%d%d, %d SM, %.1f GiB)\n", prop.name, prop.major,
                prop.minor, prop.multiProcessorCount,
                static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024 * 1024));

    std::vector<int32_t> ids = model.tokenizer().Encode(prompt, /*parse_special=*/true);
    if (c.add_bos && c.bos_id >= 0) ids.insert(ids.begin(), c.bos_id);
    RPG_CHECK(!ids.empty(), "the token sequence is empty");
    const int64_t seq = static_cast<int64_t>(ids.size());
    RPG_CHECK(max_tokens > 0, "max_tokens must be positive");
    RPG_CHECK(chunks >= 1, "chunks must be >= 1");

    // 容量：预填 + chunks 段 + 余量，再取整到 256。
    const int64_t span = static_cast<int64_t>(max_tokens) *
                         (chunks > 1 ? chunks : 1);
    const int64_t want = seq + span + static_cast<int64_t>(max_tokens) + 8;
    const int64_t cap = max_seq > 0 ? max_seq : ((want + 255) / 256) * 256;
    RPG_CHECK(cap <= max_pos, "KV capacity %lld exceeds the %d positions of the RoPE table",
              static_cast<long long>(cap), max_pos);

    std::printf("\n[input] %lld tokens; %d tokens x %d chunk(s); seed = %llu; kv cap = %lld\n",
                static_cast<long long>(seq), max_tokens, chunks,
                static_cast<unsigned long long>(seed), static_cast<long long>(cap));

    // 采样参数的默认值来自同源副本 sampler.h，不在这里另立一套。
    rpg::SampleOptions greedy;
    greedy.temperature = 0.0f;

    rpg::SampleOptions t1;
    t1.temperature = 1.0f;
    t1.top_k = 0;      // 关掉筛选：这一步只验"分布采样 + 随机源"
    t1.top_p = 1.0f;

    rpg::SampleOptions t1_filt = t1;
    t1_filt.top_k = 40;
    t1_filt.top_p = 0.9f;

    int fail = 0;

    fail += CompareGraph(model, ids, "A greedy (T = 0)", greedy, seed, max_tokens,
                         chunks, cap);
    fail += CompareGraph(model, ids, "B sampling (T = 1, no filtering)", t1, seed,
                         max_tokens, chunks, cap);
    fail += CompareGraph(model, ids, "C sampling (T = 1, top_k = 40, top_p = 0.9)",
                         t1_filt, seed, max_tokens, chunks, cap);

    // 先把"新的采样入口"单独钉住：它红了，上面 B/C 的红就都解释得通，
    // 而且不必再去怀疑"图录错了"。
    fail += CheckDevCounter(model, ids, t1_filt, seed, cap, 8);

    // ---------------------------------------------------------------- E 性能
    if (bench_tokens > 0) {
      std::printf("\n========== E throughput (target: prefill >= 1000 tok/s, decode >= 60 tok/s) ==========\n");
      const int64_t bcap = ((seq + bench_tokens + 256 + 255) / 256) * 256;
      RPG_CHECK(bcap <= max_pos, "the benchmark needs kv capacity %lld > max_pos %d",
                static_cast<long long>(bcap), max_pos);
      const RunResult boff = RunSession(model, ids, t1_filt, seed, bench_tokens, 1,
                                        bcap, /*use_graph=*/false);
      const RunResult bon = RunSession(model, ids, t1_filt, seed, bench_tokens, 1,
                                       bcap, /*use_graph=*/true);
      const double pre_tps = static_cast<double>(seq) / (bon.prefill_ms / 1000.0);
      const double off_tps =
          static_cast<double>(boff.generated) / (boff.decode_ms / 1000.0);
      const double on_tps =
          static_cast<double>(bon.generated) / (bon.decode_ms / 1000.0);
      std::printf("  prefill: %lld tokens in %.1f ms => %.1f tok/s  (target >= 1000)\n",
                  static_cast<long long>(seq), bon.prefill_ms, pre_tps);
      std::printf("  decode (no graph): %lld tokens in %.1f ms => %.1f tok/s\n",
                  static_cast<long long>(boff.generated), boff.decode_ms, off_tps);
      std::printf("  decode (graph)   : %lld tokens in %.1f ms => %.1f tok/s  "
                  "%.2fx vs no-graph  (target >= 60)\n",
                  static_cast<long long>(bon.generated), bon.decode_ms, on_tps,
                  on_tps / off_tps);
      if (!bon.graph_used) {
        std::printf("  **FAIL**: the throughput run did not use a graph.\n");
        ++fail;
      }
      // 性能是**目标**而不是判据的硬性前提：这里只如实报出到底达没达到，
      // 不因为没达到就让整条 P4 判据变红（P4 的硬判据是正确性 + 图真的用了）。
      std::printf("  => prefill %s the target; decode(graph) %s the target\n",
                  pre_tps >= 1000.0 ? "meets" : "is below",
                  on_tps >= 60.0 ? "meets" : "is below");
    }

    std::printf("\n==============================================================\n");
    if (fail == 0) {
      std::printf("P4 graph criteria (A-E) all passed.\n");
    } else {
      std::printf("P4 graph criteria: %d item(s) failed (see above).\n", fail);
    }
    return fail == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
