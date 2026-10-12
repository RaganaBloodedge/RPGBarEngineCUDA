// cpu_ref_gen —— P2 判据的**参考侧**：用 CPU 引擎生成，再重放收集每步 logits。
//
// ## 这个文件为什么在本仓库，而不是直接用 RPGBarEngine/tools/gen_check
//
// gen_check 是 M2 的验证工具，它**不导出每步的 logits**。而 DESIGN §7.3 要求
// "一旦分歧，打印分歧位置、两边各自的 top-5 logits" —— GPU 侧能导出，参考侧
// 不能，那条判据就只有一半。
//
// 修法不是"在 CPU 仓库加一个选项"（那要动上位仓库、走它的发版流程），而是
// **只换 main，不换实现**：这个文件通过 `#include "qwen3.h"` 直接使用 CPU 引擎
// 的 `Model` / `KvCache` / `ForwardFull` / `ForwardStep` / `GenerateGreedy` /
// `ArgMax`，源码由 tools/build_cpu_ref.sh 从 RPGBarEngine 现编过来。
// **没有一行前向或解码逻辑是抄的** —— 抄一份等于拿我对 CPU 行为的理解去对我的
// GPU 实现，两边同源的理解错误会互相掩盖（DESIGN §7 末）。
//
// ## 生成用引擎的 GenerateGreedy，logits 用"重放"收集
//
// 每步的 logits 不靠"我自己写一个解码循环"来拿（那又是重写一遍引擎行为），而是：
//
//   1. 用 CPU 引擎的 `GenerateGreedy` 生成 → `<前缀>.gen.txt`（**判据对象**）
//   2. 另建一份 cache，重新 prefill，按第 1 步得到的序列逐步 `ForwardStep`，
//      收集每一步的 logits；**并顺带校验重放的 argmax 等于第 1 步的 token**
//
// 第 2 步不是"另一个实现"，是数据收集 + 一条额外的自检：重放要能把同一条序列
// 重放出来，才说明第 1 步的 cache 状态与第 2 步一致（引擎是纯函数，这条本该成立）。
// 一旦不成立，先修引擎，别拿这份 step_logits 去对拍。
//
// 用法与 GPU 侧 tools/gen_check.cu **同名同形状**，便于 run_p2.sh 里并排看。

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cli_util.h"
#include "common.h"
#include "kernels.h"  // rpg::ArgMax —— qwen3.h 不转出来（它的 .cc 自己 include）
#include "model.h"
#include "npy.h"
#include "qwen3.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf(
        "用法: cpu_ref_gen <model.gguf> [选项]\n"
        "  --prompt <文本>      用分词器编码（默认）\n"
        "  --prompt-file <f>    从文件读提示词\n"
        "  --tokens <a,b,c>     直接给 token id\n"
        "  --max-tokens <N>     生成数量（默认 32）\n"
        "  --max-seq <N>        KV Cache 容量（默认自动）\n"
        "  --max-pos <N>        RoPE 表覆盖的位置数（默认 4096）\n"
        "  --threads <N>        并行度（默认物理核）\n"
        "  --stop <id,id>       停止 token（默认无）\n"
        "  --stop-eos           用模型自带的 eos 当停止 token\n"
        "  --out <前缀>         写 <前缀>.tokens.txt / .gen.txt\n"
        "  --step-logits        导出 prefill 与每步 logits（对拍用）\n"
        "  --top <K>            打印最后位置 top-K（默认 5）\n"
        "  --info               只打印模型信息\n");
    return 2;
  }

  const std::string model_path = argv[1];
  std::string prompt = "The capital of France is";
  std::string tokens_arg;
  std::string out_prefix;
  std::string stop_arg;
  int threads = 0;
  int32_t max_pos = 4096;
  int32_t max_tokens = 32;
  int64_t max_seq = 0;
  int top_k = 5;
  bool info_only = false;
  bool stop_eos = false;
  bool step_logits = false;

  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) rpg::Throw("选项 %s 后面缺少参数", a.c_str());
      return argv[++i];
    };
    if (a == "--prompt") prompt = next();
    else if (a == "--prompt-file") prompt = cli::ReadFile(next());
    else if (a == "--tokens") tokens_arg = next();
    else if (a == "--max-tokens") max_tokens = std::atoi(next());
    else if (a == "--max-seq") max_seq = std::atoll(next());
    else if (a == "--max-pos") max_pos = std::atoi(next());
    else if (a == "--threads") threads = std::atoi(next());
    else if (a == "--stop") stop_arg = next();
    else if (a == "--out") out_prefix = next();
    else if (a == "--step-logits") step_logits = true;
    else if (a == "--top") top_k = std::atoi(next());
    else if (a == "--stop-eos") stop_eos = true;
    else if (a == "--info") info_only = true;
    else rpg::Throw("未知选项 %s", a.c_str());
  }

  try {
    rpg::Model model = rpg::Model::Load(model_path, threads, max_pos);
    std::printf("%s", model.Describe().c_str());
    if (info_only) return 0;

    const rpg::ModelConfig& c = model.config();

    // ---- 输入 ----
    std::vector<int32_t> ids;
    if (!tokens_arg.empty()) {
      ids = cli::ParseIds(tokens_arg);
      std::printf("\n[输入] 直接指定 " RPG_SZ " 个 token id\n",
                  RPG_SZ_ARG(ids.size()));
    } else {
      ids = model.tokenizer().Encode(prompt, /*parse_special=*/true);
      if (c.add_bos && c.bos_id >= 0) ids.insert(ids.begin(), c.bos_id);
      std::printf("\n[输入] \"%s\" → " RPG_SZ " 个 token\n", prompt.c_str(),
                  RPG_SZ_ARG(ids.size()));
    }
    RPG_CHECK(!ids.empty(), "token 序列为空");
    const int64_t seq = static_cast<int64_t>(ids.size());

    // ---- KV Cache 容量（与 GPU 侧同一套算法，否则两边比的东西不是一回事）----
    RPG_CHECK(max_tokens > 0, "max_tokens 必须为正");
    const int64_t want = seq + static_cast<int64_t>(max_tokens) + 8;
    const int64_t cap = max_seq > 0 ? max_seq : ((want + 255) / 256) * 256;
    RPG_CHECK(cap <= max_pos,
              "KV 容量 %lld 超过 RoPE 表覆盖的 %d",
              static_cast<long long>(cap), max_pos);
    std::printf("[缓存] fp32 %d 层 × %d KV 头 × %lld × %d = %.1f MiB\n",
                c.n_layers, c.n_kv_heads, static_cast<long long>(cap), c.head_dim,
                static_cast<double>(cap) * c.n_layers * c.n_kv_heads * c.head_dim *
                    8 / 1048576.0);

    rpg::ForwardScratch scratch;
    std::vector<float> logits;

    // ---- prefill + 贪心生成（**引擎自己的 GenerateGreedy**）----
    rpg::KvCache kv(c, cap);
    rpg::ForwardOptions fopt;
    const auto t0 = std::chrono::steady_clock::now();
    rpg::ForwardFull(model, ids.data(), seq, &scratch, &logits, fopt, &kv);
    const auto t1 = std::chrono::steady_clock::now();
    const double prefill_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("\n[prefill] %lld 个位置 %.1f ms（%.1f tok/s）\n",
                static_cast<long long>(seq), prefill_ms,
                static_cast<double>(seq) / (prefill_ms / 1000.0));

    rpg::GenerateOptions gopt;
    gopt.max_tokens = max_tokens;
    gopt.stop_ids = cli::ParseIds(stop_arg);
    if (stop_eos && c.eos_id >= 0) gopt.stop_ids.push_back(c.eos_id);

    std::vector<int32_t> gen;
    const float* last = logits.data() + (seq - 1) * c.vocab;
    const auto t2 = std::chrono::steady_clock::now();
    rpg::GenerateGreedy(model, &kv, &scratch, last, gopt, &gen);
    const auto t3 = std::chrono::steady_clock::now();
    const double gen_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::printf("[decode]  生成 " RPG_SZ " 个 token，%.1f ms（%.2f tok/s）\n",
                RPG_SZ_ARG(gen.size()), gen_ms,
                static_cast<double>(gen.size()) / (gen_ms / 1000.0));
    std::printf("[decode]  cache 现有 %lld 个位置\n",
                static_cast<long long>(kv.len()));
    RPG_CHECK(kv.len() == seq + static_cast<int64_t>(gen.size()),
              "cache 长度不变量被破坏：%lld != %lld + " RPG_SZ,
              static_cast<long long>(kv.len()),
              static_cast<long long>(seq), RPG_SZ_ARG(gen.size()));
    std::printf("[decode]  不变量 cache.len() == prompt + 生成数：OK\n");

    std::printf("\n[生成] ");
    for (size_t i = 0; i < gen.size(); ++i) {
      std::printf(i ? ",%d" : "%d", gen[i]);
    }
    std::printf("\n[生成] 解码: \"%s\"\n", model.tokenizer().Decode(gen).c_str());

    // ---- 重放收集每步 logits（并校验能重放出同一条序列）----
    std::vector<float> step;
    if (step_logits) {
      std::printf("\n[重放] 用另一份 cache 逐步 ForwardStep，收集每步 logits\n");
      rpg::KvCache kv2(c, cap);
      rpg::ForwardScratch sc2;
      std::vector<float> lg2;
      rpg::ForwardFull(model, ids.data(), seq, &sc2, &lg2, fopt, &kv2);

      // 第一步的分布 = prefill 最后一个位置那行
      std::vector<float> cur(lg2.data() + (seq - 1) * c.vocab,
                             lg2.data() + seq * c.vocab);
      for (size_t i = 0; i < gen.size(); ++i) {
        step.insert(step.end(), cur.begin(), cur.end());
        const int32_t got = rpg::ArgMax(cur.data(), c.vocab);
        RPG_CHECK(got == gen[i],
                  "重放对不上：第 " RPG_SZ " 步 argmax = %d，生成时是 %d —— "
                  "引擎不是纯函数了？先查这个，别拿这份 step_logits 去对拍",
                  RPG_SZ_ARG(i), got, gen[i]);
        if (i + 1 < gen.size()) {
          rpg::ForwardStep(model, gen[i], kv2.len(), &kv2, &sc2, &cur);
        }
      }
      std::printf("[重放] " RPG_SZ " 步全部复现，cache 到 %lld\n",
                  RPG_SZ_ARG(gen.size()), static_cast<long long>(kv2.len()));
    }

    // ---- 最后位置 top-K ----
    if (!logits.empty() && top_k > 0) {
      const float* lp = logits.data() + (seq - 1) * c.vocab;
      std::vector<int32_t> order(static_cast<size_t>(c.vocab));
      for (int32_t i = 0; i < c.vocab; ++i) order[static_cast<size_t>(i)] = i;
      const int k = std::min<int>(top_k, c.vocab);
      std::partial_sort(order.begin(), order.begin() + k, order.end(),
                        [lp](int32_t a, int32_t b) { return lp[a] > lp[b]; });
      std::printf("\n[最后位置 top-%d]\n", k);
      for (int i = 0; i < k; ++i) {
        const int32_t id = order[static_cast<size_t>(i)];
        std::printf("  #%d  id=%-7d logit=%9.4f  %s\n", i + 1, id, lp[id],
                    model.tokenizer().TokenOf(id).c_str());
      }
    }

    // ---- 输出 ----
    if (!out_prefix.empty()) {
      cli::WriteIds(out_prefix + ".tokens.txt", ids);
      cli::WriteIds(out_prefix + ".gen.txt", gen);
      std::printf("\n[输出] %s.tokens.txt / %s.gen.txt\n", out_prefix.c_str(),
                  out_prefix.c_str());
      if (step_logits) {
        const std::string pl = out_prefix + ".logits.npy";
        RPG_CHECK(npy::WriteF32(pl, {seq, c.vocab}, logits.data(), logits.size()),
                  "写不了 %s", pl.c_str());
        std::printf("[输出] %s  [%lld, %d]\n", pl.c_str(),
                    static_cast<long long>(seq), c.vocab);
        const int64_t n_steps =
            static_cast<int64_t>(step.size()) / static_cast<int64_t>(c.vocab);
        const std::string sp = out_prefix + ".step_logits.npy";
        RPG_CHECK(npy::WriteF32(sp, {n_steps, c.vocab}, step.data(), step.size()),
                  "写不了 %s", sp.c_str());
        std::printf("[输出] %s  [%lld, %d]\n", sp.c_str(),
                    static_cast<long long>(n_steps), c.vocab);
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "错误：%s\n", e.what());
    return 1;
  }
}
