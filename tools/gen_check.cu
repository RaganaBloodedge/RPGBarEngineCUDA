// gen_check —— P2 的判据工具（GPU 侧）：增量解码 + 贪心生成。
//
// 三件事，与 CPU 引擎的 tools/gen_check 一一对应，但多一层容差判据：
//
//   1. **KV 自检**：带 cache 的 prefill 与不带 cache 的全序列前向，logits 比
//      相对误差分位数。
//      ⚠️ CPU 侧这条是"**逐位相同**"（K/V 都是 fp32，两边算的是同一串数）。
//      GPU 侧 KV 是 fp16（DESIGN §4.2）⇒ 写进 cache 的那一刻就有舍入，逐位
//      相同**不可能**。所以判据变成 `p99 ≤ 1e-3 且 argmax 全一致`。
//      **这个降级是有意的**（换来每会话省 448 MiB），但它把这条自检变成了
//      "布局错误探测器"：fp16 舍入只有 1e-3 量级，cache 布局写错（写反了头/位置
//      顺序）会让误差跳到 1e0 量级 —— 一眼能分开。
//
//   2. prefill + 贪心生成 N 个 token，写 <前缀>.gen.txt（与参考侧逐 token 比）。
//
//   3. `--step-logits` 时逐步导出 logits（[返回 token 数, vocab]），供
//      tests/cmp_gen.py 在分歧处打印两边的 top-5 与两个 token 的 logit 差。
//
// 用法：
//   gen_check model.gguf --prompt-file p.txt --max-tokens 32 --out runs/gpu
//                        --step-logits
//   （⚠️ 这里刻意**不写 shell 的续行反斜杠** —— 行尾的 `\` 在 // 注释里会让
//     GCC 认为注释续到下一行，报 -Wcomment）
//
// 选项：
//   --prompt <文本>       用内置分词器编码
//   --prompt-file <文件>  从文件读提示词（中文走命令行会被转码搞坏）
//   --tokens a,b,c        直接给 token id（对拍时两边必须用同一串）
//   --max-tokens <N>      生成多少个（默认 32）
//   --max-seq <N>         KV Cache 容量，默认按 prompt + max_tokens 取整到 256
//   --max-pos <N>         RoPE 表覆盖的位置数，默认 4096
//   --stop <id,id>        停止 token（默认空 —— 对拍时两边都生成固定数量）
//   --stop-eos            用模型自带的 eos 当停止 token
//   --out <前缀>          写 <前缀>.tokens.txt / .gen.txt（+ 可选的 npy）
//   --step-logits         导出每步 logits 与 prefill 的 logits
//   --no-kv-check         跳过 cache 一致性自检（省一半时间）
//   --top <K>             打印最后位置 top-K（默认 5）
//   --info                只打印模型信息后退出

#include <algorithm>
#include <chrono>
#include <cmath>
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
#include "npy.h"
#include "qwen3_gpu.h"

namespace {

// 与 tests/cmp_layers.py / RPGBarEngine/tools/cmp_npy.py 相同的口径：
// 只看 |ref| > REL_FLOOR 的元素，算相对误差的分位数。
constexpr double kRelFloor = 1e-3;

struct RelStats {
  double median = 0, p99 = 0, mx = 0, amax = 0;
  size_t n = 0;      // 参与统计的元素数
  size_t skipped = 0;  // 因 |ref| <= REL_FLOOR 被跳过的
};

// numpy.percentile 的默认（linear 插值）—— 与参考脚本的口径对齐，否则两边
// 同一个 p99 会差出第三位有效数字，白让人怀疑。
double Percentile(std::vector<double>& v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double idx = p / 100.0 * static_cast<double>(v.size() - 1);
  const size_t lo = static_cast<size_t>(idx);
  const size_t hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (v[hi] - v[lo]) * (idx - static_cast<double>(lo));
}

RelStats RelStatsOf(const float* ours, const float* ref, int64_t n) {
  std::vector<double> rel;
  rel.reserve(static_cast<size_t>(n) / 4);
  RelStats s;
  for (int64_t i = 0; i < n; ++i) {
    const double a = static_cast<double>(ref[i]);
    const double diff = std::fabs(static_cast<double>(ours[i]) - a);
    if (diff > s.amax) s.amax = diff;
    const double av = std::fabs(a);
    if (av <= kRelFloor) {
      ++s.skipped;
      continue;
    }
    rel.push_back(diff / av);
  }
  s.n = rel.size();
  s.median = Percentile(rel, 50.0);
  s.mx = Percentile(rel, 100.0);
  s.p99 = Percentile(rel, 99.0);
  return s;
}

int32_t ArgMaxHost(const float* x, int64_t n) {
  int32_t best = 0;
  float bv = x[0];
  for (int64_t i = 1; i < n; ++i) {
    if (x[i] > bv) {
      bv = x[i];
      best = static_cast<int32_t>(i);
    }
  }
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf(
        "用法: gen_check <model.gguf> [选项]\n"
        "  --prompt <文本>      用分词器编码（默认）\n"
        "  --prompt-file <f>    从文件读提示词（绕开 argv 转码）\n"
        "  --tokens <a,b,c>     直接给 token id\n"
        "  --max-tokens <N>     生成数量（默认 32）\n"
        "  --max-seq <N>        KV Cache 容量（默认自动）\n"
        "  --max-pos <N>        RoPE 表覆盖的位置数（默认 4096）\n"
        "  --stop <id,id>       停止 token（默认无）\n"
        "  --stop-eos           用模型自带的 eos 当停止 token\n"
        "  --out <前缀>         写 <前缀>.tokens.txt / <前缀>.gen.txt\n"
        "  --step-logits        导出 prefill 与每步 logits（对拍用）\n"
        "  --no-kv-check        跳过 cache 一致性自检\n"
        "  --top <K>            打印最后位置 top-K（默认 5）\n"
        "  --info               只打印模型信息\n");
    return 2;
  }

  const std::string model_path = argv[1];
  std::string prompt = "The capital of France is";
  std::string tokens_arg;
  std::string out_prefix;
  std::string stop_arg;
  int32_t max_pos = 4096;
  int32_t max_tokens = 32;
  int64_t max_seq = 0;
  int top_k = 5;
  bool kv_check = true;
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
    else if (a == "--stop") stop_arg = next();
    else if (a == "--out") out_prefix = next();
    else if (a == "--step-logits") step_logits = true;
    else if (a == "--no-kv-check") kv_check = false;
    else if (a == "--top") top_k = std::atoi(next());
    else if (a == "--stop-eos") stop_eos = true;
    else if (a == "--info") info_only = true;
    else rpg::Throw("未知选项 %s", a.c_str());
  }

  try {
    const auto tl0 = std::chrono::steady_clock::now();
    rpg::GpuModel model = rpg::GpuModel::Load(model_path, max_pos);
    const auto tl1 = std::chrono::steady_clock::now();
    std::printf("%s", model.Describe().c_str());
    std::printf("  权重文件  : %s\n", model.path().c_str());
    std::printf("  加载用时  : %.1f s（含搬上显存）\n",
                std::chrono::duration<double>(tl1 - tl0).count());
    if (info_only) return 0;

    const rpg::ModelMeta& c = model.config();

    // ---- 输入 ----
    std::vector<int32_t> ids;
    if (!tokens_arg.empty()) {
      ids = cli::ParseIds(tokens_arg);
      std::printf("\n[输入] 直接指定 %lld 个 token id\n",
                  static_cast<long long>(ids.size()));
    } else {
      ids = model.tokenizer().Encode(prompt, /*parse_special=*/true);
      if (c.add_bos && c.bos_id >= 0) ids.insert(ids.begin(), c.bos_id);
      std::printf("\n[输入] \"%s\" → %lld 个 token\n", prompt.c_str(),
                  static_cast<long long>(ids.size()));
    }
    RPG_CHECK(!ids.empty(), "token 序列为空");
    const int64_t seq = static_cast<int64_t>(ids.size());

    // ---- KV Cache 容量 ----
    RPG_CHECK(max_tokens > 0, "max_tokens 必须为正");
    const int64_t want = seq + static_cast<int64_t>(max_tokens) + 8;
    const int64_t cap = max_seq > 0 ? max_seq : ((want + 255) / 256) * 256;
    RPG_CHECK(cap <= max_pos,
              "KV 容量 %lld 超过 RoPE 表覆盖的 %d —— 用 --max-pos 抬高，"
              "或调小 --max-seq",
              static_cast<long long>(cap), max_pos);

    rpg::GpuKvCache kv(c, cap);
    std::printf("[缓存] %s\n", kv.Describe().c_str());

    rpg::GpuScratch scratch;
    std::vector<float> logits;

    // ---- ① KV 自检 ----
    int kv_check_fail = 0;
    if (kv_check) {
      std::printf("\n========== ① KV 自检（带 cache vs 不带 cache）==========\n");
      std::vector<float> plain;
      rpg::GpuForwardFull(model, ids.data(), seq, &scratch, &plain);
      rpg::GpuKvCache kv2(c, cap);
      std::vector<float> cached;
      rpg::GpuForwardFull(model, ids.data(), seq, &scratch, &cached,
                          rpg::GpuForwardOptions(), &kv2);
      RPG_CUDA_CHECK(cudaDeviceSynchronize());

      const RelStats st = RelStatsOf(cached.data(), plain.data(),
                                     static_cast<int64_t>(plain.size()));
      // 满量程（整个 logits 张量的最大绝对值）。**判据用绝对误差对它取比值**，
      // 不用逐元素相对误差 —— 理由见下面那段说明。
      float amax_ref = 0.0f;
      for (float v : plain) amax_ref = std::max(amax_ref, std::fabs(v));
      const double full_scale_rel = st.amax / static_cast<double>(amax_ref);

      std::printf("  logits：%lld 个数值（跳过 %lld 个 |ref| <= %g 的）\n",
                  static_cast<long long>(st.n),
                  static_cast<long long>(st.skipped), kRelFloor);
      std::printf("  逐元素相对误差：中位 %-9.3g p99 %-9.3g max %-9.3g\n",
                  st.median, st.p99, st.mx);
      std::printf("  绝对误差：max|Δ| %.4g；满量程 max|ref| %.4g ⇒ 比值 %.3g\n",
                  st.amax, static_cast<double>(amax_ref), full_scale_rel);
      std::printf("  ⚠️ 逐元素相对误差**在 fp16 KV 下不是判据**：它被\"量级 0.5 上下\n"
                  "     的元素 + 固定的绝对误差\"主导，p99 会随 logits 分布变化而漂移。\n"
                  "     fp16 尾数 10 位 ⇒ 1 ULP 相对 2^-11 = 4.88e-4；实测比值就落在\n"
                  "     这个量级 ⇒ 判据取满量程的 1e-3（约 2 个 ULP，留 2 倍余量）。\n");
      const bool tol_ok = full_scale_rel <= 1e-3;
      std::printf("  判据 max|Δ| <= 1e-3 × max|ref|：%s（%.3g）\n",
                  tol_ok ? "OK" : "**超限**", full_scale_rel);
      if (!tol_ok) {
        std::printf("  ⇒ 这不是精度问题（fp16 舍入只到 5e-4 量级），是 cache 的\n"
                    "     写入/读取布局错了 —— 先跑 tools/kv_map_check 定位。\n");
        kv_check_fail = 1;
      }

      // 逐位置的绝对误差。**这是把"精度"与"布局"分开的那把尺子**：
      // fp16 舍入对每个位置只发生一次，误差量级不随位置变；布局错了则会让
      // 某个位置起突然跳到 1e0 量级（因为读到了别的头/别的位置）。
      std::printf("  逐位置 max|Δ|（前 12 个位置）：\n            ");
      for (int64_t t = 0; t < seq && t < 12; ++t) {
        double mx_t = 0;
        for (int32_t i = 0; i < c.vocab; ++i) {
          const double d = std::fabs(static_cast<double>(cached[t * c.vocab + i]) -
                                     static_cast<double>(plain[t * c.vocab + i]));
          if (d > mx_t) mx_t = d;
        }
        std::printf("%.2g ", mx_t);
      }
      std::printf("\n");

      int n_bad = 0;
      for (int64_t t = 0; t < seq; ++t) {
        if (ArgMaxHost(cached.data() + t * c.vocab, c.vocab) !=
            ArgMaxHost(plain.data() + t * c.vocab, c.vocab)) {
          ++n_bad;
        }
      }
      std::printf("  每个位置的 argmax 一致：%lld/%lld %s\n",
                  static_cast<long long>(seq - n_bad),
                  static_cast<long long>(seq), n_bad == 0 ? "OK" : "**不一致**");
      if (n_bad != 0) kv_check_fail = 1;
      std::printf("  自检结论：%s\n", kv_check_fail == 0 ? "通过" : "**失败**");
    }

    // ---- ② prefill + 贪心生成 ----
    kv.Reset();  // 自检用的是另一份 cache（kv2），这一份从头上卡
    rpg::GpuForwardOptions fopt;
    const auto t0 = std::chrono::steady_clock::now();
    rpg::GpuForwardFull(model, ids.data(), seq, &scratch, &logits, fopt, &kv);
    RPG_CUDA_CHECK(cudaDeviceSynchronize());
    const auto t1 = std::chrono::steady_clock::now();
    const double prefill_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("\n[prefill] %lld 个位置 %.1f ms（%.1f tok/s）\n",
                static_cast<long long>(seq), prefill_ms,
                static_cast<double>(seq) / (prefill_ms / 1000.0));

    rpg::GpuGenerateOptions gopt;
    gopt.max_tokens = max_tokens;
    gopt.stop_ids = cli::ParseIds(stop_arg);
    if (stop_eos && c.eos_id >= 0) gopt.stop_ids.push_back(c.eos_id);
    std::vector<float> step;
    if (step_logits) gopt.step_logits = &step;

    std::vector<int32_t> gen;
    // first_logits 用**显存**里 prefill 的最后一行（增量解码全程不回拷 logits）
    const float* last_dev = static_cast<const float*>(scratch.logits.data()) +
                            (seq - 1) * c.vocab;
    const auto t2 = std::chrono::steady_clock::now();
    rpg::GpuGenerateGreedy(model, &kv, &scratch, last_dev, gopt, &gen);
    RPG_CUDA_CHECK(cudaDeviceSynchronize());
    const auto t3 = std::chrono::steady_clock::now();
    const double gen_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::printf("[decode]  生成 %lld 个 token，%.1f ms（%.2f tok/s）\n",
                static_cast<long long>(gen.size()), gen_ms,
                static_cast<double>(gen.size()) / (gen_ms / 1000.0));
    std::printf("[decode]  cache 现有 %lld 个位置\n",
                static_cast<long long>(kv.len()));

    // 不变量：cache.len() == prompt 长度 + 返回的 token 数。少了它，下一轮的
    // 第一个 token 会悄悄变味（那种 bug 不崩，只会让剧情莫名跑偏）。
    RPG_CHECK(kv.len() == seq + static_cast<int64_t>(gen.size()),
              "cache 长度不变量被破坏：%lld != %lld + %lld",
              static_cast<long long>(kv.len()), static_cast<long long>(seq),
              static_cast<long long>(gen.size()));
    std::printf("[decode]  不变量 cache.len() == prompt + 生成数：OK\n");

    std::printf("\n[生成] ");
    for (size_t i = 0; i < gen.size(); ++i) {
      std::printf(i ? ",%d" : "%d", gen[i]);
    }
    std::printf("\n[生成] 解码: \"%s\"\n", model.tokenizer().Decode(gen).c_str());
    std::printf("\n[全文] \"%s%s\"\n", model.tokenizer().Decode(ids).c_str(),
                model.tokenizer().Decode(gen).c_str());

    // ---- 最后位置 top-K ----
    if (!logits.empty() && top_k > 0) {
      const float* last = logits.data() + (seq - 1) * c.vocab;
      std::vector<int32_t> order(static_cast<size_t>(c.vocab));
      for (int32_t i = 0; i < c.vocab; ++i) order[static_cast<size_t>(i)] = i;
      const int k = std::min<int>(top_k, c.vocab);
      std::partial_sort(order.begin(), order.begin() + k, order.end(),
                        [last](int32_t a, int32_t b) { return last[a] > last[b]; });
      std::printf("\n[最后位置 top-%d]\n", k);
      for (int i = 0; i < k; ++i) {
        const int32_t id = order[static_cast<size_t>(i)];
        std::printf("  #%d  id=%-7d logit=%9.4f  %s\n", i + 1, id, last[id],
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
        // prefill 的 logits 也写一份（[seq, vocab]）—— 参考侧同形状，便于
        // 在"第一个 token 就已经分歧"时直接看两边 prefill 的分布差。
        const std::string lp = out_prefix + ".logits.npy";
        RPG_CHECK(npy::WriteF32(lp, {seq, c.vocab}, logits.data(), logits.size()),
                  "写不了 %s", lp.c_str());
        std::printf("[输出] %s  [%lld, %d]\n", lp.c_str(),
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
    return kv_check_fail;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "错误：%s\n", e.what());
    return 1;
  }
}
