// gen_sample_check —— P3 第二条判据：**采样在真实生成循环里**。
//
// 与第一条（tools/sample_check）的分工：
//   * sample_check 把模型**摘掉**，验"同一份 logits + 同一个 seed ⇒ 同一个 token"
//     （逐位）。它不需要权重，几秒跑完，是采样正确性的主判据。
//   * 本工具把模型**装回去**，验的是"接进生成循环之后，那套采样还讲不讲道理"。
//     这里**不能**要求"与 CPU 版逐 token 相同" —— DESIGN §6 第 10 条说得很清楚：
//     接在模型后面的采样，两边 logits 先差了 fp16 的舍入，采样把差异放大成
//     "整段序列不同"。硬去对拍只会得到一个永远查不清的混合指标。
//
// 于是换一批**只看自己**的判据（都是严格的，不是"差不多"）：
//
//   A. **贪心基准**（`temperature = 0`）：跑一遍留作参照。
//   B. **可复现**（同 seed 两遍）：`T = 1` 下同一份 prompt、同一个 seed 跑两遍，
//      生成的 token 序列必须**逐位相同**。这条直接钉住引擎的全局约定
//      "**没有有状态的 RNG，引擎是纯函数**"（CPU 版 DESIGN 的"随机性归属"）——
//      P3 之前 seed 根本没进生成循环，这条是无从验证的。
//   C. **随机性真的在起作用**（换 seed）：换一个 seed，序列必须不同。
//      ⚠️ 这条是**概率性**的，不是严格判据 —— T = 1、32 个 token 全撞上的概率
//      极低但不为零，所以报告里写成"观察到不同"，并在注释里说清它的性质。
//      它防的是一类具体错误：把 seed 忘了接下去（那样 B 照样绿、C 才红）。
//   D. **筛选参数真的传到循环里了**：
//      D1（严格）`T = 0` 且 `top_k = 40, top_p = 0.9` ⇒ 与贪心基准**逐位相同**
//         —— 贪心分支在筛选之前就返回了，这正是真实宿主路径（宿主只调
//         temperature，top_k / top_p 由引擎钉住）。
//      D2（必要条件）`T = 1, top_k = 2` ⇒ 每一步选出的 token 必须落在该步 logits
//         的**前 2 名**里。红了**一定**错；过了只说明没抓到错。
//      D3（概率性）`top_k = 2` 与 `top_k = 0` 的输出必须不同 —— 防的是"参数在
//         传递链上被丢了"。同 seed 同 logits 下 32 步全撞上的概率极低。
//      ⚠️ 这一节最初的写法是拿 `top_k = 1` 当"支持集只剩一个元素 ⇒ 必然 argmax"
//      来断言的，**红了一片**：契约（sampler.h）里 top_k 的生效条件是
//      `top_k > 1 && top_k < n`，所以 **1 与 0 一样算"关闭"**。判据的前提也要去
//      读契约，不能凭常识。
//   E. **counter 用的是绝对位置**（暖缓存 = 冷缓存）：把第一遍的输出**接在 prompt
//      后面**当新 prompt，冷着再跑一遍。同一个 seed 下，第二遍的第 i 个 token
//      对应的绝对位置是 `L + 1 + i`（L = 原 prompt 长度），正是第一遍的第
//      `i + 1` 个 —— 于是必须有
//          out2[i] == out1[i + 1]      （i = 0 .. N-2）
//      这条是 `SamplerGpu` 那句"counter 用绝对下标、不用本轮第几个"的**端到端
//      证据**。它比 B/C 都强：B 只要求同一台机器上跑两遍一样，E 要求"两条不同的
//      缓存路径（暖续写 / 冷重算）给出同一条随机流"。
//      ⚠️ 但 E 比较的是**两次真的前向**（批量 prefill 的最后一行 vs 单行增量），
//      理论上只有 fp16 KV 的舍入差；若它红了，先看打印的诊断（分歧位置的
//      两个候选 token 分数差），别直接归因到采样。见文件末尾的实测记录。
//
// 用法：
//   gen_sample_check model.gguf [--prompt-file p.txt] [--max-tokens 32]
//                     [--max-pos 4096] [--seed 12345]
//   （⚠️ 注释里不写 shell 的续行反斜杠 —— 行尾的 `\` 会让 GCC 认为注释续行）

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
#include "qwen3_gpu.h"
#include "sampler.h"

namespace {

// 一次完整的"prefill + 生成"。每调用一次都从**全新的 cache** 开始 ——
// 这是刻意的：判据 B/C/E 比较的是"两次独立的、从头开始的运行"。
struct RunResult {
  std::vector<int32_t> ids;   // 生成的 token（不含 prompt）
  int64_t kv_len = 0;         // 结束时 cache 的长度
  double ms = 0;              // 生成耗时（不含 prefill）
};

RunResult RunOnce(const rpg::GpuModel& m, const std::vector<int32_t>& prompt,
                  const rpg::SampleOptions& sample, uint64_t seed,
                  int32_t max_tokens, int64_t cap,
                  std::vector<float>* step_logits = nullptr) {
  const rpg::ModelMeta& c = m.config();
  const int64_t seq = static_cast<int64_t>(prompt.size());

  rpg::GpuKvCache kv(c, cap);
  rpg::GpuScratch s;
  // logits 传 nullptr：只算到显存（整个 decode 循环一次 607 KB 的搬运都不要）。
  rpg::GpuForwardFull(m, prompt.data(), seq, &s, /*logits=*/nullptr,
                      rpg::GpuForwardOptions(), &kv);
  RPG_CHECK(kv.len() == seq, "prefill 后 kv.len=%lld，应为 %lld",
            static_cast<long long>(kv.len()), static_cast<long long>(seq));

  rpg::GpuGenerateOptions g;
  g.max_tokens = max_tokens;
  g.sample = sample;
  g.seed = seed;
  g.step_logits = step_logits;  // 只要 D2 用得上（形状 [生成数, vocab]）
  const float* last = static_cast<const float*>(s.logits.data()) +
                      (seq - 1) * static_cast<int64_t>(c.vocab);

  RunResult r;
  const auto t0 = std::chrono::steady_clock::now();
  rpg::GpuGenerate(m, &kv, &s, last, g, &r.ids);
  RPG_CUDA_CHECK(cudaDeviceSynchronize());
  r.ms = std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0).count();
  r.kv_len = kv.len();
  return r;
}

// 某一步 logits 行里最大的 k 个下标（并列按下标升序 —— 与采样无关，只用来
// 判断"选出来的 token 在不在前 k 里"）。
std::vector<int32_t> TopKIds(const float* row, int32_t n, int k) {
  std::vector<int32_t> idx(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) idx[static_cast<size_t>(i)] = i;
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [row](int32_t a, int32_t b) {
                      if (row[a] != row[b]) return row[a] > row[b];
                      return a < b;  // 同值时确定性排序
                    });
  idx.resize(static_cast<size_t>(k));
  return idx;
}

void PrintIds(const char* tag, const std::vector<int32_t>& v) {
  std::printf("  %s [", tag);
  for (size_t i = 0; i < v.size(); ++i) {
    std::printf(i ? ",%d" : "%d", v[i]);
  }
  std::printf("]\n");
}

// 两个序列的前 len 个是否逐位相同；不同则把位置打出来。
bool SamePrefix(const std::vector<int32_t>& a, const std::vector<int32_t>& b,
                size_t len, std::vector<size_t>* bad) {
  bad->clear();
  for (size_t i = 0; i < len; ++i) {
    if (a[i] != b[i]) bad->push_back(i);
  }
  return bad->empty();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf(
        "用法: gen_sample_check <model.gguf> [选项]\n"
        "  --prompt <文本>      用分词器编码（默认）\n"
        "  --prompt-file <f>    从文件读提示词（绕开 argv 转码）\n"
        "  --max-tokens <N>     每次生成多少个（默认 32）\n"
        "  --max-seq <N>        KV Cache 容量（默认按需取整到 256）\n"
        "  --max-pos <N>        RoPE 表覆盖的位置数（默认 4096）\n"
        "  --seed <N>           基准 seed（默认 12345）\n");
    return 2;
  }

  const std::string model_path = argv[1];
  std::string prompt = "你是一位地下城的城主。冒险者推开了石门，请你描述他看到的一切。";
  int32_t max_pos = 4096;
  int32_t max_tokens = 32;
  int64_t max_seq = 0;
  uint64_t seed = 12345;

  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) rpg::Throw("选项 %s 后面缺少参数", a.c_str());
      return argv[++i];
    };
    if (a == "--prompt") prompt = next();
    else if (a == "--prompt-file") prompt = cli::ReadFile(next());
    else if (a == "--max-tokens") max_tokens = std::atoi(next());
    else if (a == "--max-seq") max_seq = std::atoll(next());
    else if (a == "--max-pos") max_pos = std::atoi(next());
    else if (a == "--seed") seed = static_cast<uint64_t>(std::strtoull(next(), nullptr, 10));
    else rpg::Throw("未知选项 %s", a.c_str());
  }

  try {
    rpg::GpuModel model = rpg::GpuModel::Load(model_path, max_pos);
    const rpg::ModelMeta& c = model.config();
    std::printf("%s", model.Describe().c_str());

    std::vector<int32_t> ids = model.tokenizer().Encode(prompt, /*parse_special=*/true);
    if (c.add_bos && c.bos_id >= 0) ids.insert(ids.begin(), c.bos_id);
    RPG_CHECK(!ids.empty(), "token 序列为空");
    const int64_t seq = static_cast<int64_t>(ids.size());
    RPG_CHECK(max_tokens > 0, "max_tokens 必须为正");

    // E 要用到 seq + max_tokens 个位置，再加一次生成的余量。
    const int64_t want = seq + 2 * static_cast<int64_t>(max_tokens) + 8;
    const int64_t cap = max_seq > 0 ? max_seq : ((want + 255) / 256) * 256;
    RPG_CHECK(cap <= max_pos, "KV 容量 %lld 超过 RoPE 表覆盖的 %d",
              static_cast<long long>(cap), max_pos);

    std::printf("\n[输入] %lld 个 token；每次生成 %d 个；seed = %llu\n",
                static_cast<long long>(seq), max_tokens,
                static_cast<unsigned long long>(seed));

    rpg::SampleOptions greedy;
    greedy.temperature = 0.0f;

    rpg::SampleOptions t1;
    t1.temperature = 1.0f;
    // top_k / top_p 显式关掉：这一步要验的是"分布采样 + 随机源"，不是筛选。
    t1.top_k = 0;
    t1.top_p = 1.0f;

    int fail = 0;

    // ---------------------------------------------------------------- A
    std::printf("\n========== A 贪心基准（T = 0）==========\n");
    const RunResult a = RunOnce(model, ids, greedy, seed, max_tokens, cap);
    std::printf("  耗时 %.1f ms（%.2f tok/s）；cache %lld 个位置\n", a.ms,
                a.ids.size() / (a.ms / 1000.0),
                static_cast<long long>(a.kv_len));
    PrintIds("token", a.ids);
    std::printf("  解码: \"%s\"\n", model.tokenizer().Decode(a.ids).c_str());
    // P2 就有的不变量，P3 换成新循环之后必须继续成立。
    if (a.kv_len != seq + static_cast<int64_t>(a.ids.size())) {
      std::printf("  **失败**：cache.len() != prompt + 生成数\n");
      ++fail;
    } else {
      std::printf("  不变量 cache.len() == prompt + 生成数：OK\n");
    }

    // ---------------------------------------------------------------- B
    std::printf("\n========== B 可复现（同 seed 两遍，T = 1）==========\n");
    const RunResult b1 = RunOnce(model, ids, t1, seed, max_tokens, cap);
    const RunResult b2 = RunOnce(model, ids, t1, seed, max_tokens, cap);
    PrintIds("第 1 遍", b1.ids);
    PrintIds("第 2 遍", b2.ids);
    std::printf("  解码: \"%s\"\n", model.tokenizer().Decode(b1.ids).c_str());
    {
      std::vector<size_t> bad;
      if (SamePrefix(b1.ids, b2.ids, b1.ids.size(), &bad)) {
        std::printf("  %lld/%lld 个 token 逐位相同：OK（引擎仍是纯函数）\n",
                    static_cast<long long>(b1.ids.size()),
                    static_cast<long long>(b1.ids.size()));
      } else {
        std::printf("  **失败**：同 seed 两遍有 %lld 处不同，位置：",
                    static_cast<long long>(bad.size()));
        for (size_t i = 0; i < bad.size() && i < 8; ++i) {
          std::printf(" %lld(%d≠%d)", static_cast<long long>(bad[i]),
                      b1.ids[bad[i]], b2.ids[bad[i]]);
        }
        std::printf("\n  ⇒ 引擎里混进了有状态的东西（缓存没清 / RNG 有内部状态）。\n");
        ++fail;
      }
    }

    // ---------------------------------------------------------------- C
    std::printf("\n========== C 随机性真的在起作用（换 seed）==========\n");
    const RunResult c1 = RunOnce(model, ids, t1, seed + 1, max_tokens, cap);
    PrintIds("seed+1", c1.ids);
    std::printf("  解码: \"%s\"\n", model.tokenizer().Decode(c1.ids).c_str());
    {
      size_t diff = 0;
      for (size_t i = 0; i < b1.ids.size() && i < c1.ids.size(); ++i) {
        if (b1.ids[i] != c1.ids[i]) ++diff;
      }
      std::printf("  与 seed 基准不同的位置：%lld/%lld\n",
                  static_cast<long long>(diff),
                  static_cast<long long>(b1.ids.size()));
      if (diff == 0) {
        std::printf("  **失败**：换 seed 得到完全一样的序列 —— seed 根本没接下去。\n");
        ++fail;
      } else {
        std::printf("  OK（⚠️ 这条是概率性判据：它防的是 seed 忘了接，不是精度）\n");
      }
    }

    // ---------------------------------------------------------------- D
    std::printf("\n========== D 筛选参数真的传到循环里了 ==========\n");
    // ⚠️ 这一节的原始写法是错的，值得记一笔：我原本拿 `top_k = 1` 当
    // "支持集只剩一个元素"的用例，断言它与贪心逐位相同。**红了一片**。
    // 查契约（sampler.h / sampler.cc）：top_k 的生效条件是
    //     `opt.top_k > 1 && opt.top_k < n`
    // ⇒ **`top_k = 1` 与 `0` 一样算"关闭"**（注释只写了 "0 = 关闭"）。
    // 于是那遍跑出来的就是纯温度采样，与 `top_k = 0` 逐位相同 —— 判据红了，
    // 代码没错。教训：**判据的前提也要去读契约，不能凭常识**（"只留 1 个"直觉上
    // 就该等于 argmax，但上游没这么实现；而 sampler.h 是**同源副本**，不许改）。
    // 换成下面三条，都建立在读过的契约上。
    {
      // D1（严格）：T = 0 时筛选参数必须**完全不参与** —— 贪心分支在
      // FilterToProbs 之前就返回了。这是真实宿主路径：宿主只调 temperature，
      // top_k / top_p 由引擎钉住（0 / 0.95），于是"不传 temperature 的老请求"
      // 必须逐位不变。
      rpg::SampleOptions g2;
      g2.temperature = 0.0f;
      g2.top_k = 40;
      g2.top_p = 0.9f;
      const RunResult d1 = RunOnce(model, ids, g2, seed, max_tokens, cap);
      std::printf("  D1 T=0 且 top_k=40,top_p=0.9 与贪心基准比：\n");
      std::vector<size_t> bad;
      if (SamePrefix(a.ids, d1.ids, a.ids.size(), &bad)) {
        std::printf("     逐位相同：OK（贪心分支不看筛选参数 —— 老请求行为不变）\n");
      } else {
        std::printf("     **失败**：%lld 处不同（第一个位置 %lld）\n",
                    static_cast<long long>(bad.size()),
                    static_cast<long long>(bad.empty() ? 0 : bad[0]));
        ++fail;
      }
    }
    {
      // D2（必要条件）：T = 1、top_k = 2 ⇒ 每一步选出的 token 必须落在该步
      // logits 的**前 2 名**里。红了**一定**是错的（支持集都算错了）；
      // 过了只说明"没抓到错" —— 因为温度 1 的分布本来也常落在前 2 名里。
      rpg::SampleOptions t1_k2 = t1;
      t1_k2.top_k = 2;
      std::vector<float> d_step;
      const RunResult d2 = RunOnce(model, ids, t1_k2, seed, max_tokens, cap, &d_step);
      PrintIds("T=1,top_k=2", d2.ids);
      RPG_CHECK(static_cast<int64_t>(d_step.size()) ==
                    static_cast<int64_t>(d2.ids.size()) * c.vocab,
                "step logits 形状不对（%lld 个，应为 %lld）",
                static_cast<long long>(d_step.size()),
                static_cast<long long>(d2.ids.size()) * c.vocab);
      int outside = 0;
      for (size_t i = 0; i < d2.ids.size(); ++i) {
        const std::vector<int32_t> top2 =
            TopKIds(d_step.data() + i * static_cast<size_t>(c.vocab), c.vocab, 2);
        if (d2.ids[i] != top2[0] && d2.ids[i] != top2[1]) {
          ++outside;
          if (outside <= 4) {
            std::printf("     第 %lld 步选中 %d，但前 2 名是 {%d, %d}\n",
                        static_cast<long long>(i), d2.ids[i], top2[0], top2[1]);
          }
        }
      }
      if (outside == 0) {
        std::printf("     每一步都落在该步前 2 名内：OK（%lld/%lld）\n",
                    static_cast<long long>(d2.ids.size()),
                    static_cast<long long>(d2.ids.size()));
      } else {
        std::printf("     **失败**：%d 步落在前 2 名之外 ⇒ top_k 没生效\n", outside);
        ++fail;
      }
      // D3（概率性）：top_k=2 与 top_k=0 的输出必须不同 —— 防的是"参数在
      // 传递链上被丢了"。同 seed 同 logits 下，剪掉尾巴会改变逆变换落点，
      // 32 步全撞上的概率极低；但它是概率判据，报告里如实写。
      size_t same = 0;
      for (size_t i = 0; i < d2.ids.size() && i < b1.ids.size(); ++i) {
        if (d2.ids[i] == b1.ids[i]) ++same;
      }
      std::printf("     与 top_k=0 那遍相同的位置：%lld/%lld\n",
                  static_cast<long long>(same),
                  static_cast<long long>(d2.ids.size()));
      if (same == d2.ids.size()) {
        std::printf("     **失败**：完全一样 ⇒ top_k 八成没传下去（种子/参数链断了）\n");
        ++fail;
      } else {
        std::printf("     OK（⚠️ 概率性判据）\n");
      }
    }

    // ---------------------------------------------------------------- E
    std::printf("\n========== E counter 用绝对位置（暖缓存 = 冷缓存）==========\n");
    std::printf("  做法：把第一遍的第 1 个 token 接在 prompt 后面当新 prompt，\n"
                "        冷着再跑一遍。同一个 seed 下第 2 遍的第 i 个 token 对应的\n"
                "        绝对位置 = 第 1 遍第 i+1 个的位置 ⇒ 序列应当整体错一位相同。\n");
    {
      std::vector<int32_t> ids2 = ids;
      ids2.push_back(b1.ids[0]);  // 接上第一遍的第一个 token
      const RunResult e = RunOnce(model, ids2, t1, seed, max_tokens, cap);
      PrintIds("原 prompt   ", b1.ids);
      PrintIds("接一个再跑  ", e.ids);
      const size_t n_cmp = b1.ids.size() > 1 ? b1.ids.size() - 1 : 0;
      std::vector<size_t> bad;
      // 比较 out1[1..] 与 out2[0..]。
      std::vector<int32_t> shifted(b1.ids.begin() + 1, b1.ids.end());
      if (SamePrefix(shifted, e.ids, n_cmp, &bad)) {
        std::printf("  out1[1..] == out2[0..]（%lld 个）逐位相同：OK\n",
                    static_cast<long long>(n_cmp));
      } else {
        std::printf("  **失败**：%lld/%lld 处不同，位置：",
                    static_cast<long long>(bad.size()),
                    static_cast<long long>(n_cmp));
        for (size_t i = 0; i < bad.size() && i < 8; ++i) {
          std::printf(" %lld(%d≠%d)", static_cast<long long>(bad[i]),
                      shifted[bad[i]], e.ids[bad[i]]);
        }
        std::printf("\n  ⇒ 分歧点之前的一致说明 counter 是对的，分歧点更像 fp16 舍入\n"
                    "     把 u 推过了边界；两边的概率都接近时属于这条判据的已知边界。\n");
        ++fail;
      }
    }

    std::printf("\n==============================================================\n");
    if (fail == 0) {
      std::printf("P3 的生成循环判据（A-E）全部通过。\n");
    } else {
      std::printf("P3 的生成循环判据有 %d 项未通过（见上）。\n", fail);
    }
    return fail == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "错误：%s\n", e.what());
    return 1;
  }
}
