// sample_check —— P3 判据工具（GPU 侧）：设备端采样的独立对拍。
//
// ⚠️ **不需要模型、不需要权重**，几秒就跑完。理由见 sampler_gpu.h「判据」那一节：
//
//     "采样不能逐 token 对拍"针对的是**接在模型后面**的采样（两边 logits 本来就差
//     fp16 的舍入）。把模型摘掉、两边喂**同一份从文件读进来的 logits**，判据立刻
//     回来了：同一个 seed ⇒ 同一个 token，且要求**逐位相同**。
//
// 于是采样的每一步都能与模型的数值精度解耦地钉死 —— 和 P2 里 kv_map_check 把
// "cache 布局"与"fp16 精度"分开是同一个套路。
//
// 用法：
//   sample_check --out DIR [--vocab N]
//
//   --vocab 默认 4096；判据脚本会再跑一遍 151936（真实词表），用来验证二分搜索
//   在**整个 uint32 key 空间**上都收敛（小词表测不出"中点溢出"那类 bug）。
//
// 产物（目录里这一套，参考侧写 ref_* 前缀的另一套）：
//   cases.tsv        用例矩阵 —— 两边**读同一份**，避免参数在两边各写一遍而漂掉
//   logits.bin       原始 f32 [cases, vocab]，两边读的是**同一批比特**
//   logits.npy       同一批数据的 npy 版（只为了排查时能直接 np.load 看一眼）
//   gpu_probs.npy    筛选后的概率分布 [cases, vocab]
//   gpu_tokens.txt   每行一个用例的采样结果（kUCount 个 token）
//   gpu_u.npy        每个用例用到的 u [cases, kUCount]
//
// 判据（tests/cmp_sample.py 里逐条判）：
//   ① u 表      逐位相同（纯整数运算，没有浮点误差可讲）
//   ② 概率分布  满量程误差 ≤ 1e-5，且**保留集合完全相同**（哪些 token 没被筛掉
//               是个离散判断，比"数值接近"更能说明问题）
//   ③ token     逐位相同；若分歧，必须打印 u 与两侧累积概率，
//               只有 u 距边界 < 1e-6 才允许存在（1 ULP 足以推过去的位置）

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cli_util.h"
#include "common.h"
#include "cuda_common.h"
#include "npy.h"
#include "sampler.h"
#include "sampler_gpu.h"

namespace {

// 用例矩阵。每一行都在考一件具体的事，不是随便填的数：
//   * temperature = 0        → 贪心：走 argmax 分支，**不消耗随机数**
//   * temperature = 0 + 筛选 → 同样走 argmax（筛选参数该被整段绕过）
//   * top_k = 2              → 只留两个，考"阈值取第 k 大值、并列一起留"
//   * top_p 很小             → 切得狠，考累积概率的切点
//   * temperature 极小/极大  → 分布近乎 one-hot / 近乎均匀，切点在两个极端
//   * tie = true             → 值量化到 1/8 网格 ⇒ 大量**精确并列**。
//                              ⚠️ 这一行必须 top_p = 1（关闭核采样）：top_p 在
//                              并列元素之间的取舍本来是未定义的（CPU 那边取决于
//                              std::sort 的同值顺序），拿来判"逐位相同"是错的。
struct Case {
  float temperature;
  int32_t top_k;
  float top_p;
  uint64_t seed;
  bool tie;
  const char* what;
};

const Case kCases[] = {
    {0.0f, 0, 1.0f, 11, false, "贪心（T=0）"},
    {0.0f, 8, 0.9f, 12, false, "贪心 + 筛选参数（应被绕过）"},
    {1.0f, 0, 1.0f, 13, false, "纯分布"},
    {0.7f, 40, 1.0f, 14, false, "温度 + top_k=40"},
    {1.3f, 0, 0.9f, 15, false, "温度 + top_p=0.9"},
    {0.5f, 20, 0.95f, 16, false, "三者都开"},
    {1.0f, 2, 1.0f, 17, false, "top_k=2（阈值边界）"},
    {1.0f, 0, 0.05f, 18, false, "top_p=0.05（切得狠）"},
    {0.001f, 0, 1.0f, 19, false, "温度极小（近乎 one-hot）"},
    {5.0f, 0, 1.0f, 20, false, "温度极大（近乎均匀）"},
    {1.0f, 8, 1.0f, 21, true, "精确并列 + top_k=8"},
    {0.0f, 0, 1.0f, 22, true, "精确并列 + 贪心（argmax 取最小下标）"},
};

constexpr int kUCount = 16;  // 每个用例抽 16 个 u

constexpr int kNCases = static_cast<int>(sizeof(kCases) / sizeof(kCases[0]));

// logits 的生成：确定性（seed 决定），范围 [-8, 8]。
// ⚠️ 用 DevSplitMix64 而不是 std::mt19937：两边（本工具与参考侧的用例矩阵）只需要
// 同一批**比特**，而"谁生成"是唯一的 —— logits 写进文件后两边都读它。
// 这里用哪个发生器其实无关紧要，但用引擎自己的那个省得再引一个概念。
void GenerateLogits(uint64_t seed, bool tie, int64_t n, float* out) {
  for (int64_t i = 0; i < n; ++i) {
    const uint64_t h = rpg::DevSplitMix64(seed * 1000003ull + static_cast<uint64_t>(i));
    const float x = static_cast<float>(h >> 40) * (1.0f / 16777216.0f);  // [0,1)
    float v = (x - 0.5f) * 16.0f;                                        // [-8, 8)
    if (tie) {
      // 量化到 1/8 网格 ⇒ 4096 个元素落在 129 个值上，必然出现精确并列。
      v = std::round(v * 8.0f) / 8.0f;
    }
    out[i] = v;
  }
}

// 反量化回来说一句：采样只关心"相对大小"，logits 的整体尺度会随温度缩放 ——
// 所以这里的取值区间（正负 8）是随手定的，不需要与真实模型的 logits 尺度一致。
// ⚠️ 但**并列**必须真实出现（tie 用例），因为 top_k 的"并列一起保留"是被判的语义。

int ArgMaxHost(const float* x, int64_t n) {
  int best = 0;
  for (int64_t i = 1; i < n; ++i) {
    if (x[i] > x[best]) best = static_cast<int>(i);
  }
  return best;
}

void PrintUsage() {
  std::printf(
      "用法：sample_check --out DIR [--vocab N]\n"
      "\n"
      "  --out DIR   写 cases.tsv / logits.bin(.npy) / gpu_probs.npy /\n"
      "              gpu_tokens.txt / gpu_u.npy\n"
      "  --vocab N   词表长度（默认 4096；判据脚本另跑一遍 151936）\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string out_dir;
  int64_t vocab = 4096;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      RPG_CHECK(i + 1 < argc, "选项 %s 缺参数", a.c_str());
      return argv[++i];
    };
    if (a == "--out") {
      out_dir = next();
    } else if (a == "--vocab") {
      vocab = std::strtoll(next().c_str(), nullptr, 10);
    } else if (a == "-h" || a == "--help") {
      PrintUsage();
      return 0;
    } else {
      std::fprintf(stderr, "不认识的选项：%s\n", a.c_str());
      PrintUsage();
      return 2;
    }
  }
  RPG_CHECK(!out_dir.empty(), "必须给 --out DIR");
  RPG_CHECK(vocab > 0, "--vocab 必须为正");

  const std::string prefix = out_dir + "/";

  // cases.tsv —— 用例矩阵**写出来给参考侧读**，两边不各写一遍参数表。
  // 否则改了这里的 T/top_k 而忘了那边，判据会报"分布不同"，而人先去查 kernel。
  {
    std::string body = "# idx\ttemperature\ttop_k\ttop_p\tseed\tie\tu_count\tn_vocab\n";
    for (int c = 0; c < kNCases; ++c) {
      const Case& k = kCases[c];
      body += std::to_string(c) + "\t" + std::to_string(k.temperature) + "\t" +
              std::to_string(k.top_k) + "\t" + std::to_string(k.top_p) + "\t" +
              std::to_string(k.seed) + "\t" + (k.tie ? "1" : "0") + "\t" +
              std::to_string(kUCount) + "\t" + std::to_string(vocab) + "\n";
    }
    RPG_CHECK(cli::WriteText(prefix + "cases.tsv", body), "写不了 cases.tsv");
  }

  std::printf("设备端采样对拍（不需要模型）\n");
  std::printf("  用例 %d 个，词表 %lld，每例 %d 个 u\n", kNCases,
              static_cast<long long>(vocab), kUCount);

  // ---- 生成 logits（写文件：两边读同一批比特）----
  std::vector<float> logits(static_cast<size_t>(kNCases) * static_cast<size_t>(vocab));
  for (int c = 0; c < kNCases; ++c) {
    GenerateLogits(kCases[c].seed, kCases[c].tie, vocab,
                   logits.data() + static_cast<size_t>(c) * vocab);
  }
  {
    FILE* f = std::fopen((prefix + "logits.bin").c_str(), "wb");
    RPG_CHECK(f != nullptr, "写不了 logits.bin");
    const size_t wrote = std::fwrite(logits.data(), sizeof(float), logits.size(), f);
    std::fclose(f);
    RPG_CHECK(wrote == logits.size(), "logits.bin 写少了");
  }
  RPG_CHECK(npy::WriteF32(prefix + "logits.npy", {kNCases, vocab}, logits.data(),
                          logits.size()),
            "写 logits.npy 失败");

  // ---- 显存上的对象 ----
  rpg::DeviceBuffer d_logits(logits.size() * sizeof(float));
  rpg::DeviceBuffer d_probs(static_cast<size_t>(vocab) * sizeof(float));
  rpg::DeviceBuffer d_idx(sizeof(int32_t));
  d_logits.Upload(logits.data(), d_logits.size());

  const float* logits_dev = static_cast<const float*>(d_logits.data());
  float* probs_dev = static_cast<float*>(d_probs.data());
  int32_t* idx_dev = static_cast<int32_t*>(d_idx.data());

  std::vector<float> probs(static_cast<size_t>(kNCases) * static_cast<size_t>(vocab));
  std::vector<float> us(static_cast<size_t>(kNCases) * kUCount);
  std::string tokens_txt;

  std::printf("\n%-4s %-28s %-6s %-6s %-6s %-8s %s\n", "用例", "内容", "T", "top_k",
              "top_p", "保留数", "概率最大的 3 个 token");
  for (int c = 0; c < kNCases; ++c) {
    const Case& k = kCases[c];
    const size_t off = static_cast<size_t>(c) * static_cast<size_t>(vocab);
    const float* row_dev = logits_dev + off;

    rpg::SampleOptions opt;
    opt.temperature = k.temperature;
    opt.top_k = k.top_k;
    opt.top_p = k.top_p;

    // ① 概率分布：**只算一次**，供判据逐元素比（GpuSampleToken 内部还会再算一遍，
    //    那是为了走"对外的那条接口"，不是绕过它去直接调 kernel）。
    rpg::GpuFilterToProbs(row_dev, vocab, opt, probs_dev);
    d_probs.Download(probs.data() + off,
                     static_cast<size_t>(vocab) * sizeof(float));

    // ② 采样：每个 u 一个 token。
    std::string line;
    for (int u = 0; u < kUCount; ++u) {
      const float uu = rpg::DevCounterUniform(k.seed, static_cast<uint64_t>(u));
      int32_t tok = -1;
      rpg::GpuSampleToken(row_dev, vocab, opt, k.seed, static_cast<uint64_t>(u),
                          probs_dev, idx_dev);
      d_idx.Download(&tok, sizeof(int32_t));
      RPG_CHECK(tok >= 0 && tok < vocab, "采样给出的下标越界：%d", tok);
      us[static_cast<size_t>(c) * kUCount + u] = uu;
      line += std::to_string(tok);
      line += (u + 1 == kUCount) ? "\n" : " ";

      // 顺带一条内部一致性：贪心用例里每个 u 都必须给出 argmax —— 因为
      // "贪心不消耗随机数"意味着结果与 u 完全无关。这条不成立就说明
      // temperature 的分支判断写错了（而且是那种"看着也在工作"的错）。
      if (!(k.temperature > 0.0f)) {
        const int want = ArgMaxHost(logits.data() + off, vocab);
        RPG_CHECK(tok == want, "贪心用例 %d 给出的 token %d != argmax %d（u=%g）",
                  c, tok, want, static_cast<double>(uu));
      }
    }
    tokens_txt += line;

    // 打印两个诊断量（判据在 Python 侧，这里只是让人一眼看出这一行是干什么的）：
    //   * 保留集合大小（有多少 token 没被筛掉）—— top_k/top_p 生效与否看这个；
    //   * 概率最大的 3 个 token。
    // ⚠️ 只打印概率 > 0 的：筛掉的那些是 0，按"最大"排下来会把它们排进前三，
    // 看起来像"采样给出了一个概率为 0 的 token"。
    int64_t kept = 0;
    for (int64_t i = 0; i < vocab; ++i) {
      if (probs[off + i] > 0.0f) ++kept;
    }
    std::vector<int32_t> order;
    order.reserve(static_cast<size_t>(kept));
    for (int64_t i = 0; i < vocab; ++i) {
      if (probs[off + i] > 0.0f) order.push_back(static_cast<int32_t>(i));
    }
    const int show = std::min<int>(3, static_cast<int>(order.size()));
    std::partial_sort(order.begin(), order.begin() + show, order.end(),
                      [&](int32_t a, int32_t b) {
                        if (probs[off + a] != probs[off + b]) {
                          return probs[off + a] > probs[off + b];
                        }
                        return a < b;  // 并列取小下标，与 ArgMax 同语义
                      });
    std::string top3;
    for (int t = 0; t < show; ++t) {
      top3 += std::to_string(order[t]) + " ";
    }
    std::printf("%-4d %-28s %-6g %-6d %-6g %-8lld %s\n", c, k.what,
                static_cast<double>(k.temperature), k.top_k,
                static_cast<double>(k.top_p), static_cast<long long>(kept),
                top3.c_str());
  }

  // ---- 落盘 ----
  RPG_CHECK(npy::WriteF32(prefix + "gpu_probs.npy", {kNCases, vocab}, probs.data(),
                          probs.size()),
            "写 gpu_probs.npy 失败");
  RPG_CHECK(npy::WriteF32(prefix + "gpu_u.npy", {kNCases, kUCount}, us.data(),
                          us.size()),
            "写 gpu_u.npy 失败");
  RPG_CHECK(cli::WriteText(prefix + "gpu_tokens.txt", tokens_txt),
            "写 gpu_tokens.txt 失败");

  std::printf("\n写出：%scases.tsv / logits.bin / gpu_probs.npy / gpu_tokens.txt / gpu_u.npy\n",
              prefix.c_str());
  std::printf("下一步：编参考侧并跑 tests/cmp_sample.py（见 tests/run_p3.sh）\n");
  return 0;
}
