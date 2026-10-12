// cpu_ref_sample —— P3 判据的**参考侧**：用 CPU 引擎的采样实现跑同一批 logits。
//
// ## 为什么这个文件在本仓库
//
// 与 cpu_ref_gen.cc 同一个理由（DESIGN §7 末）：参考实现必须是"真的在跑的那一份"。
// 这里 `#include "sampler.h"`，直接调 CPU 引擎的
//
//     rpg::FilterToProbs / rpg::SampleToken / rpg::CounterUniform
//
// —— 源码由 tools/build_cpu_ref.sh 从 RPGBarEngine 现编。**没有一行筛选或采样
// 逻辑是抄的。** 在本仓库再写一份"我以为 CPU 是怎么做的"，两边同源的理解错误
// 会互相掩盖，判据就废了。
//
// ## 为什么采样这块能逐 token 对拍（与 P2 的贪心对拍不同）
//
// P2 的逐 token 对拍是**接在模型后面**的：两边 logits 差 fp16 的舍入，采样把它
// 放大成"整段序列不同"。这里把模型摘掉，两边读的是**同一份从文件来的 logits**
// （逐位相同）⇒ 同一个 seed 下两边必须给出**同一个 token**。
//
// 输入输出与 GPU 侧 tools/sample_check.cu 一一对应（cases.tsv 是两边共用的用例
// 矩阵，避免参数在两边各写一遍然后漂掉）：
//
//   读：<dir>/cases.tsv、<dir>/logits.bin
//   写：<dir>/ref_probs.npy、<dir>/ref_tokens.txt、<dir>/ref_u.npy

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cli_util.h"
#include "common.h"
#include "npy.h"
#include "sampler.h"

namespace {

struct Case {
  int idx = 0;
  float temperature = 1.0f;
  int32_t top_k = 0;
  float top_p = 1.0f;
  uint64_t seed = 0;
  int tie = 0;
  int u_count = 0;
  int64_t vocab = 0;
};

// cases.tsv 的格式：制表符分隔，'#' 开头是注释。
// ⚠️ 解析失败要报错而不是"用默认值继续" —— 参数悄悄变成默认值，症状是
// "两边分布不同"，而人会先去查 kernel。
std::vector<Case> ReadCases(const std::string& path) {
  const std::string text = cli::ReadFile(path);
  std::vector<Case> out;
  size_t pos = 0;
  while (pos <= text.size()) {
    size_t eol = text.find('\n', pos);
    if (eol == std::string::npos) eol = text.size();
    const std::string line = text.substr(pos, eol - pos);
    pos = eol + 1;
    if (line.empty() || line[0] == '#') {
      if (eol >= text.size()) break;
      continue;
    }
    std::vector<std::string> f;
    size_t a = 0;
    while (a <= line.size()) {
      size_t t = line.find('\t', a);
      if (t == std::string::npos) t = line.size();
      f.push_back(line.substr(a, t - a));
      a = t + 1;
    }
    RPG_CHECK(f.size() == 8, "%s 里这一行的字段数不是 8：%s", path.c_str(),
              line.c_str());
    Case c;
    c.idx = std::atoi(f[0].c_str());
    c.temperature = std::strtof(f[1].c_str(), nullptr);
    c.top_k = std::atoi(f[2].c_str());
    c.top_p = std::strtof(f[3].c_str(), nullptr);
    c.seed = std::strtoull(f[4].c_str(), nullptr, 10);
    c.tie = std::atoi(f[5].c_str());
    c.u_count = std::atoi(f[6].c_str());
    c.vocab = std::strtoll(f[7].c_str(), nullptr, 10);
    out.push_back(c);
    if (eol >= text.size()) break;
  }
  RPG_CHECK(!out.empty(), "%s 里一条用例都没读到", path.c_str());
  return out;
}

// 读原始 f32（GPU 侧写 logits.bin 用的就是这个格式：没有头、纯 C 顺序）。
std::vector<float> ReadF32(const std::string& path, size_t n) {
  FILE* f = std::fopen(path.c_str(), "rb");
  RPG_CHECK(f != nullptr, "读不了 %s", path.c_str());
  std::vector<float> v(n);
  const size_t got = std::fread(v.data(), sizeof(float), n, f);
  std::fclose(f);
  RPG_CHECK(got == n, "%s 只读到 %zu / %zu 个 float", path.c_str(), got, n);
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--dir") {
      RPG_CHECK(i + 1 < argc, "--dir 缺参数");
      dir = argv[++i];
    } else if (a == "-h" || a == "--help") {
      std::printf(
          "用法: cpu_ref_sample --dir DIR\n"
          "\n"
          "  读  DIR/cases.tsv 与 DIR/logits.bin（都由 tools/sample_check 写出），\n"
          "  用 CPU 引擎的 sampler.cc 跑同一条路径，写 DIR/ref_*.npy|txt。\n"
          "\n"
          "  ⚠️ 实现来自 RPGBarEngine（由 tools/build_cpu_ref.sh 现编），\n"
          "     本文件只有 main。\n");
      return 0;
    } else {
      std::fprintf(stderr, "不认识的选项：%s\n", a.c_str());
      return 2;
    }
  }
  RPG_CHECK(!dir.empty(), "必须给 --dir DIR");
  const std::string prefix = dir + "/";

  const std::vector<Case> cases = ReadCases(prefix + "cases.tsv");
  const int64_t vocab = cases[0].vocab;
  for (const Case& c : cases) {
    RPG_CHECK(c.vocab == vocab, "用例 %d 的词表长度 %lld 与第一条的 %lld 不同 —— "
                                "本工具的产物是定形数组，要求全表一致",
              c.idx, static_cast<long long>(c.vocab),
              static_cast<long long>(vocab));
  }
  const size_t n_cases = cases.size();

  std::printf("参考侧（CPU 引擎的 sampler.cc）：%zu 个用例，词表 %lld\n", n_cases,
              static_cast<long long>(vocab));

  const std::vector<float> logits = ReadF32(prefix + "logits.bin", n_cases * vocab);

  std::vector<float> probs(n_cases * static_cast<size_t>(vocab));
  std::string tokens_txt;

  for (size_t ci = 0; ci < n_cases; ++ci) {
    const Case& c = cases[ci];
    const float* row = logits.data() + ci * static_cast<size_t>(vocab);

    rpg::SampleOptions opt;
    opt.temperature = c.temperature;
    opt.top_k = c.top_k;
    opt.top_p = c.top_p;

    std::vector<float> p;
    rpg::FilterToProbs(row, vocab, opt, &p);
    RPG_CHECK(static_cast<int64_t>(p.size()) == vocab, "FilterToProbs 给的分布长度不对");
    std::memcpy(probs.data() + ci * static_cast<size_t>(vocab), p.data(),
                static_cast<size_t>(vocab) * sizeof(float));

    std::string line;
    for (int u = 0; u < c.u_count; ++u) {
      // ⚠️ u 由**绝对位置**（这里就是 counter）推出来 —— 与 GPU 侧同一条约定。
      const float uu = rpg::CounterUniform(c.seed, static_cast<uint64_t>(u));
      const int32_t tok = rpg::SampleToken(row, vocab, opt, uu);
      RPG_CHECK(tok >= 0 && tok < vocab, "SampleToken 给出的下标越界：%d", tok);
      line += std::to_string(tok);
      line += (u + 1 == c.u_count) ? "\n" : " ";
    }
    tokens_txt += line;
  }

  // ref_u.npy：把 u 表也落盘，判据那边比的是**比特**（纯整数运算，没有容差可说）。
  // ⚠️ u_count 从 cases.tsv 读，**不在两边各写一个常量** —— 那个 16 只在
  // sample_check.cu 里存在（kUCount），这里跟着表走。
  const int u_count = cases[0].u_count;
  RPG_CHECK(u_count > 0, "cases.tsv 里的 u_count 不是正数");
  for (const Case& c : cases) {
    RPG_CHECK(c.u_count == u_count,
              "用例 %d 的 u_count = %d 与第一条的 %d 不同 —— u 表是定形数组，"
              "要求全表一致",
              c.idx, c.u_count, u_count);
  }
  std::vector<float> us(n_cases * static_cast<size_t>(u_count));
  for (size_t ci = 0; ci < n_cases; ++ci) {
    for (int u = 0; u < u_count; ++u) {
      us[ci * static_cast<size_t>(u_count) + u] =
          rpg::CounterUniform(cases[ci].seed, static_cast<uint64_t>(u));
    }
  }

  RPG_CHECK(npy::WriteF32(prefix + "ref_probs.npy", {static_cast<int64_t>(n_cases), vocab},
                          probs.data(), probs.size()),
            "写 ref_probs.npy 失败");
  RPG_CHECK(npy::WriteF32(prefix + "ref_u.npy",
                          {static_cast<int64_t>(n_cases), u_count}, us.data(),
                          us.size()),
            "写 ref_u.npy 失败");
  RPG_CHECK(cli::WriteText(prefix + "ref_tokens.txt", tokens_txt),
            "写 ref_tokens.txt 失败");

  std::printf("写出：%sref_probs.npy / ref_tokens.txt / ref_u.npy\n", prefix.c_str());
  return 0;
}
