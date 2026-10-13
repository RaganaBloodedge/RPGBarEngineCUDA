// layer_check —— P1 的判据工具：在 GPU 上跑整段前向，把结果写成 npy。
//
// 输出与 CPU 引擎的 tools/forward_check **同名同形状**（<前缀>.hidden.npy
// [n_layers+1, seq, hidden] / <前缀>.logits.npy [seq, vocab]），所以对拍是：
//
//   forward_check <model.gguf> --columns ... --out runs/ref   # CPU 参考侧
//   layer_check   <model.gguf> --columns ... --out runs/gpu   # 本工具
//   python tests/cmp_layers.py runs/gpu runs/ref              # 逐层比
//
// 为什么参考侧要用 CPU 引擎的可执行文件、而不是在本仓库里再写一份 CPU 前向：
// 参考实现必须是"真的在跑的那一份"。自己抄一份 CPU 前向进来，等于拿我对
// CPU 行为的理解去对我的 GPU 实现 —— 两边同源的理解错误会互相掩盖。
// 所以参考侧的源码留在它自己的仓库里（tests/run_p1.sh 里那条 g++ 命令）。
//
// 用法：
//   layer_check <model.gguf> [选项]
//     --tokens a,b,c     直接给 token id（与参考侧共用同一串才谈得上对拍）
//     --prompt <文本>    用内置分词器编码
//     --prompt-file <f>  从文件读提示词（中文走命令行会被转码搞坏）
//     --out <前缀>       写 <前缀>.tokens.txt / .logits.npy / .hidden.npy
//     --layers <N>       只跑前 N 层（定位误差起点；此时不写 logits.npy）
//     --max-pos <N>      RoPE 表覆盖的位置数（默认 4096）
//     --top <K>          打印最后位置 top-K（默认 5）
//     --no-hidden        不导出逐层残差流
//     --info             只打印模型信息后退出

#include <algorithm>
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
#include "npy.h"
#include "qwen3_gpu.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf(
        "用法: layer_check <model.gguf> [选项]\n"
        "  --tokens <a,b,c>   直接给 token id\n"
        "  --prompt <文本>     用内置分词器编码\n"
        "  --prompt-file <f>  从文件读提示词\n"
        "  --out <前缀>       写 <前缀>.tokens.txt / .logits.npy / .hidden.npy\n"
        "  --layers <N>       只跑前 N 层（此时不写 logits.npy）\n"
        "  --max-pos <N>      RoPE 表覆盖的位置数（默认 4096）\n"
        "  --top <K>          打印最后位置 top-K（默认 5）\n"
        "  --no-hidden        不导出逐层残差流\n"
        "  --info             只打印模型信息\n");
    return 2;
  }

  const std::string model_path = argv[1];
  std::string prompt = "你好，请用一句话介绍你自己。";
  std::string tokens_arg;
  std::string out_prefix;
  int32_t max_pos = 4096;
  int32_t layers = 0;
  int top_k = 5;
  bool dump_hidden = true;
  bool info_only = false;

  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) rpg::Throw("选项 %s 后面缺少参数", a.c_str());
      return argv[++i];
    };
    if (a == "--prompt") prompt = next();
    else if (a == "--prompt-file") prompt = cli::ReadFile(next());
    else if (a == "--tokens") tokens_arg = next();
    else if (a == "--out") out_prefix = next();
    else if (a == "--max-pos") max_pos = std::atoi(next());
    else if (a == "--no-hidden") dump_hidden = false;
    else if (a == "--layers") layers = std::atoi(next());
    else if (a == "--top") top_k = std::atoi(next());
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

    std::vector<int32_t> ids;
    if (!tokens_arg.empty()) {
      ids = cli::ParseIds(tokens_arg);
      std::printf("\n[输入] 直接指定 %lld 个 token id\n",
                  static_cast<long long>(ids.size()));
    } else {
      ids = model.tokenizer().Encode(prompt, /*parse_special=*/true);
      // 与 CPU 版同一个判断：add_bos 是 false（Qwen3 不加 BOS），但若某个
      // 转换版本设成 true 也要照做 —— 参考侧读的是同一个键。
      if (c.add_bos && c.bos_id >= 0) ids.insert(ids.begin(), c.bos_id);
      std::printf("\n[输入] \"%s\" → %lld 个 token\n", prompt.c_str(),
                  static_cast<long long>(ids.size()));
      std::printf("       id: ");
      for (size_t i = 0; i < ids.size() && i < 40; ++i) {
        std::printf(i ? ",%d" : "%d", ids[i]);
      }
      std::printf("%s\n", ids.size() > 40 ? " ..." : "");
      std::printf("       还原: \"%s\"\n", model.tokenizer().Decode(ids).c_str());
    }
    RPG_CHECK(!ids.empty(), "token 序列为空");

    rpg::GpuScratch scratch;
    std::vector<float> logits;
    rpg::GpuForwardOptions opt;
    opt.layer_limit = layers;
    std::vector<float> hidden;
    if (dump_hidden) opt.capture_hidden = &hidden;

    const auto t0 = std::chrono::steady_clock::now();
    rpg::GpuForwardFull(model, ids.data(), static_cast<int64_t>(ids.size()),
                        &scratch, &logits, opt);
    RPG_CUDA_CHECK(cudaDeviceSynchronize());
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const int64_t seq = static_cast<int64_t>(ids.size());
    const int32_t used = layers > 0 && layers < c.n_layers ? layers : c.n_layers;
    std::printf("\n[前向] 跑 %d/%d 层，%lld 个位置，用时 %.1f ms（显存 %lld / 暂存 %lld）\n",
                used, c.n_layers, static_cast<long long>(seq), ms,
                static_cast<long long>(model.device_bytes()),
                static_cast<long long>(scratch.bytes()));
    std::printf("        %.1f token/s（prefill 口径）\n",
                static_cast<double>(seq) / (ms / 1000.0));

    // ---- 最后位置的 top-K，肉眼判个合理 ----
    if (!logits.empty()) {
      const float* last = logits.data() + (seq - 1) * c.vocab;
      std::vector<int32_t> order(static_cast<size_t>(c.vocab));
      for (int32_t i = 0; i < c.vocab; ++i) order[static_cast<size_t>(i)] = i;
      const int k = std::min<int>(top_k, c.vocab);
      std::partial_sort(order.begin(), order.begin() + k, order.end(),
                        [last](int32_t a, int32_t b) { return last[a] > last[b]; });
      std::printf("\n[最后位置 top-%d]  logits 范围 [%.3f, %.3f]\n", k,
                  *std::min_element(last, last + c.vocab),
                  *std::max_element(last, last + c.vocab));
      for (int i = 0; i < k; ++i) {
        const int32_t id = order[static_cast<size_t>(i)];
        std::printf("  #%d  id=%-7d logit=%9.4f  %s\n", i + 1, id, last[id],
                    model.tokenizer().TokenOf(id).c_str());
      }
    } else {
      std::printf("（截断模式：不计算末端归一化与 lm_head，无 logits）\n");
    }

    if (out_prefix.empty()) return 0;

    cli::WriteIds(out_prefix + ".tokens.txt", ids);

    if (!logits.empty()) {
      const std::string lp = out_prefix + ".logits.npy";
      RPG_CHECK(npy::WriteF32(lp, {seq, c.vocab}, logits.data(), logits.size()),
                "写不了 %s", lp.c_str());
      std::printf("\n[输出] %s  [%lld, %d]\n", lp.c_str(),
                  static_cast<long long>(seq), c.vocab);
    }
    if (dump_hidden) {
      const std::string hn = out_prefix + ".hidden.npy";
      const int64_t nl = static_cast<int64_t>(hidden.size()) / (seq * c.hidden);
      RPG_CHECK(npy::WriteF32(hn, {nl, seq, c.hidden}, hidden.data(), hidden.size()),
                "写不了 %s", hn.c_str());
      std::printf("[输出] %s  [%lld, %lld, %d]\n", hn.c_str(),
                  static_cast<long long>(nl), static_cast<long long>(seq),
                  c.hidden);
    }
    std::printf("[输出] %s.tokens.txt\n", out_prefix.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "错误：%s\n", e.what());
    return 1;
  }
}
