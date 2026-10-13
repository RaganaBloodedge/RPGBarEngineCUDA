// matmul_bench —— P4 的 kernel 级基准：**权重矩阵读一遍到底要多久**。
//
// 为什么需要它（而不是只看引擎的 tok/s）：P4 的第一步（CUDA Graph）做完了，
// 正确性全绿，但 decode 只快了 1.09x —— 说明瓶颈**不在 kernel 启动开销**上。
// 那么就必须把"一层前向的时间花在哪"量出来，否则后面每一步优化都是在猜。
//
// 这个工具只做一件事：拿模型**真实的那几张权重矩阵**，在不同的 n_tokens 下把
// `GpuMatMulQuant` 跑很多遍，报出**有效带宽**（权重字节数 / 耗时）。
//
// 为什么这一个数就够定位：
//   * decode 每步只算 1 个 token，而权重必须**整套读一遍** ⇒ 如果 kernel 是
//     带宽受限的，每步的耗时下限就是 `权重量 / 显存带宽`。1.7B Q4（1.03 GiB）
//     在 384 GB/s 上约 3 ms/token（≈ 350 tok/s）。
//   * 量出来的 GB/s 与这个下限一比，就知道差在"带宽没打满"还是"算力被打满"。
//   * `n_tokens` 那条曲线同时给出结论：**耗时随 token 数怎么变**。如果
//     n_tokens=1→4 时**总耗时几乎不变**（而 token 数是 4 倍），说明瓶颈是
//     "每个权重元素的处理成本"（反量化是 ALU），它与 token 数无关 —— 那正是
//     `DeqElemQ4K` 的结构：每个元素两次 F16ToF32 + 一次 6-bit scale/min 位拼接
//     + 三次**非收缩**乘法，约 30-40 条指令，而每个元素只读 0.5625 字节。
//
// ⚠️ 本工具**只测时间，不验数值** —— 数值正确性是 dequant_check / layer_check /
// run_p2 的活。混在一起会让"快了"和"对了"两件事互相掩盖。
//
// 用法：
//   matmul_bench model.gguf [--iters 20] [--tokens 1,4,16,64]

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cli_util.h"
#include "common.h"
#include "cuda_common.h"
#include "gpu_model.h"
#include "kernels_gpu.h"

namespace {

double NowMs() {
  static const auto t0 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0).count();
}

// 把一整层的 7 张矩阵 + lm_head 跑一遍，返回 (总字节数, 总毫秒)。多跑 iters 遍取平均。
struct Row {
  int32_t n_tokens = 0;
  int64_t weight_bytes = 0;
  double ms = 0;
};

Row BenchOnce(const rpg::GpuModel& m, int32_t n_tokens, int32_t iters) {
  const rpg::GpuLayer& L = m.layers()[0];

  struct Item {
    const char* name;
    const rpg::GpuTensor* w;
  };
  const Item items[] = {
      {"attn_q", &L.attn_q},   {"attn_k", &L.attn_k},   {"attn_v", &L.attn_v},
      {"attn_o", &L.attn_o},   {"ffn_gate", &L.ffn_gate},
      {"ffn_up", &L.ffn_up},   {"ffn_down", &L.ffn_down},
      {"lm_head", &m.lm_head()},
  };

  Row r;
  r.n_tokens = n_tokens;

  // 每张矩阵单独分配自己尺寸的 X / OUT —— 因为它们的 cols / rows 完全不同。
  for (const Item& it : items) {
    const rpg::GpuTensor& w = *it.w;
    RPG_CHECK(w.matrix && w.valid(), "tensor %s is not a valid matrix", it.name);
    rpg::DeviceBuffer X(static_cast<size_t>(n_tokens) * static_cast<size_t>(w.cols) * sizeof(float));
    rpg::DeviceBuffer OUT(static_cast<size_t>(n_tokens) * static_cast<size_t>(w.rows) * sizeof(float));
    const float* xd = static_cast<const float*>(X.data());
    float* od = static_cast<float*>(OUT.data());

    // 预热：第一次调用会把 kernel 的 shared memory 属性等一次性开销付掉。
    for (int i = 0; i < 2; ++i) rpg::GpuMatMulQuant(w, xd, n_tokens, od);
    RPG_CUDA_CHECK(cudaDeviceSynchronize());

    const double t0 = NowMs();
    for (int i = 0; i < iters; ++i) rpg::GpuMatMulQuant(w, xd, n_tokens, od);
    RPG_CUDA_CHECK(cudaDeviceSynchronize());
    const double t1 = NowMs();

    r.ms += (t1 - t0) / iters;
    r.weight_bytes += w.row_bytes() * w.rows;
  }
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: matmul_bench <model.gguf> [--iters N] [--tokens 1,4,16,64]\n");
    return 2;
  }
  const std::string model_path = argv[1];
  int32_t iters = 20;
  std::vector<int32_t> toks = {1, 2, 4, 8, 16, 32, 64};

  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) rpg::Throw("option %s needs a value", a.c_str());
      return argv[++i];
    };
    if (a == "--iters") iters = std::atoi(next());
    else if (a == "--tokens") toks = cli::ParseIds(next());
    else rpg::Throw("unknown option %s", a.c_str());
  }

  try {
    rpg::GpuModel m = rpg::GpuModel::Load(model_path, 4096);
    cudaDeviceProp p{};
    RPG_CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    std::printf("%s", m.Describe().c_str());
    std::printf("device: %s (sm_%d%d, %d SM)\n", p.name, p.major, p.minor,
                p.multiProcessorCount);

    // 带宽下限：decode 每步必须把整套权重读一遍（含 tied 的 lm_head）。
    const int64_t all_weights = m.device_bytes();
    std::printf(
        "\nall weights on device: %.3f GiB (a decode step reads essentially all of "
        "it: every layer + lm_head)\n",
        static_cast<double>(all_weights) / (1024.0 * 1024 * 1024));

    std::printf(
        "\n  n_tokens   weight bytes     ms (1 layer + lm_head)      GB/s\n"
        "  --------   --------------   -----------------------   --------\n");

    double gbps_at_1 = 0.0;
    double ms_at_1 = 0.0;
    for (const int32_t nt : toks) {
      const Row r = BenchOnce(m, nt, iters);
      const double gbps =
          static_cast<double>(r.weight_bytes) / (r.ms / 1000.0) / 1e9;
      std::printf("  %8d   %14lld   %21.3f   %8.1f\n", nt,
                  static_cast<long long>(r.weight_bytes), r.ms, gbps);
      if (nt == 1) {
        gbps_at_1 = gbps;
        ms_at_1 = r.ms;
      }
    }

    if (gbps_at_1 > 0.0) {
      // 把"一层 + lm_head"的量按整个模型的比例放大，得到 decode 每步的估计。
      // 比例 = 全部权重的字节数 / (一层 + lm_head 的字节数)。
      const Row r1 = BenchOnce(m, 1, iters);
      const double frac =
          static_cast<double>(all_weights) / static_cast<double>(r1.weight_bytes);
      const double ms_per_token = ms_at_1 * frac;
      std::printf(
          "\nn_tokens = 1: %.1f GB/s => one decode step ~= %.1f ms => %.1f tok/s\n",
          gbps_at_1, ms_per_token, 1000.0 / ms_per_token);
      std::printf(
          "  (the whole model is %.3fx the bytes measured above, so the number is\n"
          "   scaled linearly; it assumes the rest of the step is negligible)\n",
          frac);
      std::printf(
          "  if the kernel were bandwidth-bound at 384 GB/s, one step would be "
          "%.1f ms (%.0f tok/s)\n",
          static_cast<double>(all_weights) / 384e9 * 1000.0,
          384e9 / static_cast<double>(all_weights));
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
