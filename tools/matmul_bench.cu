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

// ---------------------------------------------------------------- 逐矩阵

// 上面的表把"一层 + lm_head"合成一个数——它能定位"慢不慢"，但定位不了"谁慢"。
// 4B 上就吃过这个亏：聚合带宽 41 GB/s 看着只是"没打满"，拆开才发现 `ffn_down`
// 根本没走专用 kernel（cols=9728 ⇒ nb=38 > kMaxQuantBlocks），而它占每层 32%
// 的权重字节。所以单列一张逐矩阵表：**每张矩阵各自的 ms 与有效带宽**。
struct ItemRow {
  const char* name;
  const char* type;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t bytes = 0;
  double ms = 0;
};

ItemRow BenchItem(const rpg::GpuTensor& w, const char* name, int32_t n_tokens,
                  int32_t iters) {
  ItemRow r;
  r.name = name;
  r.type = w.type_name();
  r.rows = w.rows;
  r.cols = w.cols;
  r.bytes = w.row_bytes() * w.rows;
  rpg::DeviceBuffer X(static_cast<size_t>(n_tokens) * static_cast<size_t>(w.cols) * sizeof(float));
  rpg::DeviceBuffer OUT(static_cast<size_t>(n_tokens) * static_cast<size_t>(w.rows) * sizeof(float));
  const float* xd = static_cast<const float*>(X.data());
  float* od = static_cast<float*>(OUT.data());
  for (int i = 0; i < 2; ++i) rpg::GpuMatMulQuant(w, xd, n_tokens, od);
  RPG_CUDA_CHECK(cudaDeviceSynchronize());
  const double t0 = NowMs();
  for (int i = 0; i < iters; ++i) rpg::GpuMatMulQuant(w, xd, n_tokens, od);
  RPG_CUDA_CHECK(cudaDeviceSynchronize());
  r.ms = (NowMs() - t0) / iters;
  return r;
}

// ---------------------------------------------------------------- --verify
//
// "P4 重写完 matmul 之后算术没变"这句话的直接证据。
//
// P1/P2 给的是**相对误差**（p99 ≤ 1e-3），那是"数值正确"的判据 —— 它推不出
// "逐位相同"。这里把两条路径都跑一遍（`GpuSetForceGenericMatMul` 在之间切换），
// 对同一份权重与输入做**位比较**：只要有一个 bit 不同就是 FAIL。
//
// ⚠️ 用**真实权重矩阵**而不是合成数据：专用 kernel 走的是"按类型硬编码块布局"
// 的快路径，只有真权重的块数与类型分布才确保每条分支都被走到（打印出来的
// type= 就是覆盖证据：Q4_K 与 Q6_K 两条专用 kernel 都要出现在表里）。
//
// 输入用确定性的 sin 而不是随机数：随机数靠种子复现，而这里两条路径读的是
// **同一块显存**，本来就不需要复现性 —— 确定性只是让 FAIL 时的报数可重跑。
int VerifyExact(const rpg::GpuModel& m, const std::vector<int32_t>& toks) {
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

  int failures = 0;
  std::printf("\n== --verify: 通用 kernel vs 专用 kernel（逐位对拍）==\n");
  for (const Item& it : items) {
    const rpg::GpuTensor& w = *it.w;
    RPG_CHECK(w.matrix && w.valid(), "tensor %s is not a valid matrix", it.name);
    std::printf("  %-9s type=%-6s [%lld x %lld]\n", it.name, w.type_name(),
                static_cast<long long>(w.rows), static_cast<long long>(w.cols));
    for (const int32_t nt : toks) {
      const size_t xn = static_cast<size_t>(nt) * static_cast<size_t>(w.cols);
      const size_t on = static_cast<size_t>(nt) * static_cast<size_t>(w.rows);
      rpg::DeviceBuffer X(xn * sizeof(float));
      rpg::DeviceBuffer Og(on * sizeof(float));
      rpg::DeviceBuffer Os(on * sizeof(float));
      std::vector<float> xh(xn);
      for (size_t i = 0; i < xn; ++i) {
        xh[i] = static_cast<float>(std::sin(static_cast<double>(i) * 0.017) * 0.5);
      }
      RPG_CUDA_CHECK(cudaMemcpy(X.data(), xh.data(), xn * sizeof(float),
                                cudaMemcpyHostToDevice));

      rpg::GpuSetForceGenericMatMul(true);
      rpg::GpuMatMulQuant(w, static_cast<const float*>(X.data()), nt,
                          static_cast<float*>(Og.data()));
      rpg::GpuSetForceGenericMatMul(false);
      rpg::GpuMatMulQuant(w, static_cast<const float*>(X.data()), nt,
                          static_cast<float*>(Os.data()));
      RPG_CUDA_CHECK(cudaDeviceSynchronize());

      std::vector<float> og(on), os(on);
      RPG_CUDA_CHECK(cudaMemcpy(og.data(), Og.data(), on * sizeof(float),
                                cudaMemcpyDeviceToHost));
      RPG_CUDA_CHECK(cudaMemcpy(os.data(), Os.data(), on * sizeof(float),
                                cudaMemcpyDeviceToHost));

      size_t diff = 0, first = 0;
      for (size_t i = 0; i < on; ++i) {
        if (std::memcmp(&og[i], &os[i], sizeof(float)) != 0) {
          if (diff == 0) first = i;
          ++diff;
        }
      }
      std::printf("    n_tokens=%-4d  %zu/%zu differ%s\n", nt, diff, on,
                  diff ? "   <== FAIL" : "");
      if (diff != 0) {
        ++failures;
        std::printf("      first diff @%zu: generic=%.9g specialized=%.9g\n",
                    first, static_cast<double>(og[first]),
                    static_cast<double>(os[first]));
      }
    }
  }
  if (failures == 0) {
    std::printf("  => 两条路径所有用例逐位相同。\n");
  } else {
    std::printf("  => %d 个用例有 bit 不同。\n", failures);
  }
  return failures;
}

// ---------------------------------------------------------------- --bw 带宽天花板

// 一个纯流式的显存带宽探针。**为什么必须有一个**：上面那张表给的是"matmul 拿了
// 多少 GB/s"，但"这个数算好还是坏"取决于这台机器**实际**能跑多少 —— 拿规格书的
// 峰值（384 GB/s）当分母会得出"只值 12%"这种结论，而 12% 到底是"kernel 写得烂"
// 还是"这台笔记本的显存就这么快"，没有天花板就分不出来。
//
// 两个核都用 float4（16 字节/次）：这正是"每个线程一次读一个 1 字节"的 matmul
// 对照 —— 同样的字节数，一次飞 16 字节和一次飞 1 字节差的就是 MLP。
__global__ void BwCopyKernel(const float4* __restrict__ in, float4* __restrict__ out,
                             int64_t n4) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n4; i += stride) {
    out[i] = in[i];
  }
}

__global__ void BwReadKernel(const float4* __restrict__ in, float* __restrict__ sink,
                            int64_t n4) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n4; i += stride) {
    const float4 v = in[i];
    a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
  }
  // 写回要"不可能成立"才不会被优化掉。
  if (a.x == 12345.678f) sink[0] = a.x + a.y + a.z + a.w;
}

int RunBwProbe(int64_t bytes) {
  const int64_t n4 = bytes / 16;
  rpg::DeviceBuffer a(static_cast<size_t>(bytes));
  rpg::DeviceBuffer b(static_cast<size_t>(bytes));
  RPG_CUDA_CHECK(cudaMemset(a.data(), 1, static_cast<size_t>(bytes)));
  cudaDeviceProp p{};
  RPG_CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  const int blocks = p.multiProcessorCount * 32;
  std::printf("\n== --bw: 纯流式显存带宽（对照 ceiling，不是规格峰值）==\n");
  std::printf("  device %s, %d SM, %d block x 256 threads, 16 B/thread/iter\n",
              p.name, p.multiProcessorCount, blocks);
  std::printf("  buffer %.2f GiB x2\n", static_cast<double>(bytes) / 1073741824.0);
  for (int rep = 0; rep < 3; ++rep) {
    cudaEvent_t s, e;
    RPG_CUDA_CHECK(cudaEventCreate(&s));
    RPG_CUDA_CHECK(cudaEventCreate(&e));
    float ms = 0.f;
    RPG_CUDA_CHECK(cudaEventRecord(s));
    BwCopyKernel<<<blocks, 256, 0, rpg::EngineStream()>>>(
        static_cast<const float4*>(a.data()), static_cast<float4*>(b.data()), n4);
    RPG_CUDA_CHECK(cudaEventRecord(e));
    RPG_CUDA_CHECK(cudaEventSynchronize(e));
    RPG_CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    std::printf("  copy 2x %.2f GiB : %8.2f ms  -> %7.1f GB/s\n",
                static_cast<double>(bytes) / 1073741824.0, ms,
                2.0 * static_cast<double>(bytes) / (ms / 1000.0) / 1e9);
    RPG_CUDA_CHECK(cudaEventRecord(s));
    BwReadKernel<<<blocks, 256, 0, rpg::EngineStream()>>>(
        static_cast<const float4*>(a.data()), static_cast<float*>(b.data()), n4);
    RPG_CUDA_CHECK(cudaEventRecord(e));
    RPG_CUDA_CHECK(cudaEventSynchronize(e));
    RPG_CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    std::printf("  read   %.2f GiB   : %8.2f ms  -> %7.1f GB/s\n",
                static_cast<double>(bytes) / 1073741824.0, ms,
                static_cast<double>(bytes) / (ms / 1000.0) / 1e9);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
  std::printf("usage: matmul_bench <model.gguf> [--iters N] [--tokens 1,4,16,64]\n"
              "                        [--per-matrix-tokens N] [--verify] [--verify-tokens 1,3,5,8]\n"
              "                        [--bw [GiB]]\n"
              "  --per-matrix-tokens: 逐矩阵表在哪个 n_tokens 下量（默认 1 = decode 形状）\n"
              "  --verify: 另外跑一遍通用 kernel vs 专用 kernel 的逐位对拍\n"
              "  --bw: 另外量一次纯流式显存带宽（天花板；默认 1 GiB）\n");
  return 2;
  }
  const std::string model_path = argv[1];
  int32_t iters = 20;
  std::vector<int32_t> toks = {1, 2, 4, 8, 16, 32, 64};
  bool verify = false;
  std::vector<int32_t> vtoks = {1, 3, 5, 8};
  // 逐矩阵表在哪个 n_tokens 下量。默认 1（decode 形状）—— 但那**不是**扫参时唯一
  // 该看的点：形状的收益在 decode 与 prefill 上是反的，只量 n=1 会把 prefill 的
  // 结论整个漏掉（写这段之前刚踩过：以为在扫 n=64，其实逐矩阵表写死 n=1，
  // 两次"不同 n"的扫描给出逐位相同的排名）。
  int32_t pm_tokens = 1;
  int64_t bw_gib = 0;

  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) rpg::Throw("option %s needs a value", a.c_str());
      return argv[++i];
    };
    if (a == "--iters") iters = std::atoi(next());
    else if (a == "--tokens") toks = cli::ParseIds(next());
    else if (a == "--per-matrix-tokens") pm_tokens = std::atoi(next());
    else if (a == "--verify") verify = true;
    else if (a == "--verify-tokens") vtoks = cli::ParseIds(next());
    else if (a == "--bw") {
      bw_gib = 1;
      if (i + 1 < argc && argv[i + 1][0] != '-') bw_gib = std::atoll(argv[++i]);
    } else rpg::Throw("unknown option %s", a.c_str());
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

    // 逐矩阵表：见 BenchItem 的注释（聚合数定位不了"谁慢"）。
    {
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
      std::printf("\n  per-matrix, n_tokens=%d%s\n"
                  "  --------------------------------------------------------------------\n"
                  "  %-9s %-6s %18s %14s %8s %9s\n",
                  pm_tokens,
                  pm_tokens == 1 ? " (decode shape)" : " (prefill shape)",
                  "matrix", "type", "[rows x cols]", "bytes", "ms", "GB/s");
      int64_t total_bytes = 0;
      double total_ms = 0.0;
      for (const Item& it : items) {
        const rpg::GpuTensor& w = *it.w;
        const ItemRow r = BenchItem(w, it.name, pm_tokens, iters);
        std::printf("  %-9s %-6s %18s %14lld %8.3f %9.1f\n", r.name, r.type,
                    (std::to_string(r.rows) + " x " + std::to_string(r.cols)).c_str(),
                    static_cast<long long>(r.bytes), r.ms,
                    static_cast<double>(r.bytes) / (r.ms / 1000.0) / 1e9);
        total_bytes += r.bytes;
        total_ms += r.ms;
      }
      std::printf("  %-9s %-6s %18s %14lld %8.3f %9.1f\n", "TOTAL", "",
                  "", static_cast<long long>(total_bytes), total_ms,
                  static_cast<double>(total_bytes) / (total_ms / 1000.0) / 1e9);
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

    if (bw_gib > 0) {
      RunBwProbe(bw_gib * 1024LL * 1024 * 1024);
    }

    if (verify) {
      const int bad = VerifyExact(m, vtoks);
      return bad == 0 ? 0 : 1;
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
