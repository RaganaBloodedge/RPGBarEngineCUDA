// dequant_check —— P1 的第一条判据：**反量化必须逐位相同**。
//
// 为什么这条要单独、且要求 0 个 bit 不同：
//   反量化不涉及浮点求和顺序，只涉及"整数拆包 + 一次乘加"。如果这里就有 1 ULP
//   的差异，那么点积的差异就永远说不清是谁贡献的 —— 是求和顺序（正常舍入），
//   还是权重值本身就不同（真 bug）。CPU 版当年在对拍时吃过这个亏（"我们离
//   float64 真值比官方实现远几倍"），所以两边都单独留了这一层。
//
// 对拍的两侧：
//   参考侧 = src/quant.cc 的 DequantizeRow —— **从 RPGBarEngine 逐字复制的**
//            标量实现，也就是 CPU 引擎在生产用的那一份。
//   被测侧 = src/quant_gpu.cuh 的 DeqElem  —— 设备端按元素解，融合点积用它。
// 参考侧要求用 -ffp-contract=off 编（见 CMakeLists.txt 里那行），否则 host 侧
// 会把 `d1 * raw - m1` 缩成 FMA、设备侧用 __fmul_rn/__fsub_rn 不缩，差 1 ULP。
//
// 另跑一条穷举：f16 → f32 的 65536 个值逐位比（设备端那份位操作是手写的，
// 与其相信它写对了，不如让它自己证明）。
//
// 用法：
//   dequant_check <model.gguf> [--rows N] [--tensors a,b,c] [--quiet]

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cli_util.h"
#include "common.h"
#include "cuda_common.h"
#include "dtypes.h"       // host 版 F16ToF32（参考）
#include "gguf.h"
#include "kernels_gpu.h"  // GpuDequantRow（被测侧入口）
#include "quant.h"
#include "quant_gpu.cuh"  // 设备版 DeqElem / F16ToF32

namespace {

// f16 → f32 穷举（65536 个值）。
__global__ void F16SweepKernel(float* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 65536) return;
  out[i] = rpg::gpu::F16ToF32(static_cast<uint16_t>(i));
}

// 有序 ULP 距离。同号时直接减；跨零要绕开 ±0 的那一格；NaN 约定为"不同"。
// 这一项存在的意义是：判据是"0 个 bit 不同"，万一红了，ULP 距离能立刻区分
// "整体差 1 ULP（收缩/舍入）"与"差出几百 ULP（拆包错了）"。
int64_t UlpDistance(float a, float b) {
  uint32_t ua, ub;
  std::memcpy(&ua, &a, 4);
  std::memcpy(&ub, &b, 4);
  if (ua == ub) return 0;
  const bool na = (ua & 0x7F800000u) == 0x7F800000u && (ua & 0x7FFFFFu);
  const bool nb = (ub & 0x7F800000u) == 0x7F800000u && (ub & 0x7FFFFFu);
  if (na || nb) return -1;  // NaN / Inf：按"无法比较"记，调用方当不同处理
  const int64_t ia = (ua & 0x80000000u) ? -static_cast<int64_t>(ua & 0x7FFFFFFFu)
                                        : static_cast<int64_t>(ua);
  const int64_t ib = (ub & 0x80000000u) ? -static_cast<int64_t>(ub & 0x7FFFFFFFu)
                                        : static_cast<int64_t>(ub);
  return ia > ib ? ia - ib : ib - ia;
}

struct RowResult {
  int64_t elems = 0;
  int64_t diff_bits = 0;
  int64_t max_ulp = 0;
};

// 逐行对拍一个张量。
RowResult CheckTensor(const rpg::GgufFile& g, const rpg::GgufTensor& t,
                      int64_t rows_wanted, bool quiet) {
  const int64_t cols = t.cols();
  const int64_t row_bytes = rpg::TypeRowBytes(t.type, cols);
  const uint8_t* base = g.TensorBytes(t);
  const int64_t rows = std::min<int64_t>(rows_wanted, t.rows());

  // 一行搬一次，避免把整张 151936×2048 的 embedding 全搬上卡。
  rpg::DeviceBuffer dev_in(static_cast<size_t>(row_bytes));
  rpg::DeviceBuffer dev_out(static_cast<size_t>(cols) * sizeof(float));

  std::vector<float> ref(static_cast<size_t>(cols));
  std::vector<float> got(static_cast<size_t>(cols));

  RowResult r;
  for (int64_t i = 0; i < rows; ++i) {
    const uint8_t* src = base + i * row_bytes;
    // 参考：就是从 RPGBarEngine 复制来的那份标量实现。
    rpg::DequantizeRow(t.type, src, cols, ref.data());

    dev_in.Upload(src, static_cast<size_t>(row_bytes));
    rpg::GpuDequantRow(t.type, static_cast<const uint8_t*>(dev_in.data()), cols,
                       static_cast<float*>(dev_out.data()));
    dev_out.Download(got.data(), static_cast<size_t>(cols) * sizeof(float));

    r.elems += cols;
    for (int64_t j = 0; j < cols; ++j) {
      uint32_t a, b;
      std::memcpy(&a, &ref[static_cast<size_t>(j)], 4);
      std::memcpy(&b, &got[static_cast<size_t>(j)], 4);
      if (a != b) {
        ++r.diff_bits;
        const int64_t u = UlpDistance(ref[static_cast<size_t>(j)],
                                      got[static_cast<size_t>(j)]);
        if (u < 0) {
          r.max_ulp = -1;  // 出现 NaN/Inf：单独标记
        } else if (r.max_ulp >= 0 && u > r.max_ulp) {
          r.max_ulp = u;
        }
      }
    }
  }
  if (!quiet) {
    std::printf("  %-34s %-6s %5lld %10lld %12lld %9lld  %s\n", t.name.c_str(),
                rpg::TypeName(t.type), static_cast<long long>(rows),
                static_cast<long long>(r.elems), static_cast<long long>(r.diff_bits),
                static_cast<long long>(r.max_ulp),
                r.diff_bits == 0 ? "OK" : "**不同**");
  }
  return r;
}

// 挑默认张量：每种类型取第一个（覆盖 F32 / Q4_K / Q6_K 各一次）。
std::vector<const rpg::GgufTensor*> DefaultPicks(const rpg::GgufFile& g,
                                                int per_type) {
  std::vector<const rpg::GgufTensor*> out;
  std::vector<uint32_t> seen;
  for (const rpg::GgufTensor& t : g.tensors()) {
    if (!t.is_matrix()) continue;
    int n = 0;
    for (uint32_t s : seen) {
      if (s == t.type) ++n;
    }
    if (n >= per_type) continue;
    seen.push_back(t.type);
    out.push_back(&t);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf(
        "用法: dequant_check <model.gguf> [选项]\n"
        "  --rows N        每个张量对拍多少行（默认 32）\n"
        "  --per-type N    每个量化类型取几个张量（默认 1）\n"
        "  --tensors a,b   指定张量名（逗号分隔，覆盖上面的自动挑选）\n"
        "  --quiet         只打印总结\n");
    return 2;
  }

  try {
    const std::string model_path = argv[1];
    int64_t rows_wanted = 32;
    int per_type = 1;
    std::string tensors_arg;
    bool quiet = false;
    for (int i = 2; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> const char* {
        if (i + 1 >= argc) rpg::Throw("选项 %s 后面缺参数", a.c_str());
        return argv[++i];
      };
      if (a == "--rows") rows_wanted = std::atoll(next());
      else if (a == "--per-type") per_type = std::atoi(next());
      else if (a == "--tensors") tensors_arg = next();
      else if (a == "--quiet") quiet = true;
      else rpg::Throw("未知选项 %s", a.c_str());
    }

    rpg::GgufFile g = rpg::GgufFile::Open(model_path);
    RPG_CHECK(g.valid(), "GGUF 打不开：%s", model_path.c_str());

    const rpg::DeviceInfo dev = rpg::QueryDevice(0);
    RPG_CHECK(dev.available, "没有可用的 CUDA 设备：%s", dev.reason.c_str());
    std::printf("%s\n", rpg::DescribeDevice(dev).c_str());

    bool ok = true;

    // ---- 1) f16 → f32 穷举 ----
    {
      rpg::DeviceBuffer d(65536 * sizeof(float));
      F16SweepKernel<<<256, 256>>>(static_cast<float*>(d.data()));
      RPG_CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<float> got(65536);
      d.Download(got.data(), got.size() * sizeof(float));
      int64_t bad = 0;
      int64_t worst = 0;
      for (int i = 0; i < 65536; ++i) {
        const float ref = rpg::F16ToF32(static_cast<uint16_t>(i));
        uint32_t a, b;
        std::memcpy(&a, &ref, 4);
        std::memcpy(&b, &got[static_cast<size_t>(i)], 4);
        if (a != b) {
          ++bad;
          const int64_t u = UlpDistance(ref, got[static_cast<size_t>(i)]);
          if (u > worst) worst = u;
        }
      }
      std::printf("\n[1] f16 → f32 穷举 65536 个值：不同 %lld 个，最大 ULP %lld  %s\n",
                  static_cast<long long>(bad), static_cast<long long>(worst),
                  bad == 0 ? "OK" : "**不同**");
      if (bad != 0) ok = false;
    }

    // ---- 2) 反量化逐位对拍 ----
    std::printf("\n[2] 反量化逐位对拍（参考 = quant.cc 的 DequantizeRow）\n");
    std::printf("  %-34s %-6s %5s %10s %12s %9s\n", "张量", "类型", "行", "元素",
                "不同 bit", "max ULP");

    std::vector<const rpg::GgufTensor*> picks;
    if (!tensors_arg.empty()) {
      size_t i = 0;
      while (i < tensors_arg.size()) {
        size_t j = tensors_arg.find(',', i);
        if (j == std::string::npos) j = tensors_arg.size();
        const std::string nm = tensors_arg.substr(i, j - i);
        if (!nm.empty()) {
          const rpg::GgufTensor* t = g.Find(nm);
          RPG_CHECK(t != nullptr, "找不到张量 %s", nm.c_str());
          RPG_CHECK(t->is_matrix(), "%s 不是矩阵（本工具只对拍矩阵）", nm.c_str());
          picks.push_back(t);
        }
        i = j + 1;
      }
    } else {
      picks = DefaultPicks(g, per_type);
    }
    RPG_CHECK(!picks.empty(), "没有可对拍的量化矩阵");

    int64_t total_elems = 0, total_bad = 0;
    for (const rpg::GgufTensor* t : picks) {
      const RowResult r = CheckTensor(g, *t, rows_wanted, quiet);
      total_elems += r.elems;
      total_bad += r.diff_bits;
      if (r.diff_bits != 0) ok = false;
    }

    std::printf("\n总结：对拍 %lld 个元素，%lld 个 bit 不同  →  %s\n",
                static_cast<long long>(total_elems),
                static_cast<long long>(total_bad),
                ok ? "PASS（逐位相同）" : "FAIL");
    std::printf(
        "判据：反量化是纯整数拆包 + 一次乘加，不涉浮点求和顺序，所以要求\n"
        "      **0 个 bit 不同**。有差异时先看 max ULP：1 就是 FMA 收缩没对齐，\n"
        "      几百以上就是拆包写错了（Q4_K / Q6_K 的位布局互不通用，混用不报错）。\n");
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "错误：%s\n", e.what());
    return 1;
  }
}
