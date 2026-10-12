// kv_map_check —— KV Cache 写入的**下标映射**测试（不需要权重、不需要模型）。
//
// ## 为什么单独立一条判据
//
// `GpuKvWrite` 要做一次下标重排：源是本段的 `[t][head][dim]`（一个 token 的多头
// 挨在一起），目标是 cache 的 `[head][pos][dim]`（一个头的连续位置挨在一起）。
// **写反了不会崩**，只会让注意力读到别的头、别的位置的历史 —— 症状是"从某个位置
// 起开始胡说"，与"kv 的 fp16 精度不够"长得一模一样。后者是可以接受的（DESIGN §4.2
// 明确拿了 448 MiB 显存去换），前者是 bug。
//
// 这条判据把两者彻底分开：
//
//   * 值用**fp16 能精确表示的整数**（< 2048），所以 fp16 的往返是**无损**的 ——
//     这条判据是"**逐位相同**"，与任何精度问题无关；
//   * 构造时把 cache **先清零**，写完再检查**没写过的那两段仍然是 0** ——
//     这是"没有写串到区间之外"的直接证据（越界写不会报错，只会悄悄改掉邻居）；
//   * 形状全部压到个位数（3 个 KV 头 / head_dim 8 / 容量 12 / 写 5 个位置
//     从 pos0 = 4 开始），所以失败时可以直接把 (head, pos, dim) 三元组打出来看。
//
// ⇒ 它红了 = 映射错了（先修这个）；它绿了而 §5 的数值判据只是"fp16 量级"的偏差
//   = 一切正常。**两条判据的分工写在 tests/run_p2.sh 里，顺序不能反。**
//
// 用法：
//   kv_map_check
//   kv_map_check --heads 4 --dim 16 --max-seq 32 --pos0 7 --seq 9

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_common.h"
#include "kernels_gpu.h"

namespace {

int g_fail = 0;

void Check(bool ok, const char* what) {
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what);
  if (!ok) ++g_fail;
}

// fp16 能精确表示的整数上限是 2048（11 位有效位）。编码成一个**可逆**的值：
//   (head, t, dim) → head*100 + t*10 + dim
// 只要三者都 < 10 就能唯一还原，且 <= 247 << 2048 ⇒ 往返无损。
float Encode(int32_t h, int64_t t, int32_t d) {
  return static_cast<float>(h * 100 + static_cast<int32_t>(t) * 10 + d);
}

}  // namespace

int main(int argc, char** argv) {
  int32_t n_kv_heads = 3;
  int32_t head_dim = 8;
  int64_t max_seq = 12;
  int64_t pos0 = 4;
  int64_t seq = 5;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* { return argv[++i]; };
    if (a == "--heads") n_kv_heads = std::atoi(next());
    else if (a == "--dim") head_dim = std::atoi(next());
    else if (a == "--max-seq") max_seq = std::atoll(next());
    else if (a == "--pos0") pos0 = std::atoll(next());
    else if (a == "--seq") seq = std::atoll(next());
    else {
      std::printf("未知参数 %s\n", a.c_str());
      return 2;
    }
  }

  // 编码要求三者都 < 10（见 Encode 的说明）。这条检查不是洁癖：一旦超过，
  // 编码不再可逆，失败时会打出误导人的三元组。
  if (n_kv_heads >= 10 || seq >= 10 || head_dim >= 10) {
    std::printf("这条测试要求 --heads / --seq / --dim 都 < 10（编码可逆）\n");
    return 2;
  }
  if (pos0 + seq > max_seq) {
    std::printf("pos0 + seq 超出 max_seq\n");
    return 2;
  }

  std::printf("KV 映射测试：%d 个 KV 头 × %d 维 × 容量 %lld，"
              "把 %lld 个位置写到 [%lld, %lld)\n",
              n_kv_heads, head_dim, static_cast<long long>(max_seq),
              static_cast<long long>(seq), static_cast<long long>(pos0),
              static_cast<long long>(pos0 + seq));

  try {
    const int64_t src_n = seq * n_kv_heads * head_dim;      // [t][h][d]
    const int64_t dst_n = n_kv_heads * max_seq * head_dim;  // [h][pos][d]

    std::vector<float> k_src(static_cast<size_t>(src_n));
    std::vector<float> v_src(static_cast<size_t>(src_n));
    for (int64_t t = 0; t < seq; ++t) {
      for (int32_t h = 0; h < n_kv_heads; ++h) {
        for (int32_t d = 0; d < head_dim; ++d) {
          const int64_t i =
              (t * n_kv_heads + h) * head_dim + d;  // 源的 [t][h][d] 行主
          k_src[static_cast<size_t>(i)] = Encode(h, t, d);
          // V 用另一个偏移，这样"K 写进了 V、V 写进了 K"也能被抓到
          v_src[static_cast<size_t>(i)] = Encode(h, t, d) + 1000.0f;
        }
      }
    }

    rpg::DeviceBuffer k_src_dev(static_cast<size_t>(src_n) * sizeof(float));
    rpg::DeviceBuffer v_src_dev(static_cast<size_t>(src_n) * sizeof(float));
    k_src_dev.Upload(k_src.data(), k_src.size() * sizeof(float));
    v_src_dev.Upload(v_src.data(), v_src.size() * sizeof(float));

    rpg::DeviceBuffer k_dst(static_cast<size_t>(dst_n) * sizeof(uint16_t));
    rpg::DeviceBuffer v_dst(static_cast<size_t>(dst_n) * sizeof(uint16_t));
    // 先清零：写完要检查"没写过的地方仍是 0"。不先清零的话，"越界写到了邻居"
    // 这种错会被掩盖（邻居本来就没写过、读到的是垃圾，无从判断）。
    RPG_CUDA_CHECK(cudaMemset(k_dst.data(), 0, k_dst.size()));
    RPG_CUDA_CHECK(cudaMemset(v_dst.data(), 0, v_dst.size()));

    // nvcc 不接受 void* → const float* 的隐式转换（g++ 会接受），所以显式转。
    rpg::GpuKvWrite(static_cast<const float*>(k_src_dev.data()),
                    static_cast<const float*>(v_src_dev.data()), n_kv_heads,
                    head_dim, pos0, seq,
                    static_cast<uint16_t*>(k_dst.data()),
                    static_cast<uint16_t*>(v_dst.data()), max_seq);
    RPG_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> k_out(k_dst.size() / sizeof(uint16_t));
    std::vector<uint16_t> v_out(v_dst.size() / sizeof(uint16_t));
    k_dst.Download(k_out.data(), k_dst.size());
    v_dst.Download(v_out.data(), v_dst.size());

    // ---- 检查 1：写过的位置必须逐位等于期望值 ----
    int64_t bad = 0;
    int64_t first_bad = -1;
    for (int32_t h = 0; h < n_kv_heads; ++h) {
      for (int64_t t = 0; t < seq; ++t) {
        for (int32_t d = 0; d < head_dim; ++d) {
          const int64_t oi = ((h * max_seq + pos0 + t) * head_dim) + d;
          const float want_k = Encode(h, t, d);
          const float want_v = want_k + 1000.0f;
          const float got_k = __half2float(*reinterpret_cast<__half*>(&k_out[static_cast<size_t>(oi)]));
          const float got_v = __half2float(*reinterpret_cast<__half*>(&v_out[static_cast<size_t>(oi)]));
          if (got_k != want_k || got_v != want_v) {
            ++bad;
            if (first_bad < 0) {
              first_bad = oi;
              std::printf("  第一处不符：head %d / pos %lld / dim %d —— "
                          "期望 K=%g V=%g，实际 K=%g V=%g\n",
                          h, static_cast<long long>(pos0 + t), d,
                          static_cast<double>(want_k), static_cast<double>(want_v),
                          static_cast<double>(got_k), static_cast<double>(got_v));
            }
          }
        }
      }
    }
    {
      char buf[256];
      std::snprintf(buf, sizeof(buf),
                    "写入区间内 %lld 个元素逐位正确（K 与 V 分别核对）",
                    static_cast<long long>(seq) * n_kv_heads * head_dim);
      Check(bad == 0, buf);
    }

    // ---- 检查 2：没写过的地方必须仍是 0 ----
    int64_t dirty = 0;
    for (int32_t h = 0; h < n_kv_heads; ++h) {
      for (int64_t p = 0; p < max_seq; ++p) {
        if (p >= pos0 && p < pos0 + seq) continue;
        for (int32_t d = 0; d < head_dim; ++d) {
          const int64_t oi = ((h * max_seq + p) * head_dim) + d;
          if (k_out[static_cast<size_t>(oi)] != 0 ||
              v_out[static_cast<size_t>(oi)] != 0) {
            ++dirty;
          }
        }
      }
    }
    Check(dirty == 0,
          "未写入的位置保持为 0（没有越界写到邻居 —— 那是最隐蔽的一类错）");
    if (dirty != 0) {
      std::printf("         **有 %lld 个元素被动了** —— 写入的边界算错了\n",
                  static_cast<long long>(dirty));
    }

    std::printf("\n%s\n", g_fail == 0
                              ? "PASS —— KV 写入的下标映射与边界都正确"
                              : "FAIL —— 映射或边界有问题");
    return g_fail == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "错误：%s\n", e.what());
    return 1;
  }
}
