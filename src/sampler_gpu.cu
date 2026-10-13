// sampler_gpu.cu —— 采样的设备端实现（P3）。接口与理由见 sampler_gpu.h。
//
// 一个 kernel 干完"确定性的一半"（FilterProbsKernel），另一个（一个线程的）
// 干完"逆变换"。为什么逆变换不并行，见 InverseSampleKernel 的头注释。
//
// ⚠️ **两级二分是全文件最容易写错的地方**（top_k 找第 k 大值、top_p 找切点）：
//   * 搜索空间是 **uint32 的 key 全序**，中点必须用 uint64 算 —— 见 top_k 那段；
//   * 谓词必须**单调**，否则二分收敛到的地方看起来"差不多对"，只在某些分布上错。
// 两处都写了注释。它们是"不崩、只是偶尔选错 token"那一类 bug。

#include "cuda_common.h"

#ifdef RPG_HAS_CUDA

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <string>

#include "common.h"
#include "kernels_gpu.h"  // GpuArgMax —— 贪心分支复用它（不重写一遍）
#include "sampler.h"
#include "sampler_gpu.h"

namespace rpg {
namespace {

constexpr int kSampleThreads = 1024;

void CheckLaunch(const char* what) {
  const cudaError_t e = cudaGetLastError();
  RPG_CHECK(e == cudaSuccess, "%s 启动/执行失败：%s", what, cudaGetErrorString(e));
}

// ---------------------------------------------------------------- 有序 key
//
// 把 float 的位模式映射成 uint32 且**保序**。为什么需要它：top_k 要找"第 k 大的
// 值"、top_p 要找"累积概率的切点"，两者都是对**值**做二分。而 float 的位模式在
// 负数段是反的（-0.0 的位模式 0x80000000 "大于" 最小正数的 0x00000001），直接拿
// 位模式当 key 会得到一段段颠倒的顺序 —— 症状是"大部分时候对、偶尔选错一个 token"。
//
// 这条 trick 让"值的比较"变成"无符号整数的比较"，于是二分可以直接在整数空间做。
// NaN 不在讨论范围（CPU 版也没处理 NaN：它在 FilterToProbs 之前就被绕过了）。
__device__ inline uint32_t OrderKey(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ inline float FromOrderKey(uint32_t k) {
  const uint32_t u = (k & 0x80000000u) ? (k & 0x7fffffffu) : ~k;
  return __uint_as_float(u);
}

// ---------------------------------------------------------------- block 归约
//
// 每个都是"全体线程都要调用"的（内部有 __syncthreads）。它们都在**对整块统一的
// 分支**里被调用（条件是标量参数），所以合法。
//
// ⚠️ 每次进来先 __syncthreads 一次：否则上一轮读完 sh[0] 的慢线程会被这一轮
// 写 sh[tid] 的快线程覆盖 —— 一个只在"归约被连续调用两次"时才出现的偶发错误。

__device__ float BlockMaxFloat(float v, float* sh) {
  __syncthreads();
  sh[threadIdx.x] = v;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (static_cast<int>(threadIdx.x) < st) {
      const float a = sh[threadIdx.x], b = sh[threadIdx.x + st];
      sh[threadIdx.x] = a > b ? a : b;
    }
    __syncthreads();
  }
  return sh[0];
}

__device__ double BlockSumDouble(double v, double* sh) {
  __syncthreads();
  sh[threadIdx.x] = v;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (static_cast<int>(threadIdx.x) < st) sh[threadIdx.x] += sh[threadIdx.x + st];
    __syncthreads();
  }
  return sh[0];
}

__device__ int64_t BlockSumInt(int64_t v, int64_t* sh) {
  __syncthreads();
  sh[threadIdx.x] = v;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (static_cast<int>(threadIdx.x) < st) sh[threadIdx.x] += sh[threadIdx.x + st];
    __syncthreads();
  }
  return sh[0];
}

// 块内 argmax。**并列取最小下标**（严格大于），与 CPU 版 ArgMax 同语义 ——
// 差一个并列规则，top_k 的 one-hot 与兜底分支就会指向不同的 token。
__device__ int32_t BlockArgMax(const float* __restrict__ x, int64_t n, float* shv,
                               int32_t* shi) {
  float bv = -INFINITY;
  int32_t bi = 0;
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = x[i];
    if (v > bv) {
      bv = v;
      bi = static_cast<int32_t>(i);
    }
  }
  __syncthreads();
  shv[threadIdx.x] = bv;
  shi[threadIdx.x] = bi;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (static_cast<int>(threadIdx.x) < st) {
      const float vr = shv[threadIdx.x + st];
      const int32_t ir = shi[threadIdx.x + st];
      if (vr > shv[threadIdx.x] || (vr == shv[threadIdx.x] && ir < shi[threadIdx.x])) {
        shv[threadIdx.x] = vr;
        shi[threadIdx.x] = ir;
      }
    }
    __syncthreads();
  }
  return shi[0];
}

// ---------------------------------------------------------------- 筛选

__global__ void FilterProbsKernel(const float* __restrict__ logits, int64_t n,
                                  float temperature, int32_t top_k, float top_p,
                                  float* __restrict__ probs) {
  const int tid = threadIdx.x;
  const int nt = blockDim.x;
  __shared__ float gshv[kSampleThreads];
  __shared__ int32_t gshi[kSampleThreads];
  __shared__ double gshd[kSampleThreads];
  __shared__ int64_t gshc[kSampleThreads];

  for (int64_t i = tid; i < n; i += nt) probs[i] = logits[i];

  // ---- 贪心：直接给 argmax 一个 one-hot，绕过下面所有筛选（也就不用碰 NaN）。
  // 与 CPU 版 FilterToProbs 的第一条分支**同序**：先判 temperature。
  if (!(temperature > 0.0f)) {
    const int32_t best = BlockArgMax(probs, n, gshv, gshi);
    for (int64_t i = tid; i < n; i += nt) probs[i] = 0.0f;
    if (tid == 0) probs[best] = 1.0f;
    return;  // 对整块统一的分支（temperature 是标量参数），不是发散分支
  }

  // ---- 温度：缩放 logits。温度 < 1 让分布变尖，> 1 让它变平。
  // ⚠️ 用 `!= 1.0f` 判断、并且用 `1.0f / temperature` 这个**浮点除法**，
  // 都是为了与 CPU 版逐字一致（换个写法差 1 ULP，判据会在边界上抖）。
  if (temperature != 1.0f) {
    const float inv = 1.0f / temperature;
    for (int64_t i = tid; i < n; i += nt) probs[i] *= inv;
  }
  __syncthreads();

  // ---- top_k：阈值取**第 k 大的值**，`< 阈值` 的全剔除（并列的一起保留，
  // 与 HF TopKLogitsWarper 的 `<` 语义一致；CPU 版用的是 std::nth_element）。
  //
  // 怎么在 GPU 上求"第 k 大的值"而不排序：对谓词
  //     C(t) = |{ i : key_i >= t }|
  // 做二分（C 在 t 上单调不增，我们要找**最大的** t 使 C(t) >= k —— 那个 t 就是
  // 第 k 大的元素的值，并列时正好是"阈值"本身）。32 次迭代遍历整个 vocab,
  // 每次一个块内计数归约。
  if (top_k > 1 && top_k < n) {
    uint32_t lo = 0u, hi = 0xFFFFFFFFu;
    while (lo < hi) {
      // ⚠️ 中点必须用 uint64 算：`hi - lo + 1` 在 lo=0 / hi=0xFFFFFFFF 时
      // 溢出成 0，mid 就永远不动 —— 死循环，而且是在第一次迭代就卡住。
      const uint32_t mid = static_cast<uint32_t>(
          lo + ((static_cast<uint64_t>(hi) - lo + 1) >> 1));
      int64_t cnt = 0;
      for (int64_t i = tid; i < n; i += nt) {
        if (OrderKey(probs[i]) >= mid) ++cnt;
      }
      cnt = BlockSumInt(cnt, gshc);
      if (cnt >= static_cast<int64_t>(top_k)) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }
    const float kth = FromOrderKey(lo);
    for (int64_t i = tid; i < n; i += nt) {
      if (probs[i] < kth) probs[i] = -INFINITY;
    }
  }
  __syncthreads();

  // ---- softmax（与 CPU 的 SoftmaxInPlace 同构：先减最大值）。
  // expf(-inf - mx) = 0，所以上面置的 -inf 在这里自然变成概率 0。
  float lmax = -INFINITY;
  for (int64_t i = tid; i < n; i += nt) {
    const float v = probs[i];
    if (v > lmax) lmax = v;
  }
  const float mx = BlockMaxFloat(lmax, gshv);
  double lsum = 0.0;
  for (int64_t i = tid; i < n; i += nt) {
    const float e = expf(probs[i] - mx);
    probs[i] = e;
    lsum += static_cast<double>(e);
  }
  const double sum = BlockSumDouble(lsum, gshd);
  const float inv = static_cast<float>(1.0 / sum);
  for (int64_t i = tid; i < n; i += nt) probs[i] *= inv;
  __syncthreads();

  // ---- top_p（核采样）：按**累积概率**切尾巴。
  // CPU 是"按概率升序累积，累积还没过 (1 - top_p) 的整撮剔除" ⇒ 等价于
  // "累积概率 CDF(p) ≤ 1 - top_p 的那些元素被剔掉"。于是又是一个单调谓词上的
  // 二分：
  //     S(t) = Σ{ i : p_i <= t } p_i         （t 越大 S 越大，单调）
  // 找**最大的** t 使 S(t) <= (1 - top_p) —— 那个 t 就是切点值，`<= t` 的全剔掉。
  // ⚠️ 这样比 CPU 的"全量 std::sort"（151936 个元素）省掉整个排序，
  // 代价是并列概率元素之间的取舍是未定义的（CPU 那边取决于 std::sort 的同值顺序）
  // —— 并列在 fp32 softmax 的输出里几乎不出现，判据的测试数据也要避开它。
  if (top_p > 0.0f && top_p < 1.0f) {
    const double threshold = 1.0 - static_cast<double>(top_p);
    uint32_t lo = 0u, hi = 0xFFFFFFFFu;
    while (lo < hi) {
      const uint32_t mid = static_cast<uint32_t>(
          lo + ((static_cast<uint64_t>(hi) - lo + 1) >> 1));
      double s = 0.0;
      for (int64_t i = tid; i < n; i += nt) {
        if (OrderKey(probs[i]) <= mid) s += static_cast<double>(probs[i]);
      }
      s = BlockSumDouble(s, gshd);
      if (s <= threshold) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }
    const float cut = FromOrderKey(lo);
    for (int64_t i = tid; i < n; i += nt) {
      if (probs[i] <= cut) probs[i] = 0.0f;
    }
    __syncthreads();

    // 保底：至少留 1 个（CPU 版把"概率最大的那个"强制留成 1.0）。
    // ⚠️ 只在"整撮被剔光"时才会走到；并列最大值时 CPU 选的是 std::sort 顺序
    // 里的最后一个，与这里的"最小下标"未必是同一个 —— 属于上面说的并列未定义。
    const int32_t best = BlockArgMax(probs, n, gshv, gshi);
    if (tid == 0 && probs[best] == 0.0f) probs[best] = 1.0f;
    __syncthreads();

    double lsum2 = 0.0;
    for (int64_t i = tid; i < n; i += nt) lsum2 += static_cast<double>(probs[i]);
    const double sum2 = BlockSumDouble(lsum2, gshd);
    if (sum2 > 0.0) {
      const float inv2 = static_cast<float>(1.0 / sum2);
      for (int64_t i = tid; i < n; i += nt) probs[i] *= inv2;
    }
  }
}

// ---------------------------------------------------------------- 逆变换

// **一个线程、按下标顺序**串行累加。
//
// 为什么不并行：CPU 版是
//     cum += static_cast<double>(probs[i]);  if (cum > u) return i;
// 一条串行依赖链。并行扫描/分块求和会把"求和顺序"改成树形，末位就与 CPU 不同，
// 于是 u 落在累积概率边界附近时两边会选到不同的 token —— 而"同一份 logits +
// 同一个 seed ⇒ 同一个 token"正是这个模块的判据本身。
//
// 代价：n = 151936（真实词表）时一条 ~0.5 ms 的串行链。decode 一步现在要 54 ms，
// 占比 ~1%。**P4 想省这点可以改成"并行算分块和 + 串行定块"**，那时要接受
// "u 贴近边界（~1e-7）时可能与 CPU 差一个 token"这个前提；现在不必付这个代价。
__global__ void InverseSampleKernel(const float* __restrict__ probs, int64_t n,
                                    double u, int32_t* __restrict__ out_idx) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  double cum = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    cum += static_cast<double>(probs[i]);
    if (cum > u) {
      *out_idx = static_cast<int32_t>(i);
      return;
    }
  }
  // 没跨过：只可能被浮点误差推到这条分支（u 逼近 1）。与 CPU 版一样返回概率
  // 最大的那个，保证函数永远给出一个合法 token。
  float bv = probs[0];
  int32_t bi = 0;
  for (int64_t i = 1; i < n; ++i) {
    if (probs[i] > bv) {
      bv = probs[i];
      bi = static_cast<int32_t>(i);
    }
  }
  *out_idx = bi;
}

}  // namespace

// ---------------------------------------------------------------- 随机源
//
// 与 CPU 版 sampler.cc 的 SplitMix64 / CounterRng / CounterUniform **逐字对应**：
// 三行整数运算 + 乘一个 2 的幂，所以两边对拍的判据是"逐位相同"。
// ⚠️ 这里的"逐字对应"是**重写**而非复制（sampler.cc 不在本仓库，见 sampler_gpu.h），
// 所以它就是那种"改一边忘了另一边"的典型 —— 判据（sample_check 的 u 表）盯着它。
uint64_t DevSplitMix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

uint64_t DevCounterRng(uint64_t seed, uint64_t counter) {
  return DevSplitMix64(seed + 0x9E3779B97F4A7C15ull * counter);
}

float DevCounterUniform(uint64_t seed, uint64_t counter) {
  const uint32_t bits = static_cast<uint32_t>(DevCounterRng(seed, counter) >> 40);
  return static_cast<float>(bits) * (1.0f / 16777216.0f);
}

// ---------------------------------------------------------------- 对外接口

void GpuFilterToProbs(const float* logits_dev, int64_t n, const SampleOptions& opt,
                      float* probs_dev) {
  RPG_CHECK(logits_dev != nullptr && probs_dev != nullptr, "采样的缓冲不能为空");
  RPG_CHECK(n > 0, "采样的词表长度必须为正（给了 %lld）",
            static_cast<long long>(n));
  RPG_CHECK(n <= 0x7FFFFFFF, "词表长度超出 int32 下标（%lld）",
            static_cast<long long>(n));
  FilterProbsKernel<<<1, kSampleThreads>>>(logits_dev, n, opt.temperature,
                                           opt.top_k, opt.top_p, probs_dev);
  CheckLaunch("FilterProbsKernel");
}

void GpuInverseSample(const float* probs_dev, int64_t n, float u,
                      int32_t* out_idx_dev) {
  RPG_CHECK(probs_dev != nullptr && out_idx_dev != nullptr, "采样的缓冲不能为空");
  RPG_CHECK(n > 0, "采样的词表长度必须为正（给了 %lld）",
            static_cast<long long>(n));
  // ⚠️ u 必须是 [0,1)：约 1 时那个 `cum > u` 可能永远不成立（走兜底分支）。
  RPG_CHECK(u >= 0.0f && u < 1.0f, "u 必须落在 [0,1)，给了 %g",
            static_cast<double>(u));
  InverseSampleKernel<<<1, 1>>>(probs_dev, n, static_cast<double>(u), out_idx_dev);
  CheckLaunch("InverseSampleKernel");
}

void GpuSampleToken(const float* logits_dev, int64_t n, const SampleOptions& opt,
                    uint64_t seed, uint64_t counter, float* probs_dev,
                    int32_t* out_idx_dev) {
  // 贪心不消耗随机数 —— 与 CPU 版 SampleToken 的**第一条分支**同序。
  // 这条分支让"不传 temperature 的老路径"与 M2 的纯贪心逐位一致。
  if (!(opt.temperature > 0.0f)) {
    RPG_CHECK(logits_dev != nullptr && out_idx_dev != nullptr, "采样的缓冲不能为空");
    GpuArgMax(logits_dev, n, out_idx_dev);
    return;
  }
  const float u = DevCounterUniform(seed, counter);
  GpuFilterToProbs(logits_dev, n, opt, probs_dev);
  GpuInverseSample(probs_dev, n, u, out_idx_dev);
}

}  // namespace rpg

#endif  // RPG_HAS_CUDA
