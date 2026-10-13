// kernels_gpu.cu —— 算子实现（P1：先把"对不对"拿下，快是 P4 的事）
//
// 每一条算子都在 tools/dequant_check（逐位）或 P1 的层对拍（相对误差）里有
// 对应的判据，没有判据的算子不留 —— 这是 DESIGN §5 那条"没有判据的阶段不算
// 完成"在算子粒度上的展开。
//
// ⚠️ 一个贯穿全文件的取舍：**这一版不追性能**。具体表现是
//   * 融合点积的归约走 shared memory 的树形求和（4 个 token 就是 4 遍）；
//   * attention 一个 block 只算一个 (t, head)，不做分块与 KV 驻留；
//   * 反量化一个元素一个元素地算，不做向量化。
// P4 会重做这三个地方（融合 kernel / 分块 / 量化直算调优）。现在把它们写
// "快"没有意义：在算力还没被证明正确之前调优，量到的收益归因不到任何东西上。

#include "kernels_gpu.h"

#ifdef RPG_HAS_CUDA

#include <cfloat>
#include <cmath>
#include <cstdio>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "common.h"
#include "cuda_common.h"
#include "quant.h"
#include "quant_gpu.cuh"

namespace rpg {

namespace {

// 一个 block 的线程数。128 是 P1 的选择：够短促、归约 7 步、occupancy 不会
// 被寄存器卡住（融合点积里每个线程最多 4 个累加器 + 几个中间量）。
constexpr int kThreads = 128;

// 融合点积一次同时算几个 token。**这是"权重直传"能不能划算的核心参数**：
// 权重在每个 tile 里只被反量化一遍，token 越多摊得越薄。4 是 128 线程下的
// 折中（再多就要加共享内存或寄存器中转）。
constexpr int kTileTokens = 4;

// argmax 那一个 block 的线程数。词表 151936 个元素，1024 线程 → 每线程 149 个。
// 单 block 而不是多 block + 二次归约：vocab 的量级下 1024 线程的树形归约只要
// 10 步，多 block 反而要多一次 kernel 启动与一次中间结果回写。
constexpr int kArgMaxThreads = 1024;

// ggml 块的形状在 host 侧算好传进来（设备端没有 TypeInfo —— 那是一张
// host 侧的表，搬到设备上只是为了省两次整数除法，不值）。
struct BlockShape {
  int32_t elems = 1;
  int32_t bytes = 4;
};

BlockShape ShapeOf(uint32_t type) {
  const GgmlTypeInfo* t = TypeInfo(type);
  RPG_CHECK(t != nullptr, "未知的 ggml 类型 id=%u", type);
  return BlockShape{static_cast<int32_t>(t->block_elems),
                    static_cast<int32_t>(t->block_bytes)};
}

// 启动之后立刻查一次 —— 参数配置错（grid 超限、共享内存超限）在这里就报出来，
// 而不是等最后同步时报一个看不出出处的错误。
void CheckLaunch(const char* what) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    Throw("kernel 启动失败（%s）：%s", what, cudaGetErrorString(e));
  }
}

// ---------------------------------------------------------------- 反量化

__global__ void DequantRowKernel(uint32_t type, const uint8_t* src, int64_t n_elems,
                                 int32_t block_elems, int32_t block_bytes,
                                 float* dst) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n_elems) return;
  const int64_t bi = i / block_elems;
  const int ei = static_cast<int>(i - bi * block_elems);
  dst[i] = gpu::DeqElem(type, src + bi * block_bytes, ei);
}

// ---------------------------------------------------------------- 融合点积

// 一个 block 负责输出矩阵的**一行**（r = blockIdx.x），块内线程协作算
// n_tokens 个点积；token 按 kTileTokens 分片（blockIdx.y）。
//
// 数据流：外层遍历量化块（每块 block_elems 个元素），块内线程按 elems 步进；
// 每个元素反量化**一次**，然后对 tile 内每个 token 各做一次 FMA。
// 于是权重读显存的量 = 量化字节的量（对一个 token 而言），token 越多越省。
__global__ void MatMulQuantKernel(uint32_t type, const uint8_t* W, int64_t rows,
                                  int64_t row_bytes, int64_t cols,
                                  int32_t block_elems, int32_t block_bytes,
                                  const float* X, int64_t n_tokens, float* OUT) {
  const int64_t r = blockIdx.x;
  const int64_t t0 = static_cast<int64_t>(blockIdx.y) * kTileTokens;
  // 不用 device 的 min()：int64_t 在 Linux 上是 long、在 Windows 上是 long long，
  // 两个平台的重载集不一样（P1 第一次编译就踩过这类"看着一样其实不同"的东西）。
  const int64_t remain = n_tokens - t0;
  const int nt = (remain > kTileTokens) ? kTileTokens : static_cast<int>(remain);
  const uint8_t* row = W + r * row_bytes;

  // acc 用固定长度 + 编译期展开的谓词，保证它留在寄存器里；写成
  // `for (i < nt)` 会让下标变成动态 → 整个数组被打到 local memory。
  float acc[kTileTokens];
#pragma unroll
  for (int i = 0; i < kTileTokens; ++i) acc[i] = 0.0f;

  __shared__ float red[kThreads];

  const int nb = static_cast<int>(cols / block_elems);
  for (int bi = 0; bi < nb; ++bi) {
    const uint8_t* blk = row + static_cast<int64_t>(bi) * block_bytes;
    for (int e = threadIdx.x; e < block_elems; e += kThreads) {
      const float w = gpu::DeqElem(type, blk, e);
      const int64_t col = static_cast<int64_t>(bi) * block_elems + e;
#pragma unroll
      for (int i = 0; i < kTileTokens; ++i) {
        if (i < nt) acc[i] += w * X[(t0 + i) * cols + col];
      }
    }
  }

#pragma unroll
  for (int i = 0; i < kTileTokens; ++i) {
    if (i >= nt) continue;
    // 树形归约。每一步之间都要 __syncthreads，且**所有线程都要走到** ——
    // 这正是上面用 `continue` 而不是 `break`/提前 return 的原因。
    red[threadIdx.x] = acc[i];
    __syncthreads();
    for (int st = kThreads / 2; st > 0; st >>= 1) {
      if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
      __syncthreads();
    }
    if (threadIdx.x == 0) OUT[(t0 + i) * rows + r] = red[0];
    __syncthreads();
  }
}

// ---------------------------------------------------------------- RMSNorm

__global__ void RmsNormKernel(const float* x, const float* w, int64_t n,
                              int64_t stride, float eps, float* out) {
  const int64_t row = blockIdx.x;
  const float* xr = x + row * stride;
  float* orow = out + row * stride;

  __shared__ double red[kThreads];

  // 平方和在 double 里累加（与 CPU 版 kernels.h 同口径）。
  double s = 0.0;
  for (int64_t i = threadIdx.x; i < n; i += kThreads) {
    const double v = static_cast<double>(xr[i]);
    s += v * v;
  }
  red[threadIdx.x] = s;
  __syncthreads();
  for (int st = kThreads / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
    __syncthreads();
  }
  const double mean = red[0] / static_cast<double>(n);
  const float scale = static_cast<float>(1.0 / sqrt(mean + static_cast<double>(eps)));
  // 归约的最后一步已经带 __syncthreads，所以这里读 red[0] 是安全的；
  // 但**写 out 之前**不需要再同步（每个线程只写自己那几个元素）。
  for (int64_t i = threadIdx.x; i < n; i += kThreads) {
    orow[i] = xr[i] * scale * w[i];
  }
}

// ---------------------------------------------------------------- RoPE

__global__ void RopeKernel(float* x, int32_t n_heads, int32_t head_dim,
                           int32_t half, const float* rope_tab, int64_t pos0) {
  const int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  const int64_t per_row = static_cast<int64_t>(n_heads) * half;
  const int64_t row = idx / per_row;
  const int64_t rem = idx - row * per_row;
  const int32_t h = static_cast<int32_t>(rem / half);
  const int32_t i = static_cast<int32_t>(rem - static_cast<int64_t>(h) * half);

  const float* cos_t = rope_tab + (pos0 + row) * head_dim;
  const float* sin_t = cos_t + half;
  float* xh = x + (row * n_heads + h) * head_dim;

  // 半维配对：i 与 i + half 是一对（HF 的 rotate_half）。
  // 写成相邻配对 (2i, 2i+1) 不会崩，只会让注意力错位 —— 表现为"能出词但胡说"。
  const float a = xh[i];
  const float b = xh[i + half];
  const float c = cos_t[i];
  const float s = sin_t[i];
  xh[i] = a * c - b * s;
  xh[i + half] = b * c + a * s;
}

// ---------------------------------------------------------------- Attention

// 一个 block 负责一对 (t, head)。共享内存里放本行的 scores（长度 seq）。
__global__ void CausalAttentionKernel(const float* q, const float* k,
                                      const float* v, int64_t seq,
                                      int32_t n_heads, int32_t n_kv_heads,
                                      int32_t head_dim, float scale, float* out) {
  const int64_t th = blockIdx.x;
  const int32_t t = static_cast<int32_t>(th / n_heads);
  const int32_t h = static_cast<int32_t>(th - static_cast<int64_t>(t) * n_heads);
  const int32_t group = n_heads / n_kv_heads;  // GQA：多个 Q 头共享一个 KV 头
  const int32_t kh = h / group;
  const int64_t q_stride = static_cast<int64_t>(n_heads) * head_dim;
  const int64_t kv_stride = static_cast<int64_t>(n_kv_heads) * head_dim;

  extern __shared__ float sh[];
  float* scores = sh;  // seq 个 float
  // double 归约数组要 8 字节对齐 → 先按 float 数补到偶数。
  const size_t fl = (static_cast<size_t>(seq) + 1u) & ~static_cast<size_t>(1u);
  double* red = reinterpret_cast<double*>(sh + fl);

  const float* qt = q + static_cast<int64_t>(t) * q_stride +
                    static_cast<int64_t>(h) * head_dim;

  // 1) QKᵀ/√d —— 只算 [0, t]，掩码靠"不遍历"实现。
  for (int64_t s = threadIdx.x; s <= t; s += kThreads) {
    const float* ks = k + s * kv_stride + static_cast<int64_t>(kh) * head_dim;
    float acc = 0.0f;
    for (int32_t d = 0; d < head_dim; ++d) acc += qt[d] * ks[d];
    scores[s] = acc * scale;
  }
  __syncthreads();

  // 2) softmax：减最大值（数值稳定）→ exp → double 求和 → 归一化。
  float lmax = -FLT_MAX;
  for (int64_t s = threadIdx.x; s <= t; s += kThreads) {
    lmax = fmaxf(lmax, scores[s]);
  }
  red[threadIdx.x] = static_cast<double>(lmax);
  __syncthreads();
  for (int st = kThreads / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) {
      red[threadIdx.x] = red[threadIdx.x] > red[threadIdx.x + st]
                             ? red[threadIdx.x]
                             : red[threadIdx.x + st];
    }
    __syncthreads();
  }
  const float mx = static_cast<float>(red[0]);
  __syncthreads();  // red 马上要复用成求和累加器

  double lsum = 0.0;
  for (int64_t s = threadIdx.x; s <= t; s += kThreads) {
    const float e = expf(scores[s] - mx);
    scores[s] = e;
    lsum += static_cast<double>(e);
  }
  red[threadIdx.x] = lsum;
  __syncthreads();
  for (int st = kThreads / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
    __syncthreads();
  }
  const float inv = static_cast<float>(1.0 / red[0]);
  for (int64_t s = threadIdx.x; s <= t; s += kThreads) scores[s] *= inv;
  __syncthreads();

  // 3) 加权求和 V。这一段的累加顺序与 CPU 版一致（s 从 0 递增、float 累加）——
  // 不是为了逐位相同，而是因为这是唯一"按 s 顺序读 V"才有的访存局部性。
  float* ot = out + static_cast<int64_t>(t) * q_stride +
              static_cast<int64_t>(h) * head_dim;
  for (int32_t d = threadIdx.x; d < head_dim; d += kThreads) {
    float acc = 0.0f;
    for (int64_t s = 0; s <= t; ++s) {
      const float* vs = v + s * kv_stride + static_cast<int64_t>(kh) * head_dim;
      acc += scores[s] * vs[d];
    }
    ot[d] = acc;
  }
}

// ---------------------------------------------------------------- KV Cache（P2）

// 本段 K/V → cache。源是 [t][head][dim]，目标是 [head][pos][dim] —— **布局不同**，
// 所以是一次带下标重排的遍历，不是整段拷贝。
__global__ void KvWriteKernel(const float* __restrict__ k_src,
                              const float* __restrict__ v_src,
                              int32_t n_kv_heads, int32_t head_dim, int64_t pos0,
                              int64_t max_seq, int64_t total,
                              __half* __restrict__ k_dst,
                              __half* __restrict__ v_dst) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= total) return;
  const int64_t d = i % head_dim;
  const int64_t r = i / head_dim;  // = t * n_kv_heads + h
  const int32_t h = static_cast<int32_t>(r % n_kv_heads);
  const int64_t t = r / n_kv_heads;
  const int64_t dst =
      ((static_cast<int64_t>(h) * max_seq + pos0 + t) * head_dim) + d;
  // 源的下标正好就是 i（它就是 [t][h][d] 行主），不需要再算一遍。
  k_dst[dst] = __float2half_rn(k_src[i]);
  v_dst[dst] = __float2half_rn(v_src[i]);
}

// 从 cache 读历史的因果注意力。一个 block 一对 (t, head) —— 与不带 cache 的
// CausalAttentionKernel 同一结构，只有两处不同：KV 的来源（cache fp16）、
// 遍历上界（pos0 + t 而不是 t）。
//
// ⚠️ 这里与 CausalAttentionKernel 有**刻意的重复**（softmax 那一段一模一样）。
// 不抽公共函数的理由：那个 kernel 的数值路径已经被 P1 的逐层对拍钉死过，
// 动它就意味着 P1 的结论要重证一遍；而复制一份"跑在另一条数据来源上"的实现，
// 出问题时能独立看，代价是几十行。
__global__ void AttentionFromCacheKernel(const float* __restrict__ q,
                                         const __half* __restrict__ k_layer,
                                         const __half* __restrict__ v_layer,
                                         int64_t max_seq, int32_t n_heads,
                                         int32_t n_kv_heads, int32_t head_dim,
                                         float scale, int64_t pos0,
                                         int64_t n_scores,
                                         float* __restrict__ out) {
  const int64_t th = blockIdx.x;
  const int64_t t = th / n_heads;
  const int32_t h = static_cast<int32_t>(th - t * n_heads);
  const int32_t group = n_heads / n_kv_heads;
  const int32_t kh = h / group;
  const int64_t q_stride = static_cast<int64_t>(n_heads) * head_dim;
  const int64_t pos = pos0 + t;  // 本行在**绝对**位置上的下标

  extern __shared__ float sh[];
  // n_scores = pos0 + seq（本段里最大的那个绝对位置 + 1）。**不能按 seq 备**：
  // 本 kernel 的 scores 下标是绝对位置，pos0 > 0 时按 seq 备就会写到缓冲区
  // 后面去。CPU 版在同一处踩过一次堆越界（症状是"单轮正常、第二轮无声消失"），
  // 这里从一开始就按绝对位置数备。
  float* scores = sh;
  const size_t fl = (static_cast<size_t>(n_scores) + 1u) & ~static_cast<size_t>(1u);
  double* red = reinterpret_cast<double*>(sh + fl);  // 8 字节对齐

  const __half* kb =
      k_layer + static_cast<int64_t>(kh) * max_seq * head_dim;
  const __half* vb =
      v_layer + static_cast<int64_t>(kh) * max_seq * head_dim;
  const float* qt =
      q + t * q_stride + static_cast<int64_t>(h) * head_dim;

  // 1) QKᵀ/√d —— 遍历 [0, pos]（cache 里的全部历史，含本段排在它前面的）。
  for (int64_t s = threadIdx.x; s <= pos; s += kThreads) {
    const __half* ks = kb + s * head_dim;
    float acc = 0.0f;
    for (int32_t d = 0; d < head_dim; ++d) acc += qt[d] * __half2float(ks[d]);
    scores[s] = acc * scale;
  }
  __syncthreads();

  // 2) softmax（减最大值 → exp → double 求和 → 归一化）
  float lmax = -FLT_MAX;
  for (int64_t s = threadIdx.x; s <= pos; s += kThreads) {
    lmax = fmaxf(lmax, scores[s]);
  }
  red[threadIdx.x] = static_cast<double>(lmax);
  __syncthreads();
  for (int st = kThreads / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) {
      red[threadIdx.x] = red[threadIdx.x] > red[threadIdx.x + st]
                             ? red[threadIdx.x]
                             : red[threadIdx.x + st];
    }
    __syncthreads();
  }
  const float mx = static_cast<float>(red[0]);
  __syncthreads();  // red 马上要复用成求和累加器

  double lsum = 0.0;
  for (int64_t s = threadIdx.x; s <= pos; s += kThreads) {
    const float e = expf(scores[s] - mx);
    scores[s] = e;
    lsum += static_cast<double>(e);
  }
  red[threadIdx.x] = lsum;
  __syncthreads();
  for (int st = kThreads / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
    __syncthreads();
  }
  const float inv = static_cast<float>(1.0 / red[0]);
  for (int64_t s = threadIdx.x; s <= pos; s += kThreads) scores[s] *= inv;
  __syncthreads();

  // 3) 加权求和 V（按 s 递增读 cache —— cache 里同一头的连续位置是连续的）
  float* ot = out + t * q_stride + static_cast<int64_t>(h) * head_dim;
  for (int32_t d = threadIdx.x; d < head_dim; d += kThreads) {
    float acc = 0.0f;
    for (int64_t s = 0; s <= pos; ++s) {
      acc += scores[s] * __half2float(vb[s * head_dim + d]);
    }
    ot[d] = acc;
  }
}

// ---------------------------------------------------------------- ArgMax

__global__ void ArgMaxKernel(const float* __restrict__ x, int64_t n,
                             int32_t* __restrict__ out_idx) {
  float bv = -FLT_MAX;
  int32_t bi = 0;
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = x[i];
    // 严格大于 → 线程内并列时保留**先遇到**的那个（下标更小）
    if (v > bv) {
      bv = v;
      bi = static_cast<int32_t>(i);
    }
  }
  __shared__ float sv[kArgMaxThreads];
  __shared__ int32_t si[kArgMaxThreads];
  sv[threadIdx.x] = bv;
  si[threadIdx.x] = bi;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) {
      const float vr = sv[threadIdx.x + st];
      const int32_t ir = si[threadIdx.x + st];
      // 归约时也要保证"并列取最小下标"，否则跨线程的并列会取到任意一个 ——
      // 而 CPU 版是确定性的最小下标。这一条不写，判据会随机红。
      if (vr > sv[threadIdx.x] ||
          (vr == sv[threadIdx.x] && ir < si[threadIdx.x])) {
        sv[threadIdx.x] = vr;
        si[threadIdx.x] = ir;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *out_idx = si[0];
}

// ---------------------------------------------------------------- 逐元素

__global__ void SiluMulKernel(float* gate, const float* up, int64_t n) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  const float v = gate[i];
  // 与 CPU 版 kernels.h 的 SiluInPlace + MulInPlace 同序（先 silu 再乘）。
  gate[i] = (v / (1.0f + expf(-v))) * up[i];
}

__global__ void VecAddKernel(float* a, const float* b, int64_t n) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  a[i] += b[i];
}

__global__ void EmbeddingLookupKernel(uint32_t type, const uint8_t* emb,
                                      int64_t hidden, int64_t row_bytes,
                                      int32_t block_elems, int32_t block_bytes,
                                      const int32_t* toks, int64_t seq, float* x) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= seq * hidden) return;
  const int64_t t = i / hidden;
  const int64_t e = i - t * hidden;
  const uint8_t* row = emb + static_cast<int64_t>(toks[t]) * row_bytes;
  const int64_t bi = e / block_elems;
  x[i] = gpu::DeqElem(type, row + bi * block_bytes,
                      static_cast<int>(e - bi * block_elems));
}

// 一维 grid 的通用算法（向上取整），并把 grid 上限检查在这里做掉 ——
// 词表 151936 × 4 这种规模的 grid 在 gridDim.x 的 2^31 之内，但"想当然"是
// 一类可以一次性消除的 bug。
unsigned GridFor(int64_t n, int threads) {
  const int64_t g = (n + threads - 1) / threads;
  RPG_CHECK(g > 0 && g <= 0x7FFFFFFF, "grid 维度越界：%lld", (long long)g);
  return static_cast<unsigned>(g);
}

}  // namespace

// ---------------------------------------------------------------- 启动入口

void GpuDequantRow(uint32_t type, const uint8_t* src_dev, int64_t n_elems,
                   float* dst_dev) {
  const BlockShape bs = ShapeOf(type);
  RPG_CHECK(n_elems % bs.elems == 0, "反量化长度 %lld 不是块大小 %d 的整数倍",
            static_cast<long long>(n_elems), bs.elems);
  DequantRowKernel<<<GridFor(n_elems, 256), 256>>>(type, src_dev, n_elems,
                                                   bs.elems, bs.bytes, dst_dev);
  CheckLaunch("DequantRowKernel");
}

void GpuMatMulQuant(const GpuTensor& w, const float* X_dev, int64_t n_tokens,
                    float* OUT_dev) {
  RPG_CHECK(w.matrix, "GpuMatMulQuant 收到非矩阵张量（%s）", w.type_name());
  RPG_CHECK(n_tokens > 0, "n_tokens 必须为正");
  const BlockShape bs = ShapeOf(w.type);
  RPG_CHECK(w.cols % bs.elems == 0, "点积长度 %lld 不是块大小 %d 的整数倍",
            static_cast<long long>(w.cols), bs.elems);
  RPG_CHECK(w.buf.size() == static_cast<size_t>(w.rows) * w.row_bytes(),
            "张量字节数与 [%lld,%lld] 的形状不符 —— 加载期就该发现，不该等到这里",
            static_cast<long long>(w.rows), static_cast<long long>(w.cols));
  const unsigned gy = GridFor(n_tokens, kTileTokens);
  const dim3 grid(static_cast<unsigned>(w.rows), gy);
  MatMulQuantKernel<<<grid, kThreads>>>(
      w.type, static_cast<const uint8_t*>(w.buf.data()), w.rows, w.row_bytes(),
      w.cols, bs.elems, bs.bytes, X_dev, n_tokens, OUT_dev);
  CheckLaunch("MatMulQuantKernel");
}

void GpuRmsNorm(const float* x, const float* w, int64_t n_rows, int64_t n,
                int64_t stride, float eps, float* out) {
  RPG_CHECK(n > 0 && n_rows > 0 && stride >= n, "RmsNorm 参数不合法");
  RmsNormKernel<<<static_cast<unsigned>(n_rows), kThreads>>>(x, w, n, stride, eps,
                                                             out);
  CheckLaunch("RmsNormKernel");
}

void GpuRope(float* x, int32_t n_heads, int32_t head_dim, const float* rope_tab,
             int64_t pos0, int64_t n_rows) {
  const int32_t half = head_dim / 2;
  RPG_CHECK(head_dim % 2 == 0, "head_dim %d 必须是偶数（RoPE 半维配对）", head_dim);
  const int64_t total = n_rows * n_heads * half;
  RopeKernel<<<GridFor(total, 256), 256>>>(x, n_heads, head_dim, half, rope_tab,
                                           pos0);
  CheckLaunch("RopeKernel");
}

void GpuCausalAttention(const ModelMeta& c, const float* q_dev,
                        const float* k_dev, const float* v_dev, int64_t seq,
                        float* out_dev) {
  RPG_CHECK(seq > 0, "attention 的 seq 必须为正");
  const float scale = 1.0f / sqrtf(static_cast<float>(c.head_dim));
  // 共享内存 = scores[seq] + 8 字节对齐的 red[kThreads]。
  const size_t fl = (static_cast<size_t>(seq) + 1u) & ~static_cast<size_t>(1u);
  const size_t sh_bytes = (fl + kThreads * 2u) * sizeof(float);
  // 超过默认的 48 KiB 才需要显式申请。seq 到 4096 时是 17 KiB，用不着 ——
  // 但把这行留着，免得将来把 max_seq 调大之后收到一个"启动失败"的谜题。
  if (sh_bytes > 48 * 1024) {
    RPG_CUDA_CHECK(cudaFuncSetAttribute(
        CausalAttentionKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(sh_bytes)));
  }
  const int64_t blocks = seq * c.n_heads;
  CausalAttentionKernel<<<static_cast<unsigned>(blocks), kThreads, sh_bytes>>>(
      q_dev, k_dev, v_dev, seq, c.n_heads, c.n_kv_heads, c.head_dim, scale,
      out_dev);
  CheckLaunch("CausalAttentionKernel");
}

void GpuKvWrite(const float* k_dev, const float* v_dev, int32_t n_kv_heads,
                int32_t head_dim, int64_t pos0, int64_t seq,
                uint16_t* k_cache_layer, uint16_t* v_cache_layer,
                int64_t max_seq) {
  RPG_CHECK(seq > 0 && n_kv_heads > 0 && head_dim > 0, "KvWrite 参数不合法");
  RPG_CHECK(pos0 >= 0, "写入位置 %lld 不能为负", static_cast<long long>(pos0));
  RPG_CHECK(pos0 + seq <= max_seq, "写入区间 [%lld, %lld) 超出 cache 容量 %lld",
            static_cast<long long>(pos0), static_cast<long long>(pos0 + seq),
            static_cast<long long>(max_seq));
  const int64_t total = seq * n_kv_heads * head_dim;
  KvWriteKernel<<<GridFor(total, 256), 256>>>(
      k_dev, v_dev, n_kv_heads, head_dim, pos0, max_seq, total,
      reinterpret_cast<__half*>(k_cache_layer),
      reinterpret_cast<__half*>(v_cache_layer));
  CheckLaunch("KvWriteKernel");
}

void GpuAttentionFromCache(const ModelMeta& c, int64_t pos0, const float* q_dev,
                           int64_t seq, const uint16_t* k_layer,
                           const uint16_t* v_layer, int64_t max_seq,
                           float* out_dev) {
  RPG_CHECK(seq > 0, "attention 的 seq 必须为正");
  RPG_CHECK(pos0 >= 0, "pos0 不能为负");
  const float scale = 1.0f / sqrtf(static_cast<float>(c.head_dim));
  // scores 按**绝对位置数** n_scores = pos0 + seq 备（见 kernel 里的说明）。
  const int64_t n_scores = pos0 + seq;
  const size_t fl =
      (static_cast<size_t>(n_scores) + 1u) & ~static_cast<size_t>(1u);
  const size_t sh_bytes = (fl + kThreads * 2u) * sizeof(float);
  // 默认上限 48 KiB：n_scores 到 6144（RPGBar 的上下文上限）时是 6144 + 256 =
  // 6400 个 float = 25.6 KiB，够用。留这段是为了将来把 max_seq 调更大时，
  // 收到的是一条明确的启动失败，而不是一个"kernel 莫名返回错误"的谜题。
  if (sh_bytes > 48 * 1024) {
    RPG_CUDA_CHECK(cudaFuncSetAttribute(
        AttentionFromCacheKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(sh_bytes)));
  }
  const int64_t blocks = seq * c.n_heads;
  AttentionFromCacheKernel<<<static_cast<unsigned>(blocks), kThreads, sh_bytes>>>(
      q_dev, reinterpret_cast<const __half*>(k_layer),
      reinterpret_cast<const __half*>(v_layer), max_seq, c.n_heads,
      c.n_kv_heads, c.head_dim, scale, pos0, n_scores, out_dev);
  CheckLaunch("AttentionFromCacheKernel");
}

void GpuArgMax(const float* x_dev, int64_t n, int32_t* out_idx_dev) {
  RPG_CHECK(n > 0, "argmax 的长度必须为正");
  RPG_CHECK(n <= 0x7FFFFFFF, "argmax 的下标超出 int32（%lld）",
            static_cast<long long>(n));
  ArgMaxKernel<<<1, kArgMaxThreads>>>(x_dev, n, out_idx_dev);
  CheckLaunch("ArgMaxKernel");
}

void GpuSiluMulInPlace(float* gate_dev, const float* up_dev, int64_t n) {
  if (n <= 0) return;
  SiluMulKernel<<<GridFor(n, 256), 256>>>(gate_dev, up_dev, n);
  CheckLaunch("SiluMulKernel");
}

void GpuVecAddInPlace(float* a_dev, const float* b_dev, int64_t n) {
  if (n <= 0) return;
  VecAddKernel<<<GridFor(n, 256), 256>>>(a_dev, b_dev, n);
  CheckLaunch("VecAddKernel");
}

void GpuEmbeddingLookup(const GpuTensor& emb, const int32_t* toks_dev, int64_t seq,
                        float* x_dev) {
  const BlockShape bs = ShapeOf(emb.type);
  RPG_CHECK(emb.cols % bs.elems == 0, "embedding 宽度 %lld 不是块大小的整数倍",
            static_cast<long long>(emb.cols));
  const int64_t n = seq * emb.cols;
  EmbeddingLookupKernel<<<GridFor(n, 256), 256>>>(
      emb.type, static_cast<const uint8_t*>(emb.buf.data()), emb.cols,
      emb.row_bytes(), bs.elems, bs.bytes, toks_dev, seq, x_dev);
  CheckLaunch("EmbeddingLookupKernel");
}

}  // namespace rpg

#endif  // RPG_HAS_CUDA
