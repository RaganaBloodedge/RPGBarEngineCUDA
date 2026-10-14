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
#include <cstdlib>

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

// 通用 kernel 用的 tile（= 专用 kernel 的 TILE 下限，保持两条路径形状可比）。
constexpr int kTileTokens = 4;

// ------------------------------------------------------- 形状选择（TILE × ROWS）
//
// 一个 block 负责「ROWS 行 × TILE 个 token」，两个维度正交，各自解决一个瓶颈：
//
//   * TILE = 一个 block 装几个 token。整行权重被重读的遍数 = ceil(n/TILE) ⇒
//     **prefill 的代价几乎全由它决定**（TILE=4 时 900 个新 token 要过 225 遍）。
//   * ROWS = 一个 block 算几行。元素循环里 X 的读**只跟列有关、与行无关**（同一列
//     在每一行里都是同一个输入值）⇒ R 行共用一组 X 寄存器，X 访存与 block 数都
//     除以 R。这在 decode 上值钱：4B 的 lm_head 有 151936 行，R=1 时一个矩阵就要
//     15.2 万个 block，而实测每个 block 光"启动 + 跑完"就要 35~113 ns。
//
// ⚠️ 两个维度**都在吃寄存器**（acc[ROWS][TILE] + x[TILE]），而 decode 与 prefill 的
// 最优形状是**相反的**：
//   * prefill（n 大）要 TILE 大 —— 多装 token 才能少重读权重；
//   * decode（n=1）要 TILE=1 + ROWS 大 —— n=1 时 TILE>1 的槽位**全是废的**：
//     内层 `for (i < TILE)` 里那 3 个用不上的槽位被谓词掉，但 predicated 指令
//     照样占发射槽 ⇒ TILE=4 在 decode 上等于白跑 4 倍的发射量。
//
// 实测（RTX 5060 Laptop，见 tools/matmul_bench）：
//   * TILE 单独往上走是净亏：TILE=16 ⇒ REG 96 / 13 KiB（占用率 42%）、TILE=32 ⇒
//     REG 168 / 21 KiB（25%），单 tile 成本涨 4~8 倍。所以 TILE 只到 8。
//   * ROWS=4 配 TILE=4 也是净亏：REG 从 40 涨到 78，占用率腰斩，lm_head
//     51.9 → 40.1 GB/s。只有"轻的那一维"能大。
//
// 结论：按 n_tokens 在两条曲线之间选形状 —— decode 走 (TILE=1, ROWS 大)，
// prefill 走 (TILE 大, ROWS=1)。
struct MatMulShape {
  int tile;
  int rows;
};

// 扫参用的强制覆盖（只给 tools/matmul_bench 用，不是生产配置）：两个环境变量都设
// 且组合在下面的实例化表里时才生效。设了非法组合会走 switch 的 default 分支报错，
// 不会静默换一个形状。
const MatMulShape* ForcedShape() {
  static const MatMulShape s = [] {
    const char* t = std::getenv("RPG_MM_TILE");
    const char* r = std::getenv("RPG_MM_ROWS");
    MatMulShape v{0, 0};
    if (t != nullptr && r != nullptr) {
      v.tile = std::atoi(t);
      v.rows = std::atoi(r);
    }
    return v;
  }();
  return (s.tile > 0 && s.rows > 0) ? &s : nullptr;
}

// 默认策略（依据见 Temp/p4-report.txt 的三张扫参表）：
//
// 一个 block 的**单遍成本**几乎与它装多少 token 无关（实测 TILE=1/4/8 各约
// 4.3 / 7.25 / 9.7 ms per pass），而整行权重的**遍数** = ceil(n/TILE)。所以总成本
// ≈ ceil(n/TILE) × per_pass(TILE)，最优点随 n 在几个 TILE 之间跳：
//     n = 1      → TILE=1（4.3 ms）比 TILE=4（7.25）便宜 —— n=1 时 TILE>1 的槽位全废
//     n = 2..4   → TILE=4（7.25）胜出（ceil(2/1)=2 遍 ×4.3 = 8.6 > 7.25）
//     n >= 5     → TILE=8（9.7）胜出（ceil(5/4)=2 遍 ×7.25 = 14.5 > 9.7）
//
// ROWS 一律取 2：它对 TILE=1 是明显收益（block 数减半、X 复用翻倍），对 TILE=8 也是
// 小幅收益，而 4 以上会把 d1s/m1s 的共享内存推上去、把 occupancy 吃掉。
MatMulShape PickShape(int64_t n_tokens) {
  if (const MatMulShape* f = ForcedShape()) return *f;
  if (n_tokens <= 1) return MatMulShape{1, 2};
  if (n_tokens <= 4) return MatMulShape{4, 2};
  return MatMulShape{8, 2};
}

// 一行最多有多少个量化块（cols / 256）。专用 matmul kernel 按它开共享内存：
// Q4_K 是 nb*8 组 (d1,m1)、Q6_K 是 nb*16 个 d*sc，各自 ×4 字节。
//
// ⚠️ **这个上限原来是 32，而它是照 1.7B 的 cols 定的**（hidden 2048 / ffn 6144
// ⇒ 最大 nb = 24）。4B 的 ffn_down 是 cols=9728 ⇒ nb=38 ⇒ 直接掉进通用慢 kernel，
// 而它占每层 32% 的权重字节 —— 等于"4B 上有一整类矩阵从来没走过专用路径"。
// 抬到 64 之后 4B（最大 nb=38）全在快路径上；共享内存代价是 Q4_K 2×64×8×4=4 KiB、
// Q6_K 64×16×4=4 KiB，相对 48 KiB 的默认上限仍是小头。
constexpr int kMaxQuantBlocks = 64;

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
  RPG_CHECK(t != nullptr, "unknown ggml type id=%u", type);
  return BlockShape{static_cast<int32_t>(t->block_elems),
                    static_cast<int32_t>(t->block_bytes)};
}

// 启动之后立刻查一次 —— 参数配置错（grid 超限、共享内存超限）在这里就报出来，
// 而不是等最后同步时报一个看不出出处的错误。
void CheckLaunch(const char* what) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    Throw("kernel launch failed (%s): %s", what, cudaGetErrorString(e));
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

// --------------------------------------------- 融合点积（Q4_K / Q6_K 专用，P4 重写）
//
// 下面两个 kernel 与上面的 `MatMulQuantKernel` **数值语义逐位相同**，区别只有
// "哪些量算一次、哪些量每个元素重算一遍"：
//
//   * 通用版走 `gpu::DeqElem`，每个元素都要重算一遍块常量 —— 两次 F16ToF32
//     （各带一个分支和一个次正规数的 while 循环）、一次 6-bit scale/min 位拼接、
//     三次**非收缩**乘法，约 50 条指令；而每个元素只读 0.5625 字节 ⇒ 有效带宽
//     只有峰值的 ~10%（`tools/matmul_bench` 量出来的：n_tokens 1→4 耗时几乎不变，
//     说明瓶颈是"每个元素的处理成本"，与 token 数无关）。
//   * 专用版把每 256 个元素一份的 d/dmin、每 32 个元素（Q6_K 是每 16 个）一份的
//     "已乘上 d 的 scale"提到元素循环外**算一次**，元素级只剩"取半字节 + 1 mul +
//     1 sub"。块常量在 **block 内跨线程共享**，用共享内存承载；整行的块常量一次
//     算完（`nb*8` 组），于是元素循环里**一次 `__syncthreads` 都不需要** ——
//     按块算的话两次同步之间只有 2 个元素/线程的计算量，同步开销与计算量同量级。
//
// ⚠️ **逐位不变是硬要求**，所以下面这些一个都没动：
//   ① 线程 → 元素的映射（`e = threadIdx.x; e < 256; e += kThreads`）；
//   ② 累加顺序（同一 acc[i] 仍旧按 (bi, e) 的字典序累加）；
//   ③ 树形归约（与通用版逐字相同）；
//   ④ 三个乘法仍旧 `__fmul_rn` / `__fsub_rn` 钉死 —— 提前算出来的 (d·sc) 与
//      (dmin·m) 存进共享内存再读回是无损的（float 存 float），元素级那两次乘减
//      与重算一遍得到的是同一组 bit。理由见 `quant_gpu.cuh` 的头注释。
//
// 直接证据由 `tools/matmul_bench --verify` 给出：同一份权重与输入，通用 kernel 与
// 专用 kernel 的输出**逐位比较**（不是比相对误差 —— 那是 P1 的活）。

// ---------------------------------------------------- 归约与写回（TILE 张表并行）
//
// 树形归约的**加法序列与通用 kernel 逐字相同**：
//     part[i][t] = acc[i]_t；随后 st = 64,32,…,1，每一步 part[i][t] += part[i][t+st]；
//     最后 part[i][0] 就是第 i 个 token 的输出。
//
// ⚠️ 唯一的区别是**交错**：这里把 TILE 个 token 的归约并起来做（一次 8 步），原来
// 是每个 token 各 8 步、串行做完 TILE 次。__syncthreads 不是算术（它不参与浮点
// 结果），把哪些 token 的加法放进同一个同步区间**不改变任何一个加法的操作数与
// 顺序** ⇒ 输出逐位相同。行与行之间是**串行**做的（每行 8 步），所以 part 只需要
// TILE*kThreads 这么大，不必再乘 ROWS。
template <int TILE, int ROWS>
__device__ __forceinline__ void ReduceAndStore(float* part,
                                               const float (&acc)[ROWS][TILE],
                                               int nrow, int nt, float* OUT,
                                               int64_t rows, int64_t r0, int64_t t0) {
#pragma unroll
  for (int rr = 0; rr < ROWS; ++rr) {
#pragma unroll
    for (int i = 0; i < TILE; ++i) part[i * kThreads + threadIdx.x] = acc[rr][i];
    __syncthreads();
    // 所有线程都要走到每一步（哪怕 i >= nt 的那几张表装的是垃圾）—— 同步不能分叉。
    for (int st = kThreads / 2; st > 0; st >>= 1) {
#pragma unroll
      for (int i = 0; i < TILE; ++i) {
        if (threadIdx.x < st) {
          part[i * kThreads + threadIdx.x] += part[i * kThreads + threadIdx.x + st];
        }
      }
      __syncthreads();
    }
    // 越界的行（最后一个 block）照样参与归约，只是不写回 —— 循环里就不用分叉了。
#pragma unroll
    for (int i = 0; i < TILE; ++i) {
      if (i < nt && rr < nrow && threadIdx.x == 0) {
        OUT[(t0 + i) * rows + (r0 + rr)] = part[i * kThreads];
      }
    }
    __syncthreads();   // 复用 part 之前必须同步
  }
}

// ------------------------------------ 动态共享内存布局（host 启动 / device 取址共用）
//
// 为什么必须动态：常量表按**一行**算就是 nb*8（Q4_K）或 nb*16（Q6_K）个 float，
// 而静态数组只能按最坏情况开（cols 上限 ⇒ 64 个量化块 = 2~4 KiB）。4B 的实际 nb
// 只有 10~38 —— 按最坏情况开等于白扔一半共享内存，而共享内存直接决定每个 SM 能
// 驻几个 block。
//
// 三段的大小都由下面两个函数算，**host 与 device 用同一个公式**（抄两份必漂，
// 而且只在 nb 跨过某个值时暴露）。布局顺序固定：
//     d1s[ROWS*nb*kSubs] | m1s[ROWS*nb*kSubs] | part[TILE*kThreads]
// （Q6_K 的 dsc 就是 d1s 那一段，宽度换成 16。）
template <int TILE, int ROWS>
size_t Q4KShmemFloats(int nb) {
  return 2u * static_cast<size_t>(ROWS) * nb * 8u +
         static_cast<size_t>(TILE) * kThreads;
}
template <int TILE, int ROWS>
size_t Q6KShmemFloats(int nb) {
  return static_cast<size_t>(ROWS) * nb * 16u +
         static_cast<size_t>(TILE) * kThreads;
}

// Q4_K：块 144 字节 / 256 元素 = 8 个子块（每个 32 元素）。
// 子块号 `sub` 恰好就是 `GetScaleMinK4` 的 `j`（`j = 2*(sub>>1) + (sub&1) = sub`）。
//
// 两个"一个 block 干多少活"的模板参数，都不参与任何算术（怎么选见 PickShape）：
//   * TILE = 这个 block 负责几个 token —— 越大，整行权重被重读的遍数越少
//     （遍数 = ceil(n / TILE)），prefill 的代价几乎全由它决定；
//   * ROWS = 这个 block 负责几行 —— 越大，同一组 X 值被复用的次数越多。
//
// 为什么 ROWS 这一维值钱：元素循环里 X 的读**只跟列有关**，与行无关（同一列在每
// 一行里都是同一个输入值）。把 R 行放进同一个 block，一组 X 就能喂 R 行，X 的访存
// 直接除以 R。而 X 的访存量本来是 `rows * n * cols` 量级、与权重字节同阶甚至更大：
// 4B 的 lm_head 有 151936 行，n=1 时它一个矩阵就要 15.2 万个 block —— 实测成本
// 几乎与 block 数成正比（各矩阵都落在 35~113 ns/block），也就是说这台机器上
// "启动并把一个 block 跑完"本身就要 ~100 ns，而那个 block 真正读的权重只有 2 KB。
//
// ⚠️ 逐位不变仍然成立：每个输出元素 (rr, i) 的部分和依旧由**同一个线程**按 (bi, e)
// 的字典序累加（多出来的只是"哪个 block 在算"，行的循环不改变累加序），树形归约
// 的加法序列也一字未动。
template <int TILE, int ROWS>
__global__ void MatMulQ4KKernel(const uint8_t* W, int64_t rows, int64_t row_bytes,
                                int64_t cols, const float* X, int64_t n_tokens,
                                float* OUT) {
  constexpr int kBlockElems = 256;
  constexpr int kBlockBytes = 144;
  constexpr int kSubs = 8;

  const int64_t r0 = static_cast<int64_t>(blockIdx.x) * ROWS;
  const int64_t t0 = static_cast<int64_t>(blockIdx.y) * TILE;
  const int64_t nrow64 = rows - r0;
  const int nrow = (nrow64 > ROWS) ? ROWS : static_cast<int>(nrow64);
  const int64_t remain = n_tokens - t0;
  const int nt = (remain > TILE) ? TILE : static_cast<int>(remain);

  // acc 用固定长度 + 编译期展开的谓词，保证它留在寄存器里；写成
  // `for (i < nt)` 会让下标变成动态 → 整个数组被打到 local memory。
  float acc[ROWS][TILE];
#pragma unroll
  for (int rr = 0; rr < ROWS; ++rr) {
#pragma unroll
    for (int i = 0; i < TILE; ++i) acc[rr][i] = 0.0f;
  }

  const int nb = static_cast<int>(cols / kBlockElems);
  extern __shared__ float smem[];
  float* d1s = smem;
  float* m1s = smem + static_cast<size_t>(ROWS) * nb * kSubs;
  float* part = m1s + static_cast<size_t>(ROWS) * nb * kSubs;

  // 整行的块常量**一次算完**：(d1, m1) 共 ROWS*nb*8 组，之后元素循环里没有任何同步。
  for (int j = threadIdx.x; j < ROWS * nb * kSubs; j += kThreads) {
    const int rr = j / (nb * kSubs);
    const int jj = j - rr * (nb * kSubs);
    if (rr >= nrow) continue;
    const int bi = jj / kSubs;
    const int sub = jj - bi * kSubs;
    const uint8_t* blk =
        W + (r0 + rr) * row_bytes + static_cast<int64_t>(bi) * kBlockBytes;
    const float d = gpu::F16ToF32(static_cast<uint16_t>(blk[0] | (blk[1] << 8)));
    const float dmin = gpu::F16ToF32(static_cast<uint16_t>(blk[2] | (blk[3] << 8)));
    int sc = 0, m = 0;
    gpu::GetScaleMinK4(sub, blk + 4, &sc, &m);
    d1s[j] = __fmul_rn(d, static_cast<float>(sc));
    m1s[j] = __fmul_rn(dmin, static_cast<float>(m));
  }
  __syncthreads();

  // ⚠️ `#pragma unroll 4` 不是为了省指令，是为了**让 4 个量化块的字节读同时在飞**：
  // 循环体里每个线程只读 1~2 个字节，不展开的话每次迭代都要等上一次的显存往返，
  // 而一个 block 只有 nb（4B 上 10~38）次迭代。展开后 4 次迭代的地址互相独立。
#pragma unroll 4
  for (int bi = 0; bi < nb; ++bi) {
    const int base = bi * kSubs;
    const int64_t col0 = static_cast<int64_t>(bi) * kBlockElems;
    for (int e = threadIdx.x; e < kBlockElems; e += kThreads) {
      const int sub = e >> 5;
      const int l = e & 31;
      const int64_t col = col0 + e;
      // X 只跟列有关 ⇒ 先读进寄存器，下面 ROWS 行共用这一组值。
      float x[TILE];
#pragma unroll
      for (int i = 0; i < TILE; ++i) {
        x[i] = (i < nt) ? X[(t0 + i) * cols + col] : 0.0f;
      }
#pragma unroll
      for (int rr = 0; rr < ROWS; ++rr) {
        if (rr >= nrow) continue;
        const uint8_t* blk =
            W + (r0 + rr) * row_bytes + static_cast<int64_t>(bi) * kBlockBytes;
        const uint8_t* qs = blk + 16 + (sub >> 1) * 32;
        const int raw = (sub & 1) ? (qs[l] >> 4) : (qs[l] & 0x0F);
        const int off = rr * nb * kSubs + base + sub;
        const float w = __fsub_rn(__fmul_rn(d1s[off], static_cast<float>(raw)),
                                  m1s[off]);
#pragma unroll
        for (int i = 0; i < TILE; ++i) {
          if (i < nt) acc[rr][i] += w * x[i];
        }
      }
    }
  }

  ReduceAndStore<TILE, ROWS>(part, acc, nrow, nt, OUT, rows, r0, t0);
}

// Q6_K：块 210 字节 / 256 元素 = 2 个半块（各 128 元素）× 4 段（各 32 元素）×
// 2 个 16 元素小组 = 16 个 int8 scale；d 每块一个。
// 线程 t 的两个元素（e = t 与 t + 128）**只差一个 half**，其余下标全同 ⇒
// 它们用同一个 scale 下标（`is + 2*quarter`），只是落在两个半块各自的 sc[] 里。
// （TILE / ROWS 的含义与 Q4_K 版完全一致。）
template <int TILE, int ROWS>
__global__ void MatMulQ6KKernel(const uint8_t* W, int64_t rows, int64_t row_bytes,
                                int64_t cols, const float* X, int64_t n_tokens,
                                float* OUT) {
  constexpr int kBlockElems = 256;
  constexpr int kBlockBytes = 210;

  const int64_t r0 = static_cast<int64_t>(blockIdx.x) * ROWS;
  const int64_t t0 = static_cast<int64_t>(blockIdx.y) * TILE;
  const int64_t nrow64 = rows - r0;
  const int nrow = (nrow64 > ROWS) ? ROWS : static_cast<int>(nrow64);
  const int64_t remain = n_tokens - t0;
  const int nt = (remain > TILE) ? TILE : static_cast<int>(remain);

  float acc[ROWS][TILE];
#pragma unroll
  for (int rr = 0; rr < ROWS; ++rr) {
#pragma unroll
    for (int i = 0; i < TILE; ++i) acc[rr][i] = 0.0f;
  }

  const int nb = static_cast<int>(cols / kBlockElems);
  extern __shared__ float smem[];
  float* dsc = smem;   // dsc[rr*nb*16 + bi*16 + half*8 + idx] = d * sc
  float* part = smem + static_cast<size_t>(ROWS) * nb * 16;

  for (int j = threadIdx.x; j < ROWS * nb * 16; j += kThreads) {
    const int rr = j / (nb * 16);
    const int jj = j - rr * (nb * 16);
    if (rr >= nrow) continue;
    const int bi = jj >> 4;
    const int half = (jj >> 3) & 1;
    const int idx = jj & 7;
    const uint8_t* blk =
        W + (r0 + rr) * row_bytes + static_cast<int64_t>(bi) * kBlockBytes;
    const float d = gpu::F16ToF32(static_cast<uint16_t>(blk[208] | (blk[209] << 8)));
    const int8_t sc = reinterpret_cast<const int8_t*>(blk + 192 + half * 8)[idx];
    dsc[j] = __fmul_rn(d, static_cast<float>(sc));
  }
  __syncthreads();

  // 展开的理由见 Q4_K 版同一处注释（把 4 个量化块的字节读同时发出去）。
#pragma unroll 4
  for (int bi = 0; bi < nb; ++bi) {
    const int base = bi * 16;
    const int64_t col0 = static_cast<int64_t>(bi) * kBlockElems;
    for (int e = threadIdx.x; e < kBlockElems; e += kThreads) {
      const int half = e >> 7;
      const int rr = e & 127;
      const int quarter = rr >> 5;
      const int l = rr & 31;
      const int64_t col = col0 + e;
      float x[TILE];
#pragma unroll
      for (int i = 0; i < TILE; ++i) {
        x[i] = (i < nt) ? X[(t0 + i) * cols + col] : 0.0f;
      }
#pragma unroll
      for (int row_i = 0; row_i < ROWS; ++row_i) {
        if (row_i >= nrow) continue;
        const uint8_t* blk =
            W + (r0 + row_i) * row_bytes + static_cast<int64_t>(bi) * kBlockBytes;
        const int is = l >> 4;
        const uint8_t byte = blk[half * 64 + l + ((quarter & 1) ? 32 : 0)];
        const int nib = (quarter < 2) ? (byte & 0x0F) : (byte >> 4);
        const int hi = (blk[128 + half * 32 + l] >> (2 * quarter)) & 3;
        const int raw = (nib | (hi << 4)) - 32;
        const float w = __fmul_rn(
            dsc[row_i * nb * 16 + base + half * 8 + is + 2 * quarter],
            static_cast<float>(raw));
#pragma unroll
        for (int i = 0; i < TILE; ++i) {
          if (i < nt) acc[row_i][i] += w * x[i];
        }
      }
    }
  }

  ReduceAndStore<TILE, ROWS>(part, acc, nrow, nt, OUT, rows, r0, t0);
}

// 是否该把这次点积分派给专用 kernel。块形状必须与上面两个 kernel 的假设一致
// （256 元素 / 144、210 字节）—— 不一致时**宁可退回通用版**也不要静默按错的
// 布局拆包（症状是"能出词但胡说"）。返回 false 就退通用版。
bool MatMulHasSpecializedKernel(uint32_t type) {
  if (type == kGgmlQ4_K) {
    const GgmlTypeInfo* t = TypeInfo(kGgmlQ4_K);
    return t != nullptr && t->block_elems == 256 && t->block_bytes == 144;
  }
  if (type == kGgmlQ6_K) {
    const GgmlTypeInfo* t = TypeInfo(kGgmlQ6_K);
    return t != nullptr && t->block_elems == 256 && t->block_bytes == 210;
  }
  return false;
}

// 按 (TILE, ROWS) 实例化启动专用 kernel（Q4_K / Q6_K 各自一个模板）。
// 动态共享内存的大小与内核里取址用的是**同一个公式**（Q4K/Q6KShmemFloats）。
template <int TILE, int ROWS>
void LaunchSpecialized(uint32_t type, const dim3& grid, const uint8_t* wdev,
                       const GpuTensor& w, const float* X_dev, int64_t n_tokens,
                       float* OUT_dev) {
  const int nb = static_cast<int>(w.cols / 256);
  if (type == kGgmlQ4_K) {
    const size_t sh = Q4KShmemFloats<TILE, ROWS>(nb) * sizeof(float);
    MatMulQ4KKernel<TILE, ROWS><<<grid, kThreads, sh, EngineStream()>>>(
        wdev, w.rows, w.row_bytes(), w.cols, X_dev, n_tokens, OUT_dev);
    CheckLaunch("MatMulQ4KKernel");
  } else {
    const size_t sh = Q6KShmemFloats<TILE, ROWS>(nb) * sizeof(float);
    MatMulQ6KKernel<TILE, ROWS><<<grid, kThreads, sh, EngineStream()>>>(
        wdev, w.rows, w.row_bytes(), w.cols, X_dev, n_tokens, OUT_dev);
    CheckLaunch("MatMulQ6KKernel");
  }
}

// 把 host 侧挑出来的形状落到编译期实例上。形状只可能来自 PickShape（默认策略）
// 或 ForcedShape（扫参）—— 两者都在下面这张表里，所以 default 分支是"不可能发生"：
// 真发生了就**报错**，而不是静默换一个形状（那会让"我改的形状"和"实际跑的形状"
// 不一致，而这两者不一致时量到的收益归因不到任何东西上）。
void LaunchSpecializedByShape(const MatMulShape& s, uint32_t type, const dim3& grid,
                              const uint8_t* wdev, const GpuTensor& w,
                              const float* X_dev, int64_t n_tokens, float* OUT_dev) {
#define RPG_MM_CASE(T, R)                                                     \
  if (s.tile == (T) && s.rows == (R)) {                                       \
    LaunchSpecialized<(T), (R)>(type, grid, wdev, w, X_dev, n_tokens, OUT_dev); \
    return;                                                                    \
  }
  // 扫参用到的全部组合（见 Temp/p4-report.txt 的表格）；生产策略见 PickShape。
  RPG_MM_CASE(1, 1)
  RPG_MM_CASE(1, 2)
  RPG_MM_CASE(1, 4)
  RPG_MM_CASE(1, 8)
  RPG_MM_CASE(2, 4)
  RPG_MM_CASE(4, 1)
  RPG_MM_CASE(4, 2)
  RPG_MM_CASE(4, 4)
  RPG_MM_CASE(8, 1)
  RPG_MM_CASE(8, 2)
  RPG_MM_CASE(8, 4)
  RPG_MM_CASE(16, 1)
  RPG_MM_CASE(16, 2)
  RPG_MM_CASE(16, 4)
#undef RPG_MM_CASE
  RPG_CHECK(false, "unsupported matmul shape (tile=%d, rows=%d)", s.tile, s.rows);
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

// ⚠️ `pos0_dev` 是**设备指针**而不是值参：CUDA Graph 回放时每一个 kernel 参数
// 都是捕获那一刻的值，值参会让每一步都用同一个位置（RoPE 一直按同一个 pos 转，
// 症状是"第 1 个 token 对、从第 2 个起胡说"）。多一次全局读换掉那一整类错误。
__global__ void RopeKernel(float* x, int32_t n_heads, int32_t head_dim,
                           int32_t half, const float* rope_tab,
                           const int64_t* __restrict__ pos0_dev) {
  const int64_t pos0 = *pos0_dev;
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
                              int32_t n_kv_heads, int32_t head_dim,
                              const int64_t* __restrict__ pos0_dev,
                              int64_t max_seq, int64_t total,
                              __half* __restrict__ k_dst,
                              __half* __restrict__ v_dst) {
  const int64_t pos0 = *pos0_dev;  // 设备指针的理由见 RopeKernel
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
                                         float scale,
                                         const int64_t* __restrict__ pos0_dev,
                                         int64_t n_scores,
                                         float* __restrict__ out) {
  const int64_t pos0 = *pos0_dev;  // 设备指针的理由见 RopeKernel
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
  RPG_CHECK(g > 0 && g <= 0x7FFFFFFF, "grid dimension out of range: %lld", (long long)g);
  return static_cast<unsigned>(g);
}

}  // namespace

// ---------------------------------------------------------------- 启动入口

// 见 kernels_gpu.h：判据工具用它在"通用 kernel"与"P4 专用 kernel"之间切换。
// 默认 false ⇒ 生产路径永远走专用 kernel。
namespace {
bool g_force_generic_matmul = false;
}  // namespace

void GpuSetForceGenericMatMul(bool force) { g_force_generic_matmul = force; }
bool GpuForceGenericMatMul() { return g_force_generic_matmul; }

namespace {

// attention 的 scores 缓冲占多少动态共享内存。
//
// ⚠️ **两处必须用同一个公式**：一处是"实际启动时传多少"，一处是"要不要抬 48 KiB
// 上限"。抄成两份就会漂 —— 抬少了启动失败，抬多了白占 occupancy，而两种情况都
// 只在 n_scores 跨过某个阈值时才出现。GpuCausalAttention（整段，n = seq）与
// GpuAttentionFromCache（增量，n = pos0 + seq 或 max_seq）都用它。
size_t AttentionShmemBytes(int64_t n_scores) {
  const size_t fl =
      (static_cast<size_t>(n_scores) + 1u) & ~static_cast<size_t>(1u);
  return (fl + kThreads * 2u) * sizeof(float);
}

}  // namespace

void GpuDequantRow(uint32_t type, const uint8_t* src_dev, int64_t n_elems,
                   float* dst_dev) {
  const BlockShape bs = ShapeOf(type);
  RPG_CHECK(n_elems % bs.elems == 0, "dequant length %lld is not a multiple of the block size %d",
            static_cast<long long>(n_elems), bs.elems);
  DequantRowKernel<<<GridFor(n_elems, 256), 256, 0, EngineStream()>>>(
      type, src_dev, n_elems, bs.elems, bs.bytes, dst_dev);
  CheckLaunch("DequantRowKernel");
}

void GpuMatMulQuant(const GpuTensor& w, const float* X_dev, int64_t n_tokens,
                    float* OUT_dev) {
  RPG_CHECK(w.matrix, "GpuMatMulQuant got a non-matrix tensor (%s)", w.type_name());
  RPG_CHECK(n_tokens > 0, "n_tokens must be positive");
  const BlockShape bs = ShapeOf(w.type);
  RPG_CHECK(w.cols % bs.elems == 0, "dot length %lld is not a multiple of the block size %d",
            static_cast<long long>(w.cols), bs.elems);
  RPG_CHECK(w.buf.size() == static_cast<size_t>(w.rows) * w.row_bytes(),
            "tensor byte count does not match the [%lld,%lld] shape; this should have been caught at load time",
            static_cast<long long>(w.rows), static_cast<long long>(w.cols));
  const uint8_t* wdev = static_cast<const uint8_t*>(w.buf.data());

  // P4：权重直算的两种主力类型走"块常量外提"的专用 kernel；其余（F32 —— 实际
  // 权重里只有 norm 是 F32，不会走到这里）继续走通用 kernel。分派条件由
  // MatMulHasSpecializedKernel 把关：块形状与专用 kernel 的硬编码假设不符时退回。
  //
  // 形状按 n_tokens 挑（见 PickShape）：decode 走 (TILE=1, ROWS=8)，prefill 走
  // (TILE=8, ROWS=1) —— 前者避免 n=1 时 TILE 槽位空转，后者避免整行权重被重读
  // ceil(n/8) 遍。**算术一字未动**，变的只是"哪个 block 算哪些输出"。
  if (!g_force_generic_matmul && MatMulHasSpecializedKernel(w.type) &&
      w.cols / bs.elems <= kMaxQuantBlocks) {
    const MatMulShape s = PickShape(n_tokens);
    // 行方向按 ROWS 分组，token 方向按 TILE 分组。两者都只是"一个 block 干多少活"。
    const dim3 grid(static_cast<unsigned>((w.rows + s.rows - 1) / s.rows),
                    GridFor(n_tokens, s.tile));
    LaunchSpecializedByShape(s, w.type, grid, wdev, w, X_dev, n_tokens, OUT_dev);
    return;
  }
  const dim3 grid(static_cast<unsigned>(w.rows), GridFor(n_tokens, kTileTokens));
  // 走到这里只有三种情况：F32（实际权重里只有 norm）、块形状与专用 kernel 的
  // 硬编码假设不符、或一行量化块数超过专用 kernel 的共享内存预算。都是"退回通用
  // 版"而不是报错 —— 通用版与专用版**逐位等价**（tools/matmul_bench --verify）。
  MatMulQuantKernel<<<grid, kThreads, 0, EngineStream()>>>(
      w.type, wdev, w.rows, w.row_bytes(), w.cols, bs.elems, bs.bytes, X_dev,
      n_tokens, OUT_dev);
  CheckLaunch("MatMulQuantKernel");
}

void GpuRmsNorm(const float* x, const float* w, int64_t n_rows, int64_t n,
                int64_t stride, float eps, float* out) {
  RPG_CHECK(n > 0 && n_rows > 0 && stride >= n, "invalid RmsNorm arguments");
  RmsNormKernel<<<static_cast<unsigned>(n_rows), kThreads, 0, EngineStream()>>>(
      x, w, n, stride, eps, out);
  CheckLaunch("RmsNormKernel");
}

void GpuRope(float* x, int32_t n_heads, int32_t head_dim, const float* rope_tab,
             int64_t pos0, int64_t n_rows) {
  const int32_t half = head_dim / 2;
  RPG_CHECK(head_dim % 2 == 0, "head_dim %d must be even (RoPE pairs half-dimensions)", head_dim);
  const int64_t total = n_rows * n_heads * half;
  // 位置走暂存区：签名不变（tools/kv_map_check.cu 直接调这些算子），地址固定
  // 才能进 CUDA Graph。见 cuda_common.h 的 pos0 暂存区说明。
  const int64_t* pos_dev = StagePos0Cached(pos0);
  RopeKernel<<<GridFor(total, 256), 256, 0, EngineStream()>>>(
      x, n_heads, head_dim, half, rope_tab, pos_dev);
  CheckLaunch("RopeKernel");
}

void GpuCausalAttention(const ModelMeta& c, const float* q_dev,
                        const float* k_dev, const float* v_dev, int64_t seq,
                        float* out_dev) {
  RPG_CHECK(seq > 0, "attention seq must be positive");
  const float scale = 1.0f / sqrtf(static_cast<float>(c.head_dim));
  // 共享内存 = scores[seq] + 8 字节对齐的 red[kThreads]。
  const size_t sh_bytes = AttentionShmemBytes(seq);
  // 超过默认的 48 KiB 才需要显式申请。seq 到 4096 时是 17 KiB，用不着 ——
  // 但把这行留着，免得将来把 max_seq 调大之后收到一个"启动失败"的谜题。
  if (sh_bytes > 48 * 1024) {
    RPG_CUDA_CHECK(cudaFuncSetAttribute(
        CausalAttentionKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(sh_bytes)));
  }
  const int64_t blocks = seq * c.n_heads;
  CausalAttentionKernel<<<static_cast<unsigned>(blocks), kThreads, sh_bytes,
                          EngineStream()>>>(
      q_dev, k_dev, v_dev, seq, c.n_heads, c.n_kv_heads, c.head_dim, scale,
      out_dev);
  CheckLaunch("CausalAttentionKernel");
}

void GpuKvWrite(const float* k_dev, const float* v_dev, int32_t n_kv_heads,
                int32_t head_dim, int64_t pos0, int64_t seq,
                uint16_t* k_cache_layer, uint16_t* v_cache_layer,
                int64_t max_seq) {
  RPG_CHECK(seq > 0 && n_kv_heads > 0 && head_dim > 0, "invalid KvWrite arguments");
  RPG_CHECK(pos0 >= 0, "write position %lld must not be negative", static_cast<long long>(pos0));
  RPG_CHECK(pos0 + seq <= max_seq, "write range [%lld, %lld) exceeds the cache capacity %lld",
            static_cast<long long>(pos0), static_cast<long long>(pos0 + seq),
            static_cast<long long>(max_seq));
  const int64_t total = seq * n_kv_heads * head_dim;
  const int64_t* pos_dev = StagePos0Cached(pos0);
  KvWriteKernel<<<GridFor(total, 256), 256, 0, EngineStream()>>>(
      k_dev, v_dev, n_kv_heads, head_dim, pos_dev, max_seq, total,
      reinterpret_cast<__half*>(k_cache_layer),
      reinterpret_cast<__half*>(v_cache_layer));
  CheckLaunch("KvWriteKernel");
}

void GpuAttentionFromCache(const ModelMeta& c, int64_t pos0, const float* q_dev,
                           int64_t seq, const uint16_t* k_layer,
                           const uint16_t* v_layer, int64_t max_seq,
                           float* out_dev, int64_t sh_positions) {
  RPG_CHECK(seq > 0, "attention seq must be positive");
  RPG_CHECK(pos0 >= 0, "pos0 must not be negative");
  const float scale = 1.0f / sqrtf(static_cast<float>(c.head_dim));
  // scores 按**绝对位置数**备（见 kernel 里的说明）。
  //
  // ⚠️ 默认口径 `pos0 + seq` 从 P2 起就在用，逐字不变。但 CUDA Graph 下
  // **shared memory 大小是启动参数、会被录进图**，而回放时 pos0 每步递增 ——
  // 照捕获那一刻的 pos0 备，第 2 步起就越界写。所以图路径显式传
  // `sh_positions = kv->max_seq()`，一次备够"最大可能用到的位置数"。
  // 代价是 occupancy 下降，所以**只让图路径付这个代价**：prefill 保持原样
  //（prefill 要的正是"别拖慢"，P4 的目标里有 prefill ≥ 1000 tok/s）。
  const int64_t n_scores = sh_positions > 0 ? sh_positions : (pos0 + seq);
  RPG_CHECK(n_scores >= pos0 + seq,
            "shared-memory budget (" RPG_SZ ") is smaller than the positions this "
            "call needs (" RPG_SZ ")",
            RPG_SZ_ARG(n_scores), RPG_SZ_ARG(pos0 + seq));
  const size_t sh_bytes = AttentionShmemBytes(n_scores);
  // 默认上限 48 KiB：n_scores 到 6144（RPGBar 的上下文上限）时是 6144 + 256 =
  // 6400 个 float = 25.6 KiB，够用。留这段是为了将来把 max_seq 调更大时，
  // 收到的是一条明确的启动失败，而不是一个"kernel 莫名返回错误"的谜题。
  //
  // ⚠️ **捕获期间 cudaFuncSetAttribute 是非法 API**（它要动函数的属性，图里没有
  // 这种节点）。所以捕获中遇到"需要抬上限"就直接报错说清楚 —— 而不是让
  // cudaStreamEndCapture 返回一个看不出根因的 cudaErrorStreamCaptureUnsupported。
  // 图路径的正确做法是**在 BeginCapture 之前**调 GpuAttentionWarmupShmem
  //（CaptureDecodeStep 就是那么做的）。
  if (sh_bytes > 48 * 1024) {
    RPG_CHECK(!StreamIsCapturing(),
              "this attention call needs " RPG_SZ " bytes of dynamic shared memory, "
              "which requires cudaFuncSetAttribute -- illegal inside a CUDA Graph "
              "capture. Call GpuAttentionWarmupShmem(max_seq) before capturing, "
              "lower max_seq, or turn the graph path off.",
              RPG_SZ_ARG(sh_bytes));
    RPG_CUDA_CHECK(cudaFuncSetAttribute(
        AttentionFromCacheKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(sh_bytes)));
  }
  const int64_t blocks = seq * c.n_heads;
  const int64_t* pos_dev = StagePos0Cached(pos0);
  AttentionFromCacheKernel<<<static_cast<unsigned>(blocks), kThreads, sh_bytes,
                             EngineStream()>>>(
      q_dev, reinterpret_cast<const __half*>(k_layer),
      reinterpret_cast<const __half*>(v_layer), max_seq, c.n_heads,
      c.n_kv_heads, c.head_dim, scale, pos_dev, n_scores, out_dev);
  CheckLaunch("AttentionFromCacheKernel");
}

void GpuAttentionWarmupShmem(int64_t sh_positions) {
  RPG_CHECK(sh_positions > 0, "sh_positions must be positive (got %lld)",
            static_cast<long long>(sh_positions));
  const size_t sh_bytes = AttentionShmemBytes(sh_positions);
  // 48 KiB 以下本来就能启动，不必动函数属性 —— 少一次 API 调用，也少一个
  // "明明没超还去改全局上限"的副作用。
  if (sh_bytes <= 48 * 1024) return;
  // 这里**故意**用与 GpuAttentionFromCache 里同一个断言：万一哪天真被放在捕获
  // 窗口里调用，得到的是一条说得清的错误，而不是一个悬空的函数属性改动。
  RPG_CHECK(!StreamIsCapturing(),
            "GpuAttentionWarmupShmem must be called before cudaStreamBeginCapture");
  RPG_CUDA_CHECK(cudaFuncSetAttribute(
      AttentionFromCacheKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(sh_bytes)));
}

void GpuArgMax(const float* x_dev, int64_t n, int32_t* out_idx_dev) {
  RPG_CHECK(n > 0, "argmax length must be positive");
  RPG_CHECK(n <= 0x7FFFFFFF, "argmax index exceeds int32 (%lld)",
            static_cast<long long>(n));
  ArgMaxKernel<<<1, kArgMaxThreads, 0, EngineStream()>>>(x_dev, n, out_idx_dev);
  CheckLaunch("ArgMaxKernel");
}

void GpuSiluMulInPlace(float* gate_dev, const float* up_dev, int64_t n) {
  if (n <= 0) return;
  SiluMulKernel<<<GridFor(n, 256), 256, 0, EngineStream()>>>(gate_dev, up_dev, n);
  CheckLaunch("SiluMulKernel");
}

void GpuVecAddInPlace(float* a_dev, const float* b_dev, int64_t n) {
  if (n <= 0) return;
  VecAddKernel<<<GridFor(n, 256), 256, 0, EngineStream()>>>(a_dev, b_dev, n);
  CheckLaunch("VecAddKernel");
}

void GpuEmbeddingLookup(const GpuTensor& emb, const int32_t* toks_dev, int64_t seq,
                        float* x_dev) {
  const BlockShape bs = ShapeOf(emb.type);
  RPG_CHECK(emb.cols % bs.elems == 0, "embedding width %lld is not a multiple of the block size",
            static_cast<long long>(emb.cols));
  const int64_t n = seq * emb.cols;
  EmbeddingLookupKernel<<<GridFor(n, 256), 256, 0, EngineStream()>>>(
      emb.type, static_cast<const uint8_t*>(emb.buf.data()), emb.cols,
      emb.row_bytes(), bs.elems, bs.bytes, toks_dev, seq, x_dev);
  CheckLaunch("EmbeddingLookupKernel");
}

}  // namespace rpg

#endif  // RPG_HAS_CUDA
