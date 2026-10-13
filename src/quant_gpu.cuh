// RPGBarEngineCUDA —— 设备端反量化（F32 / Q4_K / Q6_K）
//
// ⚠️ 这个头**只许被 .cu include**（里面有 __device__ 函数）。host 侧（.cc）要用
// 反量化请用 src/quant.cc 的 DequantizeRow —— 那是同源复制来的标量参考实现。
//
// ## 与 quant.cc 的关系：公式是照抄，拆法不同
//
// 每个元素的**数值**与 src/quant.cc 的 DeqF32 / DeqQ4_K / DeqQ6_K 逐位相同
// （判据见 tools/dequant_check：要求 0 个 bit 不同）。区别只在遍历方式：
//
//   quant.cc  —— 「按块解整段」：一次解 256 个元素到 float y[256]。
//  这里       —— 「按元素解一个」：DeqElem(blk, e) 直接返回第 e 个元素。
//
// 为什么非要按元素解：融合点积不能先解整块。一个线程解 256 个 float 的本地
// 数组会被赶到 local memory（显存里），一读一写就废掉了融合的全部意义。
// 按元素解之后，线程只碰自己那几个字节，值直接进寄存器参与 FMA。
//
// ⚠️ 但按元素解就必须自己保证"同一个 scale/min 被两个位置用到时算出来一样"：
// quant.cc 里 `d1 = d * sc` 是**算一次用 32 次**，这里是每个元素重算一遍。
// 单次乘法是确定性的，重算得到同一组 bit —— 前提是两边都别做 FMA 收缩。
// 所以下面凡是有「先乘后减」的地方都用 __fmul_rn / __fsub_rn 显式钉死；
// 对应的 host 侧用 -ffp-contract=off 编（见 CMakeLists.txt 里那行注释）。
// 这一条不写死的话，x86 上不带 -mfma 编出来的参考实现与 GPU 会差 1 ULP，
// 而"逐位对拍"的判据会红一片 —— 且看起来像是拆包写错了。

#ifndef RPGBAR_QUANT_GPU_CUH_
#define RPGBAR_QUANT_GPU_CUH_

#include <cstdint>

// ⚠️ include 顺序约定：CUDA 头与项目头都必须在 namespace 之前。
// 放进 namespace rpg 里的后果见 cuda_common.h 里那段长注释（真踩过）。
#include <cuda_runtime_api.h>

#include "quant.h"  // kGgmlF32 / kGgmlQ4_K / kGgmlQ6_K / kMaxBlockElems

namespace rpg {
namespace gpu {

// ---------------------------------------------------------------- f16 -> f32
//
// 与 host 的 rpg::F16ToF32（src/dtypes.h，同源复制、不许改）**逐位相同**：
// 同样的位操作、同样的次正规数处理。fp16 → fp32 是无损转换，任何正确实现都
// 该给出同样的 bit；这里不用 __half2float 只是为了不引入 cuda_fp16.h 这个
// 只在 .cu 里可用的头（量化块的 scale 只有这一处需要 f16）。
//
// 正确性由 tools/dequant_check 的 --f16-sweep 穷举 65536 个 half 值对着
// host 版比一遍（0 差异才算过）—— 与其相信"这段位操作我写对了"，不如让它
// 自己证明。
__device__ __forceinline__ float F16ToF32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t man = h & 0x3FFu;
  uint32_t out;
  if (exp == 0) {
    if (man == 0) {
      out = sign;  // ±0
    } else {
      // 次正规数：规格化后指数是 -14，尾数左移到隐含 1 出现。
      exp = 127 - 15 + 1;
      while ((man & 0x400u) == 0) {
        man <<= 1;
        --exp;
      }
      man &= 0x3FFu;
      out = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 31) {
    out = sign | 0x7F800000u | (man << 13);  // Inf / NaN
  } else {
    out = sign | ((exp + (127 - 15)) << 23) | (man << 13);
  }
  return __uint_as_float(out);
}

// 小端 f32 读取。与 host 的 LoadF32（src/dtypes.h）同语义。
__device__ __forceinline__ float LoadF32(const uint8_t* p) {
  uint32_t u = static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
               (static_cast<uint32_t>(p[2]) << 16) |
               (static_cast<uint32_t>(p[3]) << 24);
  return __uint_as_float(u);
}

// ---------------------------------------------------------------- Q4_K
//
// 块布局（144 字节，256 个元素）：
//   d(f16) | dmin(f16) | scales[12] | qs[128]
//
// 与 quant.cc 的 GetScaleMinK4 对应：12 字节里塞了 8 组 6-bit 的 (scale, min)。
// j < 4 时是"低 6 位直接读"，j >= 4 时高 2 位藏在另外几个字节里 —— 这段
// 位拼接照抄 ggml，看着像可以简化，实际每一个 & 和 >> 都不能动。
__device__ __forceinline__ void GetScaleMinK4(int j, const uint8_t* q, int* sc,
                                              int* m) {
  if (j < 4) {
    *sc = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

// Q4_K 第 e 个元素（e ∈ [0,256)）的 f32 值。
__device__ __forceinline__ float DeqElemQ4K(const uint8_t* blk, int e) {
  // 256 个元素 = 4 组（每组 64）× 2 个子块（每组 32）。
  //   sub   = e / 32        → 0..7，第 sub 个子块
  //   l     = e % 32        → 子块内偏移
  //   group = sub / 2       → 0..3，qs 的组号
  const int sub = e >> 5;
  const int l = e & 31;
  const int group = sub >> 1;

  const uint8_t* qs = blk + 16 + group * 32;
  // 偶子块取低半字节，奇子块取高半字节（ggml 的交错布局）。
  const int raw = (sub & 1) ? (qs[l] >> 4) : (qs[l] & 0x0F);

  int sc = 0, m = 0;
  GetScaleMinK4(2 * group + (sub & 1), blk + 4, &sc, &m);

  const float d = F16ToF32(static_cast<uint16_t>(blk[0] | (blk[1] << 8)));
  const float dmin = F16ToF32(static_cast<uint16_t>(blk[2] | (blk[3] << 8)));

  // **不要把这四行合成一行表达式**：合起来编译器会缩成 FMA，而 host 侧那份
  // 参考实现（-ffp-contract=off）不会 —— 差 1 ULP，逐位判据当场红。
  const float d1 = __fmul_rn(d, static_cast<float>(sc));
  const float m1 = __fmul_rn(dmin, static_cast<float>(m));
  return __fsub_rn(__fmul_rn(d1, static_cast<float>(raw)), m1);
}

// ---------------------------------------------------------------- Q6_K
//
// 块布局（210 字节，256 个元素）：
//   ql[128] | qh[64] | scales[16](int8) | d(f16)
//
// 与 Q4_K 反过来了：scales 是 16 个现成的 int8（**直接读**，不做位重排），
// 但 qs 分成了 ql/qh 两段，6 位里 4 位在 ql、2 位在 qh。
// 这一版的 scale 是 (d * sc) 后**只乘不加**，所以不存在 FMA 收缩问题。
__device__ __forceinline__ float DeqElemQ6K(const uint8_t* blk, int e) {
  // 256 个元素 = 2 半（每半 128）× 4 段（每段 32）。
  const int half = e >> 7;
  const int r = e & 127;
  const int quarter = r >> 5;  // 0..3
  const int l = r & 31;

  const uint8_t* ql = blk + half * 64;
  const uint8_t* qh = blk + 128 + half * 32;
  const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192 + half * 8);
  const int is = l >> 4;  // 0..1

  // 4 位在 ql：第 0/1 段取低半字节，第 2/3 段取高半字节；第 1/3 段的 l 要 +32。
  const uint8_t byte = ql[l + ((quarter & 1) ? 32 : 0)];
  const int nib = (quarter < 2) ? (byte & 0x0F) : (byte >> 4);
  // 高 2 位在 qh，按段号左移 0/2/4/6。
  const int hi = (qh[l] >> (2 * quarter)) & 3;
  const int raw = (nib | (hi << 4)) - 32;  // → [-32, 31]

  const float d = F16ToF32(static_cast<uint16_t>(blk[208] | (blk[209] << 8)));
  const float scale = static_cast<float>(sc[is + 2 * quarter]);
  // host 写法：d * (float)sc[..] * (float)q1 —— 左结合，先 d*sc 再乘 raw。
  return __fmul_rn(__fmul_rn(d, scale), static_cast<float>(raw));
}

// ---------------------------------------------------------------- 统一入口
//
// 解一个元素。type 必须是 GpuModel 认可的那三种（别的在加载期就拒了，
// 见 gpu_model.cu 的 CheckGpuType）—— 所以这里是"不可能到达"的兜底，
// 返回 0 而不是抛异常：这个函数在热循环里，且设备端抛异常要额外开 RTTI。
__device__ __forceinline__ float DeqElem(uint32_t type, const uint8_t* blk, int e) {
  switch (type) {
    case kGgmlF32:
      return LoadF32(blk + e * 4);
    case kGgmlQ4_K:
      return DeqElemQ4K(blk, e);
    case kGgmlQ6_K:
      return DeqElemQ6K(blk, e);
    default:
      return 0.0f;
  }
}

}  // namespace gpu
}  // namespace rpg

#endif  // RPGBAR_QUANT_GPU_CUH_
