#include "quant.h"

#include <cstring>

#include "common.h"
#include "dtypes.h"

// 本文件的所有位布局与 ggml（ggml-org/ggml，MIT License，src/ggml-quants.c）的
// dequantize_row_* 逐行对齐。注释里标了每处对应关系，方便后续核对。
//
// 注意 Q2_K 与 Q3_K 的一个反直觉差异：
//   * Q2_K 的 scales 是 16 个现成的字节（低 4 位=scale，高 4 位=min），
//     **直接读**，不做位重排；
//   * Q3_K 的 scales 是 12 字节的 6-bit 打包，必须先用 kmask 重排成 16 个
//     6-bit 值再减 32。
//   * Q4_K / Q5_K 走 get_scale_min_k4 的 6-bit 拆包。
// 这三种布局长得像但互不通用，混用不会报错，只会静默算出垃圾。

namespace rpg {
namespace {

constexpr int64_t QK4_0 = 32;
constexpr int64_t QK4_1 = 32;
constexpr int64_t QK5_0 = 32;
constexpr int64_t QK5_1 = 32;
constexpr int64_t QK8_0 = 32;
constexpr int64_t QK_K = 256;

const GgmlTypeInfo kTypes[] = {
    {kGgmlF32, "F32", 1, 4},
    {kGgmlF16, "F16", 1, 2},
    {kGgmlBF16, "BF16", 1, 2},
    {kGgmlQ4_0, "Q4_0", QK4_0, 2 + QK4_0 / 2},
    {kGgmlQ4_1, "Q4_1", QK4_1, 2 + 2 + QK4_1 / 2},
    {kGgmlQ5_0, "Q5_0", QK5_0, 2 + 4 + QK5_0 / 2},
    {kGgmlQ5_1, "Q5_1", QK5_1, 2 + 2 + 4 + QK5_1 / 2},
    {kGgmlQ8_0, "Q8_0", QK8_0, 2 + QK8_0},
    {kGgmlQ8_1, "Q8_1", QK8_0, 2 + 2 + QK8_0},
    {kGgmlQ2_K, "Q2_K", QK_K, QK_K / 16 + QK_K / 4 + 2 + 2},
    {kGgmlQ3_K, "Q3_K", QK_K, QK_K / 8 + QK_K / 4 + 12 + 2},
    {kGgmlQ4_K, "Q4_K", QK_K, 2 + 2 + 12 + QK_K / 2},
    {kGgmlQ5_K, "Q5_K", QK_K, 2 + 2 + 12 + QK_K / 8 + QK_K / 2},
    {kGgmlQ6_K, "Q6_K", QK_K, QK_K / 2 + QK_K / 4 + QK_K / 16 + 2},
    {kGgmlQ8_K, "Q8_K", QK_K, 4 + QK_K + 2 * (QK_K / 16)},
};

inline float HalfAt(const uint8_t* p) { return F16ToF32(LoadU16(p)); }

// ---------------------------------------------------------------- fp 类型

void DeqF32(const uint8_t* src, int64_t n, float* dst) {
  std::memcpy(dst, src, static_cast<size_t>(n) * 4);
}

void DeqF16(const uint8_t* src, int64_t n, float* dst) {
  for (int64_t i = 0; i < n; ++i) dst[i] = F16ToF32(LoadU16(src + i * 2));
}

void DeqBf16(const uint8_t* src, int64_t n, float* dst) {
  for (int64_t i = 0; i < n; ++i) dst[i] = Bf16ToF32(LoadU16(src + i * 2));
}

// ---------------------------------------------------------------- 简单量化
// ggml: dequantize_row_q4_0 / q4_1 / q5_0 / q5_1 / q8_0 / q8_1

void DeqQ4_0(const uint8_t* src, int64_t n, float* dst) {
  const int64_t nb = n / QK4_0;
  for (int64_t b = 0; b < nb; ++b) {
    const uint8_t* p = src + b * 18;
    const float d = HalfAt(p + 0);
    const uint8_t* qs = p + 2;
    for (int64_t j = 0; j < 16; ++j) {
      // 低半字节 → 元素 j，高半字节 → 元素 j+16（ggml 的交错布局）。
      dst[b * 32 + j] = static_cast<float>((qs[j] & 0x0F) - 8) * d;
      dst[b * 32 + j + 16] = static_cast<float>((qs[j] >> 4) - 8) * d;
    }
  }
}

void DeqQ4_1(const uint8_t* src, int64_t n, float* dst) {
  const int64_t nb = n / QK4_1;
  for (int64_t b = 0; b < nb; ++b) {
    const uint8_t* p = src + b * 20;
    const float d = HalfAt(p + 0);
    const float m = HalfAt(p + 2);
    const uint8_t* qs = p + 4;
    for (int64_t j = 0; j < 16; ++j) {
      dst[b * 32 + j] = static_cast<float>(qs[j] & 0x0F) * d + m;
      dst[b * 32 + j + 16] = static_cast<float>(qs[j] >> 4) * d + m;
    }
  }
}

void DeqQ5_0(const uint8_t* src, int64_t n, float* dst) {
  const int64_t nb = n / QK5_0;
  for (int64_t b = 0; b < nb; ++b) {
    const uint8_t* p = src + b * 22;
    const float d = HalfAt(p + 0);
    const uint8_t* qh = p + 2;  // 4 字节高位
    const uint8_t* qs = p + 6;  // 16 字节低位
    for (int64_t j = 0; j < 16; ++j) {
      const int h0 = (qh[j >> 3] >> (j & 7)) & 1;
      const int h1 = (qh[(j + 16) >> 3] >> ((j + 16) & 7)) & 1;
      // 先做整数减法再乘 d（与 ggml 一致），不要展开成 value*d - 16*d：
      // 那样会多一次浮点舍入，对拍时逐位对不上。
      dst[b * 32 + j] =
          static_cast<float>(((qs[j] & 0x0F) | (h0 << 4)) - 16) * d;
      dst[b * 32 + j + 16] =
          static_cast<float>(((qs[j] >> 4) | (h1 << 4)) - 16) * d;
    }
  }
}

void DeqQ5_1(const uint8_t* src, int64_t n, float* dst) {
  const int64_t nb = n / QK5_1;
  for (int64_t b = 0; b < nb; ++b) {
    const uint8_t* p = src + b * 24;
    const float d = HalfAt(p + 0);
    const float m = HalfAt(p + 2);
    const uint8_t* qh = p + 4;
    const uint8_t* qs = p + 8;
    for (int64_t j = 0; j < 16; ++j) {
      const int h0 = (qh[j >> 3] >> (j & 7)) & 1;
      const int h1 = (qh[(j + 16) >> 3] >> ((j + 16) & 7)) & 1;
      dst[b * 32 + j] = static_cast<float>((qs[j] & 0x0F) | (h0 << 4)) * d + m;
      dst[b * 32 + j + 16] =
          static_cast<float>((qs[j] >> 4) | (h1 << 4)) * d + m;
    }
  }
}

void DeqQ8_0(const uint8_t* src, int64_t n, float* dst) {
  const int64_t nb = n / QK8_0;
  for (int64_t b = 0; b < nb; ++b) {
    const uint8_t* p = src + b * 34;
    const float d = HalfAt(p + 0);
    const int8_t* qs = reinterpret_cast<const int8_t*>(p + 2);
    for (int64_t j = 0; j < QK8_0; ++j) {
      dst[b * 32 + j] = static_cast<float>(qs[j]) * d;
    }
  }
}

void DeqQ8_1(const uint8_t* src, int64_t n, float* dst) {
  const int64_t nb = n / QK8_0;
  for (int64_t b = 0; b < nb; ++b) {
    const uint8_t* p = src + b * 36;
    const float d = HalfAt(p + 0);
    const int8_t* qs = reinterpret_cast<const int8_t*>(p + 4);
    for (int64_t j = 0; j < QK8_0; ++j) {
      dst[b * 32 + j] = static_cast<float>(qs[j]) * d;
    }
  }
}

// ---------------------------------------------------------------- K-quant

// 6-bit scale/min 拆包（Q4_K / Q5_K 用）。对应 ggml 的 get_scale_min_k4。
inline void GetScaleMinK4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = static_cast<uint8_t>((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
    *m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
  }
}

void DeqQ2_K(const uint8_t* src, int64_t n, float* dst) {
  const int64_t nb = n / QK_K;
  for (int64_t b = 0; b < nb; ++b) {
    // block_q2_K: scales[16] | qs[64] | d(fp16) | dmin(fp16)
    const uint8_t* p = src + b * 84;
    const uint8_t* scales = p;       // 16 字节，直接读
    const uint8_t* q = p + 16;       // 64 字节
    const float d = HalfAt(p + 80);
    const float dmin = HalfAt(p + 82);

    float* y = dst + b * QK_K;
    int is = 0;
    for (int64_t blk = 0; blk < QK_K; blk += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        uint8_t sc = scales[is++];
        float dl = d * static_cast<float>(sc & 0x0F);
        float ml = dmin * static_cast<float>(sc >> 4);
        for (int l = 0; l < 16; ++l) {
          *y++ = dl * static_cast<float>((q[l] >> shift) & 3) - ml;
        }
        sc = scales[is++];
        dl = d * static_cast<float>(sc & 0x0F);
        ml = dmin * static_cast<float>(sc >> 4);
        for (int l = 0; l < 16; ++l) {
          *y++ = dl * static_cast<float>((q[l + 16] >> shift) & 3) - ml;
        }
        shift += 2;
      }
      q += 32;
    }
  }
}

void DeqQ3_K(const uint8_t* src, int64_t n, float* dst) {
  const int64_t nb = n / QK_K;
  const uint32_t kmask1 = 0x03030303u;
  const uint32_t kmask2 = 0x0F0F0F0Fu;

  for (int64_t b = 0; b < nb; ++b) {
    // block_q3_K: hmask[32] | qs[64] | scales[12] | d(fp16)
    const uint8_t* p = src + b * 110;
    const uint8_t* hm = p;          // 32
    const uint8_t* q = p + 32;      // 64
    const uint8_t* packed = p + 96; // 12
    const float d_all = HalfAt(p + 108);

    // 12 字节 → 16 个 6-bit scale（与 ggml 的 kmask 重排逐位一致）。
    uint32_t aux[4];
    std::memcpy(aux, packed, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);

    float* y = dst + b * QK_K;
    uint8_t m = 1;
    int is = 0;
    for (int64_t blk = 0; blk < QK_K; blk += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        float dl = d_all * static_cast<float>(scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          // 高位掩码置位 ⇒ 用 0..3；未置位 ⇒ 减 4 得 -4..-1。
          *y++ = dl * static_cast<float>(static_cast<int>((q[l] >> shift) & 3) -
                                         ((hm[l] & m) ? 0 : 4));
        }
        dl = d_all * static_cast<float>(scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          *y++ = dl *
                 static_cast<float>(static_cast<int>((q[l + 16] >> shift) & 3) -
                                    ((hm[l + 16] & m) ? 0 : 4));
        }
        shift += 2;
        m = static_cast<uint8_t>(m << 1);
      }
      q += 32;
    }
  }
}

void DeqQ4_K(const uint8_t* src, int64_t n, float* dst) {
  const int64_t nb = n / QK_K;
  for (int64_t b = 0; b < nb; ++b) {
    // block_q4_K: d(fp16) | dmin(fp16) | scales[12] | qs[128]
    const uint8_t* p = src + b * 144;
    const float d = HalfAt(p + 0);
    const float dmin = HalfAt(p + 2);
    const uint8_t* scales = p + 4;
    const uint8_t* q = p + 16;

    float* y = dst + b * QK_K;
    int is = 0;
    for (int64_t j = 0; j < QK_K; j += 64) {
      uint8_t sc = 0;
      uint8_t m = 0;
      GetScaleMinK4(is + 0, scales, &sc, &m);
      const float d1 = d * static_cast<float>(sc);
      const float m1 = dmin * static_cast<float>(m);
      GetScaleMinK4(is + 1, scales, &sc, &m);
      const float d2 = d * static_cast<float>(sc);
      const float m2 = dmin * static_cast<float>(m);
      for (int l = 0; l < 32; ++l) *y++ = d1 * static_cast<float>(q[l] & 0x0F) - m1;
      for (int l = 0; l < 32; ++l) *y++ = d2 * static_cast<float>(q[l] >> 4) - m2;
      q += 32;
      is += 2;
    }
  }
}

void DeqQ5_K(const uint8_t* src, int64_t n, float* dst) {
  const int64_t nb = n / QK_K;
  for (int64_t b = 0; b < nb; ++b) {
    // block_q5_K: d | dmin | scales[12] | qh[32] | qs[128]
    const uint8_t* p = src + b * 176;
    const float d = HalfAt(p + 0);
    const float dmin = HalfAt(p + 2);
    const uint8_t* scales = p + 4;
    const uint8_t* qh = p + 16;
    const uint8_t* ql = p + 48;

    float* y = dst + b * QK_K;
    int is = 0;
    uint8_t u1 = 1;
    uint8_t u2 = 2;
    for (int64_t j = 0; j < QK_K; j += 64) {
      uint8_t sc = 0;
      uint8_t m = 0;
      GetScaleMinK4(is + 0, scales, &sc, &m);
      const float d1 = d * static_cast<float>(sc);
      const float m1 = dmin * static_cast<float>(m);
      GetScaleMinK4(is + 1, scales, &sc, &m);
      const float d2 = d * static_cast<float>(sc);
      const float m2 = dmin * static_cast<float>(m);
      for (int l = 0; l < 32; ++l) {
        *y++ = d1 * static_cast<float>((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) -
               m1;
      }
      for (int l = 0; l < 32; ++l) {
        *y++ = d2 * static_cast<float>((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) -
               m2;
      }
      ql += 32;
      is += 2;
      u1 = static_cast<uint8_t>(u1 << 2);
      u2 = static_cast<uint8_t>(u2 << 2);
    }
  }
}

void DeqQ6_K(const uint8_t* src, int64_t n, float* dst) {
  const int64_t nb = n / QK_K;
  for (int64_t b = 0; b < nb; ++b) {
    // block_q6_K: ql[128] | qh[64] | scales[16](int8) | d(fp16)
    const uint8_t* p = src + b * 210;
    const uint8_t* ql = p;
    const uint8_t* qh = p + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(p + 192);
    const float d = HalfAt(p + 208);

    float* y = dst + b * QK_K;
    for (int64_t blk = 0; blk < QK_K; blk += 128) {
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        const int q1 = static_cast<int>((ql[l + 0] & 0x0F) |
                                        (((qh[l] >> 0) & 3) << 4)) - 32;
        const int q2 = static_cast<int>((ql[l + 32] & 0x0F) |
                                        (((qh[l] >> 2) & 3) << 4)) - 32;
        const int q3 = static_cast<int>((ql[l + 0] >> 4) |
                                        (((qh[l] >> 4) & 3) << 4)) - 32;
        const int q4 = static_cast<int>((ql[l + 32] >> 4) |
                                        (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l + 0] = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
        y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
        y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
        y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
      }
      y += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }
}

void DeqQ8_K(const uint8_t* src, int64_t n, float* dst) {
  const int64_t nb = n / QK_K;
  for (int64_t b = 0; b < nb; ++b) {
    // block_q8_K: d(float32) | qs[256](int8) | bsums[16](int16)
    const uint8_t* p = src + b * 292;
    const float d = LoadF32(p);
    const int8_t* qs = reinterpret_cast<const int8_t*>(p + 4);
    for (int64_t j = 0; j < QK_K; ++j) {
      dst[b * QK_K + j] = d * static_cast<float>(qs[j]);
    }
  }
}

}  // namespace

const GgmlTypeInfo* TypeInfo(uint32_t id) {
  for (const auto& t : kTypes) {
    if (t.id == id) return &t;
  }
  return nullptr;
}

const char* TypeName(uint32_t id) {
  const GgmlTypeInfo* t = TypeInfo(id);
  return t != nullptr ? t->name : "unknown";
}

bool TypeSupported(uint32_t id) {
  switch (id) {
    case kGgmlF32:
    case kGgmlF16:
    case kGgmlBF16:
    case kGgmlQ4_0:
    case kGgmlQ4_1:
    case kGgmlQ5_0:
    case kGgmlQ5_1:
    case kGgmlQ8_0:
    case kGgmlQ8_1:
    case kGgmlQ2_K:
    case kGgmlQ3_K:
    case kGgmlQ4_K:
    case kGgmlQ5_K:
    case kGgmlQ6_K:
    case kGgmlQ8_K:
      return true;
    default:
      return false;
  }
}

int64_t TypeRowBytes(uint32_t id, int64_t n_elems) {
  const GgmlTypeInfo* t = TypeInfo(id);
  RPG_CHECK(t != nullptr, "未知的 ggml 类型 id=%u", id);
  RPG_CHECK(t->block_elems > 0 && n_elems % t->block_elems == 0,
            "元素数 %lld 不是 %s 块大小 %lld 的整数倍",
            static_cast<long long>(n_elems), t->name,
            static_cast<long long>(t->block_elems));
  return (n_elems / t->block_elems) * t->block_bytes;
}

void DequantizeRow(uint32_t id, const uint8_t* src, int64_t n_elems, float* dst) {
  switch (id) {
    case kGgmlF32: DeqF32(src, n_elems, dst); return;
    case kGgmlF16: DeqF16(src, n_elems, dst); return;
    case kGgmlBF16: DeqBf16(src, n_elems, dst); return;
    case kGgmlQ4_0: DeqQ4_0(src, n_elems, dst); return;
    case kGgmlQ4_1: DeqQ4_1(src, n_elems, dst); return;
    case kGgmlQ5_0: DeqQ5_0(src, n_elems, dst); return;
    case kGgmlQ5_1: DeqQ5_1(src, n_elems, dst); return;
    case kGgmlQ8_0: DeqQ8_0(src, n_elems, dst); return;
    case kGgmlQ8_1: DeqQ8_1(src, n_elems, dst); return;
    case kGgmlQ2_K: DeqQ2_K(src, n_elems, dst); return;
    case kGgmlQ3_K: DeqQ3_K(src, n_elems, dst); return;
    case kGgmlQ4_K: DeqQ4_K(src, n_elems, dst); return;
    case kGgmlQ5_K: DeqQ5_K(src, n_elems, dst); return;
    case kGgmlQ6_K: DeqQ6_K(src, n_elems, dst); return;
    case kGgmlQ8_K: DeqQ8_K(src, n_elems, dst); return;
    default:
      RPG_CHECK(false, "不支持的 ggml 类型 id=%u（%s）", id, TypeName(id));
  }
}

// DotRowQuant 已迁到 src/matmul.cc（M5）：那里多了 CPU 特性分派与 AVX2 内核，
// 本文件只留"与 SIMD 无关"的反量化 —— 它是 M1 的对拍基准，必须保持纯标量。
// 参考实现现在叫 DotRowQuantRef，同在新文件里。

}  // namespace rpg
