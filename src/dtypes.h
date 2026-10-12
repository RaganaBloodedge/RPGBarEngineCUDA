// RPGBarEngine —— 元素类型与 f32 转换
//
// 与 TinyGemmaCpp 的 dtypes.h 同源（bf16 那部分逐行沿用），但有两处不同：
//   1. 加了 **f16**（GGUF 的量化块 scale 是 fp16；BF16 模型也走 f16 路径）。
//   2. 去掉 sfp —— 那是 gemma.cpp 私有格式，GGUF 里不存在。
//
// 精度约定：所有转换都用 round-to-nearest-even，和 PyTorch / ggml 一致。
// 对拍时要逐位一致，任何"差不多"的舍入都会在 26 层累加后放大。

#ifndef RPGBAR_DTYPES_H_
#define RPGBAR_DTYPES_H_

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace rpg {

// f16 -> f32。手写位操作，不依赖编译器 __fp16 / _Float16 支持（MSVC 上不可靠）。
inline float F16ToF32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t man = h & 0x3FFu;
  uint32_t out;
  if (exp == 0) {
    if (man == 0) {
      out = sign;  // ±0
    } else {
      // 次正规数：规格化后指数是 -14，尾数需要左移到隐含 1 出现。
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
  float v;
  std::memcpy(&v, &out, 4);
  return v;
}

// f32 -> f16，round-to-nearest-even。
inline uint16_t F32ToF16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  const uint32_t sign = (u >> 16) & 0x8000u;
  const int32_t exp = static_cast<int32_t>((u >> 23) & 0xFFu) - 127 + 15;
  const uint32_t man = u & 0x7FFFFFu;

  if (((u >> 23) & 0xFFu) == 0xFFu) {  // Inf / NaN
    return static_cast<uint16_t>(sign | 0x7C00u | (man ? 0x200u : 0));
  }
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);  // 溢出 -> Inf
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);  // 下溢 -> ±0
    // 次正规：隐含 1 + 右移 + 舍入。
    uint32_t m = (man | 0x800000u) >> (1 - exp);
    m = (m + 0x1000u) >> 13;
    return static_cast<uint16_t>(sign | m);
  }
  uint32_t rounded = man + 0x1000u;  // 加半 ULP
  uint32_t e = static_cast<uint32_t>(exp);
  if (rounded & 0x800000u) {  // 进位到指数
    rounded = 0;
    ++e;
    if (e >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
  }
  return static_cast<uint16_t>(sign | (e << 10) | (rounded >> 13));
}

// bf16 -> f32：左移 16 位，精确无误差。
inline float Bf16ToF32(uint16_t v) {
  return std::bit_cast<float>(static_cast<uint32_t>(v) << 16);
}

// f32 -> bf16：round-to-nearest-even。
inline uint16_t F32ToBf16(float f) {
  uint32_t u = std::bit_cast<uint32_t>(f);
  const uint32_t lsb = (u >> 16) & 1u;
  u += 0x7FFFu + lsb;
  return static_cast<uint16_t>(u >> 16);
}

inline uint16_t LoadU16(const uint8_t* p) {
  uint16_t v;
  std::memcpy(&v, p, 2);
  return v;
}

inline float LoadF32(const uint8_t* p) {
  float v;
  std::memcpy(&v, p, 4);
  return v;
}

// 由 GGUF 类型 id 判断"是不是逐元素 fp 类型"，用于走快路径。
inline bool IsPlainFloatType(uint32_t ggml_type) {
  return ggml_type == 0 /*F32*/ || ggml_type == 1 /*F16*/ ||
         ggml_type == 30 /*BF16*/;
}

}  // namespace rpg

#endif  // RPGBAR_DTYPES_H_
