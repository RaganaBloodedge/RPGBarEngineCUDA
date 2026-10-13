// RPGBarEngine —— ggml 量化类型表与反量化
//
// GGUF 里一个张量可以是 fp 类型（F32/F16/BF16），也可以是分块量化类型
// （Q4_0 … Q8_0，以及 K-quant 家族的 Q2_K … Q6_K）。反量化算法按 ggml
// （ggml-org/ggml，MIT）的实现重写为标量版本，与文件里的位布局逐位对齐。
//
// —— 为什么反量化只到 f32，不做"就地融合点积" ——
// 融合点积（dequant 到寄存器里顺手 FMA）是 M5 的性能优化，会按 CPU 的 SIMD
// 宽度写出好几套。M1 的对拍阶段必须有**一份和 SIMD 无关、和平台无关**的参考
// 实现，否则"结果不对"的时候无法判断是模型逻辑错了还是内核写错了。
// 所以这里先给纯标量 DequantizeRow。M5 的 DotRowQuant（含 SIMD 分派）落在
// matmul.h / matmul.cc，其标量参考实现复用本文件的 DequantizeRow。

#ifndef RPGBAR_QUANT_H_
#define RPGBAR_QUANT_H_

#include <cstdint>

namespace rpg {

// GGML_TYPE_* 的 id 与 GGUF 文件里的数字一一对应（见 gguf.md 的 enum）。
enum GgmlType : uint32_t {
  kGgmlF32 = 0,
  kGgmlF16 = 1,
  kGgmlQ4_0 = 2,
  kGgmlQ4_1 = 3,
  kGgmlQ5_0 = 6,
  kGgmlQ5_1 = 7,
  kGgmlQ8_0 = 8,
  kGgmlQ8_1 = 9,
  kGgmlQ2_K = 10,
  kGgmlQ3_K = 11,
  kGgmlQ4_K = 12,
  kGgmlQ5_K = 13,
  kGgmlQ6_K = 14,
  kGgmlQ8_K = 15,
  kGgmlBF16 = 30,
};

struct GgmlTypeInfo {
  uint32_t id;
  const char* name;
  int64_t block_elems;  // 一个块包含多少元素
  int64_t block_bytes;  // 一个块占多少字节
};

// 返回类型信息；不认识的类型返回 nullptr。
const GgmlTypeInfo* TypeInfo(uint32_t id);

// 类型名（不认识时返回 "unknown"）。
const char* TypeName(uint32_t id);

// 该类型在 GGUF 文件中是否受支持。
bool TypeSupported(uint32_t id);

// n_elems 个该类型元素占多少字节。n_elems 必须是 block_elems 的整数倍。
int64_t TypeRowBytes(uint32_t id, int64_t n_elems);

// 所有受支持类型里最大的"块元素数"（QK_K = 256）。标量参考实现要在栈上解
// 一个块，用这个常量开空间；SIMD 内核按固定块长展开，也拿它对上限断言。
constexpr int64_t kMaxBlockElems = 256;

// 把 src 里的 n_elems 个元素反量化为 f32 写进 dst。
// 前置：TypeSupported(id)、n_elems % block_elems == 0。
void DequantizeRow(uint32_t id, const uint8_t* src, int64_t n_elems, float* dst);

// 注意：Q4_K / Q5_K 的 6-bit 拆包（ggml get_scale_min_k4 同款）**不对外导出**。
// 它只在本文件内部（DeqQ4_K / DeqQ5_K）和 SIMD 内核里就地 inline 使用。
// 曾经这里导出过一个 GetScaleMinK4Bytes 供 matmul_avx2.cc 复用，实测那是热循环里
// 一次**跨编译单元的真实函数调用**，把内核压到峰值的 7%（详见 DESIGN §19.7）。
// 别再往外开一个这样的门 —— 热循环里没有"调用次数少所以无所谓"的函数。

}  // namespace rpg

#endif  // RPGBAR_QUANT_H_
