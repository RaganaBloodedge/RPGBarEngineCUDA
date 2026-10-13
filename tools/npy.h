// 最小的 .npy 写出器（只写 f32，只给 tools/ 用）。
//
// 为什么不直接写裸二进制 + shape 注释：对拍脚本是 Python，用 np.load 读数组
// 比读裸数据 + 手工 reshape 少一类"我算错了偏移"的错误。npy 的头就是一段
// ASCII 字典，自己拼比引依赖便宜。
//
// 头对齐规则：numpy 让 (前缀 10 字节 + 头) 是 64 的整数倍，最后补一个换行。
// 不对齐 np.load 也能读，但 memmap 会退化 —— 顺手对齐。

#ifndef RPGBAR_TOOLS_NPY_H_
#define RPGBAR_TOOLS_NPY_H_

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace npy {

// dims 按 C 顺序（最外层在前）。data 需有 prod(dims) 个 float。
inline bool WriteF32(const std::string& path, const std::vector<int64_t>& dims,
                     const float* data, size_t n_floats) {
  std::string shape = "(";
  for (size_t i = 0; i < dims.size(); ++i) {
    shape += std::to_string(dims[i]);
    if (i + 1 < dims.size() || dims.size() == 1) shape += ",";
    if (i + 1 < dims.size()) shape += " ";
  }
  shape += ")";

  std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': " +
                     shape + ", }";

  // 10 = 6(magic) + 2(version) + 2(header_len)
  size_t total = 10 + dict.size() + 1;
  size_t pad = (64 - (total % 64)) % 64;
  dict.append(pad, ' ');
  dict.push_back('\n');

  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  const char magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
  const uint8_t ver[2] = {1, 0};
  const uint16_t hlen = static_cast<uint16_t>(dict.size());
  bool ok = std::fwrite(magic, 1, 6, f) == 6 &&
            std::fwrite(ver, 1, 2, f) == 2 &&
            std::fwrite(&hlen, 1, 2, f) == 2 &&
            std::fwrite(dict.data(), 1, dict.size(), f) == dict.size();
  if (ok && n_floats > 0) {
    ok = std::fwrite(data, sizeof(float), n_floats, f) == n_floats;
  }
  std::fclose(f);
  return ok;
}

}  // namespace npy

#endif  // RPGBAR_TOOLS_NPY_H_
