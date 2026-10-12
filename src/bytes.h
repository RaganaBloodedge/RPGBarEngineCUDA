// RPGBarEngine —— 小端读取
//
// GGUF 规范：模型默认小端（v3 支持大端，但文件里没有标记字段，规范自己都承认
// "无法判断是否大端"）。所以直接假设小端 —— 本项目只在 x86-64 / arm64 上跑，
// 两者都是小端。
//
// 这里全部用 memcpy 而不是 reinterpret_cast：后者在 -O3 下会踩到严格别名规则，
// 而且未对齐指针上做解引用在 ARM 上会直接崩。memcpy 会被编译器优化成一条 mov。

#ifndef RPGBAR_BYTES_H_
#define RPGBAR_BYTES_H_

#include <cstdint>
#include <cstring>
#include <string>

#include "common.h"

namespace rpg {

// 一个只前进的字节游标。GGUF 是纯顺序格式，用游标读最自然。
class Reader {
 public:
  Reader(const uint8_t* data, size_t size) : data_(data), size_(size) {}
  explicit Reader(const std::string& s)
      : data_(reinterpret_cast<const uint8_t*>(s.data())), size_(s.size()) {}

  size_t pos() const { return pos_; }
  size_t remaining() const { return size_ - pos_; }
  const uint8_t* cursor() const { return data_ + pos_; }
  bool eof() const { return pos_ >= size_; }

  void Need(size_t n) const {
    RPG_CHECK(pos_ + n <= size_, "读取越界：pos=" RPG_SZ " 需要 " RPG_SZ
                                 " 字节，共 " RPG_SZ,
              RPG_SZ_ARG(pos_), RPG_SZ_ARG(n), RPG_SZ_ARG(size_));
  }

  void Skip(size_t n) {
    Need(n);
    pos_ += n;
  }

  uint8_t U8() {
    Need(1);
    return data_[pos_++];
  }
  int8_t I8() { return static_cast<int8_t>(U8()); }

  uint16_t U16() {
    Need(2);
    uint16_t v;
    std::memcpy(&v, data_ + pos_, 2);
    pos_ += 2;
    return v;
  }
  int16_t I16() { return static_cast<int16_t>(U16()); }

  uint32_t U32() {
    Need(4);
    uint32_t v;
    std::memcpy(&v, data_ + pos_, 4);
    pos_ += 4;
    return v;
  }
  int32_t I32() { return static_cast<int32_t>(U32()); }

  uint64_t U64() {
    Need(8);
    uint64_t v;
    std::memcpy(&v, data_ + pos_, 8);
    pos_ += 8;
    return v;
  }
  int64_t I64() { return static_cast<int64_t>(U64()); }

  float F32() {
    uint32_t u = U32();
    float v;
    std::memcpy(&v, &u, 4);
    return v;
  }
  double F64() {
    uint64_t u = U64();
    double v;
    std::memcpy(&v, &u, 8);
    return v;
  }

  bool Bool() {
    uint8_t v = U8();
    RPG_CHECK(v <= 1, "GGUF bool 只能是 0/1，读到 %u", v);
    return v != 0;
  }

  // GGUF 字符串：u64 长度 + 非 NUL 结尾的 UTF-8 字节。
  std::string String() {
    uint64_t n = U64();
    Need(n);
    std::string s(reinterpret_cast<const char*>(data_ + pos_), n);
    pos_ += n;
    return s;
  }

  // 裸字节块（张量数据用得上）。
  const uint8_t* Bytes(size_t n) {
    Need(n);
    const uint8_t* p = data_ + pos_;
    pos_ += n;
    return p;
  }

 private:
  const uint8_t* data_;
  size_t size_;
  size_t pos_ = 0;
};

// 按对齐向上取整（GGUF 通用：align_offset）。
inline uint64_t AlignUp(uint64_t offset, uint64_t align) {
  if (align == 0) return offset;
  return offset + (align - (offset % align)) % align;
}

}  // namespace rpg

#endif  // RPGBAR_BYTES_H_
