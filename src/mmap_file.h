// RPGBarEngine —— 只读内存映射文件
//
// 为什么必须 mmap，而不是 fread 到 vector：
//   * Q8_0 的 1.7B 权重 1.8 GB，Q4_K_M 也 1.1 GB。整份读进堆会让进程峰值翻倍
//     （页缓存 + 我们的副本），在玩家机器上直接触发换页。
//   * mmap 之后张量数据就是"文件页"，未被触碰的部分不会占物理内存；宿主
//     连续两次加载同一个模型时页缓存直接命中。
//   * 顺带白拿"按需分页"：只跑一个 session 时，只有真正用到的层才会被读进来。
//
// 代价：文件被映射期间不能删（Windows 上尤其严格）。引擎的用法是模型常驻，
// 所以这不是问题；但 rpg_engine_unload_model 必须真的把映射关掉。

#ifndef RPGBAR_MMAP_FILE_H_
#define RPGBAR_MMAP_FILE_H_

#include <cstddef>
#include <cstdint>
#include <string>

namespace rpg {

class MappedFile {
 public:
  MappedFile() = default;
  ~MappedFile();

  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  // 打开并映射。失败抛 Error（路径不存在 / 权限 / 空文件）。
  static MappedFile Open(const std::string& path);

  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }
  const std::string& path() const { return path_; }
  bool valid() const { return data_ != nullptr; }

  void Close();

 private:
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
  std::string path_;

#if defined(_WIN32)
  void* file_ = nullptr;     // HANDLE
  void* mapping_ = nullptr;  // HANDLE
#endif
};

}  // namespace rpg

#endif  // RPGBAR_MMAP_FILE_H_
