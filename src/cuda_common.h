// cuda_common.h —— CUDA 侧公共设施：错误检查、设备信息、显存 RAII。
//
// 这一层刻意做得薄：**所有 CUDA 调用都过 RPG_CUDA_CHECK**，失败就抛 rpg::Error，
// 由 ABI 边界（engine.cc）统一翻成"负数/NULL + rpg_engine_last_error()"。
// CUDA 的错误码不穿出 ABI —— 与 host 侧"异常绝不穿 extern "C""是同一条约定。
//
// 为什么 DeviceBuffer 要自己做 RAII 而不是用 thrust / cuda::memory_resource：
// 这个库要静态链接进一个"发给别人玩"的 dll（见 RPGBarEngine 的构建约定），
// 少一个依赖就少一次"在某些机器上加载不了"的机会。cudaMalloc/cudaFree 够用。
#ifndef RPGBAR_CUDA_COMMON_H_
#define RPGBAR_CUDA_COMMON_H_

#include <cstddef>
#include <cstdint>
#include <string>

namespace rpg {

// ---------------------------------------------------------------- 设备信息

struct DeviceInfo {
  bool available = false;
  std::string name;
  int cc_major = 0;
  int cc_minor = 0;
  int sm_count = 0;
  size_t total_mem = 0;
  size_t free_mem = 0;
  std::string driver;    // 驱动支持的 CUDA 版本（如 "12.8"）
  std::string runtime;   // 工具链的 CUDA 版本
  std::string reason;    // available == false 时说明原因（永远不留空）
};

// 查询设备。**不抛异常** —— 它的一条主要用途就是"在没有 GPU 的机器上告诉人为什么"。
DeviceInfo QueryDevice(int index = 0);

// 一行摘要（给人看）与 JSON（给 rpg_engine_model_info 用）。
std::string DescribeDevice(const DeviceInfo& d);
std::string DeviceJson(const DeviceInfo& d);

#ifdef RPG_HAS_CUDA

#include <cuda_runtime.h>

[[noreturn]] void ThrowCudaError(cudaError_t err, const char* expr,
                                 const char* file, int line);

// 展开成 do/while 而不是裸 if：`if (x) RPG_CUDA_CHECK(f()); else g();` 这种
// 写法在裸宏下会把 else 挂到宏里的 if 上。
#define RPG_CUDA_CHECK(expr)                                       \
  do {                                                             \
    cudaError_t _rpg_e = (expr);                                   \
    if (_rpg_e != cudaSuccess) {                                   \
      ::rpg::ThrowCudaError(_rpg_e, #expr, __FILE__, __LINE__);    \
    }                                                              \
  } while (0)

// 显存 RAII。注意：**没有 CUDA 时这个类不存在** —— 与其留一个"能编译但一调用就
// 报未实现"的空壳，不如让引用它的代码在编译期就暴露出来。
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t bytes) { Alloc(bytes); }
  ~DeviceBuffer() { Free(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept;

  void Alloc(size_t bytes);
  void Free();
  void Upload(const void* src, size_t bytes);
  void Download(void* dst, size_t bytes) const;

  void* data() { return ptr_; }
  const void* data() const { return ptr_; }
  size_t size() const { return bytes_; }
  bool empty() const { return ptr_ == nullptr; }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
};

#endif  // RPG_HAS_CUDA

}  // namespace rpg

#endif  // RPGBAR_CUDA_COMMON_H_
