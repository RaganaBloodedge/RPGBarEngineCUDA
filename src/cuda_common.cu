// cuda_common.cu —— CUDA 侧的实现（错误检查、显存 RAII、设备查询）。
//
// ⚠️ 这个文件**只有装了 CUDA Toolkit 才参与编译**。所以这里可以放心直接用
// cuda_runtime 的类型与 API，不需要到处 #ifdef —— 与 CUDA 无关的部分（描述
// 文本、host-only 兜底）在 cuda_common.cc 里，两条构建路径都编那一份。

#include "cuda_common.h"

// 完整头（相对 cuda_common.h 里的 cuda_runtime_api.h）：这个 TU 由 nvcc 编译，
// 且要调用设备查询/内存搬运的 runtime API，用完整头最省心。
#include <cuda_runtime.h>

#include <string>

#include "common.h"   // rpg::Throw / RPG_LOG_WARN / RPG_SZ

namespace rpg {

namespace {

// CUDA 的版本号是 major*1000 + minor*10（如 12080 → 12.8）。
std::string CudaVersionString(int v) {
  if (v <= 0) return "";
  return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
}

}  // namespace

void ThrowCudaError(cudaError_t err, const char* expr, const char* file, int line) {
  Throw("CUDA 调用失败：%s\n  表达式：%s\n  位置：%s:%d",
        cudaGetErrorString(err), expr, file, line);
}

// ---------------------------------------------------------------- DeviceBuffer

DeviceBuffer::DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), bytes_(o.bytes_) {
  o.ptr_ = nullptr;
  o.bytes_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& o) noexcept {
  if (this != &o) {
    Free();
    ptr_ = o.ptr_;
    bytes_ = o.bytes_;
    o.ptr_ = nullptr;
    o.bytes_ = 0;
  }
  return *this;
}

void DeviceBuffer::Alloc(size_t bytes) {
  // 0 字节的 cudaMalloc 返回的指针不可靠，也说明调用方算错了尺寸 —— 早失败。
  RPG_CHECK(bytes > 0, "DeviceBuffer::Alloc 收到 0 字节");
  Free();
  RPG_CUDA_CHECK(cudaMalloc(&ptr_, bytes));
  bytes_ = bytes;
}

void DeviceBuffer::Free() {
  if (ptr_ == nullptr) return;
  // 析构路径不能抛（DeviceBuffer 常作为成员，析构时抛异常会让整个对象销毁变成
  // UB）—— cudaFree 失败只记日志。实践中它只在上下文已销毁时失败。
  const cudaError_t e = cudaFree(ptr_);
  if (e != cudaSuccess) {
    RPG_LOG_WARN("cudaFree 失败（上下文可能已销毁）：%s", cudaGetErrorString(e));
  }
  ptr_ = nullptr;
  bytes_ = 0;
}

void DeviceBuffer::Upload(const void* src, size_t bytes) {
  RPG_CHECK(ptr_ != nullptr, "DeviceBuffer::Upload 在未分配时被调用");
  RPG_CHECK(bytes <= bytes_,
            "DeviceBuffer::Upload 溢出：上传 " RPG_SZ " 字节到 " RPG_SZ " 字节的缓冲",
            RPG_SZ_ARG(bytes), RPG_SZ_ARG(bytes_));
  RPG_CUDA_CHECK(cudaMemcpy(ptr_, src, bytes, cudaMemcpyHostToDevice));
}

void DeviceBuffer::Download(void* dst, size_t bytes) const {
  RPG_CHECK(ptr_ != nullptr, "DeviceBuffer::Download 在未分配时被调用");
  RPG_CHECK(bytes <= bytes_,
            "DeviceBuffer::Download 溢出：下载 " RPG_SZ " 字节自 " RPG_SZ " 字节的缓冲",
            RPG_SZ_ARG(bytes), RPG_SZ_ARG(bytes_));
  RPG_CUDA_CHECK(cudaMemcpy(dst, ptr_, bytes, cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------- 设备查询

DeviceInfo QueryDevice(int index) {
  DeviceInfo d;

  int count = 0;
  const cudaError_t e = cudaGetDeviceCount(&count);
  if (e != cudaSuccess) {
    d.reason = std::string("cudaGetDeviceCount 失败：") + cudaGetErrorString(e);
    return d;
  }
  if (count <= 0) {
    d.reason = "没有检测到 CUDA 设备（显卡不支持 / 驱动太旧 / 是纯 CPU 机器）";
    return d;
  }
  if (index < 0 || index >= count) {
    d.reason = "设备下标越界（共 " + std::to_string(count) + " 个设备）";
    return d;
  }

  cudaDeviceProp p{};
  const cudaError_t pe = cudaGetDeviceProperties(&p, index);
  if (pe != cudaSuccess) {
    d.reason = std::string("cudaGetDeviceProperties 失败：") + cudaGetErrorString(pe);
    return d;
  }

  d.available = true;
  d.name = p.name;
  d.cc_major = p.major;
  d.cc_minor = p.minor;
  d.sm_count = p.multiProcessorCount;
  d.total_mem = p.totalGlobalMem;

  // 剩余显存要用 cudaMemGetInfo；它需要先有活动上下文，拿不到也不影响 available。
  size_t free_b = 0, total_b = 0;
  if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
    d.free_mem = free_b;
    if (total_b > 0) d.total_mem = total_b;
  }

  int drv = 0, rt = 0;
  if (cudaDriverGetVersion(&drv) == cudaSuccess) d.driver = CudaVersionString(drv);
  if (cudaRuntimeGetVersion(&rt) == cudaSuccess) d.runtime = CudaVersionString(rt);
  return d;
}

}  // namespace rpg
