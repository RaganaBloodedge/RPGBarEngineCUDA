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
  Throw("CUDA call failed: %s\n  expression: %s\n  at: %s:%d",
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
  RPG_CHECK(bytes > 0, "DeviceBuffer::Alloc got 0 bytes");
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
    RPG_LOG_WARN("cudaFree failed (the context may already be destroyed): %s", cudaGetErrorString(e));
  }
  ptr_ = nullptr;
  bytes_ = 0;
}

void DeviceBuffer::Upload(const void* src, size_t bytes) {
  RPG_CHECK(ptr_ != nullptr, "DeviceBuffer::Upload called on an unallocated buffer");
  RPG_CHECK(bytes <= bytes_,
            "DeviceBuffer::Upload overflow: uploading " RPG_SZ " bytes into a " RPG_SZ "-byte buffer",
            RPG_SZ_ARG(bytes), RPG_SZ_ARG(bytes_));
  RPG_CUDA_CHECK(cudaMemcpy(ptr_, src, bytes, cudaMemcpyHostToDevice));
}

void DeviceBuffer::Download(void* dst, size_t bytes) const {
  RPG_CHECK(ptr_ != nullptr, "DeviceBuffer::Download called on an unallocated buffer");
  RPG_CHECK(bytes <= bytes_,
            "DeviceBuffer::Download overflow: downloading " RPG_SZ " bytes from a " RPG_SZ "-byte buffer",
            RPG_SZ_ARG(bytes), RPG_SZ_ARG(bytes_));
  RPG_CUDA_CHECK(cudaMemcpy(dst, ptr_, bytes, cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------- 设备查询

DeviceInfo QueryDevice(int index) {
  DeviceInfo d;

  int count = 0;
  const cudaError_t e = cudaGetDeviceCount(&count);
  if (e != cudaSuccess) {
    d.reason = std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(e);
    return d;
  }
  if (count <= 0) {
    d.reason = "no CUDA device found (unsupported GPU / out-of-date driver / CPU-only machine)";
    return d;
  }
  if (index < 0 || index >= count) {
    d.reason = "device index out of range (there are " + std::to_string(count) + " devices)";
    return d;
  }

  cudaDeviceProp p{};
  const cudaError_t pe = cudaGetDeviceProperties(&p, index);
  if (pe != cudaSuccess) {
    d.reason = std::string("cudaGetDeviceProperties failed: ") + cudaGetErrorString(pe);
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

// ---------------------------------------------------------------- 引擎的 stream
//
// 两个进程级对象，**都故意不释放**（见下面各自的注释）：它们的析构会调
// cudaFree / cudaStreamDestroy，而那个时刻 CUDA 上下文可能已经销毁 —— 正是
// DESIGN §9.6 记的那个"退出时刷一屏失败"的坑。8 字节 + 一个流句柄，换掉一整类
// 退出期警告；进程退出时驱动自己回收。
namespace {

int64_t* g_pos_staging = nullptr;              // 设备指针，见 PosStagingPtr
cudaStream_t g_engine_stream = nullptr;        // nullptr ⇒ legacy default stream
cudaStream_t g_capture_stream = nullptr;       // 专用流，惰性创建
bool g_streams_ready = false;
// 暂存区里**当前装着**的值（栈上那一份的镜像）。见 StagePos0Cached 的说明：
// 它由 StagePos0 独占维护 —— 少一个写入路径，就少一类"缓存与显存不一致"。
int64_t g_pos_last = INT64_MIN;

}  // namespace

cudaStream_t EngineStream() {
  return g_engine_stream != nullptr ? g_engine_stream
                                    : static_cast<cudaStream_t>(nullptr);
}

bool EnsureCaptureStream() {
  // 已经试过一次就直接返回结果 —— 失败**不重试**：建流失败通常是上下文问题，
  // 每次生成都重试只会把一句错误刷成几百句。
  if (g_streams_ready) return g_capture_stream != nullptr;
  g_streams_ready = true;

  // ⚠️ 用默认（blocking）标志创建，**不加 cudaStreamNonBlocking**。
  // 理由：图路径之外还有同步的 cudaMemcpy（DeviceBuffer::Upload / Download）与
  // 采样结果回拷走 legacy default stream。blocking 流与 legacy 之间**有隐式
  // 同步**，顺序不会被悄悄打乱；non-blocking 省下的那点开销不值得冒这个险。
  const cudaError_t e = cudaStreamCreate(&g_capture_stream);
  if (e != cudaSuccess) {
    g_capture_stream = nullptr;
    return false;
  }
  return true;
}

ScopedEngineStream::ScopedEngineStream(cudaStream_t s) : prev_(EngineStream()) {
  g_engine_stream = s;
}

ScopedEngineStream::~ScopedEngineStream() { g_engine_stream = prev_; }

cudaStream_t CaptureStream() {
  return EnsureCaptureStream() ? g_capture_stream : nullptr;
}

int64_t* PosStagingPtr() {
  if (g_pos_staging == nullptr) {
    int64_t* p = nullptr;
    RPG_CUDA_CHECK(cudaMalloc(&p, sizeof(int64_t)));
    g_pos_staging = p;
  }
  return g_pos_staging;
}

void StagePos0(int64_t pos) {
  // 先保证已分配 —— 捕获期间**只能**走这一行，不能有 cudaMalloc（非法）。
  int64_t* dev = PosStagingPtr();
  // 捕获期间跳过写入：这个 cudaMemcpyAsync 的源是**调用方栈上的 pos**，被录进图
  // 之后那个地址早就失效了 ⇒ 回放时读的是野内存。跳过是对的 —— 地址（而不是值）
  // 已经进图，值由回放前的那一次写入提供。
  if (StreamIsCapturing()) return;
  RPG_CUDA_CHECK(cudaMemcpyAsync(dev, &pos, sizeof(int64_t),
                                 cudaMemcpyHostToDevice, EngineStream()));
  // 缓存与写入绑在一起 —— 这是"缓存永远不说谎"的唯一保证（见头文件里的说明）。
  g_pos_last = pos;
}

int64_t* StagePos0Cached(int64_t pos) {
  int64_t* dev = PosStagingPtr();  // 保证已分配（捕获期间只有这一步是合法的）
  if (StreamIsCapturing()) {
    // 捕获期间既不写值也不更新缓存：回放时会走完整的路径（地址已经进图）。
    return dev;
  }
  if (g_pos_last != pos) StagePos0(pos);
  return dev;
}

bool StreamIsCapturing() {
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  const cudaError_t e = cudaStreamIsCapturing(EngineStream(), &st);
  // 查询失败就当作"不在捕获中"：那条分支更保守（还是会真的写值）。
  return e == cudaSuccess && st != cudaStreamCaptureStatusNone;
}

}  // namespace rpg
