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

#ifdef RPG_HAS_CUDA
// ⚠️ 这个 include 必须在 `namespace rpg` **之前**，不许挪进下面去。
// 放进去的后果：libstdc++ 的 `namespace std` 会被嵌套成 `rpg::std`，此后这个
// TU 里每一个 `std::` 名字都解析不到，而报错点全落在标准库内部 —— 例如
//   error: 'vector' in namespace 'rpg::std' does not name a template type
//   error: 'is_nothrow_move_constructible' was not declared in this scope
// 从现象反推根因极难（这两个报错看着像标准库或编译器出了问题，其实是自己的
// 头文件把命名空间打开了）。**这个坑真踩过**：2026-10-12 在 WSL 里第一次编
// CUDA 路径才暴露 —— Windows 的 host-only 构建因为没定义 RPG_HAS_CUDA，整段
// 被 #ifdef 掉，所以此前一次都没被编译过（见 docs/DESIGN.md §8）。
//
// 用 cuda_runtime_api.h 而不是 cuda_runtime.h：host 侧（g++ 编的 .cc）只需要
// cudaError_t 与内存 API。完整头会带进 __host__/__device__ 一整套宏，对 host
// 编译没有收益、只有风险。真要用 kernel 的 .cu 自己 include 完整头。
#  include <cuda_runtime_api.h>
#endif

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

// ---------------------------------------------------------------- 引擎的 stream
//
// P0–P3 一直用 legacy default stream（`<<<g, b>>>` 不带 stream 参数）。P4 要录
// CUDA Graph，这里**必须**改，理由不是风格：
//
//   * **legacy default stream 不许被 cudaStreamBeginCapture 捕获**（CUDA 明确
//     禁止，会报 `cudaErrorStreamCaptureUnsupported`）；
//   * 捕获要求"这一段里的**每一次**提交都落在同一个流上"，而 kernel 启动里的
//     stream 参数是在**启动那一刻**求值的 —— 所以只要让下面这个函数在捕获窗口
//     内返回专用流，`<<<g, b, 0, EngineStream()>>>` 就自动全部落在同一条流上，
//     不必给每个 kernel 加参数。
//
// 默认仍然是 0（legacy default）⇒ **不捕获时行为与 P0–P3 逐字一致**。
//
// ⚠️ 全局可变状态。安全性来自引擎的调用约定：宿主串行调用、单 stream 提交
// （DESIGN §4.3）。多线程同时生成本来就不支持 —— 与"一个 stream"是同一条约束。
cudaStream_t EngineStream();

// 惰性创建专用流。返回 false 表示创建失败 —— 那时图路径要**退化成老路径**，
// 而不是崩（判据里会看到"图没开"，而不是一个 CUDA 错误）。
bool EnsureCaptureStream();

// 专用流本身。没建出来时返回 nullptr。
cudaStream_t CaptureStream();

// 捕获窗口内把引擎流切到 `s`，析构时切回去。RAII 而不是成对的 set/get：
// 捕获中途抛异常（RPG_CHECK 失败）时也能恢复，否则**整个进程之后的所有 kernel
// 都会提交到那条即将销毁的流上**。
class ScopedEngineStream {
 public:
  explicit ScopedEngineStream(cudaStream_t s);
  ~ScopedEngineStream();
  ScopedEngineStream(const ScopedEngineStream&) = delete;
  ScopedEngineStream& operator=(const ScopedEngineStream&) = delete;

 private:
  cudaStream_t prev_;
};

// ---------------------------------------------------------------- pos0 暂存区
//
// CUDA Graph 回放时**每个 kernel 参数都是捕获那一刻的值**。decode 里每步都在
// 变的 `pos0` 因此不能走值参 —— 否则回放出来的图永远用同一个位置：RoPE 会一直
// 按同一个 pos 旋转、KV 会一直写进同一格，症状是"第 1 个 token 对，从第 2 个
// 起开始胡说"（最容易归因成模型质量差）。
//
// 做法：一块**固定地址**的显存，host 每步写一次，kernel 从显存里读。地址固定
// ⇒ 可以进图；值每步更新 ⇒ 位置正确。
//
// ⚠️ 为什么是"进程内一块"而不是"每会话一块"：GpuRope / GpuKvWrite /
// GpuAttentionFromCache 是算子层，签名里没有会话句柄（只吃裸指针），而
// tools/kv_map_check.cu 直接调用它们 ⇒ **签名不能动**。于是暂存区只能挂在算子
// 层内部。全局共享是安全的：所有写入与 kernel 都在同一条流上，顺序由流保证。
int64_t* PosStagingPtr();     // 现成的设备地址（惰性分配；见 .cu 里"为什么不释放"）
void StagePos0(int64_t pos);  // 写入 pos；**捕获期间跳过**（源是栈变量，录进图就废了）
bool StreamIsCapturing();     // 引擎流当前是否处于捕获中

// 写入 pos 并**带值缓存**：同一层里 Rope 要被调两次（q 一次、k 一次），同一个
// pos0 还各被 KvWrite / AttentionFromCache 用一次 —— 每次 cudaMemcpyAsync 8 字节
// 约几微秒，36 层下来是几百微秒，白扔。
//
// ⚠️ 它**必须**与 StagePos0 同层，而且必须是唯一的读缓存入口。原因值得记一笔：
// 缓存要成立，前提是"缓存里的值 == 显存里的值"。只要存在**第三条**写显存的路
// （P4 的图回放循环就是直接调 `StagePos0`），缓存就可能变成谎话 —— 现象是
// "图路径跑过一次、紧接着另一个会话从位置 0 开始 prefill 时，那一次 prefill
// 的 RoPE 全部用错位置"，而且它只在特定调用顺序下出现。所以把缓存挂在
// `StagePos0` 自己身上：**没有别的写入路径**，前提就不可能被破坏。
//
// 缓存为什么安全：所有写入与所有 kernel 都在**同一条流**上（DESIGN §4.3）。
// host 看到"已经是 pos"时，流上排在最后的那次写入就是 pos，于是排在它后面的
// kernel 读到的必然也是 pos。跨会话交替调用只会让缓存命中率下降（多写几次同样
// 的值），不会读错。
//
// ⚠️ 单线程假设：与"引擎不做多线程"是同一条约束（宿主串行调用）。真要多线程
// 得给每个线程一份暂存区，那时这个状态就该进线程局部存储。
int64_t* StagePos0Cached(int64_t pos);

#endif  // RPG_HAS_CUDA

}  // namespace rpg

#endif  // RPGBAR_CUDA_COMMON_H_
