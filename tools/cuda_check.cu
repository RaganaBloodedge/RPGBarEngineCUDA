// cuda_check —— 工具链自检（P0 的判据）。
//
// 判据只有一条：**这段代码能在目标机器上编译、运行、并给出正确的数值**。
// 它不碰权重，所以任何机器上都能跑（包括没有模型的 CI）。
//
// 它顺手回答了三个最容易在"换一台机器"时踩到的问题：
//   1. 有没有 CUDA 设备？（没有的话，为什么 —— QueryDevice 会给出人能看懂的原因）
//   2. 编进去的架构支不支持这块卡？（sm_120 是新架构，用旧工具链编出来会
//      "no kernel image is available for execution" —— 那不是代码错）
//   3. cudaMalloc / kernel 启动 / cudaMemcpy 三样是不是都真的能用？
#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

#include "common.h"
#include "cuda_common.h"

namespace {

__global__ void ScaleKernel(const float* in, float* out, int n, float k) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * k;
}

}  // namespace

int main() {
  std::printf("RPGBarEngineCUDA —— 工具链自检\n");
  std::printf("编译期 CUDA 运行时：%d（%d.%d）\n", CUDART_VERSION,
              CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);

  const rpg::DeviceInfo d = rpg::QueryDevice(0);
  std::printf("设备：%s\n", rpg::DescribeDevice(d).c_str());
  if (!d.available) {
    std::printf("\n没有可用设备，自检到此为止（上层仍可用 CPU 版引擎）。\n");
    return 1;
  }

  // 最小的一次 device 计算 + 往返搬运：三样基础设施都在这一步里验证。
  constexpr int kN = 1024;
  std::vector<float> host(kN), back(kN);
  for (int i = 0; i < kN; ++i) host[i] = static_cast<float>(i);

  try {
    rpg::DeviceBuffer dev(sizeof(float) * kN);
    dev.Upload(host.data(), sizeof(float) * kN);
    // 就地做（in == out）：每个线程只碰自己那个元素，读写合并是安全的。
    ScaleKernel<<<(kN + 255) / 256, 256>>>(static_cast<const float*>(dev.data()),
                                           static_cast<float*>(dev.data()), kN, 2.0f);
    // 两步都要查：启动错误在 cudaGetLastError，执行期错误在 synchronize。
    RPG_CUDA_CHECK(cudaGetLastError());
    RPG_CUDA_CHECK(cudaDeviceSynchronize());
    dev.Download(back.data(), sizeof(float) * kN);
  } catch (const std::exception& e) {
    std::printf("自检失败：%s\n", e.what());
    return 1;
  }

  int bad = 0;
  for (int i = 0; i < kN; ++i) {
    if (back[i] != host[i] * 2.0f) {
      if (bad < 4) {
        std::printf("  第 %d 个：得到 %g，应为 %g\n", i, back[i], host[i] * 2.0f);
      }
      ++bad;
    }
  }
  if (bad != 0) {
    std::printf("自检失败：%d / %d 个结果不对\n", bad, kN);
    return 1;
  }

  std::printf("device 计算 + 搬运：%d 个 float 全部正确\n", kN);
  std::printf("自检通过：CUDA 工具链可用。\n");
  return 0;
}
