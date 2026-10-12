# RPGBarEngineCUDA

[RPGBarEngine](https://github.com/RaganaBloodedge/RPGBarEngine) 的 **GPU（CUDA）实现** ——
同一套 C ABI、同一份 GGUF 权重、同一套外部行为，前向计算跑在显卡上。

**为什么要有它**：RPGBar 的 DM 一次回合在 CPU 上要 **211 秒**（Qwen3-1.7B Q4_K_M，
16 线程：首轮 2811 token 的 prefill + 三轮工具循环）。把同一份权重放到 RTX 5060 上，
目标是把这一轮压到 **2 秒以内**。

**为什么是独立仓库**：[设计文档 §1](docs/DESIGN.md) 讲了三个理由，一句话 ——
CUDA 工具链不该绑架"只要一个 g++ 就能编"的 CPU 版；两个实现共用一份 ABI，
可以互相**对拍**。宿主（Python `ctypes` / Unity `P/Invoke`）换实现时**一行都不用改**。

## 状态

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| P0 | 工具链与骨架（ABI 层、CMake、`cuda_check`） | ✅ **完成** —— 两档判据全绿 |
| P1 | 权重上卡 + 整网前向对拍 | ✅ **完成** —— 两条判据全绿（见下） |
| P2 | KV Cache + 增量解码 + 贪心逐 token 对拍 | **下一步** |
| P3 | 采样 / 工具调用 / 接入 RPGBar | 未开始 |
| P4 | 融合 kernel、CUDA Graph、量化直算 | 未开始 |

**P1 的一句话结论**：同一份 GGUF、同一串 token，GPU 的 28 层前向与 CPU 引擎**逐层对得上**
（残差流最差 p99 = **1.05e-4**，判据 1e-3；argmax 21/21 一致），反量化**逐位相同**（0 bit 差异）。
GPU prefill 21 token 只用 **213.7 ms**，同一份 CPU 跑 **3081.1 ms** ⇒ **14.4×**。
⚠️ 但这还不是"DM 一轮 2 秒"——那要求 P2 的 KV Cache 与 P3 的接入，且 prefill 目标要在长
prompt（≥512 token）上量，见 [设计文档 §5](docs/DESIGN.md)。

**工具链现状**：CUDA Toolkit **13.2 已装在 WSL（Ubuntu 24.04）**（`/usr/local/cuda`，
版本与宿主 `nvidia-smi` 报的 13.2 精确对齐）。Windows 侧仍没有 nvcc / cmake / MSVC ——
所以 CUDA 路径的构建走 WSL（`bash build_wsl.sh`），产物是 `librpg_engine.so`。
要出宿主机（Windows）能加载的 `.dll`，得等 P3 时补 MSVC + Windows 版 Toolkit。

没有 Toolkit 时这个仓库仍能编译（host 部分），只是 `generate` 会明确报"未编译 CUDA 后端"。

## 编译

有 CUDA Toolkit：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/cuda_check          # 工具链自检：报出设备名、算力、显存（不需要权重）
```

**在 WSL 里编 CUDA 版**（源码留在 `/mnt/e`，构建目录落 WSL 原生 ext4 —— 理由见 `build_wsl.sh` 头部）：

```bash
bash build_wsl.sh                       # 产物 ~/cuda-build/librpg_engine.so
bash build_wsl.sh --clean               # 先清构建目录再编
~/cuda-build/cuda_check                 # 工具链自检
```

**没装 Toolkit / 没装 cmake**（只要一个 g++ 就能编 host 部分）：

```bash
bash build_host.sh          # 产物 build/rpg_engine.dll（host-only）
```

本机 GPU 是 **RTX 5060 Laptop（8 GB，Blackwell sm_120）**，需要 CUDA **12.8+**。
跨机器分发产物时**不要**用 `-DCMAKE_CUDA_ARCHITECTURES=native`，要显式列出目标架构。

## 验证

P0 的判据分两档。**不需要显卡的那档在 Windows 上跑**：

```bash
python tests/check_sync.py      # 22 个同源副本（20 src + 2 tools）+ 2 个 ABI 文件 = 24 项逐字节一致
python tests/abi_smoke.py --lib build/rpg_engine.dll
python tests/abi_smoke.py --lib build/rpg_engine.dll --model models/Qwen3-1.7B-Q4_K_M.gguf
```

**需要显卡的那档在 WSL 里跑**：

```bash
python3 tests/abi_smoke.py --lib ~/cuda-build/librpg_engine.so --model models/Qwen3-1.7B-Q4_K_M.gguf
~/cuda-build/cuda_check
```

**P1 的两条判据**（一键，在 WSL 里跑）：

```bash
apt install python3-numpy        # cmp_layers.py 要用（判据脚本唯一的外部依赖）
bash tests/run_p1.sh             # ① 反量化逐位对拍 ② 整网逐层对拍
```

分两档的理由 —— **"骨架能编、错误路径说人话"和"算力实现了没有"是两件事**，
前者不该被显卡卡住。

### 已验过的（2026-10-12）

| 项 | 环境 | 结果 |
| --- | --- | --- |
| `check_sync.py` | WSL | **24 一致 / 0 分叉 / 0 缺失**（20 src + 2 tools 副本 + 2 个 ABI 文件） |
| `abi_smoke.py`（host-only，带真实权重） | Windows | 32 通过 / 0 失败 |
| `abi_smoke.py`（CUDA 产物，带真实权重） | WSL | 32 通过 / 0 失败 |
| `cuda_check` | WSL + CUDA 13.2 | RTX 5060 Laptop / 算力 **12.0** / 26 SM / 显存 6.9+8.0 GiB；device 计算 + 搬运全对 |
| **`dequant_check`**（P1 ①） | WSL + CUDA 13.2 | f16→f32 穷举 65536 值全对；反量化 131072 个元素 **0 个 bit 不同** |
| **`tests/run_p1.sh`**（P1 ②） | WSL + CUDA 13.2 | 残差流 29 项 p99 ≤ **1.05e-4**（判据 1e-3）；logits p99 5.98e-5；argmax **21/21** 一致 |

### P1 实测（21 token prefill，同一 prompt / 同一份 GGUF）

| 项 | CPU（12 线程） | GPU（RTX 5060 Laptop） |
| --- | --- | --- |
| 前向耗时 | 3081.1 ms | **213.7 ms** |
| 吞吐 | ~7 tok/s | **98.3 tok/s**（prefill 口径） |
| 权重上卡 | — | 4.9 s（1.03 GiB，含从 `/mnt/e` 经 9p 读盘） |
| 显存 | — | 权重 1.03 GiB + 暂存 14.8 MB，设备剩余 5.62 GiB |

> 98 tok/s 离 P4 的 1000 tok/s 目标还远，**但现在不该优化**：21 个 token 的 GEMM 每个都是
> `1×2048` 乘 `2048×6144`，小到吃不满 26 个 SM。P4 的目标要在 ≥512 token 上量。

### ⚠️ 首次真编译（P0、P1 各一次）暴露的真问题

下面这些**都是 host-only 构建永远看不到的**（`RPG_HAS_CUDA` 未定义时整段被 `#ifdef` 掉），
也正是"没验过就不算"的实证。**P0 抓到 2 个、P1 抓到 4 个** —— 全部细节在
[设计文档 §8](docs/DESIGN.md)，这里只列名字：

P0：`cuda_common.h` 把 CUDA 头 include 进了 `namespace rpg`（报错全落在标准库内部）；
`cuda_check` 目标漏了 `cuda_common.cc`（链接期才炸）。

P1：`CMAKE_CUDA_STANDARD` 默认 17 而 `dtypes.h` 用了 `std::bit_cast`；
`add_compile_options(-O2)` 与 Release 的 CUDA `-O3` 冲突（**编译通过但悄悄降档**）；
静态成员函数里的 lambda 捕了 `this`；两个 `.cu` 漏 include。

## 用起来

产物是 `rpg_engine.dll` / `librpg_engine.so`，与 CPU 版**同名同 ABI**。
把它放到宿主的 `models/rpg_engine.dll` 位置即可 —— 宿主代码不需要改：

```bash
python scripts/engine_probe.py     # 在 RPGBar 仓库里跑，会打印引擎版本与实测 tok/s
```

## 目录

```
include/rpg_engine.h    ← 逐字复制自 RPGBarEngine，是唯一契约（两边都不许单独改）
src/                    ← host 层：ABI、GGUF、分词器、ChatML、工具解析、元数据推导（复用/新增）
                          + CUDA 计算层：gpu_model.cu / kernels_gpu.cu / qwen3_gpu.cu（重写）
docs/DESIGN.md          ← 设计决策、阶段判据、对拍策略、与上位仓库的同步方式
scripts/                ← fetch_model.py（下权重与对拍参考物）
tests/                  ← check_sync.py（副本有没有跑偏）、abi_smoke.py（ABI 冒烟）、
                          run_p1.sh（P1 一键判据）、cmp_layers.py（逐层误差统计）
tools/                  ← cuda_check / dequant_check / layer_check
                          + build_cpu_ref.sh（现编 CPU 引擎做参考侧）
```

## 许可证

MIT。

`src/` 下有一部分文件（GGUF / 分词器 / ChatML / 工具解析 / JSON / 量化表）是从
[RPGBarEngine](https://github.com/RaganaBloodedge/RPGBarEngine) **逐字复制**来的同源副本，
版权同属原作者。**这些文件的文件头仍然写着 `// RPGBarEngine —— …`，一个字都没改** ——
这是故意的，理由见 [设计文档 §3](docs/DESIGN.md)：副本保持逐字节一致，
对拍才能得出"差异只来自算力层"，跟上游的修复也才是一条 `git checkout` 的事。
跑 `python tests/check_sync.py` 可以随时确认它们没跑偏。
