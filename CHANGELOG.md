# 更新日志（Changelog）

本项目所有值得记录的版本更迭都写在这里。
格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## 版本约定

- **新功能**：`0.MINOR.0`，次版本号 +1。
- **修 bug / 小改动**：`0.MINOR.PATCH`，修订号 +1。
- **破坏性变更**：显式标注 `⚠ 破坏性` 并写明迁移方式。
- ⚠️ **ABI 变更要特别标注**：`include/rpg_engine.h` 是宿主（Python / Unity）绑定的依据，
  改它等于改两边。任何 ABI 变更都必须在条目里写明，并同步 RPGBarEngine。

版本号的**唯一来源**是 `src/engine.cc` 里的 `kEngineVersion`，与 `CMakeLists.txt` 的
`project(VERSION)` 必须一致。

发版流程（与上位仓库一致）：

1. 开发时把改动写进下面的 `[未发布]` 小节；
2. 发版时把 `[未发布]` 改成 `## [x.y.z] - YYYY-MM-DD`，并新建一个空的 `[未发布]`；
3. 同步 `kEngineVersion` 与 `CMakeLists.txt` 的版本号；
4. 提交，打 annotated tag：`git tag -a vX.Y.Z -m "..."`；
5. 推送：`git push origin main && git push origin --tags`；
6. 在 GitHub 上对该 tag 创建 Release，正文直接复制本文件里对应版本的小节。

> 记录原则：只写**使用者（宿主开发者）能感知的变化**。

---

## [未发布]

### 新增

- **项目初始化（P0）**：CUDA 版的仓库骨架。
  - `include/rpg_engine.h` + `src/rpg_engine.def`：与 RPGBarEngine 逐字相同的 C ABI
    （10 个函数）。宿主换实现时不需要改任何代码。
  - 平台无关层（GGUF / 分词器 / ChatML / 工具解析 / JSON / unicode / 量化表）从
    RPGBarEngine 同源复制 —— 这一层是"文本 ↔ token ↔ 结构化输出"，与算力无关，
    两边行为必须一致。**副本刻意保持逐字节一致**，文件头也不改（理由见 DESIGN §3）。
  - `src/engine.cc`：ABI 层最小实现（版本、错误、模型/会话句柄表、`model_info`）。
    没编译 CUDA 后端时，`generate` 会明确报错而不是静默失败。
  - `src/cuda_common.{h,cc,cu}`：CUDA 错误检查、设备信息、显存的 RAII 包装。
    拆成 `.cc` + `.cu` 两份是因为没有 `nvcc` 时 `.cu` 根本不参与编译，
    设备描述的 host-only 兜底实现必须留在 `.cc` 里，否则链接不过。
  - `CMakeLists.txt` + `build_host.sh`：前者是有 CUDA Toolkit 时的正规构建；
    后者是**没有 cmake、没有 nvcc** 时的最小 host 构建（只要一个 g++）。
  - `tools/cuda_check.cu`：**工具链自检**（不需要权重）—— 报出设备名、算力、显存，
    并跑一次最小的 device→host 搬运。
  - `tests/check_sync.py`：**同源副本的一致性判据** —— 逐字节比较复用文件与 2 个 ABI 文件
    （P1 之后是 20 src + 2 tools 个副本 + 2 个 ABI = 24 项）。对拍（DESIGN §7）要能得出
    "差异只来自算力层"，前提就是这一层没跑偏；所以这不是一次性动作，而是一条随时可以重跑的判据。
  - `tests/abi_smoke.py`：ABI 冒烟（**不需要显卡**）。不给权重时验错误路径与句柄语义；
    给 `--model` 时额外验 `load_model`、`model_info` 的 17 个键、`generate` 的报错文案。
  - `scripts/fetch_model.py`：下权重与对拍参考物（走 hf-mirror，支持断点续传）。
  - `docs/DESIGN.md`：设计决策（为什么独立仓库、为什么权重直传显存、为什么 KV 用 fp16、
    为什么单 stream）、五个阶段的判据、**对拍策略**、与上位仓库的同步方式。

### 新增（P1：权重上卡 + 整网前向对拍）

- **GPU 计算层**（重写，不与上位仓库同步）：`src/gpu_model.{h,cu}`（GGUF → 显存，
  量化字节**原样上传**，只接受 F32 / Q4_K / Q6_K，其余在加载期直接拒并给出可读原因）、
  `src/quant_gpu.cuh`（设备端反量化，`__fmul_rn`/`__fsub_rn` 显式不做 FMA 收缩）、
  `src/kernels_gpu.{h,cu}`（8 个算子：反量化 / 融合量化点积 / RMSNorm / RoPE /
  因果注意力 / SwiGLU / 残差加 / embedding 查表）、`src/qwen3_gpu.{h,cu}`（整网 28 层前向，
  本阶段无 KV Cache）。
- **`src/model_meta.{h,cc}`**：把"从 GGUF 推形状与超参"收敛成**唯一一处**
  （`head_dim` 只认 `attention.key_length`）。`engine.cc` 的 `model_info` 与 `GpuModel`
  都读它 —— 两处推导在 `hidden/n_heads == head_dim` 的模型上出错是看不出来的。
- **判据工具与脚本**：`tools/dequant_check.cu`（反量化逐位对拍 + f16→f32 全量穷举）、
  `tools/layer_check.cu`（整网前向，写出与 CPU 版同名同形状的 `.npy` + `.tokens.txt`）、
  `tests/cmp_layers.py`（逐层残差流与 logits 的分位数统计，口径与上位仓库的 `cmp_npy.py` 一致）、
  `tests/run_p1.sh`（P1 一键判据）、`tools/build_cpu_ref.sh`（**现编 CPU 引擎的
  `forward_check` 作参考侧** —— 不在本仓库抄一份 CPU 前向，否则两边同源的理解错误会互相掩盖）。
- `tools/npy.h`、`tools/cli_util.h`：与上位仓库同源复制，并**纳入 `check_sync.py`**
  （同源副本数 20 → 22 个，连同 2 个 ABI 文件共 24 项判据）。
- `CMakeLists.txt` 改为 **OBJECT 库**（`rpg_engine_objs`）：判据工具要用引擎内部的 C++ 符号
  （`rpg::GpuModel` / `rpg::GpuForwardFull`），而共享库在 Windows 上被 `.def` 限制成只导出
  那 10 个 C ABI 函数。同时加 `-ffp-contract=off`（逐位对拍的前提，见 DESIGN §7）。

### 说明（P1）

- **P1 的两条判据全绿**（2026-10-12，WSL + CUDA 13.2 + RTX 5060 Laptop）：
  ① 反量化 131072 个元素 **0 个 bit 不同**，f16→f32 穷举 65536 个值全对；
  ② 整网 29 项残差流最差 **p99 = 1.05e-4**（判据 1e-3）、logits p99 5.98e-5、
  **21/21 个位置的 argmax 一致**，最后位置 top-5 数值逐个相同。
- **P1 的判据比原计划的"单层"更严**：判据是**整网 28 层逐层 + logits 全量对拍**。理由与
  "P2 的内容一条没少"写在 DESIGN §5 的注里（无 cache 的整网前向里，"一层"与"28 层"是
  同一段代码跑 28 次；逐层比较能把误差定位到层）。
- **性能（同一 prompt，21 token）**：GPU 前向 **213.7 ms**，CPU（12 线程）3081.1 ms ⇒ **14.4×**；
  权重上卡 4.9 s（1.03 GiB）；上卡后设备剩余 5.62 GiB。
  ⚠️ 这还不是"DM 一轮 2 秒"—— 需要 P2 的 KV Cache 与 P3 的接入，且 prefill 目标应在
  ≥512 token 上量（21 个 token 的 GEMM 吃不满 26 个 SM）。
- **P1 首次真编译又抓出 4 个 host-only 构建看不到的问题**：`CMAKE_CUDA_STANDARD` 默认 17
  而同源副本 `dtypes.h` 用了 `std::bit_cast`（报错落在标准库内部）；`add_compile_options(-O2)`
  与 Release 下 nvcc 的 `-O3` 冲突（**编译通过但把 CUDA 侧悄悄降到 -O2**）；静态成员函数里的
  lambda 捕获了 `this`；`gpu_model.h` / `dequant_check.cu` 各漏一个 include。全部修掉，教训
  写进 DESIGN §8。
- **ABI 行为没有变化**：`generate` 仍然明确报错（文案更新为"整网生成还没实现，P1 只到
  无 KV Cache 的整网前向"），`rpg_engine.h` 一个字节未动。上卡接进 ABI 是 P2 的事。

### 说明

- **P0 的两档判据全绿**（2026-10-12）。不需要显卡的那一档在 Windows 上跑
  （`check_sync.py` 22 一致 / `abi_smoke.py` 32 通过）；需要显卡的那一档在 WSL +
  CUDA 13.2 上跑（`abi_smoke.py` 32 通过；`cuda_check` 报出 RTX 5060 Laptop / 算力 12.0 /
  26 SM，device 计算与搬运全对）。
- **CUDA 路径的构建放在 WSL 里**，因为 Windows 侧没有 MSVC（而 nvcc 不接受 g++ 当宿主
  编译器）。为此新增 `build_wsl.sh`：源码留在 `/mnt/e`（一份，不会分叉），构建目录落
  WSL 原生 ext4（9p 上的顺序写只有 173 MB/s，原生是 2.5 GB/s）。产物 `librpg_engine.so`
  供 P0–P2 的算法对拍使用；宿主能加载的 `.dll` 等 P3 时补 MSVC + Windows 版 Toolkit。
- **首次编译抓出两个真问题**（host-only 构建看不到，因为相关代码被 `#ifdef` 掉）：
  `cuda_common.h` 把 `<cuda_runtime.h>` include 进了 `namespace rpg`（libstdc++ 的
  `namespace std` 被嵌套成 `rpg::std`，该 TU 全线崩、报错却全落在标准库内部）；
  `cuda_check` 目标漏了 `src/cuda_common.cc`（链接期找不到 `rpg::DescribeDevice`）。
  两处都已修，教训写进 DESIGN §8。
