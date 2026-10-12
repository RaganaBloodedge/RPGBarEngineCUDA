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
  - `tests/check_sync.py`：**同源副本的一致性判据** —— 逐字节比较 22 个复用文件与
    2 个 ABI 文件。对拍（DESIGN §7）要能得出"差异只来自算力层"，前提就是这一层没跑偏；
    所以这不是一次性动作，而是一条随时可以重跑的判据。
  - `tests/abi_smoke.py`：ABI 冒烟（**不需要显卡**）。不给权重时验错误路径与句柄语义；
    给 `--model` 时额外验 `load_model`、`model_info` 的 17 个键、`generate` 的报错文案。
  - `scripts/fetch_model.py`：下权重与对拍参考物（走 hf-mirror，支持断点续传）。
  - `docs/DESIGN.md`：设计决策（为什么独立仓库、为什么权重直传显存、为什么 KV 用 fp16、
    为什么单 stream）、五个阶段的判据、**对拍策略**、与上位仓库的同步方式。

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
