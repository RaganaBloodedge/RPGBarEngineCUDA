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
| P0 | 工具链与骨架（ABI 层、CMake、`cuda_check`） | ✅ 骨架完成，host 层已验证；`cuda_check` 待装 Toolkit |
| P1 | 权重上卡 + 单层前向对拍 | 未开始 |
| P2 | 整网贪心生成对拍 | 未开始 |
| P3 | 采样 / 工具调用 / 接入 RPGBar | 未开始 |
| P4 | 融合 kernel、CUDA Graph、量化直算 | 未开始 |

⚠️ **前置条件**：本机当前**没有装 CUDA Toolkit**（`nvcc` 不存在），P1 起的工作需要它。
没有 Toolkit 时这个仓库仍能编译（host 部分），只是 `generate` 会明确报"未编译 CUDA 后端"。

## 编译

有 CUDA Toolkit：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/cuda_check          # 工具链自检：报出设备名、算力、显存（不需要权重）
```

**没装 Toolkit / 没装 cmake**（只要一个 g++ 就能编 host 部分）：

```bash
bash build_host.sh          # 产物 build/rpg_engine.dll（host-only）
```

本机 GPU 是 **RTX 5060 Laptop（8 GB，Blackwell sm_120）**，需要 CUDA **12.8+**。
跨机器分发产物时**不要**用 `-DCMAKE_CUDA_ARCHITECTURES=native`，要显式列出目标架构。

## 验证

```bash
python tests/check_sync.py      # 22 个同源副本 + 2 个 ABI 文件与上位仓库逐字节一致
python tests/abi_smoke.py --lib build/rpg_engine.dll
python tests/abi_smoke.py --lib build/rpg_engine.dll --model models/Qwen3-1.7B-Q4_K_M.gguf
```

第一条不需要显卡、不需要权重，第二条与第三条不需要显卡 —— **"骨架能编、错误路径说人话"
和"算力实现了没有"是两件事**，前者不该被显卡卡住。

## 用起来

产物是 `rpg_engine.dll` / `librpg_engine.so`，与 CPU 版**同名同 ABI**。
把它放到宿主的 `models/rpg_engine.dll` 位置即可 —— 宿主代码不需要改：

```bash
python scripts/engine_probe.py     # 在 RPGBar 仓库里跑，会打印引擎版本与实测 tok/s
```

## 目录

```
include/rpg_engine.h    ← 逐字复制自 RPGBarEngine，是唯一契约（两边都不许单独改）
src/                    ← host 层：ABI、GGUF、分词器、ChatML、工具解析（复用）
                          + CUDA 计算层（重写）
docs/DESIGN.md          ← 设计决策、阶段判据、对拍策略、与上位仓库的同步方式
scripts/                ← fetch_model.py（下权重与对拍参考物）
tests/                  ← check_sync.py（副本有没有跑偏）、abi_smoke.py（ABI 冒烟）
tools/                  ← cuda_check 等命令行工具
```

## 许可证

MIT。

`src/` 下有一部分文件（GGUF / 分词器 / ChatML / 工具解析 / JSON / 量化表）是从
[RPGBarEngine](https://github.com/RaganaBloodedge/RPGBarEngine) **逐字复制**来的同源副本，
版权同属原作者。**这些文件的文件头仍然写着 `// RPGBarEngine —— …`，一个字都没改** ——
这是故意的，理由见 [设计文档 §3](docs/DESIGN.md)：副本保持逐字节一致，
对拍才能得出"差异只来自算力层"，跟上游的修复也才是一条 `git checkout` 的事。
跑 `python tests/check_sync.py` 可以随时确认它们没跑偏。
