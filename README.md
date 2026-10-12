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
| P2 | KV Cache（fp16）+ 增量解码 + 贪心逐 token 对拍 | ✅ **完成** —— 三条判据全绿，**32/32 个 token 与 CPU 引擎完全一致** |
| P3 | 采样（**已接进生成循环**）/ 工具调用 / 接入 RPGBar | 🔶 **进行中** —— 设备端采样两条判据全绿（一条**不需要模型**、一条走真实 decode 循环）；还差挂进 ABI session 与换 `rpg_engine.dll` |
| P4 | 融合 kernel、CUDA Graph、量化直算 | 未开始 |

**P1 的一句话结论**：同一份 GGUF、同一串 token，GPU 的 28 层前向与 CPU 引擎**逐层对得上**
（残差流最差 p99 = **1.05e-4**，判据 1e-3；argmax 21/21 一致），反量化**逐位相同**（0 bit 差异）。
GPU prefill 21 token 只用 **213.7 ms**，同一份 CPU 跑 **3081.1 ms** ⇒ **14.4×**。

**P2 的一句话结论**：KV Cache 跑在显存里（fp16，@4096 是 448 MiB），增量解码 + 贪心生成
与 CPU 引擎**逐 token 完全一致**（32/32），每步 logits 的满量程误差 **6.5e-4**（判据 1e-3，
约 1.3 个 fp16 ULP）。KV 写入的下标映射另有一条**不需要模型**的确定性判据（逐位相同）。
⚠️ 但 decode 只比 CPU 快 **1.5 倍** —— 瓶颈已经从"算力"变成"每步约 400 次 kernel 启动
+ 一次同步"，那是 P4 的 CUDA Graph 要解决的事（见 [设计文档 §5](docs/DESIGN.md)）。

**P3 第一步（设备端采样）的一句话结论**：`temperature` / `top_k` / `top_p` 与
counter-based 随机源都在 GPU 上跑通了，判据**不需要模型、不需要权重**（喂同一份从文件读的
logits，两边就该给出同一个 token）—— u 表 **192/192 位模式相同**、概率满量程误差
**1.66e-7**（判据 1e-5）、**192/192 个 token 逐位相同**，词表 4096 与 151936 各跑一遍
（后者专门验二分搜索覆盖整个 key 空间）。
筛选不用排序：top_k 找"第 k 大值"、top_p 找"累积概率切点"，各 32 次迭代遍历一遍词表 ——
顺手把 CPU 版"`top_p` 全量 `std::sort`（10.25 ms）"那个待办一起解决了。

**P3 第二步（接进生成循环）的一句话结论**：`GpuGenerateGreedy` 变成 `GpuGenerate` 的
薄包装（与 CPU 版 `GenerateGreedy` 同构），采样在 decode 循环里真的生效了，而**贪心路径
逐位没变** —— `run_p2.sh` 的 32/32 逐 token 对拍与满量程 **6.48e-4** 与 P2 记录的数字
一模一样。这一层的判据不能与 CPU 逐 token 对拍（接在模型后面的采样先差了 fp16 舍入，
理由见 [设计文档 §6 第 10 条](docs/DESIGN.md)），所以换成五条**只看自己**的不变量：
同 seed 两遍**逐位相同**（32/32，证明引擎仍是纯函数）、换 seed 序列就变（31/32 处不同）、
`T=0` 加筛选参数与贪心逐位相同、`top_k=2` 时每步都落在该步前 2 名内、以及
**counter 用的是绝对位置**——把输出接在 prompt 后冷跑一遍，序列整体错一位
**逐位相同**（31/31）。

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
python tests/check_sync.py      # 23 个同源副本（21 src + 2 tools）+ 2 个 ABI 文件 = 25 项逐字节一致
python tests/abi_smoke.py --lib build/rpg_engine.dll
python tests/abi_smoke.py --lib build/rpg_engine.dll --model models/Qwen3-1.7B-Q4_K_M.gguf
```

**需要显卡的那档在 WSL 里跑**：

```bash
python3 tests/abi_smoke.py --lib ~/cuda-build/librpg_engine.so --model models/Qwen3-1.7B-Q4_K_M.gguf
~/cuda-build/cuda_check
```

**P1 / P2 / P3 的判据**（一键，在 WSL 里跑）：

```bash
apt install python3-numpy        # cmp_layers.py / cmp_gen.py / cmp_sample.py 要用
bash tests/run_p1.sh             # ① 反量化逐位对拍 ② 整网逐层对拍
bash tests/run_p2.sh             # ⓿ KV 映射（不需要模型）① KV 自检 ② 逐 token 对拍
bash tests/run_p3.sh             # 设备端采样（不需要模型、不需要权重，几秒跑完）
bash tests/run_p3_gen.sh         # 采样接进 decode 之后的五条不变量（要权重）
```

分两档的理由 —— **"骨架能编、错误路径说人话"和"算力实现了没有"是两件事**，
前者不该被显卡卡住。

### 已验过的（2026-10-12）

| 项 | 环境 | 结果 |
| --- | --- | --- |
| `check_sync.py` | WSL | **25 一致 / 0 分叉 / 0 缺失**（21 src + 2 tools 副本 + 2 个 ABI 文件） |
| `abi_smoke.py`（host-only，带真实权重） | Windows | 32 通过 / 0 失败 |
| `abi_smoke.py`（CUDA 产物，带真实权重） | WSL | 32 通过 / 0 失败 |
| `cuda_check` | WSL + CUDA 13.2 | RTX 5060 Laptop / 算力 **12.0** / 26 SM / 显存 6.9+8.0 GiB；device 计算 + 搬运全对 |
| **`dequant_check`**（P1 ①） | WSL + CUDA 13.2 | f16→f32 穷举 65536 值全对；反量化 131072 个元素 **0 个 bit 不同** |
| **`tests/run_p1.sh`**（P1 ②） | WSL + CUDA 13.2 | 残差流 29 项 p99 ≤ **1.05e-4**（判据 1e-3）；logits p99 5.98e-5；argmax **21/21** 一致 |
| **`kv_map_check`**（P2 ⓿） | WSL + CUDA 13.2 | KV 写入的下标映射：120 个元素**逐位正确**；未写入区间保持为 0（无越界） |
| **`gen_check` 的 KV 自检**（P2 ①） | WSL + CUDA 13.2 | 满量程误差 **4.09e-4**（判据 1e-3）；逐位置 argmax **21/21** 一致 |
| **`tests/run_p2.sh`**（P2 ②） | WSL + CUDA 13.2 | 逐 token **32/32 完全一致**；数值层满量程 6.48e-4；cache 长度不变量两侧都成立 |
| **`sample_check` + `cmp_sample.py`**（P3 ①，词表 4096，**不需要模型**） | WSL + CUDA 13.2 | u 表 **192/192 位模式相同**；概率满量程误差 **2.31e-7**（判据 1e-5）且保留集合逐例一致；token **192/192 逐位相同** |
| **同上（词表 151936，真实词表）** | WSL + CUDA 13.2 | u 表 192/192；满量程 **1.66e-7**；保留集合一致；token **192/192 逐位相同** |
| **`gen_sample_check` + `run_p3_gen.sh`**（P3 ②，要权重） | WSL + CUDA 13.2 | 同 seed 两遍 **32/32 逐位相同**；换 seed **31/32 处不同**；`T=0`+筛选参数与贪心**逐位相同**；`top_k=2` 每步都落前 2 名（32/32）；counter 绝对位置：暖/冷缓存 **31/31 逐位相同** |
| **回归 `run_p2.sh`（新循环）** | WSL + CUDA 13.2 | 逐 token 仍 **32/32 完全一致**，满量程仍 **6.48e-4**（与 P2 记录的数字一模一样） |
| **回归 `run_p1.sh`** | WSL + CUDA 13.2 | 29 项最差 p99 **1.05e-4**；logits max\|Δ\| **1.431e-4**（与 P1 记录一致） |

### P1 实测（21 token prefill，同一 prompt / 同一份 GGUF）

| 项 | CPU（12 线程） | GPU（RTX 5060 Laptop） |
| --- | --- | --- |
| 前向耗时 | 3081.1 ms | **213.7 ms** |
| 吞吐 | ~7 tok/s | **98.3 tok/s**（prefill 口径） |
| 权重上卡 | — | 4.9 s（1.03 GiB，含从 `/mnt/e` 经 9p 读盘） |
| 显存 | — | 权重 1.03 GiB + 暂存 14.8 MB，设备剩余 5.62 GiB |

### P2 实测（21 token prompt + 生成 32 个 token）

| 项 | CPU（12 线程） | GPU（RTX 5060 Laptop） | 倍率 |
| --- | --- | --- | --- |
| prefill 21 token | 3222.1 ms（6.5 tok/s） | **228.7 ms（91.8 tok/s）** | **14.1×** |
| decode 32 token | 2609.7 ms（12.3 tok/s） | **1728.9 ms（18.5 tok/s）** | **1.5×** |
| KV Cache(fp16) | 896 MiB（fp32，@4096） | **448 MiB**（@256 只要 28 MiB） | — |

> ⚠️ **decode 只有 1.5 倍，这个数字要留着**：decode 每步只算 1 个 token，而每步要发
> 28 层 × 十几个 kernel ≈ **400 次启动**，外加一次 argmax 的 4 字节回传同步。
> **瓶颈已经从"算力"换成了"启动开销 + 同步"** —— 这正是 P4 的 CUDA Graph 要解决的
> （把几百次启动摊成一次）。P2 的任务是"把对拍做对"，这个数字是 P4 的起点。

> 98 tok/s 离 P4 的 1000 tok/s 目标还远，**但现在不该优化**：21 个 token 的 GEMM 每个都是
> `1×2048` 乘 `2048×6144`，小到吃不满 26 个 SM。P4 的目标要在 ≥512 token 上量。

### P3 实测（采样接进生成循环之后）

| 项 | 实测 |
| --- | --- |
| decode 32 token（T=0，走新循环） | 1932.9 ms（**16.6 tok/s**）—— 与 P2 的 18.5 是同一量级的运行间波动；贪心分支调的仍是同一个 `GpuArgMax`，**没有新增显存搬运**（概率缓冲 607 KB 是 `GpuScratch` 里常驻的，不逐 token 分配） |
| decode 32 token（T=1，真采样） | 2141.7 ms（**14.9 tok/s**）—— 每步多一次 `FilterToProbs`（两次 32 轮二分 + 一次串行逆变换，~0.5 ms） |

两个数字都离 P4 的 60 tok/s 目标还远，而且**离瓶颈很远**：decode 每步约 400 次 kernel
启动 + 一次同步才是主因（见上面 P2 那张表）。采样加进去的是"几步 kernel"，不是"几百步"。

### ⚠️ 首次真编译（P0 / P1 / P2 各一次）暴露的真问题

下面这些**都是 host-only 构建永远看不到的**（`RPG_HAS_CUDA` 未定义时相关代码整段被
`#ifdef` 掉，`tools/` 下的判据工具根本不会被创建），也正是"没验过就不算"的实证。
**P0 抓到 2 个、P1 抓到 4 个、P2 抓到 5 个** —— 全部细节在
[设计文档 §8](docs/DESIGN.md)，这里只列名字：

**P0**：`cuda_common.h` 把 CUDA 头 include 进了 `namespace rpg`（报错全落在标准库内部）；
`cuda_check` 目标漏了 `cuda_common.cc`（链接期才炸）。

**P1**：`CMAKE_CUDA_STANDARD` 默认 17 而 `dtypes.h` 用了 `std::bit_cast`；
`add_compile_options(-O2)` 与 Release 的 CUDA `-O3` 冲突（**编译通过但悄悄降档**）；
静态成员函数里的 lambda 捕了 `this`；两个 `.cu` 漏 include。

**P2**：🔴 **`GpuForwardFull` 里 `logits == nullptr` 的早退差点写到 lm_head 之前** ——
那样 `decode` 循环里的 logits 缓冲永远是上一轮的垃圾，**不崩，只是从第二个 token 起
全错**；`GpuArgMax` 改了签名没改头文件里的声明；`tools/cpu_ref_gen.cc` 用了
`rpg::ArgMax` 但 `qwen3.h` 不转出 `kernels.h`；**nvcc 不接受 `void*` → `const float*`
的隐式转换**（g++ 接受，所以这是平台差异）；注释里行尾的 shell 续行 `\` 触发 `-Wcomment`。

**P3（第一步）一次编过，没有新坑** —— 不是运气：上面那些都写进了设计文档与文件头注释，
写新代码时是照着它们绕开的。**坑清单的价值就在于把"每次重新踩一遍"换成"照着单子绕开"。**
唯一需要记的是两个"设计上就避开"的陷阱（不是踩到才发现的），见
[设计文档 §8 的 P3 一节](docs/DESIGN.md)：两级二分的上中点必须用 `uint64` 算
（`hi - lo + 1` 在 `lo=0 / hi=0xFFFFFFFF` 时溢出成 0 ⇒ 死循环），以及逆变换的
求和顺序本身就是判据的一部分（并行扫描会与 CPU 选到不同的 token）。

**P3 第二步（接进生成循环）也没有编译坑，但踩了一个"判据前提"的坑**，值得单独记：
我拿 `top_k = 1` 当"支持集只剩一个元素 ⇒ 必然退化成 argmax"来写断言，**红了一片**。
查契约才发现 `sampler.h` 里 top_k 的生效条件是 **`top_k > 1 && top_k < n`**，
所以 **`1` 与 `0` 一样算"关闭"**（注释只写了 "0 = 关闭"）。
⇒ 代码没错，错的是判据的前提。**判据的前提也要去读契约，不能凭常识**——
尤其当"常识"（只留 1 个就该等于 argmax）听起来毫无破绽的时候。而 `sampler.h` 是
**同源副本**，不许改，所以这里只记录、不修上游。

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
                          + CUDA 计算层（重写）：gpu_model.cu（权重上卡）
                            / kv_cache_gpu.cu（fp16 KV Cache）/ kernels_gpu.cu（算子）
                            / qwen3_gpu.cu（前向 + 增量解码 + 生成循环）
                            / sampler_gpu.cu（采样：T / top_k / top_p + 随机源）
                          ← 另有 sampler.h：**逐字复制**的同源副本（采样参数的语义是契约）
docs/DESIGN.md          ← 设计决策、阶段判据、对拍策略、与上位仓库的同步方式
scripts/                ← fetch_model.py（下权重与对拍参考物）
tests/                  ← check_sync.py（副本有没有跑偏）、abi_smoke.py（ABI 冒烟）、
                          run_p1.sh / cmp_layers.py（P1 一键判据）、
                          run_p2.sh / cmp_gen.py（P2 一键判据）、
                          run_p3.sh / cmp_sample.py（P3 一键判据，不需要模型）、
                          run_p3_gen.sh（P3 的生成循环判据，要权重）
tools/                  ← cuda_check / dequant_check / layer_check / gen_check
                          / kv_map_check（KV 映射，不需要模型）
                          / sample_check（采样，不需要模型）
                          / gen_sample_check（采样接进 decode 后的不变量，要模型）
                          + build_cpu_ref.sh（现编 CPU 引擎做参考侧：
                            forward_check / cpu_ref_gen / cpu_ref_sample）
                          + cpu_ref_gen.cc / cpu_ref_sample.cc（只有 main，实现来自 CPU 引擎）
```

## 许可证

MIT。

`src/` 下有一部分文件（GGUF / 分词器 / ChatML / 工具解析 / JSON / 量化表）是从
[RPGBarEngine](https://github.com/RaganaBloodedge/RPGBarEngine) **逐字复制**来的同源副本，
版权同属原作者。**这些文件的文件头仍然写着 `// RPGBarEngine —— …`，一个字都没改** ——
这是故意的，理由见 [设计文档 §3](docs/DESIGN.md)：副本保持逐字节一致，
对拍才能得出"差异只来自算力层"，跟上游的修复也才是一条 `git checkout` 的事。
跑 `python tests/check_sync.py` 可以随时确认它们没跑偏。
