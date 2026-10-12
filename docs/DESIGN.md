# RPGBarEngineCUDA —— 设计文档

## 0. 这是什么

[RPGBarEngine](https://github.com/RaganaBloodedge/RPGBarEngine) 的 **GPU（CUDA）实现**。
同一个 C ABI、同一份 GGUF 权重、同一套"文本进、结构化出"的外部行为，但前向计算跑在
显卡上。

它服务的场景很具体：**RPGBar 的 DM 一轮要几十秒**。CPU 版（Qwen3-1.7B Q4_K_M，
16 线程）实测 prefill 约 145 tok/s、decode 约 10–14 tok/s，于是一次真实 DM 回合
（首轮 prompt 2811 token + 三轮工具循环）要 211 秒。把同一份权重放到 RTX 5060
Laptop 上，这个数字应该落到 **1 秒以内** —— 这不是"优化 10%"，是换了一个数量级。

## 1. 为什么单独一个仓库（而不是给 RPGBarEngine 加个 `.cu`）

三个理由，都是工程性的：

1. **工具链绑架**。CPU 版的价值之一是"零依赖"：`bash build.sh`，只要一个 g++ 就能编，
   产物丢进 Unity 就能用。一旦把 CUDA 塞进同一个 CMakeLists，**没装 CUDA Toolkit 的人
   就编不出任何东西**，包括 CPU 版。分开之后，两个仓库各自可构建、各自可发版。
2. **资源模型不同**。CPU 版的复杂度在"缓存友好 + SIMD + 线程池"；GPU 版的复杂度在
   "显存分配 + 权重布局 + kernel 融合 + 流同步"。这是两套几乎不重叠的知识，塞进一个
   `src/` 只会让人读不懂哪部分属于谁。
3. **对照实验**。两边跑同一份权重、同一段 prompt，输出可以直接互相对拍。**独立仓库让
   "对拍"成为一条真实可执行的命令，而不是同一份代码里的两个分支**。

> ⚠️ 代价要说清楚：平台无关的那一层（GGUF / 分词器 / ChatML / 工具解析）在两个仓库里
> **各有一份副本**。上游修了 bug，这里不会自动跟 —— 同步方式见 §10。

## 2. 唯一的契约：C ABI

`include/rpg_engine.h` 是**从上位仓库逐字复制**的，任何一方都不许单独改它：

```c
const char* rpg_engine_version(void);
const char* rpg_engine_last_error(void);
int32_t     rpg_engine_load_model(const char* model_path, const char* config_json);
void        rpg_engine_unload_model(int32_t model);
int32_t     rpg_engine_session_create(int32_t model, const char* session_json);
void        rpg_engine_session_destroy(int32_t session);
int32_t     rpg_engine_session_reset(int32_t session);
char*       rpg_engine_generate(int32_t session, const char* request_json);
void        rpg_engine_free(char* buf);
char*       rpg_engine_model_info(int32_t model);
```

**这正是"独立项目 → 再整合"可行的全部原因**：宿主侧的绑定是按函数名 + 参数类型写死的
（Python 是 `ctypes` 的 `CDLL`，Unity 是 `DllImport`），换实现只是把 `rpg_engine.dll`
换一个文件、或者把 `engine.lib` 指到另一个目录。**宿主一行代码都不用改。**

判据（P3 的验收条件）：把 RPGBar 的 `models/rpg_engine.dll` 换成 CUDA 版，
`python scripts/engine_probe.py` 与 `python scripts/smoke_test.py` 都照常通过。

## 3. 复用边界：哪些抄、哪些重写

| 层 | 文件 | 处置 | 理由 |
| --- | --- | --- | --- |
| 容器解析 | `gguf.*` `mmap_file.*` `bytes.h` | **复用** | 读文件、按 offset 取张量，与算力无关 |
| 文本层 | `tokenizer.*` `chat_template.*` `tool_parser.*` `unicode*` | **复用** | "文本 ↔ token ↔ 工具调用 JSON"，纯 CPU 逻辑 |
| 基础设施 | `json.*` `common.*` `dtypes.h` | **复用** | |
| 反量化 | `quant.*` | **复用（host 侧）** | 块缩放表的解析；GPU 侧的直算要另写 |
| 元数据 | `model_meta.*` | **新增（从 `model.cc` 里抽出来）** | 形状/超参的推导**只留一处**：`engine.cc`（灌进 `model_info`）与 `gpu_model.cu` 都用它。见下面的警告 |
| 权重 | `model.*` | **重写** | 权重从 mmap 变成**常驻显存**，布局也要按 kernel 重排 |
| KV Cache | `kv_cache.*` | **重写** | 放到显存，fp16 |
| 前向 | `qwen3.*` `matmul*` `kernels.h` | **重写** | 这是全部工作量所在 |
| 采样 | `sampler.cc` | **部分重写** | top-k/top-p 的排序在 GPU 上做（或用 `thrust`） |
| ABI 层 | `engine.cc` | **重写** | 句柄表、会话管理、JSON 编解码保持形状，但内部对象换成 GPU 侧 |
| 判据工具 | `tools/npy.h` `tools/cli_util.h` | **复用** | npy 布局与"命令行读文件"的写法必须两边一样，否则对拍脚本的报错会误导 |

> 🔴 **`model_meta.*` 为什么必须只有一份（P1 才定下来的一条）**：
> 上位仓库里"从 GGUF 推形状"这件事出现过两份推导（`model.cc` 与 `engine.cc`），两边对
> `head_dim` 的取法并不相同 —— Qwen3-1.7B 上 `hidden / n_heads = 2048/16 = 128` 恰好等于
> 真正的 `attention.key_length`，所以**两份里有一份是错的也看不出来**，换个 head_dim 不等于
> `hidden/n_heads` 的模型才会静默算错。本仓库因此把推导收敛成 `ReadModelMeta()` 一处
> （取值只认 `attention.key_length`），`model_info` 的 17 个键和 `GpuModel` 的布局都读它。

复用的文件在文件头保留原作者的版权与来源说明（同为 MIT，同作者）。

> **这些副本是"刻意保持逐字节一致"的**，所以它们的文件头仍然写着 `// RPGBarEngine —— …`，
> 不做任何改动。理由：
>
> 1. **对拍的前提**。§7 的对拍要能得出"差异只来自算力层"，前提就是文本层一个字都不差。
>    副本一旦被"顺手改好一点"，这个结论就没人能保证。
> 2. **同步的成本**。上游修了 bug，跟过来只需要
>    `git checkout upstream/main -- src/gguf.cc src/gguf.h` 一条命令；如果本地有改动，
>    每次同步都变成一次手工合并。
>
> 代价：在 RPGBarEngineCUDA 里读 `src/gguf.h`，第一行写的是另一个仓库的名字。这是**故意的**
> —— 它就是在告诉你这个文件从哪来、以及"别在这里改"。要确认没有跑偏，跑
> `python tests/check_sync.py`。

## 4. 关键设计决策

### 4.1 量化权重直传显存，不预先解量化

Q4_K_M 的 1.7B 权重是 **1.03 GiB**。如果加载时统一解成 fp16，会变成约 3.5 GiB；
解成 fp32 是 7 GiB —— 在 8 GB 的笔记本卡上直接出局。

所以：**权重按 GGUF 里的原始块布局拷到显存**，kernel 里 on-the-fly 反量化。
`Q4_K` 的块是 256 个权重一组，配 1 个 fp16 的 `d`、1 个 fp16 的 `dmin`、以及 12 字节
打包的 6-bit 量化尺度。反量化的公式在 CPU 版 `src/quant.cc` 里已经验证过，GPU 侧照着写，
**判据是"同一块权重，GPU 反量化结果与 CPU 逐元素一致"**（这一步可以完全对拍，
因为它是纯整数 + 定点运算，不涉及求和顺序）。

### 4.2 KV Cache 用 fp16

CPU 版用 fp32，@4096 上下文每会话 896 MiB。GPU 上用 fp16 → **448 MiB/会话**，
显存宽松得多（1.03 GiB 权重 + 几个会话的 KV + 中间张量，8 GB 绰绰有余）。

**但这会让"逐位对拍"从此不可能** —— fp16 写进 KV 的那一刻就有舍入。所以判据要改，
见 §7。

### 4.3 一个 stream，不做多流

decode 阶段每层要发十几个小 kernel（norm / QKV / rope / attention / O / 三个 FFN GEMM），
56 层下来是几百次启动。**在 CPU 版里"多线程"是为了喂满算力，在 GPU 版里"多流"反而会
引入同步开销** —— 每次 cross-stream 事件同步都是几十微秒。

所以：单 stream 串行提交。等到 P4 再考虑 CUDA Graph 把整段 decode 录成图，
把几百次 kernel 启动摊成一次（这是 decode 端最大的可省项之一）。

### 4.4 显存里只留一份权重，会话各持自己的 KV

与 CPU 版同一条不变量：**DM 与小助手共用权重、不共用上下文**。GPU 版里"权重"就是显存里
那块 1.03 GiB，"上下文"就是各自的 KV 缓冲。多会话 = 多份 KV + 一份权重。

## 5. 阶段与判据

**每个阶段都必须有一条能一键跑的判据**，没有判据的阶段不算完成。

| 阶段 | 内容 | 判据 |
| --- | --- | --- |
| **P0** 工具链与骨架 | CMake + CUDA，ABI 层最小实现（version / last_error / load_model / model_info） | 见下面「P0 的判据」 |
| **P1** 权重上卡 + 前向对拍 | 权重与反量化结果上显存；**整网（无 KV Cache）前向**跑通 | 见下面「P1 的判据」 |
| **P2** 增量解码对拍 | KV Cache 落地（fp16）、增量解码、贪心采样 | `T=0` 下与 CPU 版**逐 token 比较**（允许极少分歧，分歧必须记录位置与 logits 差距） |
| **P3** 接入与端到端 | 采样参数、工具调用、`rpg_engine.dll` 换进 RPGBar | `engine_probe.py`、`smoke_test.py`、Unity 侧 P/Invoke 全绿 |
| **P4** 性能 | 融合 kernel、量化直算调优、CUDA Graph | 1.7B Q4 上 prefill ≥ 1000 tok/s、decode ≥ 60 tok/s |

> **P1 的判据比"单层"更严，这是有意的**（2026-10-12 实施时定的）：原计划写的是"跑通一层、
> 单层输出 p99 ≤ 1e-3"，实际的判据是**整网 28 层的每一层残差流 + logits 全量对拍**。理由：
> ① 无 KV Cache 的整网前向里，"一层"和"28 层"是同一段代码跑 28 次 —— 单层过了整网还要再验一次；
> ② 逐层比较能把误差**定位到层**（第 16 层的 max 项开始冒头 ⇒ 那层的 attention/O 有问题），
> 只有一个最终 logits 就只能二分排查。
> ③ 真正需要新写的是 **KV Cache + 增量解码**（前缀复用、cache 布局、fp16 舍入），那才是 P2 的实体。
> ⇒ 这不是改了原设计，是把原设计里"单层"的判据做到了它的超集；**P2 的内容一条没少**。

**P0 的判据分两档**，因为"装没装 CUDA Toolkit"决定能跑到哪一档：

```bash
# ① 不需要 CUDA（没装 Toolkit 也能跑；CI 里就是这一档）
python tests/check_sync.py                 # 同源副本与上位仓库逐字节一致
python tests/abi_smoke.py --lib build/rpg_engine.dll
python tests/abi_smoke.py --lib build/rpg_engine.dll --model models/Qwen3-1.7B-Q4_K_M.gguf
#    ↑ host 层能 load 真实权重、model_info 的 17 个键齐全、generate 明确报"未编译 CUDA"

# ② 装了 CUDA Toolkit 之后
./build/cuda_check                         # 报出设备名/算力/显存 + 一次 device→host 往返
```

第 ① 档的意义：**"骨架能编、错误路径说人话"和"算力实现了没有"是两件事**，
前者不需要显卡就能验证完，不该被显卡卡住。第 ② 档才是"工具链通了"。

**P1 的判据也是两条，一键跑**：

```bash
# 在 WSL 里。前置：权重在 models/ 下、numpy 已装（apt install python3-numpy）
bash tests/run_p1.sh

# 拆开看（排错时用）
~/cuda-build/dequant_check models/Qwen3-1.7B-Q4_K_M.gguf --rows 32
bash tools/build_cpu_ref.sh ~/cuda-build          # 参考侧：现编 CPU 引擎的 forward_check
~/cuda-build/forward_check models/Qwen3-1.7B-Q4_K_M.gguf --prompt-file p.txt --out ~/p1-runs/ref
~/cuda-build/layer_check   models/Qwen3-1.7B-Q4_K_M.gguf --prompt-file p.txt --out ~/p1-runs/gpu
python tests/cmp_layers.py ~/p1-runs/gpu ~/p1-runs/ref --tol 1e-3
```

1. **反量化逐位对拍**（`tools/dequant_check.cu`）：设备端的 `DeqElem` vs 从 RPGBarEngine
   逐字复制的 `quant.cc::DequantizeRow`。判据 = **0 个 bit 不同**（反量化是纯整数拆包 +
   一次乘加，不涉浮点求和顺序）；外加 f16→f32 的 65536 个值全量穷举。
2. **整网逐层对拍**（`tools/layer_check.cu` + `tests/cmp_layers.py`）：GPU 与 CPU 引擎
   跑同一串 token、同一份 GGUF，比 **29 项残差流（embedding + 28 层）+ logits** 的相对误差，
   判据 = **p99 ≤ 1e-3**，且**每个位置的 argmax 一致**。统计口径与 CPU 引擎的
   `tools/cmp_npy.py` 完全一致（`REL_FLOOR=1e-3`、p99），两边的数字可以直接对照。

⚠️ **参考侧是 CPU 引擎自己的可执行文件（现编），不是本仓库里抄的一份 CPU 前向** ——
抄一份等于拿"我对 CPU 行为的理解"去对"我的 GPU 实现"，两边同源的理解错误会互相掩盖。
理由与具体编译命令见 `tools/build_cpu_ref.sh` 的头注释（顺带记录了"为什么不能直接用
RPGBarEngine 的 `build.sh`"：那是 MSYS2/MinGW 专用的）。

### 性能目标（RTX 5060 Laptop，8 GB，sm_120）

| 指标 | CPU 版实测 | GPU 目标 |
| --- | --- | --- |
| prefill | ~145 tok/s | ≥ 1000 tok/s |
| decode | 10–14 tok/s | ≥ 60 tok/s |
| DM 一轮（2811 prompt + 3 轮工具循环） | **211 s** | **< 2 s** |

## 6. 算子清单（GPU 侧要写的 kernel）

按依赖顺序，逐个都要有独立的对拍测试（✅ = P1 已实现并对拍过）：

1. ✅ **反量化 + 点积**（`quant_gpu.cuh` + `GpuMatMulQuant`）。⚠️ 实际权重里只用到
   **F32 / Q4_K / Q6_K** 三种（F32 全是 norm），所以只实现这三种，**别的类型在加载期
   直接拒**（`GpuModel::CheckGpuType`）—— 让不认识的量化块进 kernel，症状是"能出词但胡说"。
2. ✅ **RMSNorm**（`out = x * rsqrt(mean(x²) + eps) * w`）
3. ✅ **RoPE**（⚠️ **半维配对**，不是相邻配对 —— 写错不崩、只是胡说）
4. ✅ **Attention**（`softmax(QKᵀ/√d)V`，因果掩码靠"只遍历到 t"实现）。P1 版 q/k/v 全在
   显存里、无 cache；**从 KV Cache 读的那一版是 P2**
5. ✅ **SwiGLU**：`silu(gate) * up`，其中 silu 用 `x / (1 + exp(-x))`
6. ✅ **残差加 / 逐元素乘**
7. ✅ **LM head**（`hidden × vocab^T`，词表 151936 —— 这是 decode 里最宽的一次 GEMM）。
   ⚠️ 本权重 **tied_embeddings = true**（既没有 `output.weight` 也没有 `lm_head.weight`），
   LM head 复用 `token_embd`，**不重复上传第二份**（Q4 下也省 150 MiB）
8. ⏳ **采样**（P2）：temperature / top-k / top-p（softmax 在 GPU 上做，选出 token 再回传一个 int）

> ⚠️ **`top_p` 的排序是采样里最贵的部分**。CPU 版这里还是全量 `std::sort`（151936 个元素），
> GPU 版用 `thrust::nth_element` 或自己写的基数选择，顺手把 CPU 版那个待办一起解决。

## 7. 对拍策略（这一节比代码重要）

**因为 KV 用 fp16，"两边输出逐位相同"这个判据从 P1 起就不存在了。** 必须换一套：

1. ✅ **纯整数路径逐位对拍**（P1 已跑通）：反量化结果（§4.1）—— 不涉浮点求和顺序，
   可以要求完全一致。⚠️ 前提是 host 侧**用 `-ffp-contract=off` 编**：设备端用
   `__fmul_rn`/`__fsub_rn` 显式不做 FMA 收缩，host 侧若把 `d1 * raw - m1` 缩成 FMA，
   两边差 1 ULP，判据红一片 —— **看起来像拆包写错了，其实是编译选项**。
   （顺带：这条让两边更接近，不是更远 —— CPU 引擎在 Windows/MinGW 上不带 `-mfma`，本来也不收缩。）
2. ✅ **单算子容差对拍**（P1 已跑通，落在 §5 的第二条判据里）：每个 kernel 与 CPU 版
   同样输入，**相对误差 p99 ≤ 1e-3**。
   ⚠️ 定容差前**先量目标量级的 ULP** —— CPU 版踩过这个坑：Qwen3 的残差流有
   `|h| ≈ 1.2e4` 的离群维度，float32 的 1 ULP 就有 1e-3 量级，"绝对误差 < 1e-4"
   是**不可能满足**的判据。所以判据用**相对**误差 + 分位数（p99），并设
   `REL_FLOOR = 1e-3` 跳过"绝对值本来就接近 0"的那些元素。
3. ⏳ **端到端贪心逐 token**（P2）：`T=0` 时分布退化成尖刺，KV 的 fp16 舍入**多数情况下不改变
   argmax**。做法：两边同一 prompt 各生成 64 token，逐 token 比；一旦分歧，**打印分歧
   位置、两边各自的 top-5 logits**，然后看这个分歧是否在"两个 token 概率本来就接近"
   的合理范围内。
4. ⏳ **宿主层端到端**（P3）：`smoke_test.py` 全量 + `ui_check.js`，把 dll 换过去跑。

> **判据的参考侧一律是"真的在跑的那一份"**：反量化用从 RPGBarEngine 逐字复制的
> `quant.cc`，整网用**现场编译出来的** CPU 引擎 `forward_check`（`tools/build_cpu_ref.sh`）。
> 在本仓库里抄一份 CPU 前向会让"同源的理解错误"互相掩盖 —— 判据就废了。

## 8. 环境要求

| 项 | 要求 |
| --- | --- |
| GPU | 支持 CUDA 的 NVIDIA 卡；本项目开发机 = **RTX 5060 Laptop 8 GB（Blackwell，sm_120）** |
| 驱动 | ≥ 提供 CUDA 12.8 运行时的版本（sm_120 的硬性下限）；开发机 = 595.97 / CUDA 13.2 |
| 工具链 | **CUDA Toolkit 13.2**（`nvcc` V13.2.86）—— ✅ 已装在 **WSL（Ubuntu 24.04）** 里（`apt install cuda-toolkit-13-2`，走 NVIDIA 官方 apt 源：Ubuntu 自带的 `nvidia-cuda-toolkit` 只有 12.0、**不支持 sm_120**） |
| 编译器 | host 部分：MSVC 2022 或 g++ 13+；⚠️ **Windows 上 nvcc 只认 `cl.exe`**，而开发机没有 VS/MSVC |
| 判据脚本 | `tests/run_p1.sh` 需要 **numpy**（`apt install python3-numpy`，实测 1.26.4）；`check_sync.py` / `abi_smoke.py` 只要标准库 |
| 显存 | ≥ 4 GB。P1 实测（21 token）：权重 **1.03 GiB** + 前向暂存 14.8 MB，上卡后设备剩余 5.62 GiB；P2 起再加 fp16 KV |

### 为什么构建放在 WSL 里做（开发机的真实约束）

| | nvcc | cmake | g++ | MSVC | GPU 直通 |
| --- | --- | --- | --- | --- | --- |
| Windows 侧 | ✗ | ✗ | ✓（MSYS2） | **✗（整台机器没有 VS）** | — |
| WSL Ubuntu 24.04 | ✓ 13.2 | ✓ | ✓ | — | ✓ `/dev/dxg` + `nvidia-smi` 认卡 |

⇒ 在 Windows 上编 CUDA **必须同时装 MSVC**（nvcc 不接受 g++ 当宿主编译器），
而 WSL 只差一个 nvcc、`apt` 就能装。所以定成这样：**P0–P2 的算法验证走 WSL**
（产物 `librpg_engine.so`），**P3 出货时再补 MSVC + Windows 版 Toolkit** 出 `.dll`。

⚠️ WSL 通过 9p 访问 `E:\` 的顺序写只有 **173 MB/s**（原生 ext4 是 2.5 GB/s，差 14 倍）。
所以：源码只留一份（`/mnt/e`，git 管着，Windows 与 WSL 看到的是同一份，不会分叉），
**构建目录一律落 WSL 原生**（默认 `~/cuda-build`）。这就是 `build_wsl.sh` 存在的理由。

### 编译

```bash
# 在 WSL 里（脚本找不到 nvcc 会拒绝继续，并提示 host-only 路径）
bash build_wsl.sh                       # 产物 ~/cuda-build/librpg_engine.so
bash build_wsl.sh --clean
~/cuda-build/cuda_check                 # 工具链自检（不需要权重）

# 等价的手工步骤
cmake -S . -B ~/cuda-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build ~/cuda-build -j
```

`-DCMAKE_CUDA_ARCHITECTURES=native` 会按本机 GPU 定架构；跨机器分发时要显式列
（如 `-DCMAKE_CUDA_ARCHITECTURES=86;89;120`）—— ⚠️ 但**别用 `native` 编出来的产物发给
别人**，在别的卡上会"加载即失败"。

### 首次编译的实测结果（2026-10-12，P0 两档判据全绿）

```
设备：NVIDIA GeForce RTX 5060 Laptop GPU | 算力 12.0 | 26 SM | 显存 6.9 / 8.0 GiB
    | CUDA 驱动 13.2 / 运行时 13.2
device 计算 + 搬运：1024 个 float 全部正确
```

**首次编译抓出两个真问题** —— 都是 host-only 构建**永远看不到**的（`RPG_HAS_CUDA`
未定义时相关代码整段被 `#ifdef` 掉）：

1. `cuda_common.h` 把 `#include <cuda_runtime.h>` 放在了 `namespace rpg {` **里面** ——
   libstdc++ 的 `namespace std` 被嵌套成 `rpg::std`，此后该 TU 里每个 `std::` 名字都解析
   不到，而报错点全落在标准库内部（`'vector' in namespace 'rpg::std' does not name a
   template type`）。**修法**：include 移到 namespace 之前，并改用 `cuda_runtime_api.h`
   （host 侧只需要 `cudaError_t` 与内存 API；完整头会带进 `__host__`/`__device__` 那套宏）。
2. `CMakeLists.txt` 的 `cuda_check` 目标漏了 `src/cuda_common.cc` —— 链接期报
   `undefined reference to rpg::DescribeDevice`。

⇒ **通用教训**：当 `#ifdef` 把一条构建路径整段藏起来时，"另一条路径编得过"**不能**
推出"这条路径编得过"。这就是 P0 判据必须分两档、且第二档要真跑一次的原因 ——
别把"没报错的推测"当成"已验证的结论"。

### P1 实测结果（2026-10-12，两条判据全绿）

```
① 反量化逐位对拍（Q6_K token_embd + Q4_K attn_k，各 32 行）
   131072 个元素，**0 个 bit 不同**；f16→f32 穷举 65536 个值全对

② 整网逐层对拍（21 token prefill，同一份 GGUF）
   残差流 29 项：最差 p99 = 1.05e-4（判据 1e-3）
   logits：p99 = 5.98e-5，max|Δ| = 1.43e-4
   argmax：21/21 个位置一致，最后位置 top-5 数值逐个相同

性能（同一 prompt、CPU 12 线程）
   权重上卡 4.9 s（1.03 GiB，含从 /mnt/e 经 9p 读盘）
   GPU prefill 21 token：213.7 ms（98.3 tok/s）
   CPU 同一份：3081.1 ms  ⇒  **14.4x**
```

> 98.3 tok/s 离 P4 的 1000 tok/s 目标还远，**但现在不该优化**：这个数字里大头是
> "21 个 token 的 GEMM 每个都是 `1×2048` 乘 `2048×6144`"—— 小到吃不满 26 个 SM。
> P4 的 prefill 目标要在 ≥512 token 的 prompt 上量。**先把 P2 的对拍做对，再谈性能。**

**P1 首次真编译又抓出几个坑**（同样是 host-only 构建看不到的）：

1. 🔴 **`CMAKE_CUDA_STANDARD` 默认 17 → `std::bit_cast` 不存在**。同源副本 `dtypes.h`
   用了它（`#include <bit>`），nvcc 在 C++17 下报 `namespace "std" has no member "bit_cast"`，
   **报错点在标准库里、看着像环境问题**。同源副本一个字都不能改（§3），所以只能把
   CUDA 标准抬到 20：`set(CMAKE_CUDA_STANDARD 20)`。
2. ⚠️ **`add_compile_options(-O2)` 会同时落到 CUDA 上**，Release 下 CMake 已经给 nvcc
   传了 `-O3`（`CMAKE_CUDA_FLAGS_RELEASE`）⇒ nvcc 报
   `incompatible redefinition for option 'optimize'` 并且**实际以 -O2 生效**。
   一次"编译通过但把 CUDA 侧悄悄降档"的坑。修法：`$<$<COMPILE_LANGUAGE:CXX>:-O2>`。
3. **静态成员函数里的 lambda 不能捕 `this`**（`GpuModel::Load` 是 `static`）——
   `error: "this" may only be used inside a nonstatic member function`。改成捕 `&m`。
4. `gpu_model.h` 少了 `#include "quant.h"`（`kGgmlF32` / `TypeName` / `TypeRowBytes` 都在那），
   `tools/dequant_check.cu` 少了 `#include "kernels_gpu.h"`。

⇒ 与 P0 同一条教训的**第二次应验**：这两条都是"另一条构建路径编得过"推不出来的。
所以判据脚本 `tests/run_p1.sh` 从第一天起就是**一键、两条一起跑**，不给人"挑着跑"的机会。

## 9. 已知风险

1. **sm_120 是新架构**。CUDA 12.8 之前根本没有它的 target；工具链版本选错会得到
   "no kernel image is available for execution" —— **不是代码错，是架构没编进去**。
2. **显存够但不宽裕**：8 GB 里还要和桌面/浏览器分。跑 4B（约 2.5 GiB）时要把 KV 压到
   2048 或减少并发会话。
3. **笔记本卡的持续性能**：短时的峰值与长时间生成的热降频不是一回事，性能报表要注明
   跑了多久、温度多少。
4. **`quant.cc` 是 host 侧复用的**，GPU 侧反量化必须**另写**并独立对拍（§7.1），
   不能"看着公式一样"就跳过。

## 10. 与 RPGBarEngine 的同步

平台无关层是副本，上游的修复不会自动过来。约定：

```bash
git remote add upstream https://github.com/RaganaBloodedge/RPGBarEngine.git
git fetch upstream
# 只挑平台无关层的提交
git cherry-pick <sha>          # 或 git checkout upstream/main -- src/gguf.cc src/gguf.h
```

**跟之前先跑一遍，跟之后再跑一遍**：

```bash
python tests/check_sync.py     # 22 个同源副本（20 src + 2 tools）+ 2 个 ABI 文件 = 24 项，逐字节比较
python tests/check_sync.py --quiet   # 只看结论，适合放进脚本
```

它存在的意义是：**"复制得对不对"不是一次性动作，而是一条随时可以重跑的判据**。
对拍（§7）要能得出"差异只来自算力层"，前提就是这一层没跑偏。

**判断某个上游改动要不要跟**：看它动的是"文本/容器"还是"计算"。
- 动 `gguf.*` / `tokenizer.*` / `chat_template.*` / `tool_parser.*` → **要跟**（两边行为必须一致）
- 动 `matmul*` / `qwen3.*` / `kernels.h` / `sampler.cc` → **不跟**（这里本来就是重写的）
- 动 `include/rpg_engine.h`（ABI） → **两边同时改**，且是破坏性变更，要升版本号

反向也一样：这个仓库里对 ABI 的修正要回流到 CPU 仓库。

> ⚠️ **想在本仓库里"顺手改好"某个副本时**，先停一下：那等于把它从"复用"栏挪到"重写"栏，
> 要同时改 §3 的表格，并说明为什么不能回流上游。**不要偷偷改** —— `check_sync.py` 会立刻
> 报出分叉，但报出来之后需要有人做这个决定。
