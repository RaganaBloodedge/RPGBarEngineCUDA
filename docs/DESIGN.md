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
| 权重 | `model.*` | **重写** | 权重从 mmap 变成**常驻显存**，布局也要按 kernel 重排 |
| KV Cache | `kv_cache.*` | **重写** | 放到显存，fp16 |
| 前向 | `qwen3.*` `matmul*` `kernels.h` | **重写** | 这是全部工作量所在 |
| 采样 | `sampler.cc` | **部分重写** | top-k/top-p 的排序在 GPU 上做（或用 `thrust`） |
| ABI 层 | `engine.cc` | **重写** | 句柄表、会话管理、JSON 编解码保持形状，但内部对象换成 GPU 侧 |

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
| **P1** 权重上卡 + 单层对拍 | 权重与反量化结果上显存；跑通**一层**前向 | 与 CPU 版同一 prompt、同一权重，单层输出**相对误差 p99 ≤ 1e-3** |
| **P2** 整网生成对拍 | KV Cache 落地、整网前向、贪心解码 | `T=0` 下与 CPU 版**逐 token 比较**（允许极少分歧，分歧必须记录位置与 logits 差距） |
| **P3** 接入与端到端 | 采样参数、工具调用、`rpg_engine.dll` 换进 RPGBar | `engine_probe.py`、`smoke_test.py`、Unity 侧 P/Invoke 全绿 |
| **P4** 性能 | 融合 kernel、量化直算调优、CUDA Graph | 1.7B Q4 上 prefill ≥ 1000 tok/s、decode ≥ 60 tok/s |

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

### 性能目标（RTX 5060 Laptop，8 GB，sm_120）

| 指标 | CPU 版实测 | GPU 目标 |
| --- | --- | --- |
| prefill | ~145 tok/s | ≥ 1000 tok/s |
| decode | 10–14 tok/s | ≥ 60 tok/s |
| DM 一轮（2811 prompt + 3 轮工具循环） | **211 s** | **< 2 s** |

## 6. 算子清单（GPU 侧要写的 kernel）

按依赖顺序，逐个都要有独立的对拍测试：

1. **反量化 + 点积**（`Q4_K` / `Q6_K` / `Q8_0` firmware 版本，先做 `Q4_K`）
2. **RMSNorm**（`out = x * rsqrt(mean(x²) + eps) * w`）
3. **RoPE**（⚠️ **半维配对**，不是相邻配对 —— 写错不崩、只是胡说）
4. **Attention**：`softmax(QKᵀ/√d + mask)V`，k/v 从显存 KV Cache 读
5. **SwiGLU**：`silu(gate) * up`，其中 silu 用 `x / (1 + exp(-x))`
6. **残差加 / 逐元素乘**
7. **LM head**（`hidden × vocab^T`，词表 151936 —— 这是 decode 里最宽的一次 GEMM）
8. **采样**：temperature / top-k / top-p（softmax 在 GPU 上做，选出 token 再回传一个 int）

> ⚠️ **`top_p` 的排序是采样里最贵的部分**。CPU 版这里还是全量 `std::sort`（151936 个元素），
> GPU 版用 `thrust::nth_element` 或自己写的基数选择，顺手把 CPU 版那个待办一起解决。

## 7. 对拍策略（这一节比代码重要）

**因为 KV 用 fp16，"两边输出逐位相同"这个判据从 P1 起就不存在了。** 必须换一套：

1. **纯整数路径逐位对拍**：反量化结果（§4.1）—— 不涉浮点求和顺序，可以要求完全一致。
2. **单算子容差对拍**：每个 kernel 与 CPU 版同样输入，**相对误差 p99 ≤ 1e-3**。
   ⚠️ 定容差前**先量目标量级的 ULP** —— CPU 版踩过这个坑：Qwen3 的残差流有
   `|h| ≈ 1.2e4` 的离群维度，float32 的 1 ULP 就有 1e-3 量级，"绝对误差 < 1e-4"
   是**不可能满足**的判据。
3. **端到端贪心逐 token**：`T=0` 时分布退化成尖刺，KV 的 fp16 舍入**多数情况下不改变
   argmax**。做法：两边同一 prompt 各生成 64 token，逐 token 比；一旦分歧，**打印分歧
   位置、两边各自的 top-5 logits**，然后看这个分歧是否在"两个 token 概率本来就接近"
   的合理范围内。
4. **宿主层端到端**：`smoke_test.py` 全量 + `ui_check.js`，把 dll 换过去跑。

## 8. 环境要求

| 项 | 要求 |
| --- | --- |
| GPU | 支持 CUDA 的 NVIDIA 卡；本项目开发机 = **RTX 5060 Laptop 8 GB（Blackwell，sm_120）** |
| 驱动 | ≥ 提供 CUDA 12.8 运行时的版本（sm_120 的硬性下限） |
| 工具链 | **CUDA Toolkit（`nvcc`）** —— ⚠️ 开发机上当前**未安装**，这是 P0 的前置条件 |
| 编译器 | MSVC 2022 或 g++ 13+（host 部分） |
| 显存 | ≥ 4 GB（1.03 GiB 权重 + 448 MiB/会话 KV + 中间张量） |

### 编译

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# 工具链自检（不需要权重）
./build/cuda_check
```

`-DCMAKE_CUDA_ARCHITECTURES=native` 会按本机 GPU 定架构；跨机器分发时要显式列
（如 `-DCMAKE_CUDA_ARCHITECTURES=86;89;120`）—— ⚠️ 但**别用 `native` 编出来的 dll 发给
别人**，在别的卡上会"加载即失败"。

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
python tests/check_sync.py     # 22 个同源副本 + 2 个 ABI 文件，逐字节比较
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
