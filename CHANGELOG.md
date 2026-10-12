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

### 新增（P2：KV Cache（fp16）+ 增量解码 + 贪心对拍）

- **`src/kv_cache_gpu.{h,cu}`**：显存里的 KV Cache，元素 **fp16**，维度顺序与 CPU 版
  一个字不改（层 → KV 头 → 位置 → 维度）。@4096 上下文从 896 MiB 降到 **448 MiB**。
- **三个新算子**（`src/kernels_gpu.{h,cu}`）：
  - `GpuKvWrite`：本段的 `[t][head][dim]` → cache 的 `[head][pos][dim]`（**布局不同，
    不是 memcpy**），一次 fp32→fp16；
  - `GpuAttentionFromCache`：从 cache 读历史的因果注意力，遍历上界是**绝对位置** `pos0 + t`；
  - `GpuArgMax`：设备端归约，只回传 4 个字节（logits 是 151936 个 float = 607 KB，
    取一个下标却回拷它是纯浪费）；**并列取最小下标**，与 CPU 版 `ArgMax` 同语义。
- **`src/qwen3_gpu.{h,cu}`**：`GpuForwardLayer` 加 `pos0` 与 `kv`（RoPE 用绝对位置；
  kv 非空时先写 cache、再从 cache 读注意力）；`GpuForwardFull` 加 `kv` 参数（与 CPU 版
  同语义：pos0 = cache 当前长度、**所有层写完后才推进 len**）；新增 `GpuForwardStep`
  与 `GpuGenerateGreedy`。⚠️ 增量解码不是"另一份实现"—— `GpuForwardStep` 就是
  `GpuForwardFull(seq = 1, kv)`，与 CPU 版一模一样；两条路径共用同一份 `GpuForwardLayer`
  正是"带 cache 与不带 cache 只该差一个 fp16 舍入"这句话能成立的原因。
- **判据工具与脚本**：
  - `tools/kv_map_check.cu`：KV 写入的下标映射与边界。**不需要模型、不需要权重** ——
    值取 fp16 能精确表示的整数（往返无损 ⇒ 判据是**逐位相同**），另加一条"没写过的
    区间必须仍是 0"（越界写到邻居不报错，只会悄悄改掉它）。它把"布局错"与"精度不够"
    彻底分开，所以放在判据脚本的最前面。
  - `tools/gen_check.cu`：增量解码 + 贪心生成 + **KV 自检**（带 cache 的 prefill 与
    不带 cache 的全序列前向比 logits），写 `.gen.txt` 与可选的 `.step_logits.npy`。
  - `tools/cpu_ref_gen.cc`：**参考侧的 main**（实现在 CPU 引擎里）—— 它
    `#include "qwen3.h"` 直接用 CPU 引擎的 `Model` / `ForwardFull` / `ForwardStep` /
    `GenerateGreedy` / `ArgMax`，**没有一行前向或解码逻辑是抄的**。生成用引擎自己的
    `GenerateGreedy`（那才是判据对象），每步 logits 靠"**重放**"收集，并顺带校验
    重放的 argmax 等于生成时的 token（引擎是纯函数，这条本该成立）。
  - `tests/cmp_gen.py`（逐 token 对拍：数值层 + token 层；分歧处打印两边 top-5 与
    两个 token 的 logit 差）、`tests/run_p2.sh`（P2 一键判据，三条）。
- `build_cpu_ref.sh` 增加第二个产物 `cpu_ref_gen`（同一个 `.o` 集合，**只换 main**）。

### 说明（P2）

- **P2 的三条判据全绿**（2026-10-12，WSL + CUDA 13.2 + RTX 5060 Laptop）：
  - ⓿ KV 映射：写入区间 120 个元素**逐位正确**、未写入区间保持为 0；
  - ① KV 自检：满量程误差 **4.09e-4**（判据 1e-3），逐位置 argmax **21/21** 一致；
  - ② 逐 token 对拍：**32/32 完全一致**，数值层满量程 6.48e-4，cache 长度不变量两侧成立。
- ⚠️ **判据口径从"逐元素相对误差 p99"换成了"绝对误差 / 满量程"**。KV 过 fp16 之后，
  误差的**绝对**量级由舍入决定（稳定在 ~1e-2 上下），而相对误差 = 它 ÷ 元素自身的值 ——
  于是 p99 完全由"有多少元素落在 0.5 附近"决定，换个 prompt 就从 1.7e-2 漂到 5.2e-2
  （同一份实现、同一份权重）。新口径有物理依据：fp16 尾数 10 位 ⇒ 1 ULP = 2⁻¹¹ = 4.88e-4，
  判据取满量程的 **1e-3**（约 2 个 ULP）。两点标定：**seq=1 → 8.97e-5（0.18 ULP）、
  seq=21 → 5.6e-4（1.15 ULP）**，比值随长度按 **√n** 增长 —— 正是独立随机舍入误差
  累加的标度律（布局错会是 1e0 量级且不随 n 平滑变化）。
- **性能**（21 token prompt + 生成 32 个 token，同一份权重）：
  prefill GPU **228.7 ms** vs CPU 3222.1 ms ⇒ **14.1×**；
  decode GPU **1728.9 ms（18.5 tok/s）** vs CPU 2609.7 ms（12.3 tok/s）⇒ **1.5×**；
  KV Cache(fp16)：**28 MiB @256 位置 / 448 MiB @4096**。
  ⚠️ decode 只有 1.5 倍是**预期内**的：每步只算 1 个 token，却要发 28 层 × 十几个
  kernel ≈ **400 次启动** + 一次 argmax 回传同步 —— 瓶颈已经从"算力"变成"启动开销
  + 同步"，那正是 P4 的 CUDA Graph 要解决的（DESIGN §4.3）。**这个数字是 P4 的起点。**
- **P2 首次真编译抓出 5 个问题**（host-only 构建看不到 —— 没定义 `RPG_HAS_CUDA` 时
  `tools/` 下的判据工具根本不会被创建）。最值得记的是：`GpuForwardFull` 里
  `logits == nullptr` 的早退**差点写在 lm_head 之前** —— 那样 decode 循环里的 logits
  缓冲永远是上一轮的垃圾，**不崩，只是第一个 token 对、从第二个起全错**（这种形态最
  容易被归因成"小模型生成质量差"而不是 bug）。另有：改了 `GpuArgMax` 的签名没改头文件
  声明（报错在调用方）、`cpu_ref_gen.cc` 用了 `rpg::ArgMax` 但 `qwen3.h` 不转出
  `kernels.h`、**nvcc 不接受 `void*` → `const float*` 的隐式转换（g++ 接受，平台差异）**、
  注释里行尾的 shell 续行 `\` 触发 `-Wcomment`。全部修掉，教训写进 DESIGN §8。
- **ABI 行为仍然没有变化**：`generate` 依旧明确报错（文案更新为"生成还没接进 ABI ——
  P2 已完成整网前向 + KV Cache + 增量解码，把显存状态挂进 session 是 P3 的事"），
  `include/rpg_engine.h` 一个字节未动。`Session` 到现在还没有显存状态，`session_reset`
  仍是空操作（不假装清了个东西）—— 这与上面 P1 那条说明里"上卡接进 ABI 是 P2 的事"
  的说法不同：**真做下去才发现，P2 的全部工作量在"把增量解码做对"，接 ABI 是 P3 的
  实体**（DESIGN §5 的阶段表本来也是这么分的）。

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
