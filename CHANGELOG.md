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

### 新增（P3 第一步：设备端采样）

- **`src/sampler.h`**：从 RPGBarEngine **逐字复制**的同源副本 —— 采样参数的字段 /
  默认值 / 语义（HF 同款的 top-k 并列规则、top-p 至少留一个 token、`temperature <= 0`
  即贪心且**不消耗随机数**）是**契约**，抄成两份迟早会漂。只有头文件是副本：
  `sampler.cc` 依赖 CPU 侧的 `kernels.h`，搬过来等于把 CPU 前向也搬进来（那是"重写"
  那一栏的事）。副本数 22 → 23，连 2 个 ABI 文件共 **25 项**由 `check_sync.py` 盯着。
- **`src/sampler_gpu.{h,cu}`**：设备端采样（重写，不与上游同步）。
  - `GpuFilterToProbs`：温度缩放 → top_k → softmax → top_p，与 CPU 版
    `FilterToProbs` 同语义。**两条筛选都不排序**：top_k 找"第 k 大值"
    （谓词 `|{key ≥ t}| ≥ k`）、top_p 找"累积概率的切点"（谓词 `Σ{p ≤ t} ≤ 1 - top_p`），
    各 32 次迭代遍历一遍词表（每次一个块内归约）。CPU 版那边是
    `nth_element` + 全量 `std::sort`（151936 个元素，`FilterToProbs` 11 ms 里排序占
    10.25 ms）—— 这条顺手把它一起解决了。
  - `GpuSampleToken`：贪心（`temperature <= 0`）退化为 argmax 且**不消耗随机数**
    （与 CPU 版同序，保住"不传 temperature 的老路径逐位不变"）；否则用
    `u = CounterUniform(seed, 绝对位置)` 在筛选后的分布上做逆变换。
  - `GpuInverseSample`：**一个线程、按下标顺序**串行累加。并行扫描会改掉求和顺序，
    于是 u 落在累积概率边界上时会与 CPU 选到不同的 token —— 而"同一份 logits +
    同一个 seed ⇒ 同一个 token"正是判据本身。代价是词表 151936 时 ~0.5 ms 的串行链
    （decode 一步 54 ms，占比 ~1%）。
  - `DevSplitMix64` / `DevCounterRng` / `DevCounterUniform`：与 CPU 版逐字对应
    （三行整数运算 + 乘一个 2 的幂 ⇒ 两边判据是**逐位相同**）。
- **判据工具与脚本**（都**不需要模型、不需要权重**，几秒跑完）：
  `tools/sample_check.cu`（写出两边共用的用例矩阵 `cases.tsv` 与 `logits.bin`，
  跑 GPU 侧）、`tools/cpu_ref_sample.cc`（参考侧：只有 main，实现用 CPU 引擎的
  `sampler.cc`，由 `build_cpu_ref.sh` 现编 —— 它多了第 4 个产物 `cpu_ref_sample`）、
  `tests/cmp_sample.py`（判据：u 表逐位 / 概率满量程 / 保留集合 / token）、
  `tests/run_p3.sh`（一键，词表 4096 与 151936 各跑一遍）。
- `GpuScratch` 加 `probs` 缓冲（长度 = **vocab**，与 seq 无关 —— 一步只要一行分布）。

### 说明（P3）

- **P3 第一步的三条判据全绿**（2026-10-12，WSL + CUDA 13.2 + RTX 5060 Laptop；
  12 条用例 × 每例 16 个 u）：
  - u 表（`CounterUniform`）：**192/192 位模式相同**（两个词表各一遍）；
  - 概率分布：满量程误差最大 **2.31e-7**（词表 4096）/ **1.66e-7**（151936），
    判据 1e-5，且**保留集合逐例一致**（哪些 token 没被 top_k/top_p 筛掉是离散判断）；
  - token：**192/192 逐位相同**。
  - 词表 151936 那一遍专门用来验二分搜索在**整个 uint32 key 空间**上收敛
    （小词表测不出"上中点溢出"那类 bug）。
- ⚠️ **"采样不能逐 token 对拍"这句话要加一个限定**：它针对的是**接在模型后面**的采样
  （两边 logits 先差了 fp16 的舍入，采样再把差异放大成"整段序列不同"）。把模型摘掉、
  两边喂同一份从文件读的 logits 之后，判据立刻回来，而且是**逐位**的。
  这与 P2 用 `kv_map_check` 把"cache 布局"与"fp16 精度"分开是同一个套路：
  **先把变量的个数降到 1，再判。**
- 两个**在设计上避开**的陷阱（不是踩到才发现的）：
  ① 两级二分的上中点必须用 `uint64` 算 —— `lo + (hi - lo + 1) / 2` 在
  `lo = 0 / hi = 0xFFFFFFFF` 时 `hi - lo + 1` 溢出成 0 ⇒ 第一次迭代就 `mid == lo`
  ⇒ **死循环**（只有词表大到覆盖半个 key 空间才可能踩到）；
  ② `top_p` 的并列是**未定义**的 —— CPU 在概率精确相等的元素之间取决于 `std::sort`
  的同值顺序，GPU 的阈值法会把并列的整撮一起剔除。fp32 softmax 的输出里精确并列
  几乎不出现，判据的测试数据也刻意避开（唯一的并列用例只开 `top_k`，那条是良定义的：
  "阈值取第 k 大值、并列一起保留"）。
- 顺带：**P3 这一步一次编译通过，没有新的"编得过但错了"的坑** —— 前两个阶段抓到的
  那些（CUDA 标准版本、`-O2` 落到 nvcc、include 不能进 namespace、nvcc 不接受
  `void*` 隐式转换、注释行尾的 `\`）都写在设计文档与文件头注释里，写新代码时照着绕开。
- **ABI 行为仍然没有任何变化**：`include/rpg_engine.h` 一个字节未动，`generate` 的
  报错文案与 P2 时相同（采样要真正生效，得等它接进生成循环与 `Session`）。

### 新增（P3 第二步：采样接进生成循环）

- **`GpuGenerate`**（`src/qwen3_gpu.{h,cu}`）：`GpuGenerateGreedy` 升级成通用的
  `GpuGenerate`，`GpuGenerateOptions` 加 `SampleOptions sample` 与 `uint64_t seed`。
  - **`GpuGenerateGreedy` 保留为薄包装**（与 CPU 版 `GenerateGreedy` 同构：
    拷一份 options、把 `temperature` 置 0、调 `GpuGenerate`），**不是第二份循环** ——
    两份只差一行 argmax/sample，但"哪个分支忘了处理停止 token"这种漂移迟早会发生。
    P2 的调用方（`tools/gen_check`、`tests/run_p2.sh`）一行都不用改。
  - `counter` 取 `kv->len()`（**绝对位置**，即这个 token 将要落进 cache 的下标），
    与 CPU 版逐字一致 —— 用绝对下标才能让"暖缓存（续写）与冷缓存（从头）拿到同一个 u"。
  - 默认 `temperature = 0` ⇒ **不传采样参数的老请求行为逐位不变**（这条由
    `run_p2.sh` 的逐 token 对拍继续盯着，实测满量程比值与 P2 记录的数字一模一样）。
  - 贪心分支走 `GpuArgMax`（并列取最小下标）——与 P2 是同一个 kernel、同一条路径，
    所以"贪心没变"是**由构造保证**的，不是靠测出来的。
- **`tools/gen_sample_check.cu`**：新判据工具（**要权重**），验采样接进 decode 之后的
  不变量 —— 同 seed 两遍逐位相同（引擎仍是纯函数）、换 seed 序列就变、
  `T=0` 加筛选参数与贪心逐位相同、`top_k=2` 时每步都落在该步前 2 名内、
  以及 **counter 用绝对位置**（输出接在 prompt 后冷跑一遍 ⇒ 整体错一位逐位相同）。
- **`tests/run_p3_gen.sh`**：一键跑上面这条判据。

### 说明（P3 第二步）

- **五条不变量全绿**（2026-10-12，WSL + CUDA 13.2 + RTX 5060 Laptop，21 token prompt、
  生成 32 个）：B **32/32 逐位相同**、C 换 seed **31/32 处不同**、D1 逐位相同、
  D2 **32/32 落在前 2 名内**、D3 **0/32 相同**、E **31/31 逐位相同**。
  回归：`run_p2.sh` 逐 token 仍 **32/32**、满量程仍 **6.48e-4**；
  `run_p1.sh` 最差 p99 仍 **1.05e-4**；`check_sync.py` 25 一致 / 0 分叉；
  `abi_smoke.py` 32 通过 / 0 失败。
- **这一层的判据为什么不能沿用 P3 第一步**：这里的 logits 来自模型，两边先差了
  fp16 KV 的舍入，采样把那点差异放大成"整段序列不同"。所以换成一批**只看自己**的
  不变量 —— **严格的**（同 seed 两遍必须逐位、`T=0` 必须与贪心逐位、
  暖/冷缓存的随机流必须对齐）与**概率性的**（换 seed 会变、剪枝会改选）分开标注，
  不混成"差不多就行"。
- ⚠️ **踩到一个"判据前提"的坑**（不是代码 bug）：第一版拿 `top_k = 1` 当
  "支持集只剩一个元素 ⇒ 必然退化成 argmax"来断言，**红了 32 处**。契约里 top_k 的
  生效条件是 **`top_k > 1 && top_k < n`** ⇒ **`1` 与 `0` 一样算"关闭"**
  （注释只写了 "0 = 关闭"）。**判据的前提也要去读契约，不能凭常识。**
  `sampler.h` 是同源副本、不许改，所以只记录，不动上游。
- **一个设计决定如果能写成一个便宜的断言，就一定要写**：`counter` 用绝对位置这件事
  原本只活在注释里，现在有了 E —— 谁把它改成"本轮第几个"都会立刻红。
- 性能数据（P4 的起点）：decode 32 token，`T=0` **1932.9 ms（16.6 tok/s）**、
  `T=1` 真采样 **2141.7 ms（14.9 tok/s）**。采样加的是"几步 kernel"，不是"几百步"——
  decode 的主因仍是每步约 400 次 kernel 启动 + 一次同步。
- **ABI 行为仍然没有任何变化**：`include/rpg_engine.h` 一个字节未动，`generate` 的
  报错文案也不变（`Session` 里还没有显存句柄 —— 那是 P3 的下一步）。

### 新增（P3 第三步：接入 ABI）

- **`rpg_engine_generate` 真的实现了**（`src/engine.cc`，370 → 约 590 行）：逐条对齐
  CPU 版 `src/engine.cc` 的九步 —— 解析请求 → 渲染 ChatML + 分词 → 与 `cached_ids`
  对最长公共前缀（`lcp == total` 时回退一位）→ 生成长度与容量检查（**会失败的条件
  全部前置**，之后才动 cache）→ prefill → 停止 token（`eos` + `<|im_end|>` +
  请求/会话的 stop 串）→ 生成 → 解码（`skip_special=false`）→ 拼响应 JSON
  （`text` / `tool_calls` / `finish_reason` / `usage`）。
  **`include/rpg_engine.h` 一个字节没改**：宿主（Python / Unity）不需要改任何代码。
- **会话持显存状态**：`ModelEntry` 加 `std::unique_ptr<GpuModel>`（权重上卡，量化字节
  原样上传 1.03 GiB）；`Session` 加 `std::unique_ptr<GpuKvCache>` + `GpuScratch` +
  `cached_ids`。KV 在 **`session_create` 时分配**（显存不够要在建会话时报出来，而不是
  生成到一半炸在 kernel 里）；`session_reset` 只归零长度、保留显存。
- **`load_model` 的降级路径明确化**：设备完全不可用时只读元数据（P0 起就有的行为 ——
  CI 与无卡机器要能 load、能读 `model_info`），此时 `ModelEntry::gpu == nullptr`，
  建会话与生成各自报**不同**的错；但**设备可用而上传失败必须抛**（OOM、未知量化类型）。
- **`rpg_engine_model_info` 多一项 `device_bytes`**（权重在显存里的实际占用）。
  设备不可用时该键**不出现**，而不是报 0 —— 0 会被读成"权重不占显存"，那是另一回事。
- **`engine.cc` 新增三个辅助函数**（`AppendStopString` / `AppendStopStrings` / `ContainsId`），
  语义与 CPU 版逐字相同：停止串必须能变成单个 token，否则记警告并忽略（引擎按 token
  粒度停，多 token 的文本级回看是宿主的职责）。
- **`tests/abi_smoke.py`**：P0 时那条"generate 必须明确报错"的断言**换掉了** ——
  改成验响应的形状（返回非空 / 合法 JSON / 四个键 / `finish_reason` 取值 / `usage` 字段）。
  从 32 通过变成 **38 通过 / 0 失败**。

### 说明（P3 第三步）

- **端到端真的出文本了**（2026-10-13，WSL + CUDA 13.2 + RTX 5060 Laptop）：

  ```text
  问：用一句话介绍你自己。
  答：我是一个AI助手，可以回答问题、提供信息和协助完成各种任务。  finish=stop
      usage={'prompt_tokens': 17, 'completion_tokens': 18}

  同会话第 2 轮（验前缀复用）：
  答：我刚才说的是一句话，介绍我自己，说我是人工智能助手。
      usage={'prompt_tokens': 37, 'completion_tokens': 16}
  ```

- **资源实测**：权重上卡 **1.05 GiB**（量化字节 1.03 GiB）/ 上传 **6.6–8.9 s**
  （从 `/mnt/e` 经 9p 读盘）/ 设备剩余 **5.60 GiB**；会话 KV：512 容量 **56 MiB**、
  1024 容量 **112 MiB**。
- **与 CPU 版唯一的实质差异是"少做事"**：CPU 版 prefill 后要把 `rows × vocab` 的 logits
  下载回 host 再取最后一行；这里传 `logits = nullptr`，`GpuGenerate` 直接吃显存地址 ——
  **整个 decode 循环只回传 4 个字节**。
- **回归全绿、数字一字不差**：`run_p3` 192/192 逐位、`run_p2` 32/32 逐 token + 满量程
  **6.48e-4**、`run_p3_gen` A–E 全通过（E 31/31）、`run_p1` 最差 p99 **1.05e-4**、
  `check_sync.py` 25 一致 / 0 分叉。**ABI 层的改动没有碰算力层。**
- ⚠️ **P3 只剩一项没做完**：Windows 侧能加载的 `.dll`。nvcc 在 Windows 上只认 `cl.exe`，
  这台机器没有 VS/MSVC ⇒ `models/rpg_engine.dll` 仍是 CPU 版，RPGBar 的 `smoke_test.py` /
  `ui_check.js` 还没在 CUDA 引擎上跑过。**卡住的是编译器，不是接口。**
  （→ 2026-10-13 补完，见下面「P3 第四步」。）

### 新增（P3 第四步：Windows 产物 + 接入 RPGBar）

- **`build_win.ps1`**：Windows 侧的构建脚本（`build_wsl.sh` 的对应物）。
  刻意**不调 vcvars** —— 直接从文件系统推导 `PATH` / `INCLUDE` / `LIB`，全程不碰注册表；
  显式传 `-DCMAKE_CUDA_COMPILER_TOOLKIT_ROOT` 等，让 `find_package(CUDAToolkit)` 不必去查注册表。
  **纯 ASCII**（PowerShell 5.1 无 BOM 时按 ANSI 读 `.ps1`，中文注释会吃掉行首字符）。
- **默认架构改成 `75;86;89;120`**（Turing / Ampere / Ada / Blackwell）：`native` 只出本机 SASS，
  产物发给别人是"选项亮着、一生成就失败"。`cuobjdump --list-elf` 确认四份 SASS 齐全；
  产物体积 558,080 B（单 sm_120）→ **1,061,376 B**（四架构）。
- **`CMakeLists.txt` 的 MSVC 分支修掉一处漏限定**：`add_compile_options(/O2 /utf-8 /fp:precise)`
  改成三条 `$<$<COMPILE_LANGUAGE:CXX>:...>`。`else()` 分支早有同样的限定与注释，
  MSVC 分支因为从没在 Windows 上跑过而漏了（见下面的坑 1）。
- **37 个源文件加 UTF-8 BOM**（`src/` 20 + `tools/` 2 + `include/` 1 个 `.h`，加 14 个 `.cu`/`.cuh`）。
  ⚠️ **这连带改了上位仓库的 16 个同名文件** —— 带 BOM 的文件里有一半是同源副本，
  上游不加 BOM，`check_sync.py` 会整批报"第 1 行不同"。两边一起改才自洽。
- **`tests/check_sync.py` 加一条 BOM 自检**（第二件事，与"副本有没有跑偏"无关）：
  扫 `include/` / `src/` / `tools/` 下**所有** `.cu` / `.cuh` / `.h`，逐个查开头三个字节。
  理由：Windows 构建的成败**逐文件判定** —— `.cu` 加了 BOM、它 include 的 `.h` 没加一样炸，
  而"新加一个头文件时记得加 BOM"这种事靠人记迟早会漏。现在 37/37 带 BOM，
  负向也验过（临时塞一个无 BOM 的 `.h` ⇒ 立刻报出文件名并 exit=1）。
- **产物换进 RPGBar**：`models/rpg_engine_cuda.dll`（与 CPU 版**同名同 ABI**，
  宿主按"能不能用"探测后再选，见下面的说明）。

### 说明（P3 第四步）

- **三个坑，全在工具链与编码上，源码一个字没改就能编过**：

  1. **裸的 MSVC 风格开关会被 nvcc 当成输入文件** —— Windows 上 `/` 也是路径分隔符，
     `/O2` / `/utf-8` / `/fp:precise` 裸传给 nvcc 就变成"第二个输入文件" ⇒
     `nvcc fatal : A single input file is required for a non-link phase when an outputfile
     is specified`（13 个 `.cu` 全挂）。必须用语言限定生成式
     （`$<$<COMPILE_LANGUAGE:CXX>:/O2>`）—— 与下面那条 `-O2` 落到 CUDA 的坑**同源**，
     只是一个让编译失败、一个让它悄悄降档。
     **排查手段**：用一行 `tiny2.cu` **逐级加参数二分**（t1–t15），十几分钟钉死问题开关，
     而不是盯着 13 个文件的报错猜。
  2. **`.cu` / `.cuh` / `.h` 必须带 UTF-8 BOM** —— nvcc 在 Windows 上按**本地代码页**（936）
     读源文件，中文字符串字面量按 GBK 解读时某些字节序列产生**假的反斜杠 `\`**，
     把结束引号转义掉 ⇒ `error: missing closing quote`（62 次）+ `unrecognized token`，
     且错误**连锁**到后面几十行。试过并**证伪**的方案：`nvcc -utf-8`（只管 execution charset）、
     `-Xcompiler=/utf-8`（只影响交给 `cl` 的那半，且实测会**破坏**带 BOM 的文件）、
     `chcp 65001`、换 clang 当 host（本机只有 ARM64 版 `clang-cl.exe`）。**只认 BOM，且逐文件判**。
     ⚠️ 只有**中文字符串字面量**会炸；纯中文**注释**无害（预处理阶段就被剥离）。
     给 `.cu` 加完 BOM 后又暴露 `.h` 也得加（`fatal error C1020: 意外的 #endif`，
     `cl` 读无 BOM 的 `.h` 时行尾产生 `\` 把结构吃掉）。
  3. **cudart 是静态链进去的** —— `dumpbin /dependents rpg_engine.dll` 里只有
     `KERNEL32` / `MSVCP140` / `VCRUNTIME140*` / `api-ms-win-crt-*`，**没有 `cudart64_13.dll`**；
     `nvcuda.dll` 是**延迟加载**的。这两件事一起改变了宿主侧的分发与探测逻辑
     （不用带 cudart；而"库能不能加载"与"有没有 N 卡"必须分开判，见 RPGBar 的 `server/engine.py`）。

- **根因是 `nvcc` 每次真正编译都要调 `reg.exe`**（去找 MSVC 与 Windows SDK 的位置）——
  `nvcc --version` **不调**它，所以之前看不出来。这是唯一的**环境**卡点，
  放行之后一次编过。

- **Windows 侧实测**（2026-10-13，RTX 5060 Laptop + MSVC + CUDA 13.2）：

  | 项 | 结果 |
  | --- | --- |
  | `build_win.ps1` | exit=0，`rpg_engine.dll` **1,061,376 B** |
  | `abi_smoke.py --lib build-win/rpg_engine.dll --model …` | **38 通过 / 0 失败**（`backend=cuda`） |
  | `cuda_check.exe` | RTX 5060 Laptop / 算力 **12.0** / 26 SM / 显存 6.9+8.0 GiB；搬运全对 |
  | `kv_map_check.exe` | PASS |
  | `check_sync.py`（含新 BOM 判据） | 25 一致 / 0 分叉 / 0 缺失；37/37 带 BOM |
  | 接入 RPGBar 后 `engine_probe.py --backend gpu` | `实际后端: gpu`；**0.9 s / 15 token（16.3 tok/s）** |
  | 游戏内三轮真实回合（同一串动作） | GPU **37.5 / 27.7 / 21.5 s** vs CPU **199.3 / 114.4 / 109.2 s** ⇒ **4.1–5.3×** |
  | 退出时 `cudaFree` 警告 | **0 行** |

- ⚠️ **第四个坑（未修，已记入 DESIGN §9 第 7 条）：`.cu` 里的中文字符串在二进制里是 GBK，
  `.cc` 里的是 UTF-8。** nvcc 的宿主侧字面量按**执行字符集**（= 系统代码页）落盘，
  交给 `cl` 的 `/utf-8` 改不动它。实测排除：

  | 试过的 | 结果 |
  | --- | --- |
  | `-Xcompiler=/source-charset:.936 -Xcompiler=/execution-charset:utf-8` | **完全无效** —— 字面量在 .obj 里仍是 GBK |
  | `-Xcompiler=/utf-8` | **能把字面量变成 UTF-8**，但会让**一部分中文字符串编不过**（同一次实验：`"测试"` 通过、`"权重已上卡"` 报 `missing closing quote`）⇒ **不可用** |

  证据：`models/rpg_engine_cuda.dll` 里 `权重已上卡` **同时存在 UTF-8 与 GBK 两份**
  （`engine.cc:287` 与 `gpu_model.cu:237` 是同一条消息），而 `已创建会话` / `分词器就绪`
  只有 UTF-8 一份。⇒ 症状是 GPU 后端的引擎日志在 UTF-8 终端上部分乱码。
  **修法待定**（要么在日志 sink 上把 CUDA 侧的 ACP 转成 UTF-8，要么让 `.cu` 不再带中文
  字符串），**动之前要先问用户**：它跨到"日志用什么语言/什么编码"这个约定上了。

  > **2026-10-13 晚更新**：已按用户拍板**修掉了** —— 见下面
  > 「说明（P3 第四步补：`.cu` 中文串改英文）」。这里保留当时的排除记录。

### 说明（P3 第四步补：`.cu` 中文串改英文）

- ✅ **第四个坑修掉了**：`src/` 下会进产物的中文字符串字面量**一律改英文（97 处）**，
  **注释里的中文一个没动**。用户拍板选这条路的理由是它唯一满足三条：**确定性**
  （源码里没有非 ASCII ⇒ 编出来不可能有编码问题）、**零运行时依赖**、**不碰同源副本**。
  - **影响面实测 97 处**（⚠️ 不是早先记的 309）：`cuda_common.cu` 16、`gpu_model.cu` 33、
    `kernels_gpu.cu` 19、`kv_cache_gpu.cu` 6、`qwen3_gpu.cu` 15、`sampler_gpu.cu` 8；
    `quant_gpu.cuh` **0 处**（中文全在注释里）。`tools/*.cu` 的 110 处**没动**（判据工具、
    不进产物，且主要在 WSL 的 GCC 线上跑）。这 6 个文件都不在 §3「复用」表里 ⇒ `check_sync.py`
    仍 25 一致。
    ⚠️ **309 → 97 的差是计数方法错的**：旧数把**注释里被引号括起来的短语**也算成了字面量。
    教训：**扫描器必须先剥注释、再取字符串字面量，两步不能合并**。
  - **改法**：**按扫描器给出的精确区间替换**，不做全文查找替换（后者会误伤注释里同形的
    短语，且手抄的空格/全角半角对不上时会静默漏改）。
  - **验收三条，必须一起过**：① 代码里的字符串字面量不再含任何非 ASCII；② 注释里的中文
    **逐字未动**（六个文件按非 ASCII 字符数对比：`253 / 588 / 1716 / 78 / 1715 / 1484`
    全部一一相等）；③ **格式串数量与顺序完全不变**（`%s`/`%lld`/`%.1f`/`%g`/`%u` 逐项对拍）
    —— **这条最关键：漏一个 `%s` 是运行期未定义行为，而且编译期不报错**。
  - **实测**：6 个文件三条全过；重建产物 **1,061,888 B**（构建只多出 CUDA 自带头文件
    `cuda_runtime_api.h` 的 C4819，与我们的源码无关）；`abi_smoke.py` **38 通过 / 0 失败**；
    GPU 全量冒烟与 CPU 一致。分层结果：**库（`.cu`）出英文、宿主（`.cc`）出中文** ——
    同一段启动日志里 `RoPE table uploaded to device: …`（原先乱码）与
    `已加载 Qwen3-1.7B（…）`（出自 `engine.cc`）并存且都正常。
  - **顺带消掉一个隐患**：原来说"Windows 上的 CUDA 构建依赖构建机的代码页"，改成英文后
    自动消失 —— 产物里没有非 ASCII 了。
  - **没选那三条"集中式"的路**：① 日志 sink 转 ACP→UTF-8（`tools/*.cu` 的直接 `printf`
    管不到，且依赖系统代码页）；② 搬 `.cc`（结构改动大）；③ 运行时 `MultiByteToWideChar`
    （与 ① 同病，WSL 侧多余）。
  - **编译期修法全部证伪**（见上文 P3 第四步那张表的补充五条）。根因：**字面量不是 `cl`
    落盘的**，是 nvcc 自己的前端写进 `.obj` 字符串表 ⇒ `-Xcompiler` 的字符集开关打不到
    那一层。⚠️ 新记一条：**`u8"…"` 前缀是陷阱 —— 编译通过、字节错**（UTF-8 字节被当
    Latin-1 又编一遍，`权` → `c3 a6 c2 9d c2 83`）；`#pragma execution_character_set`
    同样失败。

### 新增（P4 第一步：CUDA Graph）

- **`EngineStream()` / `ScopedEngineStream`**：legacy default stream 不允许被
  `cudaStreamBeginCapture` 捕获 ⇒ 加一层流抽象：默认返回 `nullptr`（不捕获时与 P0–P3
  逐字一致），捕获窗口内切到专用流，于是所有 kernel 自动落在同一条流上、**每个 kernel
  的启动代码一行没改**。
- **`pos0` 暂存区**：回放时 kernel 参数被冻结 ⇒ 每步变化的 `pos0` 不能走值参。
  `GpuRope` / `GpuKvWrite` / `GpuAttentionFromCache` 的 `pos0` 由值参改**设备指针**
  （host 启动函数签名不变），指向一块进程级固定地址的暂存区。
- **采样设备端 counter 路径**（`GpuSampleTokenDev`）：`counter` 也留在设备端 —— 直接
  指向 `pos0` 暂存区（同一个数），`u = H(seed, counter)` 由 kernel 算。
- `use_graph` **默认 false**：P0–P3 的行为逐字不变；图路径由 `tools/graph_check` 显式开。
- 新增 `tools/graph_check.cu`、`tools/matmul_bench.cu`、`tests/run_p4.sh`。

### 说明（P4 第一步）

- 判据 A–E 全绿：图路径 vs 老路径**逐位相同**（贪心 / 纯采样 / 带筛选采样）、同会话
  两段连续生成 + 断言第二段没重录图、一条不涉图的定点判据（host 算 u vs device 算 u）。
  ⚠️ 每组比较都**必须**同时断言 `graph_exec != nullptr` —— 录不出图会静默退回老路径，
  那样"老路径 vs 老路径"也全绿。
- 🔴 **性能只 1.00–1.09x ⇒ 原判断"decode 慢 = 每步约 400 次 kernel 启动"被证伪。**
  `tools/matmul_bench` 量出真瓶颈：n_tokens 1→4 耗时几乎不变 ⇒ 成本与 token 数无关，
  由"把每个权重元素过一遍"的**逐元素反量化 ALU** 决定。
- 顺带修掉一个真实的静默错误：`StagePos0Cached` 把暂存区当前值缓存在**函数局部
  static**，而图回放直接调 `StagePos0` 绕过缓存 ⇒ 缓存说谎，下一个会话从位置 0 开始
  prefill 时被判定"已经是 0"而跳过写入、那次 prefill 的 RoPE 位置全错（症状极隐蔽：
  贪心组刚好撞对、采样组 32/32 全不同）。已把缓存搬进 `cuda_common`、由 `StagePos0`
  独占维护。

### 新增（P4 第二步：matmul 热循环重写）

- **`MatMulQ4KKernel` / `MatMulQ6KKernel`**（`src/kernels_gpu.cu`）：块常量外提（每 256
  元素一份的 d/dmin、每 32 元素（Q6_K 每 16）一份的"已乘 d 的 scale"提到元素循环外）
  + 整行一次预计算 ⇒ 元素级只剩"取半字节 + I2F + 1 mul + 1 sub"，元素循环里**零同步**。
- `GpuMatMulQuant` 按类型分派；F32（实际权重里只有 norm）与"块形状不符"继续走通用
  `MatMulQuantKernel` —— 通用版与专用版**逐位等价**，退回是安全的。
- **新增判据开关** `GpuSetForceGenericMatMul`（`src/kernels_gpu.h`）：仅供判据工具在
  两条路径间切换，生产路径不调。
- `tools/matmul_bench --verify`：同一份权重/输入下，通用 kernel 与专用 kernel 的输出
  **逐位比较**（不是相对误差 —— 那是 P1 的活）。

### 说明（P4 第二步）

- **逐位对拍 32 个用例（8 张矩阵 × {1,3,5,8} token）全部 0 个 bit 不同**，覆盖 Q4_K 与
  Q6_K 两条专用 kernel；最大单用例 `lm_head`（151936 行输出）也逐位相同。
- **实测**（1.7B Q4_K_M，RTX 5060 Laptop）：matmul 有效带宽 35.7 → **52.3 GB/s**
  （n_tokens=1，8.057 → 5.496 ms）；端到端 decode 19.0 → **26.9 tok/s**、
  prefill 86.9 → **132.5 tok/s**。
- 回归全绿：`check_sync` 25 一致、P1（最差 p99 1.05e-4）/ P2（逐 token 完全一致）/
  P3 / P3-gen / P4 全过。
- **瓶颈定位（nsys）**：matmul 占 **96.8%** 的 GPU kernel 时间（Q4K 64.6% + Q6K 32.2%），
  RmsNorm / Attention / 其余算子合计 3.2% ⇒ 下一步仍在 matmul 上。
- ⚠️ **两项仍未达 DESIGN §5 的目标**（decode 26.9 / 60、prefill 132.5 / 1000），且以上
  数字全部是 **1.7B**；产品 `config.json` 跑的是 **4B**（权重约 2.3x）。P4 的最终验收口径
  仍是"换 `.dll` 进 RPGBar、量 4B 的 DM 一轮耗时"。

### 说明（P0）

- **P0 的两档判据全绿**（2026-10-12）。不需要显卡的那一档在 Windows 上跑
  （`check_sync.py` 22 一致 / `abi_smoke.py` 32 通过）；需要显卡的那一档在 WSL +
  CUDA 13.2 上跑（`abi_smoke.py` 32 通过；`cuda_check` 报出 RTX 5060 Laptop / 算力 12.0 /
  26 SM，device 计算与搬运全对）。
  （数字随阶段推进而变：到 P3 结束时是 25 项 / 38 通过 —— 每一条都记在对应阶段的小节里。）
- **CUDA 路径的构建放在 WSL 里**，因为 Windows 侧没有 MSVC（而 nvcc 不接受 g++ 当宿主
  编译器）。为此新增 `build_wsl.sh`：源码留在 `/mnt/e`（一份，不会分叉），构建目录落
  WSL 原生 ext4（9p 上的顺序写只有 173 MB/s，原生是 2.5 GB/s）。产物 `librpg_engine.so`
  供 P0–P2 的算法对拍使用；宿主能加载的 `.dll` 等 P3 时补 MSVC + Windows 版 Toolkit
  （**已于 P3 第四步补完**，见上）。
- **首次编译抓出两个真问题**（host-only 构建看不到，因为相关代码被 `#ifdef` 掉）：
  `cuda_common.h` 把 `<cuda_runtime.h>` include 进了 `namespace rpg`（libstdc++ 的
  `namespace std` 被嵌套成 `rpg::std`，该 TU 全线崩、报错却全落在标准库内部）；
  `cuda_check` 目标漏了 `src/cuda_common.cc`（链接期找不到 `rpg::DescribeDevice`）。
  两处都已修，教训写进 DESIGN §8。
