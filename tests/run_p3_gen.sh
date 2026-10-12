#!/usr/bin/env bash
#
# P3 的第二条判据（DESIGN §5）：**采样接进生成循环之后**的不变量。
#
# 与 run_p3.sh 的分工（两条都要有）：
#   * run_p3.sh   —— 把模型摘掉，验"采样本身"：同一份 logits + 同一个 seed ⇒
#                    同一个 token（逐位）。**不需要权重**，几秒跑完。
#   * 本脚本      —— 把模型装回去，验"接起来之后还讲不讲道理"。
#
# 为什么不与 CPU 版逐 token 对拍：DESIGN §6 第 10 条 —— 接在模型后面的采样，
# 两边 logits 先差了 fp16 的舍入，采样把差异放大成"整段序列不同"。硬对拍
# 只会得到一个查不清的混合指标。所以换一批**只看自己**的判据：
#
#   A. 贪心基准（T = 0）+ 不变量 cache.len() == prompt + 生成数
#   B. 可复现：T = 1 下同 seed 两遍 ⇒ 序列逐位相同（引擎仍是纯函数 ——
#      "没有有状态 RNG"这条全局约定在 P3 之前根本无从验证）
#   C. 随机性真的在起作用：换 seed ⇒ 序列不同（⚠️ 概率性判据，防的是
#      "seed 忘了接下去"，不是精度）
#   D. 筛选参数真的传到循环里了：D1（严格）T=0 + top_k/top_p ⇒ 与贪心逐位相同；
#      D2（必要条件）T=1, top_k=2 ⇒ 每步 token 落在该步前 2 名内；
#      D3（概率性）top_k=2 与 top_k=0 的输出必须不同。
#      ⚠️ 这里最初拿 top_k=1 当"支持集只剩一个"来断言，红了一片 —— 契约里
#      top_k 的生效条件是 `top_k > 1 && top_k < n`，**1 与 0 一样算关闭**。
#   E. counter 用的是**绝对位置**：把第一遍的输出接在 prompt 后冷跑一遍，
#      序列应整体错一位相同（暖缓存 = 冷缓存）
#
# 「贪心没被改动」由 run_p2.sh 的逐 token 对拍继续盯着 —— 那个脚本用的
# tools/gen_check 现在走的就是这条新循环（GpuGenerate），一行调用都没改。
#
# 用法：
#   bash tests/run_p3_gen.sh
#   RPG_MAX_TOKENS=64 RPG_SEED=777 bash tests/run_p3_gen.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${RPG_BUILD_DIR:-$HOME/cuda-build}"
MODEL="${RPG_MODEL:-$ROOT/models/Qwen3-1.7B-Q4_K_M.gguf}"
RUNS="${RPG_P3GEN_RUNS:-$HOME/p3gen-runs}"
MAXTOK="${RPG_MAX_TOKENS:-32}"
SEED="${RPG_SEED:-12345}"

mkdir -p "$RUNS"

# 提示词走文件而不是命令行：中文经 argv 会被控制台代码页转码搞坏。
PROMPT="$RUNS/prompt.txt"
if [ ! -f "$PROMPT" ]; then
  cat > "$PROMPT" <<'EOF'
你是一位地下城的城主。冒险者推开了石门，请你描述他看到的一切。
EOF
fi

[ -x "$BUILD/gen_sample_check" ] || {
  echo "没找到 $BUILD/gen_sample_check —— 先构建：bash build_wsl.sh" >&2; exit 1; }
[ -f "$MODEL" ] || {
  echo "缺权重 $MODEL（python scripts/fetch_model.py，或放到 models/ 下）" >&2; exit 1; }

echo "=============================================================="
echo "P3 生成循环判据：采样接进 decode 之后的五条不变量"
echo "=============================================================="
"$BUILD/gen_sample_check" "$MODEL" --prompt-file "$PROMPT" \
  --max-tokens "$MAXTOK" --seed "$SEED" | tee "$RUNS/out.txt"
status="${PIPESTATUS[0]}"

echo
if [ "$status" -eq 0 ]; then
  echo "P3 的生成循环判据全部通过。"
else
  echo "P3 的生成循环判据有未通过项（见上）。"
fi
exit "$status"
