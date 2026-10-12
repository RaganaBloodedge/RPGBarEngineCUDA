#!/usr/bin/env bash
#
# P2 的一键判据（DESIGN §5）。三条，都必须过：
#
#   ⓿ KV 映射 —— GpuKvWrite 的下标重排（[t][h][d] → [h][pos][d]）与写入边界。
#                **不需要权重、不需要模型、与精度无关**（值用 fp16 能精确表示的
#                整数，判据是逐位相同）。放在最前面：映射错了不会崩，只会读到
#                别的头/别的位置 —— 症状与"kv 的 fp16 精度不够"一模一样，
#                而后者是本项目明确接受的取舍（DESIGN §4.2）。这条负责把两者分开。
#
#   ① KV 自检 —— 同一段 prompt，带 cache 的 prefill 与不带 cache 的全序列前向，
#                比 logits。判据：**max|Δ| ≤ 1e-3 × max|ref|** 且每个位置的 argmax
#                一致。
#                ⚠️ CPU 侧这条是"逐位相同"（KV 是 fp32）；GPU 侧 KV 是 fp16，
#                逐位相同不可能。**但判据不是"逐元素相对误差的 p99"** —— 那个量在
#                这条链路上会被"量级 0.5 上下的元素 + 固定的绝对误差"主导，随
#                logits 分布漂移，不是一个能定的判据。换成"绝对误差 / 满量程"：
#                fp16 尾数 10 位 ⇒ 1 ULP = 2^-11 = 4.88e-4，1e-3 就是约 2 个 ULP。
#                （教训与 CPU 版那次"绝对误差 < 1e-4 不可能满足"同源：
#                 **定数值判据前先量目标量级的 ULP**。）
#
#   ② 逐 token 对拍 —— 两边同一 prompt、同一 max_tokens，T=0 贪心各生成一遍，
#                     逐步比 token 与 logits。判据：完全一致；若有分歧，每一处
#                     都必须能用"两个 token 本来就在伯仲之间"解释并记录在案。
#
# 参考侧刻意用 CPU 引擎自己的源码现编（tools/build_cpu_ref.sh），不在本仓库里
# 抄一份 CPU 前向 —— 理由见那个脚本的头注释。
#
# 用法：
#   bash tests/run_p2.sh
#   RPG_MAX_TOKENS=64 RPG_TOL=1e-3 bash tests/run_p2.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${RPG_BUILD_DIR:-$HOME/cuda-build}"
MODEL="${RPG_MODEL:-$ROOT/models/Qwen3-1.7B-Q4_K_M.gguf}"
RUNS="${RPG_RUNS:-$HOME/p2-runs}"
TOL="${RPG_TOL:-1e-3}"
GAP="${RPG_GAP:-1e-2}"
MAXTOK="${RPG_MAX_TOKENS:-32}"
PY="${PY:-python3}"

mkdir -p "$RUNS"

# 提示词走文件而不是命令行：中文经 argv 会被控制台代码页转码搞坏。
PROMPT="$RUNS/prompt.txt"
if [ ! -f "$PROMPT" ]; then
  cat > "$PROMPT" <<'EOF'
你是一位地下城的城主。冒险者推开了石门，请你描述他看到的一切。
EOF
fi

[ -x "$BUILD/gen_check" ] || {
  echo "没找到 $BUILD/gen_check —— 先构建：bash build_wsl.sh" >&2; exit 1; }
[ -f "$MODEL" ] || {
  echo "缺权重 $MODEL（python scripts/fetch_model.py，或放到 models/ 下）" >&2; exit 1; }

fail=0

echo "=============================================================="
echo "⓿ KV 写入的下标映射（不需要模型，纯确定性）"
echo "=============================================================="
"$BUILD/kv_map_check" || fail=1

echo
echo "=============================================================="
echo "参考侧：CPU 引擎的 cpu_ref_gen（源码来自 RPGBarEngine）"
echo "=============================================================="
if [ ! -x "$BUILD/cpu_ref_gen" ]; then
  bash "$ROOT/tools/build_cpu_ref.sh" "$BUILD"
fi

echo
echo "=============================================================="
echo "被测侧：GPU 的 gen_check（含 ① KV 自检）"
echo "=============================================================="
if ! "$BUILD/gen_check" "$MODEL" --prompt-file "$PROMPT" --max-tokens "$MAXTOK" \
        --out "$RUNS/gpu" --step-logits; then
  echo "**① KV 自检未通过**（或 gen_check 本身报错）" >&2
  fail=1
fi

echo
echo "=============================================================="
echo "参考侧生成（CPU 引擎的 GenerateGreedy + 重放收集 logits）"
echo "=============================================================="
"$BUILD/cpu_ref_gen" "$MODEL" --prompt-file "$PROMPT" --max-tokens "$MAXTOK" \
    --out "$RUNS/ref" --step-logits

echo
echo "=============================================================="
echo "② 逐 token 对拍（数值层 max|Δ| <= $TOL × max|ref|；分歧处 gap <= $GAP 可接受）"
echo "=============================================================="
"$PY" "$ROOT/tests/cmp_gen.py" "$RUNS/gpu" "$RUNS/ref" --tol "$TOL" --gap "$GAP" \
  || fail=1

echo
if [ "$fail" -eq 0 ]; then
  echo "P2 判据全部通过。"
else
  echo "P2 有判据未通过（见上）。"
fi
exit "$fail"
