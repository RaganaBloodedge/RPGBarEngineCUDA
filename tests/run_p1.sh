#!/usr/bin/env bash
#
# P1 的一键判据（DESIGN §5）。两条，都必须过：
#
#   ① 反量化逐位对拍  —— 设备端 vs src/quant.cc（从 RPGBarEngine 逐字复制的标量
#                        参考）。判据：**0 个 bit 不同**。反量化是纯整数拆包 +
#                        一次乘加，不涉浮点求和顺序，所以可以要求逐位相同。
#   ② 整段前向逐层对拍 —— GPU 的 layer_check vs CPU 引擎的 forward_check，同一串
#                        token、同一份 GGUF。判据：逐层残差流与 logits 的
#                        **p99 相对误差 ≤ 1e-3**，且每个位置的 argmax 一致。
#
# 参考侧刻意用 CPU 引擎自己的可执行文件（由 tools/build_cpu_ref.sh 现编），
# 不在本仓库里抄一份 CPU 前向 —— 理由见那个脚本的头注释。
#
# 用法：
#   bash tests/run_p1.sh
#   RPG_BUILD_DIR=~/other RPG_TOL=1e-3 bash tests/run_p1.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${RPG_BUILD_DIR:-$HOME/cuda-build}"
MODEL="${RPG_MODEL:-$ROOT/models/Qwen3-1.7B-Q4_K_M.gguf}"
RUNS="${RPG_RUNS:-$HOME/p1-runs}"
TOL="${RPG_TOL:-1e-3}"
PY="${PY:-python3}"

mkdir -p "$RUNS"

# 提示词走文件而不是命令行：中文经 argv 会被控制台代码页转码搞坏（两边仓库
# 都吃过这个亏，CPU 仓库的 cli_util.h 专门为此写了 ReadFile）。
PROMPT="$RUNS/prompt.txt"
if [ ! -f "$PROMPT" ]; then
  cat > "$PROMPT" <<'EOF'
你是一位地下城的城主。冒险者推开了石门，请你描述他看到的一切。
EOF
fi

[ -x "$BUILD/layer_check" ] || {
  echo "没找到 $BUILD/layer_check —— 先构建：bash build_wsl.sh" >&2; exit 1; }
[ -f "$MODEL" ] || {
  echo "缺权重 $MODEL（python scripts/fetch_model.py，或放到 models/ 下）" >&2; exit 1; }

fail=0

echo "=============================================================="
echo "① 反量化逐位对拍 + f16→f32 穷举"
echo "=============================================================="
"$BUILD/dequant_check" "$MODEL" --rows 32 || fail=1

echo
echo "=============================================================="
echo "参考侧：CPU 引擎的 forward_check（源码来自 RPGBarEngine）"
echo "=============================================================="
if [ ! -x "$BUILD/forward_check" ]; then
  bash "$ROOT/tools/build_cpu_ref.sh" "$BUILD"
fi
"$BUILD/forward_check" "$MODEL" --prompt-file "$PROMPT" --out "$RUNS/ref" --top 5

echo
echo "=============================================================="
echo "被测侧：GPU 的 layer_check"
echo "=============================================================="
"$BUILD/layer_check" "$MODEL" --prompt-file "$PROMPT" --out "$RUNS/gpu" --top 5

echo
echo "=============================================================="
echo "② 逐层残差流 + logits 对拍（p99 相对误差 <= $TOL）"
echo "=============================================================="
"$PY" "$ROOT/tests/cmp_layers.py" "$RUNS/gpu" "$RUNS/ref" --tol "$TOL" || fail=1

echo
if [ "$fail" -eq 0 ]; then
  echo "P1 判据全部通过。"
else
  echo "P1 有判据未通过（见上）。"
fi
exit "$fail"
