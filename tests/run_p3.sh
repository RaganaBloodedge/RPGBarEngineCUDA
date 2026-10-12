#!/usr/bin/env bash
#
# P3 的一键判据（DESIGN §5）。目前是**第一条**（设备端采样），后面的接通 ABI /
# 换 dll / Unity 那些会陆续加进来。
#
# ⓿ 设备端采样 —— temperature / top_k / top_p 筛选 + counter-based 随机源 + 逆变换。
#                 **不需要权重、不需要模型**，几秒就跑完。
#
#               为什么这一块能判"逐 token 相同"：见 tests/cmp_sample.py 的头注释 ——
#               把模型摘掉、两边喂**同一份 logits**，"采样不可对拍"那半句话就不成立了。
#
#               判据三条：
#                 a. u 表（CounterUniform）**逐位相同** —— 纯整数运算，没有容差可说；
#                    ⚠️ 随机源是"重写"的（CPU 版在 RPGBarEngine 的 sampler.cc 里），
#                    最容易"改一边忘另一边"，所以单独判一条。
#                 b. 概率分布满量程误差 ≤ 1e-5，且**保留集合完全相同**（哪些 token
#                    没被 top_k/top_p 筛掉是离散判断，比"数值接近"说明问题）。
#                 c. token 逐位相同；若分歧，只有 u 距累积概率边界 < 1e-6 才允许
#                    （那是 GPU 与 glibc 的 expf 差 1 ULP 能推过去的唯一位置）。
#
#               跑两遍：4096（快，用例覆盖各种筛选组合）与 151936（真实词表，
#               验二分搜索在整个 uint32 key 空间上都收敛 —— 小词表测不出"中点
#               溢出导致死循环"那类 bug）。
#
# 用法：
#   bash tests/run_p3.sh
#   RPG_SAMPLE_VOCAB=8192 bash tests/run_p3.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${RPG_BUILD_DIR:-$HOME/cuda-build}"
RUNS="${RPG_P3_RUNS:-$HOME/p3-runs}"
SMALL_VOCAB="${RPG_SAMPLE_VOCAB:-4096}"
REAL_VOCAB="${RPG_REAL_VOCAB:-151936}"
TOL="${RPG_SAMPLE_TOL:-1e-5}"
BORDER="${RPG_SAMPLE_BORDER:-1e-6}"
PY="${PY:-python3}"

mkdir -p "$RUNS"

[ -x "$BUILD/sample_check" ] || {
  echo "没找到 $BUILD/sample_check —— 先构建：bash build_wsl.sh" >&2; exit 1; }

fail=0

# run_one <vocab> <子目录名> <标题>
run_one() {
  local vocab="$1" sub="$2" title="$3"
  local dir="$RUNS/$sub"
  mkdir -p "$dir"
  echo
  echo "=============================================================="
  echo "⓿ 设备端采样：$title（词表 $vocab）"
  echo "=============================================================="
  "$BUILD/sample_check" --out "$dir" --vocab "$vocab" || { fail=1; return; }

  echo
  echo "---- 参考侧：CPU 引擎的 sampler.cc（源码来自 RPGBarEngine）----"
  if [ ! -x "$BUILD/cpu_ref_sample" ]; then
    bash "$ROOT/tools/build_cpu_ref.sh" "$BUILD"
  fi
  "$BUILD/cpu_ref_sample" --dir "$dir" || { fail=1; return; }

  echo
  echo "---- 判据：u 表 / 概率分布 / token ----"
  "$PY" "$ROOT/tests/cmp_sample.py" "$dir" --tol "$TOL" --border "$BORDER" || fail=1
}

run_one "$SMALL_VOCAB" "vocab$SMALL_VOCAB" "用例矩阵"
run_one "$REAL_VOCAB" "vocab$REAL_VOCAB" "真实词表（验二分搜索的全 key 空间）"

echo
if [ "$fail" -eq 0 ]; then
  echo "P3 的采样判据全部通过。"
else
  echo "P3 有判据未通过（见上）。"
fi
exit "$fail"
