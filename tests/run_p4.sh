#!/usr/bin/env bash
#
# P4 判据（DESIGN §5）：**CUDA Graph 路径** —— 从架构上把 decode 的启动开销摊掉。
#
# 背景：P2 实测 GPU 的 decode 只比 CPU 快 1.5x（prefill 快 14.1x）。不是算力不够：
# decode 每步只算 1 个 token，却要发 28 层 x 十几个 kernel ≈ 400 次 kernel 启动。
# CUDA Graph 把这几百次启动录成一张图，回放时一次提交。
#
# 本脚本跑的是 tools/graph_check（要模型、要权重）。它验两组东西：
#
#   正确性（同一份 prompt、同一个 seed，图路径 vs 老路径）：
#     A. 贪心 T = 0              —— 图里的 argmax 分支必须逐位等于 P2 的贪心
#     B. 采样 T = 1（不筛选）    —— 整条采样链都在图里：counter 从设备端读、
#                                  u 由 kernel 算、逆变换读设备指针
#     C. 采样 T = 1 + 筛选       —— top_k / top_p 是**启动参数**、会被录进图
#     D. 两段连续生成（同一会话）—— 宿主多轮工具循环的形态：图跨调用复用、
#                                  pos0 继续递增（单段测不到"第二次调用之后"）
#     判据口径是**逐位相同**，不是"差不多"：图不是另一份实现，是同一段计算被录
#     下来（同一批 kernel、同一组参数、同一个顺序）。
#
#   ⚠️ **必须**同时断言 `graph_exec != nullptr`（工具里叫 "graph was captured and
#     used"）：GenerateGraph 在录不出图时会**静默退回老路径**，那样"老路径 vs
#     老路径"当然逐位相同 —— 全绿而一行新代码都没跑到。这条防御比正确性那条更
#     容易漏掉，所以写在同一个工具里、同一份报告里。
#
#   性能（如实报出，不因未达标而红）：
#     prefill >= 1000 tok/s、decode >= 60 tok/s（DESIGN §5 的目标表）
#     —— 这两个数是**目标**；P4 的硬判据是上面那两组正确性。
#
# 与 run_p2.sh 的关系：run_p2.sh 盯着"贪心没被改动"（逐 token 与 CPU 对拍），
# 它用的 tools/gen_check 走的还是老路径（use_graph 默认 false）—— 两者不重叠。
#
# 用法：
#   bash tests/run_p4.sh
#   RPG_MAX_TOKENS=64 RPG_BENCH_TOKENS=256 RPG_SEED=777 bash tests/run_p4.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${RPG_BUILD_DIR:-$HOME/cuda-build}"
MODEL="${RPG_MODEL:-$ROOT/models/Qwen3-1.7B-Q4_K_M.gguf}"
RUNS="${RPG_P4_RUNS:-$HOME/p4-runs}"
MAXTOK="${RPG_MAX_TOKENS:-32}"
CHUNKS="${RPG_CHUNKS:-2}"
BENCH="${RPG_BENCH_TOKENS:-128}"
SEED="${RPG_SEED:-12345}"

mkdir -p "$RUNS"

# 提示词走文件而不是命令行：中文经 argv 会被控制台代码页转码搞坏。
PROMPT="$RUNS/prompt.txt"
if [ ! -f "$PROMPT" ]; then
  cat > "$PROMPT" <<'EOF'
你是一位地下城的城主。冒险者推开了石门，请你描述他看到的一切。
EOF
fi

[ -x "$BUILD/graph_check" ] || {
  echo "没找到 $BUILD/graph_check —— 先构建：bash build_wsl.sh" >&2; exit 1; }
[ -f "$MODEL" ] || {
  echo "缺权重 $MODEL（python scripts/fetch_model.py，或放到 models/ 下）" >&2; exit 1; }

echo "=============================================================="
echo "P4 判据：CUDA Graph（图路径 vs 老路径逐位相同 + 图确实被用到 + 吞吐）"
echo "=============================================================="
# 报表要把"这组数是在什么上、什么时候量的"钉住（DESIGN §9.3）：换一张卡、
# 换一版驱动，这些数就不该被拿来互相比。温度只作参考 —— 笔记本 GPU 长时间
# 跑会被功耗墙压下来，那是"机器热了"，不是"代码变慢了"。
#
# ⚠️ WSL 里 nvidia-smi 在 /usr/lib/wsl/lib/，**不在默认 PATH 上**（而且这个
# 脚本自己 export 过一个精简 PATH，更找不到）⇒ 显式往那里找一下。
SMI="$(command -v nvidia-smi || true)"
if [ -z "$SMI" ] && [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
  SMI=/usr/lib/wsl/lib/nvidia-smi
fi
gpu_line() {
  [ -n "$SMI" ] || return 0
  "$SMI" --query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw \
         --format=csv,noheader 2>/dev/null | sed "s/^/$1/" || true
}

echo "date : $(date -Iseconds)"
echo "build: $BUILD"
gpu_line "gpu  : "
echo
"$BUILD/graph_check" "$MODEL" --prompt-file "$PROMPT" \
  --max-tokens "$MAXTOK" --chunks "$CHUNKS" --bench-tokens "$BENCH" \
  --seed "$SEED" | tee "$RUNS/out.txt"
status="${PIPESTATUS[0]}"

echo
gpu_line "gpu after run: "
echo "date : $(date -Iseconds)"
if [ "$status" -eq 0 ]; then
  echo "P4 的 CUDA Graph 判据全部通过。"
else
  echo "P4 的 CUDA Graph 判据有未通过项（见上）。"
fi
exit "$status"
