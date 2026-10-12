#!/usr/bin/env bash
#
# 用 **CPU 引擎（RPGBarEngine）自己的源码** 编出 forward_check —— P1 对拍的参考侧。
#
# ## 为什么不把 CPU 前向抄进本仓库
#
# 参考实现必须是"真的在跑的那一份"。在本仓库里再抄一份 CPU 前向，等于拿我对
# CPU 行为的理解去对我的 GPU 实现 —— 两边同源的理解错误会互相掩盖，判据就废了。
# 所以这里只给它一条编译命令，源码原封不动留在 RPGBarEngine 仓库里；本仓库
# 一个字节都不复制。
#
# ## 为什么不能直接用 CPU 仓库的 build.sh
#
# 那个脚本是给 MSYS2/MinGW 写的：-static、rpg_engine.def、--exclude-all-symbols
# 都是 Windows 专有，在 Linux 上第一条链接就失败。所以这里只取它的编译约定
# （matmul_avx2.cc 单独开 -mavx2 -mfma，其余保持基线），重写一条 Linux 命令。
#
# 用法：
#   bash tools/build_cpu_ref.sh [输出目录]
#   RPG_CPU_ENGINE_DIR=/path/to/RPGBarEngine bash tools/build_cpu_ref.sh
#
# 产物：<输出目录>/forward_check
#
set -euo pipefail

SRC="${RPG_CPU_ENGINE_DIR:-/mnt/e/Workspace/RPGBarEngine}"
OUT="${1:-$HOME/cuda-build}"
CXX="${CXX:-g++}"
JOBS="${RPG_JOBS:-$(nproc)}"

if [ ! -f "$SRC/tools/forward_check.cc" ]; then
  echo "找不到 $SRC/tools/forward_check.cc" >&2
  echo "用 RPG_CPU_ENGINE_DIR=<RPGBarEngine 路径> 指定 CPU 引擎仓库的位置。" >&2
  exit 1
fi

mkdir -p "$OUT"
OBJDIR="$OUT/cpu-ref-obj"
mkdir -p "$OBJDIR"

echo "CPU 引擎源码：$SRC"
echo "输出：$OUT/forward_check"
echo "编译器：$($CXX --version | head -1)"

INCLUDES="-I$SRC/include -I$SRC/src -I$SRC/tools"
# -pipe：汇编走管道，不落临时 .s（本机删除/截断临时文件会被安全策略拦住）。
CXXFLAGS="-std=c++20 -O2 -fPIC -pipe -Wall -Wextra -DRPG_ENGINE_BUILD $INCLUDES"

# 编译器认不认 -mavx2 -mfma 决定 AVX2 内核那条 TU 怎么编。
# 认不出来就不加 —— 该文件的 AVX2 代码全在 __AVX2__ 之后，会自动退化成
# "返回 nullptr"，由分派层走标量（与 CPU 仓库的 build.sh 同一个判断）。
AVX2FLAGS=""
if printf 'int main(){return 0;}' | "$CXX" -mavx2 -mfma -x c++ -fsyntax-only - \
     >/dev/null 2>&1; then
  AVX2FLAGS="-mavx2 -mfma"
else
  echo "  (编译器不支持 -mavx2 -mfma，AVX2 内核退化为标量)"
fi

echo "[1/2] 编译 CPU 引擎源码..."
OBJS=""
for f in "$SRC"/src/*.cc; do
  b="$(basename "${f%.cc}")"
  o="$OBJDIR/$b.o"
  extra=""
  if [ "$b" = "matmul_avx2" ]; then extra="$AVX2FLAGS"; fi
  "$CXX" $CXXFLAGS $extra -c "$f" -o "$o"
  OBJS="$OBJS $o"
done

echo "[2/2] 链接 forward_check..."
# 先链 .new 再 mv -f：与两边仓库的构建脚本同一个环境对策（直接覆盖已存在
# 的产物在某些沙箱下会静默失败）。
"$CXX" $CXXFLAGS -o "$OUT/forward_check.new" "$SRC/tools/forward_check.cc" $OBJS -pthread
mv -f "$OUT/forward_check.new" "$OUT/forward_check"

ls -la "$OUT/forward_check"
echo "OK"
