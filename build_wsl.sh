#!/usr/bin/env bash
#
# 在 WSL 里构建 RPGBarEngineCUDA。
#
# 为什么要有这个脚本，而不是直接 cmake：
#   WSL 通过 9p 访问 E:\ —— 实测顺序写只有 173 MB/s，而 WSL 原生 ext4 是 2.5 GB/s（差 14 倍）。
#   所以约定：源码只留一份（就在 /mnt/e 下，git 管着，Windows 与 WSL 看到的是同一份，不会分叉），
#   但构建目录一律落在 WSL 原生文件系统（默认 ~/cuda-build）。
#   落到磁盘的是 .o / .so / nvcc 中间件，全在 ext4 上；只剩几百 KB 源文件读取走 9p。
#
# 用法：
#   bash build_wsl.sh                 # Release 构建
#   bash build_wsl.sh --clean         # 先删掉构建目录再构建
#   RPG_BUILD_DIR=~/other bash build_wsl.sh
#   RPG_CUDA_ARCH=120 bash build_wsl.sh
#
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${RPG_BUILD_DIR:-$HOME/cuda-build}"
ARCH="${RPG_CUDA_ARCH:-120}"
JOBS="${RPG_JOBS:-$(nproc)}"

CLEAN=0
for a in "$@"; do
  case "$a" in
    --clean) CLEAN=1 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "未知参数：$a（用 --help）" >&2; exit 2 ;;
  esac
done

if [ ! -x /usr/local/cuda/bin/nvcc ]; then
  echo "找不到 /usr/local/cuda/bin/nvcc —— 先装 CUDA Toolkit（apt install cuda-toolkit-13-2）" >&2
  echo "没有 Toolkit 时可以用 host-only 路径：bash build_host.sh" >&2
  exit 1
fi

export PATH="/usr/local/cuda/bin:${PATH}"
export CUDACXX="${CUDACXX:-/usr/local/cuda/bin/nvcc}"

echo "源码：$SRC   （$(df -T "$SRC" | tail -1 | awk '{print $2}')）"
echo "构建：$BUILD （$(df -T "$(dirname "$BUILD")" | tail -1 | awk '{print $2}')）"
echo "算力：sm_$ARCH   并行：$JOBS   编译器：$(nvcc --version | tail -1)"

if [ "$CLEAN" = "1" ]; then
  echo "--- 清空构建目录 ---"
  rm -rf "$BUILD"
fi

cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build "$BUILD" -j"$JOBS"

echo "--- 产物 ---"
ls -la "$BUILD"/librpg_engine.so "$BUILD"/cuda_check "$BUILD"/dequant_check \
       "$BUILD"/layer_check 2>/dev/null || ls -la "$BUILD" | grep -E "\.so$|_check$" || true
echo
echo "下一步（P0 第二档判据）："
echo "  $BUILD/cuda_check"
echo "  python3 tests/abi_smoke.py --lib $BUILD/librpg_engine.so --model models/Qwen3-1.7B-Q4_K_M.gguf"
echo
echo "P1 的一键判据（会顺带编 CPU 侧参考实现）："
echo "  bash tests/run_p1.sh"
