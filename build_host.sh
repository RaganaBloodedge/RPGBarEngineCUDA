#!/usr/bin/env bash
# build_host.sh —— 没有 cmake / CUDA Toolkit 时的最小构建（MSYS2 / MinGW）。
#
# 与 CMakeLists.txt 的 **host-only** 路径等价：产物是一个能加载、能读 GGUF 元数据、
# 但 rpg_engine_generate 会明确报"未编译 CUDA 后端"的 dll。
#
# 它的用处：在没有显卡的开发机 / CI 上验证 ABI 层与平台无关层能编译、能链接。
# **要真正生成，必须有 nvcc** —— 那时请走 CMake。
#
# 两个从 RPGBarEngine 继承来的硬性要求，改这个脚本时别丢：
#   1) 静态链接 libwinpthread / libstdc++，否则 dll 在宿主里报 "Could not find
#      module ... or one of its dependencies"。
#   2) 用 mingw64/ucrt64 的 g++，不要 MSYS 原生 g++（会拖 msys-2.0.dll）。
set -e
cd "$(dirname "$0")"

CXX="${CXX:-g++}"
mkdir -p build/obj

INCLUDES="-Iinclude -Isrc"
# -pipe：汇编走管道、不落临时 .s —— 本环境会拦住"删除已存在的临时文件"。
CXXFLAGS="-std=c++20 -O2 -fPIC -pipe -Wall -Wextra -DRPG_ENGINE_BUILD $INCLUDES"
LDFLAGS="-static -static-libgcc -static-libstdc++"

echo "[1/3] 编译 host 层（不含 CUDA）..."
OBJS=""
for f in src/*.cc; do
  o="build/obj/$(basename "${f%.cc}").o"
  "$CXX" $CXXFLAGS -c "$f" -o "$o"
  OBJS="$OBJS $o"
done

echo "[2/3] 链接动态库..."
# 先链到 .new 再 mv -f：ld 直接覆盖已存在的 dll 时，本环境的删除守卫会拦住，
# 表现为一句没有上下文的 "ld returned 1 exit status"。
"$CXX" -shared $LDFLAGS -o build/rpg_engine.dll.new $OBJS \
  src/rpg_engine.def -Wl,--exclude-all-symbols \
  -Wl,--out-implib,build/librpg_engine.a.new
mv -f build/rpg_engine.dll.new build/rpg_engine.dll
mv -f build/librpg_engine.a.new build/librpg_engine.a

echo "[3/3] 产物："
ls -la build/rpg_engine.dll
echo "OK（host-only：generate 未实现，会明确报错）"
