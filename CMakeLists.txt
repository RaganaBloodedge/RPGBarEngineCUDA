cmake_minimum_required(VERSION 3.24)
project(RPGBarEngineCUDA VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(MSVC)
  # /fp:precise 关掉浮点收缩 —— 与下面的 -ffp-contract=off 同一个目的（见注释）。
  add_compile_options(/O2 /utf-8 /fp:precise)
else()
  # -O2 只给 host（CXX）。**别让它也落到 CUDA 上**：Release 下 CMake 会给
  # nvcc 传 -O3（CMAKE_CUDA_FLAGS_RELEASE），再叠一个 -O2 会得到
  #   "nvcc warning : incompatible redefinition for option 'optimize'"
  # 且**实际以 -O2 生效** —— 一次"编译通过但把 CUDA 侧悄悄降档"的坑。
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O2> -Wall -Wextra)
endif()

# ------------------------------------------------------------------ CUDA
#
# CUDA **可选**：没有 nvcc 时只编 host 部分（ABI 层 + GGUF + 分词器）。
# 这样做的理由不是"聊胜于无"，而是：
#   * CI / 开发机上没有显卡时，仍然能构建、能跑 ABI 层与元数据解析的测试；
#   * 宿主拿到的错误信息是"本产物未编译 CUDA 后端"，而不是"加载 dll 失败"。
# ⚠️ 但这不改变一件事：**真正要生成，必须有 nvcc**。
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(RPG_HAS_CUDA ON)
  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    # C++20，与 host 侧一致 —— **不是偏好，是硬要求**：同源副本 dtypes.h 用了
    # `std::bit_cast`，nvcc 默认给的是 C++17，会报
    #   error: namespace "std" has no member "bit_cast"
    # 而报错位置在**标准库**里（看起来像环境问题，实际是标准版本）。
    # ⚠️ 同源副本一个字都不许改（DESIGN §3），所以只能在这里抬标准。
    set(CMAKE_CUDA_STANDARD 20)
  endif()
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # native = 按本机 GPU 定架构。⚠️ **别把这样编出来的产物发给别人** ——
    # 到别的卡上会"no kernel image is available"。分发时显式列，如
    #   -DCMAKE_CUDA_ARCHITECTURES=86;89;120
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  message(STATUS "CUDA 工具链：${CMAKE_CUDA_COMPILER}")
  message(STATUS "CUDA 目标架构：${CMAKE_CUDA_ARCHITECTURES}")
else()
  set(RPG_HAS_CUDA OFF)
  message(WARNING
    "没有找到 CUDA 工具链（nvcc）—— 将只构建 host 部分。"
    "GGUF / 分词器 / ABI 层可用，但 rpg_engine_generate 会明确报『未编译 CUDA 后端』。"
    "要真正生成，请先安装 CUDA Toolkit（≥12.8；sm_120 需要）。")
endif()

# ------------------------------------------------------------------ 源文件
#
# host 层：与计算无关的部分（GGUF 容器、分词器、ChatML、工具解析、JSON）。
# 这些是从 RPGBarEngine **同源复制**的 —— 两边的文本行为必须逐字一致，
# 详见 docs/DESIGN.md §3 与 §10（同步方式）。
set(RPG_HOST_SOURCES
  src/engine.cc
  src/model_meta.cc
  src/common.cc
  src/json.cc
  src/quant.cc
  src/gguf.cc
  src/tokenizer.cc
  src/chat_template.cc
  src/tool_parser.cc
  src/mmap_file.cc
  # 与 CUDA 无关的设备描述 + host-only 兜底：两条构建路径都编它（见该文件注释）。
  src/cuda_common.cc
)

# 计算层：**全部重写**（不跟上游同步，见 DESIGN §3 的复用边界表）。
set(RPG_CUDA_SOURCES
  src/cuda_common.cu
  src/gpu_model.cu
  src/kv_cache_gpu.cu
  src/kernels_gpu.cu
  src/qwen3_gpu.cu
  # 采样的设备端实现（P3）。⚠️ sampler.h 是**同源副本**（语义与默认值的唯一来源，
  # 在 RPG_HOST_SOURCES 之外单独说明 —— 它没有 .cc，实现由设备侧重写）；
  # sampler_gpu.cu 才是本仓库自己的代码。
  src/sampler_gpu.cu
)

# ------------------------------------------------------------------ object 库
#
# 为什么不是"直接把源文件编进 SHARED 目标"：tools/ 下的判据工具
# （layer_check / dequant_check）要用到引擎**内部的 C++ 符号**（rpg::GpuModel、
# rpg::GpuForwardFull），而共享库在 Windows 上被 rpg_engine.def 限制成只导出
# 那 10 个 C ABI 函数 —— 工具链不上。抽成 OBJECT 库之后，同一批 .o 既进共享库
# 也进工具，两边各取所需，且**不会出现两份编译结果不一致**。
add_library(rpg_engine_objs OBJECT ${RPG_HOST_SOURCES})
target_compile_definitions(rpg_engine_objs PRIVATE RPG_ENGINE_BUILD)
# -ffp-contract=off —— 这一行不是洁癖，是 P1 判据的前提：
#   设备端的反量化（quant_gpu.cuh）用 __fmul_rn/__fsub_rn 显式不做 FMA 收缩，
#   如果 host 侧的 quant.cc 把 `d1 * raw - m1` 缩成 FMA，两边就差 1 ULP，
#   而"反量化逐位对拍"这条判据会红一片 —— 看起来像是拆包写错了。
#   （顺带：CPU 引擎在 Windows/MinGW 上不带 -mfma，本来也不收缩，所以这条
#   反而让两边更接近，而不是更远。）
target_compile_options(rpg_engine_objs PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-ffp-contract=off>>)
target_include_directories(rpg_engine_objs PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_include_directories(rpg_engine_objs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(rpg_engine_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(RPG_HAS_CUDA)
  target_sources(rpg_engine_objs PRIVATE ${RPG_CUDA_SOURCES})
  # host 侧代码（engine.cc）也要知道自己编没编 CUDA —— 它据此选错误文案与
  # model_info 里的 backend 字段。
  target_compile_definitions(rpg_engine_objs PRIVATE RPG_HAS_CUDA=1)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(rpg_engine_objs PUBLIC CUDA::cudart)
endif()

# ------------------------------------------------------------------ 共享库
add_library(rpg_engine SHARED)
target_link_libraries(rpg_engine PRIVATE rpg_engine_objs)

if(WIN32)
  # 显式导出清单：不这么写，MinGW 会把内部 C++ 符号全导出（见 src/rpg_engine.def）。
  target_sources(rpg_engine PRIVATE src/rpg_engine.def)
  if(MINGW)
    target_link_options(rpg_engine PRIVATE -Wl,--exclude-all-symbols)
    # ⚠️ 全静态链接（-static）与 CUDA 运行时不兼容 —— Windows 上的 CUDA 构建
    # 请用 MSVC（nvcc 的默认宿主编译器），MinGW 只用于 host-only 构建。
    if(NOT RPG_HAS_CUDA)
      target_link_options(rpg_engine PRIVATE -static -static-libgcc -static-libstdc++)
    endif()
  else()
    set_target_properties(rpg_engine PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS OFF)
  endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(rpg_engine PRIVATE Threads::Threads)

set_target_properties(rpg_engine PROPERTIES
  OUTPUT_NAME rpg_engine
  POSITION_INDEPENDENT_CODE ON
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ------------------------------------------------------------------ 工具
#
# 工具都链 rpg_engine_objs（拿内部 C++ 符号），不是链共享库 —— 理由见上面
# object 库那段。tools/ 自己也要 -Itools（npy.h / cli_util.h）。
function(rpg_add_tool name)
  add_executable(${name} ${ARGN})
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tools)
  # 工具只在有 CUDA 时被创建，所以这里直接定义；少了它，gpu_model.h /
  # qwen3_gpu.h 会整段被 #ifdef 掉，报一堆"未声明"。
  target_compile_definitions(${name} PRIVATE RPG_HAS_CUDA=1)
  target_link_libraries(${name} PRIVATE rpg_engine_objs Threads::Threads)
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endfunction()

if(RPG_HAS_CUDA)
  # 工具链自检：报出设备名/算力/显存 + 一次 device→host 往返。P0 第二档判据。
  rpg_add_tool(cuda_check tools/cuda_check.cu)
  # P1 判据之一：反量化逐位对拍 + f16→f32 穷举。
  rpg_add_tool(dequant_check tools/dequant_check.cu)
  # P1 判据之二：整段前向，写 npy 供与 CPU 引擎对拍。
  rpg_add_tool(layer_check tools/layer_check.cu)
  # P2 判据之一：KV Cache 写入的**下标映射**（不需要模型，纯确定性）。
  # 单独立一条的理由：映射错了不会崩，只会读到别的头/别的位置的历史 —— 症状与
  # "kv 的 fp16 精度不够"长得一模一样，而后者是可接受的取舍。这条把两者分开。
  rpg_add_tool(kv_map_check tools/kv_map_check.cu)
  # P2 判据：增量解码 + 贪心生成 + KV 自检，写 gen.txt / step_logits.npy
  # 供与 CPU 引擎逐 token 对拍。注意**参考侧不在这里** —— 它是
  # tools/cpu_ref_gen.cc，由 tools/build_cpu_ref.sh 从 RPGBarEngine 的源码
  # 现编（理由见 DESIGN §7 末：参考实现必须是"真的在跑的那一份"）。
  rpg_add_tool(gen_check tools/gen_check.cu)
  # P3 判据之一：设备端采样（temperature / top_k / top_p + counter-based 随机源）。
  # **不需要模型、不需要权重** —— 两边喂同一份从文件来的 logits 就能逐 token 对拍
  # （理由见 tools/sample_check.cu 的头注释与 DESIGN §7）。参考侧是
  # tools/cpu_ref_sample.cc，由 build_cpu_ref.sh 从 CPU 引擎的 sampler.cc 现编。
  rpg_add_tool(sample_check tools/sample_check.cu)
endif()
