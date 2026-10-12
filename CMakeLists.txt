cmake_minimum_required(VERSION 3.24)
project(RPGBarEngineCUDA VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(MSVC)
  add_compile_options(/O2 /utf-8)
else()
  add_compile_options(-O2 -Wall -Wextra)
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
    set(CMAKE_CUDA_STANDARD 17)
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

set(RPG_CUDA_SOURCES
  src/cuda_common.cu
)

add_library(rpg_engine SHARED ${RPG_HOST_SOURCES})
target_compile_definitions(rpg_engine PRIVATE RPG_ENGINE_BUILD)
target_include_directories(rpg_engine PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_include_directories(rpg_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(RPG_HAS_CUDA)
  target_sources(rpg_engine PRIVATE ${RPG_CUDA_SOURCES})
  # host 侧代码（engine.cc）也要知道自己编没编 CUDA —— 它据此选错误文案与
  # model_info 里的 backend 字段。
  target_compile_definitions(rpg_engine PRIVATE RPG_HAS_CUDA=1)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(rpg_engine PRIVATE CUDA::cudart)
endif()

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
if(RPG_HAS_CUDA)
  add_executable(cuda_check
    tools/cuda_check.cu
    src/cuda_common.cu
    src/common.cc
  )
  target_include_directories(cuda_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_definitions(cuda_check PRIVATE RPG_HAS_CUDA=1)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(cuda_check PRIVATE CUDA::cudart Threads::Threads)
  set_target_properties(cuda_check PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
