#!/usr/bin/env python3
"""check_sync.py —— 校验"同源副本"有没有和上位仓库跑偏。

## 它解决什么问题

本仓库的 src/ 里有一部分文件是从
[RPGBarEngine](https://github.com/RaganaBloodedge/RPGBarEngine) **逐字复制**来的
（对照表在 docs/DESIGN.md §3）。这是有意为之，不是懒：

    对拍要能得出"差异只来自算力层"这个结论，
    前提是文本层/容器层在两边**一个字都不差**。

所以"复制得对不对"不是一次性的动作，而是一条**随时可以重跑的判据**。这个脚本
就是那条判据。改了任何一边的副本之后跑一遍，能立刻看出哪几个文件分叉了。

## 用法

    # upstream 默认取 ../RPGBarEngine（两个仓库并排放的情况）
    python tests/check_sync.py

    # 也可以显式指定
    python tests/check_sync.py --upstream E:/Workspace/RPGBarEngine

    # CI / 脚本里用：有分叉时退出码非 0
    python tests/check_sync.py --quiet

退出码：0 = 全部一致；1 = 有分叉或缺失；2 = 找不到上位仓库（不算错，只提示）。

## 分叉了怎么办

- 上游修了文本层的 bug → 按 docs/DESIGN.md §10 跟过来
  （`git checkout upstream/main -- src/gguf.cc src/gguf.h`），再跑一遍本脚本确认。
- 本仓库**有意**要改某个副本 → 那说明它不该再留在"复用"那一栏，
  要同时改 docs/DESIGN.md §3 的表格。**不要偷偷改**。
"""
from __future__ import annotations

import argparse
import os
import sys

# 与 docs/DESIGN.md §3 的"复用"那一栏一一对应。
# 加文件/挪文件时，这里和那张表要同时改。
REUSED_SRC = [
    # 容器解析
    "bytes.h", "gguf.h", "gguf.cc", "mmap_file.h", "mmap_file.cc",
    # 文本层
    "tokenizer.h", "tokenizer.cc",
    "chat_template.h", "chat_template.cc",
    "tool_parser.h", "tool_parser.cc",
    "unicode.h", "unicode_tables.h",
    # 基础设施
    "json.h", "json.cc", "common.h", "common.cc", "dtypes.h",
    # 反量化（host 侧复用；GPU 侧另有实现）
    "quant.h", "quant.cc",
    # 采样参数的**定义**（SampleOptions 的字段/默认值/语义是契约）。
    # ⚠️ 只有头文件是副本：sampler.cc 依赖 CPU 侧的 kernels.h（ArgMax / SoftmaxInplace），
    # 搬过来等于把 CPU 前向也搬进来 —— 那是 §3 明确要"重写"的一栏。
    # 设备端实现在 src/sampler_gpu.*，参考侧由 tools/build_cpu_ref.sh 现编 sampler.cc。
    "sampler.h",
]

# 这两个是 ABI 契约，两边必须逐字相同（docs/DESIGN.md §2）。
ABI_FILES = [
    ("include/rpg_engine.h", "include/rpg_engine.h"),
    ("src/rpg_engine.def", "src/rpg_engine.def"),
]

# tools/ 下的两个也是逐字复制来的（npy 读写 + 平台无关的命令行工具）。
# 为什么要一起盯着：判据脚本要拿同一套 npy 布局和同一套读文件写法，
# 两边不一致会表现为"对拍脚本报形状不符"，而去查模型 —— 又是一次归因跑偏。
# ⚠️ 复制过来之后**不许改**；要改就先改上游，再同步（docs/DESIGN.md §10）。
REUSED_TOOLS = ["npy.h", "cli_util.h"]


def human_size(n: int) -> str:
    for unit in ("B", "KiB", "MiB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.0f} GiB"


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)                       # 仓库根
    default_up = os.path.join(os.path.dirname(root), "RPGBarEngine")

    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=default_up,
                    help="上位仓库 RPGBarEngine 的路径")
    ap.add_argument("--quiet", action="store_true", help="只报分叉，不报每一项")
    args = ap.parse_args()

    up = os.path.abspath(args.upstream)
    if not os.path.isdir(os.path.join(up, "src")):
        print(f"找不到上位仓库：{up}")
        print("（用 --upstream 指定 RPGBarEngine 的路径；两个仓库并排时无需指定）")
        return 2

    print(f"本仓库：{root}")
    print(f"上位仓库：{up}\n")

    same = diff = missing = 0
    bad: list[str] = []

    def compare(mine_rel: str, up_rel: str) -> None:
        nonlocal same, diff, missing
        a = os.path.join(root, mine_rel)
        b = os.path.join(up, up_rel)
        if not os.path.isfile(b):
            missing += 1
            bad.append(f"上游没有这个文件：{up_rel}")
            print(f"  [上游缺]  {mine_rel:<28} ← {up_rel}")
            return
        if not os.path.isfile(a):
            missing += 1
            bad.append(f"本仓库缺文件：{mine_rel}")
            print(f"  [本仓库缺] {mine_rel}")
            return
        ba = open(a, "rb").read()
        bb = open(b, "rb").read()
        if ba == bb:
            same += 1
            if not args.quiet:
                print(f"  [一致]    {mine_rel:<28} {human_size(len(ba))}")
        else:
            diff += 1
            # 逐行找第一个不同的地方，省得人再手动 diff
            la = ba.splitlines()
            lb = bb.splitlines()
            where = "行数不同"
            for i in range(min(len(la), len(lb))):
                if la[i] != lb[i]:
                    where = f"第 {i + 1} 行不同"
                    break
            bad.append(f"{mine_rel}（{where}）")
            print(f"  [分叉]    {mine_rel:<28} {where}"
                  f"  本仓库 {human_size(len(ba))} vs 上游 {human_size(len(bb))}")

    print("== ABI 契约（改了就是破坏性变更，两边要同时升版本号）==")
    for mine, up_rel in ABI_FILES:
        compare(mine, up_rel)

    print("\n== 同源副本（docs/DESIGN.md §3 的『复用』栏）==")
    for f in REUSED_SRC:
        compare(os.path.join("src", f), os.path.join("src", f))

    print("\n== 同源副本（tools/，同上）==")
    for f in REUSED_TOOLS:
        compare(os.path.join("tools", f), os.path.join("tools", f))

    print(f"\n结果：{same} 一致，{diff} 分叉，{missing} 缺失")
    if bad:
        print("\n需要处理：")
        for b in bad:
            print(f"  - {b}")
        print("\n跟法见 docs/DESIGN.md §10。若是**有意**分叉，"
              "要同时改 §3 的复用表 —— 不要偷偷改。")
        return 1
    print("同源副本与上位仓库逐字节一致。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
