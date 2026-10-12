#!/usr/bin/env python3
"""fetch_model.py —— 下载引擎需要的权重与对拍参考物。

下载到 models/（已在 .gitignore 里，不进版本库）。

用法：
    python scripts/fetch_model.py --preset qwen3-1.7b-q4km
    python scripts/fetch_model.py --preset qwen3-1.7b-q8
    python scripts/fetch_model.py --preset qwen3-1.7b-q4km --refs-only
    python scripts/fetch_model.py --list
    python scripts/fetch_model.py --all-small      # Q8_0 + Q4_K_M

关于镜像：huggingface.co 在国内直连不通，脚本默认走 hf-mirror.com（与 RPGBar
的 scripts/fetch_embedding.py 同一策略）。可用 --endpoint 换回官方站。

断点续传：用 HTTP Range 从中断处接着下。1.8 GB 的文件在抖动的网络下基本没有
一次到位的，所以这不是可选项。

与上位仓库（RPGBarEngine）的 scripts/fetch_model.py 是同一份，只改了这段说明：
那边下的是 CPU 版的权重，这边下的是**同一批文件** —— 两个引擎共用一份 GGUF，
"同一份权重、两种算力"正是对拍策略（docs/DESIGN.md §7）能成立的前提。
"""
import argparse
import os
import sys
import urllib.error
import urllib.request

MIRROR = "https://hf-mirror.com"
OFFICIAL = "https://huggingface.co"

# preset -> (仓库, 文件名, 说明)
PRESETS = {
    "qwen3-1.7b-q8": (
        "Qwen/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q8_0.gguf",
        "官方 Q8_0，1.71 GiB。数值上几乎无损，作为单层/整网对拍的基准。"),
    "qwen3-1.7b-q4km": (
        "unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf",
        "Q4_K_M，1.03 GiB。目标出货档位（llama.cpp 生态的标准 4-bit 混合量化），"
        "也是 RPGBar 默认用的那一份。"),
    "qwen3-1.7b-q5km": (
        "unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q5_K_M.gguf",
        "Q5_K_M，1.17 GiB。质量/体积折中档，备用。"),
    "qwen3-4b-q4km": (
        "unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf",
        "大档 DM 权重，Q4_K_M，约 2.5 GiB。"),
}

# 对拍参考物：只用官方 Qwen3-1.7B 的分词器定义与 chat 模板。
#
# 这一层（分词器 / ChatML 渲染）在本仓库里是从 RPGBarEngine **逐字复制**来的
# （docs/DESIGN.md §3）。P1/P2 的对拍要能得出"差异只来自算力层"这个结论，
# 前提就是这些副本没被改过 —— 参考物就是用来确认这一点的。
REFS = [
    ("Qwen/Qwen3-1.7B", "tokenizer.json", "qwen3-1.7b-tokenizer.json",
     "官方分词器定义（HF tokenizers 格式）。与 src/tokenizer.cc 的输出逐 token 对齐。"),
    ("Qwen/Qwen3-1.7B", "tokenizer_config.json", "tokenizer_config.json",
     "官方 chat_template 原文（Jinja 字符串）。src/chat_template.cc 的 ChatML\n"
     "     渲染就是照它复刻的 —— 没有它就没法逐字节对拍。"),
]


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def download(endpoint, repo, filename, dest, resume=True):
    url = f"{endpoint}/{repo}/resolve/main/{filename}"
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    have = os.path.getsize(dest) if (resume and os.path.exists(dest)) else 0

    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:      # 已经下完了
            print(f"  已是最新：{dest}")
            return True
        print(f"  HTTP {e.code}：{e.reason}", file=sys.stderr)
        return False
    except Exception as e:      # noqa: BLE001
        print(f"  连接失败：{e}", file=sys.stderr)
        return False

    total = None
    ctype = resp.headers.get("Content-Range")
    if ctype and "/" in ctype:
        total = int(ctype.split("/")[-1])
        if have and have == total:
            resp.close()
            print(f"  已完整：{dest}（{human(total)}）")
            return True
    elif resp.headers.get("Content-Length"):
        total = have + int(resp.headers["Content-Length"])

    mode = "ab" if have else "wb"
    done = have
    try:
        with open(dest, mode) as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100.0 / total
                    sys.stdout.write(
                        f"\r  {filename}: {pct:5.1f}%  {human(done)} / {human(total)}")
                    sys.stdout.flush()
    finally:
        resp.close()
    sys.stdout.write("\n")
    if total and done != total:
        print(f"  !! 未下完：{done} / {total}（重跑本脚本可续传）", file=sys.stderr)
        return False
    print(f"  完成：{dest}（{human(done)}）")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", action="append", default=[])
    ap.add_argument("--all-small", action="store_true",
                    help="下载 Q8_0 与 Q4_K_M 两个档位")
    ap.add_argument("--refs-only", action="store_true",
                    help="只下对拍参考物（分词器定义），不下权重")
    ap.add_argument("--endpoint", default=MIRROR)
    ap.add_argument("--out", default="models")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("可用 preset：")
        for k, (repo, fn, desc) in PRESETS.items():
            print(f"  {k:<18} {fn:<28} {desc}")
        print("\n对拍参考物：")
        for repo, fn, out, desc in REFS:
            print(f"  {fn:<28} -> {out:<28} {desc}")
        return 0

    presets = list(args.preset)
    if args.all_small:
        presets += ["qwen3-1.7b-q8", "qwen3-1.7b-q4km"]
    if not presets and not args.refs_only:
        print("没指定 preset。用 --list 看可选项，或 --all-small。", file=sys.stderr)
        return 2

    ok = True
    for name in presets:
        if name not in PRESETS:
            print(f"未知 preset：{name}", file=sys.stderr)
            ok = False
            continue
        repo, fn, desc = PRESETS[name]
        print(f"[{name}] {desc}")
        ok &= download(args.endpoint, repo, fn, os.path.join(args.out, fn))

    print("[对拍参考物]")
    for repo, fn, out_name, desc in REFS:
        print(f"  {desc}")
        ok &= download(args.endpoint, repo, fn, os.path.join(args.out, out_name))

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
