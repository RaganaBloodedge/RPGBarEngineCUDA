#!/usr/bin/env python3
"""P1 主检查：GPU 前向 vs CPU 引擎前向，逐层残差流 + logits。

## 与 CPU 仓库 tools/cmp_npy.py 的关系

统计量**完全一致**（`REL_FLOOR=1e-3` 的相对误差分位数，判据取 p99），所以两边的
数字可以直接对照。唯一的区别：这里允许 logits 缺失 —— `--layers N` 截断跑时
两边都不写 logits（截断模式不计算末端归一化与 lm_head），而 cmp_npy 硬要读它。

    Ours  = <ours>.hidden.npy / <ours>.logits.npy    由 tools/layer_check 写
    Ref   = <ref>.hidden.npy  / <ref>.logits.npy     由 CPU 引擎 tools/forward_check 写

## 判据为什么不是 max|Δ|

Qwen3 的残差流里有量级 ~1.2e4 的离群维度，float32 的 1 ULP 就有 1e-3 量级 ——
"绝对误差 < 1e-4" 是**不可能**满足的判据（与实现好坏无关）。所以看的是
**逐元素相对误差的分位数**：中位（大部分元素准不准）、p99（尾部准不准）。
max|Δ| 只作参考 —— 它被巨值维度与近零坐标两类元素主导，没有信息量。

判据：每一项（embedding + 每一层 + logits）的 p99 ≤ --tol（默认 1e-3，与
DESIGN §5 的 P1 判据一致）。

用法：
    cmp_layers.py <ours_prefix> <ref_prefix> [--tol 1e-3] [--top 5]
"""

import argparse
import os
import sys

import numpy as np

# 只统计 |ref| > 此值的元素，剔除"近零坐标"导致的相对误差失真。
# 与 RPGBarEngine/tools/cmp_npy.py 取值相同，便于两边对照。
REL_FLOOR = 1e-3


def rel_stats(ours, ref):
    d = np.abs(ours.astype(np.float64) - ref.astype(np.float64))
    a = np.abs(ref.astype(np.float64))
    m = a > REL_FLOOR
    if not m.any():
        return 0.0, 0.0, 0.0, float(d.max())
    rel = d[m] / a[m]
    return (float(np.median(rel)), float(np.percentile(rel, 99)), float(rel.max()),
            float(d.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours", help="GPU 侧前缀（tools/layer_check 的 --out）")
    ap.add_argument("ref", help="CPU 侧前缀（forward_check 的 --out）")
    ap.add_argument("--tol", type=float, default=1e-3, help="p99 相对误差上限")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    miss = [p + s for p, s in ((args.ours, ".hidden.npy"), (args.ref, ".hidden.npy"))
            if not os.path.exists(p + s)]
    if miss:
        print("缺文件：", ", ".join(miss), file=sys.stderr)
        return 2

    ours_hidden = np.load(args.ours + ".hidden.npy")
    ref_hidden = np.load(args.ref + ".hidden.npy")
    if ours_hidden.shape != ref_hidden.shape:
        print(f"hidden 形状不同：我们 {ours_hidden.shape} vs 参考 {ref_hidden.shape}",
              file=sys.stderr)
        return 2

    n_all, seq, hidden = ours_hidden.shape
    n_layers = n_all - 1

    logits = None
    if os.path.exists(args.ours + ".logits.npy") and os.path.exists(
            args.ref + ".logits.npy"):
        ours_logits = np.load(args.ours + ".logits.npy")
        ref_logits = np.load(args.ref + ".logits.npy")
        if ours_logits.shape != ref_logits.shape:
            print(f"logits 形状不同：我们 {ours_logits.shape} vs 参考 {ref_logits.shape}",
                  file=sys.stderr)
            return 2
        logits = (ours_logits, ref_logits)
    else:
        print("（没有 logits.npy —— 截断模式，只比逐层残差流）")

    # token 序列必须完全一致，否则后面比的东西不是同一个输入。
    ours_tok = args.ours + ".tokens.txt"
    ref_tok = args.ref + ".tokens.txt"
    if os.path.exists(ours_tok) and os.path.exists(ref_tok):
        a = open(ours_tok, encoding="utf-8").read().strip()
        b = open(ref_tok, encoding="utf-8").read().strip()
        if a != b:
            print("**两侧的 token 序列不同** —— 对拍无意义，先修这个：", file=sys.stderr)
            print(f"  我们: {a[:200]}", file=sys.stderr)
            print(f"  参考: {b[:200]}", file=sys.stderr)
            return 2
        print(f"[输入] {seq} 个 token，两侧一致")

    print(f"\n[逐层残差流] {n_layers + 1} 项（embedding + {n_layers} 层），"
          f"相对误差（|ref| > {REL_FLOOR:g}），判据 p99 <= {args.tol:g}")
    hdr = f"  {'项':<18}{'中位':>11}{'p99':>11}{'max':>11}{'max|Δ|':>12}"
    print(hdr)

    ok_all = True
    worst_p99 = 0.0
    first_bad = None
    for k in range(n_all):
        label = "embedding" if k == 0 else f"layer {k - 1} 输出"
        med, p99, mx, amax = rel_stats(ours_hidden[k], ref_hidden[k])
        worst_p99 = max(worst_p99, p99)
        ok = p99 <= args.tol
        print(f"  {label:<18}{med:>11.3g}{p99:>11.3g}{mx:>11.3g}{amax:>12.4g}"
              f"   {'OK' if ok else '**超限**'}")
        if not ok:
            ok_all = False
            if first_bad is None:
                first_bad = k

    if logits is not None:
        ours_logits, ref_logits = logits
        print(f"\n[输出层] logits{ours_logits.shape}（覆盖 output_norm + lm_head）")
        print(hdr)
        med, p99, mx, amax = rel_stats(ours_logits, ref_logits)
        worst_p99 = max(worst_p99, p99)
        ok = p99 <= args.tol
        print(f"  {'logits':<18}{med:>11.3g}{p99:>11.3g}{mx:>11.3g}{amax:>12.4g}"
              f"   {'OK' if ok else '**超限**'}")
        if not ok:
            ok_all = False

        print("\n[离散一致性] 每个位置的 argmax 必须相同（差一点就够生成跑偏）")
        n_bad = 0
        for t in range(ours_logits.shape[0]):
            if int(np.argmax(ours_logits[t])) != int(np.argmax(ref_logits[t])):
                n_bad += 1
                print(f"  位置 {t}: argmax 不同 —— 我们 {int(np.argmax(ours_logits[t]))}"
                      f" vs 参考 {int(np.argmax(ref_logits[t]))}")
        ok_all &= n_bad == 0
        if n_bad == 0:
            print(f"  {ours_logits.shape[0]} 个位置的 argmax 全部一致")

        lo, lr = ours_logits[-1], ref_logits[-1]
        print(f"  最后位置 top-{args.top} 我们: " + ", ".join(
            f"{int(i)}({lo[i]:.4f})" for i in np.argsort(-lo)[:args.top]))
        print(f"  最后位置 top-{args.top} 参考: " + ", ".join(
            f"{int(i)}({lr[i]:.4f})" for i in np.argsort(-lr)[:args.top]))

    print()
    if ok_all:
        print(f"PASS —— 全部 {n_all} 项满足 p99 相对误差 <= {args.tol:g}"
              + ("，argmax 全一致" if logits is not None else ""))
    else:
        which = ("embedding" if first_bad == 0 else f"第 {first_bad - 1} 层之后") \
            if first_bad is not None else "logits"
        print(f"FAIL —— 第一处超限在 {which}；最差 p99 = {worst_p99:.3g}")
    print(f"（最差 p99 = {worst_p99:.3g}，判据 {args.tol:g}）")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
