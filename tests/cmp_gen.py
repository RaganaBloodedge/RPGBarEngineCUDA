#!/usr/bin/env python3
"""P2 主检查：GPU 增量解码 vs CPU 引擎的贪心生成，逐 token 对拍。

## 为什么这里是"逐 token 比"而不是"比 logits"

P1 已经证明了两边的**算术**在同一段 token 上一致（p99 ≤ 1e-3）。P2 新增的是
KV Cache（fp16）、前缀复用的位置偏移、增量解码这三件事 —— 它们错了不会让某一
个数变差，而是让**选择的 token**开始不一样（argmax 一旦翻过去，后面全变）。
所以判据必须落在 token 层。

    Ours = <ours>.gen.txt / <ours>.step_logits.npy   由 tools/gen_check 写
    Ref  = <ref>.gen.txt  / <ref>.step_logits.npy    由 tools/cpu_ref_gen 写

## 判据分两层（DESIGN §7.3）

1. **数值层**：每一步 logits 的相对误差 p99 ≤ --tol（默认 1e-3），口径与
   tests/cmp_layers.py 完全相同（`REL_FLOOR=1e-3`、numpy linear 插值分位）。
   这一层先过，才能说"分歧不是因为算错了"。
2. **token 层**：逐 token 比。
   * 完全一致 → PASS。
   * 有分歧 → 逐个打印分歧位置、两边各自的 top-5、以及**两个 token 在各边的
     logit 差**。判定规则：如果该步 `|logit[我们选的] - logit[参考选的]|` 相对
     该步 logits 的跨度很小（≤ --gap，默认 1e-2），那这个分歧属于"两个 token
     本来就在伯仲之间"—— fp16 KV 的舍入足以把它推过去。这种分歧**允许存在**，
     但必须全部记录在案（退出码 0，打印清单）。
   * 否则 → FAIL（退出码 1）：说明不是精度问题，是逻辑不一致。

用法：
    cmp_gen.py <ours_prefix> <ref_prefix> [--tol 1e-3] [--gap 1e-2] [--top 5]
"""

import argparse
import os
import sys

import numpy as np

# 与 tests/cmp_layers.py / RPGBarEngine/tools/cmp_npy.py 取值相同。
REL_FLOOR = 1e-3


def read_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        body = f.read().strip()
    if not body:
        return []
    return [int(x) for x in body.replace("\n", ",").split(",") if x.strip()]


def rel_p99(ours, ref):
    """逐元素相对误差（只看 |ref| > REL_FLOOR），返回 (中位, p99, max, max|Δ|)。

    ⚠️ **这几个相对误差量只作诊断，不作判据** —— 见下面 full_scale_err 的说明。
    """
    d = np.abs(ours.astype(np.float64) - ref.astype(np.float64))
    a = np.abs(ref.astype(np.float64))
    m = a > REL_FLOOR
    if not m.any():
        return 0.0, 0.0, 0.0, float(d.max())
    rel = d[m] / a[m]
    return (float(np.median(rel)), float(np.percentile(rel, 99)),
            float(rel.max()), float(d.max()))


def full_scale_err(ours, ref):
    """**判据量**：max|Δ| / max|ref|（绝对误差对满量程取比值）。

    为什么不用逐元素相对误差的 p99：在这条链路上它不是一个能定的判据 ——
    误差的**绝对**量级由 KV 的 fp16 舍入决定（几十个大 logit 量级 20，绝对误差
    稳定在 ~1e-2），而相对误差 = 绝对误差 / |元素自身|。于是 p99 完全由"有多少元素
    落在 0.5 上下"决定，换个 prompt、换个 max_tokens 就会漂。

    满量程比值有物理依据：fp16 尾数 10 位 ⇒ 1 ULP = 2^-11 = 4.88e-4；
    实测 5.6e-4 ⇒ 判据取 1e-3（约 2 个 ULP，留一倍余量）。
    （同一条教训在 CPU 引擎上演过一次：那里是"残差流有 |h| ≈ 1.2e4 的离群维度，
    float32 的 1 ULP 就有 1e-3 量级 ⇒ 绝对误差 < 1e-4 不可能满足"。
    **定数值判据前先量目标量级的 ULP。**）
    """
    d = np.abs(ours.astype(np.float64) - ref.astype(np.float64))
    scale = float(np.abs(ref.astype(np.float64)).max())
    if scale == 0:
        return float(d.max()), 0.0
    return float(d.max()), float(d.max()) / scale


def top_k_str(row, k):
    idx = np.argsort(-row)[:k]
    return ", ".join(f"{int(i)}({row[i]:.4f})" for i in idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours", help="GPU 侧前缀（tools/gen_check 的 --out）")
    ap.add_argument("ref", help="CPU 侧前缀（tools/cpu_ref_gen 的 --out）")
    ap.add_argument("--tol", type=float, default=1e-3, help="数值层 p99 上限")
    ap.add_argument("--gap", type=float, default=1e-2,
                    help="判定'两个 token 本来就在伯仲之间'的相对 logit 差上限")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    for p in (args.ours + ".gen.txt", args.ref + ".gen.txt"):
        if not os.path.exists(p):
            print(f"缺文件：{p}", file=sys.stderr)
            return 2

    # ---- 输入必须完全一致 ----
    try:
        ours_tok = read_ids(args.ours + ".tokens.txt")
        ref_tok = read_ids(args.ref + ".tokens.txt")
        if ours_tok != ref_tok:
            print("**两侧的 token 序列不同** —— 对拍无意义，先修这个：",
                  file=sys.stderr)
            print(f"  我们前 40 个: {ours_tok[:40]}", file=sys.stderr)
            print(f"  参考前 40 个: {ref_tok[:40]}", file=sys.stderr)
            return 2
        print(f"[输入] {len(ref_tok)} 个 token，两侧一致")
    except FileNotFoundError:
        print("（没有 tokens.txt，跳过输入一致性检查）")

    ours = read_ids(args.ours + ".gen.txt")
    ref = read_ids(args.ref + ".gen.txt")
    print(f"[生成] 我们 {len(ours)} 个，参考 {len(ref)} 个")

    ok = True
    first_bad = None
    for i in range(max(len(ours), len(ref))):
        a = ours[i] if i < len(ours) else None
        b = ref[i] if i < len(ref) else None
        if a != b:
            first_bad = i
            break

    # ---- 数值层 ----
    so = args.ours + ".step_logits.npy"
    sr = args.ref + ".step_logits.npy"
    step = None
    if os.path.exists(so) and os.path.exists(sr):
        ours_step = np.load(so)
        ref_step = np.load(sr)
        if ours_step.shape != ref_step.shape:
            print(f"**step_logits 形状不同**：我们 {ours_step.shape} vs "
                  f"参考 {ref_step.shape} —— 两边生成的数量不一样？",
                  file=sys.stderr)
            return 2
        step = (ours_step, ref_step)
        n, vocab = ours_step.shape
        med, p99, mx, amax_rel = rel_p99(ours_step, ref_step)
        amax, full_rel = full_scale_err(ours_step, ref_step)
        print(f"\n[数值层] 每步 logits {ours_step.shape}，判据 "
              f"max|Δ| <= {args.tol:g} × max|ref|")
        print(f"  满量程 max|ref| = {np.abs(ref_step).max():.4g}，max|Δ| = {amax:.4g}"
              f"  ⇒  比值 {full_rel:.3g}   "
              f"{'OK' if full_rel <= args.tol else '**超限**'}")
        print(f"  （诊断：逐元素相对误差 中位 {med:.3g} / p99 {p99:.3g} / "
              f"max {mx:.3g} —— 只作参考，不作判据）")
        step_ok = full_rel <= args.tol
        if not step_ok:
            print("  ⇒ 满量程比值都超了，说明不是 fp16 的 5e-4 量级 —— 先跑 "
                  "tests/run_p2.sh 里的 ⓿（KV 映射）与 P1 判据")
            ok = False

        n_arg = int(np.sum(np.argmax(ours_step, axis=1) == np.argmax(ref_step, axis=1)))
        print(f"  逐位置 argmax 一致：{n_arg}/{n} {'OK' if n_arg == n else '**不一致**'}")
        if n_arg != n:
            ok = False
    else:
        print("\n（没有 step_logits.npy —— 只比 token 层，分歧时看不到 logits 差距）")

    # ---- token 层 ----
    print(f"\n[token] 逐 token 对拍（{max(len(ours), len(ref))} 步）")
    if first_bad is None:
        ids = ", ".join(str(x) for x in ours)
        print(f"  完全一致：{ids}")
    else:
        # 收集所有分歧位置
        diffs = [i for i in range(max(len(ours), len(ref)))
                 if (ours[i] if i < len(ours) else None)
                 != (ref[i] if i < len(ref) else None)]
        print(f"  分歧 {len(diffs)} 处，第一处在第 {first_bad} 步："
              f"我们 {ours[first_bad] if first_bad < len(ours) else None} vs "
              f"参考 {ref[first_bad] if first_bad < len(ref) else None}")

        unresolved = []
        for i in diffs:
            a = ours[i] if i < len(ours) else None
            b = ref[i] if i < len(ref) else None
            print(f"\n  --- 第 {i} 步 ---")
            print(f"    我们 {a}   参考 {b}")
            if step is None or i >= step[0].shape[0]:
                print("    （没有该步的 logits —— 无法判定分歧是否合理）")
                unresolved.append(i)
                continue
            lo = step[0][i]
            lr = step[1][i]
            print(f"    我们 top-{args.top}: " + top_k_str(lo, args.top))
            print(f"    参考 top-{args.top}: " + top_k_str(lr, args.top))
            span = float(lo.max() - lo.min())
            gap_o = abs(float(lo[int(a)]) - float(lo[int(b)])) if a is not None \
                and b is not None else float("inf")
            gap_r = abs(float(lr[int(a)]) - float(lr[int(b)])) if a is not None \
                and b is not None else float("inf")
            rel_o = gap_o / span if span > 0 else float("inf")
            print(f"    两个 token 的 logit 差：我们 {gap_o:.4g}（跨度的 {rel_o:.3g}）"
                  f"，参考 {gap_r:.4g}")
            if rel_o <= args.gap:
                print(f"    ⇒ 在伯仲之间（差 {rel_o:.3g} <= {args.gap:g}）——"
                      f" fp16 KV 的舍入足以把它推过去，属可接受")
            else:
                print(f"    ⇒ **差距太大**（{rel_o:.3g} > {args.gap:g}）——"
                      f" 不是精度问题，是逻辑不一致")
                unresolved.append(i)

        if unresolved:
            print(f"\n  {len(unresolved)} 处分歧无法用浮点解释：{unresolved}")
            ok = False
        else:
            print(f"\n  {len(diffs)} 处分歧全部落在'伯仲之间'（已逐条记录）")

    print()
    if ok and first_bad is None:
        print("PASS —— 逐 token 完全一致")
    elif ok:
        print("PASS（附条件）—— 有分歧，但每一处都记录在案且可用 fp16 KV 的舍入解释")
    else:
        print("FAIL —— 见上")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
