#!/usr/bin/env python3
"""P3 主检查：设备端采样 vs CPU 引擎的采样，**喂同一份 logits** 逐 token 对拍。

## 为什么这一块能判"逐位相同"

DESIGN §6 第 10 条写过"采样不是能对拍的那一半"。那句话针对的是**接在模型后面**
的采样：两边 logits 本来就差 fp16 的舍入（P2 的判据是"满量程比 ≤ 1e-3"），采样
把这个差异放大成"整段序列不同"，于是只能退成"统计上说得过去"。

把**模型摘掉**之后判据立刻回来：两边读的是同一份 `logits.bin`（逐位相同），
`FilterToProbs` 与 `SampleToken` 都是纯函数 ⇒ 同一个 seed 必须给出同一个 token。

    Ours = <dir>/gpu_probs.npy / gpu_tokens.txt / gpu_u.npy   由 tools/sample_check 写
    Ref  = <dir>/ref_probs.npy / ref_tokens.txt / ref_u.npy   由 tools/cpu_ref_sample 写

## 判据三条（DESIGN §5「P3 的判据」）

1. **u 表**：`CounterUniform(seed, counter)` 只有整数运算 + 乘一个 2 的幂 ⇒
   判据是**逐位相同**。这一条红说明随机源实现分叉了（它是最容易"改一边忘另一边"
   的地方 —— CPU 版在 RPGBarEngine 的 sampler.cc 里，本仓库是重写的）。
2. **概率分布**：满量程误差 ≤ --tol，且**保留集合完全相同**。
   保留集合（哪些 token 没被 top_k / top_p 筛掉）是个**离散判断** —— 它比"数值
   接近"更能说明问题：筛错了集合，后面采样出来的 token 一定是错的。
3. **token**：逐位相同。若分歧，打印该 u、参考侧的累积概率、以及 **u 到边界的
   距离**；只有距离 < --border（默认 1e-6）才允许存在 —— 那是"GPU 的 expf 与
   glibc 的 expf 差 1 ULP"足以把 u 推过一格的唯一位置（P2 那条"两个 token 本来
   就在伯仲之间"的同一个纪律）。

用法：
    cmp_sample.py <dir> [--tol 1e-5] [--border 1e-6]
"""

import argparse
import sys

import numpy as np


def read_cases(path):
    """读 cases.tsv（GPU 侧写出、参考侧读入的**同一份**用例矩阵）。"""
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            f8 = line.split("\t")
            if len(f8) != 8:
                raise SystemExit(f"cases.tsv 里这一行的字段数不是 8：{line}")
            cases.append(
                dict(
                    idx=int(f8[0]),
                    temperature=float(f8[1]),
                    top_k=int(f8[2]),
                    top_p=float(f8[3]),
                    seed=int(f8[4]),
                    tie=int(f8[5]),
                    u_count=int(f8[6]),
                    vocab=int(f8[7]),
                )
            )
    if not cases:
        raise SystemExit("cases.tsv 里一条用例都没有")
    return cases


def read_tokens(path):
    with open(path, "r", encoding="utf-8") as f:
        body = f.read().strip()
    if not body:
        return []
    return [[int(x) for x in line.split()] for line in body.splitlines() if line.strip()]


def full_scale_err(ours, ref):
    """**判据量**：max|Δ| / max|ref|（绝对误差对满量程取比值）。

    与 P2 的 KV 自检同一个口径：采样分布的绝对值在 [0,1] 里，用相对误差会
    在 0 附近的元素上爆掉（那些元素本来就是被筛掉的 0）。
    """
    d = np.abs(ours.astype(np.float64) - ref.astype(np.float64))
    scale = float(np.abs(ref.astype(np.float64)).max())
    if scale <= 0.0:
        return float(d.max()), 0.0
    return float(d.max()), float(d.max()) / scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="sample_check 与 cpu_ref_sample 共同的输出目录")
    ap.add_argument("--tol", type=float, default=1e-5,
                    help="概率分布的满量程误差上限（默认 1e-5）")
    ap.add_argument("--border", type=float, default=1e-6,
                    help="u 距累积概率边界的允许距离（默认 1e-6）")
    args = ap.parse_args()

    d = args.dir.rstrip("/")
    cases = read_cases(f"{d}/cases.tsv")
    vocab = cases[0]["vocab"]

    gpu_probs = np.load(f"{d}/gpu_probs.npy")
    ref_probs = np.load(f"{d}/ref_probs.npy")
    gpu_u = np.load(f"{d}/gpu_u.npy")
    ref_u = np.load(f"{d}/ref_u.npy")
    gpu_tok = read_tokens(f"{d}/gpu_tokens.txt")
    ref_tok = read_tokens(f"{d}/ref_tokens.txt")
    logits = np.load(f"{d}/logits.npy")

    print(f"用例 {len(cases)} 个，词表 {vocab}")
    print(f"  gpu_probs {gpu_probs.shape}  ref_probs {ref_probs.shape}  "
          f"logits {logits.shape}")
    if gpu_probs.shape != ref_probs.shape:
        print(f"[FAIL] 概率数组形状不一致：{gpu_probs.shape} vs {ref_probs.shape}")
        return 1
    if gpu_probs.shape != (len(cases), vocab):
        print(f"[FAIL] 概率数组形状与 cases.tsv 不符：{gpu_probs.shape} "
              f"vs ({len(cases)}, {vocab})")
        return 1

    bad = []

    # ---------------------------------------------------------------- ① u 表
    if gpu_u.shape != ref_u.shape:
        bad.append(f"u 表形状不一致：{gpu_u.shape} vs {ref_u.shape}")
    else:
        # 逐位比：两边都是 float32，先看位模式，再看数值（NaN 时位模式更可靠）。
        a = np.ascontiguousarray(gpu_u, dtype=np.float32).view(np.uint32)
        b = np.ascontiguousarray(ref_u, dtype=np.float32).view(np.uint32)
        n_diff = int((a != b).sum())
        print(f"\n① u 表：{a.size} 个值，位模式不同的 {n_diff} 个")
        if n_diff:
            i = int(np.argmax((a != b).flatten()))
            bad.append(f"u 表有 {n_diff} 个值位模式不同（第一个在第 "
                       f"{i // a.shape[1]} 行第 {i % a.shape[1]} 列："
                       f"{gpu_u.flatten()[i]!r} vs {ref_u.flatten()[i]!r}）")

    # ---------------------------------------------------------------- ② 概率
    print("\n② 概率分布（满量程误差 / 保留集合）")
    print(f"{'用例':<5}{'内容参数':<34}{'max|Δ|':<12}{'满量程比':<12}{'保留集合'}")
    worst = 0.0
    for c in cases:
        i = c["idx"]
        go, ro = gpu_probs[i], ref_probs[i]
        mx, fs = full_scale_err(go, ro)
        worst = max(worst, fs)
        keep_g = np.nonzero(go > 0.0)[0]
        keep_r = np.nonzero(ro > 0.0)[0]
        same_set = keep_g.size == keep_r.size and np.array_equal(keep_g, keep_r)
        param = (f"T={c['temperature']:g} k={c['top_k']} p={c['top_p']:g}"
                 f"{' 并列' if c['tie'] else ''}")
        print(f"{i:<5}{param:<34}{mx:<12.3e}{fs:<12.3e}"
              f"{keep_g.size}{'' if same_set else ' ← 不同！'}")
        if fs > args.tol:
            bad.append(f"用例 {i} 的概率分布满量程误差 {fs:.3e} > {args.tol:g}")
        if not same_set:
            only_g = np.setdiff1d(keep_g, keep_r)
            only_r = np.setdiff1d(keep_r, keep_g)
            bad.append(
                f"用例 {i} 的保留集合不同（GPU {keep_g.size} 个 / 参考 {keep_r.size} 个；"
                f"只在 GPU 里 {only_g[:5]}{'...' if only_g.size > 5 else ''}，"
                f"只在参考里 {only_r[:5]}{'...' if only_r.size > 5 else ''}）")

    # ---------------------------------------------------------------- ③ token
    print("\n③ 采样结果（逐 token）")
    if len(gpu_tok) != len(ref_tok):
        bad.append(f"token 行数不一致：{len(gpu_tok)} vs {len(ref_tok)}")
    else:
        n_total = 0
        n_diff = 0
        for c in cases:
            i = c["idx"]
            if len(gpu_tok[i]) != len(ref_tok[i]):
                bad.append(f"用例 {i} 的 token 个数不同："
                           f"{len(gpu_tok[i])} vs {len(ref_tok[i])}")
                continue
            n_total += len(gpu_tok[i])
            # 参考侧的累积概率（float32 逐元素累加到 float64 —— 与 CPU 的
            # `cum += (double)probs[i]` 是同一串加法，所以能直接用来算边界距离）。
            cum = np.cumsum(ref_probs[i].astype(np.float64))
            diffs = []
            for u_i, (g, r) in enumerate(zip(gpu_tok[i], ref_tok[i])):
                if g == r:
                    continue
                n_diff += 1
                uu = float(ref_u[i][u_i])
                # 距离边界多远：|u - 参考侧在该下标处的累积概率|
                dist = abs(uu - float(cum[r]))
                diffs.append((u_i, uu, g, r, dist))
            if diffs:
                for (u_i, uu, g, r, dist) in diffs[:5]:
                    print(f"  [分歧] 用例 {i} u#{u_i}: GPU {g} vs 参考 {r}"
                          f"（u={uu:.9f}，距边界 {dist:.3e}"
                          f"{'，在容差内' if dist < args.border else '，**超容差**'}）")
                for (_u_i, _uu, _g, _r, dist) in diffs:
                    if dist >= args.border:
                        bad.append(
                            f"用例 {i} 的 token 分歧不在边界容差内"
                            f"（距边界 {dist:.3e} ≥ {args.border:g}）—— 这不是 "
                            f"1 ULP 能推过去的，是筛选或逆变换不一致")
                        break
        print(f"  {len(cases)} 个用例 × 每例 {cases[0]['u_count']} 个 u = "
              f"{n_total} 个 token，不同 {n_diff} 个")

        # 贪心用例：结果必须与 u 无关，且等于 argmax（这是"贪心不消耗随机数"的
        # 外部可见形式）。GPU 侧工具内部也断言了，这里再独立算一遍。
        for c in cases:
            if c["temperature"] > 0.0:
                continue
            i = c["idx"]
            want = int(np.argmax(logits[i]))
            got = set(gpu_tok[i]) | set(ref_tok[i])
            if got != {want}:
                bad.append(f"贪心用例 {i} 应给 argmax {want}，实际出现 {sorted(got)}")

        # u 真的在起作用吗（防止"筛选/采样写错了、结果恒等于 argmax"也能过）：
        # 拿"纯分布"那个用例（T=1、无筛选）来验，16 个 u 里出现多个不同 token。
        for c in cases:
            if c["temperature"] == 1.0 and c["top_k"] == 0 and c["top_p"] == 1.0 \
                    and not c["tie"]:
                if len(set(gpu_tok[c["idx"]])) < 2:
                    bad.append(f"用例 {c['idx']}（纯分布）16 个 u 给出同一个 token —— "
                               f"u 没起作用，采样可能退化成了 argmax")
                break

    # ---------------------------------------------------------------- 结论
    print()
    if bad:
        print(f"[FAIL] {len(bad)} 项不通过：")
        for b in bad:
            print(f"  - {b}")
        return 1
    print(f"[PASS] u 表逐位相同；概率满量程误差最大 {worst:.3e}（判据 {args.tol:g}）"
          f"且保留集合一致；token 逐位相同。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
