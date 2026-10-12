"""abi_smoke.py —— C ABI 的冒烟测试。

两种跑法，都不需要 CUDA：

    # 1) 只验 ABI 形状与错误路径（不需要权重，CI 里可跑）
    python tests/abi_smoke.py --lib build/rpg_engine.dll

    # 2) 带上权重：额外验 load_model / model_info / generate 的报错文案
    python tests/abi_smoke.py --lib build/rpg_engine.dll --model models/Qwen3-1.7B-Q4_K_M.gguf

它验的是**宿主能看到的东西**：函数能不能调、返回值形状对不对、错误路径说的话
是不是人话。不验数值 —— 那是 P1/P2 的对拍工作（见 docs/DESIGN.md §7）。
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys

PASS = 0
FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f"  {extra}" if extra else ""))


def bind(lib_path: str):
    """按 include/rpg_engine.h 的签名绑定。**名字与类型必须逐字对上** ——
    宿主（Unity P/Invoke / Python ctypes）就是这么绑的，这里是它的替身。"""
    lib = ctypes.CDLL(lib_path)

    lib.rpg_engine_version.restype = ctypes.c_char_p
    lib.rpg_engine_last_error.restype = ctypes.c_char_p

    lib.rpg_engine_load_model.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.rpg_engine_load_model.restype = ctypes.c_int32

    lib.rpg_engine_unload_model.argtypes = [ctypes.c_int32]
    lib.rpg_engine_unload_model.restype = None

    lib.rpg_engine_session_create.argtypes = [ctypes.c_int32, ctypes.c_char_p]
    lib.rpg_engine_session_create.restype = ctypes.c_int32

    lib.rpg_engine_session_destroy.argtypes = [ctypes.c_int32]
    lib.rpg_engine_session_destroy.restype = None

    lib.rpg_engine_session_reset.argtypes = [ctypes.c_int32]
    lib.rpg_engine_session_reset.restype = ctypes.c_int32

    # ⚠️ generate / model_info 返回 char*：**必须用 c_void_p 接**。
    # 声明成 c_char_p 的话 ctypes 会把它转成 bytes 对象、原指针当场丢掉，
    # 于是 rpg_engine_free 无从调用 —— 每次生成漏几十 KB（Unity 侧那条同款坑）。
    lib.rpg_engine_generate.argtypes = [ctypes.c_int32, ctypes.c_char_p]
    lib.rpg_engine_generate.restype = ctypes.c_void_p

    lib.rpg_engine_model_info.argtypes = [ctypes.c_int32]
    lib.rpg_engine_model_info.restype = ctypes.c_void_p

    lib.rpg_engine_free.argtypes = [ctypes.c_void_p]
    lib.rpg_engine_free.restype = None

    return lib


def take_str(lib, ptr) -> str:
    """读一个引擎分配的字符串并释放它。"""
    if not ptr:
        return ""
    s = ctypes.cast(ptr, ctypes.c_char_p).value or b""
    lib.rpg_engine_free(ptr)
    return s.decode("utf-8", "replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=os.path.join("build", "rpg_engine.dll"))
    ap.add_argument("--model", default="")
    args = ap.parse_args()

    if not os.path.isfile(args.lib):
        print(f"找不到动态库：{args.lib}（先跑 build_host.sh 或 cmake --build build）")
        return 2

    print(f"== ABI 冒烟（{args.lib}）==")
    lib = bind(args.lib)

    ver = (lib.rpg_engine_version() or b"").decode()
    check("version() 返回非空", bool(ver), f"得到 {ver!r}")
    print(f"  引擎版本：{ver}")

    # 错误路径：句柄无效时必须失败，且 last_error 是人话。
    bad = lib.rpg_engine_session_create(-999, None)
    check("无效模型句柄建会话 → 返回负数", bad < 0, f"得到 {bad}")
    err = (lib.rpg_engine_last_error() or b"").decode("utf-8", "replace")
    check("last_error 给出可读原因", "无效的模型句柄" in err, f"得到 {err!r}")

    bad2 = lib.rpg_engine_load_model(None, None)
    check("model_path 为空 → 返回负数", bad2 < 0, f"得到 {bad2}")

    bad3 = lib.rpg_engine_load_model(b"Z:/nope/nope.gguf", None)
    check("文件不存在 → 返回负数（不崩溃）", bad3 < 0, f"得到 {bad3}")

    bad4 = lib.rpg_engine_load_model(b"Z:/nope.gguf", b"{not json}")
    check("config 不是合法 JSON → 返回负数", bad4 < 0, f"得到 {bad4}")

    # 无害操作：无效句柄的 unload / destroy / free 必须是空操作。
    lib.rpg_engine_unload_model(-999)
    lib.rpg_engine_session_destroy(-999)
    lib.rpg_engine_free(None)
    check("无效句柄的 unload/destroy/free 是空操作", True)

    if not args.model:
        print("\n（没给 --model，跳过模型相关断言）")
    elif not os.path.isfile(args.model):
        print(f"\n（--model 指向的文件不存在：{args.model}，跳过）")
    else:
        print(f"\n== 模型：{args.model} ==")
        m = lib.rpg_engine_load_model(args.model.encode(), None)
        check("load_model 成功", m >= 0, f"句柄={m}，err={lib.rpg_engine_last_error()}")

        if m >= 0:
            info = take_str(lib, lib.rpg_engine_model_info(m))
            o = json.loads(info) if info else {}
            # 键名与 CPU 版逐字对齐 —— 宿主靠它们取值，换实现不该让宿主改代码。
            for key in ("arch", "name", "n_layers", "hidden", "n_heads", "n_kv_heads",
                        "head_dim", "intermediate", "vocab", "context_length",
                        "rope_theta", "tied_embeddings", "path", "max_pos",
                        "default_max_seq", "weight_bytes", "sessions"):
                check(f"model_info 有 {key}", key in o)
            check("model_info 标出 backend", o.get("backend") in ("cuda", "host-only"),
                  str(o.get("backend")))
            print(f"  {o.get('name')} · {o.get('n_layers')} 层 · vocab {o.get('vocab')} "
                  f"· backend={o.get('backend')}")

            s = lib.rpg_engine_session_create(m, b'{"max_seq": 512, "max_tokens": 32}')
            check("session_create 成功", s >= 0, f"句柄={s}")
            if s >= 0:
                check("session_reset 成功", lib.rpg_engine_session_reset(s) == 0)
                # generate 必须**明确报错**，而不是返回一段假文本。
                # P1/P2 之后这条依然要守：整网前向与增量解码都已经跑通并对拍过
                # （tools/layer_check、tools/gen_check），但它们**没接进 ABI** ——
                # 报错文案必须把这个区别说清，否则宿主会以为是"没 GPU / 没编
                # CUDA"，去查错方向。
                out = lib.rpg_engine_generate(s, b'{"messages":[{"role":"user","content":"hi"}]}')
                check("generate 尚未实现 → 返回 NULL", not out)
                gerr = (lib.rpg_engine_last_error() or b"").decode("utf-8", "replace")
                check("generate 的报错说明原因（不是空话）",
                      ("未编译 CUDA" in gerr) or ("还没接进 ABI" in gerr),
                      f"得到 {gerr!r}")
                print(f"  generate 的报错：{gerr.splitlines()[0] if gerr else '(空)'}")
                lib.rpg_engine_session_destroy(s)

            info2 = take_str(lib, lib.rpg_engine_model_info(m))
            check("销毁会话后 sessions 归零",
                  json.loads(info2).get("sessions") == 0 if info2 else False)

            lib.rpg_engine_unload_model(m)
            check("unload 之后句柄失效",
                  lib.rpg_engine_model_info(m) is None or
                  (lib.rpg_engine_last_error() or b"").decode("utf-8", "replace") != "")

    print(f"\n结果：{PASS} 通过，{FAIL} 失败")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
