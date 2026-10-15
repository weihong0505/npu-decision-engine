"""ModernBERT-base（ONNX INT4，HF 官方导出）在各设备上的编译与延迟基准。

复现下列结果 JSON —— 整理前它们来自 REPL 手抄、没有生产者脚本，因此不可复现：

===========================  ==========================================================
modernbert-npu.json          动态形状：NPU 编译失败，CPU/iGPU 可用
modernbert-npu-static.json   静态 seq_len=128：NPU 可编译，且快于 CPU
modernbert-tuned.json        NPU_TURBO / CACHE_DIR 配置扫描（**注意 ab 模式的裁决**）
modernbert-ab.json           TURBO 的配对 A/B（多轮、交替顺序）→ 真实增益
scaling.json                 静态长度 128→768 的 NPU/CPU/iGPU 延迟与 npu_over_cpu
===========================  ==========================================================

结论要点：**静态形状是 NPU 可编译的前提**；配置扫描给出的 +14.7% 在配对测量下不成立
（见 docs/03-约束与踩坑.md）。

用法::

    python src/bench/bench_encoder.py --mode static
    python src/bench/bench_encoder.py --mode all --repeat 20
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # -> src/
import paths  # noqa: E402
import harness as H  # noqa: E402  同目录

SCRIPT = "src/bench/bench_encoder.py"
MODEL = "models/ModernBERT-base/onnx/model_q4.onnx"


def load_model(core, seq_len: int | None = None):
    """读 model_q4.onnx；给了 seq_len 就把两个输入 reshape 成静态 [1, seq_len]。"""
    m = core.read_model(str(paths.BERT_ONNX_Q4))
    if seq_len:
        shapes = {}
        for inp in m.inputs:
            tail = [d.get_length() for d in inp.get_partial_shape()[2:]]
            shapes[inp.get_any_name()] = [1, seq_len] + tail
        m.reshape(shapes)
    return m


def _run_devices(core, model, feeds, devices, repeat, warmup=3):
    return {d: H.run_on(core, model, d, feeds, repeat, warmup) for d in devices}


# --------------------------------------------------------------------------- 模式


def mode_dynamic(core, args):
    """动态形状（[?,?] 不 reshape）→ NPU 编译失败。"""
    model = load_model(core)
    feeds = H.feeds_for(model, seq_len=args.dyn_len)
    payload = {
        "onnx": MODEL,
        "maxlen": args.dyn_len,
        "repeat": args.repeat,
        "devices": _run_devices(core, model, feeds, ["NPU", "CPU", "GPU.0"], args.repeat, args.warmup),
    }
    return H.write_result("modernbert-npu.json", payload, args.outdir, SCRIPT)


def mode_static(core, args):
    """静态 seq_len → NPU 可编译。"""
    model = load_model(core, args.seq_len)
    feeds = H.feeds_for(model, args.seq_len)
    payload = {
        "onnx": MODEL,
        "seq_len": args.seq_len,
        "repeat": args.repeat,
        "devices": _run_devices(core, model, feeds, ["NPU", "CPU", "GPU.0"], args.repeat, args.warmup),
    }
    return H.write_result("modernbert-npu-static.json", payload, args.outdir, SCRIPT)


def mode_tuned(core, args):
    """配置扫描。这是**非配对**的单次扫描，结论须由 --mode ab 复核。"""
    model = load_model(core, args.seq_len)
    feeds = H.feeds_for(model, args.seq_len)
    variants = [
        ("NPU 默认", "NPU", {}),
        ("NPU TURBO", "NPU", {"NPU_TURBO": "YES"}),
        ("NPU TURBO+CACHE", "NPU", {"NPU_TURBO": "YES", "CACHE_DIR": str(paths.NPU_CACHE)}),
        ("CPU", "CPU", {}),
        ("iGPU (GPU.0)", "GPU.0", {}),
        ("dGPU (GPU.1)", "GPU.1", {}),
    ]
    out = []
    base = None
    for label, device, config in variants:
        st = H.run_on(core, model, device, feeds, args.repeat, args.warmup, config)
        rec = {"label": label, "device": device, "config": config}
        rec.update(st)
        p50 = st.get("p50_ms")
        if base is None and p50:
            base = p50
        if base and p50:
            rec["vs_base_pct"] = round((base - p50) / base * 100, 1)
        out.append(rec)
        print(f"  {label:<18} {rec.get('p50_ms', rec.get('error', '?'))}")
    payload = {"onnx": MODEL, "seq_len": args.seq_len, "repeat": args.repeat, "variants": out}
    return H.write_result("modernbert-tuned.json", payload, args.outdir, SCRIPT)


def mode_ab(core, args):
    """NPU 默认 vs NPU_TURBO 的配对 A/B。

    设计要点（这是它比 tuned 更可信的原因）：
      - 两个配置都**只编译一次**，在轮次之间复用，排除编译耗时与缓存状态的干扰；
      - 每轮的测量顺序交替（偶数轮 A→B，奇数轮 B→A），抵消热漂移与顺序效应；
      - 记录每轮 p50 与轮间标准差 —— 增益若小于轮间波动，就是噪声而非收益。

    符号约定：A = 默认（TURBO 关），B = TURBO 开；
    `paired_rel_pct = (B - A) / A * 100`，**负值表示 TURBO 更快**。
    `true_gain_pct` = 各轮 paired_rel_pct 的均值。
    """
    model = load_model(core, args.seq_len)
    feeds = H.feeds_for(model, args.seq_len)
    cm_a, cs_a = H.compile_model(core, model, "NPU", {})
    cm_b, cs_b = H.compile_model(core, model, "NPU", {"NPU_TURBO": "YES"})

    rounds, pcts = [], []
    for r in range(args.rounds):
        if r % 2 == 0:
            a = H.time_infer(cm_a, feeds, args.ab_repeat, args.warmup)
            b = H.time_infer(cm_b, feeds, args.ab_repeat, args.warmup)
        else:
            b = H.time_infer(cm_b, feeds, args.ab_repeat, args.warmup)
            a = H.time_infer(cm_a, feeds, args.ab_repeat, args.warmup)
        rel = (b["p50_ms"] - a["p50_ms"]) / a["p50_ms"] * 100
        pcts.append(rel)
        rounds.append({"round": r + 1, "A_ms": a["p50_ms"], "B_ms": b["p50_ms"],
                       "ratio": round(b["p50_ms"] / a["p50_ms"], 4)})
        print(f"  round {r+1}: A={a['p50_ms']:.2f} B={b['p50_ms']:.2f} rel={rel:+.2f}%")

    payload = {
        "onnx": MODEL,
        "seq_len": args.seq_len,
        "method": "paired A/B, alternating order, compiled once per config, warmup discarded",
        "A": "NPU 默认", "B": "NPU TURBO=YES",
        "sign_convention": "paired_rel_pct = (B-A)/A*100 ; 负值 = TURBO 更快",
        "ab_repeat": args.ab_repeat,
        "compile_s": {"A_default": round(cs_a, 2), "B_turbo": round(cs_b, 2)},
        "rounds": rounds,
        "A_median": round(statistics.median([r["A_ms"] for r in rounds]), 3),
        "B_median": round(statistics.median([r["B_ms"] for r in rounds]), 3),
        "paired_rel_pct": [round(p, 2) for p in pcts],
        "true_gain_pct": round(statistics.fmean(pcts), 2),
        "round_spread_pct": round(max(pcts) - min(pcts), 2),
        "verdict": ("TURBO 增益在轮间波动范围内，判定为噪声" if abs(statistics.fmean(pcts)) < max(0.5, max(pcts) - min(pcts) if len(pcts) > 1 else 0)
                    else "TURBO 增益超出轮间波动，判定为真实收益"),
    }
    return H.write_result("modernbert-ab.json", payload, args.outdir, SCRIPT)


def mode_scaling(core, args):
    """静态长度扫描：NPU 在 128→768 全程快于 CPU（npu_over_cpu < 1）。"""
    out = []
    for L in args.lengths:
        model = load_model(core, L)
        feeds = H.feeds_for(model, L)
        row = {"L": L}
        for tag, dev in (("npu", "NPU"), ("cpu", "CPU"), ("igpu", "GPU.0")):
            st = H.run_on(core, model, dev, feeds, args.repeat, args.warmup)
            row[tag] = st.get("p50_ms")
            if st.get("error"):
                row[f"{tag}_error"] = st["error"][:200]
        if row.get("npu") and row.get("cpu"):
            row["npu_over_cpu"] = round(row["npu"] / row["cpu"], 2)
        out.append(row)
        print(f"  L={L:<5} npu={row.get('npu')} cpu={row.get('cpu')} igpu={row.get('igpu')} ratio={row.get('npu_over_cpu')}")
    return H.write_result("scaling.json", out, args.outdir, SCRIPT)


MODES = {
    "dynamic": mode_dynamic,
    "static": mode_static,
    "tuned": mode_tuned,
    "ab": mode_ab,
    "scaling": mode_scaling,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=[*MODES, "all"], default="all")
    ap.add_argument("--seq-len", type=int, default=128, dest="seq_len", help="静态序列长度")
    ap.add_argument("--dyn-len", type=int, default=61, dest="dyn_len", help="动态模式的推理长度")
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 384, 512, 768])
    ap.add_argument("--repeat", type=int, default=20, help="每设备计时次数")
    ap.add_argument("--ab-repeat", type=int, default=30, dest="ab_repeat", help="AB 每轮每配置计时次数")
    ap.add_argument("--rounds", type=int, default=4, help="AB 轮数")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--outdir", default=None, help="默认 output/results/rerun")
    args = ap.parse_args()

    core = H.make_core()
    print(f"可用设备: {H.available_devices(core)}")
    print(f"设备名:   {H.device_names(core)}")
    print(f"模型:     {MODEL}\n")

    todo = list(MODES) if args.mode == "all" else [args.mode]
    if "dynamic" in todo and args.mode == "all":
        todo.remove("dynamic")          # 动态模式的 NPU 编译必然失败，只在显式指定时跑
    for name in todo:
        print(f"=== mode: {name} ===")
        MODES[name](core, args)


if __name__ == "__main__":
    main()
