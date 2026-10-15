"""JEV-lite（ModernBERT 150M + Linear(d,1) 标签头）在 NPU 上的基准。

复现下列结果 JSON：

=========================  ==================================================================
jev-npu-final.json         **整图**丢给 NPU：编译失败（图里有 22 个 IsNaN），只有 CPU 能跑
jev-hybrid-final.json      **拆图**：NPU 只跑编码器、head 回 CPU → 端到端大幅快于纯 CPU
=========================  ==================================================================

模型切分（见 engine/model.py:54-59）::

    h = encoder(input_ids, attention_mask).last_hidden_state   # [1, L, 768]
    picked = h.gather(1, label_pos[..., None].expand(-1, -1, 768))
    logits = Linear(768, 1)(dropout(picked)).squeeze(-1)       # [1, M]

head 只有 `score.weight[1,768]` + `score.bias[1]` 两个张量，所以可以用 numpy 逐位复现，
从而在不依赖 torch 前向的前提下量出 head 的真实耗时（整理前的记录是 0.199 ms）。

用法::

    python src/bench/bench_jev.py --mode full     --seq 768 --labels 18
    python src/bench/bench_jev.py --mode hybrid   --encoder-onnx output/onnx/jev_encoder.onnx
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # -> src/
import paths  # noqa: E402
import harness as H  # noqa: E402  同目录

SCRIPT = "src/bench/bench_jev.py"


def static_full_model(core, seq: int, labels: int):
    """jevlite.onnx：把 [?,?] 固定成 [1,seq] / [1,labels]。NPU 只接受静态形状。"""
    m = core.read_model(str(paths.JEV_ONNX))
    shapes = {"input_ids": [1, seq], "attention_mask": [1, seq], "label_pos": [1, labels]}
    m.reshape({k: v for k, v in shapes.items() if k in [i.get_any_name() for i in m.inputs]})
    return m


def load_head(ckpt=None):
    """从检查点取出 head 的两个张量。"""
    import torch
    ck = torch.load(ckpt or paths.require(paths.JEV_CKPT), map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    return (sd["score.weight"].numpy().astype(np.float32),
            sd["score.bias"].numpy().astype(np.float32),
            float(ck.get("val_acc", 0.0)), ck.get("max_len"))


def head_logits(hidden: np.ndarray, label_pos: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """numpy 版 head：gather(label_pos) → Linear(768,1)。等价于 engine/model.py:54-59 的 torch 实现。

    hidden ``[B, L, D]``、label_pos ``[B, M]`` → 返回 ``[B, M]``。
    用 take_along_axis 而不是 hidden[0][pos[0]]，这样批大小不受限，
    并且与 torch 的 ``h.gather(1, ...)`` 语义逐位对应（见 tests/test_head_parity.py）。
    """
    picked = np.take_along_axis(hidden, label_pos[..., None], axis=1)   # [B, M, D]
    return (picked @ W.T + b)[..., 0]                                   # [B, M]


def time_head(hidden, label_pos, W, b, repeat=200):
    for _ in range(5):
        head_logits(hidden, label_pos, W, b)
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        head_logits(hidden, label_pos, W, b)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return round(statistics.median(ts), 3), repeat


# --------------------------------------------------------------------------- 模式


def mode_full(core, args):
    """整图上 NPU：预期编译失败；CPU 可跑，作为后续混合方案的对照。

    注意 NPU 部分走**子进程隔离**：本机上编译含 `IsNaN` 的整图会让 OpenVINO 原生层
    直接段错误（0xC0000005），进程级崩溃 try/except 抓不到，会静默带走整个基准。
    """
    model = static_full_model(core, args.seq, args.labels)
    feeds = H.feeds_for(model, args.seq)

    cm_cpu = core.compile_model(model, "CPU")
    cpu_out = H.infer_outputs(cm_cpu, feeds)
    cpu_stats = H.time_infer(cm_cpu, feeds, args.repeat, args.warmup)

    variants = []
    for label, config in (("NPU 默认", {}), ("NPU TURBO", {"NPU_TURBO": "YES"})):
        st = H.run_isolated_device(paths.JEV_ONNX, "NPU", args.seq, args.labels, config,
                                   args.repeat, args.warmup)
        npy = st.pop("_output_npy", None)
        rec = {"label": label, "device": "NPU", "config": config}
        rec.update(st)
        if npy:
            rec["max_abs_err"] = H.max_abs_err(cpu_out, [np.load(npy)])
        variants.append(rec)
        if "p50_ms" in rec:
            print(f"  {label:<12} {rec['p50_ms']} ms   err={rec.get('max_abs_err')}")
        else:
            print(f"  {label:<12} FAILED: {rec.get('error', '')[:110]}")

    cpu_rec = {"label": "CPU", "device": "CPU", "config": {}, "max_abs_err": 0.0}
    cpu_rec.update(cpu_stats)
    variants.append(cpu_rec)
    print(f"  {'CPU':<12} {cpu_stats['p50_ms']} ms")

    payload = {"onnx": "output/onnx/jevlite.onnx", "seq": args.seq, "labels": args.labels,
               "repeat": args.repeat, "variants": variants,
               "isolation_note": "NPU 尝试在子进程内执行；段错误会被记为子进程退出码而非中断本次基准。"}
    return H.write_result("jev-npu-final.json", payload, args.outdir, SCRIPT)


def mode_hybrid(core, args):
    """拆图：编码器 ONNX 放设备上跑，head 用 numpy 在 CPU 上跑。"""
    enc_path = pathlib.Path(args.encoder_onnx)
    if not enc_path.is_absolute():
        enc_path = paths.PROJECT_ROOT / enc_path
    if not enc_path.exists():
        raise FileNotFoundError(f"找不到编码器 ONNX：{enc_path}")

    W, b, val_acc, max_len = load_head(args.ckpt)

    enc = core.read_model(str(enc_path))
    enc_feeds = H.feeds_for(enc, args.seq)
    for k in list(enc_feeds):
        if k not in [i.get_any_name() for i in enc.inputs]:
            del enc_feeds[k]

    full = static_full_model(core, args.seq, args.labels)
    full_feeds = H.feeds_for(full, args.seq)

    # 参考：整图 CPU 前向（既是数值基准，也是「不拆图」的对照延迟）
    cm_cpu_full = core.compile_model(full, "CPU")
    ref_out = H.infer_outputs(cm_cpu_full, full_feeds)

    runs = []
    for device, config in (("NPU", {}), ("CPU", {}), ("GPU.0", {})):
        try:
            if device == "NPU":
                # 子进程隔离：本机 NPU 编译此编码器图会直接段错误（见 harness.run_isolated_device）
                st = H.run_isolated_device(enc_path, "NPU", args.seq, None, config,
                                           args.repeat, args.warmup)
                npy = st.pop("_output_npy", None)
                if "error" in st:
                    runs.append({"label": f"混合: {device}编码器 + CPU head", "device": device,
                                 "error": st["error"], "stderr_tail": st.get("stderr_tail", "")})
                    print(f"  {device:<6} FAILED: {st['error'][:110]}")
                    continue
                hidden = np.load(npy)
                compile_s, p50_ms = st.get("compile_s"), st["p50_ms"]
            else:
                cm, cs = H.compile_model(core, enc, device, config)
                stt = H.time_infer(cm, enc_feeds, args.repeat, args.warmup)
                hidden = np.asarray(cm.create_infer_request().infer(enc_feeds)[cm.outputs[0]])
                compile_s, p50_ms = round(cs, 2), stt["p50_ms"]

            hp50, _ = time_head(hidden, full_feeds["label_pos"], W, b, args.head_repeat)
            hybrid = head_logits(hidden, full_feeds["label_pos"], W, b)
            runs.append({
                "label": f"混合: {device}编码器 + CPU head",
                "device": device,
                "compile_s": compile_s,
                "encoder_ms": p50_ms,
                "head_ms": hp50,
                "end_to_end_ms": round(p50_ms + hp50, 2),
                "max_abs_err": H.max_abs_err(ref_out, [hybrid]),
                "isolated": device == "NPU",
            })
            print(f"  {device:<6} encoder={p50_ms} ms + head={hp50} ms "
                  f"= {runs[-1]['end_to_end_ms']} ms   err={runs[-1]['max_abs_err']:.2e}")
        except Exception as e:
            runs.append({"label": f"混合: {device}编码器 + CPU head", "device": device,
                         "error": str(e)[:500]})
            print(f"  {device:<6} FAILED: {str(e)[:120]}")

    # 对照：整图 CPU（唯一能跑通整图的设备）
    full_cpu = H.time_infer(cm_cpu_full, full_feeds, args.repeat, args.warmup)
    ref = {"cpu_full_model_ms": full_cpu["p50_ms"]}
    if args.npu_full_model_ms:
        ref["npu_full_model_ms"] = args.npu_full_model_ms

    ok = [r for r in runs if "end_to_end_ms" in r]
    best = min(ok, key=lambda r: r["end_to_end_ms"]) if ok else None
    payload = {
        "encoder_onnx": str(enc_path.relative_to(paths.PROJECT_ROOT)).replace("\\", "/"),
        "seq": args.seq, "labels": args.labels, "repeat": args.repeat,
        "ckpt_val_acc": val_acc, "ckpt_max_len": max_len,
        "runs": runs + [{"label": "对照: CPU整图", "device": "CPU",
                         "end_to_end_ms": full_cpu["p50_ms"], "p50_ms": full_cpu["p50_ms"]}],
        "reference": ref,
        "best": best,
    }
    return H.write_result("jev-hybrid-final.json", payload, args.outdir, SCRIPT)


def mode_verify(core, args):
    """核对 output/onnx 里的编码器导出与当前 jevlite.pt 是不是同一份权重。

    为什么需要这个模式：2026-09-22 的 NPU 实验全部跑在 02:46 导出的权重上，
    而 `jevlite.pt` 在当天 12:38 被 `03_calibrate.py` 重新落盘过。
    标定只往 state_dict 里写 `log_temp` 并附 `temperatures`、不动编码器权重，
    所以两者**应当**等价 —— 但当时没有任何脚本验证过。
    这是"顺序隐患"，本模式把它变成一条可执行的检查（只需 CPU，不需要 NPU）。
    """
    import torch

    paths.add_engine_to_path()
    from model import JevLite

    ck = torch.load(paths.require(paths.JEV_CKPT), map_location="cpu", weights_only=False)
    model = JevLite(ck["encoder"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    enc_ov = core.read_model(str(paths.JEV_ENCODER_ONNX))
    feeds = H.feeds_for(enc_ov, args.seq)
    ids = torch.as_tensor(feeds["input_ids"])
    mask = torch.as_tensor(feeds["attention_mask"])
    label_pos = torch.zeros(1, args.labels, dtype=torch.long)

    with torch.no_grad():
        h_torch = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state.numpy()
        lg_torch = model.logits(ids, mask, label_pos).numpy()

    report = {
        "ckpt": "output/onnx/jevlite.pt",
        "ckpt_encoder_field": ck.get("encoder"),
        "ckpt_temperatures": ck.get("temperatures"),
        "ckpt_max_len": ck.get("max_len"),
        "seq": args.seq, "labels": args.labels,
        "note": "对比 onnx 导出与当前检查点的 torch 前向。整图与编码器两条路径都要看："
                "整图一致而编码器不一致 → 编码器专用导出过期；两者都不一致 → 检查点权重本身与导出不同源。",
        "comparisons": [],
    }

    # (a) 整图：torch logits vs jevlite.onnx
    try:
        full = static_full_model(core, args.seq, args.labels)
        cm = core.compile_model(full, "CPU")
        out = np.asarray(cm.create_infer_request().infer(H.feeds_for(full, args.seq))[cm.outputs[0]])
        err = float(np.max(np.abs(out - lg_torch)))
        report["full_model"] = {"onnx": "jevlite.onnx", "max_abs_err": err,
                                "verdict": "一致" if err < 1e-3 else "不一致"}
        print(f"  {'jevlite.onnx (整图)':<28} max_abs_err={err:.3e}  {report['full_model']['verdict']}")
    except Exception as e:
        report["full_model"] = {"onnx": "jevlite.onnx", "error": str(e)[:300]}
        print(f"  jevlite.onnx FAILED: {str(e)[:100]}")

    # (b) 编码器：torch last_hidden_state vs 各编码器专用导出
    ov_enc_outputs = {}
    for p in (paths.JEV_ENCODER_ONNX, paths.ENC_DYNAMO_SDPA):
        if not p.exists():
            continue
        try:
            m = core.read_model(str(p))
            cm = core.compile_model(m, "CPU")
            key_feeds = {k: v for k, v in feeds.items() if k in [i.get_any_name() for i in m.inputs]}
            out = np.asarray(cm.create_infer_request().infer(key_feeds)[cm.outputs[0]])
            ov_enc_outputs[p.name] = out
            err = float(np.max(np.abs(out - h_torch)))
            rec = {"onnx": p.name, "max_abs_err": err, "verdict": "一致" if err < 1e-3 else "不一致",
                   "output_shape": list(out.shape),
                   "l2_norm_onnx": round(float(np.linalg.norm(out)), 3),
                   "corr_with_torch": round(float(np.corrcoef(out.ravel(), h_torch.ravel())[0, 1]), 6)}
            print(f"  {p.name:<28} max_abs_err={err:.3e}  |onnx|={rec['l2_norm_onnx']}  "
                  f"corr={rec['corr_with_torch']}  {rec['verdict']}")
        except Exception as e:
            rec = {"onnx": p.name, "error": str(e)[:300]}
            print(f"  {p.name:<28} FAILED: {str(e)[:100]}")
        report["comparisons"].append(rec)

    report["torch_encoder_l2_norm"] = round(float(np.linalg.norm(h_torch)), 3)
    report["torch_encoder_shape"] = list(h_torch.shape)
    print(f"  {'torch encoder (基准)':<28} |h|={report['torch_encoder_l2_norm']}  shape={report['torch_encoder_shape']}")

    # (c) 两个编码器导出之间是否自洽（若它们彼此一致、却都与 torch 不一致，问题在对比口径而非权重）
    names = list(ov_enc_outputs)
    if len(names) == 2:
        a, bb = ov_enc_outputs[names[0]], ov_enc_outputs[names[1]]
        if a.shape == bb.shape:
            cross = float(np.max(np.abs(a - bb)))
            report["cross_onnx_max_abs_err"] = cross
            print(f"  {'两个 ONNX 互相之间':<28} max_abs_err={cross:.3e}")

    # (d) 短序列交叉检验 —— 把「权重不同」与「长序列注意力语义不同」一刀切开。
    #     ModernBERT 每 3 层一次全局注意力、其余用 128 的滑窗；L=128 时两者等价，
    #     所以：短序列一致 + 长序列不一致 ⇒ 差异在长序列注意力语义（滑窗/位置处理），不是权重。
    short = 128
    feeds_s = H.feeds_for(enc_ov, short)
    with torch.no_grad():
        h_s = model.encoder(input_ids=torch.as_tensor(feeds_s["input_ids"]),
                            attention_mask=torch.as_tensor(feeds_s["attention_mask"])).last_hidden_state.numpy()
    short_rows = []
    for p in (paths.JEV_ENCODER_ONNX, paths.ENC_DYNAMO_SDPA):
        if not p.exists():
            continue
        try:
            m2 = core.read_model(str(p))
            m2.reshape({i.get_any_name(): [1, short] for i in m2.inputs})
            cm2 = core.compile_model(m2, "CPU")
            key_feeds = {k: v for k, v in feeds_s.items() if k in [i.get_any_name() for i in m2.inputs]}
            out = np.asarray(cm2.create_infer_request().infer(key_feeds)[cm2.outputs[0]])
            err = float(np.max(np.abs(out - h_s)))
            short_rows.append({"onnx": p.name, "seq": short, "max_abs_err": err})
            print(f"  [{p.name} @L={short}]".ljust(30) + f" max_abs_err={err:.3e}")
        except Exception as e:
            short_rows.append({"onnx": p.name, "seq": short, "error": str(e)[:200]})
    report["short_seq_check"] = {
        "seq": short,
        "comparisons": short_rows,
        "reading": "短序列一致而长序列不一致 → 差异来自长序列注意力语义（滑窗/位置），非权重；"
                   "两者都不一致 → 权重确实不同源。",
    }
    return H.write_result("verify-encoder-weights.json", report, args.outdir, SCRIPT)


MODES = {"full": mode_full, "hybrid": mode_hybrid, "verify": mode_verify}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=[*MODES, "all"], default="hybrid")
    ap.add_argument("--seq", type=int, default=768, help="序列长度（真实 p50 为 588，max_len=768）")
    ap.add_argument("--labels", type=int, default=18, help="label_pos 宽度")
    ap.add_argument("--encoder-onnx", default="output/onnx/jev_encoder.onnx", dest="encoder_onnx",
                    help="编码器专用 ONNX（也可用 output/onnx/enc_dynamo_sdpa.onnx）")
    ap.add_argument("--ckpt", default=None, help="默认 output/onnx/jevlite.pt")
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--head-repeat", type=int, default=200, dest="head_repeat")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--npu-full-model-ms", type=float, default=None, dest="npu_full_model_ms",
                    help="整图 NPU 延迟（由 --mode full 得到，用于对照记录）")
    ap.add_argument("--outdir", default=None, help="默认 output/results/rerun")
    args = ap.parse_args()

    core = H.make_core()
    print(f"可用设备: {H.available_devices(core)}\n")
    for name in (list(MODES) if args.mode == "all" else [args.mode]):
        print(f"=== mode: {name} ===")
        MODES[name](core, args)


if __name__ == "__main__":
    main()
