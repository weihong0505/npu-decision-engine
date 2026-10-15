#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM 推理基准（设备对比：CPU / iGPU(GPU.0) / dGPU(GPU.1) / NPU）。

背景
----
2026-09 那次设备对比实验（结论：**NPU 在自回归解码上远慢于 CPU**）的基准脚本已经
丢失，只剩下 ``output/results/`` 下的三份结果 JSON。本脚本的作用就是把那次实验
重新变成可复现的：读同样的 prompt、同样的 ``max_new``、同样的 repeat 次数，产出
同样 schema 的结果文件。

推理路径选择
------------
**首选** ``openvino_genai.LLMPipeline``：模型目录 ``models/Qwen2.5-1.5B-Instruct-int4-ov``
里带的是 ``openvino_tokenizer.xml`` / ``openvino_detokenizer.xml`` 配对的
``openvino_model.xml``（OV 官方 INT4 导出格式），genai 能直接吃这套产物，且不走
optimum 的 torch 前端，最接近当年那条路径。
**回退** ``optimum.intel.OVModelForCausalLM``：只有当 ``import openvino_genai``
失败时才启用（回退理由会写进结果 JSON 的 ``backend`` 字段，不再靠人脑记）。
本机实测 genai 可用（openvino-genai 2026.4.0.0），所以实际走的是第一条路径。

与旧结果的已知差异（详见下方「口径自定」）
------------------------------------------
- 旧文件里 ``model`` 写的是绝对的 ``G:\\npu-experiment\\...``。本脚本按项目规约
  只存相对路径（``backend.model_path`` 之外不再出现盘符），因此 ``model`` 字段
  的值会与新路径一致、与旧文件不同——这是有意的。
- 生成文本长度随采样变化（``do_sample=true``），所以 ``tokens`` / ``tok_s``
  逐次不同，需要看多轮均值而不是单次。

无法从残留 JSON 反推、因而属于「本次口径自定」的字段
----------------------------------------------------
1. ``tokens`` 的计数口径：本次＝对生成文本重新 encode 取长度（genai 的
   ``generate(str)`` 返回 ``str``，拿不到 token id）。与解码器真实吐出的 token
   数在特殊 token 边界上可能差 ±1。
2. ``load_s`` 是否含首次编译：本次＝只含 ``LLMPipeline`` 构造耗时，首次生成
   作为 warmup 单独丢弃、不计入。
3. ``device_names`` 旧值是手抄的，本次由 OpenVINO ``FULL_DEVICE_NAME`` 实测填充
   （新增 ``host`` 同一份映射，作为可追溯字段）。
4. 旧文件 ``model`` 是绝对路径 ``G:\\npu-experiment\\...``，本次只写相对项目根。

用法
----
    python src/bench/bench_llm.py --mode latency --outdir output/results/rerun
    python src/bench/bench_llm.py --mode short   --outdir output/results/rerun
    python src/bench/bench_llm.py --mode long    --outdir output/results/rerun
    python src/bench/bench_llm.py --mode all     --outdir output/results/rerun

设备是**独占**资源（尤其 NPU 只有一块），所以本脚本从头到尾严格串行：同一时刻
只有一个 LLMPipeline 活着，跑完一个设备立刻释放。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import statistics
import sys
import time
import traceback

# 与 src/probe、src/fetch 保持同样的 import 姿态：先上跳一级拿到 src/，再 import paths。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

DEFAULT_DEVICES = ["CPU", "GPU.0", "GPU.1", "NPU"]

# ---------------------------------------------------------------------------
# 各模式的参数。prompt / max_new 均照抄旧结果 JSON 里的取值，保证可比。
# ---------------------------------------------------------------------------
MODES = {
    # output/results/results-1.5b.json
    "short": {
        "filename": "results-1.5b.json",
        "prompt": "用一句话解释什么是复利。",
        "max_new": 128,
        "repeat": 2,          # 旧文件里恰好 2 条 run（首轮已丢弃，见下）
        "discard_warmup": True,
    },
    # output/results/results-1.5b-long.json
    "long": {
        "filename": "results-1.5b-long.json",
        "prompt": "请用中文写一段关于复利原理的详细说明，至少三百字。",
        "max_new": 512,
        "repeat": 2,
        "discard_warmup": True,
    },
    # output/results/decision-latency.json
    "latency": {
        "filename": "decision-latency.json",
        "prompt": "用一句话解释什么是复利。",
        "max_new": 24,
        "repeat": 5,          # 旧文件的 repeat=5 全额计入统计，丢首轮就凑不够 5 个样本
        "discard_warmup": False,
    },
}

# 旧文件里长文本只留了 head，本脚本留全量但按此长度另存一份 head，两边都能对。
TEXT_HEAD_LEN = 128

BACKEND = {
    "kind": None,             # "openvino_genai" | "optimum_intel"
    "reason": "",
    "model_path": "",
}


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _load_backend():
    """挑一条推理路径，并把选择理由写进模块级 BACKEND。"""
    try:
        import openvino_genai  # noqa: F401
    except Exception as exc:  # pragma: no cover - 取决于本机安装
        BACKEND["kind"] = "optimum_intel"
        BACKEND["reason"] = (
            "openvino_genai 不可用（"
            + str(exc)[:160]
            + "），回退 optimum.intel.OVModelForCausalLM；该路径经 torch 前端，"
            "数字与 genai 路径不可直接混用。"
        )
    else:
        BACKEND["kind"] = "openvino_genai"
        BACKEND["reason"] = "openvino_genai 可用，走 LLMPipeline（首选路径，与模型目录的 OV 导出格式匹配）。"
    return BACKEND["kind"]


def _available_device_names():
    """读 OpenVINO 的完整设备名，用于 host 映射（等价旧文件的 device_names）。"""
    import openvino as ov

    core = ov.Core()
    names = {}
    for dev in core.available_devices:
        try:
            names[dev] = str(core.get_property(dev, "FULL_DEVICE_NAME"))
        except Exception as exc:  # pragma: no cover
            names[dev] = f"<不可读: {str(exc)[:60]}>"
    return names


def _openvino_version():
    import openvino as ov

    return str(ov.__version__)


def _rel(path) -> str:
    """把绝对路径写成相对项目根的 posix 形式，避免在结果里留下盘符。"""
    p = pathlib.Path(path).resolve()
    try:
        return p.relative_to(paths.PROJECT_ROOT).as_posix()
    except ValueError:
        return str(p)


class Runner:
    """一条推理路径的薄封装：只暴露 load / generate / 释放。"""

    def __init__(self, model_dir: str, device: str, kind: str):
        self.device = device
        self.kind = kind
        self.model_dir = model_dir
        self.pipe = None
        self.tok = None
        self._load_s = None

    def load(self):
        """加载模型并计时；返回 load_s。"""
        t0 = time.perf_counter()
        if self.kind == "openvino_genai":
            import openvino_genai as og

            self.pipe = og.LLMPipeline(self.model_dir, self.device)
            self.tok = self.pipe.get_tokenizer()
        else:  # pragma: no cover - 只有 genai 缺失时才会走到
            import openvino as ov
            from optimum.intel import OVModelForCausalLM
            from transformers import AutoTokenizer

            self.pipe = OVModelForCausalLM.from_pretrained(
                self.model_dir, device=self.device, ov_config={"PERFORMANCE_HINT": "LATENCY"}
            )
            self.tok = AutoTokenizer.from_pretrained(self.model_dir)
            _ = ov  # 保持 import 语义清晰
        self._load_s = time.perf_counter() - t0
        return self._load_s

    def generate(self, prompt: str, max_new: int):
        """跑一次生成，返回 (耗时秒, 生成 token 数, 文本)。

        计时代码只包住 generate() 一行——与旧实验「端到端 wall clock」口径一致，
        不含 tokenize / detokenize 之外的任何后处理。token 数的取法见 count_tokens。
        """
        t0 = time.perf_counter()
        if self.kind == "openvino_genai":
            text = self.pipe.generate(prompt, max_new_tokens=max_new)
        else:  # pragma: no cover
            import torch

            inputs = self.tok(prompt, return_tensors="pt")
            with torch.no_grad():
                out = self.pipe.generate(**inputs, max_new_tokens=max_new, do_sample=True)
            text = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        gen_s = time.perf_counter() - t0
        return gen_s, self.count_tokens(text), text

    def count_tokens(self, text: str) -> int:
        """数「新生成」的 token 数。

        口径说明：genai 的 ``LLMPipeline.generate(str)`` 直接返回 detokenize 后的
        ``str``（实测，不是带 .texts 的 DecodedResults），拿不到 token id，所以这里
        对生成文本重新 encode 取长度。这是旧实验最可能的口径，但它与「解码器实际
        吐出的 token 数」在极少数特殊 token 边界上会有 ±1 的出入，属于无法从残留
        JSON 反推的字段。
        """
        try:
            if self.kind == "openvino_genai":
                enc = self.tok.encode(text)
                arr = enc.input_ids.data
                return int(len(arr[0]) if hasattr(arr, "__len__") else len(arr))
            return int(len(self.tok.encode(text)))
        except Exception:
            return -1

    def release(self):
        """显式释放，避免下一个设备抢显存/内存（NPU 是独占设备，尤其不能叠着）。"""
        self.pipe = None
        self.tok = None


def _percentile(values, q):
    """最近秩法，不引入 numpy 依赖（样本量最大 5，插值没有意义）。"""
    if not values:
        return None
    ordered = sorted(values)
    import math

    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


def _err(exc: BaseException, limit: int = 200) -> str:
    return f"{type(exc).__name__}: {exc}"[:limit]


def _base_meta(mode_cfg, devices, host_names) -> dict:
    """所有模式共用的头部字段 + 可追溯三件套（generated_by/at/host）。"""
    return {
        "openvino": _openvino_version(),
        "backend": {
            "kind": BACKEND["kind"],
            "reason": BACKEND["reason"],
            "model_path": BACKEND["model_path"],
        },
        "model": _rel(paths.QWEN_1_5B),
        "prompt": mode_cfg["prompt"],
        "max_new": mode_cfg["max_new"],
        "host": host_names,
        "device_names": host_names,  # 旧字段名，保留兼容
        "generated_by": _rel(pathlib.Path(__file__)),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host_python": platform.python_version(),
        "note": "首轮生成作 warmup 丢弃，不计入 runs（latency 模式除外，repeat 全额计入）。",
    }


def _timeout_hit(t0, budget_s):
    return budget_s > 0 and (time.perf_counter() - t0) > budget_s


def run_short_or_long(devices, mode_cfg, budget_s):
    """short / long 模式：每设备 load 一次，跑 1 轮 warmup（丢弃）+ repeat 轮计入。"""
    result = _base_meta(mode_cfg, devices, _available_device_names())
    result["available"] = devices
    devs = {}
    for dev in devices:
        t_dev = time.perf_counter()
        entry = {}
        runner = None
        try:
            runner = Runner(str(paths.QWEN_1_5B), dev, BACKEND["kind"])
            entry["load_s"] = round(runner.load(), 2)
            runs = []
            if mode_cfg["discard_warmup"]:
                # warmup：第一次 generate 会做设备侧的图编译/内存分配，这一轮的数据
                # 明显偏高（实测 CPU 首次 2.49s vs 稳态 1.01s），所以显式丢弃，
                # 其 token 数记进 warmup_tokens 供排查。
                _, warm_tokens, _ = runner.generate(mode_cfg["prompt"], mode_cfg["max_new"])
                entry["warmup_tokens"] = warm_tokens
            for i in range(mode_cfg["repeat"]):
                if _timeout_hit(t_dev, budget_s):
                    entry["stopped_early"] = f"设备累计耗时超过 {budget_s:.0f}s 预算，剩余轮次跳过"
                    break
                gen_s, ntok, text = runner.generate(mode_cfg["prompt"], mode_cfg["max_new"])
                runs.append(
                    {"gen_s": round(gen_s, 3), "tokens": ntok, "tok_s": round(ntok / gen_s, 2) if gen_s > 0 else None}
                )
                entry["text"] = text
            if not runs:
                entry["error"] = entry.get("stopped_early", "没有任何有效轮次")
            else:
                # 旧 schema：顶层 tok_s/gen_s/tokens 取「最后一轮」的值；这里沿用，
                # 同时补 runs 平均值，避免单轮采样波动被误读成设备差异。
                last = runs[-1]
                entry["runs"] = runs
                entry["tok_s"] = last["tok_s"]
                entry["gen_s"] = last["gen_s"]
                entry["tokens"] = last["tokens"]
                entry["mean_tok_s"] = round(
                    statistics.fmean([r["tok_s"] for r in runs if r["tok_s"] is not None]), 2
                )
                if mode_cfg["max_new"] >= 128:
                    entry["text_head"] = entry.get("text", "")[:TEXT_HEAD_LEN]
        except Exception as exc:
            # 设备失败只记错误，绝不让整个脚本崩——GPU.1(独显)/NPU 都可能缺驱动。
            entry = {"error": _err(exc), "traceback_tail": traceback.format_exc(limit=3)[-400:]}
        finally:
            if runner is not None:
                runner.release()
        devs[dev] = entry
    result["devices"] = devs
    return result


def run_latency(devices, mode_cfg, budget_s):
    """latency 模式：短问答（max_new=24），repeat 轮全额计入，统计 p50/mean/min/max。"""
    result = _base_meta(mode_cfg, devices, _available_device_names())
    result["repeat"] = mode_cfg["repeat"]
    devs = {}
    for dev in devices:
        t_dev = time.perf_counter()
        try:
            runner = Runner(str(paths.QWEN_1_5B), dev, BACKEND["kind"])
            load_s = round(runner.load(), 2)
            ms = []
            toks = []
            texts = []
            for _ in range(mode_cfg["repeat"]):
                if _timeout_hit(t_dev, budget_s):
                    break
                gen_s, ntok, text = runner.generate(mode_cfg["prompt"], mode_cfg["max_new"])
                ms.append(gen_s * 1000.0)
                toks.append(ntok)
                texts.append(text)
            runner.release()
            if not ms:
                devs[dev] = {"load_s": load_s, "error": "没有任何有效轮次（超时）"}
                continue
            mean_ms = statistics.fmean(ms)
            mean_tok = statistics.fmean(toks)
            devs[dev] = {
                "load_s": load_s,
                # 命名照抄旧文件（short_p50_ms 其实是「本轮所有 max_new=24 生成」的 p50，
                # 不特指短 prompt；保留旧名以免下游解析断掉）。
                "short_p50_ms": round(_percentile(ms, 0.5), 1),
                "mean_ms": round(mean_ms, 1),
                "min_ms": round(min(ms), 1),
                "max_ms": round(max(ms), 1),
                "mean_tokens": round(mean_tok, 1),
                # 旧口径：tok_s = 平均 token 数 / 平均毫秒数（端到端，含固定开销）。
                "tok_s": round(mean_tok / (mean_ms / 1000.0), 1),
                "runs_ms": [round(v, 1) for v in ms],
                "runs_tokens": toks,
                "text": texts[-1],
            }
        except Exception as exc:
            devs[dev] = {"error": _err(exc), "traceback_tail": traceback.format_exc(limit=3)[-400:]}
    result["devices"] = devs
    return result


def write_json(payload: dict, outdir: pathlib.Path, filename: str, force: bool) -> pathlib.Path:
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / filename
    # 护栏：默认输出目录里躺的是 2026-09 的原始证据，任何情况下都不许被覆盖。
    if target.exists() and not force:
        raise SystemExit(
            f"拒绝覆盖已存在的文件：{target}\n"
            f"  该目录里的旧结果是原始证据。要写到别处请用 --outdir；"
            f"确实想覆盖请显式加 --force。"
        )
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return target


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="LLM 设备基准（CPU / GPU.0 / GPU.1 / NPU），重跑 2026-09 的三份结果。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：python src/bench/bench_llm.py --mode latency --outdir output/results/rerun",
    )
    ap.add_argument("--mode", default="all", choices=["short", "long", "latency", "all"])
    ap.add_argument(
        "--outdir",
        default=str(paths.RESULTS),
        help="结果目录（默认 paths.RESULTS；相对路径按项目根解析）",
    )
    ap.add_argument(
        "--devices",
        default="",
        help=f"逗号分隔的设备列表，默认 {'/'.join(DEFAULT_DEVICES)}，按 OpenVINO available_devices 过滤",
    )
    ap.add_argument("--repeat", type=int, default=0, help="每个设备计入统计的轮数（0=用模式内置值）")
    ap.add_argument(
        "--per-device-timeout",
        type=float,
        default=300.0,
        help="单设备累计耗时预算（秒），超出后跳过其剩余轮次；0=不限（默认 300）",
    )
    ap.add_argument("--force", action="store_true", help="允许覆盖已存在的结果文件（默认拒绝）")
    args = ap.parse_args(argv)

    kind = _load_backend()
    BACKEND["model_path"] = _rel(paths.QWEN_1_5B)
    paths.ensure_output_dirs()

    if not paths.QWEN_1_5B.exists():
        print(f"[FATAL] 模型目录不存在：{paths.QWEN_1_5B}", file=sys.stderr)
        return 2
    if "7B" in str(paths.QWEN_1_5B):
        print("[FATAL] 拒绝在 7B 目录上跑（加载极慢且原实验中是悬空的）", file=sys.stderr)
        return 2

    host_names = _available_device_names()
    wanted = [s.strip() for s in args.devices.split(",") if s.strip()] or list(DEFAULT_DEVICES)
    devices = [d for d in wanted if d in host_names]
    skipped = [d for d in wanted if d not in host_names]
    if not devices:
        print(f"[FATAL] 没有可用设备。请求={wanted} 本机={list(host_names)}", file=sys.stderr)
        return 2

    outdir = pathlib.Path(args.outdir)
    if not outdir.is_absolute():
        outdir = (paths.PROJECT_ROOT / outdir).resolve()

    modes = ["latency", "short", "long"] if args.mode == "all" else [args.mode]

    print(f"backend      : {kind}  ({BACKEND['reason'][:80]})")
    print(f"model        : {BACKEND['model_path']}")
    print(f"openvino     : {_openvino_version()}")
    print(f"devices      : {devices}" + (f"  (跳过未列出的: {skipped})" if skipped else ""))
    print(f"outdir       : {outdir}")
    print(f"modes        : {modes}")
    sys.stdout.flush()

    for mode in modes:
        cfg = dict(MODES[mode])
        if args.repeat > 0:
            cfg["repeat"] = args.repeat
        print(f"\n=== mode={mode}  max_new={cfg['max_new']}  repeat={cfg['repeat']} ===")
        sys.stdout.flush()
        t0 = time.perf_counter()
        if mode == "latency":
            payload = run_latency(devices, cfg, args.per_device_timeout)
        else:
            payload = run_short_or_long(devices, cfg, args.per_device_timeout)
        payload["mode"] = mode
        payload["repeat"] = cfg["repeat"]
        target = write_json(payload, outdir, cfg["filename"], args.force)
        print(f"--- mode={mode} 用时 {time.perf_counter() - t0:.1f}s -> {target}")
        for dev, entry in payload["devices"].items():
            if "error" in entry:
                print(f"    {dev:<6} ERROR: {entry['error'][:120]}")
            elif mode == "latency":
                print(
                    f"    {dev:<6} load={entry['load_s']}s p50={entry['short_p50_ms']}ms "
                    f"mean={entry['mean_ms']}ms tok_s={entry['tok_s']} mean_tokens={entry['mean_tokens']}"
                )
            else:
                print(
                    f"    {dev:<6} load={entry['load_s']}s tok_s={entry.get('tok_s')} "
                    f"tokens={entry.get('tokens')} gen_s={entry.get('gen_s')} mean_tok_s={entry.get('mean_tok_s')}"
                )
        sys.stdout.flush()

    print("\n完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
