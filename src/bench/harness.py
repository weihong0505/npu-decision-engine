"""基准测试公共设施：设备枚举、编译计时、推理计时、配对 A/B、数值校验、带溯源落盘。

整理前那批结果 JSON 是手工从 REPL 抄出来的 —— 没有生产者脚本，也看不出是哪台机器、
什么配置跑的。本模块存在的意义就是让每个结果都能被 `src/bench/*.py` 重新生成，
并且自带 `generated_by` / `generated_at` / `openvino` / `device_names` 溯源字段。
"""
from __future__ import annotations

import datetime
import json
import pathlib
import platform
import statistics
import sys
import time

import numpy as np
import openvino as ov

import paths

# 结果 JSON 里统一记录的设备顺序
DEVICE_ORDER = ["NPU", "CPU", "GPU.0", "GPU.1"]


def make_core() -> ov.Core:
    return ov.Core()


def available_devices(core: ov.Core) -> list[str]:
    return list(core.available_devices)


def device_names(core: ov.Core) -> dict[str, str]:
    """设备 → 人类可读名（如 NPU → Intel(R) AI Boost）。取不到就记 '?'。"""
    names = {}
    for d in core.available_devices:
        try:
            names[d] = core.get_property(d, "FULL_DEVICE_NAME")
        except Exception:
            names[d] = "?"
    return names


def order_devices(devices) -> list[str]:
    """按 NPU/CPU/GPU.0/GPU.1 的固定顺序排列，未知设备排在最后。"""
    devs = list(devices)
    head = [d for d in DEVICE_ORDER if d in devs]
    tail = sorted(d for d in devs if d not in DEVICE_ORDER)
    return head + tail


def compile_model(core: ov.Core, model, device: str, config: dict | None = None):
    """编译并计时。返回 (compiled_model, compile_seconds)。"""
    t0 = time.perf_counter()
    cm = core.compile_model(model, device, config or {})
    return cm, time.perf_counter() - t0


def feeds_for(model, seq_len: int | None = None, batch: int = 1, seed: int = 0) -> dict:
    """按模型的输入签名造 dummy 输入。

    - 静态维度沿用模型自身；动态维度：第 0 维用 batch，其余用 seq_len。
    - 名字含 mask → 全 1；含 pos → 全 0（label_pos 语义）；其余 → 词表内随机 id。

    注意：这里刻意不加载 tokenizer —— 基准只关心延迟与图的可编译性，
    用真 token 还是随机 id 对算子耗时没有影响。
    """
    rng = np.random.default_rng(seed)
    feeds = {}
    for inp in model.inputs:
        name = inp.get_any_name()
        dims = []
        for i, d in enumerate(inp.get_partial_shape()):
            if d.is_static:
                dims.append(d.get_length())
            else:
                dims.append(batch if i == 0 else (seq_len or 128))
        low = name.lower()
        if "mask" in low:
            arr = np.ones(dims, dtype=np.int64)
        elif "pos" in low:
            arr = np.zeros(dims, dtype=np.int64)
        else:
            arr = rng.integers(1000, 50000, size=dims, dtype=np.int64)
        feeds[name] = arr
    return feeds


def time_infer(compiled, feeds: dict, repeat: int = 20, warmup: int = 3) -> dict:
    """推理计时：先 warmup 次预热（丢弃），再 repeat 次取分位数。"""
    req = compiled.create_infer_request()
    for _ in range(warmup):
        req.infer(feeds)
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        req.infer(feeds)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return {
        "compile_s": None,
        "p50_ms": round(statistics.median(ts), 2),
        "mean_ms": round(statistics.fmean(ts), 2),
        "min_ms": round(min(ts), 2),
        "max_ms": round(max(ts), 2),
        "runs": repeat,
    }


def run_on(core: ov.Core, model, device: str, feeds: dict, repeat: int = 20,
           warmup: int = 3, config: dict | None = None) -> dict:
    """在单个设备上编译 + 计时。失败不抛异常，而是返回 {"error": ...} 保留现场。"""
    try:
        cm, cs = compile_model(core, model, device, config)
        st = time_infer(cm, feeds, repeat, warmup)
        st["compile_s"] = round(cs, 2)
        return st
    except Exception as e:
        return {"error": str(e)[:500]}


def infer_outputs(compiled, feeds: dict) -> list[np.ndarray]:
    res = compiled.create_infer_request().infer(feeds)
    return [np.asarray(res[o]) for o in compiled.outputs]


def max_abs_err(ref_outputs, test_outputs) -> float:
    """两个模型输出之间的最大绝对误差（数值保真度校验）。"""
    return float(max(np.max(np.abs(a - b)) for a, b in zip(ref_outputs, test_outputs)))


def time_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def run_isolated_device(model_path, device: str, seq: int | None = None, labels: int | None = None,
                        config: dict | None = None, repeat: int = 20, warmup: int = 3,
                        timeout: int = 900) -> dict:
    """在**子进程**里编译并计时一个设备，用来隔离会让 NPU 段错误的图。

    背景：JEV 系列 ONNX 在本机 NPU 上会让 OpenVINO 原生层报 0xC0000005 访问违例，
    这是进程级崩溃，Python 的 try/except 抓不到，会把整个基准进程静默带走
    （实测表现为"零输出 + 退出码 -1073741819"）。放进子进程后，父进程能把它
    记成一条 ``{"error": "子进程退出码 ..."}`` 并继续跑别的设备。

    成功时返回与 ``run_on`` 同构的统计 dict，另加 ``_output_npy``（主输出落盘路径）
    与 ``isolated: True``。
    """
    import json as _json
    import subprocess
    import uuid

    tmpdir = paths.RESULTS / "rerun" / "_probe"
    tmpdir.mkdir(parents=True, exist_ok=True)
    tag = f"{device.replace('.', '_')}_{uuid.uuid4().hex[:8]}"
    out_json = tmpdir / f"{tag}.json"
    out_npy = tmpdir / f"{tag}.npy"

    cmd = [sys.executable, str(pathlib.Path(__file__).resolve().parent / "_device_probe.py"),
           "--model", str(model_path), "--device", device,
           "--repeat", str(repeat), "--warmup", str(warmup),
           "--out", str(out_json), "--save-output", str(out_npy)]
    if seq:
        cmd += ["--seq", str(seq)]
    if labels:
        cmd += ["--labels", str(labels)]
    if config:
        cmd += ["--config", _json.dumps(config)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"子进程超时 {timeout}s", "isolated": True}

    if proc.returncode != 0:
        hint = ""
        if proc.returncode in (-1073741819, 3221225477):     # 0xC0000005 的两种写法
            hint = "（0xC0000005 访问违例：OpenVINO NPU 原生层崩溃，进程级、try/except 抓不到）"
        return {"error": f"子进程退出码 {proc.returncode}{hint}",
                "stderr_tail": (proc.stderr or "")[-400:], "isolated": True}

    data = _json.loads(out_json.read_text(encoding="utf-8"))
    if out_npy.exists():
        data["_output_npy"] = str(out_npy)
    data["isolated"] = True
    return data


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def write_result(name: str, payload, outdir=None, script: str | None = None) -> str:
    """落盘结果 JSON，并注入溯源字段。返回写入的路径。

    payload 允许是 dict，也允许是 list（历史上有裸数组的结果文件，如 scaling.json）。
    裸数组会被包装成 ``{"runs": [...], ...溯源字段}`` —— 溯源信息必须有地方挂，
    而且自描述的包装比裸数组更好读。
    """
    outdir = pathlib.Path(outdir) if outdir else (paths.RESULTS / "rerun")
    outdir.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, dict):
        payload = dict(payload)
    else:
        payload = {"runs": list(payload)}
    payload.setdefault("openvino", ov.get_version())
    payload["generated_by"] = script or "src/bench/"
    payload["generated_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    payload["host"] = f"{platform.node()} / {platform.system()} {platform.release()} / py{platform.python_version()}"
    path = outdir / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, cls=_NumpyEncoder), encoding="utf-8")
    print(f"\nwrote {path}")
    return str(path)
