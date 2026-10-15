"""内部工具：在**独立进程**里做一次「读模型 → 编译 → 计时 → 取输出」。

为什么需要它：JEV 系列 ONNX 在本机 NPU 上会让 OpenVINO 原生层直接段错误
（Windows 退出码 ``0xC0000005`` 访问违例），这**不是 Python 异常，try/except 抓不到**，
会连累整个基准进程静默死掉（实测：什么都不打印，退出码 -1073741819）。

所以把"会让 NPU 崩的那次调用"隔离到子进程：崩了只丢子进程，父进程能记录
"进程级崩溃"并继续跑其它设备。

被 `harness.run_isolated()` 调用，不建议手工使用；真要手工调试可以用：

    python src/bench/_device_probe.py --model output/onnx/jev_encoder.onnx \
        --device NPU --seq 768 --labels 18 --repeat 20 --out probe.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # -> src/
import paths  # noqa: E402
import harness as H  # noqa: E402  同目录


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="ONNX 路径（相对项目根或绝对）")
    ap.add_argument("--device", required=True)
    ap.add_argument("--seq", type=int, default=None, help="给了就把动态维 reshape 成 [1,seq]")
    ap.add_argument("--labels", type=int, default=None, help="label_pos 宽度（如 18）")
    ap.add_argument("--config", default=None, help="OpenVINO 配置的 JSON，如 '{\"NPU_TURBO\":\"YES\"}'")
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", required=True, help="结果 JSON 落点")
    ap.add_argument("--save-output", default=None, help="把主输出存成 .npy，供父进程做 parity")
    args = ap.parse_args()

    model_path = pathlib.Path(args.model)
    if not model_path.is_absolute():
        model_path = paths.PROJECT_ROOT / model_path

    core = H.make_core()
    model = core.read_model(str(model_path))
    if args.seq:
        shapes = {}
        for inp in model.inputs:
            name = inp.get_any_name()
            if "label" in name.lower() or "pos" in name.lower():
                shapes[name] = [1, args.labels or 1]
            else:
                tail = [d.get_length() for d in inp.get_partial_shape()[2:]]
                shapes[name] = [1, args.seq] + tail
        model.reshape(shapes)

    config = json.loads(args.config) if args.config else {}
    feeds = H.feeds_for(model, args.seq)

    cm, compile_s = H.compile_model(core, model, args.device, config)
    stats = H.time_infer(cm, feeds, args.repeat, args.warmup)
    stats["compile_s"] = round(compile_s, 2)
    stats["device"] = args.device
    stats["config"] = config

    if args.save_output:
        out = np.asarray(cm.create_infer_request().infer(feeds)[cm.outputs[0]])
        np.save(args.save_output, out)
        stats["saved_output"] = str(args.save_output)

    pathlib.Path(args.out).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
