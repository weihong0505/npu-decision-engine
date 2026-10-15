"""比对 ONNX 侧车里的权重与 pytorch 检查点里的权重，判断两者是否**同源**。

为什么需要它：`output/results/export-out.txt` 记录导出当时自检过 parity
（``parity vs pytorch: max abs diff 6.32e-06 OK``），说明 02:46 那份 ONNX 与
**当时的**检查点是同一份权重。但 `jevlite.pt` 在当天 12:38 被 03_calibrate.py 重新落盘，
于是"NPU 结论到底跑在哪份权重上"就成了必须回答的问题。
`bench_jev.py --mode verify` 从输出侧比对，本工具从**权重本体**侧比对，两条路互相印证。

用法::

    python src/tools/onnx_weight_fingerprint.py --onnx output/onnx/jev_encoder.onnx
    python src/tools/onnx_weight_fingerprint.py --onnx output/onnx/jevlite.onnx --list-names
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import onnx
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # -> src/
import paths  # noqa: E402

DTYPE = {1: np.float32, 2: np.uint8, 3: np.int8, 6: np.int32, 7: np.int64, 10: np.float16, 11: np.float64}


def read_initializer(init, onnx_dir: pathlib.Path) -> np.ndarray | None:
    """取一个 initializer 的数值：优先内嵌 raw_data，否则按 external_data 的 offset/length 读侧车。"""
    dt = DTYPE.get(init.data_type)
    if dt is None:
        return None
    if init.raw_data:
        return np.frombuffer(init.raw_data, dtype=dt)
    ext = {kv.key: kv.value for kv in init.external_data}
    loc = ext.get("location")
    if not loc:
        return None
    offset = int(ext.get("offset", 0))
    length = int(ext.get("length", 0))
    side = onnx_dir / loc
    if not side.exists() or not length:
        return None
    with open(side, "rb") as f:
        f.seek(offset)
        raw = f.read(length)
    return np.frombuffer(raw, dtype=dt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--ckpt", default=None, help="默认 output/onnx/jevlite.pt")
    ap.add_argument("--list-names", action="store_true", help="只打印命名，不比对")
    ap.add_argument("--sample", type=int, default=6)
    args = ap.parse_args()

    onnx_path = pathlib.Path(args.onnx)
    if not onnx_path.is_absolute():
        onnx_path = paths.PROJECT_ROOT / onnx_path

    ck = torch.load(args.ckpt or paths.JEV_CKPT, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]

    # 只读图结构，不加载 570 MB 侧车
    m = onnx.load(str(onnx_path), load_external_data=False)
    inits = list(m.graph.initializer)
    print(f"ONNX: {onnx_path.name}  initializer={len(inits)}  state_dict={len(sd)}")

    if args.list_names:
        print("\n--- ONNX 前 12 个 ---")
        for i in inits[:12]:
            print(f"  {i.name}  {list(i.dims)}")
        print("--- state_dict 前 12 个 ---")
        for k, v in list(sd.items())[:12]:
            print(f"  {k}  {list(v.shape)}")
        return

    # 名字能对上的就逐元素比；对不上就只报告命名差异
    # 命名映射：ExportWrapper 把模型挂在 self.m 上，所以 ONNX 侧名字带 "m." 前缀
    def lookup(name: str):
        candidates = [name]
        if name.startswith("m."):
            candidates.append(name[2:])
        for c in candidates:
            if c in sd:
                return sd[c]
        return None

    matched, mismatched, missing = 0, [], 0
    for init in inits:
        ref = lookup(init.name)
        if ref is None:
            missing += 1
            continue
        arr = read_initializer(init, onnx_path.parent)
        if arr is None:
            continue
        ref_np = ref.detach().numpy().ravel()
        if ref_np.shape != arr.shape:
            mismatched.append((init.name, f"形状不同 onnx={arr.shape} ckpt={ref_np.shape}", None))
            continue
        d = float(np.max(np.abs(arr.astype(np.float64) - ref_np.astype(np.float64))))
        matched += 1
        if d > 1e-6:
            mismatched.append((init.name, "数值不同", d))

    print(f"\n可配对张量: {matched}   不一致: {len(mismatched)}   名字对不上: {missing}")
    if missing and matched == 0:
        print("提示：ONNX 导出器可能给参数加了前缀（如 '/encoder/...'），用 --list-names 看命名方案。")
    for name, why, d in mismatched[: args.sample]:
        print(f"  [NG] {name}: {why}" + (f"  max|delta|={d:.3e}" if d is not None else ""))
    if matched and not mismatched:
        print("  [OK] 全部配对张量逐元素一致 → ONNX 与检查点同源")
    elif matched:
        print("  [NG] 存在不一致 → ONNX 与检查点**不同源**，NPU 结论的权重归属需要重新标注")


if __name__ == "__main__":
    main()
