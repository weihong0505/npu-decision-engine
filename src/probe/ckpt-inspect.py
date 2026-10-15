"""检查 jevlite.pt 的结构：顶层键、qid/lid 映射、logits 头相关权重、校准温度。

整理前 `sys.path.insert` 写死 `g:\\npu-experiment\\open-jev-typed-decision-engine`，
检查点写死 `g:\\npu-experiment\\jevlite.pt`。
"""
import sys, json, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # -> src/
import paths
paths.add_engine_to_path()

import torch

ck = torch.load(paths.require(paths.JEV_CKPT), map_location="cpu", weights_only=False)
print("=== 检查点顶层键 ===")
for k, v in ck.items():
    if isinstance(v, dict):
        print(f"  {k:<14} dict(keys={list(v.keys())[:6]}{'...' if len(v) > 6 else ''} len={len(v)})")
    elif isinstance(v, (list, tuple)):
        print(f"  {k:<14} {type(v).__name__} len={len(v)} 前几项={list(v)[:5]}")
    else:
        print(f"  {k:<14} {type(v).__name__} = {v}")
print("\n=== qid / lid（问题与标签映射）===")
if "qid" in ck: print("  qid:", ck["qid"])
if "lid" in ck: print("  lid:", ck["lid"])
print("\n=== state_dict 中与 logits 头相关的键 ===")
for k in ck.get("state_dict", {}):
    if any(s in k for s in ["logit", "head", "label", "classifier", "proj"]):
        print(f"  {k}  {tuple(ck['state_dict'][k].shape)}")
print("\n=== 温度（校准结果，若有）===")
print("  temperatures:", ck.get("temperatures"))
