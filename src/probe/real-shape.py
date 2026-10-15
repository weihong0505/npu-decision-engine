"""用真实检查点加载 JevLite，打印 qid/lid/max_len 与 collate 后的张量形状。

这是判读 ONNX 静态形状的权威脚本，取代 shape-probe.py（后者硬编码 max_len=768
且不加载 lid 映射）。整理前路径写死为 `g:\\npu-experiment\\...`。
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # -> src/
import paths
paths.add_engine_to_path()

from td_data import load_split, TypedDecisions, collate
from model import JevLite, require_ckpt

ck = require_ckpt(str(paths.require(paths.JEV_CKPT)))
model = JevLite(ck["encoder"])
model.load_state_dict(ck["state_dict"])
model.eval()
print("qid（问题→id）:", model.qid)
print("lid（标签→id）:", model.lid)
print("max_len:", ck["max_len"])

row = load_split("test", limit=1)[0]
ds = TypedDecisions([row], model.tok, ck["max_len"], model.qid, model.lid)
b = collate([ds[0]], model.tok.pad_token_id)
print("\n=== 单样本 batch 形状 ===")
for k, v in b.items():
    print(f"  {k:<14} {tuple(v.shape)}  {v.dtype}")

# 多样本 batch（看 label_pos 是否随样本数变化）
rows = load_split("test", limit=3)
ds3 = TypedDecisions(rows, model.tok, ck["max_len"], model.qid, model.lid)
b3 = collate([ds3[i] for i in range(3)], model.tok.pad_token_id)
print("\n=== 3 样本 batch 形状 ===")
for k, v in b3.items():
    print(f"  {k:<14} {tuple(v.shape)}  {v.dtype}")
