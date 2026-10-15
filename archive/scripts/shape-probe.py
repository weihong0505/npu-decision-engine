import sys, json
sys.path.insert(0, r"g:\npu-experiment\open-jev-typed-decision-engine")
import torch
from td_data import load_split, TypedDecisions, collate
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(r"g:\npu-experiment\ModernBERT-base")
rows = load_split("train", "all")
ds = TypedDecisions(rows, tok, 768)
print("问题数 qid:", len(ds.qid), " 标签数 lid:", len(ds.lid))
print("qid:", list(ds.qid.items())[:8])
print("lid:", list(ds.lid.items())[:10])
bs = collate([ds[i] for i in range(4)])
print("\n=== collate 输出张量 ===")
for k, v in bs.items():
    if torch.is_tensor(v):
        print(f"  {k:<14} {tuple(v.shape)}  {v.dtype}")
    else:
        print(f"  {k:<14} {type(v)}")
# 全数据集标签位宽统计
widths = set()
for i in range(len(ds)):
    widths.add(tuple(ds[i]["label_pos"].shape))
print("\nlabel_pos 形状集合:", sorted(widths))
maxw = max(w[0] for w in widths)
print("label_pos 最大宽度:", maxw)
