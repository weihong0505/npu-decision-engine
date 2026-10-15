"""统计数据集的最大标签数与标签数分布 —— 决定 ONNX 里 label_pos 的宽度。

整理前 `sys.path.insert` 写死盘符。这里用官方 typed_schema.iter_labels，
取代了 archive/scripts/inspect-data.py 里那套靠启发式猜键的自写 count_labels。
"""
import sys, json, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # -> src/
import paths
paths.add_engine_to_path()

from td_data import load_split
from typed_schema import iter_labels

maxlbl, dist = 0, {}
for split in ["train", "test"]:
    for cfg in ["all"]:
        try: rows = load_split(split, cfg)
        except Exception as e: print("skip", split, cfg, e); continue
        for r in rows:
            n = len(list(iter_labels(r["questions"])))
            maxlbl = max(maxlbl, n)
            dist[n] = dist.get(n, 0) + 1
print("最大标签数:", maxlbl)
print("标签数分布:", dict(sorted(dist.items())))
