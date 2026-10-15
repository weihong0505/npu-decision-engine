import sys, json
sys.path.insert(0, r"e:\npu-experiment".replace("pU","pu") + r"\open-jev-typed-decision-engine")
from td_data import load_split
rows = load_split("train", "all")
r = rows[0]
print("=== 一条样本的键 ===")
print(list(r.keys()))
print("\n=== questions 原始结构 ===")
q = r["questions"]
print(type(q))
print(json.dumps(q, ensure_ascii=False)[:800])
print("\n=== gold ===")
print(json.dumps(r["gold"], ensure_ascii=False)[:300])
# 统计标签数
def count_labels(qq):
    if isinstance(qq, dict):
        # 可能是 {qname: {...}}
        tot = 0
        for v in qq.values():
            if isinstance(v, dict):
                if "labels" in v: tot += len(v["labels"])
                elif "options" in v: tot += len(v["options"])
                else: tot += 1
            else: tot += 1
        return tot
    return 0
import numpy as np
counts = [count_labels(x["questions"]) for x in rows]
print("\n标签数: min", min(counts), "max", max(counts), "p50", int(np.percentile(counts,50)))
