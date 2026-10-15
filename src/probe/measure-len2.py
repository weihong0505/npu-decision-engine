"""统计 state+questions 的 token 长度分布，回答「max_len 该设多少」。

这是 measure-len.py 的修正版（全量样本 + 多一档 cap + 打印超限例数），
取代关系见 archive/scripts/README.md。

整理前 `sys.path.insert` 写成 `r"e:\\npu-experiment".replace("pU","pu")`：
`.replace` 是空操作（原始串里没有大写 `pU`），而盘符 `e:` 在本机**并不存在**，
所以这个脚本整理前是跑不通的（ModuleNotFoundError: td_data）。现已修正为项目根相对解析。
"""
import sys, json, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # -> src/
import paths
paths.add_engine_to_path()

import numpy as np
from td_data import load_split
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(str(paths.BERT))
rows = load_split("train", "all")
print("train 行数:", len(rows))
lens = []
for r in rows:
    txt = json.dumps(r.get("state"), ensure_ascii=False) + " " + json.dumps(r.get("questions"), ensure_ascii=False)
    lens.append(len(tok(txt)["input_ids"]))
print("token 长度: min %d  p50 %d  p90 %d  p99 %d  max %d" % (
    min(lens), int(np.percentile(lens,50)), int(np.percentile(lens,90)),
    int(np.percentile(lens,99)), max(lens)))
for cap in [128,192,256,384,512,768,1024]:
    cov = sum(1 for L in lens if L <= cap)/len(lens)
    print(f"  max_len={cap:<5} 覆盖 {cov*100:.1f}%   (超出 {sum(1 for L in lens if L>cap)} 例)")
