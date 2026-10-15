import sys, json
sys.path.insert(0, r"g:\npu-experiment\open-jev-typed-decision-engine")
import numpy as np
from td_data import load_split, encode
from transformers import AutoTokenizer
import statistics

tok = AutoTokenizer.from_pretrained(r"g:\npu-experiment\ModernBERT-base")
rows = load_split("train", "all")
print("train 行数:", len(rows))
lens = []
for r in rows[:400]:
    s = encode(r) if callable(encode) else None
    # 直接按 state+questions 拼出文本再分词，模拟真实编码
    txt = json.dumps(r.get("state"), ensure_ascii=False) + " " + json.dumps(r.get("questions"), ensure_ascii=False)
    lens.append(len(tok(txt)["input_ids"]))
print("token 长度: min %d  p50 %d  p90 %d  p99 %d  max %d" % (
    min(lens), int(np.percentile(lens,50)), int(np.percentile(lens,90)),
    int(np.percentile(lens,99)), max(lens)))
for cap in [128,256,384,512,768,1024]:
    cov = sum(1 for L in lens if L <= cap)/len(lens)
    print(f"  max_len={cap:<5} 覆盖 {cov*100:.1f}%")
