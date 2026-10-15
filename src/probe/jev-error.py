"""定位「整个 jevlite.onnx 无法在 NPU 上编译」的根因。

做法：reshape 成 [1,768]/[1,18] 静态形状 → 统计算子类型直方图 → 直接尝试 NPU 编译
并打印完整异常。产出证据：图里有 22 个 IsNaN，NPU 报
"Unsupported operation node_IsNaN_247 with type IsNaN and version opset10"
（原始输出见 output/results/jev-error.txt）。

这是 inspect-onnx.py 的超集（后者只打印 IO 形状），故后者已归档。

整理前 ONNX 路径写死为 `g:\\npu-experiment\\jevlite.onnx`。
"""
import numpy as np, traceback, sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # -> src/
import paths

import openvino as ov

ONNX = str(paths.require(paths.JEV_ONNX))
core = ov.Core()
m = core.read_model(ONNX)
m.reshape({"input_ids": [1, 768], "attention_mask": [1, 768], "label_pos": [1, 18]})

print("=== 模型算子类型统计（顶层）===", flush=True)
from collections import Counter
c = Counter()
for op in m.get_ordered_ops():
    c[op.get_type_name()] += 1
for k, v in c.most_common(40):
    print(f"  {k:<28} {v}", flush=True)

print(f"\n总算子数: {len(m.get_ordered_ops())}", flush=True)

print("\n=== 尝试 NPU 编译，输出完整错误 ===", flush=True)
try:
    c2 = core.compile_model(m, "NPU")
    print("  编译成功", flush=True)
except Exception as e:
    print(str(e), flush=True)
