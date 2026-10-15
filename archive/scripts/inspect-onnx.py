import openvino as ov, numpy as np
M = r"g:\npu-experiment\jevlite.onnx"
core = ov.Core()
m = core.read_model(M)
print("=== ONNX 原始输入/输出 ===")
for i in m.inputs:  print(f"  IN  {i.get_any_name():<16} {i.partial_shape}  {i.get_element_type()}")
for o in m.outputs: print(f"  OUT {o.get_any_name():<16} {o.partial_shape}  {o.get_element_type()}")
print(f"  参数总量: {sum(int(np.prod(p.shape)) for p in m.get_parameters()):,}")
