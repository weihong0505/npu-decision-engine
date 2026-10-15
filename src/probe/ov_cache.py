"""读取 NPU 的 CACHE_DIR 与各设备 GOPS（探测用，无文件依赖）。

注：原文件是 PowerShell 直接生成的 UTF-8 with BOM，整理时去掉 BOM（BOM 会污染
某些解析器读到的首行）。逻辑未改。
"""
import openvino as ov
core = ov.Core()
try:
    print("NPU CACHE_DIR =", core.get_property("NPU", "CACHE_DIR") or "(未设置)")
except Exception as e:
    print("读取失败:", str(e)[:100])
try:
    print("NPU DEVICE_GOPS =", core.get_property("NPU", "DEVICE_GOPS"))
except Exception as e:
    print("GOPS 不可读:", str(e)[:80])
for d in ["CPU","GPU.0","GPU.1","NPU"]:
    try:
        print(f"{d} DEVICE_GOPS =", core.get_property(d, "DEVICE_GOPS"))
    except Exception as e:
        print(f"{d} GOPS 不可读: {str(e)[:60]}")
