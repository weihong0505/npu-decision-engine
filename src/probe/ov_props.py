"""枚举可用设备及其 FULL_DEVICE_NAME / SUPPORTED_PROPERTIES / OPTIMIZATION_CAPABILITIES。

探测用，无文件依赖。原文件是 UTF-8 with BOM，整理时去掉 BOM，逻辑未改。
"""
import openvino as ov
core = ov.Core()
for d in core.available_devices:
    print(f"--- {d} ---")
    for prop in ["FULL_DEVICE_NAME", "SUPPORTED_PROPERTIES", "OPTIMIZATION_CAPABILITIES"]:
        try:
            v = core.get_property(d, prop)
            if prop == "SUPPORTED_PROPERTIES":
                keys = [k for k in v if any(s in k for s in ["OPTIM","PERF","CACHE","TURBO","EXEC","DEVICE","MODEL","COMPIL"])]
                print(f"   {prop}: {len(v)} 项；相关键: {keys[:12]}")
            else:
                print(f"   {prop}: {v}")
        except Exception as e:
            print(f"   {prop}: 不可读 ({str(e)[:60]})")
