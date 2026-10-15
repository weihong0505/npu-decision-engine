"""环境自检：Python / OpenVINO / 设备 / 大件是否就位。

放在 Python 而不是 PowerShell 里，是为了避免跨 shell 的引号与 here-string 差异
（本项目实测：Windows PowerShell 5.1 对 .ps1 里的 here-string 解析会失败），
同时让 `run.ps1`、bash、CI 都能调用同一份逻辑。

用法::

    python src/tools/env_report.py
    python src/tools/env_report.py --json     # 机器可读
"""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # -> src/
import paths  # noqa: E402

# (显示名, 路径, 是否可缺省, 缺失时的获取方式)
ARTIFACTS = [
    ("ModernBERT-base 配置", paths.BERT / "config.json", False, "从 HuggingFace 获取 answerdotai/ModernBERT-base"),
    ("ModernBERT ONNX INT4", paths.BERT_ONNX_Q4, False, "hf_hub_download answerdotai/ModernBERT-base onnx/model_q4.onnx"),
    ("Qwen2.5-1.5B INT4-OV", paths.QWEN_1_5B, False, "python src/fetch/dl_1_5b.py"),
    ("Qwen2.5-7B INT4-OV", paths.QWEN_7B, True, "python src/fetch/dl_7b.py（悬空资产，非必需）"),
    ("typed-decisions 数据集", paths.DATASET, False, "python src/fetch/dl-data.py"),
    ("jevlite.pt 检查点", paths.JEV_CKPT, False, "engine/02_train.py 训练，或从备份恢复"),
    ("编码器 ONNX", paths.JEV_ENCODER_ONNX, False, "engine/06_export_onnx.py"),
    ("整图 ONNX", paths.JEV_ONNX, False, "engine/06_export_onnx.py"),
    ("NPU 编译缓存", paths.NPU_CACHE, True, "首次 NPU 编译时自动生成（可选）"),
    ("原始结果目录", paths.RESULTS, False, "随仓库提供"),
]


def dir_size_mb(p: pathlib.Path) -> float | None:
    if p.is_file():
        return round(p.stat().st_size / 1024 / 1024, 1)
    if p.is_dir():
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        return round(total / 1024 / 1024, 1)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    report: dict = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
        "project_root": str(paths.PROJECT_ROOT),
        "openvino": None,
        "devices": {},
        "artifacts": [],
    }

    try:
        import openvino as ov
        core = ov.Core()
        report["openvino"] = ov.get_version()
        for d in core.available_devices:
            try:
                report["devices"][d] = core.get_property(d, "FULL_DEVICE_NAME")
            except Exception:
                report["devices"][d] = "?"
    except Exception as e:
        report["openvino_error"] = str(e)[:200]

    for name, path, optional, how in ARTIFACTS:
        exists = path.exists()
        report["artifacts"].append({
            "name": name,
            "path": str(path.relative_to(paths.PROJECT_ROOT)) if path.is_relative_to(paths.PROJECT_ROOT) else str(path),
            "present": exists,
            "optional": optional,
            "size_mb": dir_size_mb(path) if exists else None,
            "how_to_get": how,
        })

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print(f"Python      : {report['python']}  ({report['platform']})")
    print(f"项目根      : {report['project_root']}")
    print(f"OpenVINO    : {report['openvino'] or '读取失败: ' + report.get('openvino_error', '')}")
    print("\n可用设备:")
    if report["devices"]:
        for d, n in report["devices"].items():
            flag = "  <-- NPU" if d == "NPU" else ""
            print(f"  {d:<7} {n}{flag}")
    else:
        print("  （无）")
    if "NPU" not in report["devices"]:
        print("  [注意] 未检测到 NPU：装 Intel NPU 驱动后重试；CPU 路径仍可跑测试与部分基准")

    print("\n大件:")
    missing_required = []
    for a in report["artifacts"]:
        mark = "[OK]  " if a["present"] else ("[skip]" if a["optional"] else "[MISS]")
        size = f"{a['size_mb']:>9.1f} MB" if a["size_mb"] is not None else " " * 12
        print(f"  {mark} {a['name']:<24} {size}  {a['path']}")
        if not a["present"] and not a["optional"]:
            missing_required.append(a)

    if missing_required:
        print("\n缺失的必需大件（获取方式）:")
        for a in missing_required:
            print(f"  - {a['name']}: {a['how_to_get']}")
    else:
        print("\n必需大件全部就位。")


if __name__ == "__main__":
    main()
