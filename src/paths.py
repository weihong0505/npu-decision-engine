"""统一路径与运行环境解析。

本项目的所有脚本都必须通过本模块取路径，**禁止再写死 `g:\\` / `e:\\` 之类的盘符**——
2026-09 的实验脚本里出现过 8 处硬编码绝对路径，其中 2 处指向并不存在的 `e:\\npu-experiment`，
直接导致脚本 `ModuleNotFoundError`。这是那次整理要根治的问题。

用法（脚本位于 src/fetch/ 或 src/probe/，深度一致）::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # -> src/
    import paths

    ckpt = paths.require(paths.JEV_CKPT)

覆盖项目根：设置环境变量 ``NPUDE_ROOT``（CI 或迁移目录时用）。
覆盖下载代理：设置 ``HF_PROXY``（留空则直连，不再硬编码本机 7897 端口）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "PROJECT_ROOT", "ENGINE", "SRC", "MODELS", "DATA", "OUTPUT", "RESULTS",
    "ONNX", "CACHE", "DOCS",
    "BERT", "QWEN_1_5B", "QWEN_7B", "DATASET", "NPU_CACHE",
    "JEV_CKPT", "JEV_ONNX", "JEV_ONNX_DATA", "JEV_STATIC_ONNX",
    "JEV_ENCODER_ONNX", "JEV_ENCODER_ONNX_DATA", "JEV_ENCODER_INT8_ONNX",
    "ENC_DYNAMO_SDPA", "ENC_DYNAMO_EAGER", "BERT_ONNX_Q4",
    "HF_PROXY", "add_engine_to_path", "require", "ensure_output_dirs",
]

# --------------------------------------------------------------------------
# 根目录：环境变量优先，否则取本文件（src/paths.py）的上两级
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("NPUDE_ROOT") or Path(__file__).resolve().parents[1]).resolve()

# Windows 上 Python 的 stdout 默认跟随系统代码页（本机是 GBK/cp936），
# 打印中文会乱码、打印 ✓/✗ 这类字符会直接 UnicodeEncodeError 崩掉。
# 统一改成 UTF-8（errors="replace" 保证绝不因编码而中断基准跑分）。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:      # 被重定向成非文本流时忽略
        pass

ENGINE = PROJECT_ROOT / "engine"
SRC = PROJECT_ROOT / "src"
MODELS = PROJECT_ROOT / "models"
DATA = PROJECT_ROOT / "data"
OUTPUT = PROJECT_ROOT / "output"
RESULTS = OUTPUT / "results"
ONNX = OUTPUT / "onnx"
CACHE = PROJECT_ROOT / "cache"
DOCS = PROJECT_ROOT / "docs"

# 模型
BERT = MODELS / "ModernBERT-base"
QWEN_1_5B = MODELS / "Qwen2.5-1.5B-Instruct-int4-ov"
QWEN_7B = MODELS / "Qwen2.5-7B-Instruct-int4-ov"

# 数据与缓存
DATASET = DATA / "typed-decisions"
NPU_CACHE = CACHE / "npu-cache"

# JEV-lite 产物（权威检查点 = jevlite.pt，训练于 engine/02_train.py）
JEV_CKPT = ONNX / "jevlite.pt"
JEV_ONNX = ONNX / "jevlite.onnx"
JEV_ONNX_DATA = ONNX / "jevlite.onnx.data"
JEV_STATIC_ONNX = ONNX / "jevlite_static.onnx"
JEV_ENCODER_ONNX = ONNX / "jev_encoder.onnx"
JEV_ENCODER_ONNX_DATA = ONNX / "jev_encoder.onnx.data"
JEV_ENCODER_INT8_ONNX = ONNX / "jev_encoder_int8.onnx"
ENC_DYNAMO_SDPA = ONNX / "enc_dynamo_sdpa.onnx"
ENC_DYNAMO_EAGER = ONNX / "enc_dynamo_eager.onnx"

# 参考模型：HF 官方的 OpenVINO INT4 导出，用来做「自导 vs 官方」的隔离对照
BERT_ONNX_Q4 = BERT / "onnx" / "model_q4.onnx"

# 下载代理：默认不启用。历史上这里硬编码过 http://127.0.0.1:<port>
HF_PROXY = os.environ.get("HF_PROXY", "").strip()


def add_engine_to_path() -> Path:
    """把 engine/ 加入 import 路径，返回该目录（供 `from td_data import ...` 使用）。"""
    if str(ENGINE) not in sys.path:
        sys.path.insert(0, str(ENGINE))
    return ENGINE


def require(path: str | os.PathLike) -> Path:
    """断言产物存在，否则给出「怎么生成」的可执行提示。"""
    p = Path(path)
    if not p.exists():
        hints = {
            JEV_CKPT.name: "跑 engine/02_train.py 训练，或从备份恢复 output/onnx/jevlite.pt",
            JEV_ONNX.name: "跑 engine/06_export_onnx.py --quantize",
            JEV_ENCODER_ONNX.name: "跑 engine/06_export_onnx.py（仅导出编码器）",
        }
        hint = hints.get(p.name, "检查路径或先跑对应的 fetch/export 脚本")
        raise FileNotFoundError(f"缺少产物：{p}\n  生成方式：{hint}")
    return p


def ensure_output_dirs() -> None:
    """保证输出目录存在（新机器/新 checkout 上直接能跑）。"""
    for d in (RESULTS, ONNX, CACHE):
        d.mkdir(parents=True, exist_ok=True)
