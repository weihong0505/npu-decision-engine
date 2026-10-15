"""拉取 Qwen2.5-7B 的 OpenVINO INT4 权重到 models/Qwen2.5-7B-Instruct-int4-ov/。

整理前 local_dir 写死为 `g:\\npu-experiment\\...`。

注意：这个 7B 模型在 2026-09 那轮实验里**没有任何下游脚本或结论引用它**（悬空资产，
见 docs/02-实验结论汇总.md）。保留本下载脚本以保证可重新获取，但默认不参与基准。
"""
from huggingface_hub import snapshot_download
import pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # -> src/
import paths

p = snapshot_download(
    repo_id="OpenVINO/Qwen2.5-7B-Instruct-int4-ov",
    local_dir=str(paths.QWEN_7B),
    allow_patterns=["*.xml","*.bin","*.json","*.model","*.txt","*.yaml"],
)
print("downloaded ->", p)
