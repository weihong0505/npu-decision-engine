"""拉取 Qwen2.5-1.5B 的 OpenVINO INT4 权重到 models/Qwen2.5-1.5B-Instruct-int4-ov/。

整理前 local_dir 写死为 `g:\\npu-experiment\\...`。
"""
from huggingface_hub import snapshot_download
import pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # -> src/
import paths

p = snapshot_download(
    repo_id="OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov",
    local_dir=str(paths.QWEN_1_5B),
    allow_patterns=["*.xml","*.bin","*.json","*.model","*.txt","*.yaml"],
)
print("下载完成 ->", p)
