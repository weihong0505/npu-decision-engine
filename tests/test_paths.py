"""路径层测试。

重点是一条**回归测试**：整理前 `src/` 里有 8 处写死的盘符路径，其中 2 处指向并不存在的
`e:\\npu-experiment`，导致脚本直接 `ModuleNotFoundError`。这条测试防止它再长回来。
"""
import ast
import re

import paths


def test_project_root_and_key_dirs():
    assert (paths.PROJECT_ROOT / "src" / "paths.py").exists()
    for d in (paths.ENGINE, paths.MODELS, paths.DATA, paths.OUTPUT, paths.RESULTS, paths.ONNX):
        assert d.is_dir(), f"缺少目录: {d}"


def test_models_layout():
    """模型目录即便被 .gitignore 排除，也应在本地就位（否则基准跑不了）。"""
    assert (paths.BERT / "config.json").exists()
    assert paths.QWEN_1_5B.is_dir()
    assert paths.BERT_ONNX_Q4.exists(), "model_q4.onnx 是 encoder 基准的被测对象"


def test_require_gives_actionable_hint(tmp_path):
    missing = tmp_path / "jevlite.pt"
    try:
        paths.require(missing)
    except FileNotFoundError as e:
        assert "生成方式" in str(e), "缺产物时应告诉用户怎么生成"
    else:
        raise AssertionError("require() 未对缺失文件报错")


def test_no_hardcoded_drive_paths_in_src():
    """src/ 下不得再出现写死的项目绝对路径（注释与文档字符串里记录历史不算）。"""
    pattern = re.compile(r"^[a-zA-Z]:\\")
    offenders = []
    for f in (paths.PROJECT_ROOT / "src").rglob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                v = node.value
                if pattern.match(v) and "npu-" in v.lower():
                    rel = f.relative_to(paths.PROJECT_ROOT)
                    offenders.append(f"{rel}:{node.lineno} -> {v[:70]}")
    assert not offenders, "发现写死的盘符路径：\n" + "\n".join(offenders)


def test_hf_proxy_defaults_to_empty():
    """代理默认不启用 —— 整理前它被硬编码成 127.0.0.1:<port>。"""
    assert paths.HF_PROXY == "" or paths.HF_PROXY.startswith("http")
