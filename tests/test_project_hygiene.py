"""工程卫生：那些"改一次就悄悄坏掉"的约束，用测试守住而不是只写在文档里。

目前守两件事：
1. `run.ps1` 必须是 **UTF-8 with BOM** —— Windows PowerShell 5.1 读无 BOM 的 UTF-8
   脚本会按系统代码页(GBK)解码，中文注释的字节错位会破坏**解析**，
   症状是 `无法将"Do-Test"项识别为 cmdlet` 甚至把脚本源码当输出打印。
   这个坑 2026-10-15 整理时踩过一次，修复只需加 BOM，但普通编辑器保存又会丢。
2. 原始结果文件不得被重跑结果覆盖 —— `output/results/*.json|txt` 是历史证据，
   重跑一律写 `output/results/rerun/`。
"""
import pathlib

import paths

ROOT = paths.PROJECT_ROOT


def test_run_ps1_has_utf8_bom():
    raw = (ROOT / "run.ps1").read_bytes()
    assert raw[:3] == b"\xef\xbb\xbf", (
        "run.ps1 缺少 UTF-8 BOM：Windows PowerShell 5.1 会按 GBK 解码并破坏解析。"
        "修复：用 UTF8Encoding($true) 重新保存。"
    )


def test_ps1_files_are_bom_encoded():
    """后续若新增 .ps1，同样要求带 BOM。"""
    ps1s = [p for p in ROOT.rglob("*.ps1") if "backup" not in p.parts]
    assert ps1s, "应当至少有一个 .ps1（run.ps1）"
    for p in ps1s:
        assert p.read_bytes()[:3] == b"\xef\xbb\xbf", f"{p.name} 缺少 UTF-8 BOM"


def test_original_results_not_overwritten_by_rerun():
    """原始结果目录里必须没有 rerun 的产物，反之亦然（两者是不同批次）。"""
    results = paths.RESULTS
    rerun = results / "rerun"
    assert results.is_dir(), "原始结果目录应存在"
    for f in results.glob("*.json"):
        assert "generated_by" not in f.read_text(encoding="utf-8"), (
            f"{f.name} 看起来是重跑产物（含 generated_by），它应该在 output/results/rerun/ 下"
        )
    if rerun.is_dir():
        for f in rerun.glob("*.json"):
            assert "generated_by" in f.read_text(encoding="utf-8"), (
                f"{f.name} 缺少溯源字段 generated_by，重跑结果必须自带 provenance"
            )


def test_gitignore_has_no_trailing_comments():
    """gitignore 不支持行尾注释；`models/   # 说明` 会让整条规则失效。

    2026-10-15 整理时靠这条才没把 16 GB 大件提交进仓库。
    """
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for i, raw in enumerate(lines, 1):
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        assert "#" not in line, f".gitignore 第 {i} 行含行尾注释，规则会失效：{raw!r}"


def test_ignored_dirs_stay_untracked():
    """大件目录必须真的被忽略（用 git check-ignore 的等价规则名做静态检查）。"""
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("models/*", "data/*", "cache/", "output/onnx/*", "backup/"):
        assert pattern in text, f".gitignore 缺少 {pattern}"
    for readme in ("!models/README.md", "!data/README.md", "!output/onnx/README.md"):
        assert readme in text, f".gitignore 缺少例外 {readme}（被排除目录里的 README 需要放行）"
