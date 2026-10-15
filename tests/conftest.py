"""pytest 公共配置：把 src/ 与 src/bench/ 加进 import 路径。

这样测试里可以直接 `import paths` / `import harness` / `from bench_jev import ...`，
不必在每个测试文件里重复 sys.path 操作（脚本自身靠 `parents[1]` 解析，见 src/paths.py）。
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "src" / "bench"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
