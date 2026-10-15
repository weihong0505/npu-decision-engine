"""把 PowerShell 重定向产生的日志整理成干净的 UTF-8 文本，同时**逐字保留原始副本**。

背景：2026-09 那批实验日志是 `python xxx.py > out.txt` 在 Windows PowerShell 5.1 下的产物，
有三个毛病：

1. **UTF-16LE + BOM** —— 大多数文本工具（含 IDE、grep、read 工具）会判定为二进制而拒绝读取；
2. 开头裹了一段 PowerShell 的 stderr 包装（``At line:`` / ``CategoryInfo`` / ``FullyQualifiedErrorId``），
   因为它把 stderr 也并进了捕获文件；
3. 含 ANSI 颜色转义（transformers 的彩色日志）与**本机路径**（``C:\\Users\\<user>\\...``）。

本脚本的处理是"转换而非改写"：只做编码转换、剥包装、去 ANSI、遮蔽本机路径，
**不动任何实验数值**；并把 PowerShell 包装里唯一有价值的信息 —— 原始命令行 ——
提取成文件头注释保留下来（那正是证明 02→03→04→06→07 调用顺序的证据）。

用法::

    python src/tools/normalize_logs.py --dry-run     # 先看会改什么
    python src/tools/normalize_logs.py               # 转换并写回 output/results/*.txt
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # -> src/
import paths  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
CMD_LINE = re.compile(r"^\s*\+\s*(&?\s*python(?:\.exe)?\b.*)$")
PS_FRAME = re.compile(r"^\s*(\+\s*~+|At line:|.*CategoryInfo\s*:|.*FullyQualifiedErrorId\s*:|python(\.exe)?\s*:)", re.I)
# 注意：这些都是 re.sub 的**替换模板**，反斜杠必须写成 \\ ，否则 \U 会被当成转义而报错
SANITIZE = [
    (re.compile(r"C:\\Users\\[^\\\s]+", re.I), r"C:\\Users\\<user>"),
    (re.compile(r"[A-Z]:\\[^\\\s]*[\\/]site-packages", re.I), r"<SITE_PACKAGES>"),
]


def decode(raw: bytes) -> tuple[str, str]:
    """按 BOM 判断真实编码。返回 (文本, 编码名)。"""
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16"), "utf-16"
    if raw[:3] == b"\xef\xbb\xbf":
        return raw.decode("utf-8-sig"), "utf-8-sig"
    return raw.decode("utf-8", errors="replace"), "utf-8"


def normalize(text: str) -> tuple[str, str | None]:
    """返回 (清洗后文本, 提取到的原始命令行)。"""
    text = ANSI.sub("", text)
    lines = text.splitlines()
    cmd = None
    for ln in lines:
        m = CMD_LINE.match(ln)
        if m:
            cmd = " ".join(m.group(1).split())
            break

    # 剥掉 PowerShell 包装：从最后一行 FullyQualifiedErrorId 之后开始算正文
    cut = 0
    for i, ln in enumerate(lines):
        if "FullyQualifiedErrorId" in ln:
            cut = i + 1
    body = lines[cut:] if cut else lines
    while body and not body[0].strip():
        body.pop(0)
    # 剩下的裸 stderr 帧（At line / CategoryInfo 续行 / + ~~~）也去掉
    body = [ln for ln in body if not PS_FRAME.match(ln) or ln.strip().startswith(("noul", "choice", "score"))]

    out = "\n".join(body).rstrip() + "\n"
    for pat, rep in SANITIZE:
        out = pat.sub(rep, out)
    if cmd:
        out = f"# 原始命令（由 PowerShell 包装中提取）: {cmd}\n" + out
    return out, cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None, help="默认 output/results")
    ap.add_argument("--archive", default="archive/results-20260922",
                    help="原始副本落点（UTF-8 逐字副本，仅转码不改内容）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_dir = pathlib.Path(args.dir) if args.dir else paths.RESULTS
    arc_dir = paths.PROJECT_ROOT / args.archive
    arc_dir.mkdir(parents=True, exist_ok=True)

    for p in sorted(src_dir.glob("*.txt")):
        raw = p.read_bytes()
        text, enc = decode(raw)
        # 原始副本：只转码，不改内容
        if not args.dry_run:
            (arc_dir / p.name).write_text(text, encoding="utf-8")
        clean, cmd = normalize(text)
        changed = clean != text
        print(f"{p.name:<20} 编码={enc:<10} {len(text):>7} → {len(clean):>7} 字符  "
              f"{'改写' if changed else '不变'}  命令={'有' if cmd else '无'}")
        if not args.dry_run and changed:
            p.write_text(clean, encoding="utf-8")
    if not args.dry_run:
        print(f"\n原始副本（UTF-8，内容逐字保留）→ {arc_dir}")


if __name__ == "__main__":
    main()
