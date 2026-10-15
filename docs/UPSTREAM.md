# UPSTREAM · 上游来源与许可

## engine/ 的来源

`engine/` 不是本项目原创代码，是下面这个公开仓库的**浅克隆**，整理时重命名而来：

| 项 | 值 |
|---|---|
| remote | `https://github.com/intikhab49/open-jev-typed-decision-engine.git` |
| HEAD | `refs/heads/main` |
| commit | `5977431` — *Address the review in #1: scope the Jev comparison, bind loopback, add CI* |
| 克隆方式 | shallow clone |
| 许可 | Apache-2.0（`engine/LICENSE` 原样保留） |
| 原目录名 | `open-jev-typed-decision-engine/` |
| 整理动作 | 重命名为 `engine/`；其 `.git` 已迁至 `backup/20261015-113854/upstream-git-dotGit/` |

**为什么要去掉它的 `.git`**：本项目要成为一个**自包含**的仓库 ——
若 `engine/` 保留嵌套 `.git`，外层仓库只能把它当 gitlink 忽略，
引擎代码就进不了本项目的历史。上游 commit 与地址记录在此文件，
需要时可用下面的命令恢复一个可 `git pull` 的上游副本：

```powershell
git clone https://github.com/intikhab49/open-jev-typed-decision-engine.git upstream-tmp
git -C upstream-tmp checkout 5977431
```

若只想恢复被归档的 `.git`（保留 shallow 状态）：

```powershell
Move-Item backup\20261015-113854\upstream-git-dotGit engine\.git
```

## 上游内容里的第三方引用

- **注意力实现/编码器**：`answerdotai/ModernBERT-base`（HF 上的模型，训练时才需要）
- **数据集**：`LocalLLaMA/typed-decisions`（`data/typed-decisions/`）
- **对照对象**：TypeSafe AI 的 Jev（商业模型，仅作引用，无代码/权重依赖）
- `engine/legacy/*.py` 里有 `import anthropic`（上游早期用 Claude 做标注的遗留路径）。
  **本项目未使用该路径**；若要发布 `engine/`，需单独审计其 key 取法。

## 本项目原创的部分

| 目录 | 性质 |
|---|---|
| `src/bench/` | **全新**（本次整理补写）：三类基准 + 公共设施 |
| `src/paths.py` | **全新**：统一路径层 |
| `src/tools/` | **全新**：日志规范化、ONNX 权重指纹 |
| `src/probe/`、`src/fetch/` | 2026-09 实验脚本，本次整理改路径、去 BOM、补文档字符串 |
| `docs/`、`README.md`、`REVIEW.md`、`run.ps1`、`tests/` | **全新** |
| `output/results/` | 2026-09-22 实测产物（原样保留） |

## 许可提示

本仓库整体沿用上游的 **Apache-2.0**（见 `engine/LICENSE`）。
`engine/` 内的文件版权归上游作者；本项目新增部分（`src/`、`docs/`、`tests/` 等）
在同样许可下提供。模型权重与数据集各自的许可以其原始发布方为准。
