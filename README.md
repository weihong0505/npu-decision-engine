# npu-decision-engine

把 **JEV-lite**（ModernBERT-150M 的 typed decision engine，一次前向回答任意类型化问题）
和 **Qwen2.5-1.5B** 放到 Intel **NPU（Intel AI Boost）** 上，量清楚它到底值不值。

一句话结论：**NPU 适合静态形状的编码器前向（比 CPU 快约 3×），不适合自回归解码（比 CPU 慢约 6×）。**

> ⚠️ **当前状态（2026-10-15 复测）**：结论线 1、2 已用新基准脚本干净复现；
> **结论线 3 的 NPU 半边失效** —— NPU 编译 JEV 那两个 fp32 图会让进程直接段错误
> （`0xC0000005`），同机同刻 INT4 的 ModernBERT 图却正常。所以下表里 189 ms 一行是
> **2026-09-22 的历史测量，现在跑不出来**。详见
> [docs/02 第七节](docs/02-实验结论汇总.md#七2026-10-15-重跑对照outputresultsrerun)。

| 任务 | 设备 | 延迟 | 结论 |
|---|---|---|---|
| JEV-lite 编码器（静态 L=768） | **NPU** | **189 ms**（历史值，现不可复现） | 拆图后曾比纯 CPU 快 **3.0×** |
| JEV-lite 整图（含 head） | **NPU** | **编译失败**（现为段错误） | 图里有 22 个 `IsNaN`，NPU 不支持 |
| JEV-lite 整图 | CPU | 570.9 ms（2026-10 复测） | 唯一能跑通整图的设备 |
| ModernBERT 编码器（静态 L=128） | **NPU** | **40.6 ms** | 比 CPU（61.8 ms）快 1.5× |
| ModernBERT 编码器（动态形状） | **NPU** | **编译失败** | 静态形状是 NPU 可编译的**前提** |
| Qwen2.5-1.5B 解码（max_new=24） | **NPU** | 3305 ms / **3.6 tok/s** | 比 CPU（19.7 tok/s）慢 **5.5×** |

## 硬件与环境

本机（2026-09 实验 + 2026-10 整理）实测：

| 设备 | 型号 | OpenVINO 设备名 |
|---|---|---|
| CPU | Intel Core Ultra 9 275HX | `CPU` |
| iGPU | Intel(R) Graphics | `GPU.0` |
| dGPU | NVIDIA GeForce RTX 5060 Laptop | `GPU.1` |
| **NPU** | **Intel(R) AI Boost** | `NPU` |

OpenVINO 2026.4.0 · Python 3.14.6 · torch 2.11.0+cu128

## 三条结论线

### 1. NPU 跑 LLM 解码：不划算

`Qwen2.5-1.5B-Instruct-int4-ov`，短 prompt：

| 设备 | p50 (max_new=24) | 吞吐 |
|---|---|---|
| CPU | 553 ms | **19.7 tok/s** |
| iGPU (GPU.0) | 1190 ms | 18.6 tok/s |
| dGPU (GPU.1) | — | 7.8 tok/s |
| **NPU** | **3305 ms** | **3.6 tok/s** |

长 prompt / 128 new token 下 NPU 也只有 6.6 tok/s，CPU 是 28.1 tok/s。
自回归解码是逐 token 的串行小算子流，NPU 的并行优势完全用不上。

### 2. NPU 跑编码器：需要先"驯服"形状

ModernBERT-base（HF 官方 INT4 导出，`model_q4.onnx`）：

| 形状 | NPU | CPU | iGPU | NPU/CPU |
|---|---|---|---|---|
| 动态 `[?,?]` | **编译失败** | 37.2 ms | 34.1 ms | — |
| 静态 L=128 | **38.4 ms** | 60.7 ms | 47.3 ms | 0.63 |
| 静态 L=768 | **253.3 ms** | 383.0 ms | 282.7 ms | 0.66 |

静态化之后 NPU 在 128→768 全程领先，比值稳定在 0.55–0.66。

### 3. JEV-lite 整图：拆开才行

```
整图丢 NPU   → 编译失败（22 个 IsNaN）        → 只能 CPU 568 ms
补掉 IsNaN   → 能编译，但 1625 ms + 精度崩(0.81) → 无意义
拆成两半     → NPU 编码器 189 ms + CPU head 0.2 ms = 189 ms  ← ✅ 成立
```

拆图依据（`engine/model.py:54-59`）：head 只是 `Linear(768,1)` 作用在
`gather(label_pos)` 上，只有 `score.weight[1,768]` + `score.bias[1]` 两个张量，
所以 head 留在 CPU 只花 0.2 ms，而编码器（224 MB 权重）交给 NPU。

## 快速开始

```powershell
# 0) 环境自检（设备、模型、算例是否就位）
.\run.ps1 check

# 1) 跑测试（不需要 NPU，约 2 秒）
python -m pytest

# 2) 复现 NPU 编码器结论（约 1 分钟）
python src/bench/bench_encoder.py --mode static

# 3) 复现 JEV 混合方案（约 1 分钟）
python src/bench/bench_jev.py --mode hybrid

# 4) 复现 LLM 设备对比（较慢，NPU 上每轮数十秒）
python src/bench/bench_llm.py --mode latency
```

结果写到 `output/results/rerun/`，每个 JSON 自带 `generated_by` / `generated_at` / `openvino` / `host`
溯源字段。**`output/results/` 里的原作是 2026-09-22 的手抄证据，原样保留不覆盖**——
两批数据的关系见 [docs/02-实验结论汇总.md](docs/02-实验结论汇总.md)。

引擎侧（训练 / 标定 / 评估 / 集成）在 [engine/](engine/) 里，用法见其
[README](engine/README.md)：

```powershell
python engine/01_ceiling.py            # 先看要打败的基线
python engine/smoke_test.py --all      # 全量数据校验，无需 GPU
python engine/04_eval.py               # 官方测试集评估
```

## 目录

```
npu-decision-engine/
├─ README.md  REVIEW.md  requirements.txt  .gitignore  pytest.ini  run.ps1
├─ src/
│  ├─ paths.py            统一路径解析（禁止写死盘符）
│  ├─ fetch/              数据与模型下载
│  ├─ bench/              ★ 基准脚本：把结论变成可复现
│  │   ├─ harness.py      设备枚举 / 计时 / 配对 A-B / 带溯源落盘
│  │   ├─ bench_llm.py    Qwen2.5 设备对比（结论线 1）
│  │   ├─ bench_encoder.py ModernBERT 形状与配置扫描（结论线 2）
│  │   └─ bench_jev.py    JEV 整图 vs 拆图（结论线 3）
│  ├─ probe/              形状探针、检查点/ONNX 检视、设备属性
│  └─ tools/              日志规范化等工程工具
├─ engine/                JEV-lite 训练与评估流水线（01→07，上游代码，见 docs/UPSTREAM.md）
├─ tests/                 路径回归、基准设施、head 数值等价、工程卫生（23 项，无需 NPU）
├─ docs/                  背景 / 结论汇总 / 约束与踩坑 / 复现手册 / 上游来源
├─ output/
│  ├─ results/            19 个原始结果 + rerun/ 重跑结果
│  └─ onnx/               ONNX 图、侧车权重、jevlite.pt
├─ models/  data/  cache/ 大件（已 gitignore，用 src/fetch/ 重新获取）
├─ archive/               被取代的脚本与结果（保留留痕）
└─ backup/                整理前的完整快照
```

## 已知限制

- **NPU 只接受静态形状**。这是本轮所有"NPU 失败/成功"的总开关。据 Intel 材料，
  动态形状要到 Lunar Lake 一代才支持（见 `docs/03-约束与踩坑.md`）。
- **NPU 数值精度约 1e-2**：同样的图在 CPU 上误差 3e-6，在 NPU 上 0.024，iGPU 0.078。
  对 logits 做 argmax 无影响，但别指望它做需要高精度的数值计算。
- **INT8 量化在 NPU 上是负优化**（3374 ms，比 fp32 慢 2×）。
- **dGPU（GPU.1）经 OpenVINO 极慢**（比 NPU 慢 97%），不要把它当备选。
- **本仓库不含大件**：模型权重、数据集、ONNX 图、NPU 缓存都被 `.gitignore` 排除，
  新克隆后需按 [docs/04-复现手册.md](docs/04-复现手册.md) 重新获取。
- **`engine/` 是上游代码**，不是本项目原创；来源与 commit 见 [docs/UPSTREAM.md](docs/UPSTREAM.md)。

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/01-背景与目标.md](docs/01-背景与目标.md) | 这个项目要回答什么、边界在哪 |
| [docs/02-实验结论汇总.md](docs/02-实验结论汇总.md) | 19 个结果文件的结论、取代关系、冲突裁决 |
| [docs/03-约束与踩坑.md](docs/03-约束与踩坑.md) | 静态形状、IsNaN、INT8、侧车改名等硬约束与坑 |
| [docs/04-复现手册.md](docs/04-复现手册.md) | 从零重建环境的每一步 |
| [docs/UPSTREAM.md](docs/UPSTREAM.md) | 上游仓库来源、commit、许可 |
| [REVIEW.md](REVIEW.md) | 整理这次工作的复盘 |
