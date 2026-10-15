# REVIEW · 把 npu-experiment 整理成 npu-decision-engine

> 整理时间：2026-10-15（已校准系统时间）· 原目录 `G:\npu-experiment` → `G:\npu-decision-engine`
> 执行方式：单个 Lead + 2 个并行子代理（清点分类、LLM 基准脚本），无 Agent Teams
> 本次任务的目标：**整理此项目，整合为一个全套都有的正常项目**（用户原话）

---

## 一、做了什么

把一次性的实验堆（16 GB、50 个顶层散落文件、无 git、无 README）收敛成可复现的项目：

### 结构（16 GB 全部同盘归位，删掉 2.24 GB 废旧导出）

```
顶层散落 50 个文件 → 0 个
├─ 14 个脚本    → src/fetch/（3）+ src/probe/（7）+ archive/scripts/（4，已取代）
├─ 17 个二进制  → output/onnx/（删除 6 个被取代的）
├─ 19 个结果    → output/results/（原样保留；另存转码副本 archive/results-20260922/）
├─ 5 个大目录   → models/（3 个模型）+ data/ + cache/npu-cache/
└─ 上游仓库     → engine/（其 .git 归档到 backup/，升级为自包含仓库）
```

### 新增（本次原创）

| 交付物 | 作用 |
|---|---|
| `src/paths.py` | 统一路径层，消灭 8 处写死盘符 |
| `src/bench/{harness,bench_llm,bench_encoder,bench_jev}.py` | **把 14 个"没有生产者脚本"的结果 JSON 变成可重跑** |
| `src/tools/normalize_logs.py` | UTF-16LE 日志 → 干净 UTF-8，并提取原始命令行 |
| `src/tools/onnx_weight_fingerprint.py` | ONNX 侧车权重与检查点逐张量比对 |
| `tests/`（23 项） | 路径回归、基准设施、numpy head 与 torch 等价、工程卫生（BOM/证据分离/gitignore） |
| `README.md`、`docs/01`–`04`、`docs/UPSTREAM.md`、`REVIEW.md`、`run.ps1` | 文档与统一入口 |
| `.gitignore`、`requirements.txt`、`pytest.ini` | 工程化配套 |

### 数字

| 项 | 整理前 | 整理后 |
|---|---|---|
| 顶层文件 | 50 | 0 |
| 结果文件有生产者脚本的比例 | 5/19（且都是引擎侧日志） | 16/19 |
| 写死的盘符路径 | 8 处（2 处指向不存在的盘） | 0（有回归测试守着） |
| 测试 | 0 | 23 项，约 2 秒，无需 NPU |
| git | 无 | 2 次提交，112+ 文件 |
| G: 可用空间 | 14.9 GB | 17.1 GB |

## 二、关键决策

| 决策 | 理由 |
|---|---|
| **目录改名**（用户拍板） | 原名 `npu-experiment` 是"实验"，与"正常项目"语义不符 |
| **上游 `.git` 归档而非删除** | 移入 `engine/` 后主仓才能版本管理引擎代码；上游 commit 记入 `docs/UPSTREAM.md`，`.git` 留在 `backup/` 可整体恢复 |
| **原始结果一律不覆盖，重跑写 `rerun/`** | 原文件是"当时发生了什么"的证据，重跑是"现在能否复现"的证据，两者对照本身就是结论 |
| **采信 `modernbert-ab.json`，作废 tuned 的 +14.7%** | 配对 4 轮交替顺序 vs 单次扫描；后者是典型单次测量伪影 |
| **删 6 个被取代的导出前先算 SHA256** | 用户批准删除，但要留下"删了什么、怎么重生成"的可查记录 |
| **不重跑训练** | 不在验收标准内；且训练产物与整理目标无关（但发现的问题见下） |
| **放弃原计划的 `config.example.yaml`** | 所有可调项都已经是 CLI 参数或环境变量（`NPUDE_ROOT`/`HF_PROXY`），再加一份没人读的配置文件是负资产。这是对既定计划的**有意偏离**，在此留痕 |

## 三、教训与坑（本次踩到或挖出来的）

### 1. 整理前的项目有 8 处写死盘符，其中 2 处指向**不存在的盘**

`inspect-data.py` 与 `measure-len2.py` 里的
`r"e:\npu-experiment".replace("pU","pu")` —— `.replace` 是空操作（串里没有大写 `pU`），
盘符是 `e:`，而本机 `E:\npu-experiment` 并不存在。讽刺的是"取代版" `measure-len2.py`
反而是跑不通的那个。**低代码质量的典型症状：复制粘贴比验证快。**
→ 已修，并加了 `tests/test_paths.py::test_no_hardcoded_drive_paths_in_src` 用 AST 守门。

### 2. `.gitignore` 不支持行尾注释（差点把 16 GB 提交进 git）

我第一版 `.gitignore` 写成：

```
models/          # ModernBERT-base + Qwen2.5 权重（~5.4 GB）
```

git 把整行（含空格和注释）当作模式，**规则全部失效**。`git add -A` 跑了 2 分钟、
写下 2.68 GB 对象才被我掐断。**教训：gitignore 的注释必须独立成行；改完要
`git check-ignore -v <路径>` 逐条验证，而不是相信肉眼。**

### 3. `目录/` 与 `目录/*` 的区别会让目录里的 README 一起消失

`models/` 这种写法会让 git 不进入该目录，**无法**用 `!models/README.md` 重新包含。
正解是 `models/*` + `!models/README.md`。这是第二个 gitignore 坑。

### 4. 一个 ONNX 改名引发的连锁反应

把 `enc_dynamo_sdpa_默认.onnx` 改成 ASCII 名后直接加载失败：

```
Invalid usage of method for externally stored data in file
(enc_dynamo_sdpa_默认.onnx.data, offset: 443547648, ...)
```

原因：**侧车文件名被写进了 `.onnx` 内部**（`external_data.location`）。
修法是重写图里的 `location`。**推论：`.onnx` 与 `.onnx.data` 必须成对改名。**

### 5. 挖出一个真问题：ONNX 产物与当前检查点**不同源**

子代理清点时发现"NPU 实验（02:46–03:35）跑的是导出时的权重，而 `jevlite.pt` 在 12:38
被 `03_calibrate.py` 重新落盘；标定只写 `log_temp`，所以**应当**等价，但没人验证过"。
我当时也倾向于"应当没问题"。

**结果是被两条独立证据推翻：**

| 核对 | 结果 |
|---|---|
| 权重指纹（46/46、47/47 张量） | 全部不一致 |
| 输出侧（整图 logits / 编码器隐藏状态） | 差 1.65 / 28.5 |

**教训：`应当等价` 不是证据。** 这正是"跨 AI 纠错须对账一手原文"的价值 ——
如果停在"标定只改了 log_temp，所以没问题"，这个坑就会一直埋在项目里。

### 6. 顺带发现：`max_abs_err` 这个指标本身不可靠

进一步查证时发现：**两个权重指纹完全相同的 ONNX 文件（dynamo sdpa / eager），
输出互相之间差 22.97**，各自与 torch 参考差 28.5/28.8，三者范数相同、相关 0.93。
**同权重、不同输出、差 23** ⇒ 差异来自计算语义（ModernBERT 的滑窗局部注意力 /
RoPE 位置处理在不同实现路径下的分歧），不是权重。

想用"缩到 L=128 再比一次"来分离原因的路也走不通：两个编码器 ONNX 都把 768
烘死在常量里，reshape 被形状推断直接拒绝。

**含义：`max_abs_err` 只能读作"同一次运行内、同一张图跨设备的差异"，
绝不能读作模型精度，更不能拿来论证"NPU 精度够用"。**

### 7. 其它踩到的

- PowerShell 5.1 不支持 heredoc（`<<'EOF'`），提交信息得写文件走 `git commit -F`。
- 本机 Python stdout 是 GBK：打印 `✓` 直接 `UnicodeEncodeError` 崩掉基准脚本。
  已在 `src/paths.py` 导入时把 stdout/stderr 重设为 UTF-8。
- 测试首版容差比 float32 精度还紧（numpy 与 torch 不同 BLAS 内核，绝对值 50 的 logit
  上会有 ~2e-5 舍入差）——**测试写太严也会变成噪声**，已改为同时断言"远小于 NPU 误差"。
- `harness.write_result` 最初只接受 dict，而 `scaling.json` 原本是裸数组，
  于是第一次重跑在落盘时崩掉。已改为兼容并自动包装成 `{"runs": [...]}`。

### 8. 重跑发现：JEV 的 NPU 路径在本机已不可复现（环境回归）

用新脚本重跑时，`bench_jev.py` **零输出、退出码 -1073741819**。
逐层隔离后确认是 `0xC0000005` 访问违例 —— **进程级段错误，try/except 抓不到**：

| 测试 | 退出码 |
|---|---|
| `jevlite.onnx` + CPU 编译 + 推理 | 0 |
| `jevlite.onnx` + **NPU** 编译 | **-1073741819** |
| `jev_encoder.onnx` + **NPU** 编译 + 推理 | **-1073741819** |
| **`model_q4.onnx` + NPU 编译 + 推理（同机同刻，L=768）** | **0，252.82 ms** |

最后一行是关键对照：不是 NPU 坏了，而是**这个 fp32 JEV 图的编译回归**。
已排除侧车假设（合并成自包含单文件后同样崩）。

**工程对策**：新增 `harness.run_isolated_device()` + `src/bench/_device_probe.py`，
把会崩的调用放进子进程，崩溃被记成一条 `{"error": "子进程退出码 3221225477…"}`，
基准继续跑完 CPU/iGPU 并正常落盘。**这比"跑通"更重要** ——
一个会静默把基准进程带走的项目，比一个明确报"这条路现在走不通"的项目危险得多。

**代价**：结论线 3 的招牌数字（NPU 189 ms、3.0× 加速）在当前机器上**无法复现**。
已在 README、docs/02、docs/03、docs/04 四处同步标注为"历史测量"。

### 9. 复现结果总账

| 结论 | 复现情况 |
|---|---|
| NPU 编码器静态快于 CPU（L=128→768 全程） | ✅ 比值曲线几乎重合 |
| 动态形状 NPU 编译失败 | ✅（`bench_encoder --mode dynamic`） |
| TURBO 无真实收益 | ✅ 两次独立配对测量一负一正（−0.72% / +1.02%），波动 2–7% |
| NPU 解码慢于 CPU | ✅ 定性成立；倍数从 5.47× 漂到 3.88×，long 模式 2.71×→2.76× |
| JEV 整图 CPU | ✅ 差 0.4%（568.43 → 570.85 ms） |
| JEV NPU 编码器 189 ms | ❌ 环境回归，段错误 |

### 10. 我自己的坑：批量替换把三份文档改了字符

更新文档里的测试计数（18 → 23）时，我图快写了个 PowerShell 循环做字符串替换。
它在 `@{ '文档.md' = @( @('旧','新') ) }` 这种结构上**把单元素外层数组拍平了**，
于是 `$pair[0]` 从"整条旧文本"退化成**第一个字符**：

- `docs/01`：`'|'` → `' '`，整张表格的竖线没了
- `README.md`：`'├'` → `' '`，目录树变形
- `docs/04`：`'.'` → `'\'`，所有点号变反斜杠（`.\run.ps1` → `\\run\ps1`）

（两个内部数组的那份 `REVIEW.md` 反而没事 —— 因为外层数组没被拍平，正好说明这是
"不会报错的静默类型退化"。）

**教训（两条）**：
1. **批量文本替换别用 PowerShell 的嵌套数组 + 字符串索引**：`$pair[0]` 在字符串上取到的是字符，
   编译器不报错，跑起来照样"成功"。要么用 `edit` 工具逐处改，要么用 Python。
2. **批量改动前先 commit**。这次靠 commit 1 一条 `git checkout HEAD --` 就完整复原了三份文档，
   否则得手写重建。**"改动前先留快照"这条准则，救的往往就是这种自己造的错。**

### 11. PS 5.1 的编码坑：`.ps1` 必须是 UTF-8 **with BOM**

`run.ps1` 里 `Do-Test` 报"无法将 Do-Test 项识别为 cmdlet"，甚至把脚本源码当输出打印。
根因是 Windows PowerShell 5.1 读**无 BOM 的 UTF-8 `.ps1`** 时按系统代码页(GBK)解码，
中文注释的字节错位破坏了**语法解析**（不是显示问题，是解析问题）。

修复只需加 BOM，但普通编辑器保存又会丢，所以加了
`tests/test_project_hygiene.py::test_run_ps1_has_utf8_bom` 守住 ——
**凡"改一次就悄悄坏掉"的约束，都该有测试而不是只写在文档里。**
（同一个测试文件里还守住了 gitignore 不能有行尾注释、原始结果不得被重跑覆盖。）

## 四、下一步（按优先级）

1. **隔离 NPU 编译段错误的根因**（最高优先）：候选是驱动/编译器版本变化、
   或 fp32 图（570 MB 权重）编译期内存压力。可试：降级/升级 NPU 驱动、
   用 `nncf` 把 JEV 编码器量化成 INT8/INT4 再上 NPU（INT4 的 `model_q4.onnx` 是能跑的）、
   缩小 seq 到 256/512 看是否与图规模相关。
2. **重新导出 ONNX 并恢复同源**：用当前 `jevlite.pt` 跑
   `engine/06_export_onnx.py`，然后 `.\run.ps1 fingerprint` 确认指纹相合。
   在此之前的 NPU 数字与模型数字"各说各话"（延迟结论不受影响）。
3. **查清 12:38 那次到底做了什么**：是否重跑过 `02_train.py`。这决定了丢失的
   是否只是"一次训练产物"还是也包含别的实验配置。
4. **补齐 3 个不可复现的结果**：`export-variants.json`、`isolate-export.json`、
   `jev-fixed.json`。前两者的导出命令行可重建（`06_export_onnx.py` 的不同导出路径），
   后者需要在 `bench_jev.py` 里实现"替换 22 个 IsNaN"的图改写。
5. **给"NPU 数值一致性"定一个可接受的判据**：目前没有 —— 同权重跨实现差 23，
   而记录里的设备间差异是 0.024，两者不在同一尺度上，需要先定义"怎么算一致"。
6. **`bench_llm.py` 的协议统一**：旧 `decision-latency.json` 的 NPU 因提前 EOS 只生成
   12 token，导致 tok/s 不可比。建议统一按"固定 token 数 + 强制忽略 EOS"测。
7. **7B 悬空资产处置**：要么给它一个用途（对比 1.5B/7B 的 NPU 缩放），要么删目录留脚本。
