# archive/scripts/ — 被取代的脚本（保留留痕）

这 4 个脚本**已被取代或已损坏**，移到这里是为了留下决策痕迹，不建议再直接使用。

| 脚本 | 被谁取代 | 原因 |
|---|---|---|
| `measure-len.py` | [`../../src/probe/measure-len2.py`](../../src/probe/measure-len2.py) | 只取前 400 条样本；len2 跑全量、多一档 cap 192、打印超限例数。原件还含死代码 `s = encode(r) if callable(encode) else None`（`s` 从未使用）与未使用的 `import statistics` |
| `inspect-onnx.py` | [`../../src/probe/jev-error.py`](../../src/probe/jev-error.py) | 前者只打印 IO 形状；后者是超集（reshape + 算子直方图 + NPU 编译尝试），真正产出了根因结论 |
| `shape-probe.py` | [`../../src/probe/real-shape.py`](../../src/probe/real-shape.py) | 前者硬编码 `max_len=768` 且不加载 `lid` 映射；后者从真实检查点取 `max_len/qid/lid`，形状才作数 |
| `inspect-data.py` | [`../../src/probe/max-labels.py`](../../src/probe/max-labels.py) | 前者用自写的启发式 `count_labels` 猜键；后者用官方 `typed_schema.iter_labels`。**且前者 `sys.path` 写死为不存在的 `e:\npu-experiment`**，直接跑会 `ModuleNotFoundError` |

## 关于 `e:` 盘那个坏路径

`inspect-data.py` 与 `src/probe/measure-len2.py` 原本都写着：

```python
sys.path.insert(0, r"e:\npu-experiment".replace("pU","pu") + r"\open-jev-typed-decision-engine")
```

两个问题：`"npu-experiment"` 里没有大写 `pU`，`.replace` 是空操作；
盘符是 `e:`，而本机 **`E:\npu-experiment` 并不存在**（E: 是另一块空盘）。
所以这是复制粘贴留下的坏路径。

`measure-len2.py` 作为"取代者"反而跑不通，本次整理已把它修成走 `src/paths.py`；
`inspect-data.py` 因为功能已被 `max-labels.py` 完全覆盖，就留在归档里不再修。

## 同目录的 results-20260922/

`archive/results-20260922/` 是 5 个实验日志的**转码副本**（UTF-16LE → UTF-8，内容逐字保留）。
`output/results/` 里那份是进一步清洗过的（剥 PowerShell 包装、去 ANSI、遮蔽本机路径），
转换逻辑见 `src/tools/normalize_logs.py`。
