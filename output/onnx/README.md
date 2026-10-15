# output/onnx/ — 导出产物（不入库）

约 7 GB，被 `.gitignore` 排除。**这些文件是 2026-09-22 的产物，可以重新生成**，
但见文末的重要提醒。

## 清单

| 文件 | 大小 | 是什么 | 谁在用 |
|---|---|---|---|
| `jevlite.pt` | 568 MB | **训练好的检查点**（权重 + qid/lid + 温度） | `bench_jev.py` 的 head 权重来源；`probe/*` |
| `jevlite.onnx` + `.data` | 2.3 MB + 568 MB | 整图导出（encoder + head，含 22 个 `IsNaN`） | `bench_jev.py --mode full`、`probe/jev-error.py` |
| `jevlite_static.onnx` | 573 MB | 整图的静态形状变体 | 历史产物 |
| `jev_encoder.onnx` + `.data` | 2.0 MB + 569 MB | **编码器专用**导出（静态 [1,768]），NPU 混合方案的被测对象 | `bench_jev.py --mode hybrid/verify` |
| `jev_encoder_int8.onnx` | 144 MB | 编码器 INT8 量化版（NPU 上是负优化：3374 ms） | 反面对照 |
| `enc_dynamo_sdpa.onnx` + `.data` | 2.0 MB + 571 MB | dynamo 导出、SDPA 注意力（NPU p50 189 ms） | `bench_encoder.py` 的变体对照、`bench_jev.py --mode verify` |
| `enc_dynamo_eager.onnx` + `.data` | 1.8 MB + 571 MB | dynamo 导出、eager 注意力（NPU p50 180 ms，最快） | 变体对照 |

导出（`--out` 指定文件名）：

```powershell
python engine/06_export_onnx.py --quantize
```

> **`.onnx` 与 `.onnx.data` 必须成对且同名**：侧车文件名被写进了 `.onnx` 内部
> （`external_data.location`）。改名/移动要同时改内部引用，否则加载时报
> `Invalid usage of method for externally stored data`。修法见
> [docs/03-约束与踩坑.md](../docs/03-约束与踩坑.md#6-onnx-侧车权重改名会炸)。

## ⚠️ 重要提醒：这批 ONNX 与当前 `jevlite.pt` **不同源**

本次整理核对过（`src/tools/onnx_weight_fingerprint.py` + `bench_jev.py --mode verify`）：

- `jev_encoder.onnx`：46 个可配对张量**全部**与当前检查点不一致
- `jevlite.onnx`：47 个可配对张量**全部**不一致
- 输出侧：整图 logits 差 1.65，编码器隐藏状态差 28.5

`output/results/export-out.txt` 显示导出当时 parity 是 6.32e-06，说明**导出时是同源的**；
差异出现在 `jevlite.pt` 于当天 12:38 被 `03_calibrate.py` 重新落盘之后。
**延迟结论不受影响**（延迟与权重取值无关），但不要把 `max_abs_err` 读成权重保真度。

要恢复一致性：用当前 `jevlite.pt` 重新导出，再跑 `bench_jev.py --mode verify` 确认指纹相合。

## 已删除的产物

2026-10-15 整理时按确认删除了 6 个"明确被取代"的文件（约 2.24 GB）。
清单含 SHA256 与重生成说明：`backup/20261015-113854/DELETED-MANIFEST.md`。
