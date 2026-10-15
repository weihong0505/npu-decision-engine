# models/ — 模型权重（不入库）

本目录被 `.gitignore` 排除（约 5.4 GB）。新克隆后按需重新获取。

| 目录 | 大小 | 来源 | 用途 |
|---|---|---|---|
| `ModernBERT-base/` | 216 MB | HuggingFace `answerdotai/ModernBERT-base`（另含 `onnx/model_q4.onnx` 224.8 MB 的 INT4 导出） | 编码器主体；`bench_encoder.py` 的被测对象 |
| `Qwen2.5-1.5B-Instruct-int4-ov/` | 893 MB | `OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov` | `bench_llm.py` 的被测对象 |
| `Qwen2.5-7B-Instruct-int4-ov/` | 4.3 GB | `OpenVINO/Qwen2.5-7B-Instruct-int4-ov` | **悬空资产**：2026-09 那轮实验里没有任何脚本或结论引用它 |

获取：

```powershell
python src/fetch/dl_1_5b.py     # Qwen2.5-1.5B
python src/fetch/dl_7b.py       # Qwen2.5-7B（4.3 GB，非必需）
```

`ModernBERT-base/` 由 `transformers` 在训练/导出时自动从 HF 缓存取用，
若缓存里没有，需联网或从别处拷贝。

> **7B 为什么留着**：它占了 4.3 GB 且无下游引用，整理时经确认**不删**——
> 保留 `src/fetch/dl_7b.py` 就保证了可重新获取，删除是不可逆的。
> 如果将来确认无用，删目录即可，脚本会重新下载。
