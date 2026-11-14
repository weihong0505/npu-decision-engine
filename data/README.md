# data/ — 数据集（不入库）

## typed-decisions

评测集 [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)，
用于 JEV-lite 的训练与评估。

```
data/typed-decisions/
├─ all/                       train + test（约 1,016 训练 / 400 测试）
├─ customer_service/
├─ security_incidents/
├─ invoice_processing/
└─ agent_trace_observability/
```

每个子目录下各有 `train-00000-of-00001.parquet` 与 `test-...parquet`。
数据集分成 4 个 workflow，**同时用多个 workflow 时注意标签宽度不同**（16/17/18/20），
见 [docs/03-约束与踩坑.md](../docs/03-约束与踩坑.md#8-数据与形状坑引擎侧来自上游-readme-与本次核实)。

获取：

```powershell
python src/fetch/dl-data.py            # 直连
$env:HF_PROXY = "http://127.0.0.1:<port>"; python src/fetch/dl-data.py   # 走代理（端口仅为示例，按自己的代理软件填）
```

脚本是幂等的：已存在且非空的文件会跳过。

> **离线也能跑**：`engine/td_data.py` 在联网失败时会回落到 HuggingFace 的本地缓存
> （本机是 `C:\Users\<user>\.cache\huggingface\datasets\...`）。
> `output/results/*.txt` 里那句 "Using the latest cached version of the dataset"
> 就是这个回落，不是错误。
