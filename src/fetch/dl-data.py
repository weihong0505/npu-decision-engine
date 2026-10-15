"""下载 typed-decisions 数据集的 10 个 parquet 分片到 data/typed-decisions/。

整理前：目标目录写死为 `g:\\npu-experiment\\typed-decisions`，并硬编码本机代理
`http://127.0.0.1:<port>`。现在统一走 paths，代理改由环境变量 HF_PROXY 控制（默认直连）。
"""
import pathlib, sys, os, time, socket, urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # -> src/
import paths

REPO="LocalLLaMA/typed-decisions"; DEST=str(paths.DATASET)
BASE=f"https://huggingface.co/datasets/{REPO}/resolve/main/"
FILES=["all/train-00000-of-00001.parquet","all/test-00000-of-00001.parquet",
       "customer_service/train-00000-of-00001.parquet","customer_service/test-00000-of-00001.parquet",
       "security_incidents/train-00000-of-00001.parquet","security_incidents/test-00000-of-00001.parquet",
       "invoice_processing/train-00000-of-00001.parquet","invoice_processing/test-00000-of-00001.parquet",
       "agent_trace_observability/train-00000-of-00001.parquet","agent_trace_observability/test-00000-of-00001.parquet"]
if paths.HF_PROXY:
    op=urllib.request.build_opener(urllib.request.ProxyHandler({"http":paths.HF_PROXY,"https":paths.HF_PROXY}))
    print(f"使用代理 {paths.HF_PROXY}")
else:
    op=urllib.request.build_opener()
for f in FILES:
    p=os.path.join(DEST,f.replace("/",os.sep)); os.makedirs(os.path.dirname(p),exist_ok=True)
    if os.path.exists(p) and os.path.getsize(p)>0: print("  跳过",f); continue
    for a in range(6):
        try:
            with op.open(urllib.request.Request(BASE+f,headers={"User-Agent":"Mozilla/5.0"}),timeout=60) as r:
                data=r.read()
            open(p,"wb").write(data); print(f"  OK {f} {len(data)/1024:.0f} KB"); break
        except Exception as e:
            print(f"  retry{a} {f}: {str(e)[:50]}"); time.sleep(2)
