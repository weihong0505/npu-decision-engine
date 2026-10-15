<#
.SYNOPSIS
    npu-decision-engine 的统一入口。

.DESCRIPTION
    把"环境自检 → 测试 → 跑基准 → 核对产物"这些常用动作收成一条命令，
    避免每次从文档里抄一长串 python 调用。

.EXAMPLE
    .\run.ps1 check          # 环境、设备、大件自检
    .\run.ps1 test           # 跑测试（不需要 NPU）
    .\run.ps1 bench-jev      # 复现 JEV 拆图结论
    .\run.ps1 all            # 全部基准（慢，约 15 分钟）
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('check', 'test', 'bench-llm', 'bench-encoder', 'bench-jev', 'all', 'quality', 'logs', 'fingerprint', 'help')]
    [string]$Command = 'check',

    [string]$OutDir = 'output/results/rerun',
    [string]$Devices = 'CPU,GPU.0,GPU.1,NPU'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# 输出一律按 UTF-8 走，避免 PowerShell 5.1 把中文写成乱码
$env:PYTHONIOENCODING = 'utf-8'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

function Write-Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Invoke-Py { param([string[]]$PyArgs) & python @PyArgs; if ($LASTEXITCODE -ne 0) { Write-Warning "python 退出码 $LASTEXITCODE" } }

function Do-Check {
    Write-Step 'Python'
    Invoke-Py @('-V')

    Write-Step 'OpenVINO 与设备'
    $code = @'
import openvino as ov
core = ov.Core()
print("openvino", ov.get_version())
print("devices ", core.available_devices)
for d in core.available_devices:
    print(f"  {d:<7} {core.get_property(d, 'FULL_DEVICE_NAME')}")
'@
    Invoke-Py @('-c', $code)

    Write-Step '大件是否就位'
    foreach ($p in @('models\ModernBERT-base\config.json',
                     'models\ModernBERT-base\onnx\model_q4.onnx',
                     'models\Qwen2.5-1.5B-Instruct-int4-ov',
                     'data\typed-decisions',
                     'output\onnx\jevlite.pt',
                     'output\onnx\jev_encoder.onnx',
                     'output\results')) {
        $mark = if (Test-Path $p) { '[OK]  ' } else { '[MISS]' }
        $color = if (Test-Path $p) { 'Green' } else { 'Yellow' }
        Write-Host "  $mark $p" -ForegroundColor $color
    }

    Write-Step '关键计数'
    Write-Host ("  原始结果文件: {0}" -f (Get-ChildItem 'output\results' -File -ErrorAction SilentlyContinue).Count)
    Write-Host ("  重跑结果文件: {0}" -f (Get-ChildItem $OutDir -File -ErrorAction SilentlyContinue).Count)
    Write-Host ("  测试文件:     {0}" -f (Get-ChildItem 'tests' -Filter 'test_*.py' -ErrorAction SilentlyContinue).Count)
}

function Do-Test { Write-Step 'pytest'; Invoke-Py @('-m', 'pytest') }

function Do-BenchLlm {
    Write-Step "LLM 设备对比 -> $OutDir"
    Invoke-Py @('src/bench/bench_llm.py', '--mode', 'all', '--outdir', $OutDir, '--devices', $Devices)
}

function Do-BenchEncoder {
    Write-Step "编码器基准 -> $OutDir"
    Invoke-Py @('src/bench/bench_encoder.py', '--mode', 'all', '--outdir', $OutDir)
}

function Do-BenchJev {
    Write-Step "JEV 整图 -> $OutDir"
    Invoke-Py @('src/bench/bench_jev.py', '--mode', 'full', '--outdir', $OutDir)
    Write-Step "JEV 拆图 -> $OutDir"
    Invoke-Py @('src/bench/bench_jev.py', '--mode', 'hybrid', '--outdir', $OutDir)
}

function Do-Quality {
    Write-Step '引擎全量校验（无需 GPU）'
    Push-Location engine
    try { Invoke-Py @('smoke_test.py', '--all') } finally { Pop-Location }
}

function Do-Fingerprint {
    Write-Step 'ONNX 权重 vs 检查点'
    foreach ($f in @('jev_encoder.onnx', 'jevlite.onnx', 'enc_dynamo_sdpa.onnx')) {
        if (Test-Path "output\onnx\$f") {
            Write-Host "--- $f"
            Invoke-Py @('src/tools/onnx_weight_fingerprint.py', '--onnx', "output/onnx/$f", '--sample', '3')
        }
    }
    Write-Step '输出侧二次核对'
    Invoke-Py @('src/bench/bench_jev.py', '--mode', 'verify', '--outdir', $OutDir)
}

function Do-Logs {
    Write-Step '规范化 output/results/*.txt（先 dry-run）'
    Invoke-Py @('src/tools/normalize_logs.py', '--dry-run')
    Write-Host '  确认无误后去掉 --dry-run 执行' -ForegroundColor Yellow
}

$sw = [System.Diagnostics.Stopwatch]::StartNew()
switch ($Command) {
    'check'       { Do-Check }
    'test'        { Do-Test }
    'bench-llm'   { Do-BenchLlm }
    'bench-encoder' { Do-BenchEncoder }
    'bench-jev'   { Do-BenchJev }
    'quality'     { Do-Quality }
    'fingerprint' { Do-Fingerprint }
    'logs'        { Do-Logs }
    'all'         { Do-Check; Do-Test; Do-Quality; Do-BenchEncoder; Do-BenchJev; Do-BenchLlm }
    default {
        Write-Host '用法: .\run.ps1 <check|test|bench-llm|bench-encoder|bench-jev|all|quality|fingerprint|logs>' -ForegroundColor Cyan
        Write-Host '详见 docs/04-复现手册.md'
    }
}
$sw.Stop()
Write-Host ("`n[完成] {0}  用时 {1:N1}s" -f $Command, $sw.Elapsed.TotalSeconds) -ForegroundColor Green
