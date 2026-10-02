$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$environmentPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$env:PIP_CACHE_DIR = Join-Path $projectRoot 'tools\pip_cache'
$env:YOLO_CONFIG_DIR = Join-Path $projectRoot '.settings'
if (-not (Test-Path -LiteralPath $environmentPython)) {
    $codexPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (Test-Path -LiteralPath $codexPython) { & $codexPython -m venv (Join-Path $projectRoot '.venv') }
    elseif (Get-Command py -ErrorAction SilentlyContinue) { & py -3.12 -m venv (Join-Path $projectRoot '.venv') }
    else { & python -m venv (Join-Path $projectRoot '.venv') }
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 and retry setup.' }
}
& $environmentPython -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'CPU detector runtime installation failed.' }
& $environmentPython -m pip install -r (Join-Path $projectRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Project dependencies installation failed.' }
New-Item -ItemType Directory -Path (Join-Path $projectRoot 'models') -Force | Out-Null
$weightsPath = Join-Path $projectRoot 'models\yolov8n.pt'
if (-not (Test-Path -LiteralPath $weightsPath)) {
    Invoke-WebRequest -Uri 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt' -OutFile $weightsPath
}
& $environmentPython -c 'import cv2,torch,streamlit; print("DriftLens environment ready. CPU PyTorch:",torch.__version__)'
Write-Host 'Setup finished. Open launch.cmd to start DriftLens.'
