$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython -PathType Leaf)) {
    throw 'DriftLens Python environment is missing. Run the project setup first.'
}
Push-Location -LiteralPath $projectRoot
try {
    & $projectPython -m streamlit run app.py --server.address 127.0.0.1 --server.port 8510 --browser.gatherUsageStats false
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
