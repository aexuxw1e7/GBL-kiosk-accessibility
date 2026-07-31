$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$bundledPython = "C:\Users\luevi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue

if ($pythonCommand) {
    $python = $pythonCommand.Source
} elseif (Test-Path -LiteralPath $bundledPython) {
    $python = $bundledPython
} else {
    throw "Python 3.11 이상이 필요합니다."
}

& $python (Join-Path $projectRoot "run.py") @args
exit $LASTEXITCODE
