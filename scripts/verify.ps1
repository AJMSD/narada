Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "==> compileall"
python -m compileall src tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> ruff format --check ."
ruff format --check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> ruff check ."
ruff check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> mypy src"
mypy src
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> pytest -q"
pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "All checks passed."
