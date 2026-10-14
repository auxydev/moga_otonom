# Moga Otonom - kurulum ve demo (Windows PowerShell)
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_and_demo.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\setup_and_demo.ps1 -MetaDrive
#
# Python python.org'dan kurulmalı ("py" başlatıcısı ile). MetaDrive için ayrıca Python 3.11.
param([switch]$MetaDrive)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Komut başarısız: $Exe $($Arguments -join ' ')" }
}

# Python 3.10+ bul
if (Get-Command py -ErrorAction SilentlyContinue) { $pyExe = "py"; $pyArgs = @("-3") }
elseif (Get-Command python -ErrorAction SilentlyContinue) { $pyExe = "python"; $pyArgs = @() }
else { throw "Python bulunamadı. https://www.python.org/downloads/ adresinden Python 3.10+ kurun." }
Invoke-Checked $pyExe ($pyArgs + @("-c", "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 'Python 3.10+ gerekli')"))

Write-Host "==> 1/3 Sanal ortam (.venv) ve paketler"
if (-not (Test-Path ".venv")) { Invoke-Checked $pyExe ($pyArgs + @("-m", "venv", ".venv")) }
$venvPy = ".venv\Scripts\python.exe"
Invoke-Checked $venvPy @("-m", "pip", "install", "-q", "--upgrade", "pip")
Invoke-Checked $venvPy @("-m", "pip", "install", "-q", "-r", "requirements.txt")

Write-Host "==> 2/3 Hızlı testler (~30 s)"
Invoke-Checked $venvPy @("-m", "pytest", "-q", "-n", "4", "-m", "not scenario", "-p", "no:cacheprovider")

Write-Host "==> 3/3 Demo: kırsal yolda karşı trafiği bekleyip arızalı aracı geçme (~1 dk)"
Invoke-Checked $venvPy @("scripts\run_scenario.py", "m4_rural_bypass", "--gif", "out\demo.gif", "--overview", "out\demo.png")
Write-Host "Çıktılar: out\demo.gif (animasyon), out\demo.png (rota ve hız)"
Write-Host "Canlı izlemek için: .venv\Scripts\python.exe scripts\run_scenario.py m5_urban_mission --live"

if ($MetaDrive) {
    Write-Host "==> MetaDrive ortamı (.venv-metadrive, Python 3.11)"
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw "MetaDrive için 'py -3.11' gerekli (python.org'dan Python 3.11 kurun)." }
    if (-not (Test-Path ".venv-metadrive")) { Invoke-Checked "py" @("-3.11", "-m", "venv", ".venv-metadrive") }
    $mdPy = ".venv-metadrive\Scripts\python.exe"
    Invoke-Checked $mdPy @("-m", "pip", "install", "-q", "--upgrade", "pip")
    Invoke-Checked $mdPy @("-m", "pip", "install", "-q", "-r", "requirements-metadrive.txt")
    Invoke-Checked $mdPy @("-m", "pip", "install", "-q", "--no-deps", "metadrive-simulator==0.2.6.0")
    Invoke-Checked $mdPy @("scripts\run_metadrive.py", "--seeds", "0", "1", "--map", "3", "--plot", "--out", "out\metadrive")
    Write-Host "Çıktılar: out\metadrive\ (her sürüş için harita + hız grafiği, JSON)"
    Write-Host "3B pencerede izlemek için: .venv-metadrive\Scripts\python.exe scripts\run_metadrive.py --seeds 3 --render"
}
