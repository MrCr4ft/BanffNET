<#
Install the BanffNET pipeline in a conda environment. See INSTALL.md.
Usage: ./install.ps1
CUDA PyTorch is installed before the editable packages. xformers is optional
on Windows when a compatible wheel is unavailable.
#>
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$EnvName = if ($env:BANFFNET_ENV)         { $env:BANFFNET_ENV }         else { "banffnet" }
$PyVer   = if ($env:BANFFNET_PYTHON)      { $env:BANFFNET_PYTHON }      else { "3.10" }
$Cuda    = if ($env:BANFFNET_CUDA)        { $env:BANFFNET_CUDA }        else { "cu128" }
$Torch   = if ($env:BANFFNET_TORCH)       { $env:BANFFNET_TORCH }       else { "2.11.0" }
$TVis    = if ($env:BANFFNET_TORCHVISION) { $env:BANFFNET_TORCHVISION } else { "0.26.0" }
$Xform   = if ($env:BANFFNET_XFORMERS)    { $env:BANFFNET_XFORMERS }    else { "0.0.35" }
$DataRoot = if ($env:BANFFNET_DATA_ROOT)  { $env:BANFFNET_DATA_ROOT }   else { "" }

Write-Host "==> BanffNET pipeline -> conda env '$EnvName' (Python $PyVer, $Cuda)"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda not found. Install Miniforge/Miniconda first: https://github.com/conda-forge/miniforge"
}

function Invoke-InEnv {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$CmdArgs)
    conda run -n $EnvName --no-capture-output @CmdArgs
    if ($LASTEXITCODE -ne 0) { throw "command failed: $($CmdArgs -join ' ')" }
}

conda run -n $EnvName python -c "import sys" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "==> Creating conda env '$EnvName' (Python $PyVer)"
    conda create -y -n $EnvName "python=$PyVer" pip wheel
    if ($LASTEXITCODE -ne 0) { throw "conda create failed" }
} else {
    Write-Host "==> Reusing existing conda env '$EnvName'"
}
Invoke-InEnv python -m pip install --upgrade pip wheel setuptools

Write-Host "==> Installing torch $Torch+$Cuda / torchvision $TVis+$Cuda"
Invoke-InEnv python -m pip install --index-url "https://download.pytorch.org/whl/$Cuda" "torch==$Torch+$Cuda" "torchvision==$TVis+$Cuda"
Write-Host "==> Installing xformers $Xform (optional)"
conda run -n $EnvName --no-capture-output python -m pip install "xformers==$Xform"
if ($LASTEXITCODE -ne 0) { Write-Warning "xformers install failed; continuing without it (pipeline runs without xformers)." }

Write-Host "==> Installing BanffNET packages (editable)"
Invoke-InEnv python -m pip install -e ./wsipatch -e ./binarization -e ./features_extraction -e ./banffnet -e ./explainability

conda run -n $EnvName python -c "import openslide" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "==> Installing openslide-bin wheel"
    Invoke-InEnv python -m pip install "openslide-bin>=4.0.0"
}

if (-not $DataRoot -and [Environment]::UserInteractive) {
    $DataRoot = Read-Host "==> Path for BANFFNET_DATA_ROOT (models + staining configs), blank to skip"
}
if ($DataRoot) {
    if (-not (Test-Path $DataRoot)) { Write-Host "   (note: '$DataRoot' does not exist yet)" }
    conda env config vars set "BANFFNET_DATA_ROOT=$DataRoot" -n $EnvName
    if ($LASTEXITCODE -ne 0) { throw "failed to set BANFFNET_DATA_ROOT" }
    Write-Host "==> BANFFNET_DATA_ROOT='$DataRoot' will auto-export on 'conda activate $EnvName'"
} else {
    Write-Host "==> BANFFNET_DATA_ROOT not set. Configure it later with:"
    Write-Host "    conda env config vars set BANFFNET_DATA_ROOT=C:\path -n $EnvName"
}

Write-Host "==> Verifying"
$verify = @'
import importlib
for n in ("wsipatch", "binarization", "features_extraction", "banffnet", "explainability"):
    m = importlib.import_module(n)
    print(f"   {n:<20} {getattr(m, '__version__', 'ok')}")
import openslide
print(f"   {'openslide':<20} {openslide.__version__}")
import torch
print(f"   {'torch':<20} {torch.__version__}  (cuda build: {torch.version.cuda}, available: {torch.cuda.is_available()})")
from wsipatch.annotations import ASAPAnnotations as A1
from binarization import ASAPAnnotations as A2
assert A1 is A2, "ASAPAnnotations is not the single shared class!"
print("   annotations          single shared class OK")
'@
Invoke-InEnv python -c $verify

Write-Host ""
Write-Host "==> Done."
Write-Host "    Activate:    conda activate $EnvName"
Write-Host "    CLIs:        banffnet --help | tissue-binarize --help | wsipatch --help | features-extract --help | run-explainability --help"
Write-Host "    Model root:  BANFFNET_DATA_ROOT is exported automatically on activate; change it with"
Write-Host "                 conda env config vars set BANFFNET_DATA_ROOT=C:\path -n $EnvName"
