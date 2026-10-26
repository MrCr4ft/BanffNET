#!/usr/bin/env bash
# Install the BanffNET pipeline in a conda environment. See INSTALL.md.
# Install CUDA PyTorch before the editable packages to preserve the chosen build.
# Usage: bash install.sh
# Conda activation hooks may use unset variables, so nounset is disabled.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${BANFFNET_ENV:-banffnet}"
PY_VER="${BANFFNET_PYTHON:-3.10}"
CUDA="${BANFFNET_CUDA:-cu128}"
TORCH="${BANFFNET_TORCH:-2.11.0}"
TORCHVISION="${BANFFNET_TORCHVISION:-0.26.0}"
XFORMERS="${BANFFNET_XFORMERS:-0.0.35}"
DATA_ROOT="${BANFFNET_DATA_ROOT:-}"

echo "==> BanffNET pipeline -> conda env '$ENV_NAME' (Python $PY_VER, $CUDA)"

if ! command -v conda >/dev/null 2>&1; then
  echo "!! conda not found. Install Miniforge/Miniconda first:"
  echo "   https://github.com/conda-forge/miniforge"
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda activate "$ENV_NAME" 2>/dev/null; then
  echo "==> Creating conda env '$ENV_NAME' (Python $PY_VER)"
  conda create -y -n "$ENV_NAME" "python=$PY_VER" pip wheel
  conda activate "$ENV_NAME"
else
  echo "==> Reusing existing conda env '$ENV_NAME'"
fi
python -m pip install --upgrade pip wheel setuptools >/dev/null

echo "==> Installing torch ${TORCH}+${CUDA} / torchvision ${TORCHVISION}+${CUDA}"
python -m pip install --index-url "https://download.pytorch.org/whl/${CUDA}" \
  "torch==${TORCH}+${CUDA}" "torchvision==${TORCHVISION}+${CUDA}"
echo "==> Installing xformers ${XFORMERS}"
python -m pip install "xformers==${XFORMERS}"

echo "==> Installing BanffNET packages (editable)"
python -m pip install \
  -e ./wsipatch \
  -e ./binarization \
  -e ./features_extraction \
  -e ./banffnet \
  -e ./explainability

if ! python -c "import openslide" >/dev/null 2>&1; then
  echo "==> OpenSlide C library not importable; trying conda-forge"
  conda install -y -c conda-forge openslide \
    || python -m pip install "openslide-bin>=4.0.0" \
    || { echo "!! Could not provide OpenSlide. On HPC try: module load openslide"; exit 1; }
fi

if [ -z "$DATA_ROOT" ] && [ -t 0 ]; then
  read -rp "==> Path for BANFFNET_DATA_ROOT (models + staining configs), blank to skip: " DATA_ROOT
fi
if [ -n "$DATA_ROOT" ]; then
  [ -d "$DATA_ROOT" ] || echo "   (note: '$DATA_ROOT' does not exist yet)"
  conda env config vars set "BANFFNET_DATA_ROOT=$DATA_ROOT" -n "$ENV_NAME"
  echo "==> BANFFNET_DATA_ROOT='$DATA_ROOT' will auto-export on 'conda activate $ENV_NAME'"
else
  echo "==> BANFFNET_DATA_ROOT not set. Configure it later with:"
  echo "    conda env config vars set BANFFNET_DATA_ROOT=/path -n $ENV_NAME"
fi

echo "==> Verifying"
python - <<'PY'
import importlib
for n in ("wsipatch", "binarization", "features_extraction", "banffnet", "explainability"):
    m = importlib.import_module(n)
    print(f"   {n:<20} {getattr(m, '__version__', 'ok')}")
import openslide
print(f"   {'openslide':<20} {openslide.__version__}")
import torch
print(f"   {'torch':<20} {torch.__version__}  (cuda build: {torch.version.cuda}, "
      f"available: {torch.cuda.is_available()})")
from wsipatch.annotations import ASAPAnnotations as A1
from binarization import ASAPAnnotations as A2
assert A1 is A2, "ASAPAnnotations is not the single shared class!"
print("   annotations          single shared class OK")
PY

cat <<EOF

==> Done.
    Activate:    conda activate $ENV_NAME
    CLIs:        banffnet --help | tissue-binarize --help | wsipatch --help | features-extract --help | run-explainability --help
    Model root:  BANFFNET_DATA_ROOT is exported automatically on activate; change it with
                 conda env config vars set BANFFNET_DATA_ROOT=/path -n $ENV_NAME
EOF
