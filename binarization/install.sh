#!/usr/bin/env bash
# Install the binarization package and wsipatch in a conda environment. See ../INSTALL.md.
# Install CUDA PyTorch before the editable packages to preserve the chosen build.
# Usage: bash install.sh
# Conda activation hooks may use unset variables, so nounset is disabled.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSIPATCH_DIR="$SCRIPT_DIR/../wsipatch"
cd "$SCRIPT_DIR"

ENV_NAME="${BINARIZATION_ENV:-binarization}"
PY_VER="${BINARIZATION_PYTHON:-3.10}"
CUDA="${BINARIZATION_CUDA:-cu128}"
TORCH="${BINARIZATION_TORCH:-2.11.0}"
TORCHVISION="${BINARIZATION_TORCHVISION:-0.26.0}"

echo "==> binarization -> conda env '$ENV_NAME' (Python $PY_VER, $CUDA)"

if ! command -v conda >/dev/null 2>&1; then
  echo "!! conda not found. Install Miniforge/Miniconda first:"
  echo "   https://github.com/conda-forge/miniforge"
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "$WSIPATCH_DIR" ]; then
  echo "!! Sibling 'wsipatch' package not found at $WSIPATCH_DIR"
  echo "   binarization needs it for ASAP annotation parsing."
  exit 1
fi

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

echo "==> Installing wsipatch + binarization (editable)"
python -m pip install -e "$WSIPATCH_DIR" -e .

if ! python -c "import openslide" >/dev/null 2>&1; then
  echo "==> OpenSlide C library not importable; trying conda-forge"
  conda install -y -c conda-forge openslide \
    || python -m pip install "openslide-bin>=4.0.0" \
    || { echo "!! Could not provide OpenSlide. On HPC try: module load openslide"; exit 1; }
fi

echo "==> Verifying"
python - <<'PY'
import importlib
for n in ("wsipatch", "binarization"):
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
    CLI:         tissue-binarize --help
    Checkpoint:  supply a segmentation checkpoint with --model-path
EOF
