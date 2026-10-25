#!/usr/bin/env bash

# Stop if a command fails or an unset variable is used.
set -euo pipefail

# Resolve paths relative to this script, so it can be run from any directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Change the environment name with: ENV_NAME=my_env bash build_env.sh
# Keep Python, PyTorch, and CUDA versions aligned with the prebuilt wheel.
ENV_NAME="${ENV_NAME:-medvtok}"
PYTHON_VERSION="3.11"
WHL_FILE="flash_attn-2.7.4+cu126torch2.7-cp311-cp311-linux_x86_64.whl"
WHL_PATH="$SCRIPT_DIR/$WHL_FILE"
# This release's SHA-256 matches the wheel originally used by this repository.
# Source: https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/tag/v0.5.4
WHL_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.7.4%2Bcu126torch2.7-cp311-cp311-linux_x86_64.whl"
WHL_SHA256="077510a9a200499eac38926beda558c607c685f96f0a39c3dbe0969027e090c8"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    fail "This wheel requires Linux x86_64."
fi
command -v conda >/dev/null 2>&1 || fail "Install Conda and add it to PATH first."
[[ -f "$SCRIPT_DIR/requirements.txt" ]] || fail "requirements.txt is missing."

# Enable 'conda activate' inside this non-interactive shell.
eval "$(conda shell.bash hook)"

# Reuse an existing environment without deleting it, or create a new one.
if conda activate "$ENV_NAME" 2>/dev/null; then
    echo ">>> Using existing Conda environment: $ENV_NAME"
else
    echo ">>> Creating Conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
    conda activate "$ENV_NAME"
fi

# The cp311 wheel only works with Python 3.11.
python - <<'PY'
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        "This environment must use Python 3.11. "
        "Choose a new environment with ENV_NAME=my_env bash build_env.sh."
    )
PY

# Install the dependencies listed in requirements.txt from the repository root.
echo ">>> Installing project dependencies"
pip install -r requirements.txt

verify_wheel() {
    [[ -f "$1" ]] && printf '%s  %s\n' "$WHL_SHA256" "$1" | sha256sum --check --status
}

# Cache the wheel in the repository root. Reuse it when the checksum matches.
if verify_wheel "$WHL_PATH"; then
    echo ">>> Using cached wheel: $WHL_PATH"
else
    echo ">>> Downloading FlashAttention wheel to: $WHL_PATH"
    # Download to a temporary file, then place the verified wheel at its final path.
    if command -v wget >/dev/null 2>&1; then
        wget --tries=3 --timeout=30 -O "$WHL_PATH.part" "$WHL_URL"
    elif command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 3 --connect-timeout 30 \
            --output "$WHL_PATH.part" "$WHL_URL"
    else
        fail "Install wget or curl to download the FlashAttention wheel."
    fi
    verify_wheel "$WHL_PATH.part" || fail "FlashAttention wheel checksum mismatch. Run the script again to retry."
    mv -- "$WHL_PATH.part" "$WHL_PATH"
fi

# Dependencies are already installed; preserve the selected PyTorch/CUDA versions.
echo ">>> Installing FlashAttention 2.7.4"
python -m pip install --no-deps "$WHL_PATH"

# Confirm that the compiled extension imports and report GPU visibility.
echo ">>> Checking the environment"
python - <<'PY'
import torch
import flash_attn

if torch.__version__ != "2.7.1+cu126" or torch.version.cuda != "12.6":
    raise SystemExit("Expected PyTorch 2.7.1+cu126 with CUDA 12.6.")
if flash_attn.__version__ != "2.7.4":
    raise SystemExit("Expected FlashAttention 2.7.4.")

print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("FlashAttention:", flash_attn.__version__)
print("GPU available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("Run feature extraction and training on a node with an NVIDIA GPU.")
PY

echo ">>> Environment setup complete: $ENV_NAME"
echo ">>> Activate it in your shell with: conda activate $ENV_NAME"
