#!/bin/bash
set -e

# =============================================================================
# First clone the ARENA_3.0 repo using:
#   git clone -b alignment-science https://github.com/callummcdougall/ARENA_3.0.git
# Then, usage:
#   bash ARENA_3.0/install.sh                        # RunPod (default), with llm-context repo
#   bash ARENA_3.0/install.sh --platform vastai      # Vast.ai platform
#   bash ARENA_3.0/install.sh --no-llm-context       # Skip cloning arena-llm-context
#
# Assumes a PyTorch base image (e.g. RunPod's runpod/pytorch:*) so that torch +
# CUDA libs are already installed in the system Python. We create a venv with
# --system-site-packages so the env inherits that preinstalled torch (no multi-GB
# re-download) and only the remaining ARENA deps get installed, via uv.
# =============================================================================

# Defaults
PLATFORM="runpod"
ENV_NAME="arena-env"
VENV_DIR="$HOME/$ENV_NAME"
CLONE_LLM_CONTEXT=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --no-llm-context) CLONE_LLM_CONTEXT=false; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== Setup: platform=$PLATFORM, clone_llm_context=$CLONE_LLM_CONTEXT ==="

# --- Prefer IPv4 ---
# Some pods have a broken/black-holed IPv6 route, which makes pip/uv/git/apt crawl as
# they wait on dead IPv6 connections. Make getaddrinfo hand out IPv4 first. Idempotent.
echo "=== Preferring IPv4 for downloads ==="
GAI_RULE='precedence ::ffff:0:0/96  100'
grep -qxF "$GAI_RULE" /etc/gai.conf 2>/dev/null || echo "$GAI_RULE" >> /etc/gai.conf

# --- Install git ---
echo "=== Installing system packages ==="
if [[ "$PLATFORM" == "runpod" ]]; then
    apt update && apt install -y git curl
elif [[ "$PLATFORM" == "vastai" ]]; then
    sudo apt update && sudo apt install -y git
fi

# Maybe clone the repo which gives you extra context for LLMs (to help with exercises)
if $CLONE_LLM_CONTEXT; then
    REPO="callummcdougall/arena-llm-context"
    BRANCH="main"
    echo "=== Cloning $REPO (branch: $BRANCH) ==="
    git clone -b "$BRANCH" "https://github.com/${REPO}.git"
fi

# --- Create venv that inherits the base image's preinstalled torch ---
echo "=== Creating venv '$ENV_NAME' from base Python: $(which python) ==="
python -m venv --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"
echo "=== Active Python: $(which python) ==="

# Fail fast if the base image didn't provide the full torch stack (wrong template).
# These are inherited via --system-site-packages, so uv will skip re-downloading them.
echo "=== Verifying inherited torch stack ==="
python - <<'PY' || { echo "ERROR: base image is missing part of the torch stack. Use a full PyTorch base image."; exit 1; }
import torch, torchvision, torchaudio
print("Found torch", torch.__version__, "CUDA", torch.version.cuda)
print("Found torchvision", torchvision.__version__, "torchaudio", torchaudio.__version__)
PY

# --- Install Python deps from primary repo with uv (parallel, fast) ---
PRIMARY_REPO_DIR="ARENA_3.0"
echo "=== Installing Python dependencies from $PRIMARY_REPO_DIR ==="
cd "$PRIMARY_REPO_DIR"
pip install -U pip uv
# torch/torchvision/torchaudio are already satisfied via system-site-packages and are skipped.
# --index-strategy unsafe-best-match: pick the best version across PyPI + the pytorch index,
# instead of locking a package to the first index that lists it (avoids the stale `requests` pin).
uv pip install --index-strategy unsafe-best-match -r requirements.txt
cd ..

# --- Register Jupyter kernel for this env ---
echo "=== Registering Jupyter kernel '$ENV_NAME' ==="
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

# --- VS Code workspace settings ---
echo "=== Configuring VS Code workspace settings ==="

HOME_DIR="$HOME"
mkdir -p "$HOME_DIR/.vscode"
cat > "$HOME_DIR/.vscode/settings.json" << EOF
{
    "python.defaultInterpreterPath": "$VENV_DIR/bin/python",
    "python.analysis.extraPaths": [
        "$HOME_DIR/$PRIMARY_REPO_DIR/chapter0_fundamentals/exercises",
        "$HOME_DIR/$PRIMARY_REPO_DIR/chapter1_transformer_interp/exercises",
        "$HOME_DIR/$PRIMARY_REPO_DIR/chapter2_rl/exercises",
        "$HOME_DIR/$PRIMARY_REPO_DIR/chapter3_llm_evals/exercises",
        "$HOME_DIR/$PRIMARY_REPO_DIR/chapter4_alignment_science/exercises"
    ]
}
EOF

echo "=== Done! Activate with: source $VENV_DIR/bin/activate ==="
