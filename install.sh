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
# Creates an isolated conda env on Python 3.11 (required by ARENA's pinned RL stack,
# gymnasium[atari]==0.29.0 -> ale-py 0.8.x, which has no cp312 wheels) and installs its
# own CUDA torch from requirements.txt. Independent of the base image's Python/torch, so
# it works on any GPU image. uv is used for fast, parallel dependency installs.
# =============================================================================

# Defaults
PLATFORM="runpod"
ENV_NAME="arena-env"
PYTHON_VERSION="3.11"
PRIMARY_REPO_DIR="ARENA_3.0"
REPO_PATH="$HOME/$PRIMARY_REPO_DIR"
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

# --- Ensure conda is available (reuse the image's, else install Miniconda) ---
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    echo "=== Using existing conda at $CONDA_BASE ==="
else
    echo "=== Installing Miniconda ==="
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    CONDA_BASE="$HOME/miniconda3"
    "$CONDA_BASE/bin/conda" init bash
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

# --- Accept conda TOS (required before creating an env from the default channels) ---
echo "=== Accepting Conda TOS ==="
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# --- Create and activate the env ---
echo "=== Creating conda env '$ENV_NAME' (python $PYTHON_VERSION) ==="
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
conda activate "$ENV_NAME"
ENV_PYTHON="$CONDA_PREFIX/bin/python"
echo "=== Active Python: $(which python) ==="

# --- Install Python deps from primary repo with uv (parallel, fast) ---
# requirements.txt installs its own CUDA torch from the pytorch index.
# --index-strategy unsafe-best-match: pick the best version across PyPI + the pytorch
# index, instead of locking a package to the first index that lists it (the pytorch index
# ships a stale `requests` that otherwise makes the resolve unsatisfiable).
echo "=== Installing Python dependencies from $PRIMARY_REPO_DIR ==="
cd "$PRIMARY_REPO_DIR"
pip install -U pip uv
uv pip install --index-strategy unsafe-best-match -r requirements.txt
cd ..

# --- Verify torch sees the GPU ---
echo "=== Verifying torch + CUDA ==="
python - <<'PY' || { echo "ERROR: torch cannot see a GPU (CPU-only build, or no GPU on this pod)."; exit 1; }
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
print("CUDA OK:", torch.cuda.get_device_name(0))
PY

# --- Register Jupyter kernel for this env ---
echo "=== Registering Jupyter kernel '$ENV_NAME' ==="
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

# --- VS Code workspace settings ---
# Must live in the OPENED workspace folder's .vscode/, not $HOME/.vscode (which VS Code
# ignores). Pointing at the conda env's python makes VS Code surface it as a kernel.
echo "=== Configuring VS Code workspace settings ==="
mkdir -p "$REPO_PATH/.vscode"
cat > "$REPO_PATH/.vscode/settings.json" << EOF
{
    "python.defaultInterpreterPath": "$ENV_PYTHON",
    "python.analysis.extraPaths": [
        "$REPO_PATH/chapter0_fundamentals/exercises",
        "$REPO_PATH/chapter1_transformer_interp/exercises",
        "$REPO_PATH/chapter2_rl/exercises",
        "$REPO_PATH/chapter3_llm_evals/exercises",
        "$REPO_PATH/chapter4_alignment_science/exercises"
    ]
}
EOF

echo "=== Done! Activate with: conda activate $ENV_NAME ==="
