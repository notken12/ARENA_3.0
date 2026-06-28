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
PRIMARY_REPO_DIR="ARENA_3.0"
REPO_PATH="$HOME/$PRIMARY_REPO_DIR"
# Put the venv at <repo>/.venv: VS Code auto-discovers, auto-selects, and auto-activates
# a `.venv` in the workspace root, so the kernel shows up with no manual interpreter path.
VENV_DIR="$REPO_PATH/.venv"
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
# Fail fast on the wrong Python: ARENA's pinned RL stack (gymnasium[atari]==0.29.0 ->
# ale-py 0.8.x) only has wheels up to cp311, so 3.12+ images can't resolve. Use a py3.11
# PyTorch base image (whose torch is also built for cp311, so --system-site-packages works).
echo "=== Verifying base Python version ==="
python - <<'PY' || { echo "ERROR: ARENA needs Python 3.11. Recreate the pod from a py3.11 PyTorch base image."; exit 1; }
import sys
assert sys.version_info[:2] == (3, 11), f"found {sys.version.split()[0]}, need 3.11"
print("Base Python", sys.version.split()[0])
PY

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
# Must live in the OPENED workspace folder's .vscode/, not $HOME/.vscode (which VS Code ignores).
# Setting the interpreter to the venv makes VS Code surface it as a kernel automatically.
echo "=== Configuring VS Code workspace settings ==="

mkdir -p "$REPO_PATH/.vscode"
cat > "$REPO_PATH/.vscode/settings.json" << EOF
{
    "python.defaultInterpreterPath": "$VENV_DIR/bin/python",
    "python.analysis.extraPaths": [
        "$REPO_PATH/chapter0_fundamentals/exercises",
        "$REPO_PATH/chapter1_transformer_interp/exercises",
        "$REPO_PATH/chapter2_rl/exercises",
        "$REPO_PATH/chapter3_llm_evals/exercises",
        "$REPO_PATH/chapter4_alignment_science/exercises"
    ]
}
EOF

echo "=== Done! Activate with: source $VENV_DIR/bin/activate ==="
