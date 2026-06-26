#!/bin/bash
# =============================================================================
# macOS (Apple Silicon / Intel) setup for ARENA_3.0.
# Equivalent of install.sh, adapted for Darwin:
#   - downloads the macOS Miniconda installer for the host arch (not Linux)
#   - uses curl instead of wget
#   - skips apt (git/curl ship with macOS / Xcode CLT)
#   - initialises zsh (the macOS default shell) instead of bash
#
# Usage:
#   bash ARENA_3.0/install_mac.sh                 # core install
#   bash ARENA_3.0/install_mac.sh --llm-context   # also clone arena-llm-context
# =============================================================================

set -e

CONDA_ENV="arena-env"
PYTHON_VERSION="3.11"
CLONE_LLM_CONTEXT=false
PRIMARY_REPO_DIR="ARENA_3.0"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --llm-context) CLONE_LLM_CONTEXT=true; shift ;;
        --no-llm-context) CLONE_LLM_CONTEXT=false; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Pick the right Miniconda installer for this Mac's architecture
ARCH="$(uname -m)"
if [[ "$ARCH" == "arm64" ]]; then
    MINICONDA_FILE="Miniconda3-latest-MacOSX-arm64.sh"
elif [[ "$ARCH" == "x86_64" ]]; then
    MINICONDA_FILE="Miniconda3-latest-MacOSX-x86_64.sh"
else
    echo "Unsupported macOS arch: $ARCH"; exit 1
fi

echo "=== Setup (macOS): arch=$ARCH, clone_llm_context=$CLONE_LLM_CONTEXT ==="

# --- Install Miniconda ---
echo "=== Installing Miniconda ($MINICONDA_FILE) ==="
# Remove any broken/partial prior install (e.g. a Linux installer that failed here)
rm -rf ~/miniconda3
mkdir -p ~/miniconda3
curl -fsSL "https://repo.anaconda.com/miniconda/$MINICONDA_FILE" -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init zsh

# Source conda.sh so `conda activate` works inside this script
source ~/miniconda3/etc/profile.d/conda.sh

# --- Accept conda TOS ---
echo "=== Accepting Conda TOS ==="
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# --- Create and activate conda env ---
echo "=== Creating conda env '$CONDA_ENV' (python $PYTHON_VERSION) ==="
conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y
conda activate "$CONDA_ENV"
echo "=== Active Python: $(which python) ==="

# Maybe clone the extra LLM-context repo
if $CLONE_LLM_CONTEXT; then
    REPO="callummcdougall/arena-llm-context"
    echo "=== Cloning $REPO ==="
    git clone -b main "https://github.com/${REPO}.git"
fi

# --- Install Python deps ---
echo "=== Installing Python dependencies from $PRIMARY_REPO_DIR ==="
cd "$PRIMARY_REPO_DIR"
pip install -U pip setuptools wheel

# Try the full requirements first. On Apple Silicon a few CUDA/Linux-oriented
# packages (bitsandbytes, gymnasium[mujoco-py]) often fail to build; we don't
# want that to abort the whole env, so fall back to installing the rest and
# report what was skipped.
if ! pip install -r requirements.txt; then
    echo ""
    echo "!!! Full requirements install failed (expected on Mac for some CUDA/Linux pkgs)."
    echo "!!! Retrying without the known Mac-problematic packages..."
    grep -viE '^(bitsandbytes|gymnasium)' requirements.txt > /tmp/requirements_mac.txt
    pip install -r /tmp/requirements_mac.txt
    echo ""
    echo "!!! Skipped on Mac: bitsandbytes, gymnasium[mujoco-py]."
    echo "!!! These are only needed for specific GPU/RL exercises; install manually if required."
fi

conda install -n "$CONDA_ENV" ipykernel --update-deps --force-reinstall -y
cd ..

# --- VS Code workspace settings ---
echo "=== Configuring VS Code workspace settings ==="
HOME_DIR="$HOME"
mkdir -p "$HOME_DIR/.vscode"
cat > "$HOME_DIR/.vscode/settings.json" << EOF
{
    "python.defaultInterpreterPath": "$HOME_DIR/miniconda3/envs/$CONDA_ENV/bin/python",
    "python.analysis.extraPaths": [
        "$HOME_DIR/$PRIMARY_REPO_DIR/chapter0_fundamentals/exercises",
        "$HOME_DIR/$PRIMARY_REPO_DIR/chapter1_transformer_interp/exercises",
        "$HOME_DIR/$PRIMARY_REPO_DIR/chapter2_rl/exercises",
        "$HOME_DIR/$PRIMARY_REPO_DIR/chapter3_llm_evals/exercises",
        "$HOME_DIR/$PRIMARY_REPO_DIR/chapter4_alignment_science/exercises"
    ]
}
EOF

echo "=== Done! Run 'conda activate $CONDA_ENV' in a new terminal. ==="
