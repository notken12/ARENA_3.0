# #!/bin/bash
# set -e

# =============================================================================
# First clone the ARENA_3.0 repo using:
#   git clone -b alignment-science https://github.com/callummcdougall/ARENA_3.0.git
# Then, usage:
#   bash ARENA_3.0/install.sh                        # RunPod (default), with llm-context repo
#   bash ARENA_3.0/install.sh --platform vastai      # Vast.ai platform
#   bash ARENA_3.0/install.sh --no-llm-context       # Skip cloning arena-llm-context
# =============================================================================

# Defaults
CONDA_ENV="arena-env"
PYTHON_VERSION="3.11"
CLONE_LLM_CONTEXT=true
PLATFORM=""

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
  --platform)
    PLATFORM="$2"
    shift 2
    ;;
  --no-llm-context)
    CLONE_LLM_CONTEXT=false
    shift
    ;;
  *)
    echo "Unknown option: $1"
    exit 1
    ;;
  esac
done

# Auto-detect platform if not specified
if [[ -z "$PLATFORM" ]]; then
  if [[ -d "/workspace" ]]; then
    PLATFORM="runpod"
  else
    PLATFORM="local"
  fi
fi

echo "=== Setup: platform=$PLATFORM, clone_llm_context=$CLONE_LLM_CONTEXT ==="

# --- Set Miniconda path based on platform ---
if [[ "$PLATFORM" == "runpod" ]]; then
  MINICONDA_DIR="/workspace/miniconda3"
else
  MINICONDA_DIR="$HOME/miniconda3"
fi

# --- Install Miniconda (skip if already present) ---
if [[ -d "$MINICONDA_DIR/bin" && -f "$MINICONDA_DIR/bin/conda" ]]; then
  echo "=== Miniconda already present at $MINICONDA_DIR, skipping install ==="
else
  echo "=== Installing Miniconda to $MINICONDA_DIR ==="
  mkdir -p "$MINICONDA_DIR"
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$MINICONDA_DIR/miniconda.sh"
  bash "$MINICONDA_DIR/miniconda.sh" -b -u -p "$MINICONDA_DIR"
  rm -f "$MINICONDA_DIR/miniconda.sh"
  "$MINICONDA_DIR/bin/conda" init bash
fi

# Source conda.sh to get conda activate working in this script
source "$MINICONDA_DIR/etc/profile.d/conda.sh"

# --- Accept conda TOS ---
echo "=== Accepting Conda TOS ==="
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# --- Create and activate conda env ---
echo "=== Creating conda env '$CONDA_ENV' (python $PYTHON_VERSION) ==="
conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y
conda activate "$CONDA_ENV"
echo "=== Active Python: $(which python) ==="

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

# # --- Git config ---
# git config --global user.name callummcdougall
# git config --global user.email cal.s.mcdougall@gmail.com

# --- Install Python deps from primary repo ---
PRIMARY_REPO_DIR="ARENA_3.0"
echo "=== Installing Python dependencies from $PRIMARY_REPO_DIR ==="
cd "$PRIMARY_REPO_DIR"
pip install -U pip setuptools wheel
pip install -r requirements.txt
conda install -n "$CONDA_ENV" ipykernel --update-deps --force-reinstall -y
cd ..

# --- VS Code workspace settings ---
echo "=== Configuring VS Code workspace settings ==="

HOME_DIR="$HOME"
mkdir -p "$HOME_DIR/.vscode"
cat >"$HOME_DIR/.vscode/settings.json" <<EOF
{
    "python.defaultInterpreterPath": "$MINICONDA_DIR/envs/$CONDA_ENV/bin/python",
    "python.analysis.extraPaths": [
        "$HOME_DIR/$PRIMARY_REPO_DIR/chapter0_fundamentals/exercises",
        "$HOME_DIR/$PRIMARY_REPO_DIR/chapter1_transformer_interp/exercises",
        "$HOME_DIR/$PRIMARY_REPO_DIR/chapter2_rl/exercises",
        "$HOME_DIR/$PRIMARY_REPO_DIR/chapter3_llm_evals/exercises",
        "$HOME_DIR/$PRIMARY_REPO_DIR/chapter4_alignment_science/exercises"
    ]
}
EOF

echo "=== Done! ==="
