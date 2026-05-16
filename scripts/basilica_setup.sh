#!/usr/bin/env bash
# Basilica environment setup for elite miner Phase 2 deployment.
#
# Runs on the rented A100 80GB box. Idempotent — safe to re-run.
# Expects the box to have Ubuntu/Debian + CUDA toolchain pre-installed (Basilica default).
#
# Usage (via `basilica exec`):
#   basilica exec "$(cat scripts/basilica_setup.sh)" --target <rental-id>
# Or after `basilica ssh`:
#   bash basilica_setup.sh

set -euo pipefail

REPO_URL="https://github.com/synapz-org/nova.git"
REPO_DIR="${HOME}/nova"
PY_VERSION="3.10"  # timelock package only ships cp310 wheels

echo "=== System info ==="
uname -a
nvidia-smi | head -10 || echo "(nvidia-smi not available)"

echo "=== Apt prerequisites ==="
sudo apt-get update -qq
sudo apt-get install -y -qq git python3.10 python3.10-venv python3.10-dev build-essential

echo "=== Clone repo ==="
if [ -d "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR"
    git fetch --all --quiet
    git checkout competitive-miner
    git pull --quiet
else
    git clone -b competitive-miner "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

echo "=== Init submodule ==="
git submodule update --init --recursive

echo "=== Python venv ==="
if [ ! -d ".venv" ]; then
    python3.10 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

echo "=== Install repo requirements (this is the slow part) ==="
pip install -r requirements/requirements.txt

echo "=== Install Phase 2 extras ==="
pip install lightgbm pyarrow

echo "=== Install boltz (vendored) ==="
pip install -e ./external_tools/boltz

echo "=== Install boltzgen (vendored) ==="
pip install -e ./external_tools/boltzgen

echo "=== Install torch 2.7.1 cu126 (boltz needs this exact version, will pull torch 2.12 otherwise) ==="
pip install --quiet 'torch==2.7.1' 'torchvision==0.22.1' 'torchaudio==2.7.1' \
    --index-url https://download.pytorch.org/whl/cu126

echo "=== Stub timelock (label collection doesn't need real timelock encryption) ==="
SITE_PKG="$(python -c 'import site; print(site.getsitepackages()[0])')"
cat > "${SITE_PKG}/timelock.py" <<'PYEOF'
# Stub for label collection. Real mining (which calls drand encrypt) requires
# the maturin-built timelock from external_tools/timelock/. Build via:
#   cd external_tools/timelock && uv pip install --upgrade build && python3 -m build && uv pip install timelock
class Timelock:
    def __init__(self, *a, **kw): pass
    def encrypt(self, *a, **kw): raise NotImplementedError("timelock stub")
    def decrypt(self, *a, **kw): raise NotImplementedError("timelock stub")
PYEOF

echo "=== Download combinatorial DB ==="
if [ ! -s combinatorial_db/molecules.sqlite ]; then
    wget -q 'https://huggingface.co/datasets/Metanova/Mol-Rxn-DB/resolve/main/molecules.sqlite?download=true' \
        -O combinatorial_db/molecules.sqlite
fi

echo "=== Sanity check ==="
python -c "
import sys
sys.path.insert(0, '.')
import torch
print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')
from elite_miner.molecule import CombinatorialSearcher, ValidityFilter
print('elite_miner imports ok')
from external_tools.boltz.boltz_wrapper import BoltzWrapper
print('boltz_wrapper imports ok')
"

echo "=== Setup complete ==="
echo "To collect labels:"
echo "  cd ${REPO_DIR} && source .venv/bin/activate"
echo "  python -m elite_miner.scripts.collect_labels --target Q6P6W3 --rxn-id 1 --n-candidates 2000 --output cache/labels/Q6P6W3.parquet"
