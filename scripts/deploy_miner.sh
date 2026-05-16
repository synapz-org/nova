#!/usr/bin/env bash
# Deploy elite miner to mainnet (or testnet) on a Basilica box.
#
# Prerequisites (on your local machine, not the box):
#   1. Wallet exists with `btcli wallet list` showing `<coldkey>/<hotkey>`
#   2. Hotkey is registered on netuid 68 (mainnet) or 379 (testnet) — see ./scripts/register_hotkey.sh
#   3. Surrogate model trained: models/surrogate_{TARGET}/ for the current target
#   4. GitHub repo configured: env vars below are set
#
# Usage:
#   ./scripts/deploy_miner.sh \
#       --rental-id <basilica-rental-uuid> \
#       --wallet <coldkey-name> \
#       --hotkey <hotkey-name> \
#       --netuid 68 \
#       --network finney \
#       --mol-surrogate models/surrogate_Q6P6W3 \
#       --nb-surrogate models/nb_surrogate_Q9NZQ7

set -euo pipefail

RENTAL_ID=""
WALLET=""
HOTKEY=""
NETUID=68
NETWORK="finney"
MOL_SURROGATE=""
NB_SURROGATE=""
USE_INFERENCE=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rental-id) RENTAL_ID="$2"; shift 2 ;;
        --wallet) WALLET="$2"; shift 2 ;;
        --hotkey) HOTKEY="$2"; shift 2 ;;
        --netuid) NETUID="$2"; shift 2 ;;
        --network) NETWORK="$2"; shift 2 ;;
        --mol-surrogate) MOL_SURROGATE="$2"; shift 2 ;;
        --nb-surrogate) NB_SURROGATE="$2"; shift 2 ;;
        --no-inference) USE_INFERENCE=0; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$RENTAL_ID" || -z "$WALLET" || -z "$HOTKEY" ]]; then
    echo "Required: --rental-id, --wallet, --hotkey" >&2
    echo "See header of this file for full usage." >&2
    exit 1
fi

# --- Env var checks ---
echo "=== Checking required env vars locally ==="
: "${GITHUB_TOKEN:?GITHUB_TOKEN must be set in your env (used by miner to upload submissions)}"
: "${GITHUB_REPO_OWNER:?GITHUB_REPO_OWNER must be set (e.g. synapz-org)}"
: "${GITHUB_REPO_NAME:?GITHUB_REPO_NAME must be set (e.g. nova-submissions)}"
: "${GITHUB_REPO_BRANCH:?GITHUB_REPO_BRANCH must be set (e.g. main)}"
GITHUB_REPO_PATH="${GITHUB_REPO_PATH:-}"
echo "  GITHUB_REPO_OWNER=$GITHUB_REPO_OWNER"
echo "  GITHUB_REPO_NAME=$GITHUB_REPO_NAME"
echo "  GITHUB_REPO_BRANCH=$GITHUB_REPO_BRANCH"
echo "  GITHUB_REPO_PATH=${GITHUB_REPO_PATH:-<empty>}"

# --- Verify rental is alive ---
echo "=== Checking rental status ==="
basilica status "$RENTAL_ID" | grep -q "Status: running" || {
    echo "Rental $RENTAL_ID is not running. Aborting." >&2
    exit 1
}
echo "Rental running."

# --- Push wallet to box ---
WALLET_DIR="$HOME/.bittensor/wallets/$WALLET"
if [[ ! -d "$WALLET_DIR" ]]; then
    echo "Wallet directory $WALLET_DIR not found. Aborting." >&2
    exit 1
fi
if [[ ! -f "$WALLET_DIR/hotkeys/$HOTKEY" ]]; then
    echo "Hotkey $HOTKEY not found under $WALLET_DIR/hotkeys. Aborting." >&2
    exit 1
fi
echo "=== Copying wallet to rental ==="
basilica exec "mkdir -p /home/ubuntu/.bittensor/wallets/$WALLET/hotkeys" --target "$RENTAL_ID"
basilica cp "$WALLET_DIR/coldkeypub.txt" "$RENTAL_ID:/home/ubuntu/.bittensor/wallets/$WALLET/coldkeypub.txt"
basilica cp "$WALLET_DIR/hotkeys/$HOTKEY" "$RENTAL_ID:/home/ubuntu/.bittensor/wallets/$WALLET/hotkeys/$HOTKEY"
echo "Wallet pushed. (coldkey itself NOT pushed — only coldkeypub + hotkey are needed for miner)"

# --- Push surrogate models if provided ---
if [[ -n "$MOL_SURROGATE" && -d "$MOL_SURROGATE" ]]; then
    echo "=== Copying molecule surrogate $MOL_SURROGATE ==="
    basilica exec "mkdir -p /home/ubuntu/nova/$MOL_SURROGATE" --target "$RENTAL_ID"
    basilica cp "$MOL_SURROGATE/model.txt" "$RENTAL_ID:/home/ubuntu/nova/$MOL_SURROGATE/model.txt"
    basilica cp "$MOL_SURROGATE/holdout_metrics.json" "$RENTAL_ID:/home/ubuntu/nova/$MOL_SURROGATE/holdout_metrics.json"
    basilica cp "$MOL_SURROGATE/feature_version.json" "$RENTAL_ID:/home/ubuntu/nova/$MOL_SURROGATE/feature_version.json"
fi
if [[ -n "$NB_SURROGATE" && -d "$NB_SURROGATE" ]]; then
    echo "=== Copying nanobody surrogate $NB_SURROGATE ==="
    basilica exec "mkdir -p /home/ubuntu/nova/$NB_SURROGATE" --target "$RENTAL_ID"
    basilica cp "$NB_SURROGATE/model.txt" "$RENTAL_ID:/home/ubuntu/nova/$NB_SURROGATE/model.txt"
    basilica cp "$NB_SURROGATE/holdout_metrics.json" "$RENTAL_ID:/home/ubuntu/nova/$NB_SURROGATE/holdout_metrics.json"
    basilica cp "$NB_SURROGATE/feature_version.json" "$RENTAL_ID:/home/ubuntu/nova/$NB_SURROGATE/feature_version.json"
fi

# --- Push .env to box ---
echo "=== Pushing GitHub env to box ==="
basilica exec "cat > /home/ubuntu/nova/.env <<EOF
GITHUB_TOKEN=$GITHUB_TOKEN
GITHUB_REPO_OWNER=$GITHUB_REPO_OWNER
GITHUB_REPO_NAME=$GITHUB_REPO_NAME
GITHUB_REPO_BRANCH=$GITHUB_REPO_BRANCH
GITHUB_REPO_PATH=$GITHUB_REPO_PATH
SUBTENSOR_NETWORK=$NETWORK
EOF
chmod 600 /home/ubuntu/nova/.env" --target "$RENTAL_ID"

# --- Build the REAL timelock package (the stub doesn't work for actual encryption) ---
echo "=== Building real timelock on box (one-time, ~3 min) ==="
basilica exec "cd /home/ubuntu/nova && \
    if ! python -c 'import timelock; t = timelock.Timelock; t()' 2>/dev/null; then \
        echo 'Stub or missing timelock detected, building from external_tools/timelock...' && \
        rm -f .venv/lib/python3.10/site-packages/timelock.py && \
        if [ ! -d external_tools/timelock ]; then \
            cd external_tools && git clone https://github.com/ideal-lab5/timelock.git && \
            cd timelock && git checkout 23fe963f17175e413b7434180d2d0d0776722f1f && cd ../.. ; \
        fi && \
        source .venv/bin/activate && \
        pip install --quiet maturin==1.8.3 patchelf && \
        if ! command -v cargo > /dev/null; then \
            wget -qO- https://sh.rustup.rs | sh -s -- -y && source \$HOME/.cargo/env ; \
        fi && \
        source \$HOME/.cargo/env 2>/dev/null || true && \
        export PYO3_CROSS_PYTHON_VERSION=3.10 && \
        cd external_tools/timelock/wasm && ./wasm_build_py.sh && cd ../../.. && \
        cd external_tools/timelock/py && pip install --upgrade build && python3 -m build && pip install timelock && \
        cd ../../.. ; \
    fi && \
    python -c 'import timelock; t = timelock.Timelock()' && echo 'Real timelock import OK'" --target "$RENTAL_ID"

# --- Verify wallet is registered on the subnet ---
echo "=== Verifying hotkey is registered on netuid $NETUID ==="
basilica exec "cd /home/ubuntu/nova && source .venv/bin/activate && \
    python -c \"
import bittensor as bt
sub = bt.subtensor(network='$NETWORK')
mg = sub.metagraph(netuid=$NETUID)
wallet = bt.wallet(name='$WALLET', hotkey='$HOTKEY')
ss58 = wallet.hotkey.ss58_address
if ss58 in mg.hotkeys:
    uid = mg.hotkeys.index(ss58)
    print(f'Hotkey {ss58[:10]}... registered as UID {uid}')
else:
    print(f'NOT REGISTERED: {ss58}')
    import sys; sys.exit(1)
\"" --target "$RENTAL_ID"

# --- Build the launch command ---
INFERENCE_FLAG=""
[[ "$USE_INFERENCE" -eq 1 ]] && INFERENCE_FLAG="--use_inference"

MOL_SURROGATE_FLAG=""
[[ -n "$MOL_SURROGATE" ]] && MOL_SURROGATE_FLAG="--surrogate_model_dir $MOL_SURROGATE --surrogate_min_spearman 0.5"

NB_SURROGATE_FLAG=""
[[ -n "$NB_SURROGATE" ]] && NB_SURROGATE_FLAG="--nb_surrogate_model_dir $NB_SURROGATE --nb_surrogate_min_spearman 0.5"

LAUNCH_CMD="cd /home/ubuntu/nova && source .venv/bin/activate && nohup python -m elite_miner.run \
    --wallet.name $WALLET \
    --wallet.hotkey $HOTKEY \
    --network $NETWORK \
    --netuid $NETUID \
    $MOL_SURROGATE_FLAG \
    $NB_SURROGATE_FLAG \
    $INFERENCE_FLAG \
    > /home/ubuntu/miner.log 2>&1 & echo \$! > /home/ubuntu/miner.pid"

echo ""
echo "=== Ready to launch ==="
echo "Command:"
echo "  $LAUNCH_CMD"
echo ""
echo "To launch:"
echo "  basilica exec \"$LAUNCH_CMD\" --target $RENTAL_ID"
echo ""
echo "To monitor:"
echo "  basilica exec \"tail -f /home/ubuntu/miner.log\" --target $RENTAL_ID"
echo "  basilica exec \"cat /home/ubuntu/miner.pid && ps -p \\\$(cat /home/ubuntu/miner.pid)\" --target $RENTAL_ID"
echo ""
echo "To stop:"
echo "  basilica exec \"kill \\\$(cat /home/ubuntu/miner.pid)\" --target $RENTAL_ID"
echo ""
echo "Run this script with --launch to also start the miner immediately (TODO)."
