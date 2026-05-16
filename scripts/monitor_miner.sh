#!/usr/bin/env bash
# Quick health check + recent submission log for a running miner.
#
# Usage:
#   ./scripts/monitor_miner.sh --rental-id <basilica-uuid>

set -euo pipefail

RENTAL_ID=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rental-id) RENTAL_ID="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$RENTAL_ID" ]]; then
    echo "Required: --rental-id" >&2; exit 1
fi

echo "=== Rental status ==="
basilica status "$RENTAL_ID" | head -8

echo ""
echo "=== Miner process ==="
basilica exec "
if [ -f /home/ubuntu/miner.pid ]; then
    PID=\$(cat /home/ubuntu/miner.pid)
    if ps -p \$PID > /dev/null 2>&1; then
        echo 'Miner running (PID '\$PID')'
        ps -o etime= -p \$PID | awk '{print \"  uptime:\", \$1}'
    else
        echo 'Miner PID \$PID is not running'
    fi
else
    echo 'No miner.pid file'
fi
" --target "$RENTAL_ID" 2>&1 | tail -5

echo ""
echo "=== Recent submissions ==="
basilica exec "grep -E 'chain commit successful|epoch:.+complete|fallback submission' /home/ubuntu/miner.log 2>/dev/null | tail -10" --target "$RENTAL_ID" 2>&1 | tail -15

echo ""
echo "=== Errors in last 100 log lines ==="
basilica exec "grep -E 'ERROR|FAIL|exception' /home/ubuntu/miner.log 2>/dev/null | tail -5" --target "$RENTAL_ID" 2>&1 | tail -10

echo ""
echo "=== GPU utilization (right now) ==="
basilica exec "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader" --target "$RENTAL_ID" 2>&1 | tail -3

echo ""
echo "=== Streamed labels (free training data this run) ==="
basilica exec "ls -la /home/ubuntu/nova/cache/labels/ /home/ubuntu/nova/cache/nb_labels/ 2>/dev/null" --target "$RENTAL_ID" 2>&1 | tail -10
