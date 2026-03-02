#!/usr/bin/env bash
# cipher/start.sh  —  launch pipeline + mitmproxy in the right order
#
# Usage:
#   bash start.sh
#
# Assumes:
#   • Android emulator already booted and proxy configured (run android_setup.sh first)
#   • Twitter/X app already installed and logged-in in the emulator
#   • python3, mitmdump available in PATH

set -euo pipefail

MITM_PORT=8080
CIPHER_DIR="$(cd "$(dirname "$0")" && pwd)"

G="\033[0;32m"; Y="\033[0;33m"; N="\033[0m"
ok()  { echo -e "${G}[+]${N} $*"; }
inf() { echo -e "${Y}[*]${N} $*"; }

# ── cleanup on exit ───────────────────────────────────────────────────────────
PIDS=()
cleanup() {
    inf "Shutting down …"
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    ok "Done"
}
trap cleanup EXIT INT TERM

# ── 1. pipeline server ────────────────────────────────────────────────────────
inf "Starting pipeline server …"
python3 "${CIPHER_DIR}/pipeline.py" &
PIPELINE_PID=$!
PIDS+=("$PIPELINE_PID")
ok "pipeline.py  PID=$PIPELINE_PID"

# Give it a moment to bind the port before mitmproxy tries to connect
sleep 1

# ── 2. mitmproxy interceptor ──────────────────────────────────────────────────
inf "Starting mitmproxy on port ${MITM_PORT} …"
mitmdump \
    -s "${CIPHER_DIR}/interceptor.py" \
    --mode regular \
    --listen-port "$MITM_PORT" \
    --ssl-insecure \
    --quiet &
MITM_PID=$!
PIDS+=("$MITM_PID")
ok "mitmproxy     PID=$MITM_PID"

echo ""
ok "System running.  Open Twitter/X in the emulator to begin interception."
echo ""
echo "  Launch Twitter in emulator:"
echo "    adb shell monkey -p com.twitter.android -c android.intent.category.LAUNCHER 1"
echo ""
echo "  Watch signals:"
echo "    tail -f /tmp/cipher_pipeline.log   (if you redirect pipeline output there)"
echo ""
echo "  Stop everything:  Ctrl-C"
echo ""

# ── wait for either process to exit ──────────────────────────────────────────
wait -n "${PIDS[@]}" 2>/dev/null || true
inf "A process exited — shutting down everything"
