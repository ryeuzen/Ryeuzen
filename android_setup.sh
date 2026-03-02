#!/usr/bin/env bash
# cipher/android_setup.sh
#
# One-shot setup script to prepare the Android emulator for Twitter interception.
#
# Usage:
#   bash android_setup.sh [path/to/twitter.apk]
#
# Prerequisites (already installed per project spec):
#   • Android emulator + AVD  cipher_pixel6  (Android 13)
#   • adb  in PATH
#   • mitmproxy  in PATH
#   • python3 with cryptography package
#
# What this script does:
#   1.  Start  cipher_pixel6  emulator
#   2.  Wait for full boot
#   3.  Set emulator HTTP(S) proxy → mitmproxy
#   4.  Start mitmproxy briefly to generate the CA cert
#   5.  Install the mitmproxy CA cert as a system cert (Android 13 ignores user certs)
#   6.  (Optional) sideload Twitter APK if provided
#   7.  Print next-step instructions

set -euo pipefail

# ── config ────────────────────────────────────────────────────────────────────
AVD="cipher_pixel6"
MITM_PORT=8080
APK_PATH="${1:-}"
MITMPROXY_CERT_DIR="${HOME}/.mitmproxy"
MITMPROXY_CERT_PEM="${MITMPROXY_CERT_DIR}/mitmproxy-ca-cert.pem"

# ── colours ───────────────────────────────────────────────────────────────────
G="\033[0;32m"; Y="\033[0;33m"; R="\033[0;31m"; N="\033[0m"
ok()  { echo -e "${G}[+]${N} $*"; }
inf() { echo -e "${Y}[*]${N} $*"; }
err() { echo -e "${R}[!]${N} $*" >&2; }

# ── 1. start emulator ─────────────────────────────────────────────────────────
inf "Starting emulator $AVD …"
nohup emulator \
    -avd "$AVD" \
    -no-audio \
    -no-window \
    -gpu swiftshader_indirect \
    -writable-system \
    >/tmp/emulator.log 2>&1 &
EMULATOR_PID=$!
ok "Emulator PID $EMULATOR_PID (log → /tmp/emulator.log)"

# ── 2. wait for boot ──────────────────────────────────────────────────────────
inf "Waiting for boot_completed …"
adb wait-for-device

BOOT_DONE=0
for i in $(seq 1 90); do
    STATUS=$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')
    if [[ "$STATUS" == "1" ]]; then
        BOOT_DONE=1
        break
    fi
    sleep 2
done

if [[ $BOOT_DONE -eq 0 ]]; then
    err "Emulator did not finish booting in 180 s — check /tmp/emulator.log"
    exit 1
fi
ok "Emulator booted"

# ── 3. set proxy ──────────────────────────────────────────────────────────────
inf "Configuring proxy → 127.0.0.1:${MITM_PORT}"
# The emulator sees the host as 10.0.2.2
adb shell settings put global http_proxy  "10.0.2.2:${MITM_PORT}"
# For apps that bypass the global proxy, also set WiFi proxy on the default AP
adb shell am broadcast \
    -a android.net.ConnectivityManager.CONNECTIVITY_ACTION \
    >/dev/null 2>&1 || true
ok "Proxy configured"

# ── 4. generate mitmproxy CA cert ─────────────────────────────────────────────
if [[ ! -f "$MITMPROXY_CERT_PEM" ]]; then
    inf "Generating mitmproxy CA cert (quick start then kill) …"
    mitmdump --port "$MITM_PORT" &
    MITM_TEMP_PID=$!
    sleep 3
    kill "$MITM_TEMP_PID" 2>/dev/null || true
    wait "$MITM_TEMP_PID" 2>/dev/null || true
fi

if [[ ! -f "$MITMPROXY_CERT_PEM" ]]; then
    err "mitmproxy CA cert not found at $MITMPROXY_CERT_PEM"
    exit 1
fi
ok "mitmproxy CA cert ready"

# ── 5. install cert as system CA ─────────────────────────────────────────────
inf "Computing cert filename (OpenSSL subject hash) …"
CERT_HASH=$(python3 - <<'PYEOF'
import hashlib, struct, sys
from pathlib import Path
from cryptography import x509
from cryptography.hazmat.backends import default_backend

pem = Path("${MITMPROXY_CERT_PEM}".replace("${MITMPROXY_CERT_PEM}",
    __import__("os").path.expanduser("~/.mitmproxy/mitmproxy-ca-cert.pem")
)).read_bytes()

cert = x509.load_pem_x509_certificate(pem, default_backend())
# OpenSSL subject_hash_old algorithm: MD5 of canonical DER subject, little-endian 4 bytes
subject_der = cert.subject.public_bytes(default_backend())
digest = hashlib.md5(subject_der).digest()[:4]
h = struct.unpack('<I', digest)[0]
print(format(h, '08x'))
PYEOF
)

CERT_NAME="${CERT_HASH}.0"
DEVICE_TMP="/data/local/tmp/${CERT_NAME}"
SYSTEM_CERTS="/system/etc/security/cacerts"

inf "Pushing cert to device as ${CERT_NAME} …"
adb push "$MITMPROXY_CERT_PEM" "$DEVICE_TMP"

inf "Installing cert to system store (requires root in AVD) …"
# Remount /system read-write; works in Google API (non-Play) AVD images
adb shell "su 0 mount -o remount,rw /system" 2>/dev/null || \
    adb shell "su 0 mount -o rw,remount /system"
adb shell "su 0 cp ${DEVICE_TMP} ${SYSTEM_CERTS}/${CERT_NAME}"
adb shell "su 0 chmod 644 ${SYSTEM_CERTS}/${CERT_NAME}"
adb shell "su 0 mount -o remount,ro /system" 2>/dev/null || true

ok "System CA cert installed: ${CERT_NAME}"

# ── 6. install Twitter APK (optional) ────────────────────────────────────────
if [[ -n "$APK_PATH" && -f "$APK_PATH" ]]; then
    inf "Installing Twitter APK: $APK_PATH"
    adb install -r "$APK_PATH"
    ok "Twitter APK installed"
else
    inf "No APK supplied — install Twitter/X via Play Store in the emulator"
    inf "  (Pass path as argument:  bash android_setup.sh /path/to/twitter.apk)"
fi

# ── 7. instructions ───────────────────────────────────────────────────────────
echo ""
ok "Android setup complete. Next steps:"
echo ""
echo "  1. Start the pipeline server (terminal 1):"
echo "       python3 pipeline.py"
echo ""
echo "  2. Start mitmproxy with the interceptor (terminal 2):"
echo "       mitmdump -s interceptor.py --mode regular -p ${MITM_PORT} --ssl-insecure"
echo ""
echo "  3. Open Twitter/X in the emulator:"
echo "       adb shell monkey -p com.twitter.android -c android.intent.category.LAUNCHER 1"
echo "     Or via AVD GUI."
echo ""
echo "  4. Log into the target accounts and browse the Home timeline."
echo "     Intercepted tweets will appear in the pipeline terminal."
echo ""
echo "  Emulator PID : $EMULATOR_PID"
echo "  Proxy        : 10.0.2.2:${MITM_PORT}"
echo "  System cert  : ${SYSTEM_CERTS}/${CERT_NAME}"
