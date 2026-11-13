#!/usr/bin/env bash
# Dolphin CapFaces smoke probe for out/lostcoastgc.iso
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

ISO="${1:-$LCGC_ISO}"
[[ -f "$ISO" ]] || { echo "error: missing ISO $ISO" >&2; exit 1; }

DOLPHIN_USER="${LCGC_DOLPHIN_USER:-$HOME/.local/share/lostcoast-gc/dolphin}"
NOTE_DIR="$LCGC_NOTES"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOGDIR="$NOTE_DIR/dolphin-probe-$STAMP"
mkdir -p "$DOLPHIN_USER" "$LOGDIR" "$LCGC_SHOTS"

lcgc_pkill_own_dolphin "$DOLPHIN_USER"

if [[ -z "${DOLPHIN_BIN:-}" ]]; then
	if [[ -x "$LCGC_ENGINE/3rdparty/dolphin/build/Binaries/xash-gc-dolphin" ]]; then
		DOLPHIN_BIN="$LCGC_ENGINE/3rdparty/dolphin/build/Binaries/xash-gc-dolphin"
	elif [[ -x "$LCGC_ENGINE/3rdparty/dolphin/build/Binaries/dolphin-emu" ]]; then
		DOLPHIN_BIN="$LCGC_ENGINE/3rdparty/dolphin/build/Binaries/dolphin-emu"
	elif command -v dolphin-emu >/dev/null 2>&1; then
		DOLPHIN_BIN="$(command -v dolphin-emu)"
	elif command -v dolphin-emu-nogui >/dev/null 2>&1; then
		DOLPHIN_BIN="$(command -v dolphin-emu-nogui)"
	else
		echo "error: no Dolphin found (build engine/3rdparty/dolphin or set DOLPHIN_BIN)" >&2
		exit 1
	fi
fi

# Ensure OSReport lands in Logs/dolphin.log for batch probes.
mkdir -p "$DOLPHIN_USER/Config" "$DOLPHIN_USER/Logs"
if [[ ! -f "$DOLPHIN_USER/Config/Logger.ini" ]]; then
	cat > "$DOLPHIN_USER/Config/Logger.ini" <<'EOF'
[Options]
WriteToFile = True
WriteToConsole = True
Verbosity = 4
[Logs]
OSREPORT = True
BOOT = True
OSREPORT_HLE = True
EOF
fi

TIMEOUT="${DOLPHIN_TIMEOUT:-90}"
BATCH_ARGS=()
# Headless Null video avoids GUI segfaults on this host; OSReport still flows.
# Set DOLPHIN_VIDEO=OGL (and DOLPHIN_BATCH=0) for DumpFramesAsImages stills.
VIDEO_BACKEND="${DOLPHIN_VIDEO:-Null}"
if [[ "${DOLPHIN_BATCH:-1}" != "0" ]]; then
	BATCH_ARGS=(-b -v "$VIDEO_BACKEND")
else
	BATCH_ARGS=(-v "$VIDEO_BACKEND")
fi
echo "==> Dolphin CapFaces smoke ($TIMEOUT s) bin=$(basename "$DOLPHIN_BIN") user=$DOLPHIN_USER ${BATCH_ARGS[*]}"
: > "$DOLPHIN_USER/Logs/dolphin.log"
set +e
timeout --signal=KILL "$TIMEOUT" "$DOLPHIN_BIN" -u "$DOLPHIN_USER" "${BATCH_ARGS[@]}" -e "$ISO" \
	>"$LOGDIR/stdout.log" 2>"$LOGDIR/stderr.log"
RC=$?
set -e

# Collect Dolphin Logger output if present
if [[ -f "$DOLPHIN_USER/Logs/dolphin.log" ]]; then
	cp -f "$DOLPHIN_USER/Logs/dolphin.log" "$LOGDIR/dolphin.log"
fi

# DumpFrames / framedumps (also honor GFX.ini DumpPath redirect)
DUMP_DIRS=(
	"$DOLPHIN_USER/Screenshots"
	"$DOLPHIN_USER/Dump/Frames"
	"$DOLPHIN_USER/Dump"
)
if [[ -f "$DOLPHIN_USER/Config/GFX.ini" ]]; then
	dump_path="$(awk -F' *= *' '/^DumpPath *=/{print $2; exit}' "$DOLPHIN_USER/Config/GFX.ini" || true)"
	if [[ -n "$dump_path" ]]; then
		DUMP_DIRS+=("$dump_path" "$(dirname "$dump_path")")
	fi
fi
for dir in "${DUMP_DIRS[@]}"; do
	[[ -d "$dir" ]] || continue
	find "$dir" -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.bmp' -o -name '*.avif' -o -name '*.tga' \) \
		-printf '%T@ %p\n' 2>/dev/null \
		| sort -nr | head -40 | while read -r _ p; do
			cp -f "$p" "$LCGC_SHOTS/" 2>/dev/null || true
			cp -f "$p" "$LOGDIR/" 2>/dev/null || true
		done
done

PASS=0
# Require CapFaces pack load evidence (boot flag alone is not enough).
if grep -Eiq 'CapFaces pack loaded.*faces=[1-9]|CapFaces pack loaded faces=[1-9]' \
	"$LOGDIR/dolphin.log" "$LOGDIR/stderr.log" "$LOGDIR/stdout.log" 2>/dev/null; then
	PASS=1
fi
# Hard fail markers override.
if grep -Eiq 'FATAL: Cannot initialize game data path|MISSING GAME DATA|boot fatal|CapFaces pack reject|CapFaces pack open miss' \
	"$LOGDIR/dolphin.log" "$LOGDIR/stderr.log" "$LOGDIR/stdout.log" 2>/dev/null; then
	PASS=0
fi

{
	echo "# CapFaces smoke $STAMP"
	echo "iso=$ISO"
	echo "rc=$RC pass=$PASS"
	echo "logdir=$LOGDIR"
	echo "dolphin=$DOLPHIN_BIN"
	grep -Eih 'CapFaces|gcmap|MAP_READY|SOUNDSCAPE|soundscape|audio telemetry|world texture pack|gamecube.cfg|gamedir|fatal|DSI|gccapfacessmoke|linked gamedir|base path|DVD boot' \
		"$LOGDIR/dolphin.log" "$LOGDIR/stderr.log" 2>/dev/null | tail -80 || true
} | tee "$LOGDIR/summary.txt"

if (( PASS )); then
	echo "CAPFACES_SMOKE: PASS"
	exit 0
fi
echo "CAPFACES_SMOKE: FAIL (see $LOGDIR)"
exit 1
