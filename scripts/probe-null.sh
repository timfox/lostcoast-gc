#!/usr/bin/env bash
# Headless flythrough check: build the DOL and a private VST ISO, run it on
# Dolphin's Null video backend (no GPU needed), print ride progress and the
# engine's per-stage frame timing.
#   scripts/probe-null.sh [seconds=100]
# Needs LCGC_ENGINE (and XASH_GAMECUBE_BUILD_LOCK for a private build lock).
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
U="${LCGC_NULL_DOLPHIN_USER:-$HOME/.local/share/lostcoast-gc-claude/dolphin}"
ISO="$ROOT/out/lostcoastgc-vst.iso"
bash scripts/build-lostcoastgc.sh --dol-only --skip-bake --skip-hlsdk > out/probe-null-dol.log 2>&1 \
	|| { grep -E "rror:" out/probe-null-dol.log | head; exit 1; }
LCGC_VST=1 LCGC_ISO="$ISO" XASH3D_GC_ISO_CACHE=/dev/shm/lostcoastgc-iso-vst \
	bash scripts/build-lostcoastgc.sh --iso-only --skip-bake > out/probe-null-iso.log 2>&1 \
	|| { tail -5 out/probe-null-iso.log; exit 1; }
mkdir -p "$U/Logs"
: > "$U/Logs/dolphin.log"
timeout --signal=KILL "${1:-100}" "${LCGC_ENGINE:-$ROOT/engine}/3rdparty/dolphin/build/Binaries/xash-gc-dolphin" \
	-u "$U" -b -v Null -e "$ISO" > /dev/null 2>&1
L="$U/Logs/dolphin.log"
grep -E "VST ride seg" "$L" | tail -1 | cut -c60-150
grep -E "stage ms" "$L" | cut -c95-200 | head -6
grep -E "texture slots per|prop horizon|prop draw|arena tail=(FAIL|OVERLAP)" "$L" | cut -c75-200 | head -6
