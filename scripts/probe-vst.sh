#!/usr/bin/env bash
# Video stress test probe: the retail ptrack_vst flythrough covers the whole
# map, so it is the foundation for render-parity checks.
#   scripts/probe-vst.sh <label> ["note"]
# Packages a private out/lostcoastgc-vst.iso (gcvst in gamecube.cfg) from the
# current DOL + bake, runs it in a private Dolphin profile, and documents the
# run (still / contact sheet / clip / PROGRESS.md row).
# Safe next to other engineers: own ISO, own stage dir, own Dolphin user dir.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LABEL="${1:?label}"
NOTE="${2:-}"
export LCGC_VST=1
export LCGC_ISO="$ROOT/out/lostcoastgc-vst.iso"
export XASH3D_GC_ISO_CACHE="${LCGC_VST_ISO_CACHE:-/dev/shm/lostcoastgc-iso-vst}"
if [[ "${LCGC_VST_SKIP_ISO:-0}" != "1" ]]; then
	bash "$SCRIPT_DIR/build-lostcoastgc.sh" --iso-only --skip-bake \
		>"$ROOT/out/lostcoastgc-vst-build.log" 2>&1 \
		|| { tail -20 "$ROOT/out/lostcoastgc-vst-build.log" >&2; exit 1; }
fi
export DOLPHIN_TIMEOUT="${DOLPHIN_TIMEOUT:-240}"
export LCGC_DOC_VST=1
exec bash "$SCRIPT_DIR/probe-doc.sh" "$LABEL" "$NOTE"
