#!/usr/bin/env bash
# Shared paths for the isolated Lost Coast GameCube workspace.
# Source from other scripts:  source "$(dirname "$0")/env.sh"

set -euo pipefail

LCGC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LCGC_ROOT

export LCGC_ENGINE="${LCGC_ENGINE:-$LCGC_ROOT/engine}"
# A build against an engine checkout that is far behind master gives today's
# bake to an old renderer (the default `engine` link sat 486 commits behind
# for days). Say so; fetch nothing, change nothing.
if [[ -d "$LCGC_ENGINE/.git" || -f "$LCGC_ENGINE/.git" ]]; then
	_lcgc_behind="$(git -C "$LCGC_ENGINE" rev-list --count HEAD..origin/master 2>/dev/null || echo 0)"
	if [[ "${_lcgc_behind:-0}" -gt 20 ]]; then
		echo "WARNING: engine checkout $LCGC_ENGINE is ${_lcgc_behind} commits behind origin/master" >&2
		echo "         ($(git -C "$LCGC_ENGINE" log -1 --format='%h %cd' --date=short 2>/dev/null)). Set LCGC_ENGINE to a current checkout." >&2
	fi
	unset _lcgc_behind
fi
if [[ ! -d "$LCGC_ENGINE/scripts" ]]; then
	echo "error: engine symlink missing or incomplete: $LCGC_ENGINE" >&2
	echo "  run scripts/link-deps.sh" >&2
	exit 1
fi

export LCGC_CONTENT_LOSTCOAST="${LCGC_CONTENT_LOSTCOAST:-$LCGC_ROOT/content/lostcoast}"
export LCGC_CONTENT_HL2="${LCGC_CONTENT_HL2:-$LCGC_ROOT/content/hl2}"

export LCGC_OUT="${LCGC_OUT:-$LCGC_ROOT/out}"
export LCGC_DOL="${LCGC_DOL:-$LCGC_OUT/lostcoastgc.dol}"
export LCGC_ISO="${LCGC_ISO:-$LCGC_OUT/lostcoastgc.iso}"
export LCGC_CACHE="${LCGC_CACHE:-$LCGC_ROOT/cache}"
export LCGC_PLAYTESTS="${LCGC_PLAYTESTS:-$LCGC_ROOT/playtests}"
export LCGC_SHOTS="${LCGC_SHOTS:-$LCGC_PLAYTESTS/screenshots}"
export LCGC_NOTES="${LCGC_NOTES:-$LCGC_PLAYTESTS/notes}"

if [[ -z "${XASH3D_GC_ISO_CACHE:-}" ]]; then
	if [[ -d /dev/shm ]]; then
		export XASH3D_GC_ISO_CACHE="/dev/shm/lostcoastgc-iso-build"
	else
		export XASH3D_GC_ISO_CACHE="$LCGC_OUT/iso-cache"
	fi
fi

mkdir -p "$LCGC_OUT" "$LCGC_SHOTS" "$LCGC_NOTES" "$LCGC_CACHE" "$XASH3D_GC_ISO_CACHE"

lcgc_require_content() {
	if [[ ! -f "$LCGC_CONTENT_LOSTCOAST/maps/d2_lostcoast.bsp" ]]; then
		echo "error: Lost Coast map missing at $LCGC_CONTENT_LOSTCOAST/maps/d2_lostcoast.bsp" >&2
		echo "  symlink content/lostcoast → Steam Half-Life 2/lostcoast" >&2
		exit 1
	fi
	if [[ ! -d "$LCGC_CONTENT_HL2" ]]; then
		echo "error: HL2 content missing at $LCGC_CONTENT_HL2" >&2
		exit 1
	fi
}

lcgc_pkill_own_dolphin() {
	local pattern
	for pattern in "$@"; do
		[[ -n "$pattern" ]] || continue
		pkill -f -- "-u ${pattern}" 2>/dev/null || true
	done
}
