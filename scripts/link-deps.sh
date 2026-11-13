#!/usr/bin/env bash
# Recreate engine + content symlinks for this machine.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

ENGINE_DEFAULT="/run/media/tim/Expansion1/xash3d-gc"
STEAM_HL2_DEFAULT="/home/tim/snap/steam/common/.local/share/Steam/steamapps/common/Half-Life 2"

ENGINE="${LCGC_ENGINE_TARGET:-$ENGINE_DEFAULT}"
STEAM_HL2="${LCGC_STEAM_HL2:-$STEAM_HL2_DEFAULT}"

usage() {
	cat <<EOF
usage: scripts/link-deps.sh [--engine PATH] [--steam-hl2 PATH]

Defaults:
  engine:    $ENGINE_DEFAULT
  steam-hl2: $STEAM_HL2_DEFAULT
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--engine) ENGINE="${2:-}"; shift ;;
		--steam-hl2) STEAM_HL2="${2:-}"; shift ;;
		-h|--help) usage; exit 0 ;;
		*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
	esac
	shift
done

[[ -d "$ENGINE/scripts" ]] || { echo "error: engine path invalid: $ENGINE" >&2; exit 1; }
[[ -d "$STEAM_HL2/lostcoast" ]] || { echo "error: Lost Coast missing: $STEAM_HL2/lostcoast" >&2; exit 1; }
[[ -d "$STEAM_HL2/hl2" ]] || { echo "error: HL2 missing: $STEAM_HL2/hl2" >&2; exit 1; }

mkdir -p "$ROOT/content"
ln -sfn "$ENGINE" "$ROOT/engine"
ln -sfn "$STEAM_HL2/lostcoast" "$ROOT/content/lostcoast"
ln -sfn "$STEAM_HL2/hl2" "$ROOT/content/hl2"

echo "linked:"
printf '  %s -> %s\n' engine "$(readlink -f "$ROOT/engine")"
printf '  %s -> %s\n' content/lostcoast "$(readlink -f "$ROOT/content/lostcoast")"
printf '  %s -> %s\n' content/hl2 "$(readlink -f "$ROOT/content/hl2")"
