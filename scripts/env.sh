#!/usr/bin/env bash
# Shared paths for the isolated Lost Coast GameCube workspace.
# Source from other scripts:  source "$(dirname "$0")/env.sh"

set -euo pipefail

LCGC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LCGC_ROOT

# The engine a plain build uses. LCGC_ENGINE wins when set. Otherwise the
# checkout kept for these builds (it carries a .lcgc-managed file and is
# moved to origin/master below), and only without it the old `engine` link.
LCGC_ENGINE_MANAGED="${LCGC_ENGINE_MANAGED:-/home/tim/Desktop/lcgc-engine-master}"
if [[ -z "${LCGC_ENGINE:-}" && -f "$LCGC_ENGINE_MANAGED/.lcgc-managed" ]]; then
	export LCGC_ENGINE="$LCGC_ENGINE_MANAGED"
fi
export LCGC_ENGINE="${LCGC_ENGINE:-$LCGC_ROOT/engine}"
# Populated third-party tree the managed checkout links to.
LCGC_ENGINE_3RDPARTY="${LCGC_ENGINE_3RDPARTY:-/home/tim/Desktop/lcgc-integrate/3rdparty}"
# The managed checkout is moved to origin/master before every build, so a
# plain build always uses the current engine. Skipped when the fetch fails
# (offline: build what is there) or when someone has edited tracked files in
# it. LCGC_ENGINE_NO_REFRESH=1 turns it off.
if [[ -f "$LCGC_ENGINE/.lcgc-managed" && "${LCGC_ENGINE_NO_REFRESH:-0}" != "1" ]]; then
	if git -C "$LCGC_ENGINE" fetch -q origin master 2>/dev/null; then
		# Its third-party tree is a link to a populated copy, so git lists
		# the submodule paths as deleted; anything else modified is real work.
		if [[ -z "$(git -C "$LCGC_ENGINE" status --porcelain --untracked-files=no 2>/dev/null | grep -v ' 3rdparty/' || true)" ]]; then
			if [[ "$(git -C "$LCGC_ENGINE" rev-parse HEAD)" != "$(git -C "$LCGC_ENGINE" rev-parse origin/master)" ]]; then
				git -C "$LCGC_ENGINE" checkout -q --detach origin/master 2>/dev/null || true
				echo "engine now at: $(git -C "$LCGC_ENGINE" log -1 --format='%h %s' | cut -c1-90)" >&2
			fi
			# A checkout can put a plain 3rdparty directory back in place of
			# the link (it holds no submodule content). Set it aside under a
			# dated name and relink; nothing is deleted.
			if [[ -d "$LCGC_ENGINE/3rdparty" && ! -L "$LCGC_ENGINE/3rdparty" && -d "$LCGC_ENGINE_3RDPARTY" ]]; then
				mv "$LCGC_ENGINE/3rdparty" "$LCGC_ENGINE/3rdparty.checkout.$(date +%Y%m%d-%H%M%S)"
				ln -s "$LCGC_ENGINE_3RDPARTY" "$LCGC_ENGINE/3rdparty"
			fi
		else
			echo "WARNING: $LCGC_ENGINE has local changes; not moving it to origin/master" >&2
		fi
	fi
fi

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
