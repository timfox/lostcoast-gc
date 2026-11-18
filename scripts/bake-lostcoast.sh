#!/usr/bin/env bash
# Host-bake d2_lostcoast → GCWT + GCcf + GoldSrc stub BSP under cache/bake.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
lcgc_require_content

MAP="${1:-d2_lostcoast}"
EXTRACT="$LCGC_CACHE/extracted"
BAKE="$LCGC_CACHE/bake"

mkdir -p "$EXTRACT" "$BAKE"

FULL_GEO_ARGS=()
if [[ "${LCGC_FULL_GEO:-1}" == "1" ]]; then
	FULL_GEO_ARGS=(--full-geo)
	echo "==> [lostcoast-gc] baking $MAP (FULL geo spatial packs)"
else
	echo "==> [lostcoast-gc] baking $MAP (CapFaces smoke)"
fi
# Default DumpFrames = retail info_player_start (matches forced dumpeye).
EYE_ARGS=()
if [[ -n "${LCGC_BAKE_EYE:-}" ]]; then
	EYE_ARGS+=(--eye "$LCGC_BAKE_EYE")
else
	EYE_ARGS+=(--eye "1655.6,2212.7,-48")
fi
if [[ -n "${LCGC_BAKE_ANGLES:-}" ]]; then
	EYE_ARGS+=(--angles "$LCGC_BAKE_ANGLES")
else
	EYE_ARGS+=(--angles "8,150,0")
fi
python3 "$LCGC_ROOT/tools/bake_lostcoast_map.py" \
	--map "$LCGC_CONTENT_LOSTCOAST/maps/${MAP}.bsp" \
	--lostcoast "$LCGC_CONTENT_LOSTCOAST" \
	--hl2 "$LCGC_CONTENT_HL2" \
	--out "$BAKE" \
	--extract "$EXTRACT" \
	"${FULL_GEO_ARGS[@]}" \
	"${EYE_ARGS[@]}"

echo "==> [lostcoast-gc] baking Source skybox → GX CMPR"
bash "$SCRIPT_DIR/bake-lostcoast-sky.sh" "$BAKE/media"

echo "==> [lostcoast-gc] baking Source water → GX water pack"
python3 "$LCGC_ROOT/tools/bake_lostcoast_water.py" \
	--map "$LCGC_CONTENT_LOSTCOAST/maps/${MAP}.bsp" \
	--lostcoast "$LCGC_CONTENT_LOSTCOAST" \
	--hl2 "$LCGC_CONTENT_HL2" \
	--out "$BAKE"

echo "==> [lostcoast-gc] baking characters → engine studio models (shared r_studio.c path)"
if [[ "${LCGC_STUDIO_NPCS:-1}" == "1" ]]; then
	python3 "$LCGC_ROOT/tools/bake_lostcoast_studio.py" \
		--lostcoast "$LCGC_CONTENT_LOSTCOAST" \
		--hl2 "$LCGC_CONTENT_HL2" \
		--out "$BAKE/studio" \
		--map "$LCGC_CONTENT_LOSTCOAST/maps/${MAP}.bsp" \
		--stub-bsp "$BAKE/maps/${MAP}.bsp"
	# What they do: scenes, node graph, voice (lostcoast_npcs.cpp reads these)
	python3 "$LCGC_ROOT/tools/bake_lostcoast_actors.py" \
		--lostcoast "$LCGC_CONTENT_LOSTCOAST" \
		--hl2 "$LCGC_CONTENT_HL2" \
		--map "$LCGC_CONTENT_LOSTCOAST/maps/${MAP}.bsp" \
		--out "$BAKE/studio"
	# The gestures those scenes play, one rotation track each (r_gx_gesture.c)
	python3 "$LCGC_ROOT/tools/bake_lostcoast_gestures.py" \
		--lostcoast "$LCGC_CONTENT_LOSTCOAST" \
		--hl2 "$LCGC_CONTENT_HL2" \
		--out "$BAKE/studio"
else
	rm -rf "$BAKE/studio"
fi

echo "==> [lostcoast-gc] baking Source props → GX prop pack + mip chains"
python3 "$LCGC_ROOT/tools/bake_lostcoast_props.py" \
	--map "$LCGC_CONTENT_LOSTCOAST/maps/${MAP}.bsp" \
	--lostcoast "$LCGC_CONTENT_LOSTCOAST" \
	--hl2 "$LCGC_CONTENT_HL2" \
	--out "$BAKE/media" \
	--eye 1655.6 2212.7 -48 \
	--yaw 150

echo "==> [lostcoast-gc] baking VST path + Source PVS for CapFaces cull"
python3 "$LCGC_ROOT/tools/bake_lostcoast_vst.py" --map "$MAP" --media "$BAKE/media"

echo "bake complete:"
ls -lh "$BAKE/media"/wt_*.gct "$BAKE/media"/cf_*.gcp "$BAKE/media"/gw_*.gcw \
	"$BAKE/media"/gp_*.gcp "$BAKE/media"/pt_*.gct "$BAKE/media"/prop_*.cmp \
	"$BAKE/media"/sky_*.cmp "$BAKE/media"/water_gx.cmp \
	"$BAKE/media"/vis_*.gcv "$BAKE/media"/vst_*_path.bin "$BAKE/media"/cf_*.cls \
	"$BAKE/maps"/*.bsp 2>/dev/null || true
