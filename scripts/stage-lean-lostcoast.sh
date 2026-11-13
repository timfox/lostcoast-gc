#!/usr/bin/env bash
# Stage CapFaces smoke tree into the ISO cache (never mutates Steam).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
lcgc_require_content

BAKE="${LCGC_BAKE:-$LCGC_CACHE/bake}"
STAGE="${LCGC_STAGE:-$XASH3D_GC_ISO_CACHE/lostcoast-smoke}"
MAP="${1:-d2_lostcoast}"

if [[ ! -f "$BAKE/media/cf_${MAP}.gcp" || ! -f "$BAKE/media/wt_${MAP}.gct" ]]; then
	echo "error: missing bake packs under $BAKE/media (run scripts/bake-lostcoast.sh)" >&2
	exit 1
fi
if [[ ! -f "$BAKE/maps/${MAP}.bsp" ]]; then
	echo "error: missing stub BSP $BAKE/maps/${MAP}.bsp" >&2
	exit 1
fi

rm -rf "$STAGE"
mkdir -p "$STAGE"/{maps,media,resource}

cp -a "$BAKE/maps/${MAP}.bsp" "$STAGE/maps/"
cp -a "$BAKE/media/cf_${MAP}.gcp" "$BAKE/media/wt_${MAP}.gct" "$STAGE/media/"
# Source PVS + VST on-rails path (CapFaces densify; real cluster cull)
if [[ -f "$BAKE/media/vis_${MAP}.gcv" ]]; then
	cp -a "$BAKE/media/vis_${MAP}.gcv" "$STAGE/media/"
	echo "  + Source vis pack vis_${MAP}.gcv"
fi
if [[ -f "$BAKE/media/eye_${MAP}.gce" ]]; then
	cp -a "$BAKE/media/eye_${MAP}.gce" "$STAGE/media/"
	echo "  + dump-eye Source cluster eye_${MAP}.gce"
fi
if [[ -f "$BAKE/media/vst_${MAP}_path.bin" ]]; then
	cp -a "$BAKE/media/vst_${MAP}_path.bin" "$STAGE/media/"
	echo "  + VST path vst_${MAP}_path.bin"
fi
if [[ -f "$BAKE/media/cf_${MAP}.cls" ]]; then
	cp -a "$BAKE/media/cf_${MAP}.cls" "$STAGE/media/"
	echo "  + CapFaces clusters cf_${MAP}.cls"
fi
# Single FST-safe pak with clusters for every spatial CapFaces pack
if [[ ! -f "$BAKE/media/cf_${MAP}_cls.pak" ]]; then
	python3 "$SCRIPT_DIR/../tools/bake_lostcoast_cls_pak.py" --map "$MAP" --media "$BAKE/media" || true
fi
if [[ -f "$BAKE/media/cf_${MAP}_cls.pak" ]]; then
	cp -a "$BAKE/media/cf_${MAP}_cls.pak" "$STAGE/media/"
	echo "  + CapFaces cluster pak cf_${MAP}_cls.pak ($(wc -c <"$BAKE/media/cf_${MAP}_cls.pak") bytes)"
fi
# One FST-safe pak for all spatial CapFaces Source clusters (VST PVS).
if [[ -f "$BAKE/media/cf_${MAP}_cls.pak" ]]; then
	cp -a "$BAKE/media/cf_${MAP}_cls.pak" "$STAGE/media/"
	echo "  + CapFaces cluster pak cf_${MAP}_cls.pak"
elif [[ -f "$BAKE/media/cf_${MAP}.cls" ]]; then
	python3 "$SCRIPT_DIR/../tools/bake_lostcoast_cls_pak.py" --map "$MAP" --media "$BAKE/media" || true
	if [[ -f "$BAKE/media/cf_${MAP}_cls.pak" ]]; then
		cp -a "$BAKE/media/cf_${MAP}_cls.pak" "$STAGE/media/"
		echo "  + CapFaces cluster pak cf_${MAP}_cls.pak"
	fi
fi
# Spatial full-geo packs — optional on disc (manifest open hung CapFaces prefetch
# after sky in dolphin-probe-20261109-095416/101442). Default OFF until FS path
# is fixed; set LCGC_SPATIAL_DISC_MAX>0 to ship a near-spawn subset.
shopt -s nullglob
# Default ships a near-spawn subset now that spatial.txt EOF parse is fixed.
SPATIAL_MAX="${LCGC_SPATIAL_DISC_MAX:-48}"
MANIFEST="$BAKE/media/cf_${MAP}_spatial.txt"
if [[ -f "$MANIFEST" && "$SPATIAL_MAX" -gt 0 ]]; then
	# Manifest is spawn-nearest first (primary then p01…).
	n=0
	while read -r fname _rest; do
		[[ -z "$fname" || "$fname" == \#* ]] && continue
		# Primary cf_<map>.gcp already staged; only copy extras.
		[[ "$fname" == "cf_${MAP}.gcp" ]] && continue
		src="$BAKE/media/$fname"
		[[ -f "$src" ]] || continue
		cp -a "$src" "$STAGE/media/"
		# Spatial .cls omitted — 12+ sidecars broke GameCube DVD FST mount.
		n=$((n + 1))
		(( n >= SPATIAL_MAX )) && break
	done < "$MANIFEST"
	{
		echo "# spatial CapFaces packs for ${MAP} (disc subset max=${SPATIAL_MAX})"
		n=0
		while read -r line; do
			[[ -z "$line" || "$line" == \#* ]] && continue
			echo "$line"
			n=$((n + 1))
			(( n >= SPATIAL_MAX )) && break
		done < "$MANIFEST"
	} > "$STAGE/media/cf_${MAP}_spatial.txt"
	echo "  + spatial CapFaces packs on disc: $n extras + primary (max=${SPATIAL_MAX})"
else
	echo "  + spatial CapFaces packs on disc: skipped (LCGC_SPATIAL_DISC_MAX=${SPATIAL_MAX})"
fi
# GX water pack + scroll CMPR
if [[ -f "$BAKE/media/gw_${MAP}.gcw" ]]; then
	cp -a "$BAKE/media/gw_${MAP}.gcw" "$STAGE/media/"
	echo "  + water pack gw_${MAP}.gcw ($(wc -c <"$BAKE/media/gw_${MAP}.gcw") bytes)"
fi
if [[ -f "$BAKE/media/water_gx.cmp" ]]; then
	cp -a "$BAKE/media/water_gx.cmp" "$STAGE/media/"
	echo "  + water_gx.cmp"
fi
if [[ -f "$BAKE/media/water_gx.ia8" ]]; then
	cp -a "$BAKE/media/water_gx.ia8" "$STAGE/media/"
	echo "  + water_gx.ia8"
fi
if [[ ! -f "$BAKE/media/sand_nrm.cmp" || ! -f "$BAKE/media/sand_spec.cmp" ]]; then
	if [[ -d "$LCGC_CONTENT_LOSTCOAST" ]]; then
		python3 "$SCRIPT_DIR/../tools/bake_sand_env.py" \
			--lostcoast "$LCGC_CONTENT_LOSTCOAST" --out "$BAKE/media" || true
	fi
fi
# Source material → TEV recipe table (detail / alphatest; ref/gx/r_gx_tevmat.c)
if [[ -f "$BAKE/media/tevmat.gcm" ]]; then
	cp -a "$BAKE/media/tevmat.gcm" "$STAGE/media/"
	echo "  + tevmat.gcm (Source material TEV table)"
fi
# Full BSP as baked GX display lists + real PVS (ref/gx/r_gx_worlddl.c).
# Rebaked whenever the material tables are newer (ids must match tevmat.gcm).
if [[ "${LCGC_WORLD_DL:-1}" == "1" ]]; then
	if [[ ! -f "$BAKE/media/wd_${MAP}.gcd" || "$BAKE/texture-names.json" -nt "$BAKE/media/wd_${MAP}.gcd" \
		|| "$SCRIPT_DIR/../tools/bake_worlddl.py" -nt "$BAKE/media/wd_${MAP}.gcd" ]]; then
		python3 "$SCRIPT_DIR/../tools/bake_worlddl.py" --out "$BAKE" || true
	fi
	if [[ -f "$BAKE/media/wd_${MAP}.gcd" ]]; then
		cp -a "$BAKE/media/wd_${MAP}.gcd" "$STAGE/media/"
		echo "  + wd_${MAP}.gcd (world display lists, $(wc -c <"$BAKE/media/wd_${MAP}.gcd") bytes)"
	fi
fi
if [[ -f "$BAKE/media/sand_spec.cmp" ]]; then
	cp -a "$BAKE/media/sand_spec.cmp" "$STAGE/media/"
	echo "  + sand_spec.cmp (envmapmask)"
fi
if [[ -f "$BAKE/media/sand_nrm.cmp" ]]; then
	cp -a "$BAKE/media/sand_nrm.cmp" "$STAGE/media/"
	echo "  + sand_nrm.cmp (\$bumpmap)"
fi
if [[ -f "$BAKE/media/sand_sun.cmp" ]]; then
	cp -a "$BAKE/media/sand_sun.cmp" "$STAGE/media/"
	echo "  + sand_sun.cmp (specular disc)"
fi
if [[ -f "$BAKE/media/gp_${MAP}.gcp" ]]; then
	cp -a "$BAKE/media/gp_${MAP}.gcp" "$STAGE/media/"
	echo "  + gp_${MAP}.gcp (Source props)"
fi
if [[ -f "$BAKE/media/pt_${MAP}.gct" ]]; then
	cp -a "$BAKE/media/pt_${MAP}.gct" "$STAGE/media/"
	echo "  + pt_${MAP}.gct (prop mip chains)"
fi
for pcmp in "$BAKE/media"/prop_*.cmp; do
	[[ -f "$pcmp" ]] || continue
	cp -a "$pcmp" "$STAGE/media/"
	echo "  + $(basename "$pcmp")"
done
shopt -u nullglob

# Source→GX skybox (sky_lostcoast_hdr → media/sky_sky_lostcoast_hdr*.cmp)
SKYNAME="sky_lostcoast_hdr"
if [[ ! -f "$BAKE/media/sky_${SKYNAME}rt.cmp" ]]; then
	bash "$SCRIPT_DIR/bake-lostcoast-sky.sh" "$BAKE/media" "$SKYNAME"
fi
cp -a "$BAKE/media"/sky_"${SKYNAME}"*.cmp "$STAGE/media/"
echo "  + skybox $SKYNAME ($(ls "$STAGE/media"/sky_"${SKYNAME}"*.cmp | wc -l) sides)"

# Minimal gameinfo / liblist so FS mounts the gamedir.
cat > "$STAGE/liblist.gam" <<'EOF'
game "Half-Life 2 Lost Coast (GX smoke)"
gamedir "lostcoast"
fallback_dir "valve"
type "single"
EOF

# Eye: pack eye0 + 1000u along yaw=150, Quake pitch +10 (look down at water).
# Host fullgeo sweep: best corr vs retail-lostcoast-target.jpg at this aim.
# Disc cfg token is "dumpeye" (sys_gamecube → -gcdumpeye).
cat > "$STAGE/gamecube.cfg" <<EOF
# CapFaces smoke — Source→GX packs, stub GoldSrc BSP loader hook
gccapfacessmoke
gcworldrender
map ${MAP}
dumpframes
# Retail still ≈ info_player_start (CapFaces rebaked for this eye).
dumpeye 1655.6 2212.7 -48 8 150
# Pad look after DumpFrames hold — CapFaces becomes free-look playable.
gcplayerinput
gcplayercam
EOF

# Empty markers the engine may probe for
mkdir -p "$STAGE/gfx" "$STAGE/sound" "$STAGE/resource"
: > "$STAGE/resource/.keep"

# Source soundscapes → sound/ + tip-safe sound/sc_* oneshots (ASND stream bed).
HL2_CONTENT="${LCGC_CONTENT_HL2:-$LCGC_ROOT/content/hl2}"
if [[ -d "$LCGC_CONTENT_LOSTCOAST" && -d "$HL2_CONTENT" ]]; then
	python3 "$SCRIPT_DIR/stage_lostcoast_audio.py" \
		--stage "$STAGE" \
		--lostcoast "$LCGC_CONTENT_LOSTCOAST" \
		--hl2 "$HL2_CONTENT"
else
	echo "  (skip audio: need LCGC_CONTENT_LOSTCOAST + hl2 content)"
fi

# Retail Lost Coast title plate (console/background01 → resource/gc_menu).
TITLE_MENU="${LCGC_CACHE}/title/gc_menu"
if [[ ! -f "$TITLE_MENU/background.tga" || ! -f "$TITLE_MENU/SOURCE.txt" ]]; then
	bash "$SCRIPT_DIR/bake-lostcoast-title.sh" "$TITLE_MENU"
fi
mkdir -p "$STAGE/resource/gc_menu"
cp -a "$TITLE_MENU/." "$STAGE/resource/gc_menu/"
echo "  + retail Lost Coast gc_menu ($(wc -c <"$STAGE/resource/gc_menu/background.tga") bytes background.tga)"

# GoldSrc boot essentials from HL1 valve (read-only copy; never mutate Steam).
HL1_VALVE=""
for cand in \
	"$LCGC_ENGINE/Half-Life/valve" \
	"$LCGC_ENGINE/OUT/valve" \
	/home/tim/snap/steam/common/.local/share/Steam/steamapps/common/Half-Life/valve
do
	if [[ -d "$cand" ]]; then
		HL1_VALVE="$cand"
		break
	fi
done

if [[ -n "$HL1_VALVE" ]]; then
	[[ -f "$HL1_VALVE/gfx.wad" ]] && cp -f "$HL1_VALVE/gfx.wad" "$STAGE/gfx.wad"
	# Host_Init fatals without delta.lst once -gcmap engages.
	[[ -f "$HL1_VALVE/delta.lst" ]] && cp -f "$HL1_VALVE/delta.lst" "$STAGE/delta.lst"
	for f in default.cfg titles.txt skill.cfg; do
		[[ -f "$STAGE/$f" && -s "$STAGE/$f" ]] && continue
		[[ -f "$HL1_VALVE/$f" ]] && cp -f "$HL1_VALVE/$f" "$STAGE/$f"
	done
	mkdir -p "$STAGE/gfx"
	for f in palette.lmp colormap.lmp; do
		[[ -f "$STAGE/gfx/$f" ]] && continue
		[[ -f "$HL1_VALVE/gfx/$f" ]] && cp -f "$HL1_VALVE/gfx/$f" "$STAGE/gfx/$f"
	done
	# HL1 357 as store-only pk3 — loose models/ on ISO9660 breaks libogc DVD
	# mount; gc_studio/ mirror is the CapFaces BSS load path.
	if [[ -f "$HL1_VALVE/models/v_357.mdl" ]]; then
		vm_tmp="$(mktemp -d)"
		mkdir -p "$vm_tmp/gc_studio" "$vm_tmp/models"
		cp -f "$HL1_VALVE/models/v_357.mdl" "$vm_tmp/gc_studio/v_357.mdl"
		cp -f "$HL1_VALVE/models/v_357.mdl" "$vm_tmp/models/v_357.mdl"
		( cd "$vm_tmp" && zip -0 -r "$STAGE/viewmodel.pk3" gc_studio models )
		rm -rf "$vm_tmp"
		echo "  + viewmodel.pk3 (HL1 v_357 + gc_studio mirror, store-only)"
	fi
fi
if [[ ! -f "$STAGE/viewmodel.pk3" ]]; then
	echo "  (WARNING: no viewmodel.pk3 — CapFaces DumpFrames will miss the gun)"
fi

echo "staged lean Lost Coast → $STAGE"
du -sh "$STAGE" "$STAGE/media"/*
[[ -f "$STAGE/gfx.wad" ]] && echo "  + gfx.wad" || echo "  (no gfx.wad)"
[[ -f "$STAGE/delta.lst" ]] && echo "  + delta.lst" || echo "  (WARNING: no delta.lst — Host_Init will fatal)"
