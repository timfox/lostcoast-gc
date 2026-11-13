#!/usr/bin/env bash
# Build lostcoastgc.dol + lostcoastgc.iso into this repo's out/ only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

usage() {
	cat <<'EOF'
usage: scripts/build-lostcoastgc.sh [options]

  --dol-only     build/link lostcoastgc.dol only
  --iso-only     package out/lostcoastgc.iso from existing DOL + bake
  --skip-bake    reuse cache/bake (default: bake if packs missing)
  --skip-hlsdk   reuse existing OUT/hlsdk-gamecube/lostcoast (or valve copy)
  -h, --help
EOF
}

DOL_ONLY=0
ISO_ONLY=0
SKIP_BAKE=0
SKIP_HLSDK=0

while [[ $# -gt 0 ]]; do
	case "$1" in
		--dol-only) DOL_ONLY=1 ;;
		--iso-only) ISO_ONLY=1 ;;
		--skip-bake) SKIP_BAKE=1 ;;
		--skip-hlsdk) SKIP_HLSDK=1 ;;
		-h|--help) usage; exit 0 ;;
		*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
	esac
	shift
done

lcgc_require_content

BAKE="$LCGC_CACHE/bake"
ENGINE_DOL="$LCGC_ENGINE/OUT/bin/lostcoastgc.dol"
ENGINE_HLSDK_VALVE="$LCGC_ENGINE/OUT/hlsdk-gamecube/valve"
ENGINE_HLSDK_LC="$LCGC_ENGINE/OUT/hlsdk-gamecube/lostcoast"

if (( ! SKIP_BAKE )); then
	if [[ ! -f "$BAKE/media/cf_d2_lostcoast.gcp" ]]; then
		bash "$SCRIPT_DIR/bake-lostcoast.sh"
	else
		echo "==> reusing bake at $BAKE"
	fi
fi

if (( ! ISO_ONLY )); then
	echo "==> [lostcoast-gc] building CapFaces-smoke DOL (profile=lostcoast)"
	if [[ -z "${XASH_GAMECUBE_OGC_STACK:-}" ]]; then
		if [[ -d /opt/devkitpro/libogc ]]; then
			export XASH_GAMECUBE_OGC_STACK=libogc
		elif [[ -d /home/tim/Desktop/dod-gc/out/network-toolchain/devkitpro/libogc2 ]]; then
			export DEVKITPRO="${DEVKITPRO:-/home/tim/Desktop/dod-gc/out/network-toolchain/devkitpro}"
			export XASH_GAMECUBE_OGC_STACK=libogc2
		else
			export XASH_GAMECUBE_OGC_STACK=auto
		fi
	fi
	export XASH_GAMECUBE_LINK_BBA="${XASH_GAMECUBE_LINK_BBA:-0}"
	# Share the engine flock with dod-gc/HL1 — a private lock lets concurrent
	# configures poison XASH_GAMEDIR into valve/dod while still emitting lostcoastgc.dol.
	export XASH_GAMECUBE_BUILD_LOCK="${XASH_GAMECUBE_BUILD_LOCK:-$LCGC_ENGINE/.ai/build-gamecube.lock}"
	echo "    OGC stack=$XASH_GAMECUBE_OGC_STACK DEVKITPRO=${DEVKITPRO:-/opt/devkitpro}"
	echo "    build lock=$XASH_GAMECUBE_BUILD_LOCK"

	if (( ! SKIP_HLSDK )); then
		if [[ ! -s "$ENGINE_HLSDK_LC/dlls/libhl_gamecube_ppc.a" ]]; then
			if [[ -s "$ENGINE_HLSDK_VALVE/dlls/libhl_gamecube_ppc.a" ]]; then
				echo "==> copying valve HLSDK archives → lostcoast profile"
				mkdir -p "$ENGINE_HLSDK_LC"
				cp -a "$ENGINE_HLSDK_VALVE/." "$ENGINE_HLSDK_LC/"
			else
				echo "==> building valve HLSDK then cloning to lostcoast"
				cd "$LCGC_ENGINE"
				HLSDK_GAMECUBE_GAMEDIR=valve bash scripts/hlsdk-gamecube-build.sh
				mkdir -p "$ENGINE_HLSDK_LC"
				cp -a "$ENGINE_HLSDK_VALVE/." "$ENGINE_HLSDK_LC/"
			fi
		fi
	fi

	cd "$LCGC_ENGINE"
	# Distclean when the tip DOL still embeds the wrong gamedir (stale .o defines).
	# Avoid `grep -q` under pipefail — early close SIGPIPEs strings → false negative.
	if [[ -f "$ENGINE_DOL" ]] && ! strings "$ENGINE_DOL" | grep 'Need sd:/xash3d/lostcoast' >/dev/null; then
		echo "==> tip DOL gamedir stale; waf distclean before lostcoast rebuild"
		./waf distclean || true
		rm -rf build
	fi
	XASH3D_GC_SKIP_DISC_BUILD=1 \
	HLSDK_GAMECUBE_GAMEDIR=lostcoast \
		bash scripts/build-gamecube.sh --game-profile=lostcoast

	[[ -f "$ENGINE_DOL" ]] || { echo "error: expected $ENGINE_DOL" >&2; exit 1; }
	if ! strings "$ENGINE_DOL" | grep 'Need sd:/xash3d/lostcoast' >/dev/null; then
		echo "error: $ENGINE_DOL linked without XASH_GAMEDIR=lostcoast (waf race?)" >&2
		strings "$ENGINE_DOL" | grep 'Need sd:/xash3d' >&2 || true
		exit 1
	fi
	mkdir -p "$LCGC_OUT"
	cp -f "$ENGINE_DOL" "$LCGC_DOL"
	echo "DOL → $LCGC_DOL (gamedir=lostcoast verified)"
fi

if (( DOL_ONLY )); then
	exit 0
fi

# Prefer the tip engine DOL when out/ is stale or wrong-gamedir.
if [[ -f "$ENGINE_DOL" ]] && strings "$ENGINE_DOL" | grep 'Need sd:/xash3d/lostcoast' >/dev/null; then
	mkdir -p "$LCGC_OUT"
	cp -f "$ENGINE_DOL" "$LCGC_DOL"
fi
[[ -f "$LCGC_DOL" ]] || { echo "error: missing $LCGC_DOL (build DOL first)" >&2; exit 1; }
if ! strings "$LCGC_DOL" | grep 'Need sd:/xash3d/lostcoast' >/dev/null; then
	echo "error: $LCGC_DOL is not a lostcoast-linked DOL; rebuild with --dol-only" >&2
	strings "$LCGC_DOL" | grep 'Need sd:/xash3d' >&2 || true
	exit 1
fi

bash "$SCRIPT_DIR/stage-lean-lostcoast.sh" d2_lostcoast
STAGE="$XASH3D_GC_ISO_CACHE/lostcoast-smoke"

# Minimal GoldSrc boot files the smoke/retail disc tooling expects.
for f in default.cfg config.cfg valve.rc; do
	[[ -f "$STAGE/$f" ]] || echo "" > "$STAGE/$f"
done
# Tiny palette/colormap stubs if absent (engine may soft-fail without them).
mkdir -p "$STAGE/gfx"
if [[ ! -f "$STAGE/gfx/palette.lmp" && -f "$LCGC_ENGINE/OUT/valve/gfx/palette.lmp" ]]; then
	cp -f "$LCGC_ENGINE/OUT/valve/gfx/palette.lmp" "$STAGE/gfx/" 2>/dev/null || true
	cp -f "$LCGC_ENGINE/OUT/valve/gfx/colormap.lmp" "$STAGE/gfx/" 2>/dev/null || true
fi
# Prefer engine valve palette from a known HL1 tree if present.
if [[ ! -f "$STAGE/gfx/palette.lmp" ]]; then
	for cand in \
		"$LCGC_ENGINE/3rdparty/hlsdk-portable/valve/gfx/palette.lmp" \
		/home/tim/snap/steam/common/.local/share/Steam/steamapps/common/Half-Life/valve/gfx/palette.lmp
	do
		if [[ -f "$cand" ]]; then
			cp -f "$cand" "$STAGE/gfx/palette.lmp"
			cp -f "$(dirname "$cand")/colormap.lmp" "$STAGE/gfx/colormap.lmp" 2>/dev/null || true
			break
		fi
	done
fi

# Nest as parent/lostcoast so GC FST is xash3d/lostcoast/... matching boot cfg path.
NEST="$XASH3D_GC_ISO_CACHE/lostcoast-nest"
rm -rf "$NEST"
mkdir -p "$NEST"
cp -a "$STAGE" "$NEST/lostcoast"

# CapFaces smoke + retail title plaque (Lost Coast background01 in gc_menu).
# CapFaces eye0 bake angles (HL: +pitch looks down toward waterline).
cat > "$NEST/lostcoast/gamecube.cfg" <<'EOF'
gccapfacessmoke
gcworldrender
map d2_lostcoast
dumpframes
# Opening eye, turned toward the boats (yaw 304). Yaw 91 looks up the cliff.
# Frame 256 of the reference ride, a third of the way from node 6
# (1458, 1863, 170 yaw 120) to node 7 (764, 3406, 75 yaw 175).
dumpeye 1236 2357 140 0 138
# Optional stress cam: uncomment gcvst to ride ptrack_vst instead of dumpeye.
#gcvst
gcplayerinput
gcplayercam
EOF
# Optional still-eye override: LCGC_DUMP_EYE="x y z [pitch] [yaw]"
if [[ -n "${LCGC_DUMP_EYE:-}" ]]; then
	sed -i "s/^dumpeye .*/dumpeye ${LCGC_DUMP_EYE}/" "$NEST/lostcoast/gamecube.cfg"
	echo "  + gamecube.cfg: dumpeye ${LCGC_DUMP_EYE}"
fi
# LCGC_VST=1: video-stress-test disc — ride the retail ptrack_vst flythrough
# (whole-map coverage for render parity). Pair with a private LCGC_ISO and
# XASH3D_GC_ISO_CACHE so the still-eye disc is left alone (scripts/probe-vst.sh).
if [[ "${LCGC_VST:-0}" == "1" ]]; then
	sed -i 's/^#gcvst$/gcvst/' "$NEST/lostcoast/gamecube.cfg"
	# Prefer the on-rails ride; leave dumpeye commented so GetArgv does not
	# bake a still-eye path when gcvst wins later.
	sed -i 's/^dumpeye /#dumpeye /' "$NEST/lostcoast/gamecube.cfg"
	echo "  + gamecube.cfg: gcvst (stress flythrough)"
fi
# HL1 357: viewmodel.pk3 fails DVD fopen on Flipper (only extras.pk3 mounts).
# Disc builder also swaps nest extras for valve extras — patch every extras we
# can find so CapFaces DumpFrames keep the gun.
VM_MDL=""
for cand in \
	"$LCGC_ENGINE/Half-Life/valve/models/v_357.mdl" \
	/home/tim/snap/steam/common/.local/share/Steam/steamapps/common/Half-Life/valve/models/v_357.mdl
do
	if [[ -f "$cand" ]]; then
		VM_MDL="$cand"
		break
	fi
done
inject_357() {
	local extras="$1"
	[[ -n "$VM_MDL" && -f "$extras" ]] || return 0
	python3 - "$extras" "$VM_MDL" <<'PY'
import sys, zipfile
from pathlib import Path
extras, mdl = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(extras, "a", compression=zipfile.ZIP_STORED) as z:
	names = set(z.namelist())
	for arc in ("models/v_357.mdl", "gc_studio/v_357.mdl"):
		if arc not in names:
			z.write(mdl, arc)
			print(f"  + {extras.name} {arc}")
PY
}
inject_357 "$NEST/lostcoast/extras.pk3"
# Path the retail disc builder prefers over nest extras:
inject_357 "/run/media/tim/Expansion1/xash3d-gc/OUT/valve/extras.pk3"
inject_357 "$LCGC_ENGINE/../OUT/valve/extras.pk3"
if [[ -n "$VM_MDL" && ! -f "$NEST/lostcoast/viewmodel.pk3" ]]; then
	vm_tmp="$(mktemp -d)"
	mkdir -p "$vm_tmp/gc_studio" "$vm_tmp/models"
	cp -f "$VM_MDL" "$vm_tmp/gc_studio/v_357.mdl"
	cp -f "$VM_MDL" "$vm_tmp/models/v_357.mdl"
	( cd "$vm_tmp" && zip -0 -r "$NEST/lostcoast/viewmodel.pk3" gc_studio models )
	rm -rf "$vm_tmp"
	echo "  + nest viewmodel.pk3 (sidecar; extras is the Flipper path)"
fi
rm -rf "$NEST/lostcoast/models"
# Ensure title bake lands in the nest even if stage was reused without it.
TITLE_MENU="${LCGC_CACHE}/title/gc_menu"
if [[ ! -f "$TITLE_MENU/background.tga" ]]; then
	bash "$SCRIPT_DIR/bake-lostcoast-title.sh" "$TITLE_MENU"
fi
mkdir -p "$NEST/lostcoast/resource/gc_menu"
cp -a "$TITLE_MENU/." "$NEST/lostcoast/resource/gc_menu/"

echo "==> packaging $LCGC_ISO"
# Retail packaging re-bakes CapFaces/WT from the GoldSrc stub BSP (→ 0 faces).
# We still use it for hybrid DOL+ISO9660 layout, then reinject Source→GX packs.

python3 "$LCGC_ENGINE/scripts/build-gamecube-disc.py" \
	--dol "$LCGC_DOL" \
	--data "$NEST/lostcoast" \
	--gamedir lostcoast \
	--no-auto-extra-gamedirs \
	--output "$LCGC_ISO"

# Verify single-session ISO kept host-prebaked packs (disc builder preserves them).
python3 - <<'PY'
from pathlib import Path
import os
import subprocess
import tempfile

iso = Path(os.environ["LCGC_ISO"])
vtd = Path(tempfile.mkdtemp(prefix="lcgc-iso-verify-"))
try:
    checks = {
        "lostcoast/gamecube.cfg": 20,
        "lostcoast/media/cf_d2_lostcoast.gcp": 40_000,
        "lostcoast/media/wt_d2_lostcoast.gct": 100_000,
        "lostcoast/media/vis_d2_lostcoast.gcv": 10_000,
        "lostcoast/media/eye_d2_lostcoast.gce": 16,
        "lostcoast/media/vst_d2_lostcoast_path.bin": 200,
        "lostcoast/media/cf_d2_lostcoast.cls": 100,
        "lostcoast/media/cf_d2_lostcoast_cls.pak": 1000,
        "lostcoast/resource/gc_menu/background.tga": 8_000,
        "lostcoast/resource/gc_menu/SOURCE.txt": 20,
    }
    for side in ("rt", "bk", "lf", "ft", "up", "dn"):
        checks[f"lostcoast/media/sky_sky_lostcoast_hdr{side}.cmp"] = 8192
    for rel, min_sz in checks.items():
        dest = vtd / Path(rel).name
        subprocess.check_call(
            ["xorriso", "-indev", str(iso), "-osirrox", "on", "-extract", f"/xash3d/{rel}", str(dest)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        sz = dest.stat().st_size
        print(f"verify /xash3d/{rel}: {sz} bytes")
        if sz < min_sz:
            raise SystemExit(f"ISO pack too small: /xash3d/{rel} ({sz} < {min_sz})")
    cfg = (vtd / "gamecube.cfg").read_text()
    print(cfg)
    if "gccapfacessmoke" not in cfg:
        raise SystemExit("gamecube.cfg missing gccapfacessmoke")
    if (vtd / "cf_d2_lostcoast.gcp").read_bytes()[:4] != b"GCcf":
        raise SystemExit("bad CapFaces magic")
    if (vtd / "wt_d2_lostcoast.gct").read_bytes()[:4] != b"GCWT":
        raise SystemExit("bad GCWT magic")
    src = (vtd / "SOURCE.txt").read_text()
    if "Lost Coast" not in src:
        raise SystemExit("gc_menu SOURCE.txt is not Lost Coast retail title bake")
    print(src.strip())
    for side in ("rt", "bk", "lf", "ft", "up", "dn"):
        sky = vtd / f"sky_sky_lostcoast_hdr{side}.cmp"
        if sky.stat().st_size != 8192:
            raise SystemExit(f"sky side {side} size {sky.stat().st_size} != 8192")
    print("skybox: 6 x 128 CMPR (sky_sky_lostcoast_hdr*)")
finally:
    for p in vtd.iterdir():
        p.unlink(missing_ok=True)
    vtd.rmdir()
PY

echo "ISO → $LCGC_ISO ($(du -h "$LCGC_ISO" | awk '{print $1}'))"
