#!/usr/bin/env bash
# Prove Lost Coast title plate matches retail with CapFaces map loaded.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

ISO="${1:-$LCGC_ISO}"
[[ -f "$ISO" ]] || { echo "error: missing ISO $ISO" >&2; exit 1; }

STAMP="$(date +%Y%m%d-%H%M%S)"
NOTE="$LCGC_NOTES/title-proof-$STAMP"
mkdir -p "$NOTE" "$LCGC_SHOTS"
export LCGC_ISO="$ISO"
export LCGC_TITLE_NOTE="$NOTE"

python3 - <<'PY'
from pathlib import Path
import os, subprocess, tempfile, sys
import numpy as np
from PIL import Image

iso = Path(os.environ["LCGC_ISO"])
root = Path(os.environ["LCGC_ROOT"])
note = Path(os.environ["LCGC_TITLE_NOTE"])
shots = Path(os.environ["LCGC_SHOTS"])
sys.path.insert(0, str(root / "tools"))
from vpk import open_vpk_roots, read_first
from vtf import decode_vtf

vtd = Path(tempfile.mkdtemp(prefix="lcgc-title-"))
try:
	for rel in (
		"lostcoast/resource/gc_menu/background.tga",
		"lostcoast/resource/gc_menu/SOURCE.txt",
		"lostcoast/media/cf_d2_lostcoast.gcp",
		"lostcoast/gamecube.cfg",
	):
		dest = vtd / Path(rel).name
		subprocess.check_call(
			["xorriso", "-indev", str(iso), "-osirrox", "on", "-extract", f"/xash3d/{rel}", str(dest)],
			stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
		)
	src = (vtd / "SOURCE.txt").read_text()
	cfg = (vtd / "gamecube.cfg").read_text()
	cf = (vtd / "cf_d2_lostcoast.gcp").read_bytes()
	bg = Image.open(vtd / "background.tga").convert("RGB")
	if "Lost Coast" not in src:
		raise SystemExit("ISO gc_menu is not Lost Coast retail title")
	if "gccapfacessmoke" not in cfg or "d2_lostcoast" not in cfg:
		raise SystemExit("gamecube.cfg missing CapFaces coast boot")
	if cf[:4] != b"GCcf" or len(cf) < 40_000:
		raise SystemExit("CapFaces pack missing/corrupt on ISO")

	vpks = open_vpk_roots([
		root / "content/lostcoast/lostcoast_pak_dir.vpk",
		root / "content/hl2/hl2_textures_dir.vpk",
	])
	data = read_first(vpks, "materials/console/background01_widescreen.vtf")
	if not data:
		data = read_first(vpks, "materials/console/background01.vtf")
	img = decode_vtf(data)
	retail = Image.frombytes("RGBA", (img.width, img.height), img.rgba).convert("RGB")
	hires = Image.new("RGB", (640, 480), (12, 14, 18))
	fitted = retail.copy()
	fitted.thumbnail((640, 480), Image.Resampling.LANCZOS)
	hires.paste(fitted, ((640 - fitted.size[0]) // 2, (480 - fitted.size[1]) // 2))
	soft = hires.resize((320, 240), Image.Resampling.LANCZOS)
	a = np.asarray(soft, np.float32)
	b = np.asarray(bg.resize((320, 240), Image.Resampling.LANCZOS), np.float32)
	mae = float(np.mean(np.abs(a - b)))
	corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
	side = Image.new("RGB", (640, 240))
	side.paste(soft, (0, 0))
	side.paste(Image.fromarray(b.astype(np.uint8)), (320, 0))
	side_path = note / "title-retail-vs-iso.png"
	side.save(side_path)
	side.save(shots / "lostcoast-title-retail-vs-iso.png")
	summary = note / "EVIDENCE.md"
	summary.write_text(
		"# Lost Coast title proof\n\n"
		f"- ISO: `{iso}`\n"
		f"- gc_menu SOURCE:\n```\n{src.strip()}\n```\n"
		f"- CapFaces pack: {len(cf)} bytes magic={cf[:4]!r}\n"
		f"- gamecube.cfg:\n```\n{cfg.strip()}\n```\n"
		f"- retail→ISO soft plate mae={mae:.4f} corr={corr:.6f}\n"
		f"- side-by-side: `{side_path}`\n"
	)
	print(summary.read_text())
	if mae > 1.0 or corr < 0.99:
		raise SystemExit(f"title mismatch mae={mae} corr={corr}")
	print("TITLE_ASSET: PASS")
finally:
	for p in vtd.iterdir():
		p.unlink(missing_ok=True)
	vtd.rmdir()
PY

if bash "$SCRIPT_DIR/probe-capfaces-smoke.sh" "$ISO"; then
	PASS_CF=1
else
	PASS_CF=0
fi
LATEST="$(ls -1dt "$LCGC_NOTES"/dolphin-probe-* 2>/dev/null | head -1 || true)"
TITLE_MARK=0
if [[ -n "$LATEST" ]]; then
	cp -a "$LATEST" "$NOTE/dolphin" 2>/dev/null || true
	if grep -Eiq 'Lost Coast retail title plaque present|title plaque present map=' \
		"$LATEST"/dolphin.log "$LATEST"/stderr.log 2>/dev/null; then
		TITLE_MARK=1
	fi
	{
		echo ""
		echo "## Dolphin"
		echo "- probe=$LATEST"
		echo "- capfaces_pass=$PASS_CF"
		echo "- title_plaque_osreport=$TITLE_MARK"
		grep -Eih 'Lost Coast retail title|CapFaces pack loaded|MAP_READY|gccapfacessmoke' \
			"$LATEST"/dolphin.log "$LATEST"/stderr.log 2>/dev/null | tail -30 || true
	} >> "$NOTE/EVIDENCE.md"
fi

if (( PASS_CF )) && (( TITLE_MARK )); then
	echo "TITLE_PROOF: PASS ($NOTE)"
	exit 0
fi
if (( PASS_CF )) && (( ! TITLE_MARK )); then
	echo "TITLE_PROOF: PARTIAL — CapFaces OK, title plaque OSReport missing (rebuild DOL?) ($NOTE)"
	exit 2
fi
echo "TITLE_PROOF: FAIL ($NOTE)"
exit 1
