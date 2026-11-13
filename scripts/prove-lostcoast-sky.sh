#!/usr/bin/env bash
# Prove Source sky_lostcoast_hdr is on the ISO and CapFaces loads it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

ISO="${1:-$LCGC_ISO}"
[[ -f "$ISO" ]] || { echo "error: missing ISO $ISO" >&2; exit 1; }

STAMP="$(date +%Y%m%d-%H%M%S)"
NOTE="$LCGC_NOTES/sky-proof-$STAMP"
mkdir -p "$NOTE" "$LCGC_SHOTS"
export LCGC_ISO="$ISO"
export LCGC_SKY_NOTE="$NOTE"

python3 - <<'PY'
from pathlib import Path
import os, subprocess, tempfile, sys
import numpy as np
from PIL import Image

iso = Path(os.environ["LCGC_ISO"])
root = Path(os.environ["LCGC_ROOT"])
note = Path(os.environ["LCGC_SKY_NOTE"])
shots = Path(os.environ["LCGC_SHOTS"])
sys.path.insert(0, str(root / "tools"))

sides = ("rt", "bk", "lf", "ft", "up", "dn")
vtd = Path(tempfile.mkdtemp(prefix="lcgc-sky-"))
try:
	for side in sides:
		rel = f"lostcoast/media/sky_sky_lostcoast_hdr{side}.cmp"
		dest = vtd / Path(rel).name
		subprocess.check_call(
			["xorriso", "-indev", str(iso), "-osirrox", "on", "-extract", f"/xash3d/{rel}", str(dest)],
			stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
		)
	subprocess.check_call(
		["xorriso", "-indev", str(iso), "-osirrox", "on", "-extract",
		 "/xash3d/lostcoast/gamecube.cfg", str(vtd / "gamecube.cfg")],
		stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
	)
	cfg = (vtd / "gamecube.cfg").read_text()
	if "gccapfacessmoke" not in cfg or "d2_lostcoast" not in cfg:
		raise SystemExit("gamecube.cfg missing CapFaces coast boot")

	rows = []
	preview = Image.new("RGB", (128 * 6, 128))
	# Minimal GX CMPR→RGB for proof (decode endpoints only as sanity; prefer bake preview).
	bake_prev = root / "cache/title/sky-lostcoast-preview.png"
	if bake_prev.is_file():
		prev = Image.open(bake_prev).convert("RGB")
		prev.save(note / "sky-source-preview.png")
		prev.save(shots / "lostcoast-sky-source-preview.png")

	for i, side in enumerate(sides):
		p = vtd / f"sky_sky_lostcoast_hdr{side}.cmp"
		raw = p.read_bytes()
		if len(raw) != 8192:
			raise SystemExit(f"{p.name}: expected 8192, got {len(raw)}")
		nz = sum(1 for b in raw if b)
		if nz < 1000:
			raise SystemExit(f"{p.name}: too sparse ({nz} nonzero) — likely black/empty bake")
		rows.append(f"- `{p.name}`: {len(raw)} bytes, nonzero={nz}")

	summary = note / "EVIDENCE.md"
	summary.write_text(
		"# Lost Coast skybox proof\n\n"
		f"- ISO: `{iso}`\n"
		f"- skyname: `sky_lostcoast_hdr` → `media/sky_sky_lostcoast_hdr*.cmp`\n"
		f"- gamecube.cfg:\n```\n{cfg.strip()}\n```\n"
		+ "\n".join(rows) + "\n"
		+ f"- source preview: `{note / 'sky-source-preview.png'}`\n"
	)
	print(summary.read_text())
	print("SKY_ASSET: PASS")
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
SKY_MARK=0
SKY_MISS=0
if [[ -n "$LATEST" ]]; then
	cp -a "$LATEST" "$NOTE/dolphin" 2>/dev/null || true
	if grep -Eiq 'sky sky_lostcoast_hdr loaded \(6 x 128x128 CMPR\)|sky sky_lostcoast_hdr loaded' \
		"$LATEST"/dolphin.log "$LATEST"/stderr.log 2>/dev/null; then
		SKY_MARK=1
	fi
	if grep -Eiq 'sky sky_lostcoast_hdr missing|missing media/sky_sky_lostcoast_hdr' \
		"$LATEST"/dolphin.log "$LATEST"/stderr.log 2>/dev/null; then
		SKY_MISS=1
	fi
	{
		echo ""
		echo "## Dolphin"
		echo "- probe=$LATEST"
		echo "- capfaces_pass=$PASS_CF"
		echo "- sky_loaded_osreport=$SKY_MARK"
		echo "- sky_missing_osreport=$SKY_MISS"
		grep -Eih 'sky sky_lostcoast|skyname|CapFaces pack loaded|MAP_READY|PrefetchMapSky|gcworldrender' \
			"$LATEST"/dolphin.log "$LATEST"/stderr.log 2>/dev/null | tail -40 || true
	} >> "$NOTE/EVIDENCE.md"
fi

if (( PASS_CF )) && (( SKY_MARK )) && (( ! SKY_MISS )); then
	echo "SKY_PROOF: PASS ($NOTE)"
	exit 0
fi
if (( PASS_CF )) && (( ! SKY_MARK )); then
	echo "SKY_PROOF: PARTIAL — CapFaces OK, sky loaded OSReport missing ($NOTE)"
	exit 2
fi
echo "SKY_PROOF: FAIL ($NOTE)"
exit 1
