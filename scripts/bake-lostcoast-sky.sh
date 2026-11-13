#!/usr/bin/env bash
# Bake Source skybox (sky_lostcoast_hdr) → media/sky_sky_lostcoast_hdr*.cmp for CapFaces.
# Engine path: media/sky_<skyname><side>.cmp with skyname=sky_lostcoast_hdr.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
lcgc_require_content

OUT="${1:-$LCGC_CACHE/bake/media}"
SKYNAME="${2:-sky_lostcoast_hdr}"
mkdir -p "$OUT"

python3 - <<PY
from pathlib import Path
import struct
import sys
import numpy as np
from PIL import Image

ROOT = Path("$LCGC_ROOT")
OUT = Path("$OUT")
SKYNAME = "$SKYNAME"
sys.path.insert(0, str(ROOT / "tools"))
from vpk import open_vpk_roots, read_first
from vtf import decode_vtf

# Same CMPR encoder as menu/title bake (GX DXT1 layout).
def gx_cmpr_encode(rgb: np.ndarray) -> bytes:
	h, w, _ = rgb.shape
	px = rgb.astype(np.int32)
	blocks = px.reshape(h // 4, 4, w // 4, 4, 3).transpose(0, 2, 1, 3, 4)
	flat = blocks.reshape(h // 4, w // 4, 16, 3)
	lum = flat @ np.array([299, 587, 114])
	hi = np.take_along_axis(flat, lum.argmax(-1)[..., None, None], 2)[..., 0, :]
	lo = np.take_along_axis(flat, lum.argmin(-1)[..., None, None], 2)[..., 0, :]

	def to565(c):
		return ((c[..., 0] >> 3) << 11) | ((c[..., 1] >> 2) << 5) | (c[..., 2] >> 3)

	def from565(v):
		r = (v >> 11) & 31
		g = (v >> 5) & 63
		b = v & 31
		return np.stack(((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)), -1)

	c0, c1 = to565(hi), to565(lo)
	swap = c0 < c1
	c0, c1 = np.where(swap, c1, c0), np.where(swap, c0, c1)
	e0, e1 = from565(c0), from565(c1)
	pal = np.stack((e0, e1, (2 * e0 + e1) // 3, (e0 + 2 * e1) // 3), 2)
	dist = ((flat[:, :, :, None, :] - pal[:, :, None, :, :]) ** 2).sum(-1)
	idx = dist.argmin(-1)
	idx = np.where((c0 == c1)[..., None], 0, idx)
	rows = idx.reshape(h // 4, w // 4, 4, 4)
	bits = (rows[..., 0] << 6) | (rows[..., 1] << 4) | (rows[..., 2] << 2) | rows[..., 3]
	outb = bytearray()
	for ty in range(0, h // 4, 2):
		for tx in range(0, w // 4, 2):
			for by, bx in ((ty, tx), (ty, tx + 1), (ty + 1, tx), (ty + 1, tx + 1)):
				outb += struct.pack(">HH", int(c0[by, bx]), int(c1[by, bx]))
				outb += bytes(int(b) for b in bits[by, bx])
	return bytes(outb)

vpks = open_vpk_roots([
	ROOT / "content/lostcoast/lostcoast_pak_dir.vpk",
	ROOT / "content/hl2/hl2_textures_dir.vpk",
])
sides = ("rt", "bk", "lf", "ft", "up", "dn")
SIZE = 128
OUT.mkdir(parents=True, exist_ok=True)

# Prefer HDR VTFs; fall back to LDR sky_lostcoast* if HDR decode fails.
src_note = []
preview = Image.new("RGB", (SIZE * 6, SIZE))
for i, side in enumerate(sides):
	img = None
	used = None
	for stem in (f"sky_lostcoast_hdr{side}", f"sky_lostcoast{side}"):
		raw = read_first(vpks, f"materials/skybox/{stem}.vtf")
		if not raw:
			continue
		try:
			vtf = decode_vtf(raw)
			img = Image.frombytes("RGBA", (vtf.width, vtf.height), vtf.rgba).convert("RGB")
			used = stem
			break
		except Exception as e:
			src_note.append(f"{stem}: {e}")
			continue
	if img is None:
		raise SystemExit(f"missing/undecodable sky side {side}")
	img = img.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
	preview.paste(img, (i * SIZE, 0))
	cmpr = gx_cmpr_encode(np.asarray(img))
	assert len(cmpr) == SIZE * SIZE // 2, len(cmpr)
	# Engine: media/sky_<skyname><side>.cmp  (skyname already starts with sky_)
	dest = OUT / f"sky_{SKYNAME}{side}.cmp"
	dest.write_bytes(cmpr)
	print(f"  {dest.name}: {len(cmpr)} bytes from materials/skybox/{used}.vtf")

preview_path = ROOT / "cache/title/sky-lostcoast-preview.png"
preview_path.parent.mkdir(parents=True, exist_ok=True)
preview.save(preview_path)
(OUT / "SKY_SOURCE.txt").write_text(
	f"Lost Coast Source skybox bake\nskyname={SKYNAME}\n"
	f"engine_path=media/sky_{SKYNAME}<side>.cmp\n"
	f"size={SIZE}x{SIZE} GX CMPR\n"
)
print(f"baked 6 sky sides → {OUT} (preview {preview_path})")
PY

echo "sky → $OUT"
ls -lh "$OUT"/sky_"${SKYNAME}"*.cmp
