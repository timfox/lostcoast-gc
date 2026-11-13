#!/usr/bin/env bash
# Bake retail Lost Coast console title art → resource/gc_menu for GX menu.
# Source: materials/console/background01*.vtf from Steam lostcoast VPK (read-only).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
lcgc_require_content

OUT="${1:-$LCGC_CACHE/title/gc_menu}"
mkdir -p "$OUT"

python3 - <<PY
from pathlib import Path
import struct
import sys
import numpy as np
from PIL import Image

ROOT = Path("$LCGC_ROOT")
sys.path.insert(0, str(ROOT / "tools"))
from vpk import open_vpk_roots, read_first
from vtf import decode_vtf

# Inline GX CMPR encode (same layout as build-gamecube-disc.gx_cmpr_encode).
# Avoid exec_module of the full disc builder (dataclass import edge cases).
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

out = Path("$OUT")
out.mkdir(parents=True, exist_ok=True)
vpks = open_vpk_roots([
	ROOT / "content/lostcoast/lostcoast_pak_dir.vpk",
	ROOT / "content/hl2/hl2_textures_dir.vpk",
])

def load_vtf(name: str) -> Image.Image:
	data = read_first(vpks, name)
	if not data:
		raise SystemExit(f"missing {name} in Lost Coast/HL2 VPKs")
	img = decode_vtf(data)
	return Image.frombytes("RGBA", (img.width, img.height), img.rgba)

try:
	bg = load_vtf("materials/console/background01_widescreen.vtf")
	src_name = "background01_widescreen"
except SystemExit:
	bg = load_vtf("materials/console/background01.vtf")
	src_name = "background01"

hires = Image.new("RGB", (640, 480), (12, 14, 18))
fitted = bg.convert("RGB")
fitted.thumbnail((640, 480), Image.Resampling.LANCZOS)
ox = (640 - fitted.size[0]) // 2
oy = (480 - fitted.size[1]) // 2
hires.paste(fitted, (ox, oy))
soft = hires.resize((320, 240), Image.Resampling.LANCZOS)

soft.save(out / "background.tga", format="TGA")
soft.save(out / "background.bmp", format="BMP")
soft.save(out / "loading.tga", format="TGA")
soft.save(out / "intro.tga", format="TGA")

tile_w, tile_h = 160, 120
layout = [f"resolution\t320\t240", ""]
for row, y in enumerate((0, tile_h)):
	for col, x in enumerate((0, tile_w)):
		tile = soft.crop((x, y, x + tile_w, y + tile_h))
		name = f"bg_{row}_{col}.tga"
		tile.save(out / name, format="TGA")
		layout.append(f"resource/gc_menu/{name}\tfit\t{x}\t{y}")
(out / "BackgroundLayout.txt").write_text("\n".join(layout) + "\n")

cw, ch = 320, 240
for row, y in enumerate((0, ch)):
	for col, x in enumerate((0, cw)):
		quad = np.asarray(hires.crop((x, y, x + cw, y + ch)))
		header = b"GCMP" + struct.pack(">HH", cw, ch) + bytes(24)
		(out / f"bg_{row}_{col}.cmp").write_bytes(header + gx_cmpr_encode(quad))

try:
	ch_img = load_vtf("materials/gamepadui/chapter1.vtf")
	ch_img.convert("RGB").resize((230, 40), Image.Resampling.LANCZOS).save(
		out / "head_newgame.tga", format="TGA"
	)
except SystemExit:
	pass

# Side-by-side reference for evidence (retail vs baked soft plate).
ref = soft.copy()
ref.save(ROOT / "cache/title/baked-gc-background.png", format="PNG")

(out / "SOURCE.txt").write_text(
	f"Lost Coast retail title bake\nsource={src_name}\n"
	f"vtf=materials/console/{src_name}.vtf\n"
)
print(f"baked retail Lost Coast title → {out} (from {src_name})")
PY

echo "title menu → $OUT"
ls -lh "$OUT" | head -30
