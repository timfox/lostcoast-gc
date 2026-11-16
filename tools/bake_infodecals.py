"""Source `infodecal` entities -> one decal atlas texture plus a placement table.

VBSP leaves infodecal out of the BSP: the engine projects each decal onto the
nearest surface when the map loads. The world pack is full (one near slot per
unique material), so the 23 decal textures of Lost Coast are packed into ONE
small atlas that goes through the pack as a single material, and each decal is
baked as an alpha-tested quad on its host face (bake_worlddl.py).

World size = texture pixels x $decalscale (the decal VMT). Orientation is the
Source one for a static decal: image up is world up projected onto the host
plane, so graffiti and ivy hang upright on walls; on level ground the image
axes follow world +X.

Writes <out>/infodecals.json: atlas size, the tile rectangle of every texture
(u0, v0, u1, v1 in 0..1) and its world size, and the decal list (origin,
texture) read from the entity lump.
"""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import numpy as np

from vmt import load_vmt
from vtf import decode_vtf

ATLAS = 128  # one near texture slot (128x128 CMPR chain)
ATLAS_NAME = "lostcoast/decals/atlas"


def parse_decals(bsp_bytes: bytes) -> list[tuple[tuple[float, float, float], str]]:
	ofs, ln = struct.unpack_from("<ii", bsp_bytes, 8)
	ents = bsp_bytes[ofs:ofs + ln].decode("latin1")
	out = []
	for blk in re.findall(r"\{([^}]*)\}", ents):
		kv = dict(re.findall(r'"([^"]+)" "([^"]*)"', blk))
		if kv.get("classname") == "infodecal" and "origin" in kv and "texture" in kv:
			org = tuple(float(x) for x in kv["origin"].split())
			out.append((org, kv["texture"].replace("\\", "/").lower()))
	return out


def _resize(rgba: np.ndarray, w: int, h: int) -> np.ndarray:
	"""Area average on premultiplied colour, so transparent texels do not
	darken the edge of the paint."""
	sh, sw = rgba.shape[:2]
	f = rgba.astype(np.float64)
	a = f[..., 3:4] / 255.0
	pm = np.concatenate([f[..., :3] * a, f[..., 3:4]], axis=2)
	ys = np.linspace(0, sh, h + 1).astype(int)
	xs = np.linspace(0, sw, w + 1).astype(int)
	out = np.zeros((h, w, 4))
	for j in range(h):
		for i in range(w):
			blk = pm[ys[j]:max(ys[j + 1], ys[j] + 1), xs[i]:max(xs[i + 1], xs[i] + 1)]
			out[j, i] = blk.reshape(-1, 4).mean(axis=0)
	alpha = out[..., 3:4]
	col = np.where(alpha > 0.5, out[..., :3] / np.maximum(alpha / 255.0, 1e-3), 0.0)
	return np.clip(np.concatenate([col, alpha], axis=2), 0, 255).astype(np.uint8)


def _shelf_pack(sizes: list[tuple[str, int, int]], pad: int):
	"""Tallest first, left to right in rows. Returns {name: (x, y)} or None."""
	x = y = row_h = 0
	pos = {}
	for name, w, h in sorted(sizes, key=lambda s: (-s[2], -s[1], s[0])):
		if x + w + pad > ATLAS:
			x, y, row_h = 0, y + row_h + pad, 0
		if y + h + pad > ATLAS:
			return None
		pos[name] = (x, y)
		x += w + pad
		row_h = max(row_h, h)
	return pos


def build(bsp_bytes: bytes, vpks, read_first) -> tuple[dict, np.ndarray] | None:
	decals = parse_decals(bsp_bytes)
	if not decals:
		return None
	tex = {}
	for _org, name in decals:
		if name in tex:
			continue
		vm = read_first(vpks, f"materials/{name}.vmt")
		if vm is None:
			continue
		m = load_vmt(vm)
		base = m.get("$basetexture", name).replace("\\", "/").lower()
		raw = read_first(vpks, f"materials/{base}.vtf")
		if raw is None:
			continue
		img = decode_vtf(raw, max_side=256)
		scale = float(m.get("$decalscale", "1") or 1)
		arr = np.frombuffer(img.rgba, np.uint8).reshape(img.height, img.width, 4)
		tex[name] = (arr, img.width * scale, img.height * scale)
	if not tex:
		return None
	# Largest uniform shrink that fits every tile (1 px border each).
	pos = None
	for denom in range(5, 40):
		sizes = [(n, max(4, t[0].shape[1] // denom), max(4, t[0].shape[0] // denom)) for n, t in tex.items()]
		pos = _shelf_pack(sizes, 2)
		if pos:
			break
	if not pos:
		return None
	atlas = np.zeros((ATLAS, ATLAS, 4), np.uint8)
	table = {}
	for n, w, h in sizes:
		x, y = pos[n]
		atlas[y:y + h, x:x + w] = _resize(tex[n][0], w, h)
		# Inset half a texel so bilinear never reaches the neighbour.
		table[n] = {
			"uv": [(x + 0.5) / ATLAS, (y + 0.5) / ATLAS, (x + w - 0.5) / ATLAS, (y + h - 0.5) / ATLAS],
			"size": [tex[n][1], tex[n][2]],
		}
	info = {
		"atlas": ATLAS,
		"name": ATLAS_NAME,
		"tiles": table,
		"decals": [{"origin": list(o), "texture": t} for o, t in decals if t in table],
	}
	return info, atlas


def write_info(out_dir: Path, info: dict) -> None:
	(out_dir / "infodecals.json").write_text(json.dumps(info, indent=1) + "\n")
