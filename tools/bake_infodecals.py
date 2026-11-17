"""Source `infodecal` entities -> decal atlas textures plus a placement table.

VBSP leaves infodecal out of the BSP: the engine projects each decal onto the
nearest surface when the map loads. Here each decal is baked as an alpha-tested
quad on its host face (bake_worlddl.py), and the 23 decal textures of Lost
Coast are packed into a few 128x128 atlases that go through the world pack as
ordinary materials.

World size = the VTF's own pixel size x $decalscale (the decal VMT).
Orientation is the Source one for a static decal: image up is world up
projected onto the host plane, so graffiti and ivy hang upright on walls; on
level ground the image axes follow world +X.

Memory compromise (not a look choice): N_ATLAS atlases. Every texture gets the
same texels per world unit, the largest that fits, up to one whole atlas for
a texture. Textures are put into atlases in order along the map, so one view
binds few of them.

Writes <out>/infodecals.json: atlas size and names, the tile of every texture
(atlas index, u0, v0, u1, v1 in 0..1) and its world size, and the decal list
(origin, texture) read from the entity lump.
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
N_ATLAS = 6
ATLAS_NAME = "lostcoast/decals/atlas"  # + index


def atlas_name(i: int) -> str:
	return f"{ATLAS_NAME}{i}"


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


def _tile_px(world: float, density: float, src: int) -> int:
	# A texture cannot be larger than its source or than one atlas (the two
	# 256-unit ivy sheets take a whole atlas each and stop there).
	return int(max(4, min(src, ATLAS - PAD, round(world * density))))


PAD = 2  # transparent texels between tiles


def _place(free: list, w: int, h: int):
	"""Guillotine packing: take the free rectangle the tile fits most tightly,
	split what is left along the shorter leftover. Returns (x, y) or None."""
	best = None
	for k, (fx, fy, fw, fh) in enumerate(free):
		if w <= fw and h <= fh:
			score = min(fw - w, fh - h)
			if best is None or score < best[0]:
				best = (score, k)
	if best is None:
		return None
	fx, fy, fw, fh = free.pop(best[1])
	if fw - w < fh - h:
		free.append((fx + w, fy, fw - w, h))
		free.append((fx, fy + h, fw, fh - h))
	else:
		free.append((fx, fy + h, w, fh - h))
		free.append((fx + w, fy, fw - w, fh))
	return fx, fy


def _partition(order, tex, density: float):
	"""Largest first, each into the first atlas with room (ties in map order,
	so neighbours tend to share an atlas). Returns [(sizes, pos)] or None when
	the textures need more than N_ATLAS at this density."""
	items = []
	for k, n in enumerate(order):
		arr, ww, wh = tex[n][:3]
		items.append((n, _tile_px(ww, density, arr.shape[1]), _tile_px(wh, density, arr.shape[0]), k))
	bins = []
	for n, w, h, _k in sorted(items, key=lambda it: (-it[1] * it[2], it[3])):
		for sizes, pos, free in bins:
			xy = _place(free, w + PAD, h + PAD)
			if xy:
				break
		else:
			if len(bins) == N_ATLAS:
				return None
			sizes, pos, free = [], {}, [(0, 0, ATLAS, ATLAS)]
			bins.append((sizes, pos, free))
			xy = _place(free, w + PAD, h + PAD)
			if xy is None:
				return None
		sizes.append((n, w, h))
		pos[n] = xy
	return [(sizes, pos) for sizes, pos, _free in bins]


def build(bsp_bytes: bytes, vpks, read_first) -> tuple[dict, list[np.ndarray]] | None:
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
		# The decal's world size comes from the texture's full size, not from
		# the mip decoded here.
		full_w, full_h = struct.unpack_from("<HH", raw, 16)
		img = decode_vtf(raw, max_side=256)
		scale = float(m.get("$decalscale", "1") or 1)
		arr = np.frombuffer(img.rgba, np.uint8).reshape(img.height, img.width, 4)
		tex[name] = (arr, full_w * scale, full_h * scale)
	if not tex:
		return None
	# Map order: along the longest axis of the decals' spread.
	org = {n: np.mean([o for o, t in decals if t == n], axis=0) for n in tex}
	pts = np.array([org[n] for n in tex])
	axis = int(np.argmax(pts.max(axis=0) - pts.min(axis=0)))
	order = sorted(tex, key=lambda n: (org[n][axis], n))
	groups = None
	density = 2.0
	while density > 0.05:
		groups = _partition(order, tex, density)
		if groups:
			break
		density *= 0.98
	if not groups:
		return None
	atlases = []
	table = {}
	for ai, (sizes, pos) in enumerate(groups):
		atlas = np.zeros((ATLAS, ATLAS, 4), np.uint8)
		for n, w, h in sizes:
			x, y = pos[n]
			atlas[y:y + h, x:x + w] = _resize(tex[n][0], w, h)
			# Inset half a texel so bilinear never reaches the neighbour.
			table[n] = {
				"atlas": ai,
				"uv": [(x + 0.5) / ATLAS, (y + 0.5) / ATLAS, (x + w - 0.5) / ATLAS, (y + h - 0.5) / ATLAS],
				"size": [tex[n][1], tex[n][2]],
			}
		atlases.append(atlas)
	info = {
		"atlas": ATLAS,
		"names": [atlas_name(i) for i in range(len(atlases))],
		"density": density,
		"tiles": table,
		"decals": [{"origin": list(o), "texture": t} for o, t in decals if t in table],
	}
	return info, atlases


def write_info(out_dir: Path, info: dict) -> None:
	(out_dir / "infodecals.json").write_text(json.dumps(info, indent=1) + "\n")
