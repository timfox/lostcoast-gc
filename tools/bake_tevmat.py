#!/usr/bin/env python3
"""Source material features → Flipper TEV material table (media/tevmat.gcm).

Each Source VMT is classified into TEV feature flags that the runtime
(engine ref/gx/r_gx_worlddl.c) turns into extra TEV stages after the stock
diffuse × lightmap stages. CapFaces face byte 127 / world DL mat.tevmat
carries the material id.

Ported so far:
  $detail / $detailscale / $detailblendfactor (blend mode 0, mod2x)
      → shared 64² I8 detail textures with mips, one TEV stage, UV = diffuse
        ST × scale via a texture matrix (no extra vertex data).
  $alphatest / $alphatestreference
      → CMPR 1-bit alpha + GX alpha compare.
  $bumpmap (materials without an authored $detail)
      → the normal map is lit once at bake time by the map's sun and stored
        as a native-resolution, tileable 64² relief tile that rides the same
        mod2x detail stage (×8 tiling). Sharp sun-lit grain on sand and rock
        for one TEV stage and 5.4 KB, instead of per-pixel normal mapping.
  $envmap + $basealphaenvmapmask (metal / plaster / monastery frescos)
      → shared 64² I4 masks from the base texture alpha (CMPR alpha is 1-bit,
        so the mask cannot ride in the diffuse). Runtime adds sun × tint ×
        mask on top of lit colour — no sand IndTex warp.

See docs/TEV_MATERIALS.md for the full Source→TEV mapping and what is next.
"""
from __future__ import annotations

import math
import re
import struct
from pathlib import Path

GC_TEVMAT_MAGIC = b"GCmt"
GC_TEVMAT_VERSION = 1
GC_TEVMAT_MAX_MATS = 96
GC_TEVMAT_MAX_DETAIL = 8
GC_TEVMAT_REAL_DETAIL = 4  # slots kept for authored $detail textures
GC_TEVMAT_MAX_ENVMASK = 8  # shared I4 masks for $basealphaenvmapmask
GC_TEVMAT_DETAIL_SIDE = 64
GC_TEVMAT_DETAIL_BYTES = 4096 + 1024 + 256 + 64
GC_TEVMAT_ENVMASK_BYTES = 2048  # 64² GX I4
GC_TEVMAT_ENVMASK_MAGIC = b"GCem"

F_DETAIL = 0x01
F_ALPHATEST = 0x02
F_ENVMASK = 0x04


def _num(val: str | None, default: float) -> float:
	if not val:
		return default
	m = re.search(r"[-+]?\d*\.?\d+", val)
	return float(m.group(0)) if m else default


def _tex_path(val: str) -> str:
	v = val.replace("\\", "/").lower().strip()
	if v.startswith("materials/"):
		v = v[len("materials/") :]
	if v.endswith(".vtf"):
		v = v[: -len(".vtf")]
	return v


def classify(vmt: dict[str, str] | None) -> dict | None:
	"""Return TEV features for one VMT, or None when it is plain diffuse × LM."""
	if not vmt:
		return None
	out: dict = {"flags": 0}
	detail = (vmt.get("$detail") or "").strip()
	mode = int(_num(vmt.get("$detailblendmode"), 0.0))
	if detail and mode == 0:
		out["flags"] |= F_DETAIL
		out["detail"] = _tex_path(detail)
		out["detail_scale"] = max(0.05, min(255.0, _num(vmt.get("$detailscale"), 4.0)))
		out["detail_factor"] = max(0.0, min(1.0, _num(vmt.get("$detailblendfactor"), 1.0)))
	bump = (vmt.get("$bumpmap") or "").strip()
	if bump and not out["flags"] & F_DETAIL:
		out["flags"] |= F_DETAIL
		out["detail"] = BUMP_PREFIX + _tex_path(bump)
		out["detail_scale"] = float(BUMP_TILING)
		out["detail_factor"] = BUMP_FACTOR
	if _num(vmt.get("$alphatest"), 0.0) >= 1.0:
		out["flags"] |= F_ALPHATEST
		out["alpharef"] = int(max(1, min(255, round(_num(vmt.get("$alphatestreference"), 0.5) * 255))))
	env = (vmt.get("$envmap") or "").strip().lower()
	if (
		env
		and env not in ("0", '""', "none")
		and _num(vmt.get("$basealphaenvmapmask"), 0.0) >= 1.0
	):
		base = (vmt.get("$basetexture") or "").strip()
		if base:
			out["flags"] |= F_ENVMASK
			out["envmask"] = _tex_path(base)
	return out if out["flags"] else None


# $bumpmap → sun-lit relief tile.
BUMP_PREFIX = "bump:"
BUMP_TILING = 8  # relief tile repeats 8× per diffuse tile (64 px of a 512² normal map)
BUMP_FACTOR = 0.85
# d2_lostcoast env_sun: towards the sun is yaw 137°, 10° up. Floor texture
# axes in Source are S=+x, T=-y, so tangent space L = (Lx, -Ly, Lz).
_SUN_YAW = math.radians(137.0)
_SUN_ELEV = math.radians(10.0)
SUN_TANGENT = (
	math.cos(_SUN_ELEV) * math.cos(_SUN_YAW),
	-math.cos(_SUN_ELEV) * math.sin(_SUN_YAW),
	# Grazing 10° sun over-darkens: lift so relief reads without crushing.
	max(math.sin(_SUN_ELEV), 0.45),
)


def bump_relief_rgba(rgba: bytes, w: int, h: int) -> tuple[bytes, int, int]:
	"""Light a normal map with the sun → tileable grey relief tile (RGBA)."""
	side = max(16, min(w, h) // BUMP_TILING)
	lx, ly, lz = SUN_TANGENT
	ln = math.sqrt(lx * lx + ly * ly + lz * lz)
	lx, ly, lz = lx / ln, ly / ln, lz / ln

	def shade(x: int, y: int) -> float:
		i = ((y % h) * w + (x % w)) * 4
		nx = rgba[i] / 127.5 - 1.0
		ny = rgba[i + 1] / 127.5 - 1.0
		nz = rgba[i + 2] / 127.5 - 1.0
		return max(0.0, nx * lx + ny * ly + nz * lz)

	# Crop one tile; cross-fade with a half-offset crop so edges wrap cleanly.
	out = bytearray()
	half = side // 2
	for y in range(side):
		wy = min(y, side - 1 - y) / max(1.0, side / 4.0)
		for x in range(side):
			wx = min(x, side - 1 - x) / max(1.0, side / 4.0)
			wgt = max(0.0, min(1.0, min(wx, wy)))
			a = shade(x, y)
			b = shade(x + half + side, y + half + side)
			v = int(max(0, min(255, round((a * wgt + b * (1.0 - wgt)) * 255.0))))
			out += bytes((v, v, v, 255))
	return bytes(out), side, side


def _i8_tiled(lum: list[int], side: int) -> bytes:
	"""GX I8: 8×4 texel tiles, row-major."""
	out = bytearray()
	for ty in range(0, side, 4):
		for tx in range(0, side, 8):
			for y in range(4):
				row = (ty + y) * side + tx
				out += bytes(lum[row : row + 8])
	return bytes(out)


def _i4_tiled(lum: list[int], side: int) -> bytes:
	"""GX I4: 8×8 texel tiles, high nibble first."""
	out = bytearray()
	for ty in range(0, side, 8):
		for tx in range(0, side, 8):
			for y in range(8):
				row = (ty + y) * side + tx
				for x in range(0, 8, 2):
					a = lum[row + x] >> 4
					b = lum[row + x + 1] >> 4
					out.append((a << 4) | b)
	return bytes(out)


def _envmask_i4(rgba: bytes, w: int, h: int) -> bytes:
	"""Resample base-texture alpha to 64² I4 for $basealphaenvmapmask."""
	side = GC_TEVMAT_DETAIL_SIDE
	lum: list[int] = []
	for y in range(side):
		sy0 = y * h // side
		sy1 = max(sy0 + 1, (y + 1) * h // side)
		for x in range(side):
			sx0 = x * w // side
			sx1 = max(sx0 + 1, (x + 1) * w // side)
			acc = 0
			n = 0
			for sy in range(sy0, sy1):
				base = (sy * w + sx0) * 4 + 3
				for _sx in range(sx0, sx1):
					acc += rgba[base]
					base += 4
					n += 1
			lum.append(int(max(0, min(255, round(acc / max(1, n))))))
	body = _i4_tiled(lum, side)
	assert len(body) == GC_TEVMAT_ENVMASK_BYTES, len(body)
	return body


def _detail_i8(rgba: bytes, w: int, h: int, factor: float) -> bytes:
	"""Resample a detail texture to 64² I8 + mips, biased for TEV mod2x.

	TEV computes prev × texel × 2, so texel 128 is neutral. $detailblendfactor
	is folded in here: texel = 0.5 + (d - 0.5) × factor. Each mip is pulled
	further toward neutral so detail fades with distance (Source fades detail
	out the same way) instead of shimmering.
	"""
	side = GC_TEVMAT_DETAIL_SIDE
	lum: list[float] = []
	for y in range(side):
		sy0 = y * h // side
		sy1 = max(sy0 + 1, (y + 1) * h // side)
		for x in range(side):
			sx0 = x * w // side
			sx1 = max(sx0 + 1, (x + 1) * w // side)
			acc = 0.0
			n = 0
			for sy in range(sy0, sy1):
				base = (sy * w + sx0) * 4
				for _sx in range(sx0, sx1):
					acc += (rgba[base] * 0.299 + rgba[base + 1] * 0.587 + rgba[base + 2] * 0.114)
					base += 4
					n += 1
			lum.append(acc / max(1, n))
	# Source detail textures are authored around mid grey; re-centre on the
	# actual mean so mod2x does not shift overall brightness.
	mean = sum(lum) / len(lum)
	body = bytearray()
	level = [(v - mean) for v in lum]
	cur = side
	fade = 1.0
	while cur >= 8:
		texels = [int(max(0, min(255, round(128.0 + d * factor * fade)))) for d in level]
		body += _i8_tiled(texels, cur)
		nxt = cur // 2
		level = [
			(
				level[(2 * y) * cur + 2 * x]
				+ level[(2 * y) * cur + 2 * x + 1]
				+ level[(2 * y + 1) * cur + 2 * x]
				+ level[(2 * y + 1) * cur + 2 * x + 1]
			)
			/ 4.0
			for y in range(nxt)
			for x in range(nxt)
		]
		cur = nxt
		fade *= 0.7
	assert len(body) == GC_TEVMAT_DETAIL_BYTES, len(body)
	return bytes(body)


def build_table(
	mat_features: dict[str, dict], load_rgba, weights: dict[str, float] | None = None
) -> tuple[dict[str, int], bytes, list[str]]:
	"""Assign material ids and encode tevmat.gcm.

	mat_features: Source material path → classify() result (non-None only).
	load_rgba:    callable(texture path) → (rgba, w, h) or None.
	Returns (mat path → id 1..N, file bytes, log lines).
	"""
	log: list[str] = []
	# Detail textures: the most used ones get the GC_TEVMAT_MAX_DETAIL slots.
	use: dict[str, list[float]] = {}
	area: dict[str, float] = {}
	emask_area: dict[str, float] = {}
	for mat, feat in mat_features.items():
		if feat["flags"] & F_DETAIL:
			use.setdefault(feat["detail"], []).append(feat["detail_factor"])
			area[feat["detail"]] = area.get(feat["detail"], 0.0) + (weights or {}).get(mat, 1.0)
		if feat["flags"] & F_ENVMASK:
			em = feat["envmask"]
			emask_area[em] = emask_area.get(em, 0.0) + (weights or {}).get(mat, 1.0)
	detail_idx: dict[str, int] = {}
	detail_blobs: list[bytes] = []
	n_real = 0
	n_bump = 0
	# Largest drawn area first; authored details and bump reliefs have separate quotas.
	for tex, factors in sorted(use.items(), key=lambda kv: (-area[kv[0]], kv[0])):
		is_bump = tex.startswith(BUMP_PREFIX)
		quota = GC_TEVMAT_MAX_DETAIL - GC_TEVMAT_REAL_DETAIL if is_bump else GC_TEVMAT_REAL_DETAIL
		if (n_bump if is_bump else n_real) >= quota:
			log.append(f"  tevmat: detail {tex} dropped (quota {quota})")
			continue
		got = load_rgba(tex[len(BUMP_PREFIX) :] if is_bump else tex)
		if got is not None and is_bump:
			got = bump_relief_rgba(*got)
		if got is None:
			log.append(f"  tevmat: detail {tex} missing")
			continue
		rgba, w, h = got
		factor = sorted(factors)[len(factors) // 2]
		detail_idx[tex] = len(detail_blobs)
		detail_blobs.append(_detail_i8(rgba, w, h, factor))
		if is_bump:
			n_bump += 1
		else:
			n_real += 1
		log.append(f"  tevmat: detail[{detail_idx[tex]}] {tex} {w}x{h} factor={factor:.2f} mats={len(factors)} area={area[tex]:.0f}")

	envmask_idx: dict[str, int] = {}
	envmask_blobs: list[bytes] = []
	for tex, _area in sorted(emask_area.items(), key=lambda kv: (-kv[1], kv[0])):
		if len(envmask_blobs) >= GC_TEVMAT_MAX_ENVMASK:
			log.append(f"  tevmat: envmask {tex} dropped (quota {GC_TEVMAT_MAX_ENVMASK})")
			continue
		got = load_rgba(tex)
		if got is None:
			log.append(f"  tevmat: envmask {tex} missing")
			continue
		rgba, w, h = got
		envmask_idx[tex] = len(envmask_blobs)
		envmask_blobs.append(_envmask_i4(rgba, w, h))
		log.append(
			f"  tevmat: envmask[{envmask_idx[tex]}] {tex} {w}x{h} "
			f"area={emask_area[tex]:.0f}"
		)

	ids: dict[str, int] = {}
	recs = bytearray()
	for mat in sorted(mat_features):
		feat = mat_features[mat]
		flags = feat["flags"]
		didx = 0
		scale = 0
		maskidx = 0
		if flags & F_DETAIL:
			if feat["detail"] in detail_idx:
				didx = detail_idx[feat["detail"]]
				scale = int(max(1, min(0xFFFF, round(feat["detail_scale"] * 256.0))))
			else:
				flags &= ~F_DETAIL
		if flags & F_ENVMASK:
			if feat["envmask"] in envmask_idx:
				maskidx = envmask_idx[feat["envmask"]]
			else:
				flags &= ~F_ENVMASK
		if not flags:
			continue
		if len(ids) >= GC_TEVMAT_MAX_MATS:
			log.append(f"  tevmat: {mat} dropped (>{GC_TEVMAT_MAX_MATS} materials)")
			continue
		ids[mat] = len(ids) + 1
		# flags, detail idx, detail scale u16, alpharef, envmask idx, pad2
		recs += struct.pack(
			">BBHBB2x", flags, didx, scale, feat.get("alpharef", 0), maskidx
		)
	hdr = GC_TEVMAT_MAGIC + struct.pack(">HBB", GC_TEVMAT_VERSION, len(ids), len(detail_blobs))
	head = hdr + bytes(recs)
	head += b"\0" * ((-len(head)) % 32)
	body = head + b"".join(detail_blobs)
	if envmask_blobs:
		# Trailer after detail tiles: runtime probes for GCem.
		body += GC_TEVMAT_ENVMASK_MAGIC + struct.pack(
			">B3x", len(envmask_blobs)
		) + b"".join(envmask_blobs)
	return ids, body, log


def write_table(path: Path, blob: bytes) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_bytes(blob)
