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
  $bumpmap without $envmap (rock / cliffs / monastery walls)
      → shared 64² I8 height tiles (normal-map Z, or authored $parallaxmap).
        Runtime embosses lit colour: lit × (0.5 + h(uv) − h(uv+d)) × 2 with
        d = map-sun offset in UV space via a texture matrix (trailer GCeh).
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
GC_TEVMAT_MAX_ENVMASK = 8  # shared I4 masks for Source envmap alpha masks
# Past those, 32x32 masks (512 bytes) in a second table: the church's frescoes
# and walls each carry their own mask, nineteen more than the 64x64 table
# holds, and a material whose mask is not in a table gets no reflection at
# all (the frescoes' gold sheen). Mask index 8 and up is the small table.
GC_TEVMAT_MAX_ENVMASK_SMALL = 24
GC_TEVMAT_ENVMASK_SMALL_MAGIC = b"GCes"
GC_TEVMAT_MAX_EMBOSS = 16  # shared I8 height tiles for cliff / rock emboss
GC_TEVMAT_DETAIL_SIDE = 64
GC_TEVMAT_DETAIL_BYTES = 4096 + 1024 + 256 + 64
GC_TEVMAT_ENVMASK_BYTES = 2048  # 64² GX I4
GC_TEVMAT_EMBOSS_BYTES = 4096  # 64² GX I8 (no mips)
LAST_ENVMASKS: list[dict] = []  # filled by build_table, in mask index order
LAST_EMBOSS: list[dict] = []  # filled by build_table, in height index order
GC_TEVMAT_ENVMASK_MAGIC = b"GCem"
GC_TEVMAT_EMBOSS_MAGIC = b"GCeh"

F_DETAIL = 0x01
F_ALPHATEST = 0x02
F_ENVMASK = 0x04
# $decal: the alpha is the decal's shape, so the cut-out also applies on
# lightmapped surfaces (plain alpha-test materials only cut out when unlit).
F_DECAL = 0x08
# Soft decal: the atlas is opaque and its alpha is the separate I4 page
# (media/da_<map>.bin); the engine blends instead of cutting out.
F_SOFT = 0x10
# $bumpmap without $envmap: Flipper emboss from a shared height tile.
F_EMBOSS = 0x20


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
	# Cliff / rock $bumpmap without $envmap → emboss height (see F_EMBOSS).
	# An earlier stand-in lit a 1/8 crop as mod2x detail; that punched white
	# voids over lightmaps and is not used anymore.
	if _num(vmt.get("$alphatest"), 0.0) >= 1.0:
		out["flags"] |= F_ALPHATEST
		out["alpharef"] = int(max(1, min(255, round(_num(vmt.get("$alphatestreference"), 0.5) * 255))))
	# $translucent brush textures (the Combine grate, the monastery rail)
	# blend by alpha in Source. The world DL has one opaque pass, so they cut
	# out at half alpha instead of drawing as a solid sheet.
	if _num(vmt.get("$translucent"), 0.0) >= 1.0 and not out["flags"] & F_ALPHATEST:
		out["flags"] |= F_ALPHATEST
		out["alpharef"] = 128
	# Overlay decals (graffiti, rubble, trash) blend by alpha in Source. The
	# world DL has one opaque pass, so they cut out at 1-bit alpha instead.
	if _num(vmt.get("$decal"), 0.0) >= 1.0:
		out["flags"] |= F_DECAL
		if not out["flags"] & F_ALPHATEST:
			out["flags"] |= F_ALPHATEST
			out["alpharef"] = 96
	env = (vmt.get("$envmap") or "").strip().lower()
	has_env = bool(env) and env not in ("0", '""', "none")
	if has_env:
		# Lost Coast rocks and wet sand put the reflection mask in the normal
		# map alpha. Source also supports a base texture alpha mask.
		mask = ""
		if _num(vmt.get("$normalmapalphaenvmapmask"), 0.0) >= 1.0:
			mask = (vmt.get("$bumpmap") or "").strip()
			out["envmask_normal"] = True
		elif _num(vmt.get("$basealphaenvmapmask"), 0.0) >= 1.0:
			mask = (vmt.get("$basetexture") or "").strip()
		if mask:
			out["flags"] |= F_ENVMASK
			out["envmask"] = _tex_path(mask)
	# Emboss from $bumpmap. Harbor cliffs keep $envmap for the additive
	# specular pass but still need height relief on diffuse×LM. Water dudv
	# bumps ride the separate water bake. Wet sand (IndTex) still embosses
	# the diffuse base; the specular warp is a later additive pass.
	bump = (vmt.get("$bumpmap") or "").strip()
	if bump and not (vmt.get("$bottommaterial") or "").strip():
		para = (vmt.get("$parallaxmap") or "").strip()
		out["flags"] |= F_EMBOSS
		if para:
			out["emboss"] = _tex_path(para)
			out["emboss_from_normal"] = False
		else:
			out["emboss"] = _tex_path(bump)
			out["emboss_from_normal"] = True
	return out if out["flags"] else None


# d2_lostcoast env_sun: towards the sun is yaw 137°, 10° up. Floor texture
# axes in Source are S=+x, T=-y, so tangent space L = (Lx, -Ly, Lz).
# Runtime emboss uses the XY as a UV offset (texture matrix).
# How much of N.L's swing a shade tile keeps. 1 = the slope's own dot product
# with the sun's tangent direction (a 15 degree bump is +-26 %).
EMBOSS_SHADE_GAIN = 1.0
# Height difference across two texels of a 64x64 tile -> slope, for the one
# authored height map (sandfloor007c_height).
EMBOSS_HEIGHT_SLOPE = 2.0
_SUN_YAW = math.radians(137.0)
_SUN_ELEV = math.radians(10.0)
SUN_TANGENT = (
	math.cos(_SUN_ELEV) * math.cos(_SUN_YAW),
	-math.cos(_SUN_ELEV) * math.sin(_SUN_YAW),
	max(math.sin(_SUN_ELEV), 0.45),
)


def bump_height_i8(rgba: bytes, w: int, h: int, from_normal: bool) -> bytes:
	"""Resample a height or normal map to a 64² GX I8 height tile.

	Normal maps contribute their Z (blue) channel — a cheap stand-in for the
	authored height Source keeps in $parallaxmap. Authored height / greyscale
	maps use luminance. Values stay in 0..255 (not mod2x-centred): the
	runtime emboss formula is lit × (0.5 + h0 − h1) × 2.
	"""
	side = GC_TEVMAT_DETAIL_SIDE
	if w <= 0 or h <= 0 or len(rgba) != w * h * 4:
		raise ValueError("invalid height/normal dimensions or RGBA data")
	lum: list[int] = []
	for y in range(side):
		sy0, sy1 = y * h // side, max(y * h // side + 1, (y + 1) * h // side)
		for x in range(side):
			sx0, sx1 = x * w // side, max(x * w // side + 1, (x + 1) * w // side)
			acc = 0.0
			n = 0
			for sy in range(sy0, sy1):
				base = (sy * w + sx0) * 4
				for _sx in range(sx0, sx1):
					if from_normal:
						# The shade of this texel under the map's sun: half
						# of 1 + N.L over the surface, N from the normal map
						# (R along S, G along T), L the sun in tangent space.
						# The engine draws lit x tile x 2 (trailer byte 5 = 1).
						# It used to store the mean of X and Y as a height and
						# difference it four texels apart at run time: a second
						# derivative, which rimmed every bump with light and
						# quilted the cliffs. Retail's relief is this, and
						# subtle.
						nx = rgba[base] / 127.5 - 1.0
						ny = rgba[base + 1] / 127.5 - 1.0
						acc += 255.0 * 0.5 * (1.0 + EMBOSS_SHADE_GAIN * (nx * SUN_TANGENT[0] + ny * SUN_TANGENT[1]))
					else:
						acc += (
							rgba[base] * 0.299
							+ rgba[base + 1] * 0.587
							+ rgba[base + 2] * 0.114
						)
					base += 4
					n += 1
			lum.append(int(max(0, min(255, round(acc / max(1, n))))))
	# Stretch contrast so cliff grain reads at 480p after the
	# (0.5+h0-h1)×2 emboss; keep the mean near mid-grey for a neutral floor.
	if not from_normal:
		# An authored height map: its slope along the sun, as the shade a
		# normal map would give (central differences, wrapping: it tiles).
		hs = [v / 255.0 for v in lum]
		lum = []
		for y in range(side):
			for x in range(side):
				dx = hs[y * side + (x + 1) % side] - hs[y * side + (x - 1) % side]
				dy = hs[((y + 1) % side) * side + x] - hs[((y - 1) % side) * side + x]
				d = -(dx * SUN_TANGENT[0] + dy * SUN_TANGENT[1]) * EMBOSS_HEIGHT_SLOPE
				lum.append(int(max(0, min(255, round(255.0 * 0.5 * (1.0 + EMBOSS_SHADE_GAIN * d))))))
	mean = 128.0
	stretch = 1.0
	lum = [
		int(max(0, min(255, round(128.0 + (v - mean) * stretch))))
		for v in lum
	]
	body = _i8_tiled(lum, side)
	assert len(body) == GC_TEVMAT_EMBOSS_BYTES, len(body)
	return body


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


def normal_xy_ia8(rgba: bytes, w: int, h: int, side: int = 64) -> bytes:
	"""Pack a complete tangent-space normal map's X/Y channels as GX IA8.

	Each byte remains centered on 128, so the runtime can use it as a signed
	indirect-texture offset. The two channels are area filtered together; no
	crop, repetition, contrast stretch, or alpha-mask conversion is applied.
	The caller owns the separate Source alpha reflectivity mask.
	"""
	if w <= 0 or h <= 0 or side <= 0 or side % 4 or len(rgba) != w * h * 4:
		raise ValueError("invalid normal-map dimensions or RGBA data")
	xy = []
	for y in range(side):
		sy0, sy1 = y * h // side, max(y * h // side + 1, (y + 1) * h // side)
		for x in range(side):
			sx0, sx1 = x * w // side, max(x * w // side + 1, (x + 1) * w // side)
			r = g = count = 0
			for sy in range(sy0, sy1):
				for sx in range(sx0, sx1):
					o = (sy * w + sx) * 4
					r += rgba[o]
					g += rgba[o + 1]
					count += 1
			xy.append((round(r / count), round(g / count)))
	out = bytearray()
	for ty in range(0, side, 4):
		for tx in range(0, side, 4):
			for y in range(4):
				for x in range(4):
					out.extend(xy[(ty + y) * side + tx + x])
	return bytes(out)


def _envmask_i4(rgba: bytes, w: int, h: int, side: int = GC_TEVMAT_DETAIL_SIDE) -> bytes:
	"""Bake Source's authored envmap mask to a compact, high-contrast I4 map."""
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
	# The authored mask as it is, squared: the shader squares cube x mask x
	# tint in linear light and the engine keeps that product linear.
	lum = [int(round(255.0 * (value / 255.0) ** 2.0)) for value in lum]
	body = _i4_tiled(lum, side)
	assert len(body) == side * side // 2, len(body)
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
	emboss_area: dict[str, float] = {}
	emboss_from_normal: dict[str, bool] = {}
	for mat, feat in mat_features.items():
		if feat["flags"] & F_DETAIL:
			use.setdefault(feat["detail"], []).append(feat["detail_factor"])
			area[feat["detail"]] = area.get(feat["detail"], 0.0) + (weights or {}).get(mat, 1.0)
		if feat["flags"] & F_ENVMASK:
			em = feat["envmask"]
			emask_area[em] = emask_area.get(em, 0.0) + (weights or {}).get(mat, 1.0)
		if feat["flags"] & F_EMBOSS:
			eh = feat["emboss"]
			w = (weights or {}).get(mat, 1.0)
			# Harbor cliffs are the visual target: boost rockwall / blendrock
			# so monastery / metal variants don't fill the shared height quota.
			ml = mat.lower()
			if "rockwall" in ml or "blendrock" in ml or "/nature/rock" in ml:
				w *= 4.0
			emboss_area[eh] = emboss_area.get(eh, 0.0) + w
			# Prefer "from normal" if any user of this tile says so.
			emboss_from_normal[eh] = emboss_from_normal.get(eh, False) or bool(
				feat.get("emboss_from_normal")
			)
	detail_idx: dict[str, int] = {}
	detail_blobs: list[bytes] = []
	n_real = 0
	# Largest drawn area first; authored $detail only (bump relief retired).
	for tex, factors in sorted(use.items(), key=lambda kv: (-area[kv[0]], kv[0])):
		if n_real >= GC_TEVMAT_REAL_DETAIL:
			log.append(f"  tevmat: detail {tex} dropped (quota {GC_TEVMAT_REAL_DETAIL})")
			continue
		got = load_rgba(tex)
		if got is None:
			log.append(f"  tevmat: detail {tex} missing")
			continue
		rgba, w, h = got
		factor = sorted(factors)[len(factors) // 2]
		detail_idx[tex] = len(detail_blobs)
		detail_blobs.append(_detail_i8(rgba, w, h, factor))
		n_real += 1
		log.append(
			f"  tevmat: detail[{detail_idx[tex]}] {tex} {w}x{h} "
			f"factor={factor:.2f} mats={len(factors)} area={area[tex]:.0f}"
		)

	envmask_idx: dict[str, int] = {}
	envmask_blobs: list[bytes] = []
	envmask_small: list[bytes] = []
	del LAST_ENVMASKS[:]
	for tex, _area in sorted(emask_area.items(), key=lambda kv: (-kv[1], kv[0])):
		if len(envmask_blobs) >= GC_TEVMAT_MAX_ENVMASK:
			if len(envmask_small) >= GC_TEVMAT_MAX_ENVMASK_SMALL:
				log.append(f"  tevmat: envmask {tex} dropped (quota {GC_TEVMAT_MAX_ENVMASK} + {GC_TEVMAT_MAX_ENVMASK_SMALL})")
				continue
			got = load_rgba(tex)
			if got is None:
				log.append(f"  tevmat: envmask {tex} missing")
				continue
			rgba, w, h = got
			envmask_idx[tex] = GC_TEVMAT_MAX_ENVMASK + len(envmask_small)
			envmask_small.append(_envmask_i4(rgba, w, h, 32))
			# No normal map from the small table (bake_lostcoast_envnormals
			# indexes the 64x64 one).
			LAST_ENVMASKS.append({"tex": tex, "normal": False})
			log.append(f"  tevmat: envmask[{envmask_idx[tex]}] {tex} {w}x{h} -> 32x32 area={emask_area[tex]:.0f}")
			continue
		got = load_rgba(tex)
		if got is None:
			log.append(f"  tevmat: envmask {tex} missing")
			continue
		rgba, w, h = got
		envmask_idx[tex] = len(envmask_blobs)
		envmask_blobs.append(_envmask_i4(rgba, w, h))
		# Mask order for tools/bake_lostcoast_envnormals.py (envmask-names.json).
		LAST_ENVMASKS.append({"tex": tex, "normal": any(
			f.get("envmask") == tex and f.get("envmask_normal") for f in mat_features.values())})
		log.append(
			f"  tevmat: envmask[{envmask_idx[tex]}] {tex} {w}x{h} "
			f"area={emask_area[tex]:.0f}"
		)

	emboss_idx: dict[str, int] = {}
	emboss_blobs: list[bytes] = []
	del LAST_EMBOSS[:]
	for tex, _area in sorted(emboss_area.items(), key=lambda kv: (-kv[1], kv[0])):
		if len(emboss_blobs) >= GC_TEVMAT_MAX_EMBOSS:
			log.append(f"  tevmat: emboss {tex} dropped (quota {GC_TEVMAT_MAX_EMBOSS})")
			continue
		got = load_rgba(tex)
		if got is None:
			log.append(f"  tevmat: emboss {tex} missing")
			continue
		rgba, w, h = got
		from_n = emboss_from_normal.get(tex, True)
		try:
			blob = bump_height_i8(rgba, w, h, from_n)
		except ValueError as exc:
			log.append(f"  tevmat: emboss {tex} skip ({exc})")
			continue
		emboss_idx[tex] = len(emboss_blobs)
		emboss_blobs.append(blob)
		LAST_EMBOSS.append({"tex": tex, "from_normal": from_n})
		log.append(
			f"  tevmat: emboss[{emboss_idx[tex]}] {tex} {w}x{h} "
			f"from_normal={int(from_n)} area={emboss_area[tex]:.0f}"
		)

	ids: dict[str, int] = {}
	rec_ids: dict[bytes, int] = {}
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
		# Emboss height index: envmask byte when free, else the detail byte
		# (cliff rocks with $envmap rarely also author $detail).
		if flags & F_EMBOSS:
			if feat["emboss"] not in emboss_idx:
				flags &= ~F_EMBOSS
			elif flags & F_ENVMASK:
				if flags & F_DETAIL:
					# Prefer authored detail; drop emboss on this rare combo.
					flags &= ~F_EMBOSS
				else:
					didx = emboss_idx[feat["emboss"]]
			else:
				maskidx = emboss_idx[feat["emboss"]]
		if not flags:
			continue
		# flags, detail idx, detail scale u16, alpharef, envmask/emboss idx, pad2
		rec = struct.pack(
			">BBHBB2x", flags, didx, scale, feat.get("alpharef", 0), maskidx
		)
		# Materials with the same recipe share one record: the cubemap
		# patches of one texture (name_x_y_z) would otherwise fill the table
		# and push later materials out, alpha-tested grates among them.
		if rec in rec_ids:
			ids[mat] = rec_ids[rec]
			continue
		if len(rec_ids) >= GC_TEVMAT_MAX_MATS:
			log.append(f"  tevmat: {mat} dropped (>{GC_TEVMAT_MAX_MATS} recipes)")
			continue
		rec_ids[rec] = ids[mat] = len(rec_ids) + 1
		recs += rec
	hdr = GC_TEVMAT_MAGIC + struct.pack(">HBB", GC_TEVMAT_VERSION, len(rec_ids), len(detail_blobs))
	head = hdr + bytes(recs)
	head += b"\0" * ((-len(head)) % 32)
	body = head + b"".join(detail_blobs)
	if envmask_blobs:
		# Trailer after detail tiles: runtime probes for GCem.
		body += GC_TEVMAT_ENVMASK_MAGIC + struct.pack(
			">B3x", len(envmask_blobs)
		) + b"".join(envmask_blobs)
	if emboss_blobs:
		# Trailer after envmasks (or details): runtime probes for GCeh.
		# Byte 5 = 1: the tiles are shade factors (bump_height_i8 from a
		# normal map), not heights.
		body += GC_TEVMAT_EMBOSS_MAGIC + struct.pack(
			">BB2x", len(emboss_blobs), 1
		) + b"".join(emboss_blobs)
	if envmask_small:
		# Last: an engine that does not know it stops reading here.
		body += GC_TEVMAT_ENVMASK_SMALL_MAGIC + struct.pack(
			">B3x", len(envmask_small)
		) + b"".join(envmask_small)
	return ids, body, log


def write_table(path: Path, blob: bytes) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_bytes(blob)
