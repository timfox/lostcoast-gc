#!/usr/bin/env python3
"""Bake Lost Coast Source map → GCWT + GCcf + GoldSrc BSP29 stub."""
from __future__ import annotations

import argparse
import hashlib
import math
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import bake_tevmat  # noqa: E402
from vmt import basetextures, load_vmt, material_short_name  # noqa: E402
from vpk import open_vpk_roots, read_first  # noqa: E402
from vtf import decode_vtf  # noqa: E402

# CapFaces constants (match engine bake_capfaces_pack.py)
GC_CF_MAGIC = b"GCcf"
GC_CF_VERSION = 1
GC_CF_FLAG_LM_ATLAS = 1
GC_MAX_CAP_FACES = 768  # 1024 CapFaces BSS starved lean sky CMPR on Flipper
GC_CAP_MAX_VERTS = 8
GC_CAP_LM_DIM = 4
GC_LM_ATLAS_W = 128
GC_LM_ATLAS_H = 128
GC_LM_ATLAS_COLS = GC_LM_ATLAS_W // GC_CAP_LM_DIM
GC_LM_ATLAS_ROWS = GC_LM_ATLAS_H // GC_CAP_LM_DIM
BAKE_EDGE = 1
BAKE_TEX = 3

GC_WT_MAGIC = b"GCWT"
GC_WT_FLAG_ALPHA = 1
GC_WT_ENTRY = struct.Struct(">16sHHBBHI")

SURF_SKY = 0x4
SURF_WARP = 0x8
SURF_TRANS = 0x10
SURF_NODRAW = 0x80
SURF_HINT = 0x100
SURF_SKIP = 0x200
SURF_NOLIGHT = 0x400
SURF_TRIGGER = 0x40
# CapFaces bake-only: Source $envmap / normalmap-alpha specular (not a SURF_* bit).
GC_SURF_ENVMAP = 0x4000


def fnv1a32(s: str) -> int:
	h = 2166136261
	for c in s.encode("latin-1"):
		h ^= c
		h = (h * 16777619) & 0xFFFFFFFF
	return h


def gx_cmpr_encode_blocks(rgb, opaque) -> bytes:
	"""GX CMPR (DXT1) — same layout as xash3d-gc build-gamecube-disc.py."""
	import numpy as np

	h, w = opaque.shape
	bh, bw = h // 4, w // 4
	flat = rgb.astype(np.float32).reshape(bh, 4, bw, 4, 3).transpose(0, 2, 1, 3, 4).reshape(bh, bw, 16, 3)
	mask = opaque.reshape(bh, 4, bw, 4).transpose(0, 2, 1, 3).reshape(bh, bw, 16)
	weight = mask[..., None].astype(np.float32)
	count = np.maximum(weight.sum(2, keepdims=True), 1.0)
	mean = (flat * weight).sum(2, keepdims=True) / count
	diff = (flat - mean) * weight
	cov = np.einsum("...ki,...kj->...ij", diff, diff)
	axis = np.ones(cov.shape[:2] + (3,), np.float32)
	for _ in range(8):
		axis = np.einsum("...ij,...j->...i", cov, axis)
		axis /= np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-6)
	t = np.einsum("...ki,...i->...k", flat - mean, axis)
	lo_t = np.where(mask, t, np.inf).min(2)
	hi_t = np.where(mask, t, -np.inf).max(2)
	lo_t = np.where(np.isfinite(lo_t), lo_t, 0.0)
	hi_t = np.where(np.isfinite(hi_t), hi_t, 0.0)
	lo = np.clip(mean[:, :, 0] + axis * lo_t[..., None], 0, 255).astype(np.int32)
	hi = np.clip(mean[:, :, 0] + axis * hi_t[..., None], 0, 255).astype(np.int32)

	def to565(c):
		return ((c[..., 0] >> 3) << 11) | ((c[..., 1] >> 2) << 5) | (c[..., 2] >> 3)

	def from565(v):
		r, g, b = (v >> 11) & 31, (v >> 5) & 63, v & 31
		return np.stack(((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)), -1)

	a, b = to565(hi), to565(lo)
	has_alpha = ~mask.all(2)
	four = np.where(a > b, a, b), np.where(a > b, b, a)
	c0 = np.where(has_alpha, four[1], four[0])
	c1 = np.where(has_alpha, four[0], four[1])
	e0, e1 = from565(c0), from565(c1)
	pal4 = np.stack((e0, e1, (2 * e0 + e1) // 3, (e0 + 2 * e1) // 3), 2)
	pal3 = np.stack((e0, e1, (e0 + e1) // 2, np.full_like(e0, 100000)), 2)
	pal = np.where(has_alpha[..., None, None], pal3, pal4).astype(np.float32)
	dist = ((flat[:, :, :, None, :] - pal[:, :, None, :, :]) ** 2).sum(-1)
	idx = dist.argmin(-1)
	idx = np.where(mask, idx, 3)
	idx = np.where(((c0 == c1) & ~has_alpha)[..., None], 0, idx)
	rows = idx.reshape(bh, bw, 4, 4)
	bits = ((rows[..., 0] << 6) | (rows[..., 1] << 4) | (rows[..., 2] << 2) | rows[..., 3]).astype(np.uint8)
	out = bytearray()
	for ty in range(0, bh, 2):
		for tx in range(0, bw, 2):
			for by, bx in ((ty, tx), (ty, tx + 1), (ty + 1, tx), (ty + 1, tx + 1)):
				out += struct.pack(">HH", int(c0[by, bx]), int(c1[by, bx]))
				out += bits[by, bx].tobytes()
	return bytes(out)


def source_lump(data: bytes, i: int) -> bytes:
	off, ln, _ver, _fcc = struct.unpack_from("<iiii", data, 8 + i * 16)
	return data[off : off + ln]


def load_source_bsp(path: Path) -> dict:
	data = path.read_bytes()
	if data[:4] != b"VBSP":
		raise ValueError(f"{path}: not VBSP")
	ver = struct.unpack_from("<i", data, 4)[0]
	if ver not in (19, 20, 21):
		raise ValueError(f"{path}: unsupported VBSP version {ver}")

	ents = source_lump(data, 0).decode("latin-1", "replace")
	planes_raw = source_lump(data, 1)
	texdata_raw = source_lump(data, 2)
	verts_raw = source_lump(data, 3)
	texinfo_raw = source_lump(data, 6)
	faces_raw = source_lump(data, 7)
	lighting = source_lump(data, 8)
	edges_raw = source_lump(data, 12)
	surfedges_raw = source_lump(data, 13)
	string_data = source_lump(data, 43)
	string_table = source_lump(data, 44)

	verts = [struct.unpack_from("<fff", verts_raw, i) for i in range(0, len(verts_raw), 12)]
	edges = [struct.unpack_from("<hh", edges_raw, i) for i in range(0, len(edges_raw), 4)]
	surfedges = list(struct.unpack_from(f"<{len(surfedges_raw) // 4}i", surfedges_raw))

	planes = []
	for i in range(0, len(planes_raw), 20):
		nx, ny, nz, dist, typ = struct.unpack_from("<ffffi", planes_raw, i)
		planes.append(((nx, ny, nz), dist, typ))

	# texdata: Vector reflectivity(12) + 5 ints = 32
	texdatas = []
	for i in range(0, len(texdata_raw), 32):
		_rx, _ry, _rz, name_id, w, h, vw, vh = struct.unpack_from("<fffiiiii", texdata_raw, i)
		texdatas.append({"name_id": name_id, "width": w, "height": h})

	name_ids = struct.unpack_from(f"<{len(string_table) // 4}i", string_table)
	tex_names = []
	for off in name_ids:
		end = string_data.index(b"\0", off)
		tex_names.append(string_data[off:end].decode("latin-1").replace("\\", "/").lower())

	# texinfo: textureVecs[2][4] + lightmapVecs[2][4] + flags + texdata = 72
	texinfos = []
	for i in range(0, len(texinfo_raw), 72):
		vals = struct.unpack_from("<16fii", texinfo_raw, i)
		texinfos.append(
			{
				"vecs": (vals[0:4], vals[4:8]),
				"lmvecs": (vals[8:12], vals[12:16]),
				"flags": vals[16],
				"texdata": vals[17],
			}
		)

	faces = []
	face_fmt = "<HBBIhhhh4sif5iHHI"
	assert struct.calcsize(face_fmt) == 56, struct.calcsize(face_fmt)
	for i in range(0, len(faces_raw), 56):
		(
			planenum,
			side,
			_on_node,
			firstedge,
			numedges,
			texinfo,
			dispinfo,
			_sfog,
			styles,
			lightofs,
			area,
			_lmm0,
			_lmm1,
			lms0,
			lms1,
			_orig,
			_np,
			_fp,
			_sm,
		) = struct.unpack_from(face_fmt, faces_raw, i)
		faces.append(
			{
				"planenum": planenum,
				"side": side,
				"firstedge": firstedge,
				"numedges": numedges,
				"texinfo": texinfo,
				"dispinfo": dispinfo,
				"styles": styles,
				"lightofs": lightofs,
				"area": area,
				"lm_w": lms0 + 1,
				"lm_h": lms1 + 1,
			}
		)

	spawns = []
	for m in re.finditer(r"\{[^}]*?classname\"\s+\"info_player_start\"[^}]*\}", ents, re.S):
		block = m.group(0)
		om = re.search(r'"origin"\s+"([^"]+)"', block)
		am = re.search(r'"angles"\s+"([^"]+)"', block)
		if not om:
			continue
		origin = [float(x) for x in om.group(1).split()]
		angles = [float(x) for x in am.group(1).split()] if am else [0.0, 0.0, 0.0]
		spawns.append({"origin": origin, "angles": angles})

	sky = "desert"
	sm = re.search(r'"skyname"\s+"([^"]+)"', ents)
	if sm:
		sky = sm.group(1)

	return {
		"data": data,
		"verts": verts,
		"edges": edges,
		"surfedges": surfedges,
		"planes": planes,
		"texinfos": texinfos,
		"texdatas": texdatas,
		"tex_names": tex_names,
		"faces": faces,
		"lighting": lighting,
		"spawns": spawns,
		"sky": sky,
	}


def face_verts(bsp: dict, face: dict) -> list[tuple[float, float, float]]:
	pts = []
	fe = face["firstedge"]
	for i in range(face["numedges"]):
		se = bsp["surfedges"][fe + i]
		if se >= 0:
			e = bsp["edges"][se]
			v = bsp["verts"][e[0]]
		else:
			e = bsp["edges"][-se]
			v = bsp["verts"][e[1]]
		pts.append(v)
	return pts


def decimate_ring(pts: list, maxv: int = GC_CAP_MAX_VERTS) -> list:
	if len(pts) <= maxv:
		return pts
	out = []
	n = len(pts)
	for i in range(maxv):
		out.append(pts[(i * n) // maxv])
	return out


def fan_tris(pts: list) -> list[list]:
	"""Split an N-gon into triangles so CapFaces keeps the full outline.

	decimate_ring(→8) turns big cliff brushes into ribbon silhouettes; fan
	tris preserve coverage (each tri fits GC_CAP_MAX_VERTS).
	"""
	if len(pts) <= GC_CAP_MAX_VERTS:
		return [pts]
	return [[pts[0], pts[i], pts[i + 1]] for i in range(1, len(pts) - 1)]


def tex_st(texinfo: dict, p, tw: float, th: float) -> tuple[float, float]:
	v0, v1 = texinfo["vecs"]
	s = p[0] * v0[0] + p[1] * v0[1] + p[2] * v0[2] + v0[3]
	t = p[0] * v1[0] + p[1] * v1[1] + p[2] * v1[2] + v1[3]
	if tw < 1:
		tw = 64
	if th < 1:
		th = 64
	return s / tw, t / th


def facing_score(plane_n, plane_d, eye) -> float:
	return eye[0] * plane_n[0] + eye[1] * plane_n[1] + eye[2] * plane_n[2] - plane_d


def _tonemap_hdr(x: float, exposure: float = 4.0) -> int:
	import math

	if not math.isfinite(x) or x <= 0.0:
		return 0
	x *= exposure
	y = x / (1.0 + x)
	y = pow(y, 1.0 / 2.2)
	return min(255, int(y * 255.0 + 0.5))


def _rgb565(r: int, g: int, b: int) -> int:
	return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def sample_lm_rgb565(lighting: bytes, lightofs: int, w: int, h: int) -> bytes:
	"""Downsample face luxels into a 4×4 RGB565 tile (linear layout).

	Source Lost Coast uses ColorRGBExp32 (4 bytes/luxel). Fall back to RGB888
	then flat mid-grey so CapFaces never bind garbage LM tiles.
	"""
	tile = bytearray(GC_CAP_LM_DIM * GC_CAP_LM_DIM * 2)
	flat = 0x8410
	if lightofs < 0 or not lighting or w < 1 or h < 1:
		for i in range(GC_CAP_LM_DIM * GC_CAP_LM_DIM):
			struct.pack_into(">H", tile, i * 2, flat)
		return bytes(tile)

	def fill(sample_rgb) -> bytes:
		for ty in range(GC_CAP_LM_DIM):
			for tx in range(GC_CAP_LM_DIM):
				sx = min(w - 1, (tx * w) // GC_CAP_LM_DIM)
				sy = min(h - 1, (ty * h) // GC_CAP_LM_DIM)
				r, g, b = sample_rgb(sx, sy)
				struct.pack_into(">H", tile, (ty * GC_CAP_LM_DIM + tx) * 2, _rgb565(r, g, b))
		return bytes(tile)

	# ColorRGBExp32 (HDR) — preferred for Lost Coast.
	# Valve TexLightToLinear: (c/255) * 2^exponent with signed exponent.
	need4 = w * h * 4
	if lightofs + need4 <= len(lighting):
		def sample_exp(sx: int, sy: int):
			o = lightofs + (sy * w + sx) * 4
			r8, g8, b8 = lighting[o], lighting[o + 1], lighting[o + 2]
			exp = struct.unpack("b", bytes((lighting[o + 3],)))[0]
			scale = pow(2.0, float(exp))
			return (
				_tonemap_hdr((r8 / 255.0) * scale),
				_tonemap_hdr((g8 / 255.0) * scale),
				_tonemap_hdr((b8 / 255.0) * scale),
			)

		return fill(sample_exp)

	# RGB888 style-0 LDR
	need3 = w * h * 3
	if lightofs + need3 <= len(lighting):
		def sample_rgb888(sx: int, sy: int):
			o = lightofs + (sy * w + sx) * 3
			return lighting[o], lighting[o + 1], lighting[o + 2]

		return fill(sample_rgb888)

	for i in range(GC_CAP_LM_DIM * GC_CAP_LM_DIM):
		struct.pack_into(">H", tile, i * 2, flat)
	return bytes(tile)


def rgba_to_cmpr_mips(rgba: bytes, w: int, h: int, alpha: bool, max_side: int = 256):
	import numpy as np
	from PIL import Image

	arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
	rgb = arr[..., :3]
	opaque = arr[..., 3] > 127 if alpha else np.ones((h, w), bool)
	tw = 1 << max(3, (w - 1).bit_length())
	th = 1 << max(3, (h - 1).bit_length())
	cap = max(8, int(max_side))
	tw = min(tw, cap)
	th = min(th, cap)
	if (tw, th) != (w, h):
		resample = Image.Resampling.NEAREST if alpha else Image.Resampling.BICUBIC
		rgb = np.asarray(Image.fromarray(rgb).resize((tw, th), resample))
		opaque = np.asarray(Image.fromarray(opaque.astype(np.uint8) * 255).resize((tw, th), Image.Resampling.NEAREST)) > 127
	levels = bytearray()
	count = 0
	fcol = rgb.astype(np.float32)
	fcov = opaque.astype(np.float32)
	while True:
		levels += gx_cmpr_encode_blocks(np.clip(fcol + 0.5, 0, 255).astype(np.uint8), fcov >= 0.5)
		count += 1
		ch, cw = fcov.shape
		if min(ch, cw) <= 8:
			break
		cover = fcov.reshape(ch // 2, 2, cw // 2, 2).sum((1, 3))
		sumc = (fcol * fcov[..., None]).reshape(ch // 2, 2, cw // 2, 2, 3).sum((1, 3))
		fcol = np.where(cover[..., None] > 0, sumc / np.maximum(cover[..., None], 1e-6), 0)
		fcov = cover / 4
	return tw, th, count, bytes(levels)


def write_gcwt(path: Path, entries: list[tuple[str, int, int, int, int, int, bytes]]) -> None:
	blob = bytearray()
	table_meta = []
	for name, tw, th, levels, flags, tag, body in entries:
		table_meta.append((name, tw, th, levels, flags, tag, len(blob)))
		blob += body
		blob += bytes(-len(blob) & 31)
	head = GC_WT_MAGIC + struct.pack(">II", len(table_meta), len(blob))
	table = b"".join(
		GC_WT_ENTRY.pack(n.encode("latin-1")[:16], tw, th, lv, fl, tag, of)
		for n, tw, th, lv, fl, tag, of in table_meta
	)
	first = len(head) + len(table)
	pad = bytes(-first & 31)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_bytes(head + table + pad + bytes(blob))


def write_capfaces(path: Path, eyes: list[dict], faces_out: list[dict], atlas: bytes) -> dict:
	eye_blob = bytearray()
	for e in eyes:
		o, a = e["origin"], e["angles"]
		eye_blob += struct.pack(">ffffff", float(o[0]), float(o[1]), float(o[2]), float(a[0]), float(a[1]), float(a[2]))
	order_keys = [(f["tex_hash"], -f["area"], i) for i, f in enumerate(faces_out)]
	order_keys.sort()
	order = [i for _, _, i in order_keys]

	hdr_size = 32
	eyes_ofs = hdr_size
	eyes_bytes = len(eye_blob)
	face_bytes = len(faces_out) * 128
	order_bytes = len(order) * 2
	faces_ofs = (eyes_ofs + eyes_bytes + 31) & ~31
	order_ofs = faces_ofs + face_bytes
	atlas_ofs = (order_ofs + order_bytes + 31) & ~31
	buf = bytearray(atlas_ofs + len(atlas))
	buf[0:4] = GC_CF_MAGIC
	struct.pack_into(">HHHH", buf, 4, GC_CF_VERSION, GC_CF_FLAG_LM_ATLAS, len(eyes), len(faces_out))
	struct.pack_into(">I", buf, 12, atlas_ofs)
	struct.pack_into(">HHHH", buf, 16, GC_LM_ATLAS_W, GC_LM_ATLAS_H, len(order), 0)
	struct.pack_into(">II", buf, 24, faces_ofs, order_ofs)
	buf[eyes_ofs : eyes_ofs + eyes_bytes] = eye_blob
	for i, f in enumerate(faces_out):
		off = faces_ofs + i * 128
		buf[off] = f["nverts"] & 0xFF
		buf[off + 1] = f["bake_src"] & 0xFF
		struct.pack_into(">H", buf, off + 2, f["flags"])
		struct.pack_into(">I", buf, off + 4, f["area"] & 0xFFFFFFFF)
		struct.pack_into(">I", buf, off + 8, f["tex_hash"])
		for v in range(GC_CAP_MAX_VERTS):
			struct.pack_into(">hhh", buf, off + 12 + v * 6, f["pts"][v * 3], f["pts"][v * 3 + 1], f["pts"][v * 3 + 2])
			struct.pack_into(">e", buf, off + 60 + v * 4, float(f["st"][v * 2]))
			struct.pack_into(">e", buf, off + 60 + v * 4 + 2, float(f["st"][v * 2 + 1]))
			struct.pack_into(">e", buf, off + 92 + v * 4, float(f["lm_uv"][v * 2]))
			struct.pack_into(">e", buf, off + 92 + v * 4 + 2, float(f["lm_uv"][v * 2 + 1]))
		buf[off + 124] = f["lm_col"]
		buf[off + 125] = f["lm_row"]
		# Byte 126: Source $envmaptint → GX specular/env blend weight (0 = unused).
		buf[off + 126] = int(f.get("spec_strength", 0)) & 0xFF
		# Byte 127: Source material class id → TEV recipe (media/tevmat.gcm).
		buf[off + 127] = int(f.get("mat_id", 0)) & 0xFF
	for i, idx in enumerate(order):
		struct.pack_into(">H", buf, order_ofs + i * 2, idx)
	buf[atlas_ofs : atlas_ofs + len(atlas)] = atlas
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_bytes(buf)
	return {"faces": len(faces_out), "eyes": len(eyes), "bytes": len(buf)}


def write_stub_bsp29(path: Path, tex_names: list[str], eye: dict, sky: str) -> None:
	"""Minimal GoldSrc BSP29 so -gcmap can mount WT/CapFaces packs by map name."""
	ox, oy, oz = eye["origin"]
	ax, ay, az = eye["angles"]
	ents = (
		"{\n"
		f'"classname" "worldspawn"\n"skyname" "{sky}"\n"message" "Lost Coast GX smoke"\n'
		"}\n"
		"{\n"
		'"classname" "info_player_start"\n'
		f'"origin" "{ox:.2f} {oy:.2f} {oz:.2f}"\n'
		f'"angles" "{ax:.2f} {ay:.2f} {az:.2f}"\n'
		"}\n"
	).encode("latin-1")

	s = 64.0
	cx, cy, cz = ox, oy, oz + 32.0
	mins = (int(cx - s), int(cy - s), int(cz - s))
	maxs = (int(cx + s), int(cy + s), int(cz + s))
	verts = [
		(cx - s, cy - s, cz - s),
		(cx + s, cy - s, cz - s),
		(cx + s, cy + s, cz - s),
		(cx - s, cy + s, cz - s),
		(cx - s, cy - s, cz + s),
		(cx + s, cy - s, cz + s),
		(cx + s, cy + s, cz + s),
		(cx - s, cy + s, cz + s),
	]
	planes = [
		((0.0, 0.0, -1.0), -(cz - s), 2),
		((0.0, 0.0, 1.0), (cz + s), 2),
		((0.0, -1.0, 0.0), -(cy - s), 1),
		((0.0, 1.0, 0.0), (cy + s), 1),
		((-1.0, 0.0, 0.0), -(cx - s), 0),
		((1.0, 0.0, 0.0), (cx + s), 0),
	]
	edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
	surfedges = [4, 5, 6, 7]  # top face verts 4-5-6-7

	names = tex_names if tex_names else ["grey"]
	pal = bytearray()
	for i in range(256):
		pal.extend((i, i, i))
	tex_blob = bytearray()
	tex_blob += struct.pack("<i", len(names))
	ofs_table = len(tex_blob)
	tex_blob += b"\0" * (len(names) * 4)
	for i, name in enumerate(names):
		while len(tex_blob) % 4:
			tex_blob += b"\0"
		struct.pack_into("<i", tex_blob, ofs_table + i * 4, len(tex_blob))
		raw_name = name.encode("latin-1")[:15]
		w = h = 16
		pix0 = bytes([128]) * (w * h)
		pix1 = bytes([128]) * ((w // 2) * (h // 2))
		pix2 = bytes([128]) * ((w // 4) * (h // 4))
		pix3 = bytes([128]) * ((w // 8) * (h // 8))
		m0, m1 = 40, 40 + len(pix0)
		m2, m3 = m1 + len(pix1), m1 + len(pix1) + len(pix2)
		mip = bytearray()
		mip += raw_name + b"\0" * (16 - len(raw_name))
		mip += struct.pack("<IIIIII", w, h, m0, m1, m2, m3)
		mip += pix0 + pix1 + pix2 + pix3
		mip += struct.pack("<H", 256)
		mip += bytes(pal)
		tex_blob += mip

	# One texinfo per texture name (mip index = i)
	texinfo = bytearray()
	for i in range(len(names)):
		texinfo += struct.pack("<8fii", 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, i, 0)

	planes_b = b"".join(struct.pack("<ffffi", n[0], n[1], n[2], dist, typ) for n, dist, typ in planes)
	verts_b = b"".join(struct.pack("<fff", *v) for v in verts)
	edges_b = b"".join(struct.pack("<hh", a, c) for a, c in edges)
	# HL face: planenum u16, side u16, firstedge i32, numedges u16, texinfo u16, styles[4], lightofs i32
	faces_b = struct.pack("<HHiHH4Bi", 1, 0, 0, 4, 0, 0, 255, 255, 255, -1)
	node = struct.pack(
		"<ihh6hHH",
		0,
		-1,
		-2,
		mins[0],
		mins[1],
		mins[2],
		maxs[0],
		maxs[1],
		maxs[2],
		0,
		1,
	)
	leaf0 = struct.pack("<ii6hHH4B", -2, -1, *mins, *maxs, 0, 0, 0, 0, 0, 0)
	leaf1 = struct.pack("<ii6hHH4B", -1, -1, *mins, *maxs, 0, 1, 0, 0, 0, 0)
	marksurfaces = struct.pack("<H", 0)
	# dmodel_t: mins[3] maxs[3] origin[3] headnode[4] visleafs firstface numfaces
	model = struct.pack(
		"<9f7i",
		float(mins[0]),
		float(mins[1]),
		float(mins[2]),
		float(maxs[0]),
		float(maxs[1]),
		float(maxs[2]),
		0.0,
		0.0,
		0.0,
		0,
		0,
		0,
		0,
		1,
		0,
		1,
	)
	clipnodes = struct.pack("<ihh", 0, -1, -2)

	lumps = [
		ents,
		planes_b,
		bytes(tex_blob),
		verts_b,
		b"",
		node,
		bytes(texinfo),
		faces_b,
		b"",
		clipnodes,
		leaf0 + leaf1,
		marksurfaces,
		edges_b,
		struct.pack(f"<{len(surfedges)}i", *surfedges),
		model,
	]
	out = bytearray(struct.pack("<i", 29) + b"\0" * (15 * 8))
	for i, lump in enumerate(lumps):
		while len(out) % 4:
			out += b"\0"
		struct.pack_into("<ii", out, 4 + i * 8, len(out), len(lump))
		out += lump
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_bytes(out)


def _material_aliases(mat_path: str) -> list[str]:
	"""Unique/overlay materials often look like maps/<map>/<mat>_<x>_<y>_<z>."""
	mat = mat_path.replace("\\", "/").lower()
	out = [mat]
	# strip trailing _digits_digits_digits on any path (patch or raw)
	stripped = re.sub(r"(?:_-?\d+){1,3}$", "", mat)
	if stripped and stripped not in out:
		out.append(stripped)
	if mat.startswith("maps/"):
		# maps/d2_lostcoast/lostcoast/monastery/foo_123_456_789 → lostcoast/monastery/foo
		rest = mat.split("/", 2)[-1] if mat.count("/") >= 2 else mat
		out.append(rest)
		out.append(re.sub(r"(?:_-?\d+){1,3}$", "", rest))
		# also try dropping the maps/<map>/ prefix keeping path after second slash
		parts = mat.split("/")
		if len(parts) >= 3:
			out.append("/".join(parts[2:]))
			out.append(re.sub(r"(?:_-?\d+){1,3}$", "", "/".join(parts[2:])))
	# de-dupe preserve order
	seen = set()
	uniq = []
	for m in out:
		if m and m not in seen:
			seen.add(m)
			uniq.append(m)
	return uniq


def material_canonical(mat_path: str) -> str:
	"""Stable base path for TEV ids (strip maps/<map>/ and VBSP _x_y_z)."""
	aliases = _material_aliases(mat_path)
	for a in aliases:
		if not a.startswith("maps/"):
			return a
	return aliases[-1]


def _load_vmt_for_mat(vpks, mat_path: str, extract_dir: Path) -> dict[str, str] | None:
	for alias in _material_aliases(mat_path):
		vmt_rel = f"materials/{alias}.vmt"
		vmt_data = read_first(vpks, vmt_rel)
		if not vmt_data:
			continue
		(extract_dir / vmt_rel).parent.mkdir(parents=True, exist_ok=True)
		(extract_dir / vmt_rel).write_bytes(vmt_data)
		return load_vmt(vmt_data)
	return None


def material_has_envmap(vmt: dict[str, str] | None) -> bool:
	"""Lost Coast sand/rock use $envmap env_cubemap (+ normalmap alpha mask), not $phong."""
	if not vmt:
		return False
	env = (vmt.get("$envmap") or "").strip().lower()
	if not env or env in ("0", "\"\"", "none"):
		return False
	# Commented-out envmap lines are already stripped by the VMT parser.
	return True


def envmap_strength_byte(vmt: dict[str, str] | None) -> int:
	"""Map $envmaptint into CapFaces face byte 126 (0..255 blend weight)."""
	if not vmt:
		return 128
	tint = (vmt.get("$envmaptint") or "").strip()
	# "[ .5 .5 .5]" / "{ 220 200 110}" (Source float vs byte colour)
	nums = re.findall(r"[-+]?\d*\.?\d+", tint)
	if len(nums) >= 3:
		avg = (float(nums[0]) + float(nums[1]) + float(nums[2])) / 3.0
		if avg > 1.0:
			return int(max(32, min(220, round(avg))))
		return int(max(32, min(220, round(avg * 255.0))))
	return 140


def resolve_texture(vpks, mat_path: str, extract_dir: Path) -> tuple[bytes, int, int] | None:
	"""Return RGBA, w, h for a material path."""
	for alias in _material_aliases(mat_path):
		vmt_rel = f"materials/{alias}.vmt"
		vmt_data = read_first(vpks, vmt_rel)
		tex_candidates = []
		if vmt_data:
			(extract_dir / vmt_rel).parent.mkdir(parents=True, exist_ok=True)
			(extract_dir / vmt_rel).write_bytes(vmt_data)
			vmt = load_vmt(vmt_data)
			tex_candidates = basetextures(vmt)
		if not tex_candidates:
			tex_candidates = [alias]
		for tex in tex_candidates:
			# Prefer albedo over bump
			if tex.endswith("_normal") or tex.endswith("_height") or "bump" in tex:
				continue
			rel = f"materials/{tex}.vtf"
			raw = read_first(vpks, rel)
			if raw is None:
				continue
			(extract_dir / rel).parent.mkdir(parents=True, exist_ok=True)
			(extract_dir / rel).write_bytes(raw)
			img = decode_vtf(raw)
			return img.rgba, img.width, img.height
		# fall through: try alias.vtf directly
		rel = f"materials/{alias}.vtf"
		raw = read_first(vpks, rel)
		if raw is not None:
			img = decode_vtf(raw)
			return img.rgba, img.width, img.height
	return None


def _build_face_records(
	bsp: dict,
	cands: list,
	short_for_mat: dict,
	tex_size: dict[str, tuple[int, int]] | None = None,
	envmap_mats: dict[str, int] | None = None,
	tevmat_ids: dict[str, int] | None = None,
) -> tuple[list[dict], bytes]:
	"""Encode ≤GC_MAX_CAP_FACES candidates into CapFaces records + LM atlas.

	Diffuse ST must be normalized by *Source* texdata WxH (texvec luxels), not
	the lean WT max_side. Dividing by encoded 32/64 made UVs tile 8–16× →
	Flipper rainbow stripes on cliffs.
	"""
	del tex_size  # retained for call-site compat; ST uses Source dims below
	atlas = bytearray(GC_LM_ATLAS_W * GC_LM_ATLAS_H * 2)
	for i in range(0, len(atlas), 2):
		atlas[i], atlas[i + 1] = 0x18, 0xC3
	faces_out: list[dict] = []
	for slot, c in enumerate(cands[:GC_MAX_CAP_FACES]):
		_score, area, _fi, mat, texinfo, _n, _dist, pts, _cx, _cy, _cz, face = c
		short = short_for_mat[mat]
		td = texinfo["texdata"]
		tw = bsp["texdatas"][td]["width"] if 0 <= td < len(bsp["texdatas"]) else 64
		th = bsp["texdatas"][td]["height"] if 0 <= td < len(bsp["texdatas"]) else 64
		if tw < 1:
			tw = 64
		if th < 1:
			th = 64
		dpts = decimate_ring(pts, GC_CAP_MAX_VERTS)
		bake_src = BAKE_EDGE if len(pts) <= GC_CAP_MAX_VERTS else BAKE_TEX
		v0, v1 = texinfo["vecs"]
		ss = [p[0] * v0[0] + p[1] * v0[1] + p[2] * v0[2] + v0[3] for p in pts]
		tt = [p[0] * v1[0] + p[1] * v1[1] + p[2] * v1[2] + v1[3] for p in pts]
		lm_w = max(1, face["lm_w"])
		lm_h = max(1, face["lm_h"])
		col = slot % GC_LM_ATLAS_COLS
		row = slot // GC_LM_ATLAS_COLS
		if row >= GC_LM_ATLAS_ROWS:
			break
		tile = sample_lm_rgb565(bsp["lighting"], face["lightofs"], lm_w, lm_h)
		for ty in range(GC_CAP_LM_DIM):
			for tx in range(GC_CAP_LM_DIM):
				src = (ty * GC_CAP_LM_DIM + tx) * 2
				dx = col * GC_CAP_LM_DIM + tx
				dy = row * GC_CAP_LM_DIM + ty
				dst = (dy * GC_LM_ATLAS_W + dx) * 2
				atlas[dst : dst + 2] = tile[src : src + 2]
		nv = len(dpts)
		pts_s16: list[int] = []
		st: list[float] = []
		lm_uv: list[float] = []
		for i in range(GC_CAP_MAX_VERTS):
			if i < nv:
				p = dpts[i]
				pts_s16.extend(
					[
						int(max(-32767, min(32767, round(p[0])))),
						int(max(-32767, min(32767, round(p[1])))),
						int(max(-32767, min(32767, round(p[2])))),
					]
				)
				s, t = tex_st(texinfo, p, float(tw), float(th))
				st.extend([s, t])
				s_lux = p[0] * v0[0] + p[1] * v0[1] + p[2] * v0[2] + v0[3]
				t_lux = p[0] * v1[0] + p[1] * v1[1] + p[2] * v1[2] + v1[3]
				ls = (s_lux - min(ss)) / 16.0
				lt = (t_lux - min(tt)) / 16.0
				ls = max(0.0, min(float(lm_w - 1), ls)) / max(1, lm_w - 1) if lm_w > 1 else 0.5
				lt = max(0.0, min(float(lm_h - 1), lt)) / max(1, lm_h - 1) if lm_h > 1 else 0.5
				u = (col * GC_CAP_LM_DIM + 0.5 + ls * (GC_CAP_LM_DIM - 1)) / GC_LM_ATLAS_W
				v = (row * GC_CAP_LM_DIM + 0.5 + lt * (GC_CAP_LM_DIM - 1)) / GC_LM_ATLAS_H
				lm_uv.extend([u, v])
			else:
				pts_s16.extend([0, 0, 0])
				st.extend([0.0, 0.0])
				lm_uv.extend([0.0, 0.0])
		face_flags = 0
		spec_strength = 0
		if envmap_mats and mat in envmap_mats:
			face_flags |= GC_SURF_ENVMAP
			spec_strength = int(envmap_mats[mat]) & 0xFF
		faces_out.append(
			{
				"nverts": nv,
				"bake_src": bake_src,
				"flags": face_flags,
				"area": area,
				"tex_hash": fnv1a32(short),
				"pts": pts_s16,
				"st": st,
				"lm_uv": lm_uv,
				"lm_col": col,
				"lm_row": row,
				"spec_strength": spec_strength,
				"mat_id": (tevmat_ids or {}).get(mat, 0),
			}
		)
	return faces_out, bytes(atlas)


def _append_eye_vista_packs(
	remaining: list,
	packs: list[list],
	eye: tuple[float, float, float],
	eye_angles: tuple[float, float, float],
	face_cap: int,
	vista_packs: int,
	max_r: float = 4000.0,
	label: str = "vista",
) -> list:
	"""Consume look-cone faces near eye into ≤vista_packs CapFaces packs."""
	if vista_packs <= 0 or not remaining:
		return remaining
	ex, ey, ez = eye
	pitch, yaw, _roll = eye_angles
	yr = math.radians(yaw)
	pr = math.radians(pitch)
	cp, sp = math.cos(pr), math.sin(pr)
	cy, sy = math.cos(yr), math.sin(yr)
	fwd = (cp * cy, cp * sy, -sp)
	max_r2 = max_r * max_r

	def dist2(c: tuple) -> float:
		dx = c[8] - ex
		dy = c[9] - ey
		dz = c[10] - ez
		return dx * dx + dy * dy + dz * dz

	def view_keep(c: tuple) -> float:
		dx = c[8] - ex
		dy = c[9] - ey
		dz = c[10] - ez
		d2 = dx * dx + dy * dy + dz * dz
		dist = math.sqrt(d2) if d2 > 1e-6 else 1.0
		view_dot = (dx * fwd[0] + dy * fwd[1] + dz * fwd[2]) / dist
		view_w = 0.08 + max(0.0, view_dot) ** 1.35
		nx, ny, nz = c[5]
		face_dot = max(0.08, -(nx * fwd[0] + ny * fwd[1] + nz * fwd[2]))
		vertical = 1.0 + 4.5 * (1.0 - min(1.0, abs(nz)))
		# Prefer elevated industrial stacks / monastery walls into the
		# dump-eye vista — floor-heavy scoring left chimneys out of densify.
		cz = float(c[10])
		elev_w = 1.0 + (2.8 if cz > 60.0 else 0.0) * min(1.0, max(0.0, (cz - 40.0) / 180.0))
		area = float(c[1])
		size_w = max(64.0, area)
		return size_w * face_dot * view_w * vertical * elev_w / (1.0 + d2 * 0.000004)

	vista_pool = [c for c in remaining if dist2(c) <= max_r2]
	vista_pool.sort(key=view_keep, reverse=True)
	for vi in range(vista_packs):
		chunk = vista_pool[:face_cap]
		if len(chunk) < max(64, face_cap // 4):
			break
		vista_pool = vista_pool[face_cap:]
		have_v = {id(c) for c in chunk}
		remaining = [c for c in remaining if id(c) not in have_v]
		packs.append(chunk)
		print(
			f"  {label} pack {vi}: {len(chunk)} faces "
			f"(eye=({ex:.0f},{ey:.0f},{ez:.0f}) yaw={yaw:.0f} "
			f"d_min={min(dist2(c) for c in chunk) ** 0.5:.0f} "
			f"d_max={max(dist2(c) for c in chunk) ** 0.5:.0f})"
		)
	return remaining


def _partition_spatial_packs(
	cands: list,
	face_cap: int,
	cell: float = 512.0,
	eye: tuple[float, float, float] | None = None,
	eye_angles: tuple[float, float, float] | None = None,
	vista_packs: int = 0,
	extra_vista_eyes: list[tuple[tuple[float, float, float], tuple[float, float, float]]] | None = None,
) -> list[list]:
	"""Split all candidates into ≤face_cap packs with spatial locality.

	When eye is set, pack 0 is view-scored for that eye (screen coverage of
	cliffs/walls in the look cone) rather than pure 3D-nearest — nearest-only
	filled CapFaces with underfoot patches and left Flipper cliffs as ribbons.

	vista_packs>0: also emit that many additional packs from the next-best
	view-cone faces so DumpFrames densify can fill cliff volume.

	extra_vista_eyes: VST / flythrough samples — each gets 2 look-cone packs so
	CapFaces densify covers the on-rails path (cove-only bake left beach void).
	"""
	remaining = list(cands)
	packs: list[list] = []
	if eye is not None and remaining:
		ex, ey, ez = eye
		pitch, yaw, _roll = eye_angles or (0.0, 0.0, 0.0)
		# Quake angles: yaw 0 → +X, yaw 90 → +Y; pitch + looks down.
		yr = math.radians(yaw)
		pr = math.radians(pitch)
		cp, sp = math.cos(pr), math.sin(pr)
		cy, sy = math.cos(yr), math.sin(yr)
		fwd = (cp * cy, cp * sy, -sp)

		def dist2(c: tuple) -> float:
			dx = c[8] - ex
			dy = c[9] - ey
			dz = c[10] - ez
			return dx * dx + dy * dy + dz * dz

		# Hard radius — beyond this, area-weighted cliffs steal the primary pack
		# (d_max≈16k left Flipper looking at empty near-eye water).
		# 2000u only held ~1.8k candidates; cliffs that fill the vista sit ~2–3.5k out.
		max_r = 4000.0
		max_r2 = max_r * max_r
		local = [c for c in remaining if dist2(c) <= max_r2]
		if len(local) < face_cap:
			local = sorted(remaining, key=dist2)[: max(face_cap * 2, face_cap)]

		# Camera basis for a cheap screen projection (yaw/pitch look).
		# right = normalize(cross(fwd, world_up)); up = cross(right, fwd)
		world_up = (0.0, 0.0, 1.0)
		rx = fwd[1] * world_up[2] - fwd[2] * world_up[1]
		ry = fwd[2] * world_up[0] - fwd[0] * world_up[2]
		rz = fwd[0] * world_up[1] - fwd[1] * world_up[0]
		rlen = math.sqrt(rx * rx + ry * ry + rz * rz) or 1.0
		right = (rx / rlen, ry / rlen, rz / rlen)
		ux = right[1] * fwd[2] - right[2] * fwd[1]
		uy = right[2] * fwd[0] - right[0] * fwd[2]
		uz = right[0] * fwd[1] - right[1] * fwd[0]
		up = (ux, uy, uz)
		# Match CapFaces DumpFrames / Source default FOV (75°), not HL1 90°.
		fov_x = math.radians(75.0)
		tan_half = math.tan(fov_x * 0.5)
		grid_w, grid_h = 20, 15

		def view_keep(c: tuple) -> float:
			dx = c[8] - ex
			dy = c[9] - ey
			dz = c[10] - ez
			d2 = dx * dx + dy * dy + dz * dz
			dist = math.sqrt(d2) if d2 > 1e-6 else 1.0
			view_dot = (dx * fwd[0] + dy * fwd[1] + dz * fwd[2]) / dist
			view_w = 0.08 + max(0.0, view_dot) ** 1.35
			nx, ny, nz = c[5]
			face_dot = max(0.08, -(nx * fwd[0] + ny * fwd[1] + nz * fwd[2]))
			# Strong cliff/wall bias — Flipper DumpFrames were sand-floor heavy.
			vertical = 1.0 + 4.5 * (1.0 - min(1.0, abs(nz)))
			cz = float(c[10])
			elev_w = 1.0 + (2.8 if cz > 60.0 else 0.0) * min(1.0, max(0.0, (cz - 40.0) / 180.0))
			area = float(c[1])
			# Prefer large covering faces (full-geo solid cliffs). Fan-tris keep
			# big rings as many small tris, so no mid-size downrank is needed.
			size_w = max(64.0, area)
			return size_w * face_dot * view_w * vertical * elev_w / (1.0 + d2 * 0.000004)

		def screen_cell(c: tuple) -> tuple[int, int] | None:
			dx = c[8] - ex
			dy = c[9] - ey
			dz = c[10] - ez
			z = dx * fwd[0] + dy * fwd[1] + dz * fwd[2]
			if z < 32.0:
				return None
			x = dx * right[0] + dy * right[1] + dz * right[2]
			y = dx * up[0] + dy * up[1] + dz * up[2]
			sx = (x / (z * tan_half) + 1.0) * 0.5
			sy = (-y / (z * tan_half) + 1.0) * 0.5
			if sx < -0.05 or sx > 1.05 or sy < -0.05 or sy > 1.05:
				return None
			col = min(grid_w - 1, max(0, int(sx * grid_w)))
			row = min(grid_h - 1, max(0, int(sy * grid_h)))
			return col, row

		# Screen-grid coverage: top face per cell, then fill by view_keep.
		# Pure nearest/area picks left CapFaces as underfoot ribbons.
		buckets: dict[tuple[int, int], list] = {}
		for c in local:
			cell_key = screen_cell(c)
			if cell_key is None:
				continue
			buckets.setdefault(cell_key, []).append(c)
		for cell_key in buckets:
			buckets[cell_key].sort(key=view_keep, reverse=True)

		primary: list = []
		have: set[int] = set()
		# Round-robin across cells so cliffs + ground both fill the frame.
		depth = 0
		while len(primary) < face_cap and buckets:
			progress = False
			for cell_key in list(buckets.keys()):
				if len(primary) >= face_cap:
					break
				bucket = buckets[cell_key]
				while bucket and id(bucket[0]) in have:
					bucket.pop(0)
				if not bucket:
					del buckets[cell_key]
					continue
				c = bucket.pop(0)
				primary.append(c)
				have.add(id(c))
				progress = True
			if not progress:
				break
			depth += 1
			if depth > 64:
				break

		n_grid = len(primary)
		if len(primary) < face_cap:
			for c in sorted(local, key=view_keep, reverse=True):
				if len(primary) >= face_cap:
					break
				if id(c) in have:
					continue
				primary.append(c)
				have.add(id(c))
		# Guarantee a near-eye grounding set if grid starved underfoot.
		n_near_pad = min(48, face_cap // 8)
		if n_near_pad > 0:
			by_near = sorted(remaining, key=dist2)
			for c in by_near:
				if n_near_pad <= 0 or len(primary) >= face_cap:
					# Replace weakest far face if at capacity.
					break
				if id(c) in have:
					continue
				if len(primary) < face_cap:
					primary.append(c)
					have.add(id(c))
					n_near_pad -= 1

		remaining = [c for c in remaining if id(c) not in have]
		packs.append(primary)
		d_min = min(dist2(c) for c in primary) ** 0.5
		d_max = max(dist2(c) for c in primary) ** 0.5
		print(
			f"  screen-grid primary: {len(primary)} faces "
			f"(grid={n_grid} cells={grid_w}x{grid_h} yaw={yaw:.0f} local={len(local)} "
			f"d_min={d_min:.0f} d_max={d_max:.0f})"
		)

		# Cove DumpFrames densify first (retail bake-eye parity). VST packs
		# take remaining faces so beach stress-cam still has densify cover
		# without stealing the cove look-cone (mae 37→61 when VST-first).
		if vista_packs > 1:
			remaining = _append_eye_vista_packs(
				remaining,
				packs,
				eye,
				(float(pitch), float(yaw), float(_roll)),
				face_cap,
				vista_packs - 1,
				max_r=max_r,
				label="vista",
			)

		if extra_vista_eyes:
			for ei, (e_origin, e_angles) in enumerate(extra_vista_eyes):
				remaining = _append_eye_vista_packs(
					remaining,
					packs,
					e_origin,
					e_angles,
					face_cap,
					1,  # one pack per VST sample — leave room for spatial fill
					max_r=2800.0,
					label=f"vst{ei}",
				)

	buckets: dict[tuple[int, int, int], list] = {}
	for c in remaining:
		key = (int(c[8] // cell), int(c[9] // cell), int(c[10] // cell))
		buckets.setdefault(key, []).append(c)
	# Larger buckets first so secondary packs stay dense near landmarks.
	keys = sorted(buckets.keys(), key=lambda k: -len(buckets[k]))
	cur: list = []
	for key in keys:
		bucket = buckets[key]
		bucket.sort(key=lambda x: -x[0])
		for c in bucket:
			if len(cur) >= face_cap:
				packs.append(cur)
				cur = []
			cur.append(c)
	if cur:
		packs.append(cur)
	return packs


def _load_vst_extra_eyes(map_bsp: Path) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
	"""Sample HL2 VST path nodes as CapFaces densify eyes (beach→monastery).

	The stress path lives in vst_lostcoast.bsp (ptrack_vst_*), not d2_lostcoast.
	Fall back to a previously baked path.bin / JSON if the VST VBSP is absent.
	"""
	nodes: list[dict] = []
	try:
		from bake_lostcoast_vst import extract_vst_path, load_source_tree
	except Exception as exc:  # noqa: BLE001
		print(f"  warn: cannot import VST path helper: {exc}")
		load_source_tree = None  # type: ignore[assignment]
		extract_vst_path = None  # type: ignore[assignment]

	vst_bsp = map_bsp.parent / "vst_lostcoast.bsp"
	if load_source_tree and extract_vst_path and vst_bsp.is_file():
		try:
			tree = load_source_tree(vst_bsp)
			_speed, nodes = extract_vst_path(tree["ents"], tree)
		except (Exception, SystemExit) as exc:  # noqa: BLE001
			print(f"  warn: VST VBSP extract failed: {exc}")
			nodes = []

	if not nodes:
		# Prefer baked path.bin next to CapFaces media (already on disc).
		for cand in (
			map_bsp.parent.parent / "cache/bake/media/vst_d2_lostcoast_path.bin",
			ROOT / "cache/bake/media/vst_d2_lostcoast_path.bin",
			ROOT / "cache/bake/media/vst_lostcoast_path.json",
		):
			if not cand.is_file():
				continue
			if cand.suffix == ".bin":
				data = cand.read_bytes()
				if len(data) < 16 or data[:5] != b"GCvst":
					continue
				n = struct.unpack_from(">H", data, 8)[0]
				off = 16
				for _ in range(n):
					ox, oy, oz, ax, ay, az = struct.unpack_from(">6f", data, off)
					nodes.append({"origin": [ox, oy, oz], "angles": [ax, ay, az]})
					off += 28
				print(f"  VST densify eyes from {cand.name} ({len(nodes)} nodes)")
				break
			if cand.suffix == ".json":
				import json

				payload = json.loads(cand.read_text())
				nodes = list(payload.get("nodes") or [])
				print(f"  VST densify eyes from {cand.name} ({len(nodes)} nodes)")
				break

	if not nodes:
		print("  warn: no VST path nodes for CapFaces densify eyes")
		return []

	def norm_pitch(p: float) -> float:
		# Source path_track often stores pitch as 356 instead of -4.
		p = float(p)
		while p > 180.0:
			p -= 360.0
		while p < -180.0:
			p += 360.0
		return p

	# Early path (beach) + mid climb + monastery — every 2nd node, max 12.
	picked = []
	for i, node in enumerate(nodes):
		if i % 2 != 0 and i not in (1, 5, 7):
			continue
		o = node["origin"]
		a = node.get("angles") or [0.0, 0.0, 0.0]
		picked.append(
			(
				(float(o[0]), float(o[1]), float(o[2])),
				(norm_pitch(a[0]), float(a[1]), float(a[2])),
			)
		)
		if len(picked) >= 12:
			break
	print(f"  VST densify eyes: {len(picked)}/{len(nodes)} path nodes")
	return picked


def bake(
	map_bsp: Path,
	lostcoast: Path,
	hl2: Path,
	out_dir: Path,
	extract_dir: Path,
	max_faces: int = GC_MAX_CAP_FACES,
	full_geo: bool = False,
	eye_override: tuple[float, float, float] | None = None,
	angles_override: tuple[float, float, float] | None = None,
	vst_eyes: bool = True,
) -> None:
	face_cap = max(1, min(int(max_faces), GC_MAX_CAP_FACES))
	print(f"==> loading {map_bsp} (full_geo={full_geo})")
	bsp = load_source_bsp(map_bsp)
	stem = map_bsp.stem.lower()
	eyes = bsp["spawns"][:6] or [
		{
			"origin": [
				sum(v[0] for v in bsp["verts"]) / max(1, len(bsp["verts"])),
				sum(v[1] for v in bsp["verts"]) / max(1, len(bsp["verts"])),
				sum(v[2] for v in bsp["verts"]) / max(1, len(bsp["verts"])) + 64,
			],
			"angles": [0.0, 0.0, 0.0],
		}
	]
	# Retail cove DumpFrames: force CapFaces spatial ranking from dump eye.
	if eye_override is not None:
		ang = list(angles_override) if angles_override is not None else [0.0, 150.0, 0.0]
		eyes = [{"origin": list(eye_override), "angles": ang}] + list(eyes)
		print(
			f"  eye override origin=({eye_override[0]:.1f},{eye_override[1]:.1f},{eye_override[2]:.1f}) "
			f"angles=({ang[0]:.1f},{ang[1]:.1f},{ang[2]:.1f})"
		)
	extra_vista: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
	if vst_eyes and full_geo:
		extra_vista = _load_vst_extra_eyes(map_bsp)
		# Score beach/climb faces into the candidate pool (facing_score uses eyes).
		for origin, angles in extra_vista[:8]:
			eyes.append({"origin": list(origin), "angles": list(angles)})

	# Collect candidate faces + materials
	skip_flags = SURF_SKY | SURF_WARP | SURF_TRANS | SURF_NODRAW | SURF_HINT | SURF_SKIP | SURF_TRIGGER
	cands = []
	mat_refs: set[str] = set()
	min_area = 64 if full_geo else 400
	for fi, face in enumerate(bsp["faces"]):
		if face["dispinfo"] != -1 or face["numedges"] < 3:
			continue
		ti = face["texinfo"]
		if ti < 0 or ti >= len(bsp["texinfos"]):
			continue
		texinfo = bsp["texinfos"][ti]
		if texinfo["flags"] & skip_flags:
			continue
		td = texinfo["texdata"]
		if td < 0 or td >= len(bsp["tex_names"]):
			continue
		mat = bsp["tex_names"][td]
		if mat.startswith("tools/") or mat.startswith("nature/water") or "water" in mat.split("/")[-1]:
			continue
		pts = face_verts(bsp, face)
		if len(pts) < 3:
			continue
		area = max(1, int(face["area"]))
		if area < min_area:
			continue
		pl = bsp["planes"][face["planenum"] % len(bsp["planes"])]
		n, dist, _ = pl
		if face["side"]:
			n = (-n[0], -n[1], -n[2])
			dist = -dist
		score = 0.0
		cx = sum(p[0] for p in pts) / len(pts)
		cy = sum(p[1] for p in pts) / len(pts)
		cz = sum(p[2] for p in pts) / len(pts)
		for eye in eyes:
			fs = facing_score(n, dist, eye["origin"])
			if fs > 0:
				dx = cx - eye["origin"][0]
				dy = cy - eye["origin"][1]
				dz = cz - eye["origin"][2]
				d2 = dx * dx + dy * dy + dz * dz
				score += area * fs / (1.0 + d2 * 0.00001)
		if score <= 0:
			if not full_geo:
				continue
			# Full geo: keep backfaces / distant walls with a weak score.
			score = float(area) * 0.01
		mat_refs.add(mat)
		# Fan-tessellate large rings before packing so CapFaces never stores an
		# 8-vert decimated ribbon for a 20-vert cliff brush.
		for tri in fan_tris(pts):
			tcx = sum(p[0] for p in tri) / len(tri)
			tcy = sum(p[1] for p in tri) / len(tri)
			tcz = sum(p[2] for p in tri) / len(tri)
			tri_area = max(1, area // max(1, len(pts) - 2)) if len(pts) > GC_CAP_MAX_VERTS else area
			cands.append((score, tri_area, fi, mat, texinfo, n, dist, tri, tcx, tcy, tcz, face))

	# Full geo: tessellate Source displacements (cliffs/terrain) into 4-vert quads.
	if full_geo:
		from source_disp import iter_disp_quads

		bsp_bytes = map_bsp.read_bytes()
		disp_n = 0
		# max_power=2 matches host full-geo solid silhouette (power=3 micro-tiles
		# left CapFaces densify as ribbons even at ~18k faces).
		for q in iter_disp_quads(bsp, bsp_bytes, max_power=2):
			mat = q["mat"]
			mat_refs.add(mat)
			nx, ny, nz = q["n"]
			ln = (nx * nx + ny * ny + nz * nz) ** 0.5 or 1.0
			nn = (nx / ln, ny / ln, nz / ln)
			dist = nn[0] * q["cx"] + nn[1] * q["cy"] + nn[2] * q["cz"]
			score = float(q["area"]) * (0.5 + max(0.0, nn[2]))
			# Synthetic face carries LM sizing from the base brush face.
			face = dict(q["face"])
			face["lm_w"] = max(1, face.get("lm_w", 4))
			face["lm_h"] = max(1, face.get("lm_h", 4))
			cands.append(
				(
					score,
					int(q["area"]),
					-1 - disp_n,
					mat,
					q["texinfo"],
					nn,
					dist,
					q["pts"],
					q["cx"],
					q["cy"],
					q["cz"],
					face,
				)
			)
			disp_n += 1
		print(f"  + displacement quads: {disp_n}")

	cands.sort(key=lambda x: -x[0])
	if full_geo:
		ex, ey, ez = eyes[0]["origin"]
		ea = eyes[0].get("angles") or [0.0, 0.0, 0.0]
		# Vista packs: top view-scored faces in the look cone, chunked into
		# face_cap packs so DumpFrames densify emits a solid cliff silhouette
		# (spatial cells alone left Flipper ribbons even at 8×768).
		# Cover the whole 4k look-cone (~16k faces) so densify can approach
		# full-geo silhouette (~19k frustum), not stop at 8×768 ribbons.
		vista_n = 32
		packs = _partition_spatial_packs(
			cands,
			face_cap,
			eye=(ex, ey, ez),
			eye_angles=(float(ea[0]), float(ea[1]), float(ea[2])),
			vista_packs=vista_n,
			extra_vista_eyes=extra_vista or None,
		)
		print(
			f"  FULL geo: {len(cands)} faces → {len(packs)} packs ×≤{face_cap} "
			f"(vista={vista_n} vst_eyes={len(extra_vista)}) materials={len(mat_refs)}"
		)
	else:
		# Smoke: single pack with spatial coverage pick
		if len(cands) > face_cap:
			cell = 512.0
			buckets: dict = {}
			for c in cands:
				key = (int(c[8] // cell), int(c[9] // cell), int(c[10] // cell))
				buckets.setdefault(key, []).append(c)
			per = max(1, face_cap // max(1, len(buckets)))
			picked = []
			for _ in range(per):
				for key in list(buckets.keys()):
					if len(picked) >= face_cap:
						break
					if buckets[key]:
						picked.append(buckets[key].pop(0))
				if len(picked) >= face_cap:
					break
			if len(picked) < face_cap:
				have = {id(x) for x in picked}
				for c in cands:
					if len(picked) >= face_cap:
						break
					if id(c) not in have:
						picked.append(c)
			cands = picked
		else:
			cands = cands[:face_cap]
		packs = [cands]
		print(f"  CapFaces candidates: {len(cands)}/{face_cap} materials: {len(mat_refs)}")

	vpk_paths = []
	for root in (lostcoast, hl2):
		vpk_paths.extend(sorted(Path(root).glob("*_dir.vpk")))
	print(f"  opening {len(vpk_paths)} VPKs")
	vpks = open_vpk_roots(vpk_paths)

	name_map: dict[str, str] = {}
	short_for_mat = {}
	rgba_cache: dict[str, tuple[bytes, int, int]] = {}
	envmap_mats: dict[str, int] = {}  # mat path → specular strength byte
	tex2_for_mat: dict[str, str] = {}  # WVT mat → short name of $basetexture2
	needed_mats = {c[3] for pack in packs for c in pack}
	tevmat_feats: dict[str, dict] = {}
	for mat in sorted(needed_mats):
		short = material_short_name(mat, name_map)
		short_for_mat[mat] = short
		vmt = _load_vmt_for_mat(vpks, mat, extract_dir)
		feat = bake_tevmat.classify(vmt)
		if feat:
			# Key by canonical base so VBSP patch variants share one TEV id.
			tevmat_feats[material_canonical(mat)] = feat
		if material_has_envmap(vmt):
			envmap_mats[mat] = envmap_strength_byte(vmt)
		got = resolve_texture(vpks, mat, extract_dir)
		# WorldVertexTransition: second layer ($basetexture2) blended per vertex
		# by displacement alpha. Baked as its own WT texture for the world DL.
		if vmt and vmt.get("$basetexture2"):
			t2 = vmt["$basetexture2"].replace("\\", "/").lower().strip()
			if t2.endswith(".vtf"):
				t2 = t2[:-4]
			raw2 = read_first(vpks, f"materials/{t2}.vtf")
			if raw2:
				img2 = decode_vtf(raw2)
				key2 = f"tex2:{t2}"
				short2 = material_short_name(key2, name_map)
				tex2_for_mat[mat] = short2
				rgba_cache[short2] = (img2.rgba, img2.width, img2.height)
				print(f"  blend layer {mat} tex2 {t2} → {short2} {img2.width}x{img2.height}")
		if got is None:
			# 8x8 grey placeholder
			rgba_cache[short] = (bytes([140, 140, 140, 255]) * 64, 8, 8)
			print(f"  missing texture for {mat} → placeholder {short}")
		else:
			rgba_cache[short] = got
			env_note = f" envmap={envmap_mats[mat]}" if mat in envmap_mats else ""
			print(f"  texture {mat} → {short} {got[1]}x{got[2]}{env_note}")
	if envmap_mats:
		print(f"  CapFaces envmap/specular mats: {len(envmap_mats)}")

	# Encode WT pack — full-geo uses 128² CMPR chains in the GCT.
	# MEM1 only keeps low mips / 8×8 stubs; R_GXHotService streams full
	# chains from ARAM after CapFaces boot (sky 6×128 CMPR still fits tip).
	# 256² × ~320 mats exceeds the 6 MiB ARAM WT store — stay at 128.
	wt_max = 128 if full_geo else 256
	hires_shorts = {short_for_mat[m] for m in needed_mats if "sand" in m.lower()}
	hires_shorts |= set(tex2_for_mat.values())
	wt_entries = []
	tex_size: dict[str, tuple[int, int]] = {}
	# Source material features → TEV material table (detail, alphatest, …).
	def _load_detail_rgba(tex: str):
		raw = read_first(vpks, f"materials/{tex}.vtf")
		if raw is None:
			return None
		img = decode_vtf(raw)
		return img.rgba, img.width, img.height

	# Weight detail/envmask slots by drawn face area so the foreground wins.
	tevmat_area: dict[str, float] = {}
	for pack in packs:
		for c in pack:
			key = material_canonical(c[3])
			tevmat_area[key] = tevmat_area.get(key, 0.0) + float(c[1])
	tevmat_ids_base, tevmat_blob, tevmat_log = bake_tevmat.build_table(
		tevmat_feats, _load_detail_rgba, weights=tevmat_area
	)
	# Expand canonical ids to every CapFaces / patch material path.
	tevmat_ids: dict[str, int] = dict(tevmat_ids_base)
	for mat in needed_mats:
		base = material_canonical(mat)
		if base in tevmat_ids_base:
			tevmat_ids[mat] = tevmat_ids_base[base]
	for line in tevmat_log:
		print(line)
	alphatest_shorts = {
		short_for_mat[m]
		for m in needed_mats
		if material_canonical(m) in tevmat_ids_base
		and tevmat_feats.get(material_canonical(m), {}).get("flags", 0)
		& bake_tevmat.F_ALPHATEST
	}
	for short, (rgba, w, h) in sorted(rgba_cache.items()):
		alpha = short in alphatest_shorts
		# Sand and blend layers are the surfaces the eye lingers on: 256² (ARAM
		# streams them; the rest stays 128² for MEM1).
		side = 256 if (short in hires_shorts and full_geo) else wt_max
		tw, th, levels, body = rgba_to_cmpr_mips(rgba, w, h, alpha, max_side=side)
		tag = int(hashlib.md5(rgba).hexdigest()[:4], 16)
		wt_entries.append((short, tw, th, levels, GC_WT_FLAG_ALPHA if alpha else 0, tag, body))
		tex_size[short] = (tw, th)
	media = out_dir / "media"
	media.mkdir(parents=True, exist_ok=True)
	gct_path = media / f"wt_{stem}.gct"
	write_gcwt(gct_path, wt_entries)
	bake_tevmat.write_table(media / "tevmat.gcm", tevmat_blob)
	# Material → TEV id map for the world display-list bake (bake_worlddl.py).
	(out_dir / "tevmat-ids.json").write_text(
		__import__("json").dumps(tevmat_ids, indent=1, sort_keys=True) + "\n"
	)
	print(f"  wrote {media / 'tevmat.gcm'} mats={len(tevmat_ids)} bytes={len(tevmat_blob)}")
	print(f"  wrote {gct_path} entries={len(wt_entries)} size={gct_path.stat().st_size}")

	# Primary + vista packs stay first (DumpFrames densify loads p01.. in order).
	# Remaining spatial packs sorted by distance from eye0.
	ex, ey, ez = eyes[0]["origin"]

	def pack_dist(pack: list) -> float:
		cx = sum(c[8] for c in pack) / len(pack)
		cy = sum(c[9] for c in pack) / len(pack)
		cz = sum(c[10] for c in pack) / len(pack)
		return (cx - ex) ** 2 + (cy - ey) ** 2 + (cz - ez) ** 2

	# Keep primary + VST + cove vista packs ahead of pure spatial cells.
	vista_keep = 56 if full_geo else 1
	if full_geo and packs:
		head = list(range(min(vista_keep, len(packs))))
		tail = sorted(range(len(head), len(packs)), key=lambda i: pack_dist(packs[i]))
		pack_order = head + tail
	else:
		pack_order = sorted(range(len(packs)), key=lambda i: pack_dist(packs[i]))
	manifest_lines = [
		f"# spatial CapFaces packs for {stem}",
		f"# faces_total={sum(len(p) for p in packs)} packs={len(packs)} face_cap={face_cap}",
		"# file cx cy cz faces",
	]
	primary_written = False
	for ord_i, pi in enumerate(pack_order):
		pack = packs[pi]
		faces_out, atlas = _build_face_records(
			bsp, pack, short_for_mat, tex_size, envmap_mats=envmap_mats, tevmat_ids=tevmat_ids
		)
		cx = sum(c[8] for c in pack) / len(pack)
		cy = sum(c[9] for c in pack) / len(pack)
		cz = sum(c[10] for c in pack) / len(pack)
		if not primary_written:
			gcp_path = media / f"cf_{stem}.gcp"
			primary_written = True
			label = gcp_path.name
		else:
			gcp_path = media / f"cf_{stem}_p{ord_i:02d}.gcp"
			label = gcp_path.name
		info = write_capfaces(gcp_path, eyes, faces_out, atlas)
		manifest_lines.append(f"{label} {cx:.1f} {cy:.1f} {cz:.1f} {info['faces']}")
		print(f"  wrote {gcp_path} faces={info['faces']} bytes={info['bytes']} center=({cx:.0f},{cy:.0f},{cz:.0f})")
	(media / f"cf_{stem}_spatial.txt").write_text("\n".join(manifest_lines) + "\n")
	print(f"  spatial manifest → media/cf_{stem}_spatial.txt ({len(packs)} packs)")

	# Stub BSP for loader
	maps_dir = out_dir / "maps"
	write_stub_bsp29(maps_dir / f"{stem}.bsp", sorted(rgba_cache.keys()), eyes[0], bsp["sky"])
	print(f"  wrote stub BSP {maps_dir / (stem + '.bsp')}")

	# Name map for debugging
	(out_dir / "tex2-names.json").write_text(
		__import__("json").dumps(tex2_for_mat, indent=1, sort_keys=True) + "\n"
	)
	(out_dir / "texture-names.json").write_text(
		__import__("json").dumps({m: short_for_mat[m] for m in sorted(short_for_mat)}, indent=2) + "\n"
	)


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--map", type=Path, default=None)
	ap.add_argument("--lostcoast", type=Path, required=True)
	ap.add_argument("--hl2", type=Path, required=True)
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--extract", type=Path, required=True)
	ap.add_argument(
		"--max-faces",
		type=int,
		default=GC_MAX_CAP_FACES,
		help=f"CapFaces budget per pack (default {GC_MAX_CAP_FACES}; lower for MEM1 tip / DumpFrames)",
	)
	ap.add_argument(
		"--full-geo",
		action="store_true",
		help="Bake ALL solid faces into spatial CapFaces packs (cf_<map>.gcp + cf_<map>_pNN.gcp)",
	)
	ap.add_argument(
		"--eye",
		type=str,
		default=None,
		help="Override CapFaces eye0 as x,y,z (spatial pack ranking + stub spawn)",
	)
	ap.add_argument(
		"--angles",
		type=str,
		default=None,
		help="Override CapFaces eye0 angles as pitch,yaw,roll (with --eye)",
	)
	ap.add_argument(
		"--no-vst-eyes",
		action="store_true",
		help="Skip VST path densify eyes (cove-only CapFaces packs)",
	)
	args = ap.parse_args()
	map_bsp = args.map or (args.lostcoast / "maps" / "d2_lostcoast.bsp")
	eye_ov = None
	ang_ov = None
	if args.eye:
		parts = [float(x) for x in args.eye.split(",")]
		if len(parts) != 3:
			raise SystemExit("--eye needs x,y,z")
		eye_ov = (parts[0], parts[1], parts[2])
	if args.angles:
		parts = [float(x) for x in args.angles.split(",")]
		if len(parts) != 3:
			raise SystemExit("--angles needs pitch,yaw,roll")
		ang_ov = (parts[0], parts[1], parts[2])
	bake(
		map_bsp,
		args.lostcoast,
		args.hl2,
		args.out,
		args.extract,
		max_faces=args.max_faces,
		full_geo=args.full_geo,
		eye_override=eye_ov,
		angles_override=ang_ov,
		vst_eyes=not args.no_vst_eyes,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
