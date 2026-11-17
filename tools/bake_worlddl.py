#!/usr/bin/env python3
"""Bake the full Source BSP into GX display lists grouped by PVS cluster.

Replaces the lean CapFaces subset: every drawable world face and
displacement of d2_lostcoast is emitted once, as a GX display list batch
keyed by (visibility cluster, material). The runtime (engine
ref/gx/r_gx_worlddl.c) walks the real BSP tree to find the eye cluster,
decompresses that cluster's PVS row, frustum-culls cluster boxes and calls
the resident display lists.

media/wd_<map>.gcd  (big-endian)

  header (64 bytes)
     0  "GCwd"  u16 version(1)  u16 flags
     8  u32 n_clusters   u32 n_chunks   u32 n_mats   u32 n_nodes
    24  u32 n_planes     u32 ofs_mats   u32 ofs_nodes   u32 ofs_planes
    40  u32 ofs_chunkdir u32 ofs_pvsdir u32 ofs_pvs     u32 ofs_chunks
    56  u32 file_size    u32 max_chunk_bytes
  mats      n_mats   x { u32 tex_hash, u32 tex2_hash (0 = none), u8 tevmat_id, u8 flags, u16 pad }
  nodes     n_nodes  x { u16 plane, s16 child0, s16 child1 }
            child >= 0: node index; child < 0: leaf, cluster = -child - 2
            (-1 = solid / no cluster)
  planes    n_planes x { f32 nx, ny, nz, dist }
  chunkdir  n_chunks x { u32 ofs (from ofs_chunks), u32 bytes,
                         s16 mins[3], s16 maxs[3], u16 cluster, u16 pad }
  pvsdir    n_clusters x u32 offset into pvs blob (0xFFFFFFFF = none)
  pvs       Source RLE rows (zero byte followed by run length)
  chunks    32-byte aligned blobs:
               u16 n_pos, u16 n_batches, u32 pos_ofs
               n_batches x { u16 mat, u16 pad, u32 dl_ofs, u32 dl_bytes }
               pos array: n_pos x s16 xyz        (32-byte aligned)
               display lists                     (each 32-byte aligned/padded)

Display list vertex (GX_VTXFMT6, file version 2), GX attribute order:
POS index16 → s16 xyz, NRM s16 xyz (fraction bits 14), CLR0 RGBA8
(light rgb + blend alpha), TEX0 s16 ST (9 fraction bits), TEX1 s16
lightmap ST (8 fraction bits; 0 when the chunk has no page).
A chunk whose bounds span at least 256 units also carries a 32² RGB565
lightmap page (GX tiled) sampled by TEX1. The runtime binds that page
for every drawn chunk, not only the ones near the camera.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from bake_lostcoast_map import (  # noqa: E402
	SURF_HINT,
	SURF_NODRAW,
	SURF_SKIP,
	SURF_SKY,
	SURF_TRIGGER,
	SURF_WARP,
	_tonemap_hdr,
	fnv1a32,
	material_canonical,
)

GX_DRAW_TRIFAN = 0xA0
GX_DRAW_TRISTRIP = 0x98
GX_VTXFMT = 6  # POS, NRM s16, CLR0 RGBA8, TEX0 s16, TEX1 lightmap s16
ST_FRAC = 9
LM_FRAC = 8
MAT_F_WATER = 0x01
MAT_F_NOLIGHT = 0x02
MAT_F_ENVMAP = 0x04
MAT_F_BASEALPHA = 0x08  # $basealphaenvmapmask — never sand IndTex
MAT_F_ENVNOMASK = 0x10  # $envmap authored without a mask: reflects everywhere
ENV_SPLIT: set[int] = set()  # material indices whose batches carry a normal
LM_PAGES_LEFT = 512


def lump(data: bytes, i: int) -> bytes:
	off, ln, _ver, _fcc = struct.unpack_from("<iiii", data, 8 + i * 16)
	return data[off : off + ln]


def pad32(b: bytearray) -> None:
	b += b"\0" * ((-len(b)) % 32)


class Bsp:
	def __init__(self, path: Path):
		d = self.data = path.read_bytes()
		assert d[:4] == b"VBSP"
		raw = lump(d, 1)
		self.planes = [struct.unpack_from("<ffff", raw, i) for i in range(0, len(raw), 20)]
		raw = lump(d, 3)
		self.verts = [struct.unpack_from("<fff", raw, i) for i in range(0, len(raw), 12)]
		raw = lump(d, 12)
		self.edges = [struct.unpack_from("<HH", raw, i) for i in range(0, len(raw), 4)]
		raw = lump(d, 13)
		self.surfedges = struct.unpack_from(f"<{len(raw) // 4}i", raw)
		raw = lump(d, 6)
		self.texinfos = []
		for i in range(0, len(raw), 72):
			v = struct.unpack_from("<16fii", raw, i)
			self.texinfos.append((v[0:4], v[4:8], v[8:12], v[12:16], v[16], v[17]))
		raw = lump(d, 2)
		self.texdatas = [struct.unpack_from("<fffiiiii", raw, i)[3:6] for i in range(0, len(raw), 32)]
		sdata = lump(d, 43)
		stab = lump(d, 44)
		self.tex_names = []
		for off in struct.unpack_from(f"<{len(stab) // 4}i", stab):
			end = sdata.index(b"\0", off)
			self.tex_names.append(sdata[off:end].decode("latin-1").replace("\\", "/").lower())
		raw = lump(d, 7)
		fmt = "<HBBIhhhh4sif5iHHI"
		self.faces = [struct.unpack_from(fmt, raw, i) for i in range(0, len(raw), 56)]
		# Prefer HDR lighting when present; Lost Coast also contains an LDR lump.
		self.lighting = lump(d, 53) or lump(d, 8)
		raw = lump(d, 5)
		self.nodes = [struct.unpack_from("<iii", raw, i) for i in range(0, len(raw), 32)]
		raw = lump(d, 10)
		# dleaf_t v1 (32 bytes): contents, cluster, area/flags, mins, maxs,
		# firstleafface, numleaffaces, firstleafbrush, numleafbrushes, water
		self.leaves = [struct.unpack_from("<ihh6hHHHHh", raw, i) for i in range(0, len(raw), 32)]
		raw = lump(d, 16)
		self.leaffaces = struct.unpack_from(f"<{len(raw) // 2}H", raw)
		raw = lump(d, 14)
		# dmodel_t: mins, maxs, origin, headnode, firstface, numfaces (48 bytes)
		self.models = [struct.unpack_from("<9fiii", raw, i) for i in range(0, len(raw), 48)]
		self.vis = lump(d, 4)
		self.numclusters = struct.unpack_from("<i", self.vis, 0)[0]
		raw = lump(d, 26)
		self.disps = []
		for i in range(0, len(raw), 176):
			start = struct.unpack_from("<fff", raw, i)
			vert_start, _tri_start, power = struct.unpack_from("<iii", raw, i + 12)
			map_face = struct.unpack_from("<H", raw, i + 36)[0]
			self.disps.append((start, vert_start, power, map_face))
		self.dispverts = lump(d, 33)
		# Displacement faces by their first edge (a face tuple does not carry
		# its own index): the start position decides how their lightmap lies.
		self.disp_start = {self.faces[mf][3]: start for (start, _vs, _pw, mf) in self.disps if 0 <= mf < len(self.faces)}

	def face_pts(self, f) -> list:
		first, num = f[3], f[4]
		pts = []
		for k in range(num):
			se = self.surfedges[first + k]
			v = self.edges[se][0] if se >= 0 else self.edges[-se][1]
			pts.append(self.verts[v])
		return pts

	def cluster_at(self, p) -> int:
		idx = 0
		while idx >= 0:
			planenum, c0, c1 = self.nodes[idx]
			nx, ny, nz, dist = self.planes[planenum]
			idx = c0 if nx * p[0] + ny * p[1] + nz * p[2] - dist >= 0.0 else c1
		return self.leaves[-1 - idx][1]


LIGHT_EXPOSURE = 2.8


def _light_texel(x: float) -> int:
	"""Source linear light → lightmap texel for the engine's x2 modulate.

	0.5 in the texel is identity, so texel = gamma(light) / 2. The old
	Reinhard curve at exposure 4 put shade at 0.57 and sun at 0.95, which
	the x2 stage turned into 1.1 and 1.9: every lit wall clipped to white
	and lost its texture. LIGHT_EXPOSURE stands in for retail's
	auto-exposure, which sits above 1 in this map.
	"""
	import math

	if not math.isfinite(x) or x <= 0.0:
		return 0
	y = 0.5 * pow(x * LIGHT_EXPOSURE, 1.0 / 2.2)
	return min(255, int(y * 255.0 + 0.5))


def light_at(bsp: Bsp, f, base_pt) -> tuple[int, int, int]:
	"""Baked light colour at a world point on face f (flat lightmap, style 0)."""
	lightofs = f[9]
	ti = bsp.texinfos[f[5]]
	if lightofs < 0 or ti[4] & 0x400:  # SURF_NOLIGHT
		return (128, 128, 128)
	lw, lh = f[13] + 1, f[14] + 1
	s = ti[2][0] * base_pt[0] + ti[2][1] * base_pt[1] + ti[2][2] * base_pt[2] + ti[2][3] - f[11]
	t = ti[3][0] * base_pt[0] + ti[3][1] * base_pt[1] + ti[3][2] * base_pt[2] + ti[3][3] - f[12]
	dl = disp_luxel(bsp, f, base_pt)
	if dl is not None:
		s, t = dl
	s = max(0.0, min(lw - 1.0, s))
	t = max(0.0, min(lh - 1.0, t))
	x0, y0 = int(s), int(t)
	x1, y1 = min(lw - 1, x0 + 1), min(lh - 1, y0 + 1)
	fx, fy = s - x0, t - y0
	acc = [0.0, 0.0, 0.0]
	lit = bsp.lighting
	for (x, y, w) in ((x0, y0, (1 - fx) * (1 - fy)), (x1, y0, fx * (1 - fy)), (x0, y1, (1 - fx) * fy), (x1, y1, fx * fy)):
		o = lightofs + (y * lw + x) * 4
		if o + 4 > len(lit):
			continue
		r, g, b, e = lit[o], lit[o + 1], lit[o + 2], lit[o + 3]
		e = e - 256 if e > 127 else e
		sc = (2.0 ** e) / 255.0
		acc[0] += r * sc * w
		acc[1] += g * sc * w
		acc[2] += b * sc * w
	return (_light_texel(acc[0]), _light_texel(acc[1]), _light_texel(acc[2]))


def rgb565(c) -> int:
	return ((c[0] >> 3) << 11) | ((c[1] >> 2) << 5) | (c[2] >> 3)


def pack_nrm(n) -> tuple[int, int, int]:
	x, y, z = n
	length = math.sqrt(x * x + y * y + z * z) or 1.0
	scale = 16384.0 / length  # s16 fraction bits 14
	return tuple(int(max(-32767, min(32767, round(c * scale)))) for c in (x, y, z))


def disp_luxel(bsp: Bsp, f, pt):
	"""Luxel coordinates of a base point on a displacement face, or None.

	A displacement's lightmap does not follow the face's lightmap vectors.
	Source lays it over the displacement's own grid: with the face corners
	taken from the one nearest the displacement's start position, corner 0
	is luxel (0, 0), corner 1 (0, H), corner 2 (W, H), corner 3 (W, 0). The
	vectors gave some sand patches a flipped lightmap: the cliff's shadow
	landed on the wrong side of the patch and met its lit neighbour in a
	hard straight seam.
	"""
	start = bsp.disp_start.get(f[3])
	if start is None:
		return None
	pts = bsp.face_pts(f)
	if len(pts) != 4:
		return None
	k = min(range(4), key=lambda i: sum((pts[i][a] - start[a]) ** 2 for a in range(3)))
	p0, p1, _p2, p3 = pts[k:] + pts[:k]
	e_s = [p3[a] - p0[a] for a in range(3)]
	e_t = [p1[a] - p0[a] for a in range(3)]
	d = [pt[a] - p0[a] for a in range(3)]
	ls = sum(c * c for c in e_s) or 1.0
	lt = sum(c * c for c in e_t) or 1.0
	u = sum(d[a] * e_s[a] for a in range(3)) / ls
	v = sum(d[a] * e_t[a] for a in range(3)) / lt
	return u * f[13], v * f[14]


def luxel_uv(bsp: Bsp, f, pt) -> tuple[float, float, int, int]:
	ti = bsp.texinfos[f[5]]
	lw, lh = f[13] + 1, f[14] + 1
	dl = disp_luxel(bsp, f, pt)
	if dl is not None:
		return dl[0], dl[1], lw, lh
	s = ti[2][0] * pt[0] + ti[2][1] * pt[1] + ti[2][2] * pt[2] + ti[2][3] - f[11]
	t = ti[3][0] * pt[0] + ti[3][1] * pt[1] + ti[3][2] * pt[2] + ti[3][3] - f[12]
	return s, t, lw, lh


def luxel_color(bsp: Bsp, f, x: int, y: int) -> tuple[int, int, int]:
	lw = f[13] + 1
	o = f[9] + (y * lw + x) * 4
	lit = bsp.lighting
	if o < 0 or o + 4 > len(lit):
		return (128, 128, 128)
	r, g, b, e = lit[o], lit[o + 1], lit[o + 2], lit[o + 3]
	e = e - 256 if e > 127 else e
	sc = (2.0 ** e) / 255.0
	return (_light_texel(r * sc), _light_texel(g * sc), _light_texel(b * sc))


def face_samples(bsp: Bsp, f):
	lw, lh = f[13] + 1, f[14] + 1
	if f[9] < 0 or lw < 2 or lh < 2 or lw * lh > 65536:
		return None
	return [[luxel_color(bsp, f, x, y) for x in range(lw)] for y in range(lh)]


def cell_st(cell, s: float, t: float, lw: int, lh: int) -> tuple[float, float]:
	x, y, w, h = cell
	sn = min(1.0, max(0.0, s / max(lw - 1, 1)))
	tn = min(1.0, max(0.0, t / max(lh - 1, 1)))
	u = (x + sn * max(w - 1, 0) + 0.5) / 32.0
	v = (y + tn * max(h - 1, 0) + 0.5) / 32.0
	return u, v


def rgb565_page(pix) -> bytes:
	"""32×32 RGB565 in GX 4×4 tiles."""
	out = bytearray()
	for ty in range(0, 32, 4):
		for tx in range(0, 32, 4):
			for y in range(4):
				for x in range(4):
					r, g, b = pix[ty + y][tx + x]
					out += struct.pack(">H", ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3))
	return bytes(out)


DISP_CELLS: list = []  # (face, cluster, cell side) of every displacement, for the bake log


# Lightmap cells are planned per page before they are handed out. Faces used
# to take cells first come first served, 16 x 16 for a big patch, until the
# 32 x 32 page was full: of the map's 525 displacement patches 156 got no cell
# at all and drew with vertex light beside lightmapped neighbours, the hard
# seams where sand patches meet. bake() now runs twice: the first run only
# records what each page is asked for (LM_PLAN is None), then every page's
# requests are shrunk by one common factor until they all fit, and the second
# run hands out the planned cells. Pages are numbered in creation order,
# which is the same in both runs.
LM_PLAN: dict | None = None
LM_REQUESTS: dict[int, list[int]] = {}
LM_COUNTER = [0]


def plan_lightmap_pages() -> dict:
	plan = {}
	for page, sides in LM_REQUESTS.items():
		for scale in (1.0, 0.875, 0.75, 0.625, 0.5, 0.4375, 0.375, 0.3125, 0.25, 0.1875, 0.125):
			want = [max(2, int(round(sd * scale))) for sd in sides]
			order = sorted(range(len(want)), key=lambda i: -want[i])
			rects = [None] * len(want)
			x = y = row_h = 0
			ok = True
			for i in order:
				w = want[i]
				if x + w > 32:
					x, y, row_h = 0, y + row_h, 0
				if y + w > 32:
					ok = False
					break
				rects[i] = (x, y, w, w)
				x += w
				row_h = max(row_h, w)
			if ok:
				break
		plan[page] = rects
	return plan


class LmPage:
	def __init__(self) -> None:
		# Mid-grey pad — unused atlas texels used to be white, so any vert
		# with lu,lv≈0 (failed cell / unset) lit full-bright and washed cliffs.
		self.pix = [[(128, 128, 128) for _ in range(32)] for _ in range(32)]
		self.x = self.y = self.row_h = 0
		self.used = False
		self.failed = False
		self.index = LM_COUNTER[0]
		LM_COUNTER[0] += 1
		self.seq = 0

	def _try(self, w: int, h: int):
		x, y, rh = self.x, self.y, self.row_h
		if x + w > 32:
			x = 0
			y += rh
			rh = 0
		if y + h > 32:
			return None
		self.x, self.y, self.row_h = x + w, y, max(rh, h)
		return (x, y, w, h)

	def alloc(self, w: int, h: int):
		if LM_PLAN is None:
			# Recording run: note the size wanted, hand out a placeholder.
			LM_REQUESTS.setdefault(self.index, []).append(max(2, min(16, w, h)))
			self.used = True
			return (0, 0, 2, 2)
		rects = LM_PLAN.get(self.index)
		rect = rects[self.seq] if rects and self.seq < len(rects) else None
		self.seq += 1
		if rect:
			self.used = True
		return rect
		if self.failed:
			return None
		side = max(2, min(16, w, h))
		for s in (side, 8, 4, 2):
			rect = self._try(s, s)
			if rect:
				self.used = True
				return rect
		self.failed = True
		return None

	def blit(self, rect, samples) -> None:
		x, y, w, h = rect
		lh = len(samples)
		lw = len(samples[0]) if lh else 1
		for py in range(h):
			fy = py * (lh - 1) / max(h - 1, 1)
			y0 = int(fy)
			y1 = min(lh - 1, y0 + 1)
			fy -= y0
			for px in range(w):
				fx = px * (lw - 1) / max(w - 1, 1)
				x0 = int(fx)
				x1 = min(lw - 1, x0 + 1)
				fx -= x0
				self.pix[y + py][x + px] = tuple(round(
					(samples[y0][x0][c] * (1 - fx) + samples[y0][x1][c] * fx) * (1 - fy)
					+ (samples[y1][x0][c] * (1 - fx) + samples[y1][x1][c] * fx) * fy
				) for c in range(3))


def stitch_displacement_lightmaps(bsp, records) -> tuple[int, int, int]:
	"""Soften lighting edges on contiguous, same-material displacements."""
	edges = {}
	for f, material, page, cell, grid, refs in records:
		center = tuple((grid[0][0][3][k] + grid[-1][-1][3][k]) * 0.5 for k in range(3))
		for edge in (grid[0], grid[-1], [r[0] for r in grid], [r[-1] for r in grid]):
			start = tuple(round(v) for v in edge[0][0])
			end = tuple(round(v) for v in edge[-1][0])
			if start == end:
				continue
			if end < start:
				edge = list(reversed(edge))
				start, end = end, start
			edges.setdefault((material, start, end), []).append((f, page, cell, edge, center, refs))

	updates = {}
	vertex_updates = {}
	pairs = 0
	for neighbors in edges.values():
		if len(neighbors) != 2:
			continue
		left, right = neighbors
		if math.dist(left[3][len(left[3]) // 2][0], right[3][len(right[3]) // 2][0]) > 2.0:
			continue  # The displaced edges have different shapes.
		steps = 2 * max(*(max(rec[2][2:4]) if rec[2] else len(rec[3]) for rec in (left, right)))
		profile = []
		for k in range(steps + 1):
			t = k / steps
			points = [tuple(rec[3][0][3][j] * (1 - t) + rec[3][-1][3][j] * t for j in range(3))
				for rec in (left, right)]
			colors = [light_at(bsp, rec[0], point) for rec, point in zip((left, right), points)]
			profile.append((points, colors))
		mean_difference = sum(max(abs(a - b) for a, b in zip(*colors))
			for _, colors in profile) / len(profile)
		if mean_difference < 32:
			continue
		pairs += 1
		# Some chunks cannot fit an atlas page and use vertex lighting.
		# Patch their shared vertices too, preserving displacement alpha.
		for rec, other in ((left, right), (right, left)):
			f, _, _, edge, _, refs = rec
			for i, vertex in enumerate(edge):
				t = i / (len(edge) - 1)
				peer_base = tuple(other[3][0][3][j] * (1 - t) + other[3][-1][3][j] * t for j in range(3))
				own, peer = light_at(bsp, f, vertex[3]), light_at(bsp, other[0], peer_base)
				target = tuple((a + b) / 2 for a, b in zip(own, peer))
				key = tuple(round(v) for v in vertex[0])
				for dl, offset in refs.get(key, []):
					vertex_updates.setdefault((id(dl), offset), (dl, []))[1].append(target)
		for points, colors in profile:
			target = tuple((a + b) / 2 for a, b in zip(*colors))
			for rec, base in zip((left, right), points):
				f, page, cell, _, center, _ = rec
				if cell is None:
					continue
				for inward, weight in ((0.0, 1.0), (1.0 / max(cell[2], cell[3]), 0.5)):
					p = tuple(base[j] + (center[j] - base[j]) * inward for j in range(3))
					s, t, lw, lh = luxel_uv(bsp, f, p)
					u, v = cell_st(cell, s, t, lw, lh)
					x = max(cell[0], min(cell[0] + cell[2] - 1, round(u * 32 - 0.5)))
					y = max(cell[1], min(cell[1] + cell[3] - 1, round(v * 32 - 0.5)))
					updates.setdefault((page, x, y), []).append((target, weight))
	for (page, x, y), values in updates.items():
		original = page.pix[y][x]
		page.pix[y][x] = tuple(round(sum(original[c] * (1 - w) + t[c] * w for t, w in values) / len(values))
			for c in range(3))
	for (_, offset), (dl, colors) in vertex_updates.items():
		for c in range(3):
			dl[offset + c] = max(0, min(255, round(sum(v[c] for v in colors) / len(colors))))
	return pairs, len(updates), len(vertex_updates)


def displacement_normals(grid, face_normal):
	"""Estimate surface normals from displaced positions, oriented to the face.

	Source dispvert.vec is a displacement direction, not a shading normal.
	Use centered differences inside the grid and one-sided differences at its
	edges; fall back to the face normal for degenerate geometry.
	"""
	rows, cols = len(grid), len(grid[0])
	out = []
	for j in range(rows):
		row = []
		for i in range(cols):
			left, right = grid[j][max(0, i - 1)][0], grid[j][min(cols - 1, i + 1)][0]
			top, bottom = grid[max(0, j - 1)][i][0], grid[min(rows - 1, j + 1)][i][0]
			u = [right[k] - left[k] for k in range(3)]
			v = [bottom[k] - top[k] for k in range(3)]
			n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
			length = math.sqrt(sum(x * x for x in n))
			if length < 1e-8:
				n = list(face_normal)
			else:
				sign = -1.0 if sum(n[k] * face_normal[k] for k in range(3)) < 0 else 1.0
				n = [x * sign / length for x in n]
			row.append(tuple(n))
		out.append(row)
	return out


def oct12(n) -> int:
	"""Unit vector -> 12 bits (octahedral, 6 + 6)."""
	x, y, z = n
	l = abs(x) + abs(y) + abs(z) or 1.0
	x, y, z = x / l, y / l, z / l
	if z < 0.0:
		x, y = (1.0 - abs(y)) * (1.0 if x >= 0 else -1.0), (1.0 - abs(x)) * (1.0 if y >= 0 else -1.0)
	return (int(round((x * 0.5 + 0.5) * 63.0)) << 6) | int(round((y * 0.5 + 0.5) * 63.0))


def oct12_decode(code: int):
	x = ((code >> 6) & 63) / 63.0 * 2.0 - 1.0
	y = (code & 63) / 63.0 * 2.0 - 1.0
	z = 1.0 - abs(x) - abs(y)
	if z < 0.0:
		x, y = (1.0 - abs(y)) * (1.0 if x >= 0 else -1.0), (1.0 - abs(x)) * (1.0 if y >= 0 else -1.0)
	l = math.sqrt(x * x + y * y + z * z) or 1.0
	return (x / l, y / l, z / l)


def tex_orient(n, tex_axes) -> int:
	"""How the texture's S and T run in the reflection plane's frame, 3 bits.

	A normal map's X is along the surface's texture S and its Y along T
	(Source's tangent space). The engine offsets the cubemap lookup in the
	frame ( t, b ) of the plane axis nearest the batch normal, so it needs
	S and T in that frame: the nearest of the eight axis-aligned layouts.
	bit 0: S runs along b (swap), bit 1: first row negative, bit 2: second.
	The sand is mapped S = +x, T = -y: code 4, i.e. Y flipped.
	"""
	if not tex_axes:
		return 0
	from bake_lostcoast_cubemaps import AXES, axis_frame, nearest_axis
	t, b = axis_frame(AXES[nearest_axis(n)])
	def unit(v):
		l = math.sqrt(sum(c * c for c in v)) or 1.0
		return [c / l for c in v]
	sv, tv = unit(tex_axes[0][:3]), unit(tex_axes[1][:3])
	j = [[sum(sv[k] * t[k] for k in range(3)), sum(tv[k] * t[k] for k in range(3))],
		[sum(sv[k] * b[k] for k in range(3)), sum(tv[k] * b[k] for k in range(3))]]
	best, code = -9.0, 0
	for o in range(8):
		sx, sy = (-1.0 if o & 2 else 1.0), (-1.0 if o & 4 else 1.0)
		score = (sx * j[0][1] + sy * j[1][0]) if o & 1 else (sx * j[0][0] + sy * j[1][1])
		if score > best:
			best, code = score, o
	return code


def env_code(verts, tex_axes=None, base_normal=None) -> int:
	"""Batch word for a reflective primitive: 0x8000 | texture layout << 12 |
	its normal, or 0.

	A flat brush face keeps its own normal, so its reflection is exact.
	Displacement strips curve: they go to the nearest of a few directions
	(up, or 22.5 degree steps round the compass, level or raised 45 degrees)
	so that a rock face is a handful of batches, not one per triangle.
	Surfaces that face down get no reflection pass.
	"""
	if base_normal is not None:
		# A displacement: every strip reflects about the plane of the face
		# it was built on (the normal map supplies the relief). Binning each
		# strip by its own tilt put neighbouring strips of one beach up to
		# 45 degrees apart, which showed as bands where patches meet.
		n = list(base_normal)
		if n[2] < -0.5:
			return 0
		return 0x8000 | (tex_orient(n, tex_axes) << 12) | oct12(n)
	n = [sum(v[2][k] for v in verts) for k in range(3)]
	l = math.sqrt(sum(c * c for c in n))
	if l < 1e-6:
		return 0
	n = [c / l for c in n]
	if n[2] < -0.5:
		return 0
	n0 = verts[0][2]
	l0 = math.sqrt(sum(c * c for c in n0)) or 1.0
	flat = all(
		abs(v[2][k] / (math.sqrt(sum(c * c for c in v[2])) or 1.0) - n0[k] / l0) < 1e-3
		for v in verts for k in range(3)
	)
	if not flat:
		if n[2] >= 0.92:
			n = [0.0, 0.0, 1.0]
		else:
			step = math.pi / 8.0
			az = round(math.atan2(n[1], n[0]) / step) * step
			el = math.pi / 4.0 if n[2] > 0.38 else 0.0
			n = [math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)]
	return 0x8000 | (tex_orient(n, tex_axes) << 12) | oct12(n)


class Chunk:
	"""All geometry assigned to one PVS cluster."""

	def __init__(self) -> None:
		self.pos: dict[tuple[int, int, int], int] = {}
		self.batches: dict[int, bytearray] = {}
		self.mins = [32767, 32767, 32767]
		self.maxs = [-32768, -32768, -32768]
		self.tris = 0
		self.lm = LmPage()

	def pos_index(self, p) -> int:
		key = (
			int(max(-32767, min(32767, round(p[0])))),
			int(max(-32767, min(32767, round(p[1])))),
			int(max(-32767, min(32767, round(p[2])))),
		)
		idx = self.pos.get(key)
		if idx is None:
			idx = self.pos[key] = len(self.pos)
			for a in range(3):
				self.mins[a] = min(self.mins[a], key[a])
				self.maxs[a] = max(self.maxs[a], key[a])
		return idx

	def emit(self, mat: int, prim: int, verts: list, color_refs=None, tex_axes=None, base_normal=None) -> None:
		"""verts: (pos, rgba, nrm, s, t, lm_u, lm_v)."""
		# Reflective surfaces are batched by their normal: the engine mirrors
		# the view about it and looks the cubemap up with one projective
		# texgen per batch (see env_code).
		key = mat | ((env_code(verts, tex_axes, base_normal) << 16) if mat in ENV_SPLIT else 0)
		dl = self.batches.setdefault(key, bytearray())
		dl += struct.pack(">BH", prim | GX_VTXFMT, len(verts))
		for p, c, nrm, s, t, lu, lv in verts:
			nx, ny, nz = pack_nrm(nrm)
			if color_refs is not None:
				key = tuple(round(v) for v in p)
				if key in color_refs:
					color_refs[key].append((dl, len(dl) + 5))
			# 15 bytes per vertex: the chunk cache is the scarce resource, so
			# normals are s8 (6 fraction bits) and lightmap UV u8, not s16.
			dl += struct.pack(
				">Hbbb4BhhBB",
				self.pos_index(p), nx >> 8, ny >> 8, nz >> 8,
				max(0, min(255, c[0])), max(0, min(255, c[1])),
				max(0, min(255, c[2])), max(0, min(255, c[3])),
				int(max(-32768, min(32767, round(s * (1 << ST_FRAC))))),
				int(max(-32768, min(32767, round(t * (1 << ST_FRAC))))),
				int(max(0, min(255, round(lu * (1 << LM_FRAC))))),
				int(max(0, min(255, round(lv * (1 << LM_FRAC))))),
			)
		self.tris += max(0, len(verts) - 2)

	def encode(self) -> bytes:
		mats = sorted(self.batches)
		out = bytearray(16 + 12 * len(mats))
		pad32(out)
		pos_ofs = len(out)
		for key in self.pos:  # insertion order == index order
			out += struct.pack(">hhh", *key)
		pad32(out)
		for i, m in enumerate(mats):
			dl = bytearray(self.batches[m])
			pad32(dl)  # GX NOPs
			struct.pack_into(">HHII", out, 16 + 12 * i, m & 0xFFFF, m >> 16, len(out), len(dl))
			out += dl
		lm_ofs = 0
		# Pages are assigned before encode(), largest walls first, so the
		# 512-page budget is not spent on whichever cluster id came first.
		# The page itself is NOT in the chunk: it lives in a separate stream
		# after all chunks (2048 bytes each) and the runtime fetches it only
		# when the chunk is drawn. The header field is page index + 1.
		if getattr(self, "lm_keep", False):
			lm_ofs = self.lm_index + 1
		struct.pack_into(">HHII", out, 0, len(self.pos), len(mats), pos_ofs, lm_ofs)
		return bytes(out)


def st_for(ti, tw, th, pts) -> list[tuple[float, float]]:
	"""Texture repeats per point, re-centred so the face starts near 0."""
	st = [
		(
			(ti[0][0] * p[0] + ti[0][1] * p[1] + ti[0][2] * p[2] + ti[0][3]) / tw,
			(ti[1][0] * p[0] + ti[1][1] * p[1] + ti[1][2] * p[2] + ti[1][3]) / th,
		)
		for p in pts
	]
	s0 = math.floor(min(s for s, _ in st))
	t0 = math.floor(min(t for _, t in st))
	return [(s - s0, t - t0) for s, t in st]


def load_envmap(names: dict, cube_ix: dict[str, int], pak) -> tuple[dict[str, int], set[str], list, set[str]]:
	"""Material path -> packed envmap word, the $basealphaenvmapmask set and
	the tint palette.

	Word: ( cubemap index + 1 ) << 8 | tint palette index. The cubemap is the
	one VBSP patched into the material ("$envmap" "maps/<map>/cX_Y_Z"); the
	tint is $envmaptint to the power 2 / 2.2 (see bake_lostcoast_cubemaps).
	Materials without a patched cubemap get no reflection, as on PC.
	"""
	out: dict[str, int] = {}
	basealpha: set[str] = set()
	nomask: set[str] = set()
	palette: list[tuple[int, int, int]] = []
	from bake_lostcoast_map import _load_vmt_for_mat
	from vpk import open_vpk_roots
	import bake_tevmat

	roots = sorted((ROOT / "content" / "lostcoast").glob("*_dir.vpk"))
	roots += sorted((ROOT / "content" / "hl2").glob("*_dir.vpk"))
	vpks = open_vpk_roots(roots)
	extract = ROOT / "cache" / "extracted"
	pak_names = {n.lower(): n for n in pak.namelist()}
	for name in names:
		vmt = _load_vmt_for_mat(vpks, name, extract)
		if not vmt:
			continue
		# VBSP's patch material in the BSP pak names the cubemap.
		env = ""
		pk = "materials/" + name.lower() + ".vmt"
		if pk in pak_names:
			m = re.search(r'"\$envmap"\s+"([^"]+)"', pak.read(pak_names[pk]).decode("latin-1"), re.I)
			if m:
				env = m.group(1).strip().lower().replace("\\", "/")
		if env not in cube_ix:
			continue
		nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", vmt.get("$envmaptint") or "1 1 1")][:3]
		if len(nums) < 3:
			nums = [1.0, 1.0, 1.0]
		if "{" in (vmt.get("$envmaptint") or ""):
			nums = [v / 255.0 for v in nums]  # { r g b } is a byte colour
		# Engine constant = tint^2 x LIGHT_EXPOSURE / 2 (linear; the TEV scales
		# make up the rest, see R_GXWorldDLEnvOn).
		tint = tuple(int(max(0, min(255, round(255.0 * 0.5 * LIGHT_EXPOSURE * max(0.0, v) ** 2.0)))) for v in nums)
		# ( cube x tint )^2 with a tint this small is under one step of the
		# frame buffer even at the sun: no pass.
		if max(tint) <= 2:
			continue
		if tint not in palette:
			if len(palette) >= 16:
				continue
			palette.append(tint)
		out[name] = ((cube_ix[env] + 1) << 8) | palette.index(tint)
		if bake_tevmat._num(vmt.get("$basealphaenvmapmask"), 0.0) >= 1.0:
			basealpha.add(name)
		elif bake_tevmat._num(vmt.get("$normalmapalphaenvmapmask"), 0.0) < 1.0 and not (vmt.get("$envmapmask") or "").strip():
			nomask.add(name)
	print(f"  envmap materials: {len(out)} basealpha={len(basealpha)} nomask={len(nomask)} tints={palette}")
	return out, basealpha, palette, nomask


def tevmat_id_for(name: str, tevmat_ids: dict[str, int]) -> int:
	"""Look up a TEV material id, following VBSP patch → base aliases."""
	if name in tevmat_ids:
		return int(tevmat_ids[name])
	base = material_canonical(name)
	if base in tevmat_ids:
		return int(tevmat_ids[base])
	try:
		from bake_lostcoast_map import _material_aliases
	except Exception:
		return 0
	for alias in _material_aliases(name):
		if alias in tevmat_ids:
			return int(tevmat_ids[alias])
	return 0


def bake(map_bsp: Path, out_dir: Path, max_disp_power: int) -> None:
	bsp = Bsp(map_bsp)
	stem = map_bsp.stem
	names = json.loads((out_dir / "texture-names.json").read_text())
	ids_path = out_dir / "tevmat-ids.json"
	tevmat_ids = json.loads(ids_path.read_text()) if ids_path.is_file() else {}

	mat_index: dict[str, int] = {}
	mat_recs: list[tuple[int, int, int, int, int]] = []
	t2_path = out_dir / "tex2-names.json"
	tex2_names = json.loads(t2_path.read_text()) if t2_path.is_file() else {}
	from bake_lostcoast_cubemaps import cubemap_index
	from bake_lostcoast_cubemaps import read_pak
	_bsp_raw = map_bsp.read_bytes()
	envmap, basealpha_mats, env_palette, nomask_mats = load_envmap(names, cubemap_index(_bsp_raw, stem), read_pak(_bsp_raw))
	# Tint palette for the engine: media/et_<map>.bin, 16 x RGBA.
	(out_dir / "media").mkdir(parents=True, exist_ok=True)
	(out_dir / "media" / f"et_{stem}.bin").write_bytes(
		b"".join(bytes((r, g, b, 255)) for r, g, b in env_palette).ljust(64, b"\0"))

	def mat_for(texdata: int, flags: int) -> int | None:
		name = bsp.tex_names[bsp.texdatas[texdata][0]]
		if name.startswith("tools/") or name not in names:
			return None
		if name not in mat_index:
			mflags = (MAT_F_WATER if flags & SURF_WARP else 0) | (MAT_F_NOLIGHT if flags & 0x400 else 0)
			strength = int(envmap.get(name, 0))
			if strength and not (mflags & MAT_F_WATER):
				mflags |= MAT_F_ENVMAP
				if name in basealpha_mats:
					mflags |= MAT_F_BASEALPHA
				if name in nomask_mats:
					mflags |= MAT_F_ENVNOMASK
				ENV_SPLIT.add(len(mat_recs))
			mat_index[name] = len(mat_recs)
			t2 = tex2_names.get(name)
			mat_recs.append((fnv1a32(names[name]), fnv1a32(t2) if t2 else 0,
				tevmat_id_for(name, tevmat_ids), mflags, strength & 0xFFFF))
		return mat_index[name]

	def mat_named(name: str) -> int | None:
		if name not in names:
			return None
		if name not in mat_index:
			mat_index[name] = len(mat_recs)
			t2 = tex2_names.get(name)
			mat_recs.append((fnv1a32(names[name]), fnv1a32(t2) if t2 else 0,
				tevmat_id_for(name, tevmat_ids), 0, 0))
		return mat_index[name]

	# Face → cluster from the leaf face lists (first leaf that references it).
	face_cluster: dict[int, int] = {}
	for leaf in bsp.leaves:
		cluster = leaf[1]
		if cluster < 0:
			continue
		for k in range(leaf[9], leaf[9] + leaf[10]):
			face_cluster.setdefault(bsp.leaffaces[k], cluster)

	chunks: dict[int, Chunk] = {}
	skip_flags = SURF_SKY | SURF_NODRAW | SURF_HINT | SURF_SKIP | SURF_TRIGGER
	world_first, world_num = bsp.models[0][10], bsp.models[0][11]
	n_faces = n_disp = n_orphans = n_flat_blend = 0
	face_ctx: dict[int, tuple] = {}  # brush face → (cluster, lightmap cell, normal)
	ground_tris: list = []  # upward-facing world triangles, for the water depth map
	disp_light_records = []

	for fi in range(world_first, world_first + world_num):
		f = bsp.faces[fi]
		ti = bsp.texinfos[f[5]]
		if ti[4] & skip_flags:
			continue
		mat = mat_for(ti[5], ti[4])
		if mat is None:
			continue
		tw, th = bsp.texdatas[ti[5]][1] or 64, bsp.texdatas[ti[5]][2] or 64
		pts = bsp.face_pts(f)
		if len(pts) < 3:
			continue
		water = bool(ti[4] & SURF_WARP)

		if 0 <= f[6] < len(bsp.disps):
			start, vert_start, power, _mf = bsp.disps[f[6]]
			if len(pts) != 4:
				continue
			i0 = min(range(4), key=lambda i: sum((pts[i][a] - start[a]) ** 2 for a in range(3)))
			c = [pts[(i0 + k) % 4] for k in range(4)]
			full = (1 << power) + 1
			step = 1 << max(0, power - max_disp_power)
			n = (full - 1) // step + 1
			grid = []
			for j in range(n):
				row = []
				v = (j * step) / (full - 1)
				for i in range(n):
					u = (i * step) / (full - 1)
					# Source: rows run corner0→corner1, columns corner0→corner3.
					a = [c[0][k] + (c[1][k] - c[0][k]) * v for k in range(3)]
					b = [c[3][k] + (c[2][k] - c[3][k]) * v for k in range(3)]
					base = [a[k] + (b[k] - a[k]) * u for k in range(3)]
					o = (vert_start + (j * step) * full + (i * step)) * 20
					vx, vy, vz, dist, alpha = struct.unpack_from("<5f", bsp.dispverts, o)
					p = (base[0] + vx * dist, base[1] + vy * dist, base[2] + vz * dist)
					row.append((p, light_at(bsp, f, base) + (int(max(0, min(255, round(alpha)))),),
						(vx, vy, vz), base))
				grid.append(row)
			plane = bsp.planes[f[0]]
			sign = -1.0 if f[1] else 1.0
			normals = displacement_normals(grid, tuple(plane[k] * sign for k in range(3)))
			grid = [[(item[0], item[1], normals[j][i], item[3])
				for i, item in enumerate(row)] for j, row in enumerate(grid)]
			# Texture coordinates use displaced world positions, not normals.
			flat = [g[0] for row in grid for g in row]
			sts = st_for(ti, tw, th, flat)
			mid = grid[n // 2][n // 2][0]
			cluster = bsp.cluster_at((mid[0], mid[1], mid[2] + 16.0))
			if cluster < 0:
				for probe in (grid[0][0][0], grid[0][-1][0], grid[-1][0][0], grid[-1][-1][0]):
					cluster = bsp.cluster_at((probe[0], probe[1], probe[2] + 32.0))
					if cluster >= 0:
						break
			if cluster < 0:
				n_orphans += 1
				continue
			ch = chunks.setdefault(cluster, Chunk())
			samples = face_samples(bsp, f)
			cell = ch.lm.alloc(f[13] + 1, f[14] + 1) if samples else None
			DISP_CELLS.append((fi, cluster, cell[2] if cell else 0))
			if cell and samples:
				ch.lm.blit(cell, samples)
			edge_vertices = grid[0] + grid[-1] + [row[0] for row in grid] + [row[-1] for row in grid]
			color_refs = {tuple(round(v) for v in item[0]): [] for item in edge_vertices}
			if samples:
				disp_light_records.append((f, mat_recs[mat][:2], ch.lm, cell, grid, color_refs))
			for j in range(n - 1):
				strip = []
				for i in range(n):
					for jj in (j, j + 1):
						p, col, nrm, base = grid[jj][i]
						s, t = sts[jj * n + i]
						lu, lv = (0.0, 0.0)
						if cell:
							sl, tl, lw, lh = luxel_uv(bsp, f, base)
							lu, lv = cell_st(cell, sl, tl, lw, lh)
						strip.append((p, col, nrm, s, t, lu, lv))
				ch.emit(mat, GX_DRAW_TRISTRIP, strip, color_refs, tex_axes=(ti[0], ti[1]),
					base_normal=tuple(plane[k] * sign for k in range(3)))
			for j in range(n - 1):
				for i in range(n - 1):
					a, b2, c, d = grid[j][i][0], grid[j][i + 1][0], grid[j + 1][i + 1][0], grid[j + 1][i][0]
					ground_tris.append((a, b2, c))
					ground_tris.append((a, c, d))
			n_disp += 1
			continue

		cluster = face_cluster.get(fi, -1)
		if cluster < 0:
			cx = sum(p[0] for p in pts) / len(pts)
			cy = sum(p[1] for p in pts) / len(pts)
			cz = sum(p[2] for p in pts) / len(pts)
			pl = bsp.planes[f[0]]
			sign = -1.0 if f[1] else 1.0
			cluster = bsp.cluster_at((cx + pl[0] * sign * 4, cy + pl[1] * sign * 4, cz + pl[2] * sign * 4))
		if cluster < 0:
			n_orphans += 1
			continue
		sts = st_for(ti, tw, th, pts)
		pln = bsp.planes[f[0]]
		sign = -1.0 if f[1] else 1.0
		nrm = (pln[0] * sign, pln[1] * sign, pln[2] * sign)
		ch = chunks.setdefault(cluster, Chunk())
		samples = None if water else face_samples(bsp, f)
		cell = ch.lm.alloc(f[13] + 1, f[14] + 1) if samples else None
		if cell and samples:
			ch.lm.blit(cell, samples)
		verts = []
		for p, (s, t) in zip(pts, sts):
			# WorldVertexTransition on a plain brush face has no blend alpha:
			# Source shows $basetexture only. 255 here drew layer 2 solid
			# and left hard-edged pale polygons across the beach.
			blend_a = 0 if mat_recs[mat][1] else 255
			if blend_a == 0:
				n_flat_blend += 1
			col = (96, 132, 140, 255) if water else light_at(bsp, f, p) + (blend_a,)
			lu, lv = (0.0, 0.0)
			if cell:
				sl, tl, lw, lh = luxel_uv(bsp, f, p)
				lu, lv = cell_st(cell, sl, tl, lw, lh)
			verts.append((p, col, nrm, s, t, lu, lv))
		ch.emit(mat, GX_DRAW_TRIFAN, verts, tex_axes=(ti[0], ti[1]))
		if nrm[2] > 0.5 and not water:
			for k in range(1, len(pts) - 1):
				ground_tris.append((pts[0], pts[k], pts[k + 1]))
		face_ctx[fi] = (cluster, cell, nrm)
		n_faces += 1

	# --- Overlays (lump 45): graffiti, plaster and trash decals -----------------
	# doverlay_t, 352 bytes: id, texinfo, face count, 64 face ids, U[2], V[2],
	# UVPoints[4] (xy = corner in basis space, z of the first three = BasisU),
	# origin, basis normal. Each becomes one quad a unit off its host face,
	# lit by that face's lightmap cell. Hosts on displacements are skipped:
	# a flat quad does not follow the sand.
	n_overlays = 0
	oo, ol = struct.unpack_from("<ii", bsp.data, 8 + 45 * 16)
	for oi in range(ol // 352):
		r = bsp.data[oo + oi * 352:oo + (oi + 1) * 352]
		_oid, oti, ofc = struct.unpack_from("<ihH", r, 0)
		hosts = [h for h in struct.unpack_from("<64i", r, 8)[:ofc & 0x3FFF] if h in face_ctx]
		if not hosts or not (0 <= oti < len(bsp.texinfos)):
			continue
		oti_rec = bsp.texinfos[oti]
		omat = mat_for(oti_rec[5], oti_rec[4])
		if omat is None:
			continue
		U = struct.unpack_from("<2f", r, 264)
		V = struct.unpack_from("<2f", r, 272)
		uvp = struct.unpack_from("<12f", r, 280)
		org = struct.unpack_from("<3f", r, 328)
		bn = struct.unpack_from("<3f", r, 340)
		bu = (uvp[2], uvp[5], uvp[8])
		bv = (bn[1] * bu[2] - bn[2] * bu[1], bn[2] * bu[0] - bn[0] * bu[2], bn[0] * bu[1] - bn[1] * bu[0])
		host = hosts[0]
		cluster, cell, hn = face_ctx[host]
		hf = bsp.faces[host]
		corner_uv = ((U[0], V[0]), (U[0], V[1]), (U[1], V[1]), (U[1], V[0]))
		verts = []
		for k in range(4):
			x, y = uvp[k * 3], uvp[k * 3 + 1]
			pt = tuple(org[a] + bu[a] * x + bv[a] * y + hn[a] * 1.0 for a in range(3))
			lu, lv = (0.0, 0.0)
			if cell:
				sl, tl, lw, lh = luxel_uv(bsp, hf, pt)
				lu, lv = cell_st(cell, max(0.0, min(lw - 1.0, sl)), max(0.0, min(lh - 1.0, tl)), lw, lh)
			verts.append((pt, light_at(bsp, hf, pt) + (255,), hn, corner_uv[k][0], corner_uv[k][1], lu, lv))
		chunks[cluster].emit(omat, GX_DRAW_TRIFAN, verts)
		n_overlays += 1
	print(f"  overlays baked: {n_overlays}/{ol // 352}")

	# --- infodecals (tools/bake_infodecals.py) --------------------------------
	# Source projects each infodecal onto the surface it sits on when the map
	# loads. Host = brush face whose plane holds the origin (within 16 units)
	# and whose polygon contains it. Size = texture pixels x $decalscale, image
	# up = world up on the plane (X axis on level ground); one alpha-tested quad
	# a unit off the host, lit by the host's lightmap, from the shared atlas.
	n_decals = 0
	dec_path = out_dir / "infodecals.json"
	if dec_path.is_file() and "lostcoast/decals/atlas" in names:
		dinfo = json.loads(dec_path.read_text())
		dmat = mat_named("lostcoast/decals/atlas")
		for dec in dinfo["decals"]:
			org = tuple(dec["origin"])
			tile = dinfo["tiles"][dec["texture"]]
			best = None
			for fi, (cl, cell, hn) in face_ctx.items():
				f = bsp.faces[fi]
				pl = bsp.planes[f[0]]
				sg = -1.0 if f[1] else 1.0
				dist = (org[0] * hn[0] + org[1] * hn[1] + org[2] * hn[2]) - pl[3] * sg
				if abs(dist) > 16.0 or (best and abs(dist) >= best[0]):
					continue
				pts = bsp.face_pts(f)
				pr = tuple(org[a] - hn[a] * dist for a in range(3))
				pos = neg = 0
				for k in range(len(pts)):
					a0, b0 = pts[k], pts[(k + 1) % len(pts)]
					e = (b0[0] - a0[0], b0[1] - a0[1], b0[2] - a0[2])
					w = (pr[0] - a0[0], pr[1] - a0[1], pr[2] - a0[2])
					cx = (e[1] * w[2] - e[2] * w[1], e[2] * w[0] - e[0] * w[2], e[0] * w[1] - e[1] * w[0])
					d = cx[0] * hn[0] + cx[1] * hn[1] + cx[2] * hn[2]
					pos += d > 0.5
					neg += d < -0.5
				if not (pos and neg):
					best = (abs(dist), fi, pr, pts)
			if not best:
				continue
			_d, host, ctr, hpts = best
			cluster, cell, hn = face_ctx[host]
			hf = bsp.faces[host]
			if abs(hn[2]) > 0.99:
				upv = (0.0, 1.0, 0.0)
			else:
				upv = (0.0, 0.0, 1.0)
			rgt = (upv[1] * hn[2] - upv[2] * hn[1], upv[2] * hn[0] - upv[0] * hn[2], upv[0] * hn[1] - upv[1] * hn[0])
			rl = math.sqrt(sum(c * c for c in rgt)) or 1.0
			rgt = tuple(c / rl for c in rgt)
			upv = (hn[1] * rgt[2] - hn[2] * rgt[1], hn[2] * rgt[0] - hn[0] * rgt[2], hn[0] * rgt[1] - hn[1] * rgt[0])
			hw, hh = tile["size"][0] * 0.5, tile["size"][1] * 0.5
			u0, v0, u1, v1 = tile["uv"]
			corners = ((-hw, hh, u0, v0), (-hw, -hh, u0, v1), (hw, -hh, u1, v1), (hw, hh, u1, v0))
			quad = []
			for x, y, cu, cv in corners:
				pt = tuple(ctr[a] + rgt[a] * x + upv[a] * y + hn[a] * 1.0 for a in range(3))
				lu, lv = (0.0, 0.0)
				if cell:
					sl, tl, lw, lh = luxel_uv(bsp, hf, pt)
					lu, lv = cell_st(cell, max(0.0, min(lw - 1.0, sl)), max(0.0, min(lh - 1.0, tl)), lw, lh)
				quad.append((pt, light_at(bsp, hf, pt) + (255,), hn, cu, cv, lu, lv))
			# Match the winding of the host's own polygon (Source: clockwise).
			def _wind(q):
				a0, b0, c0 = q[0], q[1], q[2]
				e1 = tuple(b0[i] - a0[i] for i in range(3))
				e2 = tuple(c0[i] - a0[i] for i in range(3))
				cr = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0])
				return cr[0] * hn[0] + cr[1] * hn[1] + cr[2] * hn[2]
			if _wind([v[0] for v in quad]) * _wind(hpts) < 0:
				quad = [quad[0], quad[3], quad[2], quad[1]]
			chunks[cluster].emit(dmat, GX_DRAW_TRIFAN, quad)
			n_decals += 1
		print(f"  infodecals baked: {n_decals}/{len(dinfo['decals'])}")

	seam_pairs, seam_texels, seam_vertices = stitch_displacement_lightmaps(bsp, disp_light_records)
	print(f"  displacement lighting seams softened: {seam_pairs} edges, {seam_texels} texels, {seam_vertices} vertices")

	# --- Water depth map ----------------------------------------------------------
	# Top-down map of how far the ground lies under the harbor water plane
	# (z = -64), over the playable cove. The water pass multiplies its alpha by
	# it, so the sheet thins to nothing along the shore instead of ending at a
	# polygon edge. I4, 128 x 256 texels of 24 units, 16 KB.
	#   media/dp_<map>.gcx : "GCdp", u16 w, u16 h, f32 x0, f32 y0, f32 cell,
	#                        pad to 32, then GX I4 tiles (8x8)
	DP_W, DP_H, DP_CELL, DP_X0, DP_Y0 = 128, 256, 24.0, 0.0, -1536.0
	WATER_Z, DP_FULL = -64.0, 40.0
	ground = [[None] * DP_W for _ in range(DP_H)]
	above_water = [[False] * DP_W for _ in range(DP_H)]
	for a, b2, c in ground_tris:
		minx = min(a[0], b2[0], c[0]); maxx = max(a[0], b2[0], c[0])
		miny = min(a[1], b2[1], c[1]); maxy = max(a[1], b2[1], c[1])
		if maxx < DP_X0 or minx > DP_X0 + DP_W * DP_CELL or maxy < DP_Y0 or miny > DP_Y0 + DP_H * DP_CELL:
			continue
		det = (b2[1] - c[1]) * (a[0] - c[0]) + (c[0] - b2[0]) * (a[1] - c[1])
		if abs(det) < 1e-6:
			continue
		for gy in range(max(0, int((miny - DP_Y0) / DP_CELL)), min(DP_H, int((maxy - DP_Y0) / DP_CELL) + 1)):
			py = DP_Y0 + (gy + 0.5) * DP_CELL
			for gx in range(max(0, int((minx - DP_X0) / DP_CELL)), min(DP_W, int((maxx - DP_X0) / DP_CELL) + 1)):
				px = DP_X0 + (gx + 0.5) * DP_CELL
				w0 = ((b2[1] - c[1]) * (px - c[0]) + (c[0] - b2[0]) * (py - c[1])) / det
				w1 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / det
				w2 = 1.0 - w0 - w1
				if w0 < -0.02 or w1 < -0.02 or w2 < -0.02:
					continue
				z = w0 * a[2] + w1 * b2[2] + w2 * c[2]
				# Not the harbor bed: the map has floors thousands of units
				# down (under the 3D skybox, z = -5144). Where the beach rises
				# just above the waterline such a floor became the "seabed"
				# of the cell, and the sheet drew full-depth dark water over
				# dry sand in hard-edged patches.
				if z < WATER_Z - 1024.0:
					continue
				cur = ground[gy][gx]
				# The seabed is the highest surface at or below the waterline;
				# anything above it (a pier deck, a cliff) is not the bed.
				if z <= WATER_Z + 8.0 and (cur is None or z > cur):
					ground[gy][gx] = z
				elif z > WATER_Z + 8.0:
					# Track dry geometry separately: a later seabed triangle must
					# still be able to replace the apparent dry cell.
					above_water[gy][gx] = True
	dp = bytearray(DP_W * DP_H // 2)
	def depth_at(gx, gy):
		z = ground[gy][gx]
		if z is None:
			return 0 if above_water[gy][gx] else 15  # dry surface or open water
		return int(max(0, min(15, round((WATER_Z - z) / DP_FULL * 15.0))))
	for ty in range(0, DP_H, 8):
		for tx in range(0, DP_W, 8):
			tile = (ty // 8) * (DP_W // 8) + (tx // 8)
			for y in range(8):
				for x in range(0, 8, 2):
					hi = depth_at(tx + x, ty + y)
					lo = depth_at(tx + x + 1, ty + y)
					dp[tile * 32 + y * 4 + x // 2] = (hi << 4) | lo
	dp_path = out_dir / "media" / f"dp_{stem}.gcx"
	dp_path.write_bytes(struct.pack(">4sHHfff", b"GCdp", DP_W, DP_H, DP_X0, DP_Y0, DP_CELL) + b"\0" * 12 + bytes(dp))
	shallow = sum(1 for row in ground for z in row if z is not None and WATER_Z - DP_FULL < z <= WATER_Z)
	print(f"  water depth map {dp_path.name}: {DP_W}x{DP_H}, shallow cells {shallow}")

	# --- BSP tree for the runtime point-in-leaf --------------------------------
	plane_remap: dict[int, int] = {}
	node_blob = bytearray()

	def child(c: int) -> int:
		if c >= 0:
			return c
		cluster = bsp.leaves[-1 - c][1]
		return -1 if cluster < 0 else -(cluster + 2)

	assert len(bsp.nodes) < 32768 and bsp.numclusters < 32000
	for planenum, c0, c1 in bsp.nodes:
		pi = plane_remap.setdefault(planenum, len(plane_remap))
		node_blob += struct.pack(">Hhh", pi, child(c0), child(c1))
	plane_blob = bytearray(16 * len(plane_remap))
	for planenum, pi in plane_remap.items():
		struct.pack_into(">ffff", plane_blob, pi * 16, *bsp.planes[planenum])

	# --- PVS rows (Source RLE, PVS half of each cluster's offset pair) ----------
	visbytes = (bsp.numclusters + 7) // 8
	pvs_blob = bytearray()
	pvs_dir = bytearray()
	for cl in range(bsp.numclusters):
		ofs = struct.unpack_from("<i", bsp.vis, 4 + cl * 8)[0]
		if ofs < 0:
			pvs_dir += struct.pack(">I", 0xFFFFFFFF)
			continue
		pos = ofs
		done = 0
		while done < visbytes:
			if bsp.vis[pos]:
				pos += 1
				done += 1
			else:
				done += bsp.vis[pos + 1]
				pos += 2
		pvs_dir += struct.pack(">I", len(pvs_blob))
		pvs_blob += bsp.vis[ofs:pos]

	# --- Assemble ---------------------------------------------------------------
	order = sorted(chunks)
	# Large walls are the ones vertex colour leaves flat. Keep the 512
	# biggest pages (span ≥ 256). The runtime binds each one wherever
	# that chunk is drawn.
	# 3D skybox clusters never bind a lightmap page (drawn 16x, far away) and
	# stay resident every frame: keep them lean so they do not crowd the cache.
	sky_clusters: set[int] = set()
	_m = __import__("re").search(r'\{[^{}]*"classname"\s*"sky_camera"[^{}]*\}', lump(bsp.data, 0).decode("latin-1", "replace"))
	if _m:
		_o = __import__("re").search(r'"origin"\s*"([^"]+)"', _m.group(0))
		_scl = bsp.cluster_at([float(x) for x in _o.group(1).split()]) if _o else -1
		_ofs = struct.unpack_from("<i", bsp.vis, 4 + _scl * 8)[0] if _scl >= 0 else -1
		if _ofs >= 0:
			_c = 0
			_p = _ofs
			while _c < bsp.numclusters:
				if bsp.vis[_p]:
					for _b in range(8):
						if bsp.vis[_p] & (1 << _b):
							sky_clusters.add(_c + _b)
					_c += 8
					_p += 1
				else:
					_c += 8 * bsp.vis[_p + 1]
					_p += 2
	ranked = []
	for cl, ch in chunks.items():
		if not ch.pos or not ch.lm.used or ch.lm.failed or cl in sky_clusters:
			continue
		span = max(ch.maxs[a] - ch.mins[a] for a in range(3))
		if span >= 256:
			ranked.append((span, cl))
	ranked.sort(reverse=True)
	for _span, cl in ranked[:512]:
		chunks[cl].lm_keep = True
	lm_pages = bytearray()
	for cl in order:
		if getattr(chunks[cl], "lm_keep", False):
			chunks[cl].lm_index = len(lm_pages) // 2048
			page = rgb565_page(chunks[cl].lm.pix)
			assert len(page) == 2048, len(page)
			lm_pages += page
	LM_PAGES_LEFT = 512 - min(512, len(ranked))
	chunk_blobs = [chunks[c].encode() for c in order]
	# The (cubemap, plane axis) pairs the reflective batches look up, for
	# tools/bake_lostcoast_cubemaps.py (media/cm_<map>.gcm holds only these).
	from bake_lostcoast_cubemaps import nearest_axis
	env_pairs = set()
	n_env_batches = 0
	for ch in chunks.values():
		for key in ch.batches:
			if key >> 16 and (key & 0xFFFF) < len(mat_recs) and mat_recs[key & 0xFFFF][4] >> 8:
				env_pairs.add(((mat_recs[key & 0xFFFF][4] >> 8) - 1, nearest_axis(oct12_decode((key >> 16) & 0xFFF))))
				n_env_batches += 1
	(out_dir / "envpairs.json").write_text(json.dumps(sorted(env_pairs)) + "\n")
	print(f"  reflective batches={n_env_batches} of {sum(len(c.batches) for c in chunks.values())} cubemap views={len(env_pairs)}")
	mats_blob = b"".join(struct.pack(">IIBBH", h, h2, tid, fl, st) for h, h2, tid, fl, st in mat_recs)
	body = bytearray(64)
	# 3D skybox (Source sky_camera): 32-byte block right after the header,
	# present when ofs_mats >= 96 and it starts with "SKY1".
	#   "SKY1", f32 origin[3], f32 scale, s32 cluster, u8 fog rgb, u8 pad,
	#   f32 fogstart, f32 fogend   (fog distances in main-world units)
	sky = None
	ents = lump(bsp.data, 0).decode("latin-1", "replace")
	import re as _re
	for m in _re.finditer(r"\{[^{}]*\}", ents):
		kv = dict(_re.findall(r'"([^"]*)"\s*"([^"]*)"', m.group(0)))
		if kv.get("classname") == "sky_camera":
			so = [float(x) for x in kv.get("origin", "0 0 0").split()]
			fog = [int(float(x)) for x in kv.get("fogcolor", "255 255 255").split()][:3]
			sky = (so, float(kv.get("scale", "16")), bsp.cluster_at(so), fog,
				float(kv.get("fogstart", "0")), float(kv.get("fogend", "0")),
				kv.get("fogenable", "0") == "1")
			break
	if sky and sky[2] >= 0:
		so, sscale, scl, fog, fs, fe, fen = sky
		body += struct.pack(">4s4fi3BBff", b"SKY1", so[0], so[1], so[2], sscale, scl,
			fog[0], fog[1], fog[2], 1 if fen else 0, fs, fe)
		pad32(body)
		print(f"  sky_camera origin={so} scale={sscale} cluster={scl} fog={fog} {fs}..{fe}")
	ofs_mats = len(body)
	body += mats_blob
	pad32(body)
	ofs_nodes = len(body)
	body += node_blob
	pad32(body)
	ofs_planes = len(body)
	body += plane_blob
	pad32(body)
	ofs_chunkdir = len(body)
	run = 0
	for cl, blob in zip(order, chunk_blobs):
		ch = chunks[cl]
		body += struct.pack(">II3h3hHH", run, len(blob), *ch.mins, *ch.maxs, cl, 0)
		run += len(blob)
	pad32(body)
	ofs_pvsdir = len(body)
	body += pvs_dir
	pad32(body)
	ofs_pvs = len(body)
	body += pvs_blob
	pad32(body)
	ofs_chunks = len(body)
	for blob in chunk_blobs:
		body += blob
	# Lightmap page stream: starts right after the last chunk (the runtime
	# derives the offset from the chunk directory).
	body += lm_pages
	max_chunk = max((len(b) for b in chunk_blobs), default=0)
	struct.pack_into(
		">4sHH13I", body, 0, b"GCwd", 2, 0,
		bsp.numclusters, len(order), len(mat_recs), len(bsp.nodes),
		len(plane_remap), ofs_mats, ofs_nodes, ofs_planes,
		ofs_chunkdir, ofs_pvsdir, ofs_pvs, ofs_chunks,
		len(body),
	)
	struct.pack_into(">I", body, 60, max_chunk)
	out = out_dir / "media" / f"wd_{stem}.gcd"
	out.write_bytes(body)

	tris = sum(c.tris for c in chunks.values())
	batches = sum(len(c.batches) for c in chunks.values())
	resident = ofs_pvsdir  # mats + tree + chunk directory stay in MEM1
	print(f"world DL bake {stem}:")
	print(f"  faces={n_faces} displacements={n_disp} (power<={max_disp_power}) orphans={n_orphans} flat_blend_verts={n_flat_blend} tris={tris}")
	print(f"  clusters={bsp.numclusters} chunks={len(order)} batches={batches} mats={len(mat_recs)}")
	print(f"  tree nodes={len(bsp.nodes)} planes={len(plane_remap)} ({(len(node_blob) + len(plane_blob)) // 1024} KB)")
	print(f"  pvs rows={len(pvs_blob) // 1024} KB chunkdir={len(order) * 24 // 1024} KB")
	print(f"  MEM1 resident={resident // 1024} KB  ARAM chunks={run // 1024} KB (max chunk {max_chunk // 1024} KB)")
	import collections as _c
	_h = _c.Counter(side for _f, _cl, side in DISP_CELLS)
	print(f"  displacement lightmap cells by side: {dict(sorted(_h.items()))}")
	print(f"  envmap mats={sum(1 for *_, fl, _s in mat_recs if fl & MAT_F_ENVMAP)} lightmap pages left={LM_PAGES_LEFT}")
	print(f"  wrote {out} ({len(body) // 1024} KB)")


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--map", type=Path, default=ROOT / "content/lostcoast/maps/d2_lostcoast.bsp")
	ap.add_argument("--out", type=Path, default=ROOT / "cache/bake")
	ap.add_argument("--max-disp-power", type=int, default=4)
	args = ap.parse_args()
	global LM_PLAN
	# Run 1 records the lightmap cells each page is asked for; run 2 bakes
	# with the planned cells (see LM_PLAN).
	import contextlib, io
	with contextlib.redirect_stdout(io.StringIO()):
		bake(args.map, args.out, args.max_disp_power)
	LM_PLAN = plan_lightmap_pages()
	LM_COUNTER[0] = 0
	ENV_SPLIT.clear()
	DISP_CELLS.clear()
	_unfit = sum(1 for r in LM_PLAN.values() for c in r if c is None)
	print(f"  lightmap plan: {len(LM_PLAN)} pages, {sum(len(r) for r in LM_PLAN.values())} cells, {_unfit} without room")
	bake(args.map, args.out, args.max_disp_power)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
