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
		# Lost Coast ships HDR lighting only (lump 53); LDR lump 8 otherwise.
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


def light_at(bsp: Bsp, f, base_pt) -> tuple[int, int, int]:
	"""Baked light colour at a world point on face f (flat lightmap, style 0)."""
	lightofs = f[9]
	ti = bsp.texinfos[f[5]]
	if lightofs < 0 or ti[4] & 0x400:  # SURF_NOLIGHT
		return (128, 128, 128)
	lw, lh = f[13] + 1, f[14] + 1
	s = ti[2][0] * base_pt[0] + ti[2][1] * base_pt[1] + ti[2][2] * base_pt[2] + ti[2][3] - f[11]
	t = ti[3][0] * base_pt[0] + ti[3][1] * base_pt[1] + ti[3][2] * base_pt[2] + ti[3][3] - f[12]
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
	return (_tonemap_hdr(acc[0]), _tonemap_hdr(acc[1]), _tonemap_hdr(acc[2]))


def rgb565(c) -> int:
	return ((c[0] >> 3) << 11) | ((c[1] >> 2) << 5) | (c[2] >> 3)


def pack_nrm(n) -> tuple[int, int, int]:
	x, y, z = n
	length = math.sqrt(x * x + y * y + z * z) or 1.0
	scale = 16384.0 / length  # s16 fraction bits 14
	return tuple(int(max(-32767, min(32767, round(c * scale)))) for c in (x, y, z))


def luxel_uv(bsp: Bsp, f, pt) -> tuple[float, float, int, int]:
	ti = bsp.texinfos[f[5]]
	lw, lh = f[13] + 1, f[14] + 1
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
	return (_tonemap_hdr(r * sc), _tonemap_hdr(g * sc), _tonemap_hdr(b * sc))


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


class LmPage:
	def __init__(self) -> None:
		# Mid-grey pad — unused atlas texels used to be white, so any vert
		# with lu,lv≈0 (failed cell / unset) lit full-bright and washed cliffs.
		self.pix = [[(128, 128, 128) for _ in range(32)] for _ in range(32)]
		self.x = self.y = self.row_h = 0
		self.used = False
		self.failed = False

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
			for px in range(w):
				sx = min(lw - 1, px * lw // w)
				sy = min(lh - 1, py * lh // h)
				self.pix[y + py][x + px] = samples[sy][sx]


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

	def emit(self, mat: int, prim: int, verts: list) -> None:
		"""verts: (pos, rgba, nrm, s, t, lm_u, lm_v)."""
		dl = self.batches.setdefault(mat, bytearray())
		dl += struct.pack(">BH", prim | GX_VTXFMT, len(verts))
		for p, c, nrm, s, t, lu, lv in verts:
			nx, ny, nz = pack_nrm(nrm)
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
			struct.pack_into(">HHII", out, 16 + 12 * i, m, 0, len(out), len(dl))
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


def load_envmap(names: dict) -> tuple[dict[str, int], set[str]]:
	"""Material path → specular strength, plus the $basealphaenvmapmask set.

	VBSP patch names (maps/<map>/…_x_y_z) resolve through _material_aliases so
	monastery frescos / metal pick up $envmap from the base VMT.
	"""
	out: dict[str, int] = {}
	basealpha: set[str] = set()
	vpks = None
	try:
		from bake_lostcoast_map import (
			_load_vmt_for_mat,
			envmap_strength_byte,
			material_has_envmap,
		)
		from vpk import open_vpk_roots
		import bake_tevmat

		roots = sorted((ROOT / "content" / "lostcoast").glob("*_dir.vpk"))
		roots += sorted((ROOT / "content" / "hl2").glob("*_dir.vpk"))
		vpks = open_vpk_roots(roots) if roots else None
		has_env, strength, load_vmt = material_has_envmap, envmap_strength_byte, _load_vmt_for_mat
		extract = ROOT / "cache" / "extracted"
		_num = bake_tevmat._num
	except Exception as exc:
		print(f"  envmap VMT scan skipped: {exc}")
		has_env = strength = load_vmt = vpks = extract = _num = None
	for name in names:
		if vpks and load_vmt and has_env and strength:
			vmt = load_vmt(vpks, name, extract)
			if vmt and has_env(vmt):
				out[name] = strength(vmt)
				if _num and _num(vmt.get("$basealphaenvmapmask"), 0.0) >= 1.0:
					basealpha.add(name)
				continue
		if "sand" in name:
			out[name] = 140
	print(f"  envmap materials: {len(out)} basealpha={len(basealpha)}")
	return out, basealpha


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
	envmap, basealpha_mats = load_envmap(names)

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
			mat_index[name] = len(mat_recs)
			t2 = tex2_names.get(name)
			mat_recs.append((fnv1a32(names[name]), fnv1a32(t2) if t2 else 0,
				tevmat_id_for(name, tevmat_ids), mflags, strength & 0xFFFF))
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
	n_faces = n_disp = n_orphans = 0

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
			# g[0] is the displaced world position. g[2] is the normal, and
			# feeding that to the texvecs made a whole cliff one texel.
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
			if cell and samples:
				ch.lm.blit(cell, samples)
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
				ch.emit(mat, GX_DRAW_TRISTRIP, strip)
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
			col = (96, 132, 140, 255) if water else light_at(bsp, f, p) + (255,)
			lu, lv = (0.0, 0.0)
			if cell:
				sl, tl, lw, lh = luxel_uv(bsp, f, p)
				lu, lv = cell_st(cell, sl, tl, lw, lh)
			verts.append((p, col, nrm, s, t, lu, lv))
		ch.emit(mat, GX_DRAW_TRIFAN, verts)
		n_faces += 1

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
	print(f"  faces={n_faces} displacements={n_disp} (power<={max_disp_power}) orphans={n_orphans} tris={tris}")
	print(f"  clusters={bsp.numclusters} chunks={len(order)} batches={batches} mats={len(mat_recs)}")
	print(f"  tree nodes={len(bsp.nodes)} planes={len(plane_remap)} ({(len(node_blob) + len(plane_blob)) // 1024} KB)")
	print(f"  pvs rows={len(pvs_blob) // 1024} KB chunkdir={len(order) * 24 // 1024} KB")
	print(f"  MEM1 resident={resident // 1024} KB  ARAM chunks={run // 1024} KB (max chunk {max_chunk // 1024} KB)")
	print(f"  envmap mats={sum(1 for *_, fl, _s in mat_recs if fl & MAT_F_ENVMAP)} lightmap pages left={LM_PAGES_LEFT}")
	print(f"  wrote {out} ({len(body) // 1024} KB)")


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--map", type=Path, default=ROOT / "content/lostcoast/maps/d2_lostcoast.bsp")
	ap.add_argument("--out", type=Path, default=ROOT / "cache/bake")
	ap.add_argument("--max-disp-power", type=int, default=4)
	args = ap.parse_args()
	bake(args.map, args.out, args.max_disp_power)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
