#!/usr/bin/env python3
"""Bake Lost Coast Source props → media/gp_<map>.gcp + media/pt_<map>.gct.

Entity props (physics, dynamic, static) plus the static-prop lump. That
lump is where the coast rocks and the monastery shell live; neither is an
entity. Heavy meshes are decimated so the pack stays inside the u16 tri
ceiling. Each material is its own CMPR mip chain (largest first, down to
8×8). The disc read lands in ARAM once; the runtime keeps the 8×8 in MEM1
and lets the hot window climb.
"""
from __future__ import annotations

import argparse
import math
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from bake_lostcoast_map import rgba_to_cmpr_mips, write_gcwt  # noqa: E402
from source_mdl import SourceMesh, angle_matrix, load_source_mesh, transform_point  # noqa: E402
from vpk import open_vpk_roots, read_first  # noqa: E402
from vtf import decode_vtf  # noqa: E402

GC_PR_MAGIC = b"GCpr"
GC_PR_VERSION = 5
GC_PR_MAX_TRIS = 65535
GC_PR_MODEL_TRIS = 105000  # unique model triangles (30 B each in ARAM)
GC_PR_PIECE_TRIS = 512  # one piece = 32 B header + verts, fits the 16 KB bounce
GC_PR_MAX_INST = 512
GC_PR_MAX_MAT = 127
PROP_CLASSNAMES = {"prop_physics", "prop_dynamic", "prop_static"}
# CapFaces lean stand-ins: Source NPCs / canisters have no GoldSrc LINK_ENTITY,
# so they bake into the GCpr pack as posed meshes (same path as boats/rocks).
NPC_CLASSNAMES = {
	"npc_combine_s": "models/combine_soldier.mdl",
	"npc_seagull": "models/seagull.mdl",
	"env_headcrabcanister": "models/props_combine/headcrabcannister01a.mdl",
	# Fisherman walks via CapFaces path (hdr[6]); keep a posed instance off
	# the pack so the walker is the only draw of him.
}
# Local-space canister mesh for the CapFaces church-crash flight (hdr[5]).
CANISTER_FLIGHT_MODEL = "models/props_combine/headcrabcannister01a.mdl"
# Local-space fisherman for the CapFaces lead/walk path (hdr[6]).
FISHERMAN_WALK_MODEL = "models/lostcoast/fisherman/fisherman.mdl"
# Opening of the VST ride. Shore rocks are packed nearest-first.
SHORE_EYE = (1518.0, -1008.0, 52.0)


def _parse_vec(s: str) -> tuple[float, float, float]:
	parts = s.split()
	if len(parts) < 3:
		return 0.0, 0.0, 0.0
	return float(parts[0]), float(parts[1]), float(parts[2])


def _skip_model(model: str) -> bool:
	"""Foliage trees decimate into a solid star. Their leaves need alpha,
	which this pack does not test, so they stay out until that exists."""
	low = model.lower().replace("\\", "/")
	return "blackout" in low or "tree" in low or "/card" in low or "card_" in low


def _tri_cap(model: str, natural: int) -> int:
	"""Boats and rocks keep a readable hull. Lamps and chairs stay a silhouette."""
	low = model.lower()
	if "seagull" in low:
		# Thin wings shred under aggressive decimate — keep the full LOD2.
		cap = 270
	elif "combine_soldier" in low or "combine_super_soldier" in low:
		# Readable silhouette under the monastery; bind-pose stand-in.
		cap = 220
	elif "fisherman" in low:
		# Real Source fisherman (fixups + mesh vertexoffset). Keep enough
		# tris that short-edge decimate still reads as a person.
		cap = 4000
	elif "headcrabcannister" in low:
		cap = 160
	elif any(s in low for s in ("chandelier", "candlestick", "oil_lamp", "chair", "bottle", "bucket", "paintcan", "cannister", "bicycle")):
		cap = 64
	elif any(s in low for s in ("boat", "float", "dock")):
		cap = 1400
	elif "monastery01" in low:
		cap = 400
	elif "rock_" in low:
		cap = 256
	elif any(s in low for s in ("gate", "stair", "crate", "pallet", "crab")):
		cap = 200
	elif any(s in low for s in ("dome", "cloister", "shipwreck", "drydock", "village", "scaffold", "refinery", "bridge", "destroyed", "interior_")):
		cap = 240
	elif "window" in low:
		# A window is already a few hundred triangles. Cutting it to 80
		# keeps the dark panes and drops the frame, which reads as a shard.
		cap = 640
	elif any(s in low for s in ("pillar", "fence", "tree", "handrail", "gutter", "chimney", "ladder")):
		cap = 80
	else:
		cap = 96
	# Pack v3 stores a model once and places it with a matrix, so the
	# budget is unique triangles, not triangles times instances.
	if "fisherman" not in low:
		cap *= 4
	return max(1, min(natural, cap))


def _max_side(model: str, texname: str) -> int:
	"""Chair, hull, crate and gate keep a 256² top. Everything else tops at 128²."""
	low = f"{model} {texname}".lower()
	if any(s in low for s in ("chair", "boat", "crate", "gate")):
		return 256
	return 128


def _shell_model(model: str) -> bool:
	"""Closed shells. Picking their largest triangles leaves floating plates.
	Vertex clustering welds those back into a coarse silhouette."""
	low = model.lower()
	return any(s in low for s in (
		"dome", "cloister", "village", "destroyed", "interior_",
		"shipwreck", "refinery", "drydock",
	))


def _cluster_tris(
	positions: list[tuple[float, float, float]],
	uvs: list[tuple[float, float]],
	tris: list[tuple[int, int, int]],
	mats: list[int],
	div: int,
) -> tuple[list, list, list, list]:
	xs = [p[0] for p in positions]
	ys = [p[1] for p in positions]
	zs = [p[2] for p in positions]
	minx, maxx = min(xs), max(xs)
	miny, maxy = min(ys), max(ys)
	minz, maxz = min(zs), max(zs)
	sx = max(maxx - minx, 1e-3)
	sy = max(maxy - miny, 1e-3)
	sz = max(maxz - minz, 1e-3)
	buckets: dict[tuple[int, int, int], list[int]] = {}
	for i, p in enumerate(positions):
		key = (
			min(div - 1, int((p[0] - minx) / sx * div)),
			min(div - 1, int((p[1] - miny) / sy * div)),
			min(div - 1, int((p[2] - minz) / sz * div)),
		)
		buckets.setdefault(key, []).append(i)
	remap: dict[int, int] = {}
	new_pos: list[tuple[float, float, float]] = []
	new_uv: list[tuple[float, float]] = []
	for ids in buckets.values():
		ni = len(new_pos)
		n = float(len(ids))
		new_pos.append((
			sum(positions[i][0] for i in ids) / n,
			sum(positions[i][1] for i in ids) / n,
			sum(positions[i][2] for i in ids) / n,
		))
		new_uv.append((
			sum(uvs[i][0] for i in ids) / n,
			sum(uvs[i][1] for i in ids) / n,
		))
		for i in ids:
			remap[i] = ni
	new_tris: list[tuple[int, int, int]] = []
	new_mats: list[int] = []
	seen: set[tuple] = set()
	for t, m in zip(tris, mats):
		a, b, c = remap[t[0]], remap[t[1]], remap[t[2]]
		if a == b or b == c or c == a:
			continue
		key = (tuple(sorted((a, b, c))), m)
		if key in seen:
			continue
		seen.add(key)
		new_tris.append((a, b, c))
		new_mats.append(m)
	return new_pos, new_uv, new_tris, new_mats


def _spread_model(model: str) -> bool:
	"""Buildings need a triangle in every part of the mesh. Biggest-first
	keeps the long spanning faces and draws them as spikes."""
	low = model.lower()
	return any(s in low for s in (
		"dome", "cloister", "shipwreck", "village", "scaffold", "refinery",
		"bridge", "destroyed", "interior_", "monastery",
	))


def _hull_model(model: str) -> bool:
	"""A boat or dock is a few large plates. Dropping those long faces
	leaves the ribs and the pier reads as loose planks."""
	low = model.lower()
	return any(s in low for s in ("boat", "float", "dock"))


def _npc_model(model: str) -> bool:
	"""Combine / seagull / canister / fisherman stand-ins stay when the pack shrinks."""
	low = model.lower()
	return any(
		s in low
		for s in (
			"combine_soldier",
			"combine_super_soldier",
			"seagull",
			"headcrabcannister",
			"fisherman",
		)
	)


def _decimate_tris(
	positions: list[tuple[float, float, float]],
	uvs: list[tuple[float, float]],
	tris: list[tuple[int, int, int]],
	mats: list[int],
	max_tris: int,
	spread: bool = False,
	keep_long: bool = False,
) -> tuple[list, list, list, list]:
	if len(mats) != len(tris):
		mats = [0] * len(tris)
	if len(tris) <= max_tris:
		return positions, uvs, tris, mats
	xs = [p[0] for p in positions]
	ys = [p[1] for p in positions]
	zs = [p[2] for p in positions]
	minx, maxx = min(xs), max(xs)
	miny, maxy = min(ys), max(ys)
	minz, maxz = min(zs), max(zs)
	sx = max(maxx - minx, 1e-3)
	sy = max(maxy - miny, 1e-3)
	sz = max(maxz - minz, 1e-3)
	diag = math.sqrt(sx * sx + sy * sy + sz * sz)
	compact: list[tuple[float, tuple[int, int, int], int, tuple[int, int, int]]] = []
	spans: list[tuple[float, tuple[int, int, int], int]] = []
	for t, m in zip(tris, mats):
		a, b, c = t
		pa, pb, pc = positions[a], positions[b], positions[c]
		e1x, e1y, e1z = pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]
		e2x, e2y, e2z = pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]
		cpx = e1y * e2z - e1z * e2y
		cpy = e1z * e2x - e1x * e2z
		cpz = e1x * e2y - e1y * e2x
		area2 = cpx * cpx + cpy * cpy + cpz * cpz
		longest = max(
			math.dist(pa, pb), math.dist(pb, pc), math.dist(pc, pa),
		)
		cx = (pa[0] + pb[0] + pc[0]) / 3.0
		cy = (pa[1] + pb[1] + pc[1]) / 3.0
		cz = (pa[2] + pb[2] + pc[2]) / 3.0
		cell = (
			min(3, max(0, int((cx - minx) / sx * 4.0))),
			min(3, max(0, int((cy - miny) / sy * 4.0))),
			min(3, max(0, int((cz - minz) / sz * 4.0))),
		)
		# Longer than a third of the model: a span across the building,
		# which is what drew the star in the monastery street.
		if longest > diag * 0.34 and not keep_long:
			spans.append((area2, t, m))
		else:
			compact.append((area2, t, m, cell))
	if spread:
		cells: dict[tuple[int, int, int], list] = {}
		for item in compact:
			cells.setdefault(item[3], []).append(item)
		for bucket in cells.values():
			bucket.sort(key=lambda it: it[0], reverse=True)
		picked: list[tuple[float, tuple[int, int, int], int]] = []
		cursors = {k: 0 for k in cells}
		order = list(cells)
		while len(picked) < max_tris:
			grew = False
			for k in order:
				i = cursors[k]
				if i >= len(cells[k]):
					continue
				area2, t, m, _cell = cells[k][i]
				picked.append((area2, t, m))
				cursors[k] = i + 1
				grew = True
				if len(picked) >= max_tris:
					break
			if not grew:
				break
	else:
		compact.sort(key=lambda it: it[0], reverse=True)
		picked = [(area2, t, m) for area2, t, m, _cell in compact[:max_tris]]
	if len(picked) < max_tris and spans:
		spans.sort(key=lambda it: it[0], reverse=True)
		picked.extend(spans[: max_tris - len(picked)])
	kept = picked
	used = sorted({i for _area, t, _m in kept for i in t})
	remap = {old: new for new, old in enumerate(used)}
	new_pos = [positions[i] for i in used]
	new_uv = [uvs[i] for i in used]
	new_tris = [(remap[a], remap[b], remap[c]) for _area, (a, b, c), _m in kept]
	new_mats = [m for _area, _t, m in kept]
	return new_pos, new_uv, new_tris, new_mats


def _voxel_shell(
	positions: list[tuple[float, float, float]],
	uvs: list[tuple[float, float]],
	tris: list[tuple[int, int, int]],
	mats: list[int],
	div: int = 6,
) -> tuple[list, list, list, list]:
	"""Seal a hull. Fine clustering left holes the size of a plank, and the
	frame read those as ribs. A coarse occupied grid has one outside."""
	xs = [p[0] for p in positions]
	ys = [p[1] for p in positions]
	zs = [p[2] for p in positions]
	minx, maxx = min(xs), max(xs)
	miny, maxy = min(ys), max(ys)
	minz, maxz = min(zs), max(zs)
	sx = max(maxx - minx, 1e-3)
	sy = max(maxy - miny, 1e-3)
	sz = max(maxz - minz, 1e-3)
	occ: set[tuple[int, int, int]] = set()
	for p in positions:
		occ.add((
			min(div - 1, int((p[0] - minx) / sx * div)),
			min(div - 1, int((p[1] - miny) / sy * div)),
			min(div - 1, int((p[2] - minz) / sz * div)),
		))
	# Fill the interior so the shell is the outside, not a crust of ribs.
	filled = set(occ)
	for iy in range(div):
		for iz in range(div):
			cols = sorted(ix for ix, y, z in occ if y == iy and z == iz)
			if len(cols) >= 2:
				for ix in range(cols[0], cols[-1] + 1):
					filled.add((ix, iy, iz))
	def corner(ix: int, iy: int, iz: int) -> tuple[float, float, float]:
		return (
			minx + ix / div * sx,
			miny + iy / div * sy,
			minz + iz / div * sz,
		)
	def uv_of(p: tuple[float, float, float]) -> tuple[float, float]:
		# The authored UV of the nearest source vertex, so the Lost Coast
		# material lands on the sealed hull instead of a stretched box map.
		best = 1.0e30
		uv = (0.0, 0.0)
		for q, src in zip(positions, uvs):
			d = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2
			if d < best:
				best = d
				uv = src
		return uv
	mat = mats[0] if mats else 0
	# +axis faces, two windings each so a one-sided raster still shows the outside.
	faces = (
		(1, 0, 0, ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
		(-1, 0, 0, ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
		(0, 1, 0, ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
		(0, -1, 0, ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
		(0, 0, 1, ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
		(0, 0, -1, ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
	)
	new_pos: list[tuple[float, float, float]] = []
	new_uv: list[tuple[float, float]] = []
	new_tris: list[tuple[int, int, int]] = []
	new_mats: list[int] = []
	index: dict[tuple[float, float, float], int] = {}
	def vid(p: tuple[float, float, float]) -> int:
		got = index.get(p)
		if got is not None:
			return got
		index[p] = len(new_pos)
		new_pos.append(p)
		new_uv.append(uv_of(p))
		return index[p]
	for ix, iy, iz in filled:
		for dx, dy, dz, quad in faces:
			nx, ny, nz = ix + dx, iy + dy, iz + dz
			if (nx, ny, nz) in filled:
				continue
			ids = [vid(corner(ix + a, iy + b, iz + c)) for a, b, c in quad]
			new_tris.append((ids[0], ids[1], ids[2]))
			new_tris.append((ids[0], ids[2], ids[3]))
			new_mats.append(mat)
			new_mats.append(mat)
	if len(new_tris) < 8:
		return positions, uvs, tris, mats
	return new_pos, new_uv, new_tris, new_mats


def _simplify(
	model: str,
	positions: list[tuple[float, float, float]],
	uvs: list[tuple[float, float]],
	tris: list[tuple[int, int, int]],
	mats: list[int],
	max_tris: int,
) -> tuple[list, list, list, list]:
	low = model.lower()
	if "fisherman" in low:
		# Character mesh: never keep_long (that made the spiked blob).
		# Reserve head/face materials so the raincoat body cannot starve them.
		head_ids = {i for i, m in enumerate(mats) if m <= 4}
		body_ids = [i for i in range(len(tris)) if i not in head_ids]
		head_budget = min(len(head_ids), max(400, max_tris // 4))
		body_budget = max(64, max_tris - head_budget)
		head_pos, head_uv, head_tris, head_mats = _decimate_tris(
			positions, uvs,
			[tris[i] for i in sorted(head_ids)],
			[mats[i] for i in sorted(head_ids)],
			head_budget, spread=True, keep_long=False,
		)
		body_pos, body_uv, body_tris, body_mats = _decimate_tris(
			positions, uvs,
			[tris[i] for i in body_ids],
			[mats[i] for i in body_ids],
			body_budget, spread=True, keep_long=False,
		)
		# Merge into one vertex pool.
		merged_pos = list(head_pos) + list(body_pos)
		merged_uv = list(head_uv) + list(body_uv)
		off = len(head_pos)
		merged_tris = list(head_tris) + [
			(a + off, b + off, c + off) for a, b, c in body_tris
		]
		merged_mats = list(head_mats) + list(body_mats)
		return merged_pos, merged_uv, merged_tris, merged_mats
	if len(tris) <= max_tris:
		return positions, uvs, tris, mats
	# Weld vertices on a grid, finest grid that fits the budget. Picking a
	# subset of triangles (the old path for boats, NPCs and most props)
	# met the budget by leaving holes in every mesh.
	best = (positions, uvs, tris, mats)
	for div in (160, 128, 96, 80, 64, 56, 48, 40, 32, 28, 24, 20, 18, 16, 14, 12, 10, 8, 6, 5, 4, 3):
		clustered = _cluster_tris(positions, uvs, tris, mats, div)
		best = clustered
		if len(clustered[2]) <= max_tris:
			return clustered
	pos, uv, kept, kmats = best
	return _decimate_tris(pos, uv, kept, kmats, max_tris, spread=False)


def _cstr(buf: bytes, off: int) -> str:
	if off < 0 or off >= len(buf):
		return ""
	end = buf.find(b"\0", off)
	if end < 0:
		end = len(buf)
	return buf[off:end].decode("latin-1", "replace")


def _mdl_materials(mdl: bytes) -> tuple[list[str], list[str]]:
	"""Source studiohdr texture names and cdmaterials directories."""
	if len(mdl) < 220 or mdl[:4] != b"IDST":
		return [], []
	nt, ti = struct.unpack_from("<ii", mdl, 204)
	nc, ci = struct.unpack_from("<ii", mdl, 212)
	names: list[str] = []
	for i in range(max(0, nt)):
		base = ti + i * 64
		if base + 4 > len(mdl):
			break
		rel = struct.unpack_from("<i", mdl, base)[0]
		name = _cstr(mdl, base + rel)
		if name:
			names.append(name)
	cds: list[str] = []
	for i in range(max(0, nc)):
		slot = ci + i * 4
		if slot + 4 > len(mdl):
			break
		off = struct.unpack_from("<i", mdl, slot)[0]
		cd = _cstr(mdl, off).replace("\\", "/").strip("/")
		if cd:
			cds.append(cd)
	return cds, names


def _additive_vmt(vmt: bytes) -> bool:
	"""Source $additive. The stained-glass shafts live in this mode."""
	lines = []
	for line in vmt.decode("latin-1", "replace").splitlines():
		lines.append(line.split("//", 1)[0])
	body = "\n".join(lines).lower()
	return bool(re.search(r"\$additive\b[^\n]*\b1\b", body))


def _lift_additive(rgba: bytes) -> bytes:
	"""The shaft textures peak around 7/255. Stretch them so the pink,
	yellow and green core survives CMPR and an additive blend."""
	peak = 1
	for i in range(0, len(rgba), 4):
		peak = max(peak, rgba[i], rgba[i + 1], rgba[i + 2])
	if peak >= 96:
		return rgba
	scale = 210.0 / peak
	out = bytearray(len(rgba))
	for i in range(0, len(rgba), 4):
		out[i] = min(255, int(rgba[i] * scale + 0.5))
		out[i + 1] = min(255, int(rgba[i + 1] * scale + 0.5))
		out[i + 2] = min(255, int(rgba[i + 2] * scale + 0.5))
		out[i + 3] = 255
	return bytes(out)


def _tint_rgba(rgba: bytes, tint: tuple[int, int, int]) -> bytes:
	"""Multiply by the glass color, then put the peak back up so the
	shaft stays bright amber instead of a gray cone."""
	tr, tg, tb = tint
	out = bytearray(len(rgba))
	peak = 1
	for i in range(0, len(rgba), 4):
		r = rgba[i] * tr // 255
		g = rgba[i + 1] * tg // 255
		b = rgba[i + 2] * tb // 255
		out[i] = r
		out[i + 1] = g
		out[i + 2] = b
		out[i + 3] = 255
		peak = max(peak, r, g, b)
	if peak >= 200:
		return bytes(out)
	scale = 220.0 / peak
	for i in range(0, len(out), 4):
		out[i] = min(255, int(out[i] * scale + 0.5))
		out[i + 1] = min(255, int(out[i + 1] * scale + 0.5))
		out[i + 2] = min(255, int(out[i + 2] * scale + 0.5))
	return bytes(out)


def _basetexture(vmt: bytes) -> str | None:
	text = vmt.decode("latin-1", "replace")
	m = re.search(r'"\$baseTexture"\s+"([^"]+)"', text, re.I)
	if not m:
		m = re.search(r'\$baseTexture\s+"([^"]+)"', text, re.I)
	if not m:
		return None
	return m.group(1).replace("\\", "/").lstrip("/")


def _load_vtf(vpks, rel: str, max_side: int | None = 256):
	raw = read_first(vpks, rel)
	if not raw:
		return None
	try:
		img = decode_vtf(raw, max_side=max_side)
	except Exception as exc:  # noqa: BLE001
		print(f"  vtf skip {rel}: {exc}")
		return None
	return img.rgba, img.width, img.height


def _resolve_vtf(vpks, cd: str, name: str, max_side: int | None = 256):
	"""Return (rgba, w, h, vtf path) for one MDL material."""
	rel_base = f"materials/{cd}/{name}" if cd else f"materials/{name}"
	cands: list[str] = []
	vmt = read_first(vpks, rel_base + ".vmt")
	if vmt:
		bt = _basetexture(vmt)
		if bt:
			cands.append(f"materials/{bt}.vtf")
	cands.append(rel_base + ".vtf")
	seen: set[str] = set()
	for rel in cands:
		if rel in seen:
			continue
		seen.add(rel)
		got = _load_vtf(vpks, rel, max_side=max_side)
		if got:
			rgba, w, h = got
			return rgba, w, h, rel
	return None


def _wrap_uv(u: float, v: float) -> tuple[float, float]:
	return u - math.floor(u), v - math.floor(v)


def _static_props(bsp: bytes) -> list[tuple[str, tuple, tuple]]:
	"""sprp v5. Coast rocks and the monastery shell are not entities."""
	if len(bsp) < 8 + 36 * 16:
		return []
	off, ln = struct.unpack_from("<ii", bsp, 8 + 35 * 16)
	if off <= 0 or ln < 20 or off + ln > len(bsp):
		return []
	header = bsp[off : off + ln]
	count = struct.unpack_from("<i", header, 0)[0]
	sprp = None
	for i in range(max(0, count)):
		slot = 4 + i * 16
		if slot + 16 > len(header):
			break
		gid, _flags, version, fofs, flen = struct.unpack_from("<IHHII", header, slot)
		if gid.to_bytes(4, "little") == b"prps":
			sprp = (version, fofs, flen)
			break
	if sprp is None or sprp[0] < 4:
		return []
	version, fofs, flen = sprp
	if fofs <= 0 or flen < 8 or fofs + flen > len(bsp):
		return []
	body = bsp[fofs : fofs + flen]
	n = struct.unpack_from("<i", body, 0)[0]
	if n <= 0 or 4 + n * 128 > len(body):
		return []
	names: list[str] = []
	o = 4
	for _i in range(n):
		names.append(body[o : o + 128].split(b"\0", 1)[0].decode("latin-1", "replace"))
		o += 128
	if o + 4 > len(body):
		return []
	nleaf = struct.unpack_from("<i", body, o)[0]
	o += 4 + max(0, nleaf) * 2
	if o + 4 > len(body):
		return []
	nprops = struct.unpack_from("<i", body, o)[0]
	o += 4
	stride = 60 if version >= 5 else 56
	out: list[tuple[float, str, tuple, tuple]] = []
	for i in range(max(0, nprops)):
		at = o + i * stride
		if at + 26 > len(body):
			break
		x, y, z, pitch, yaw, roll, ptype = struct.unpack_from("<6fH", body, at)
		if ptype >= len(names):
			continue
		model = names[ptype].replace("\\", "/")
		low = model.lower()
		if _skip_model(model) or "card" in low:
			continue
		origin = (x, y, z)
		dx = x - SHORE_EYE[0]
		dy = y - SHORE_EYE[1]
		dz = z - SHORE_EYE[2]
		out.append((
			dx * dx + dy * dy + dz * dz,
			_normalize_model(model),
			origin,
			(pitch, yaw, roll),
		))
	out.sort(key=lambda row: row[0])
	return [(model, origin, angles) for _d, model, origin, angles in out]


def _normalize_model(model: str) -> str:
	model = model.replace("\\", "/")
	if model.startswith("./"):
		model = model[2:]
	if not model.startswith("models/"):
		model = "models/" + model
	return model


def bake_props(
	map_bsp: Path,
	lostcoast: Path,
	hl2: Path,
	out_dir: Path,
	eye=None,
	yaw_deg: float = 150.0,
) -> None:
	del eye, yaw_deg
	stem = map_bsp.stem.lower()
	ents_raw = Path(map_bsp).read_bytes()
	off, ln = struct.unpack_from("<ii", ents_raw, 8)
	ents = ents_raw[off : off + ln].decode("latin-1", "replace")

	props: list[tuple[str, tuple, tuple]] = []
	npcs: list[tuple[str, tuple, tuple]] = []
	cur: dict[str, str] = {}
	for line in ents.splitlines():
		line = line.strip()
		if line == "{":
			cur = {}
		elif line == "}":
			cls = cur.get("classname", "")
			model = cur.get("model", "")
			if cls in PROP_CLASSNAMES and model and not _skip_model(model):
				props.append((
					_normalize_model(model),
					_parse_vec(cur.get("origin", "0 0 0")),
					_parse_vec(cur.get("angles", "0 0 0")),
				))
			elif cls in NPC_CLASSNAMES:
				# Honor an explicit model (super soldier, etc.); else default.
				mdl = _normalize_model(model) if model else NPC_CLASSNAMES[cls]
				if not _skip_model(mdl):
					npcs.append((
						mdl,
						_parse_vec(cur.get("origin", "0 0 0")),
						_parse_vec(cur.get("angles", "0 0 0")),
					))
			cur = {}
		elif line.startswith('"'):
			parts = line.split('"')
			if len(parts) >= 4:
				cur[parts[1]] = parts[3]

	statics = _static_props(ents_raw)
	print(f"  static props: {len(statics)}")
	print(f"  npc/canister stand-ins: {len(npcs)}")
	# NPCs first so the tri ceiling cannot drop them after scenery fills the pack.
	props = npcs + props
	props.extend(statics)

	if len(props) > GC_PR_MAX_INST:
		# Keep every NPC, then fill with scenery.
		keep_npcs = [p for p in props if _npc_model(p[0])]
		rest = [p for p in props if not _npc_model(p[0])]
		room = max(0, GC_PR_MAX_INST - len(keep_npcs))
		print(
			f"  prop instances {len(keep_npcs) + len(rest)} exceed {GC_PR_MAX_INST}; "
			f"keeping {len(keep_npcs)} npc + {room} scenery"
		)
		props = keep_npcs + rest[:room]
	print(f"  prop instances: {len(props)} unique models: {len({m for m, _, _ in props})}")

	vpks = open_vpk_roots(
		sorted(Path(lostcoast).glob("*_dir.vpk")) + sorted(Path(hl2).glob("*_dir.vpk"))
	)

	model_cache: dict[str, tuple[SourceMesh, list[int]] | None] = {}
	mat_of: dict[str, int] = {}
	chains: list[tuple[str, int, int, int, int, int, bytes]] = []

	def ensure_mat(key: str, rgba: bytes, w: int, h: int, side: int, label: str) -> int:
		if key in mat_of:
			return mat_of[key]
		if len(chains) >= GC_PR_MAX_MAT:
			print(f"  mat cap {GC_PR_MAX_MAT}; sharing 0 for {label}")
			return 0
		if w != h:
			import numpy as np
			from PIL import Image

			side_px = max(w, h)
			arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
			rgba = Image.fromarray(arr, "RGBA").resize(
				(side_px, side_px), Image.Resampling.BICUBIC,
			).tobytes()
			w = h = side_px
		tw, th, lv, body = rgba_to_cmpr_mips(rgba, w, h, False, max_side=side)
		name = f"p{len(chains):03d}"
		chains.append((name, tw, th, lv, 0, 0, body))
		mat_of[key] = len(chains) - 1
		print(f"  mat {name} {label} {tw}x{th} lv={lv} {len(body)}B")
		return mat_of[key]

	def materials_for(model: str, mdl: bytes) -> list[int]:
		cds, names = _mdl_materials(mdl)
		if not names:
			names = [Path(model).stem]
		if not cds:
			cds = [""]
		ids: list[int] = []
		for ni, name in enumerate(names):
			got = None
			skipped = False
			for cd in cds:
				rel_vmt = f"materials/{cd}/{name}.vmt" if cd else f"materials/{name}.vmt"
				vmt = read_first(vpks, rel_vmt)
				if vmt and _additive_vmt(vmt):
					skipped = True
				got = _resolve_vtf(vpks, cd, name, max_side=None if skipped else 256)
				if got:
					break
			if got is None:
				pix = bytes([140, 110, 70, 255]) * 64
				ids.append(ensure_mat(
					f"fallback:{model}:{ni}", pix, 8, 8, 8,
					f"fallback {model} {name}",
				))
				print(f"  tex fallback {model} {name}")
				continue
			rgba, w, h, rel = got
			if skipped:
				rgba = _lift_additive(rgba)
				if "stain" in name.lower() or "glass" in name.lower():
					rgba = _tint_rgba(rgba, (249, 150, 65))
			idx = ensure_mat(
				rel, rgba, w, h, _max_side(model, name),
				f"{name} ← {rel}",
			)
			# 0x100 marks an additive material. The tri byte keeps the
			# chain index in the low 7 bits and the blend in bit 7.
			ids.append(idx | (0x100 if skipped else 0))
			if skipped:
				print(f"  additive {model} {name}")
		return ids or [0]

	def mesh_for(model: str):
		if model in model_cache:
			return model_cache[model]
		mdl = read_first(vpks, model)
		base = model[: -4] if model.endswith(".mdl") else model
		vvd = read_first(vpks, base + ".vvd")
		vtx = read_first(vpks, base + ".dx90.vtx") or read_first(vpks, base + ".dx80.vtx")
		if not (mdl and vvd and vtx):
			print(f"  skip missing mesh bytes: {model}")
			model_cache[model] = None
			return None
		mesh = None
		# Fisherman VVD uses fixups; lod2 drops legs once mesh vertexoffset
		# is applied. Prefer lod0 so the CapFaces walker is a whole person.
		lods = (0, 1, 2) if "fisherman" in model.lower() else (2, 1, 0)
		for lod in lods:
			try:
				mesh = load_source_mesh(mdl, vvd, vtx, lod=lod)
			except Exception as exc:  # noqa: BLE001
				print(f"  skip parse {model} lod{lod}: {exc}")
				mesh = None
			if mesh and mesh.tris:
				break
		if mesh is None or not mesh.tris:
			model_cache[model] = None
			return None
		ids = materials_for(model, mdl)
		natural = len(mesh.tris)
		cap = _tri_cap(model, natural)
		pos, uv, tris, mats = _simplify(
			model, mesh.positions, mesh.uvs, mesh.tris, mesh.tri_mat, cap,
		)
		mesh = SourceMesh(positions=pos, uvs=uv, tris=tris, tri_mat=mats)
		model_cache[model] = (mesh, ids)
		print(f"  mesh {model}: tris={len(tris)} (cap {cap}, source {natural}) mats={ids}")
		return model_cache[model]

	# Fisherman + canister first so the mat cap cannot leave CapFaces
	# templates on the grey fallback.
	mesh_for(FISHERMAN_WALK_MODEL)
	mesh_for(CANISTER_FLIGHT_MODEL)
	for model, _origin, _angles in props:
		mesh_for(model)
	used_models = []
	for model, _origin, _angles in props:
		if model_cache.get(model) and model not in used_models:
			used_models.append(model)
	wanted = sum(len(model_cache[m][0].tris) for m in used_models)
	if wanted > GC_PR_MODEL_TRIS:
		# Boats, the dock and the NPC stand-ins stay whole; the rest shrinks.
		keep_tris = sum(
			len(model_cache[m][0].tris) for m in used_models
			if _hull_model(m) or _npc_model(m)
		)
		other_tris = wanted - keep_tris
		room = GC_PR_MODEL_TRIS - keep_tris - 4000 - 160
		scale = (room / other_tris) if other_tris and room > 0 else 1.0
		print(f"  shrinking models {wanted} → {GC_PR_MODEL_TRIS} (kept {keep_tris}, other x{scale:.2f})")
		for model in used_models:
			if _hull_model(model) or _npc_model(model):
				continue
			mesh, ids = model_cache[model]
			keep = max(4, int(len(mesh.tris) * scale))
			if keep >= len(mesh.tris):
				continue
			pos, uv, tris, mats = _simplify(
				model, mesh.positions, mesh.uvs, mesh.tris, mesh.tri_mat, keep,
			)
			model_cache[model] = (
				SourceMesh(positions=pos, uvs=uv, tris=tris, tri_mat=mats),
				ids,
			)

	# ---- pack v3: models once, instances carry origin + angles ----
	def mat_byte(ids, local):
		raw = ids[local] if 0 <= local < len(ids) else ids[0]
		if raw < 0:
			return None
		return (raw & 0x7F) | (0x80 if raw & 0x100 else 0)

	model_recs = []   # (dl_ofs, first_bytes, tris, frac)
	model_index: dict[str, int] = {}
	piece_blob = bytearray()
	far_blob = bytearray()

	def emit_pieces(pts, uvs, tris, tri_mat, ids, q, blob=None):
		blob = piece_blob if blob is None else blob
		by_mat: dict[int, list] = {}
		for ti, (a, b, c) in enumerate(tris):
			local = tri_mat[ti] if ti < len(tri_mat) else 0
			mid = mat_byte(ids, local)
			if mid is None:
				continue
			uvs3 = [uvs[i] for i in (a, b, c)]
			# One whole-texture shift per triangle keeps the s16 UVs in
			# range without tearing a face that crosses the wrap.
			su = math.floor(min(u for u, _v in uvs3))
			sv = math.floor(min(v for _u, v in uvs3))
			verts = bytearray()
			for i, (u, v) in zip((a, b, c), uvs3):
				x, y, z = pts[i]
				verts += struct.pack(
					">hhhhh",
					int(max(-32767, min(32767, round(x * q)))),
					int(max(-32767, min(32767, round(y * q)))),
					int(max(-32767, min(32767, round(z * q)))),
					int(max(-32767, min(32767, round((u - su) * 256.0)))),
					int(max(-32767, min(32767, round((v - sv) * 256.0)))),
				)
			by_mat.setdefault(mid, []).append(bytes(verts))
		pieces = []
		for mid in sorted(by_mat):
			tl = by_mat[mid]
			for k in range(0, len(tl), GC_PR_PIECE_TRIS):
				part = tl[k:k + GC_PR_PIECE_TRIS]
				body = b"".join(part)
				body += b"\0" * (-len(body) % 32)
				pieces.append((mid, len(part) * 3, body))
		if not pieces:
			return None
		dl_ofs = len(blob)
		total = 0
		for n, (mid, nverts, body) in enumerate(pieces):
			nxt = (32 + len(pieces[n + 1][2])) if n + 1 < len(pieces) else 0
			blob.extend(struct.pack(">BBHII", mid, 0, nverts, len(body), nxt) + b"\0" * 20)
			blob.extend(body)
			total += nverts // 3
		return dl_ofs, 32 + len(pieces[0][2]), total

	def add_model(key, mesh, ids, rotate_walk=False):
		if key in model_index:
			return model_index[key]
		pts = [((p[1], -p[0], p[2]) if rotate_walk else p) for p in mesh.positions]
		ext = max((abs(c) for p in pts for c in p), default=1.0)
		frac = 0
		while frac < 6 and ext * (1 << (frac + 1)) <= 32000.0:
			frac += 1
		q = float(1 << frac)
		full = emit_pieces(pts, mesh.uvs, mesh.tris, mesh.tri_mat, ids, q)
		if full is None:
			return None
		# Far LOD: about a sixth of the triangles, welded on a coarser grid.
		# The engine picks it when the instance is small on screen.
		# Far chains live in their own region at the end of the file so the
		# engine can keep all of them resident and draw the horizon with no
		# ARAM read. A model too small to reduce gets a copy there.
		far = None
		# At most 96 triangles: a far instance is under 5 % of the view, and
		# the whole region has to fit in about 250 KB of MEM1.
		target = max(8, min(96, len(mesh.tris) // 8))
		if len(mesh.tris) > 16:
			for div in (48, 40, 32, 28, 24, 20, 16, 14, 12, 10, 8, 6, 5, 4, 3):
				lp, lu, lt, lm = _cluster_tris(pts, mesh.uvs, mesh.tris, mesh.tri_mat, div)
				if 0 < len(lt) <= target or (div == 3 and lt):
					# The coarsest grid is taken even when it is over the
					# target: a full copy here cost 400 KB.
					far = emit_pieces(lp, lu, lt, lm, ids, q, far_blob)
					break
		if far is None:
			far = emit_pieces(pts, mesh.uvs, mesh.tris, mesh.tri_mat, ids, q, far_blob)
		model_index[key] = len(model_recs)
		model_recs.append((full[0], full[1], full[2], frac, far[0], far[1], far[2]))
		return model_index[key]

	inst_blob = bytearray()
	n_inst = 0
	inst_tris = 0
	for model, origin, angles in props:
		got = mesh_for(model)
		if got is None:
			continue
		mesh, ids = got
		mi = add_model(model, mesh, ids)
		if mi is None:
			continue
		pitch, yaw, roll = angles
		xform = angle_matrix(pitch, yaw, roll)
		mins = [32767.0] * 3
		maxs = [-32767.0] * 3
		for p in mesh.positions:
			wp = transform_point(xform, origin, p)
			for axis in range(3):
				mins[axis] = min(mins[axis], wp[axis])
				maxs[axis] = max(maxs[axis], wp[axis])
		c16 = lambda v: int(max(-32767, min(32767, round(v))))
		a16 = lambda d: ((int(round(d * 65536.0 / 360.0)) + 32768) & 0xFFFF) - 32768
		inst_blob += struct.pack(
			">HH3h3h3h3hI",
			mi, min(65535, model_recs[mi][2]),
			*[c16(math.floor(v)) for v in mins], *[c16(math.ceil(v)) for v in maxs],
			*[c16(v) for v in origin],
			a16(pitch), a16(yaw), a16(roll),
			0,
		)
		n_inst += 1
		inst_tris += model_recs[mi][2]

	flight_model = 0xFFFF
	flight = mesh_for(CANISTER_FLIGHT_MODEL)
	if flight:
		got_i = add_model(CANISTER_FLIGHT_MODEL, flight[0], flight[1])
		flight_model = 0xFFFF if got_i is None else got_i
	# Fisherman lead walk: bind pose faces +Y, runtime yaw wants +X.
	walk_model = 0xFFFF
	walk = mesh_for(FISHERMAN_WALK_MODEL)
	if walk:
		got_i = add_model("walk:" + FISHERMAN_WALK_MODEL, walk[0], walk[1], rotate_walk=True)
		walk_model = 0xFFFF if got_i is None else got_i

	chain_bytes = sum(len(body) for *_, body in chains)
	model_tris = sum(r[2] for r in model_recs)
	print(f"  far LOD tris: {sum(r[6] for r in model_recs)} in {len(far_blob)} bytes")
	print(
		f"  models: {len(model_recs)} unique tris: {model_tris} "
		f"instances: {n_inst} (placed tris {inst_tris}) mats: {len(chains)} chain={chain_bytes}B"
	)
	out_dir.mkdir(parents=True, exist_ok=True)
	for stale in out_dir.glob("prop_*.cmp"):
		stale.unlink()
	model_blob = bytearray()
	for dl_ofs, first_bytes, tris, frac, far_ofs, far_bytes, far_tris in model_recs:
		model_blob += struct.pack(
			">IIHBBIIHH", dl_ofs, first_bytes, min(65535, tris), frac, 0,
			far_ofs, far_bytes, min(65535, far_tris), 0,
		)
	model_blob += b"\0" * (-len(model_blob) % 32)
	inst_ofs = 32
	models_ofs = inst_ofs + len(inst_blob)
	dl_base = models_ofs + len(model_blob)
	hdr = struct.pack(
		">4sHHHHIIII",
		GC_PR_MAGIC, GC_PR_VERSION, n_inst, len(model_recs), 0,
		inst_ofs, models_ofs, dl_base,
		((flight_model & 0xFFFF) << 16) | (walk_model & 0xFFFF),
	)
	# hdr[7]: file offset of the far-LOD region (runs to the end of file).
	hdr += struct.pack(">I", dl_base + len(piece_blob))
	hdr = hdr + b"\0" * (32 - len(hdr))
	path = out_dir / f"gp_{stem}.gcp"
	path.write_bytes(hdr + inst_blob + model_blob + piece_blob + far_blob)
	tex_path = out_dir / f"pt_{stem}.gct"
	write_gcwt(tex_path, chains)
	print(f"  wrote {path} ({path.stat().st_size} bytes)")
	print(f"  wrote {tex_path} ({tex_path.stat().st_size} bytes)")


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--map", type=Path, required=True)
	ap.add_argument("--lostcoast", type=Path, required=True)
	ap.add_argument("--hl2", type=Path, required=True)
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--eye", nargs=3, type=float, default=None)
	ap.add_argument("--yaw", type=float, default=150.0)
	args = ap.parse_args()
	eye = tuple(args.eye) if args.eye else None
	bake_props(args.map, args.lostcoast, args.hl2, args.out, eye=eye, yaw_deg=args.yaw)


if __name__ == "__main__":
	main()
