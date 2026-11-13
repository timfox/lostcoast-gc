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
GC_PR_VERSION = 2
GC_PR_MAX_TRIS = 65535
GC_PR_MAX_INST = 512
GC_PR_MAX_MAT = 96
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
	if _hull_model(model) or _npc_model(model):
		# Boats keep long plates; NPC wings/limbs shred under spread decimate.
		return _decimate_tris(
			positions, uvs, tris, mats, max_tris, spread=False, keep_long=True,
		)
	if not _shell_model(model) or len(tris) <= max_tris:
		return _decimate_tris(
			positions, uvs, tris, mats, max_tris,
			spread=_spread_model(model) and not _hull_model(model),
			keep_long=_hull_model(model),
		)
	best = (positions, uvs, tris, mats)
	for div in (24, 20, 18, 16, 14, 12, 10, 8, 6, 5, 4, 3):
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
		cap = _tri_cap(model, len(mesh.tris))
		pos, uv, tris, mats = _simplify(
			model, mesh.positions, mesh.uvs, mesh.tris, mesh.tri_mat, cap,
		)
		mesh = SourceMesh(positions=pos, uvs=uv, tris=tris, tri_mat=mats)
		model_cache[model] = (mesh, ids)
		print(f"  mesh {model}: tris={len(tris)} (cap {cap}) mats={ids}")
		return model_cache[model]

	# Fisherman + canister first so the mat cap cannot leave CapFaces
	# templates on the grey fallback.
	mesh_for(FISHERMAN_WALK_MODEL)
	mesh_for(CANISTER_FLIGHT_MODEL)
	for model, _origin, _angles in props:
		mesh_for(model)
	wanted = 0
	for model, _origin, _angles in props:
		got = model_cache.get(model)
		if got:
			wanted += len(got[0].tris)
	if wanted > GC_PR_MAX_TRIS and wanted > 0:
		# Largest-first on a hull drops the small skin and leaves the ribs.
		# Boats and the dock stay whole; everything else absorbs the cut.
		hull_tris = 0
		other_tris = 0
		for model, _origin, _angles in props:
			got = model_cache.get(model)
			if not got:
				continue
			n = len(got[0].tris)
			if _hull_model(model) or _npc_model(model):
				hull_tris += n
			else:
				other_tris += n
		# Reserve local-space canister flight + fisherman walker.
		room = GC_PR_MAX_TRIS - 128 - hull_tris - 96 - 4000
		scale = (room / other_tris) if other_tris and room > 0 else 1.0
		print(
			f"  shrinking meshes {wanted} → {GC_PR_MAX_TRIS} "
			f"(hulls+npc {hull_tris} kept, other x{scale:.2f})"
		)
		for model, got in list(model_cache.items()):
			if not got or _hull_model(model) or _npc_model(model):
				continue
			mesh, ids = got
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

	world_tris: list[tuple[int, tuple, tuple, tuple]] = []
	inst_recs: list[tuple[int, int, tuple, tuple]] = []
	for model, origin, angles in props:
		got = mesh_for(model)
		if got is None:
			continue
		mesh, ids = got
		pitch, yaw, roll = angles
		xform = angle_matrix(pitch, yaw, roll)
		first = len(world_tris)
		mins = [32767, 32767, 32767]
		maxs = [-32767, -32767, -32767]
		for ti, (a, b, c) in enumerate(mesh.tris):
			if len(world_tris) >= GC_PR_MAX_TRIS:
				break
			local = mesh.tri_mat[ti] if ti < len(mesh.tri_mat) else 0
			raw = ids[local] if 0 <= local < len(ids) else ids[0]
			if raw < 0:
				continue
			mid = (raw & 0x7F) | (0x80 if raw & 0x100 else 0)
			vs = []
			for idx in (a, b, c):
				wp = transform_point(xform, origin, mesh.positions[idx])
				uu, vv = _wrap_uv(*mesh.uvs[idx])
				vs.append((wp[0], wp[1], wp[2], uu, vv))
				for axis in range(3):
					mins[axis] = min(mins[axis], wp[axis])
					maxs[axis] = max(maxs[axis], wp[axis])
			world_tris.append((mid, vs[0], vs[1], vs[2]))
		count = len(world_tris) - first
		if count > 0:
			inst_recs.append((first, count, tuple(mins), tuple(maxs)))
			if "chair" in model.lower():
				print(f"  chair instance {origin} tris={count} mat={ids}")
		if len(world_tris) >= GC_PR_MAX_TRIS:
			print("  hit tri ceiling")
			break

	# Local-space canister for CapFaces flight (not an instance — hdr[5] only).
	flight_first = 0
	flight_count = 0
	flight = mesh_for(CANISTER_FLIGHT_MODEL)
	if flight and len(world_tris) + 4 < GC_PR_MAX_TRIS:
		mesh, ids = flight
		flight_first = len(world_tris)
		for ti, (a, b, c) in enumerate(mesh.tris):
			if len(world_tris) >= GC_PR_MAX_TRIS:
				break
			local = mesh.tri_mat[ti] if ti < len(mesh.tri_mat) else 0
			raw = ids[local] if 0 <= local < len(ids) else ids[0]
			if raw < 0:
				continue
			mid = (raw & 0x7F) | (0x80 if raw & 0x100 else 0)
			vs = []
			for idx in (a, b, c):
				p = mesh.positions[idx]
				uu, vv = _wrap_uv(*mesh.uvs[idx])
				vs.append((p[0], p[1], p[2], uu, vv))
			world_tris.append((mid, vs[0], vs[1], vs[2]))
		flight_count = len(world_tris) - flight_first
		print(f"  canister flight mesh: first={flight_first} count={flight_count}")
	else:
		print("  canister flight mesh: skipped")

	# Local-space Source fisherman for CapFaces lead walk (hdr[6]).
	# Rotate +Y-facing bind pose to +X so runtime yaw matches Source.
	walk_first = 0
	walk_count = 0
	walk = mesh_for(FISHERMAN_WALK_MODEL)
	if walk and len(world_tris) + 4 < GC_PR_MAX_TRIS:
		mesh, ids = walk
		walk_first = len(world_tris)
		for ti, (a, b, c) in enumerate(mesh.tris):
			if len(world_tris) >= GC_PR_MAX_TRIS:
				break
			local = mesh.tri_mat[ti] if ti < len(mesh.tri_mat) else 0
			raw = ids[local] if 0 <= local < len(ids) else ids[0]
			if raw < 0:
				continue
			mid = (raw & 0x7F) | (0x80 if raw & 0x100 else 0)
			vs = []
			for idx in (a, b, c):
				px, py, pz = mesh.positions[idx]
				# +Y face → +X face: (x,y) → (y, -x)
				rx, ry = py, -px
				uu, vv = _wrap_uv(*mesh.uvs[idx])
				vs.append((rx, ry, pz, uu, vv))
			world_tris.append((mid, vs[0], vs[1], vs[2]))
		walk_count = len(world_tris) - walk_first
		print(f"  fisherman walk mesh: first={walk_first} count={walk_count}")
	else:
		print("  fisherman walk mesh: skipped")

	chain_bytes = sum(len(body) for *_, body in chains)
	print(
		f"  world tris: {len(world_tris)} instances: {len(inst_recs)} "
		f"mats: {len(chains)} chain={chain_bytes}B"
	)

	out_dir.mkdir(parents=True, exist_ok=True)
	for stale in out_dir.glob("prop_*.cmp"):
		stale.unlink()

	def pack_vert(v):
		x, y, z, u, vv = v
		return struct.pack(
			">hhhHH",
			int(max(-32767, min(32767, round(x)))),
			int(max(-32767, min(32767, round(y)))),
			int(max(-32767, min(32767, round(z)))),
			int(max(0, min(65535, round(u * 256.0)))) & 0xFFFF,
			int(max(0, min(65535, round(vv * 256.0)))) & 0xFFFF,
		)

	inst_blob = bytearray()
	for first, count, mins, maxs in inst_recs:
		inst_blob += struct.pack(
			">HH3h3h",
			first, count,
			*[int(max(-32767, min(32767, round(v)))) for v in mins],
			*[int(max(-32767, min(32767, round(v)))) for v in maxs],
		)
	tri_blob = bytearray()
	for mid, v0, v1, v2 in world_tris:
		tri_blob.append(mid & 0xFF)
		tri_blob.append(0)
		tri_blob += pack_vert(v0)
		tri_blob += pack_vert(v1)
		tri_blob += pack_vert(v2)

	inst_ofs = 32
	tris_ofs = inst_ofs + len(inst_blob)
	# hdr[5] BE: (flight_first << 16) | flight_count
	# hdr[6] BE: (walk_first << 16) | walk_count  (fisherman lead path)
	hdr = struct.pack(
		">4sHHHHIIII",
		GC_PR_MAGIC,
		GC_PR_VERSION,
		0,  # materials live in the pt_ pack, not an atlas page
		len(inst_recs),
		len(world_tris),
		inst_ofs,
		tris_ofs,
		((flight_first & 0xFFFF) << 16) | (flight_count & 0xFFFF),
		((walk_first & 0xFFFF) << 16) | (walk_count & 0xFFFF),
	)
	hdr = hdr + b"\0" * (32 - len(hdr))
	path = out_dir / f"gp_{stem}.gcp"
	path.write_bytes(hdr + inst_blob + tri_blob)
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
