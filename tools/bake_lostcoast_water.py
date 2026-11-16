#!/usr/bin/env python3
"""Bake Lost Coast Source water faces → media/gw_<map>.gcw + water CMPR for GX."""
from __future__ import annotations

import argparse
import math
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from bake_lostcoast_map import (  # noqa: E402
	SURF_NODRAW,
	SURF_SKIP,
	SURF_SKY,
	SURF_WARP,
	decimate_ring,
	face_verts,
	load_source_bsp,
	rgba_to_cmpr_mips,
	write_gcwt,
)
from vpk import open_vpk_roots, read_first  # noqa: E402
from vtf import decode_vtf  # noqa: E402

GC_WF_MAGIC = b"GCwf"
GC_WF_VERSION = 1
GC_WATER_MAX_FACES = 512  # engine GC_WATER_ARAM_MAX; the face table lives in ARAM
GC_CAP_MAX_VERTS = 8


def _is_water_mat(mat: str) -> bool:
	low = mat.lower()
	if "waterfog" in low:
		return False
	return "water" in low.split("/")[-1] or "/water" in low or low.startswith("nature/water")


def _polygon_area_xy(poly) -> float:
	return 0.5 * sum(
		poly[i][0] * poly[(i + 1) % len(poly)][1]
		- poly[(i + 1) % len(poly)][0] * poly[i][1]
		for i in range(len(poly))
	)


def _subtract_convex_xy(subject, clip):
	"""Subtract a convex coplanar water polygon, returning convex pieces."""
	orientation = 1.0 if _polygon_area_xy(clip) >= 0 else -1.0
	inside = list(subject)
	pieces = []
	for i, a in enumerate(clip):
		b = clip[(i + 1) % len(clip)]
		if not inside:
			break
		def side(p):
			return orientation * ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))
		inner, outer = [], []
		prev, dp = inside[-1], side(inside[-1])
		for cur in inside:
			dc = side(cur)
			if (dp > 0.001 and dc < -0.001) or (dp < -0.001 and dc > 0.001):
				t = dp / (dp - dc)
				cross = tuple(prev[k] + t * (cur[k] - prev[k]) for k in range(3))
				inner.append(cross)
				outer.append(cross)
			if dc >= -0.001:
				inner.append(cur)
			if dc <= 0.001:
				outer.append(cur)
			prev, dp = cur, dc
		if len(outer) >= 3 and abs(_polygon_area_xy(outer)) > 1.0:
			pieces.append(outer)
		inside = inner
	return pieces


def _convex_parts_xy(poly):
	"""Triangulate a concave polygon; keep convex polygons intact."""
	def cross(a, b, c):
		return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

	sign = 1 if _polygon_area_xy(poly) >= 0 else -1
	if all(sign * cross(poly[i - 1], poly[i], poly[(i + 1) % len(poly)]) >= -0.001
			for i in range(len(poly))):
		return [poly]
	remaining = list(poly)
	triangles = []
	while len(remaining) > 3:
		for i, b in enumerate(remaining):
			a, c = remaining[i - 1], remaining[(i + 1) % len(remaining)]
			if sign * cross(a, b, c) <= 0.001:
				continue
			if any(sign * cross(a, b, p) >= -0.001 and sign * cross(b, c, p) >= -0.001
					and sign * cross(c, a, p) >= -0.001
					for j, p in enumerate(remaining) if j not in (i - 1 if i else len(remaining) - 1, i, (i + 1) % len(remaining))):
				continue
			triangles.append([a, b, c])
			del remaining[i]
			break
		else:
			raise ValueError("cannot triangulate water polygon")
	triangles.append(remaining)
	return triangles


def _nonoverlapping_water(cands):
	"""Remove overlap on horizontal planes while preserving source UV mapping."""
	planes = {}
	out = []
	expanded = []
	for cand in cands:
		pts, normal = cand[6], cand[4]
		if normal[2] >= 0.999 and max(p[2] for p in pts) - min(p[2] for p in pts) <= 0.01:
			for part in _convex_parts_xy(decimate_ring(pts, GC_CAP_MAX_VERTS)):
				expanded.append(cand[:6] + (part,) + cand[7:])
		else:
			expanded.append(cand)
	for cand in expanded:
		score, area, mat, texinfo, normal, dist, pts, cx, cy, cz = cand
		# The shoreline surface consists of convex, horizontal BSP faces.
		# Keep other water surfaces unchanged.
		if normal[2] < 0.999 or max(p[2] for p in pts) - min(p[2] for p in pts) > 0.01:
			out.append(cand)
			continue
		original = decimate_ring(pts, GC_CAP_MAX_VERTS)
		if len(original) < 3 or abs(_polygon_area_xy(original)) <= 1.0:
			continue
		key = round(sum(p[2] for p in original) / len(original), 2)
		blocked = planes.setdefault(key, [])
		pieces = [original]
		for other in blocked:
			ox0, oy0 = min(p[0] for p in other), min(p[1] for p in other)
			ox1, oy1 = max(p[0] for p in other), max(p[1] for p in other)
			next_pieces = []
			for poly in pieces:
				if (max(p[0] for p in poly) <= ox0 or min(p[0] for p in poly) >= ox1
						or max(p[1] for p in poly) <= oy0 or min(p[1] for p in poly) >= oy1):
					next_pieces.append(poly)
				else:
					next_pieces.extend(_subtract_convex_xy(poly, other))
			pieces = next_pieces
			if not pieces:
				break
		blocked.append(original)
		for poly in pieces:
			# Avoid losing coverage if clipping creates more than the record limit.
			parts = [poly] if len(poly) <= GC_CAP_MAX_VERTS else [
				[poly[0], poly[i], poly[i + 1]] for i in range(1, len(poly) - 1)
			]
			for part in parts:
				out.append((score, abs(_polygon_area_xy(part)), mat, texinfo, normal, dist,
					part, cx, cy, cz))
	return out


# Real Lost Coast water materials (lostcoast/nature/water_ati_*.vmt). The
# map uses cheap001 for the surface and beneath for the underwater body.
WATER_VMT = {
	"cheap": "materials/lostcoast/nature/water_ati_cheap001.vmt",
	"beneath": "materials/lostcoast/nature/water_ati_beneath.vmt",
}


def _vec3(text: str | None, default):
	if not text:
		return default
	nums = re.findall(r"[-+]?\d*\.?\d+", text)
	return tuple(float(x) for x in nums[:3]) if len(nums) >= 3 else default


def _f(text: str | None, default: float) -> float:
	if not text:
		return default
	m = re.search(r"[-+]?\d*\.?\d+", text)
	return float(m.group(0)) if m else default


def water_material_params(vpks) -> dict:
	"""Parse the two Source water materials into one parameter set.

	Returns scroll (both layers), fog colour/start/end, reflect amount and
	the distance fade of the cheap material, all straight from the VMTs.
	"""
	from vmt import parse_vmt

	out = {
		"scroll1": (0.02, 0.02), "scroll2": (-0.05, 0.05),
		"fog_rgb": (20, 42, 37), "fog_start": 1.0, "fog_end": 100.0,
		"reflect": 1.5, "cheap_start": 500.0, "cheap_end": 1000.0,
	}
	for key, rel in WATER_VMT.items():
		raw = read_first(vpks, rel)
		if not raw:
			print(f"  water VMT missing: {rel}")
			continue
		v = parse_vmt(raw.decode("latin-1", "replace"))
		if key == "cheap":
			out["scroll1"] = tuple(_vec3(v.get("$scroll1"), (0.02, 0.02, 0.1))[:2])
			out["scroll2"] = tuple(_vec3(v.get("$scroll2"), (-0.05, 0.05, 0.15))[:2])
			out["reflect"] = _f(v.get("$reflectamount"), 1.5)
			out["cheap_start"] = _f(v.get("$cheapwaterstartdistance"), 500.0)
			out["cheap_end"] = _f(v.get("$cheapwaterenddistance"), 1000.0)
		fog = _vec3(v.get("$fogcolor"), (20, 42, 37))
		out["fog_rgb"] = tuple(int(max(0, min(255, round(c)))) for c in fog)
		out["fog_start"] = _f(v.get("$fogstart"), 1.0)
		out["fog_end"] = _f(v.get("$fogend"), 100.0)
		print(f"  water VMT {rel}: fog={out['fog_rgb']} fog {out['fog_start']}..{out['fog_end']}")
	return out


def bake_water(map_bsp: Path, lostcoast: Path, hl2: Path, out_dir: Path) -> None:
	bsp = load_source_bsp(map_bsp)
	stem = map_bsp.stem.lower()
	eyes = bsp["spawns"][:1]
	eye = eyes[0]["origin"] if eyes else (0.0, 0.0, 0.0)

	cands = []
	for face in bsp["faces"]:
		if face["dispinfo"] != -1 or face["numedges"] < 3:
			continue
		ti = face["texinfo"]
		if ti < 0 or ti >= len(bsp["texinfos"]):
			continue
		texinfo = bsp["texinfos"][ti]
		flags = texinfo["flags"]
		if flags & (SURF_SKY | SURF_NODRAW | SURF_SKIP):
			continue
		td = texinfo["texdata"]
		if td < 0 or td >= len(bsp["tex_names"]):
			continue
		mat = bsp["tex_names"][td]
		if not (_is_water_mat(mat) or (flags & SURF_WARP)):
			continue
		if "waterfog" in mat.lower():
			continue
		pts = face_verts(bsp, face)
		if len(pts) < 3:
			continue
		area = max(1, int(face["area"]))
		if area < 256:
			continue
		cx = sum(p[0] for p in pts) / len(pts)
		cy = sum(p[1] for p in pts) / len(pts)
		cz = sum(p[2] for p in pts) / len(pts)
		dx, dy, dz = cx - eye[0], cy - eye[1], cz - eye[2]
		d2 = dx * dx + dy * dy + dz * dz
		score = area / (1.0 + d2 * 0.00001)
		pl = bsp["planes"][face["planenum"] % len(bsp["planes"])]
		n = pl[0]
		if face["side"]:
			n = (-n[0], -n[1], -n[2])
		cands.append((score, area, mat, texinfo, n, pl[1] if not face["side"] else -pl[1], pts, cx, cy, cz))
	cands.sort(key=lambda x: -x[0])
	before_overlap = len(cands)
	cands = _nonoverlapping_water(cands)
	print(f"  water overlap cleanup: {before_overlap} source faces -> {len(cands)} pieces")
	params = water_material_params(open_vpk_roots(
		sorted(Path(lostcoast).glob("*_dir.vpk")) + sorted(Path(hl2).glob("*_dir.vpk"))
	))
	print(f"  water face candidates: {len(cands)} (limit {GC_WATER_MAX_FACES} unique faces)")

	# Source Water shader $normalmap (water_ati_cheap001 / water_ati_beneath):
	#   "$normalmap" "dev/water_normal"  (+ "$bumpmap" "dev/water_dudv")
	# Bake the same normal into water_gx.cmp for GX IndTex reflection warp.
	vpks = open_vpk_roots(
		sorted(Path(lostcoast).glob("*_dir.vpk")) + sorted(Path(hl2).glob("*_dir.vpk"))
	)
	water_rgba = None
	# Prefer VMT $normalmap from the map's cheap water material when present.
	nrm_candidates = []
	for mat_rel in (
		"materials/lostcoast/nature/water_ati_cheap001.vmt",
		"materials/lostcoast/nature/water_ati_beneath.vmt",
	):
		vmt = read_first(vpks, mat_rel)
		if not vmt:
			continue
		text = vmt.decode("latin1", "replace")
		bump = re.search(r'\$bumpmap\s+"([^"]+)"', text, re.I)
		if bump:
			nrm_candidates.append(f"materials/{bump.group(1)}.vtf")
			print(f"  VMT {mat_rel} $bumpmap (dudv) → {bump.group(1)}")
		m = re.search(r'\$normalmap\s+"([^"]+)"', text, re.I)
		if m:
			nrm_candidates.append(f"materials/{m.group(1)}.vtf")
			print(f"  VMT {mat_rel} $normalmap → {m.group(1)}")
	nrm_candidates.extend(
		(
			"materials/dev/water_dudv.vtf",
			"materials/dev/water_normal.vtf",
			"materials/nature/water_coast01.vtf",
		)
	)
	seen = set()
	for rel in nrm_candidates:
		key = rel.lower()
		if key in seen:
			continue
		seen.add(key)
		raw = read_first(vpks, rel)
		if raw:
			img = decode_vtf(raw)
			water_rgba = (img.rgba, img.width, img.height)
			print(f"  water $normalmap {rel} {img.width}x{img.height}")
			break
	if water_rgba is None:
		# Procedural ripple fallback
		import numpy as np

		w = h = 64
		yy, xx = np.mgrid[0:h, 0:w]
		wave = (np.sin(xx * 0.4) + np.cos(yy * 0.35) + np.sin((xx + yy) * 0.2)) * 40 + 140
		rgb = np.stack((wave * 0.55, wave * 0.75, wave * 0.95), -1).clip(0, 255).astype(np.uint8)
		a = np.full((h, w, 1), 200, np.uint8)
		water_rgba = (np.concatenate((rgb, a), -1).tobytes(), w, h)
		print("  water tex procedural ripple 64x64")

	media = out_dir / "media"
	media.mkdir(parents=True, exist_ok=True)
	rgba, tw, th = water_rgba
	# Cap to 64² for MEM1 tip
	if tw > 64 or th > 64:
		from PIL import Image
		import numpy as np

		img = Image.frombytes("RGBA", (tw, th), rgba).resize((64, 64), Image.Resampling.LANCZOS)
		rgba, tw, th = img.tobytes(), 64, 64
	# Single 64² CMPR level for GX scroll (same layout as sky sides).
	from bake_lostcoast_map import gx_cmpr_encode_blocks
	import numpy as np

	arr = np.frombuffer(rgba, dtype=np.uint8).reshape(th, tw, 4)
	if (tw, th) != (64, 64):
		from PIL import Image

		img = Image.fromarray(arr, "RGBA").resize((64, 64), Image.Resampling.LANCZOS)
		arr = np.asarray(img)
		tw = th = 64
	opaque = arr[..., 3] > 8
	cmpr = gx_cmpr_encode_blocks(arr[..., :3], opaque)
	assert len(cmpr) == 64 * 64 // 2
	cmp_path = media / "water_gx.cmp"
	cmp_path.write_bytes(cmpr)
	print(f"  wrote {cmp_path} {len(cmpr)} bytes")
	write_water_ia8(vpks, media)
	wt_w = wt_h = 64

	# Pack records 128B:
	# 0 nverts, 1 flags, 2-3 pad, 4-7 area,
	# 8-11 scroll_u f32 BE, 12-15 scroll_v f32 BE,
	# 16-63 pts s16[8][3], 64-95 st s16[8][2] (/64),
	# 96-111 plane nxyz+dist f32 BE, 112-115 tint RGBA8888
	recs = bytearray()
	seen_faces = set()
	duplicates = 0
	for score, area, mat, texinfo, n, dist, pts, cx, cy, cz in cands:
		dpts = decimate_ring(pts, GC_CAP_MAX_VERTS)
		nv = len(dpts)
		rec = bytearray(128)
		rec[0] = nv & 0xFF
		rec[1] = 1 if (texinfo["flags"] & SURF_WARP) else 0
		struct.pack_into(">H", rec, 2, 0)
		struct.pack_into(">I", rec, 4, int(area) & 0xFFFFFFFF)
		# Layer 1 scroll (VMT $scroll1) in the record; engine uses it for the dudv warp.
		struct.pack_into(">ff", rec, 8, params["scroll1"][0], params["scroll1"][1])
		v0, v1 = texinfo["vecs"]
		tdw = float(max(1, bsp["texdatas"][texinfo["texdata"]]["width"]))
		tdh = float(max(1, bsp["texdatas"][texinfo["texdata"]]["height"]))
		for i in range(GC_CAP_MAX_VERTS):
			if i < nv:
				p = dpts[i]
				struct.pack_into(
					">hhh",
					rec,
					16 + i * 6,
					int(max(-32767, min(32767, round(p[0])))),
					int(max(-32767, min(32767, round(p[1])))),
					int(max(-32767, min(32767, round(p[2])))),
				)
				s = p[0] * v0[0] + p[1] * v0[1] + p[2] * v0[2] + v0[3]
				t = p[0] * v1[0] + p[1] * v1[1] + p[2] * v1[2] + v1[3]
				struct.pack_into(
					">hh",
					rec,
					64 + i * 4,
					# Source's own texture coordinate (texels over the material's
					# texdata size), x64: the water shader's bump lookups are
					# written in these units.
					int(max(-32767, min(32767, round(s / tdw * 64.0)))),
					int(max(-32767, min(32767, round(t / tdh * 64.0)))),
				)
		struct.pack_into(">ffff", rec, 96, float(n[0]), float(n[1]), float(n[2]), float(dist))
		# Body tint = VMT $fogcolor (retail cheap water). Runtime lifts +
		# distance-hazes this; the old (64,140,160) read as bright cyan.
		fog = params["fog_rgb"]
		rec[112:116] = bytes((fog[0], fog[1], fog[2], 200))
		# Extension (bytes 116-127): fog RGB, reflect x100, fog start/end,
		# cheap-reflection fade start/end (units / 4).
		rec[116:119] = bytes(fog)
		rec[119] = int(max(0, min(255, round(params["reflect"] * 100))))
		struct.pack_into(">HH", rec, 120, int(max(0, min(65535, params["fog_start"]))),
			int(max(0, min(65535, params["fog_end"]))))
		struct.pack_into(">HH", rec, 124, int(params["cheap_start"] / 4), int(params["cheap_end"] / 4))
		# BSP water can contain opposite-winding copies of the same face.
		# GX renders both sides, so identical geometry and vertex UVs would
		# draw the same translucent surface and reflection more than once.
		vertices = tuple(sorted(
			(struct.unpack_from(">hhh", rec, 16 + i * 6),
			 struct.unpack_from(">hh", rec, 64 + i * 4))
			for i in range(nv)
		))
		plane = tuple(round(v, 5) for v in (n[0], n[1], n[2], dist))
		key = (rec[1], bytes(rec[8:16]), vertices, plane, bytes(rec[112:128]))
		if key in seen_faces:
			duplicates += 1
			continue
		seen_faces.add(key)
		recs += rec
		if len(seen_faces) == GC_WATER_MAX_FACES:
			break

	hdr = bytearray(32)
	hdr[0:4] = GC_WF_MAGIC
	struct.pack_into(">HH", hdr, 4, GC_WF_VERSION, len(seen_faces))
	struct.pack_into(">I", hdr, 8, 32)  # faces ofs
	struct.pack_into(">HH", hdr, 12, wt_w, wt_h)
	# $scroll2 drives the second dudv phase. Bytes 16-23 were unused.
	struct.pack_into(">ff", hdr, 16, params["scroll2"][0], params["scroll2"][1])
	path = media / f"gw_{stem}.gcw"
	path.write_bytes(bytes(hdr) + bytes(recs))
	print(f"  wrote {path} faces={len(seen_faces)} duplicates={duplicates} bytes={path.stat().st_size}")


def write_water_ia8(vpks, media: Path, size: int = 64) -> None:
	"""Lost Coast $normalmap as GX IA8. CMPR quantizes the bump and the
	indirect warp falls apart; IA8 keeps the 128-neutral du/dv."""
	from PIL import Image
	import numpy as np

	raw = None
	src = None
	for rel in ("materials/dev/water_normal.vtf", "materials/dev/water_dudv.vtf"):
		raw = read_first(vpks, rel)
		if raw:
			src = rel
			break
	if not raw:
		print("  water IA8 skipped: no water_normal / water_dudv")
		return
	img = decode_vtf(raw)
	pic = Image.frombytes("RGBA", (img.width, img.height), img.rgba).resize(
		(size, size), Image.Resampling.BOX)
	rgba = np.asarray(pic).astype(np.int16)
	# The source normal only wanders about ±3 around 128, which is a flat
	# sheet once it is a colour. Stretch that offset so the crests reach
	# the ends of the byte. 128 stays the no-warp centre.
	rg = np.clip(128 + (rgba[..., :2] - 128) * 8, 0, 255).astype(np.uint8)
	tw = size // 4
	out = bytearray(size * size * 2)
	for ty in range(tw):
		for tx in range(tw):
			tile = (ty * tw + tx) * 32
			for y in range(4):
				for x in range(4):
					i, a = rg[ty * 4 + y, tx * 4 + x]
					o = tile + (y * 4 + x) * 2
					out[o] = int(i)
					out[o + 1] = int(a)
	path = media / "water_gx.ia8"
	path.write_bytes(out)
	print(f"  wrote {path} {len(out)} bytes from {src} {img.width}x{img.height} → IA8 {size}")


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--map", type=Path, required=True)
	ap.add_argument("--lostcoast", type=Path, required=True)
	ap.add_argument("--hl2", type=Path, required=True)
	ap.add_argument("--out", type=Path, required=True)
	args = ap.parse_args()
	bake_water(args.map, args.lostcoast, args.hl2, args.out)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
