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
	cands = cands[:GC_WATER_MAX_FACES]
	params = water_material_params(open_vpk_roots(
		sorted(Path(lostcoast).glob("*_dir.vpk")) + sorted(Path(hl2).glob("*_dir.vpk"))
	))
	print(f"  water faces selected: {len(cands)}/{GC_WATER_MAX_FACES}")

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
					int(max(-32767, min(32767, round(s / 2.0)))),
					int(max(-32767, min(32767, round(t / 2.0)))),
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
		recs += rec

	hdr = bytearray(32)
	hdr[0:4] = GC_WF_MAGIC
	struct.pack_into(">HH", hdr, 4, GC_WF_VERSION, len(cands))
	struct.pack_into(">I", hdr, 8, 32)  # faces ofs
	struct.pack_into(">HH", hdr, 12, wt_w, wt_h)
	# $scroll2 drives the second dudv phase. Bytes 16-23 were unused.
	struct.pack_into(">ff", hdr, 16, params["scroll2"][0], params["scroll2"][1])
	path = media / f"gw_{stem}.gcw"
	path.write_bytes(bytes(hdr) + bytes(recs))
	print(f"  wrote {path} faces={len(cands)} bytes={path.stat().st_size}")


def write_water_ia8(vpks, media: Path, size: int = 128) -> None:
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
