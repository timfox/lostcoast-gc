#!/usr/bin/env python3
"""Host soft-raster of Lost Coast: skybox + CapFaces and/or full Source geo."""
from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from bake_lostcoast_map import (  # noqa: E402
	SURF_HINT,
	SURF_NODRAW,
	SURF_SKIP,
	SURF_SKY,
	SURF_TRANS,
	SURF_TRIGGER,
	SURF_WARP,
	face_verts,
	load_source_bsp,
)


def decode_gx_cmpr(data: bytes, size: int = 128) -> np.ndarray:
	out = np.zeros((size, size, 3), np.uint8)
	off = 0

	def from565(v: int) -> np.ndarray:
		r = (v >> 11) & 31
		g = (v >> 5) & 63
		b = v & 31
		return np.array(((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)), np.uint8)

	for ty in range(0, size // 4, 2):
		for tx in range(0, size // 4, 2):
			for by, bx in ((ty, tx), (ty, tx + 1), (ty + 1, tx), (ty + 1, tx + 1)):
				c0, c1 = struct.unpack_from(">HH", data, off)
				bits = struct.unpack_from(">I", data, off + 4)[0]
				off += 8
				e0, e1 = from565(c0).astype(np.int16), from565(c1).astype(np.int16)
				if c0 > c1:
					pal = np.stack(
						(
							e0,
							e1,
							(2 * e0 + e1) // 3,
							(e0 + 2 * e1) // 3,
						)
					).astype(np.uint8)
				else:
					mid = ((e0 + e1) // 2).astype(np.uint8)
					pal = np.stack((e0.astype(np.uint8), e1.astype(np.uint8), mid, np.zeros(3, np.uint8)))
				for row in range(4):
					for col in range(4):
						shift = 30 - ((row * 4 + col) * 2)
						idx = (bits >> shift) & 3
						yy, xx = by * 4 + row, bx * 4 + col
						if yy < size and xx < size:
							out[yy, xx] = pal[idx]
	return out


def load_sky_sides(media: Path, skyname: str = "sky_lostcoast_hdr") -> dict[str, np.ndarray]:
	sides = {}
	for side in ("rt", "lf", "ft", "bk", "up", "dn"):
		p = media / f"sky_{skyname}{side}.cmp"
		if not p.is_file():
			raise SystemExit(f"missing sky side {p}")
		sides[side] = decode_gx_cmpr(p.read_bytes(), 128)
	return sides


def parse_capfaces(path: Path):
	d = path.read_bytes()
	if d[:4] != b"GCcf":
		raise SystemExit(f"not GCcf: {path}")
	_ver, _flags, eye_n, face_n = struct.unpack_from(">HHHH", d, 4)
	faces_ofs, _order_ofs = struct.unpack_from(">II", d, 24)
	eyes = [struct.unpack_from(">ffffff", d, 32 + i * 24) for i in range(eye_n)]
	faces = []
	for i in range(face_n):
		off = faces_ofs + i * 128
		nv = d[off]
		pts = [struct.unpack_from(">hhh", d, off + 12 + v * 6) for v in range(nv)]
		faces.append({"nverts": nv, "pts": pts, "tex_hash": struct.unpack_from(">I", d, off + 8)[0]})
	return eyes, faces


def look_basis(eye, yaw_deg, pitch_deg=0.0):
	yaw = math.radians(yaw_deg)
	pitch = math.radians(pitch_deg)
	forward = np.array(
		[math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)],
		np.float64,
	)
	world_up = np.array([0.0, 0.0, 1.0])
	right = np.cross(forward, world_up)
	n = np.linalg.norm(right)
	right = np.array([1.0, 0.0, 0.0]) if n < 1e-6 else right / n
	up = np.cross(right, forward)
	up /= max(1e-9, np.linalg.norm(up))
	return np.asarray(eye, np.float64), forward, right, up


def draw_sky(buf: np.ndarray, sky: dict[str, np.ndarray], forward, right, up, fov):
	h, w, _ = buf.shape
	f = 1.0 / math.tan(math.radians(fov) * 0.5)
	aspect = w / h
	ys = (1.0 - (np.arange(h) + 0.5) / h * 2.0) / f
	xs = ((np.arange(w) + 0.5) / w * 2.0 - 1.0) * aspect / f
	xx, yy = np.meshgrid(xs, ys)
	dirs = forward[None, None, :] + right[None, None, :] * xx[..., None] + up[None, None, :] * yy[..., None]
	dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True).clip(1e-9)
	face = np.argmax(np.abs(dirs), axis=-1)
	rgb = np.zeros((h, w, 3), np.uint8)
	for axis, pos_name, neg_name in ((0, "rt", "lf"), (1, "ft", "bk"), (2, "up", "dn")):
		mask = face == axis
		if not np.any(mask):
			continue
		sign = dirs[..., axis]
		for m, name, s in ((mask & (sign >= 0), pos_name, 1.0), (mask & (sign < 0), neg_name, -1.0)):
			if not np.any(m):
				continue
			d = dirs[m]
			t = 1.0 / np.maximum(np.abs(d[:, axis]), 1e-9)
			p = d * t[:, None]
			if axis == 0:
				u = (-s * p[:, 1] + 1.0) * 0.5
				v = (-p[:, 2] + 1.0) * 0.5
			elif axis == 1:
				u = (s * p[:, 0] + 1.0) * 0.5
				v = (-p[:, 2] + 1.0) * 0.5
			else:
				u = (p[:, 0] + 1.0) * 0.5
				v = (s * p[:, 1] + 1.0) * 0.5
			tex = sky[name]
			th, tw = tex.shape[:2]
			ui = np.clip((u * tw).astype(np.int32), 0, tw - 1)
			vi = np.clip((v * th).astype(np.int32), 0, th - 1)
			rgb[m] = tex[vi, ui]
	buf[:] = rgb


def project_poly(pts, eye, forward, right, up, fov, w, h):
	f = 1.0 / math.tan(math.radians(fov) * 0.5)
	aspect = w / h
	screen = []
	depth = 0.0
	for p in pts:
		v = np.asarray(p, np.float64) - eye
		x = float(np.dot(v, right))
		y = float(np.dot(v, up))
		z = float(np.dot(v, forward))
		if z <= 8.0:
			return None, None
		sx = (x / z * f / aspect + 1.0) * 0.5 * w
		sy = (1.0 - y / z * f) * 0.5 * h
		screen.append((sx, sy))
		depth += z
	return screen, depth / len(pts)


def face_color(n, mat: str) -> tuple[int, int, int]:
	shade = 0.45 + 0.55 * max(0.15, abs(n[2]) * 0.5 + max(0.0, n[0]) * 0.25 + max(0.0, n[1]) * 0.15)
	low = mat.lower()
	if any(k in low for k in ("rock", "cliff", "stone")):
		base = np.array([118.0, 108.0, 92.0])
	elif any(k in low for k in ("wood", "dock", "plank")):
		base = np.array([108.0, 78.0, 48.0])
	elif any(k in low for k in ("concrete", "plaster", "brick")):
		base = np.array([145.0, 142.0, 136.0])
	elif any(k in low for k in ("grass", "dirt", "sand", "ground")):
		base = np.array([88.0, 108.0, 58.0])
	elif "metal" in low:
		base = np.array([128.0, 132.0, 140.0])
	else:
		h = abs(hash(mat)) % 160 + 50
		base = np.array([float(h), float((h * 3) % 140 + 50), float((h * 7) % 120 + 60)])
	return tuple(np.clip(base * shade, 0, 255).astype(np.uint8).tolist())


def collect_all_faces(bsp, bsp_path: Path | None = None):
	skip = SURF_SKY | SURF_WARP | SURF_TRANS | SURF_NODRAW | SURF_HINT | SURF_SKIP | SURF_TRIGGER
	out = []
	for face in bsp["faces"]:
		if face["dispinfo"] != -1 or face["numedges"] < 3:
			continue
		ti = face["texinfo"]
		if ti < 0 or ti >= len(bsp["texinfos"]):
			continue
		texinfo = bsp["texinfos"][ti]
		if texinfo["flags"] & skip:
			continue
		td = texinfo["texdata"]
		if td < 0 or td >= len(bsp["tex_names"]):
			continue
		mat = bsp["tex_names"][td]
		if mat.startswith("tools/") or "water" in mat.split("/")[-1]:
			continue
		pts = face_verts(bsp, face)
		if len(pts) < 3:
			continue
		if len(pts) > 16:
			# decimate ring
			n = len(pts)
			pts = [pts[(i * n) // 16] for i in range(16)]
		pl = bsp["planes"][face["planenum"] % len(bsp["planes"])]
		n = pl[0]
		if face["side"]:
			n = (-n[0], -n[1], -n[2])
		out.append({"pts": pts, "mat": mat, "n": n})
	# Displacement / patch quads (cliffs, terrain)
	if bsp_path and bsp_path.is_file():
		from source_disp import iter_disp_quads

		for q in iter_disp_quads(bsp, bsp_path.read_bytes(), max_power=2):
			nx, ny, nz = q["n"]
			ln = (nx * nx + ny * ny + nz * nz) ** 0.5 or 1.0
			out.append({"pts": q["pts"], "mat": q["mat"], "n": (nx / ln, ny / ln, nz / ln)})
	return out


def paint(buf, items, eye, forward, right, up, fov):
	"""Painter's algorithm — far → near polygons via PIL."""
	h, w = buf.shape[:2]
	img = Image.fromarray(buf, "RGB")
	draw = ImageDraw.Draw(img, "RGBA")
	scored = []
	for pts, color in items:
		screen, depth = project_poly(pts, eye, forward, right, up, fov, w, h)
		if screen is None or len(screen) < 3:
			continue
		# reject tiny / backfacing screen polys
		area = 0.0
		for i in range(len(screen)):
			x0, y0 = screen[i]
			x1, y1 = screen[(i + 1) % len(screen)]
			area += x0 * y1 - x1 * y0
		if abs(area) < 4.0:
			continue
		scored.append((depth, screen, color))
	scored.sort(key=lambda t: -t[0])  # far first
	for _d, screen, color in scored:
		draw.polygon(screen, fill=(*color, 255))
	return np.asarray(img.convert("RGB")), len(scored)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--mode", choices=("capfaces+sky", "full+sky"), default="capfaces+sky")
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--w", type=int, default=640)
	ap.add_argument("--h", type=int, default=480)
	ap.add_argument("--yaw", type=float, default=None)
	ap.add_argument("--pitch", type=float, default=8.0)
	ap.add_argument("--fov", type=float, default=75.0)
	args = ap.parse_args()

	media = ROOT / "cache/bake/media"
	sky = load_sky_sides(media)
	bsp = load_source_bsp(ROOT / "content/lostcoast/maps/d2_lostcoast.bsp")

	if args.mode == "capfaces+sky":
		eyes, faces = parse_capfaces(media / "cf_d2_lostcoast.gcp")
		eye_origin = list(eyes[0][:3]) if eyes else bsp["spawns"][0]["origin"]
		yaw = args.yaw if args.yaw is not None else (float(eyes[0][4]) if eyes else float(bsp["spawns"][0]["angles"][1]))
		items = []
		for f in faces:
			h = f["tex_hash"]
			color = ((h >> 16) & 200) + 40, ((h >> 8) & 200) + 40, (h & 200) + 40
			items.append((f["pts"], color))
		title = f"Lost Coast in-level — skybox + CapFaces ({len(faces)} faces)"
		sub = "media/sky_sky_lostcoast_hdr*.cmp + cf_d2_lostcoast.gcp (same disc bytes)"
		nfaces = len(faces)
	else:
		eye_origin = bsp["spawns"][0]["origin"]
		yaw = args.yaw if args.yaw is not None else float(bsp["spawns"][0]["angles"][1])
		bsp_path = ROOT / "content/lostcoast/maps/d2_lostcoast.bsp"
		all_faces = collect_all_faces(bsp, bsp_path)
		items = [(f["pts"], face_color(f["n"], f["mat"])) for f in all_faces]
		# Water sheets (cyan tint) from gw pack if present
		gw = ROOT / "cache/bake/media/gw_d2_lostcoast.gcw"
		water_n = 0
		if gw.is_file():
			import struct as _st

			raw = gw.read_bytes()
			face_n = _st.unpack_from(">H", raw, 6)[0]
			for i in range(face_n):
				off = 32 + i * 128
				nv = raw[off]
				pts = [_st.unpack_from(">hhh", raw, off + 16 + v * 6) for v in range(nv)]
				items.append((pts, (48, 130, 150)))
				water_n += 1
		title = f"Lost Coast FULL geo + sky ({len(all_faces)} faces + {water_n} water)"
		sub = "VBSP brush+disp + sky_lostcoast_hdr + gw water sheets"
		nfaces = len(all_faces)

	eye, forward, right, up = look_basis(eye_origin, yaw, args.pitch)
	buf = np.zeros((args.h, args.w, 3), np.uint8)
	draw_sky(buf, sky, forward, right, up, args.fov)
	buf, drawn = paint(buf, items, eye, forward, right, up, args.fov)

	img = Image.fromarray(buf, "RGB")
	d = ImageDraw.Draw(img)
	d.rectangle((0, 0, args.w, 36), fill=(8, 10, 14))
	d.text((12, 10), title, fill=(235, 235, 230))
	d.rectangle((0, args.h - 32, args.w, args.h), fill=(8, 10, 14))
	d.text((12, args.h - 22), f"{sub}  drawn={drawn}", fill=(170, 180, 190))
	args.out.parent.mkdir(parents=True, exist_ok=True)
	img.save(args.out)
	print(f"wrote {args.out} faces={nfaces} drawn={drawn} mode={args.mode}")


if __name__ == "__main__":
	main()
