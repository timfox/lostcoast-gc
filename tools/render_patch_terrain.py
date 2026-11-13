#!/usr/bin/env python3
"""Show the displacement patches in the immediate Lost Coast area.

Left: the dump-eye view, height-shaded, with the patch grid drawn on top.
Right: the same patches from above, one colour per patch, so the seams read.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from bake_lostcoast_map import face_verts, load_source_bsp  # noqa: E402
from source_disp import parse_dispinfos, tessellate_disp  # noqa: E402

EYE = np.array([1655.6, 2212.7, -48.0], np.float64)
YAW = 150.0
PITCH = 18.0
NEAR_R = 720.0


def basis(eye, yaw, pitch):
	yr, pr = math.radians(yaw), math.radians(pitch)
	forward = np.array(
		[math.cos(pr) * math.cos(yr), math.cos(pr) * math.sin(yr), math.sin(pr)],
		np.float64,
	)
	forward /= np.linalg.norm(forward)
	right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
	right /= np.linalg.norm(right)
	up = np.cross(right, forward)
	return np.asarray(eye, np.float64), forward, right, up


def collect(bsp, data):
	disps = parse_dispinfos(data)
	patches = []
	for disp in disps:
		fi = disp["map_face"]
		if fi < 0 or fi >= len(bsp["faces"]):
			continue
		face = bsp["faces"][fi]
		ti = face["texinfo"]
		if ti < 0 or ti >= len(bsp["texinfos"]):
			continue
		td = bsp["texinfos"][ti]["texdata"]
		if td < 0 or td >= len(bsp["tex_names"]):
			continue
		mat = bsp["tex_names"][td]
		if mat.startswith("tools/"):
			continue
		pts4 = face_verts(bsp, face)
		quads = tessellate_disp(data, disp, pts4, max_power=2)
		if not quads:
			continue
		near = []
		for q in quads:
			c = np.mean(q, axis=0)
			if np.linalg.norm(c - EYE) <= NEAR_R:
				near.append(q)
		if not near:
			continue
		center = np.mean([np.mean(q, axis=0) for q in near], axis=0)
		patches.append({"quads": near, "mat": mat, "power": disp["power"], "center": center})
	return patches


def height_rgb(z, z0, z1, patch_rgb):
	t = 0.0 if z1 <= z0 else (z - z0) / (z1 - z0)
	t = max(0.0, min(1.0, t))
	if t < 0.45:
		u = t / 0.45
		base = np.array([78, 68, 46]) * (1 - u) + np.array([176, 154, 112]) * u
	else:
		u = (t - 0.45) / 0.55
		base = np.array([176, 154, 112]) * (1 - u) + np.array([118, 116, 108]) * u
	return np.clip(base * 0.72 + patch_rgb * 0.28, 0, 255).astype(np.uint8)


def project(pts, eye, forward, right, up, fov, w, h):
	rel = pts - eye
	z = rel @ forward
	x = rel @ right
	y = rel @ up
	f = 1.0 / math.tan(math.radians(fov) * 0.5)
	aspect = w / h
	ok = z > 8.0
	sx = np.zeros(len(pts))
	sy = np.zeros(len(pts))
	sx[ok] = (x[ok] / z[ok] * f / aspect + 1.0) * 0.5 * w
	sy[ok] = (1.0 - y[ok] / z[ok] * f) * 0.5 * h
	return sx, sy, z, ok


def render_persp(patches, z0, z1, w, h):
	target = np.mean([p["center"] for p in patches], axis=0)
	delta = target - EYE
	yaw = math.degrees(math.atan2(delta[1], delta[0]))
	pitch = math.degrees(math.atan2(delta[2], math.hypot(delta[0], delta[1])))
	eye, forward, right, up = basis(EYE, yaw, pitch)
	img = Image.new("RGB", (w, h), (168, 186, 196))
	draw = ImageDraw.Draw(img)
	draw.rectangle((0, h // 2, w, h), fill=(42, 48, 44))
	order = []
	wires = []
	for i, patch in enumerate(patches):
		hue = np.array(
			[80 + (i * 47) % 140, 70 + (i * 29) % 120, 50 + (i * 71) % 100],
			np.float64,
		)
		for q in patch["quads"]:
			pts = np.array(q, np.float64)
			sx, sy, depth, ok = project(pts, eye, forward, right, up, 70.0, w, h)
			if ok.sum() < 4:
				continue
			col = tuple(int(c) for c in height_rgb(float(pts[:, 2].mean()), z0, z1, hue))
			poly = [(float(sx[k]), float(sy[k])) for k in range(4)]
			order.append((float(depth.mean()), poly, col))
			wires.append(poly)
	for _d, poly, col in sorted(order, key=lambda t: -t[0]):
		draw.polygon(poly, fill=col)
	for poly in wires:
		draw.line(poly + [poly[0]], fill=(42, 34, 26), width=1)
	return img


def render_top(patches, z0, z1, w, h):
	pts = np.array([p for patch in patches for q in patch["quads"] for p in q], np.float64)
	minx, miny = pts[:, 0].min(), pts[:, 1].min()
	span = max(pts[:, 0].max() - minx, pts[:, 1].max() - miny, 1.0)
	pad = span * 0.06

	def xy_to(x, y):
		u = (x - (minx - pad)) / (span + 2 * pad)
		v = (y - (miny - pad)) / (span + 2 * pad)
		return u * w, (1.0 - v) * h

	img = Image.new("RGB", (w, h), (24, 26, 28))
	draw = ImageDraw.Draw(img)
	order = []
	wires = []
	for i, patch in enumerate(patches):
		hue = np.array(
			[90 + (i * 53) % 150, 80 + (i * 37) % 140, 60 + (i * 19) % 120],
			np.float64,
		)
		for q in patch["quads"]:
			pts4 = np.array(q, np.float64)
			poly = [xy_to(p[0], p[1]) for p in pts4]
			col = tuple(int(c) for c in height_rgb(float(pts4[:, 2].mean()), z0, z1, hue))
			order.append((float(pts4[:, 2].mean()), poly, col))
			wires.append(poly)
	for _z, poly, col in sorted(order, key=lambda t: t[0]):
		draw.polygon(poly, fill=col)
	for poly in wires:
		draw.line(poly + [poly[0]], fill=(36, 30, 24), width=1)
	return img


def main():
	bsp_path = ROOT / "content/lostcoast/maps/d2_lostcoast.bsp"
	bsp = load_source_bsp(bsp_path)
	patches = collect(bsp, bsp_path.read_bytes())
	if not patches:
		raise SystemExit("no patches near the dump eye")
	zs = [p[2] for patch in patches for q in patch["quads"] for p in q]
	z0, z1 = min(zs), max(zs)
	mats = {}
	for p in patches:
		mats[p["mat"]] = mats.get(p["mat"], 0) + 1
	left = render_persp(patches, z0, z1, 640, 480)
	right = render_top(patches, z0, z1, 640, 480)
	sheet = Image.new("RGB", (1280, 520), (12, 12, 14))
	sheet.paste(left, (0, 28))
	sheet.paste(right, (640, 28))
	d = ImageDraw.Draw(sheet)
	d.text((12, 8), "Immediate Lost Coast ground  ·  dump eye, patch grid", fill=(230, 226, 210))
	d.text((652, 8), "Same patches from above  ·  colour marks each displacement", fill=(230, 226, 210))
	out = ROOT / "playtests/notes/patch-terrain-20261111/ground-mesh.png"
	out.parent.mkdir(parents=True, exist_ok=True)
	sheet.save(out)
	print(f"wrote {out}")
	print(f"patches={len(patches)} z={z0:.0f}..{z1:.0f}")
	for name, n in sorted(mats.items(), key=lambda kv: -kv[1])[:12]:
		print(f"  {n:3d}  {name}")


if __name__ == "__main__":
	main()
