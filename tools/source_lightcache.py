#!/usr/bin/env python3
"""Source model lighting at a point, from the map's own data.

What the Source engine feeds a dynamic model (an NPC) standing at a point:

  * the ambient cube of the BSP leaf it stands in. This map is BSP v20 with
    v1 leaves and per-leaf cubes: lump 55 (HDR) / 56 (LDR) hold one
    CompressedLightCube (6 x ColorRGBExp32, +X -X +Y -Y +Z -Z) per leaf.
  * the light_environment sun (worldlight emit_skylight, lump 54 HDR / 15),
    added with N.L when a ray toward the sun reaches the sky.

Colours are linear light, the same units as the lightmaps.
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bake_worlddl import Bsp, lump  # noqa: E402

SURF_SKY2D = 0x2
SURF_SKY = 0x4
CONTENTS_SOLID = 0x1
EMIT_SKYLIGHT = 3


def _rgbe(b: bytes, o: int) -> tuple[float, float, float]:
	"""ColorRGBExp32ToVector: c * 2^e (lightmaps use TexLightToLinear, /255)."""
	r, g, bl, e = b[o], b[o + 1], b[o + 2], b[o + 3]
	e = e - 256 if e > 127 else e
	sc = 2.0 ** e
	return (r * sc, g * sc, bl * sc)


class LightCache:
	def __init__(self, bsp_path: Path):
		self.bsp = Bsp(bsp_path)
		d = self.bsp.data
		self.cubes = lump(d, 55) or lump(d, 56)
		self.sun = None
		for i in (54, 15):
			raw = lump(d, i)
			for k in range(len(raw) // 88):
				v = struct.unpack_from("<3f3f3fiiiffffffffiii", raw, k * 88)
				if v[10] == EMIT_SKYLIGHT:
					self.sun = (v[3:6], v[6:9])  # intensity, direction light travels
					break
			if self.sun:
				break
		b = self.bsp
		self.sky_faces = []
		for f in b.faces:
			if b.texinfos[f[5]][4] & (SURF_SKY | SURF_SKY2D):
				pts = b.face_pts(f)
				if not pts:
					continue
				pl = b.planes[f[0]]
				lo = tuple(min(p[a] for p in pts) - 2.0 for a in range(3))
				hi = tuple(max(p[a] for p in pts) + 2.0 for a in range(3))
				self.sky_faces.append((pl, lo, hi))

	def leaf_at(self, p) -> int:
		b = self.bsp
		idx = 0
		while idx >= 0:
			planenum, c0, c1 = b.nodes[idx]
			nx, ny, nz, dist = b.planes[planenum]
			idx = c0 if nx * p[0] + ny * p[1] + nz * p[2] - dist >= 0.0 else c1
		return -1 - idx

	def ambient_cube(self, p) -> list[tuple[float, float, float]]:
		leaf = self.leaf_at(p)
		o = leaf * 24
		if o + 24 > len(self.cubes):
			return [(0.0, 0.0, 0.0)] * 6
		return [_rgbe(self.cubes, o + 4 * i) for i in range(6)]

	def _trace(self, p0, p1):
		"""First point where the segment enters solid, or None."""
		b = self.bsp

		def rec(node, a, c):
			if node < 0:
				return a if b.leaves[-1 - node][0] & CONTENTS_SOLID else None
			planenum, c0, c1 = b.nodes[node]
			nx, ny, nz, dist = b.planes[planenum]
			d0 = nx * a[0] + ny * a[1] + nz * a[2] - dist
			d1 = nx * c[0] + ny * c[1] + nz * c[2] - dist
			if d0 >= 0 and d1 >= 0:
				return rec(c0, a, c)
			if d0 < 0 and d1 < 0:
				return rec(c1, a, c)
			t = d0 / (d0 - d1)
			m = tuple(a[i] + (c[i] - a[i]) * t for i in range(3))
			near, far = (c0, c1) if d0 >= 0 else (c1, c0)
			r = rec(near, a, m)
			return r if r is not None else rec(far, m, c)

		return rec(0, tuple(p0), tuple(p1))

	def sun_visible(self, p) -> bool:
		if not self.sun:
			return False
		d = self.sun[1]
		far = tuple(p[i] - d[i] * 65536.0 for i in range(3))
		hit = self._trace(p, far)
		if hit is None:
			return True  # left the world without hitting anything
		for (nx, ny, nz, dist), lo, hi in self.sky_faces:
			if abs(nx * hit[0] + ny * hit[1] + nz * hit[2] - dist) > 2.0:
				continue
			if all(lo[a] <= hit[a] <= hi[a] for a in range(3)):
				return True
		return False


def lighting_at(lc: LightCache, origin, height: float = 36.0):
	"""(cube[6], sun_rgb or None, sun_dir_toward_light) for a model at origin.

	Source lights a model at its illumination centre, roughly the middle of
	its bounding box; for a standing human that is ~36 units up.
	"""
	c = (origin[0], origin[1], origin[2] + height)
	cube = lc.ambient_cube(c)
	sun = None
	sdir = (0.0, 0.0, 1.0)
	if lc.sun:
		inten, d = lc.sun
		sdir = (-d[0], -d[1], -d[2])
		if lc.sun_visible(c):
			sun = tuple(inten)
	return cube, sun, sdir


if __name__ == "__main__":
	lc = LightCache(Path(sys.argv[1]))
	o = tuple(float(x) for x in sys.argv[2:5])
	cube, sun, sdir = lighting_at(lc, o)
	print("cube", [tuple(round(x, 3) for x in c) for c in cube])
	print("sun", sun, "dir", sdir)
