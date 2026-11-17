#!/usr/bin/env python3
"""Small copies of the map's env_cubemap samples for reflections on props.

A prop's $envmap resolves to the env_cubemap nearest to it. The engine turns
the cubemap nearest the camera into a 16 x 16 sphere map for the current view
each frame (GX can generate a sphere-map lookup from the vertex normal, not a
true reflection vector), so it needs the cube itself in a form it can sample
on the CPU: six 8 x 8 faces per cubemap, HDR, stored as sqrt( light / 2 ) in
8 bits per channel like the world's reflection views (bake_lostcoast_cubemaps).

  header 32 bytes : "GCcs", u16 version (1), u16 count, u16 side (8), zero padded
  origins         : count x 3 x s16, padded to 32
  faces           : count x 6 x side x side x 3 bytes (RGB), 1152 each
Face order +X -X +Y -Y +Z -Z. Texel (i, j) of a face looks along
  +X: ( 1, s, t )  -X: ( -1, s, t )  +Y: ( s, 1, t )  -Y: ( s, -1, t )
  +Z: ( s, t, 1 )  -Z: ( s, t, -1 ),  s = (i + .5) / 4 - 1, t = (j + .5) / 4 - 1.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import bake_lostcoast_cubemaps as bc  # noqa: E402

SIDE = 8


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--map", type=Path, default=ROOT / "content/lostcoast/maps/d2_lostcoast.bsp")
	ap.add_argument("--out", type=Path, default=ROOT / "cache/bake")
	args = ap.parse_args()
	stem = args.map.stem.lower()
	raw = args.map.read_bytes()
	pak = bc.read_pak(raw)
	names = {n.lower(): n for n in pak.namelist()}
	c = (np.arange(SIDE) + 0.5) / (SIDE / 2.0) - 1.0
	s, t = np.meshgrid(c, c)  # s along i (columns), t along j (rows)
	one = np.ones_like(s)
	dirs = [
		np.stack([one, s, t], -1), np.stack([-one, s, t], -1),
		np.stack([s, one, t], -1), np.stack([s, -one, t], -1),
		np.stack([s, t, one], -1), np.stack([s, t, -one], -1),
	]
	origins, blobs = [], []
	for (x, y, z) in bc.cubemap_origins(raw):
		key = f"materials/maps/{stem}/c{x}_{y}_{z}.vtf"
		if key not in names:
			continue
		hdr_key = key[:-4] + ".hdr.vtf"
		faces = bc.load_faces(pak.read(names[hdr_key if hdr_key in names else key]))
		if hdr_key not in names:
			faces = faces ** 2.2
		out = bytearray()
		for d in dirs:
			n = d / np.linalg.norm(d, axis=-1, keepdims=True)
			img = bc.sample_cube(faces, n)
			out += np.clip(np.sqrt(np.clip(img, 0.0, bc.CUBE_MAX) / bc.CUBE_MAX) * 255.0 + 0.5, 0, 255).astype(np.uint8).tobytes()
		origins.append((x, y, z))
		blobs.append(bytes(out))
	body = struct.pack(">4sHHH", b"GCcs", 1, len(origins), SIDE).ljust(32, b"\0")
	tab = b"".join(struct.pack(">3h", *[int(max(-32767, min(32767, v))) for v in o]) for o in origins)
	body += tab + b"\0" * (-len(tab) % 32)
	body += b"".join(blobs)
	path = args.out / "media" / f"cs_{stem}.bin"
	path.write_bytes(body)
	print(f"  wrote {path} cubemaps={len(origins)} bytes={len(body)}")


if __name__ == "__main__":
	main()
