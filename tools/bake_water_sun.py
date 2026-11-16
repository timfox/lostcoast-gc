#!/usr/bin/env python3
"""Bake the part of the water reflection an 8-bit mirror cannot hold.

Retail water_ati.vmt reflects the scene from an HDR render target, so the sun
in the HDR sky comes back off the water brighter than white. The GX mirror
pass is RGB565: everything above 1.0 is clipped, and the hotspot is lost.

The water material is patched by VBSP with env_cubemap samples
(maps/<map>/.../water_ati_X_Y_Z -> cX_Y_Z.hdr.vtf, linear float). This bakes
max( 0, cubemap - 1 ) looking up from the water, i.e. exactly the light the
mirror clipped, as media/ws_<map>.gcx. The engine adds it over the mirror
reflection with the same Fresnel weight and the same ripple offsets.

Of the water's cubemaps, the one with the most light above 1.0 is used (the
sun is at infinity; the others see it partly behind the cliffs).

  header 32 bytes: "GCws", u16 version (1), u16 side, u16 k * 256, zero padded
  CMPR side x side. Lookup: uv = 0.5 + R.xy / ( 2 k R.z ), R the mirrored ray.
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
from bake_lostcoast_map import gx_cmpr_encode_blocks  # noqa: E402
from bake_worlddl import Bsp  # noqa: E402

SIDE = 64
K = 8.0  # tan of the largest angle from straight up: rays down to 7 degrees above the horizon


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--map", type=Path, default=ROOT / "content/lostcoast/maps/d2_lostcoast.bsp")
	ap.add_argument("--out", type=Path, default=ROOT / "cache/bake")
	args = ap.parse_args()
	stem = args.map.stem.lower()
	raw = args.map.read_bytes()
	pak = bc.read_pak(raw)
	names = {n.lower(): n for n in pak.namelist()}
	best = None
	for tex in Bsp(args.map).tex_names:
		base = tex.rsplit("/", 1)[-1]
		if not tex.startswith(f"maps/{stem}/") or not base.startswith("water_ati_"):
			continue
		key = f"materials/maps/{stem}/c{base[len('water_ati_'):]}.hdr.vtf"
		if key not in names:
			continue
		faces = bc.load_faces(pak.read(names[key]))
		plane = bc.axis_plane(faces, 0, SIDE, K)
		excess = np.clip(plane - 1.0, 0.0, 1.0)
		score = float(excess.sum())
		if best is None or score > best[0]:
			best = (score, key, excess, float(plane.max()))
	if best is None:
		raise SystemExit("no HDR cubemap for the water material")
	_score, key, excess, peak = best
	rgb = np.clip(excess ** (1.0 / 2.2) * 255.0 + 0.5, 0, 255).astype(np.uint8)
	body = struct.pack(">4sHHH", b"GCws", 1, SIDE, int(K * 256)).ljust(32, b"\0")
	body += gx_cmpr_encode_blocks(rgb, np.ones((SIDE, SIDE), bool))
	out = args.out / "media" / f"ws_{stem}.gcx"
	out.write_bytes(body)
	lit = int((excess.max(-1) > 0).sum())
	print(f"  wrote {out} from {key}: HDR peak {peak:.2f}, {lit} of {SIDE * SIDE} texels above 1.0")


if __name__ == "__main__":
	main()
