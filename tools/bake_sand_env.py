#!/usr/bin/env python3
"""Bake Lost Coast sand $bumpmap → media/sand_nrm.cmp + sand_spec.cmp + sand_sun.cmp."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from bake_lostcoast_map import gx_cmpr_encode_blocks  # noqa: E402
from vpk import open_vpk_roots, read_first  # noqa: E402
from vtf import decode_vtf  # noqa: E402

REL = "materials/lostcoast/nature/sandfloor007c_normal.vtf"
SIDE = 64  # 64² CMPR = 2 KiB tip-safe each
SUN_SIDE = 32  # 32² CMPR = 512 bytes


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--lostcoast", type=Path, required=True)
	ap.add_argument("--out", type=Path, required=True)
	args = ap.parse_args()
	vpks = open_vpk_roots(sorted(args.lostcoast.glob("*_dir.vpk")))
	raw = read_first(vpks, REL)
	if not raw:
		raise SystemExit(f"missing {REL}")
	img = decode_vtf(raw)
	rgba = np.frombuffer(img.rgba, dtype=np.uint8).reshape(img.height, img.width, 4)
	im = np.array(Image.fromarray(rgba, "RGBA").resize((SIDE, SIDE), Image.LANCZOS))
	opaque = np.ones((SIDE, SIDE), dtype=bool)
	nrm = gx_cmpr_encode_blocks(im[:, :, :3].astype(np.float32), opaque)

	# High-contrast mask from $normalmapalphaenvmapmask (sparse wet glints).
	a = im[:, :, 3].astype(np.float32)
	thr = float(np.percentile(a, 70))
	stretched = np.clip((a - thr) * (255.0 / max(255.0 - thr, 1.0)), 0, 255)
	stretched = 255.0 * (stretched / 255.0) ** 0.65
	spec = gx_cmpr_encode_blocks(np.stack([stretched, stretched, stretched], -1), opaque)

	yy, xx = np.mgrid[0:SUN_SIDE, 0:SUN_SIDE]
	cx = cy = (SUN_SIDE - 1) / 2.0
	r2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (cx * cx)
	sun = np.clip(1.0 - r2, 0, 1.0) ** 3
	core = np.clip(1.0 - r2 * 4.0, 0, 1.0) ** 2
	sun = np.clip(sun * 255.0 + core * 255.0 * 0.6, 0, 255).astype(np.float32)
	sun_rgb = np.stack([sun, sun * 0.95, sun * 0.85], -1)
	sun_body = gx_cmpr_encode_blocks(sun_rgb, np.ones((SUN_SIDE, SUN_SIDE), dtype=bool))

	args.out.mkdir(parents=True, exist_ok=True)
	(args.out / "sand_nrm.cmp").write_bytes(nrm)
	(args.out / "sand_spec.cmp").write_bytes(spec)
	(args.out / "sand_sun.cmp").write_bytes(sun_body)
	print(
		f"  sand_nrm/spec ({SIDE}², {len(nrm)} B) + sand_sun ({SUN_SIDE}², {len(sun_body)} B)"
	)


if __name__ == "__main__":
	main()
