#!/usr/bin/env python3
"""Bake the normal maps behind the env_cubemap reflection for GX.

LightmappedGeneric reflects the cubemap about the per-pixel normal of
$bumpmap. GX does that with an indirect texture: the normal's X/Y offset the
cubemap lookup. This packs the X/Y of the first MAX_NORMALS reflection masks
that are normal maps ($normalmapalphaenvmapmask), in mask index order
(cache/bake/envmask-names.json, written by the map bake).

Output media/en_<map>.bin:
  "GCen", u8 count, count x u8 mask index, pad to 32
  count x 64x64 GX IA8 (A = normal X, I = normal Y, 128 = 0)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import bake_tevmat  # noqa: E402
import vtf  # noqa: E402
from vpk import open_vpk_roots  # noqa: E402

MAX_NORMALS = 3  # 8 KB each, resident in the map arena


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--map", default="d2_lostcoast")
	ap.add_argument("--out", type=Path, default=ROOT / "cache" / "bake")
	args = ap.parse_args()
	masks = json.loads((args.out / "envmask-names.json").read_text())
	roots = sorted((ROOT / "content" / "lostcoast").glob("*_dir.vpk"))
	roots += sorted((ROOT / "content" / "hl2").glob("*_dir.vpk"))
	vpks = open_vpk_roots(roots)
	idx, blobs = [], []
	for i, m in enumerate(masks):
		if not m.get("normal") or len(idx) >= MAX_NORMALS:
			continue
		data = None
		for v in vpks:
			try:
				data = v.read("materials/" + m["tex"] + ".vtf")
			except Exception:
				data = None
			if data:
				break
		if not data:
			print(f"  envnormals: {m['tex']} missing")
			continue
		im = vtf.decode_vtf(data)
		blobs.append(bake_tevmat.normal_xy_ia8(bytes(im.rgba), im.width, im.height, 64))
		idx.append(i)
		print(f"  envnormals: [{i}] {m['tex']} {im.width}x{im.height}")
	head = b"GCen" + bytes([len(idx)]) + bytes(idx)
	head += b"\0" * (-len(head) % 32)
	path = args.out / "media" / f"en_{args.map}.bin"
	path.write_bytes(head + b"".join(blobs))
	print(f"  wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
	main()
