#!/usr/bin/env python3
"""Bake the map's env_cubemap samples for the GameCube.

Source lights reflective surfaces with cubemaps built into the BSP (pak lump,
materials/maps/<map>/cX_Y_Z.vtf; LUMP_CUBEMAPS lists the sample origins).
GX has no cubemaps. For an up-facing reflector (water, wet sand) the mirror
direction of a view ray through surface point P is

    R = ( P.x - eye.x, P.y - eye.y, eye.z - P.z )

and R.xy / R.z is a projective function of P, which GX texgen computes per
vertex (GX_TG_MTX3x4 from position). So each cubemap's upper hemisphere is
baked on the tangent plane z = 1: texel (a, b) in [-K, K]^2 holds the cubemap
along normalize( a, b, 1 ). That is the cubemap itself, resampled; nothing
is painted in.

Output media/cm_<map>.gcm:
  header  "GCCM", u16 version=1, u16 count, u16 side, u16 K*256, pad to 32
  count x 8 bytes: s16 origin xyz, u16 pad
  count x CMPR side x side (pad each to 32)
"""
from __future__ import annotations

import argparse
import io
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from vtf import _decode_dxt  # noqa: E402

SIDE = 128
K = 6.0  # tan of the largest angle from vertical kept (80.5 degrees)


def read_pak(bsp: bytes) -> zipfile.ZipFile:
	o, n = struct.unpack_from("<II", bsp, 8 + 40 * 16)
	return zipfile.ZipFile(io.BytesIO(bsp[o:o + n]))


def cubemap_origins(bsp: bytes):
	o, n = struct.unpack_from("<II", bsp, 8 + 42 * 16)
	return [struct.unpack_from("<iii", bsp, o + i)[:3] for i in range(0, n, 16)]


def load_faces(vtf: bytes) -> np.ndarray:
	"""Six faces, float RGB 0..1 (LDR) or linear HDR, shape (6, s, s, 3)."""
	w, h = struct.unpack_from("<HH", vtf, 16)
	fmt = struct.unpack_from("<i", vtf, 52)[0]
	nfaces = 7  # v7.2 with a first frame: six faces and a sphere map
	if fmt == 13:  # DXT1
		per = max(1, w // 4) * max(1, h // 4) * 8
		off = len(vtf) - nfaces * per
		out = []
		for f in range(6):
			rgba = _decode_dxt(vtf[off + f * per: off + (f + 1) * per], w, h, "DXT1")
			out.append(np.frombuffer(bytes(rgba), np.uint8).reshape(h, w, 4)[:, :, :3] / 255.0)
		return np.stack(out)
	if fmt == 24:  # RGBA16161616F
		per = w * h * 8
		off = len(vtf) - nfaces * per
		a = np.frombuffer(vtf[off: off + 6 * per], np.float16).astype(np.float32)
		return a.reshape(6, h, w, 4)[:, :, :, :3]
	raise ValueError(f"cubemap format {fmt}")


def sample_cube(faces: np.ndarray, d: np.ndarray) -> np.ndarray:
	"""D3D texCUBE with a Source-space direction, bilinear inside a face."""
	x, y, z = d[..., 0], d[..., 1], d[..., 2]
	ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
	s = faces.shape[1]
	out = np.zeros(d.shape[:-1] + (3,), np.float32)
	sel = [
		((ax >= ay) & (ax >= az) & (x > 0), 0, -z, -y, ax),
		((ax >= ay) & (ax >= az) & (x <= 0), 1, z, -y, ax),
		((ay > ax) & (ay >= az) & (y > 0), 2, x, z, ay),
		((ay > ax) & (ay >= az) & (y <= 0), 3, x, -z, ay),
		((az > ax) & (az > ay) & (z > 0), 4, x, -y, az),
		((az > ax) & (az > ay) & (z <= 0), 5, -x, -y, az),
	]
	for mask, f, u, v, m in sel:
		if not mask.any():
			continue
		uu = (u[mask] / m[mask] * 0.5 + 0.5) * s - 0.5
		vv = (v[mask] / m[mask] * 0.5 + 0.5) * s - 0.5
		uu = np.clip(uu, 0, s - 1)
		vv = np.clip(vv, 0, s - 1)
		u0 = np.floor(uu).astype(int)
		v0 = np.floor(vv).astype(int)
		u1 = np.minimum(u0 + 1, s - 1)
		v1 = np.minimum(v0 + 1, s - 1)
		fu = (uu - u0)[:, None]
		fv = (vv - v0)[:, None]
		img = faces[f]
		out[mask] = (img[v0, u0] * (1 - fu) + img[v0, u1] * fu) * (1 - fv) \
			+ (img[v1, u0] * (1 - fu) + img[v1, u1] * fu) * fv
	return out


def up_plane(faces: np.ndarray, side: int = SIDE, k: float = K) -> np.ndarray:
	c = (np.arange(side) + 0.5) / side * 2.0 - 1.0
	a, b = np.meshgrid(c * k, c * k)  # a along +x (u), b along +y (v)
	d = np.stack([a, b, np.ones_like(a)], -1)
	d /= np.linalg.norm(d, axis=-1, keepdims=True)
	return sample_cube(faces, d)


def cubemap_index(bsp: bytes, mapname: str) -> dict[str, int]:
	"""'maps/<map>/cX_Y_Z' -> index in the pack (same order as main())."""
	names = {n.lower() for n in read_pak(bsp).namelist()}
	out, i = {}, 0
	for (x, y, z) in cubemap_origins(bsp):
		if f"materials/maps/{mapname}/c{x}_{y}_{z}.vtf" in names:
			out[f"maps/{mapname}/c{x}_{y}_{z}"] = i
			i += 1
	return out


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--map", type=Path, required=True)
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--sheet", type=Path)
	args = ap.parse_args()
	bsp = args.map.read_bytes()
	pak = read_pak(bsp)
	names = {n.lower(): n for n in pak.namelist()}
	mapname = args.map.stem
	from bake_lostcoast_map import gx_cmpr_encode_blocks
	recs, blobs, sheet = [], [], []
	for (x, y, z) in cubemap_origins(bsp):
		key = f"materials/maps/{mapname}/c{x}_{y}_{z}.vtf"
		if key not in names:
			continue
		img = up_plane(load_faces(pak.read(names[key])))
		# LightmappedGeneric: spec = ( cube * mask * tint )^2 in linear light
		# ($envmapcontrast 1). In display gamma that is cube_gamma^2 times
		# ( mask * tint )^( 2 / 2.2 ), so the texture holds the squared value
		# and the engine multiplies by the tint and the mask.
		rgb = np.clip(img * img * 255.0 + 0.5, 0, 255).astype(np.uint8)
		cm = gx_cmpr_encode_blocks(rgb, np.ones(rgb.shape[:2], bool))
		cm += b"\0" * (-len(cm) % 32)
		recs.append(struct.pack(">hhhH", x, y, z, 0))
		blobs.append(cm)
		sheet.append(rgb)
	hdr = struct.pack(">4sHHHH", b"GCCM", 2, len(recs), SIDE, int(K * 256))
	hdr += b"\0" * (32 - len(hdr))
	tab = b"".join(recs)
	tab += b"\0" * (-len(tab) % 32)
	args.out.mkdir(parents=True, exist_ok=True)
	path = args.out / f"cm_{mapname}.gcm"
	path.write_bytes(hdr + tab + b"".join(blobs))
	print(f"  wrote {path} ({path.stat().st_size} bytes, {len(recs)} cubemaps)")
	if args.sheet:
		from PIL import Image
		n = min(len(sheet), 24)
		im = Image.new("RGB", (6 * (SIDE + 2), ((n + 5) // 6) * (SIDE + 2)))
		for i in range(n):
			im.paste(Image.fromarray(sheet[i][::-1]), ((i % 6) * (SIDE + 2), (i // 6) * (SIDE + 2)))
		im.save(args.sheet, quality=90)


if __name__ == "__main__":
	main()
