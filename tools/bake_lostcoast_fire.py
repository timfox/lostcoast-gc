#!/usr/bin/env python3
"""Bake Source env_fire flames for the GX renderer: media/fi_<map>.gcf.

Every number comes from the game data:
  * env_fire entities (BSP entity lump) give the positions. C_FireSmoke picks
    the particle system by floor(firesize / 36): 3 and up is env_fire_large,
    which is what all six fires of d2_lostcoast are.
  * particles/fire_01.pcf gives the two flame layers (env_fire_large and its
    child env_fire_large_b): emission rate, radius, alpha, colour, gravity,
    rise speed, spawn disc, sequences and playback rate, fades.
  * The flame frames are the sprite sheet of the system's material
    (particle/fire_burning_character/fire_env_fire.vmt), cut by the VTF's
    sheet resource.

In the map these fires start off: the headcrab canister templates turn them
on five seconds after a canister lands, together with the smoke columns the
port already draws. They are baked for that same bombarded state.

Memory compromise (not a look choice): every second frame of each sequence
is kept, in 32x32 cells of one 256x256 CMPR atlas (32 KB). The texture is
premultiplied by its alpha and drawn additively ($addself).

Not baked yet: the smoke layer (env_fire_large_smoke, vistasmokev1), the
embers child, horizontal velocity noise and per-particle rotation.

  header 16 bytes : "GCfi", u16 version (1), u16 fires, u16 layers,
                    u16 sequences, u16 atlas side, u16 cell
  layer  40 bytes : f32 rate, f32 gravity_z, f32 tile_fps,
                    u8 radius_min, radius_max, alpha_min, alpha_max,
                    u8 colour_min, colour_max, seq_first, seq_count,
                    u8 spread, max_particles, fade_out (1/100 s), z_offset (% of radius),
                    f32 speed_z_min, f32 speed_z_max,
                    f32 colour_fade_start, f32 colour_fade_end (fractions of life; 0,0 = none)
  seq     4 bytes : u8 first tile, u8 tiles, u16 life (1/100 s)
  fire   12 bytes : f32 origin[3]
  padding to 32, then the CMPR atlas.
"""
from __future__ import annotations

import argparse
import glob
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import source_pcf  # noqa: E402
from bake_lostcoast_fx import entities  # noqa: E402
from bake_lostcoast_map import gx_cmpr_encode_blocks  # noqa: E402
from vmt import load_vmt  # noqa: E402
from vpk import open_vpk_roots, read_first  # noqa: E402
from vtf import decode_vtf  # noqa: E402

ATLAS = 256
CELL = 32
FRAME_STEP = 2


def sheet(vtf: bytes) -> dict[int, list[tuple[float, float, float, float]]]:
	"""VTF sheet resource (tag 0x10): {sequence: [(u0, v0, u1, v1), ...]}."""
	nres = struct.unpack_from("<I", vtf, 68)[0]
	for i in range(nres):
		tag = vtf[80 + i * 8:83 + i * 8]
		off = struct.unpack_from("<I", vtf, 84 + i * 8)[0]
		if tag != b"\x10\x00\x00":
			continue
		p = off + 4
		ver, nseq = struct.unpack_from("<II", vtf, p)
		p += 8
		out = {}
		for _ in range(nseq):
			seq, _clamp, nfr, _total = struct.unpack_from("<IIIf", vtf, p)
			p += 16
			frames = []
			for _f in range(nfr):
				p += 4
				n = 16 if ver == 1 else 4
				frames.append(struct.unpack_from("<4f", vtf, p))
				p += 4 * n
			out[seq] = frames
		return out
	raise ValueError("no sheet resource in the fire texture")


def op(system: dict, group: str, name: str) -> dict:
	for o in system.get(group, []):
		if o.get("functionName") == name:
			return o
	return {}


def cell_from(img: np.ndarray, rect) -> np.ndarray:
	h, w = img.shape[:2]
	x0, y0, x1, y1 = (int(round(rect[0] * w)), int(round(rect[1] * h)), int(round(rect[2] * w)), int(round(rect[3] * h)))
	src = img[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)].astype(np.float64)
	pm = src[..., :3] * (src[..., 3:4] / 255.0)  # premultiplied: drawn additively
	side = CELL - 2  # one black texel all round, so bilinear never reaches a neighbour
	ys = np.linspace(0, pm.shape[0], side + 1).astype(int)
	xs = np.linspace(0, pm.shape[1], side + 1).astype(int)
	out = np.zeros((CELL, CELL, 3))
	for j in range(side):
		for i in range(side):
			blk = pm[ys[j]:max(ys[j + 1], ys[j] + 1), xs[i]:max(xs[i + 1], xs[i] + 1)]
			out[j + 1, i + 1] = blk.reshape(-1, 3).mean(axis=0)
	return out


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--map", type=Path, default=ROOT / "content/lostcoast/maps/d2_lostcoast.bsp")
	ap.add_argument("--out", type=Path, default=ROOT / "cache/bake")
	args = ap.parse_args()
	vpks = open_vpk_roots([Path(p) for p in sorted(glob.glob(str(ROOT / "content/*/*_dir.vpk")))])

	fires = []
	for kv in entities(args.map.read_bytes()):
		if kv.get("classname") != "env_fire":
			continue
		size = float(kv.get("firesize", "64") or 64)
		if int(size // 36) < 3:
			print(f"  env_fire {kv.get('targetname', '-')}: firesize {size} is not env_fire_large, skipped")
			continue
		fires.append(tuple(float(v) for v in kv["origin"].split()))
	systems = source_pcf.systems(read_first(vpks, "particles/fire_01.pcf"))
	main_sys = systems["env_fire_large"]
	layers_src = [main_sys] + [c["child"] for c in main_sys.get("children", []) if c["child"]["_name"] == "env_fire_large_b"]
	mat = main_sys["material"].replace("\\", "/").lower()
	vm = load_vmt(read_first(vpks, "materials/" + mat))
	vtf_raw = read_first(vpks, "materials/" + vm["$basetexture"].replace("\\", "/").lower() + ".vtf")
	frames = sheet(vtf_raw)
	img = decode_vtf(vtf_raw, max_side=2048)
	arr = np.frombuffer(img.rgba, np.uint8).reshape(img.height, img.width, 4)

	atlas = np.zeros((ATLAS, ATLAS, 3))
	per_row = ATLAS // CELL
	tiles = 0
	seq_recs = {}
	layer_recs = []
	for sysdef in layers_src:
		life_fps = float(op(sysdef, "initializers", "lifetime from sequence")["Frames Per Second"])
		sr = op(sysdef, "initializers", "Sequence Random")
		s0, s1 = int(sr["sequence_min"]), int(sr["sequence_max"])
		for s in range(s0, s1 + 1):
			if s in seq_recs:
				continue
			keep = frames[s][::FRAME_STEP]
			first = tiles
			for rect in keep:
				if tiles >= per_row * per_row:
					raise SystemExit("fire atlas full")
				ty, tx = divmod(tiles, per_row)
				atlas[ty * CELL:(ty + 1) * CELL, tx * CELL:(tx + 1) * CELL] = cell_from(arr, rect)
				tiles += 1
			seq_recs[s] = (first, len(keep), len(frames[s]) / life_fps)
		rad = op(sysdef, "initializers", "Radius Random")
		alp = op(sysdef, "initializers", "Alpha Random")
		col = op(sysdef, "initializers", "Color Random")
		pos = op(sysdef, "initializers", "Position Within Sphere Random")
		off = op(sysdef, "initializers", "Position Modify Offset Random")
		noise = op(sysdef, "initializers", "Velocity Noise")
		mov = op(sysdef, "operators", "Movement Basic")
		fade = op(sysdef, "operators", "Alpha Fade Out Random")
		cfade = op(sysdef, "operators", "Color Fade")
		emit = op(sysdef, "emitters", "emit_continuously")
		lives = [seq_recs[s][2] for s in range(s0, s1 + 1)]
		fade_s = 0.5 * (fade.get("fade out time min", 0.0) + fade.get("fade out time max", 0.0))
		if fade.get("proportional 0/1"):
			fade_s *= sum(lives) / len(lives)
		vz0 = pos["speed_in_local_coordinate_system_min"][2] + (noise["output minimum"][2] if noise else 0.0)
		vz1 = pos["speed_in_local_coordinate_system_max"][2] + (noise["output maximum"][2] if noise else 0.0)
		c1, c2 = col["color1"][0], col["color2"][0]
		layer_recs.append((sysdef["_name"], struct.pack(
			">3f12B4f",
			float(emit["emission_rate"]), float(mov["gravity"][2]), life_fps / FRAME_STEP,
			int(rad["radius_min"]), int(rad["radius_max"]), int(alp["alpha_min"]), int(alp["alpha_max"]),
			min(c1, c2), max(c1, c2), s0, s1 - s0 + 1,
			int(pos["distance_max"]), int(sysdef["max_particles"]),
			int(round(fade_s * 100)), int(round(off["offset min"][2] * 100)) if off.get("offset proportional to radius 0/1") else 0,
			vz0, vz1,
			float(cfade.get("fade_start_time", 0.0)) if cfade else 0.0,
			float(cfade.get("fade_end_time", 0.0)) if cfade else 0.0,
		)))
		print(f"  layer {sysdef['_name']}: rate {emit['emission_rate']} radius {rad['radius_min']}-{rad['radius_max']} "
			f"gravity {mov['gravity'][2]} rise {vz0}-{vz1} seq {s0}-{s1} life {min(lives):.2f}-{max(lives):.2f}s")
	nseq = max(seq_recs) + 1
	seq_blob = b"".join(struct.pack(">BBH", seq_recs[s][0], seq_recs[s][1], int(round(seq_recs[s][2] * 100)))
		if s in seq_recs else b"\0\0\0\0" for s in range(nseq))
	body = struct.pack(">4s6H", b"GCfi", 1, len(fires), len(layer_recs), nseq, ATLAS, CELL)
	body += b"".join(r[1] for r in layer_recs) + seq_blob + b"".join(struct.pack(">3f", *f) for f in fires)
	body += b"\0" * (-len(body) % 32)
	rgb = np.clip(atlas + 0.5, 0, 255).astype(np.uint8)
	if __import__('os').environ.get('LCGC_FIRE_PREVIEW'):
		from PIL import Image
		Image.fromarray(rgb, 'RGB').save(__import__('os').environ['LCGC_FIRE_PREVIEW'])
	body += gx_cmpr_encode_blocks(rgb, np.ones((ATLAS, ATLAS), bool))
	out = args.out / "media" / f"fi_{args.map.stem.lower()}.gcf"
	out.write_bytes(body)
	print(f"  wrote {out} fires={len(fires)} layers={len(layer_recs)} tiles={tiles} bytes={len(body)}")


if __name__ == "__main__":
	main()
