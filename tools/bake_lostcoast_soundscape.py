#!/usr/bin/env python3
"""Bake the map's soundscapes for the GameCube.

Source drives ambience from scripts/soundscapes_<game>.txt and the map's
env_soundscape entities. Every Lost Coast soundscape is a set of "playrandom"
rules: every `time` seconds pick one wave from the rule's list and play it
whole at a random `volume` and `pitch`, either unpositioned (ambient), at a
random spot on a 36-unit ring around the listener ("position" "random") or
at one of the entity's position targets. "playsoundscape" nests another
soundscape with a volume multiplier. This tool flattens that per entity and
packs the waves, so the engine plays the same rules the PC client does
(game/client/c_soundscape.cpp).

media/ss_<map>.gcs, big-endian:
  header 32  "GCSS", u16 version=1, n_ent, n_rule, n_wave, n_ref, rate,
             u32 data_ofs, u32 data_bytes
  ents  16   s16 origin xyz, u16 radius, u16 first_rule, u16 n_rules, pad
  rules 24   u16 time_min, time_max (1/100 s), u8 vol_min, vol_max (/255),
             u8 pitch_min, pitch_max, u8 mode (0 ambient, 1 random ring,
             2 fixed), u8 soundlevel, s16 pos xyz, u16 first_ref, u8 n_ref
  refs  2    u16 wave index (table padded to 32)
  waves 12   u32 data offset, u32 samples, pad
  data       IMA ADPCM, 4 bits a sample, high nibble first, mono, each wave
             starts from predictor 0 / step index 0 and is padded to 32
"""
from __future__ import annotations

import argparse
import io
import re
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from vpk import open_vpk_roots, read_first  # noqa: E402

RATE = 22050
STEP = [7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60,
	66, 73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371,
	408, 449, 494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878,
	2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845,
	8630, 9493, 10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086,
	29794, 32767]
INDEX = [-1, -1, -1, -1, 2, 4, 6, 8]


def ima_encode(pcm: np.ndarray) -> bytes:
	pred, idx = 0, 0
	out = bytearray((len(pcm) + 1) // 2)
	for i, s in enumerate(pcm.tolist()):
		step = STEP[idx]
		diff = s - pred
		code = 0
		if diff < 0:
			code = 8
			diff = -diff
		d = step >> 3
		if diff >= step:
			code |= 4; diff -= step; d += step
		if diff >= step >> 1:
			code |= 2; diff -= step >> 1; d += step >> 1
		if diff >= step >> 2:
			code |= 1; d += step >> 2
		pred = max(-32768, min(32767, pred - d if code & 8 else pred + d))
		idx = max(0, min(88, idx + INDEX[code & 7]))
		out[i >> 1] |= code << (0 if i & 1 else 4)
	return bytes(out)


def parse_kv(text: str):
	"""Valve KeyValues -> list of (key, value|list), keeping duplicates."""
	text = re.sub(r"//[^\n]*", "", text)
	tok = re.findall(r'"([^"]*)"|([{}])', text)
	tok = [a if b == "" else b for a, b in tok]
	pos = 0

	def block():
		nonlocal pos
		items = []
		while pos < len(tok) and tok[pos] != "}":
			key = tok[pos]; pos += 1
			if pos < len(tok) and tok[pos] == "{":
				pos += 1
				items.append((key.lower(), block()))
				pos += 1
			else:
				items.append((key.lower(), tok[pos])); pos += 1
		return items
	return block()


def interval(s: str):
	p = [float(x) for x in s.replace(" ", "").split(",") if x]
	return (p[0], p[1] if len(p) > 1 else p[0])


SNDLVL = {"sndlvl_none": 0, "sndlvl_norm": 75, "sndlvl_80db": 80, "sndlvl_140db": 140}


def flatten(scapes, name, master, pos_override, positions, out, depth=0):
	body = scapes.get(name.lower())
	if body is None or depth > 8:
		return
	for key, val in body:
		if key == "playsoundscape":
			kv = dict((k, v) for k, v in val if not isinstance(v, list))
			vol = interval(kv.get("volume", "1"))[0]
			po = int(kv["positionoverride"]) if "positionoverride" in kv else pos_override
			flatten(scapes, kv.get("name", ""), master * vol, po, positions, out, depth + 1)
		elif key == "playrandom":
			kv = dict((k, v) for k, v in val if not isinstance(v, list))
			waves = [w for k, v in val if k == "rndwave" for kk, w in v if kk == "wave"]
			if not waves:
				continue
			vol = interval(kv.get("volume", "1"))
			mode, pos = 0, (0, 0, 0)
			p = kv.get("position")
			idx = -1
			if p is not None and p.lower() == "random":
				mode = 1
			elif p is not None:
				idx = int(p)
			if pos_override >= 0:  # override trumps random position
				idx, mode = pos_override, 0
			if idx >= 0:
				if idx not in positions:
					continue  # c_soundscape.cpp: suppress if the position is missing
				mode, pos = 2, positions[idx]
			lvl = kv.get("soundlevel", "sndlvl_norm").lower()
			out.append({
				"time": interval(kv.get("time", "10")),
				"vol": (vol[0] * master, vol[1] * master),
				"pitch": interval(kv.get("pitch", "100")),
				"mode": mode, "pos": pos, "lvl": SNDLVL.get(lvl, 75), "waves": waves,
			})


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--map", type=Path, required=True)
	ap.add_argument("--lostcoast", type=Path, required=True)
	ap.add_argument("--hl2", type=Path, required=True)
	ap.add_argument("--out", type=Path, required=True)
	args = ap.parse_args()
	roots = open_vpk_roots(sorted(args.lostcoast.glob("*_dir.vpk")) + sorted(args.hl2.glob("*_dir.vpk")))
	scapes = {}
	for rel in ("scripts/soundscapes_lostcoast.txt",):
		raw = read_first(roots, rel)
		if raw:
			for k, v in parse_kv(raw.decode("latin-1")):
				if isinstance(v, list):
					scapes[k] = v
	bsp = args.map.read_bytes()
	o, n = struct.unpack_from("<II", bsp, 8)
	ents = [dict(re.findall(r'"([^"]*)"\s+"([^"]*)"', b))
		for b in re.findall(r"\{[^{}]*\}", bsp[o:o + n].decode("latin-1"))]
	named = {e["targetname"]: tuple(float(x) for x in e["origin"].split())
		for e in ents if "targetname" in e and "origin" in e}
	ent_recs, rules = [], []
	for e in ents:
		if e.get("classname") != "env_soundscape" or e.get("StartDisabled", "0") != "0":
			continue
		positions = {i: named[e[f"position{i}"]] for i in range(8)
			if e.get(f"position{i}") in named}
		flat: list = []
		flatten(scapes, e.get("soundscape", ""), 1.0, -1, positions, flat)
		org = tuple(float(x) for x in e["origin"].split())
		ent_recs.append((org, float(e.get("radius", "0")), len(rules), len(flat)))
		rules += flat
		print(f"  env_soundscape {e.get('soundscape')} at {org} r={e.get('radius')}: {len(flat)} rules")
	wave_ix: dict[str, int] = {}
	refs: list[int] = []
	for r in rules:
		r["first_ref"] = len(refs)
		for w in r["waves"]:
			refs.append(wave_ix.setdefault(w.lstrip("*#@><^)(}$!?"), len(wave_ix)))
	data = bytearray()
	wave_recs = []
	with tempfile.TemporaryDirectory() as td:
		for name in wave_ix:
			raw = read_first(roots, "sound/" + name)
			if not raw:
				print(f"  missing wave {name}")
				wave_recs.append((len(data), 0))
				continue
			src, dst = Path(td) / "in.wav", Path(td) / "out.raw"
			src.write_bytes(raw)
			subprocess.run(["sox", str(src), "-r", str(RATE), "-c", "1", "-b", "16", "-e", "signed",
				"-t", "raw", str(dst)], check=True, capture_output=True)
			pcm = np.frombuffer(dst.read_bytes(), "<i2")
			wave_recs.append((len(data), len(pcm)))
			data += ima_encode(pcm)
			data += b"\0" * (-len(data) % 32)
	ent_b = b"".join(struct.pack(">hhhHHHI", int(o[0]), int(o[1]), int(o[2]), int(min(65535, rad)), fr, nr, 0)
		for o, rad, fr, nr in ent_recs)
	rule_b = b"".join(struct.pack(">HHBBBBBBhhhHBxxxxx",
		int(min(65535, r["time"][0] * 100)), int(min(65535, r["time"][1] * 100)),
		int(max(0, min(255, round(r["vol"][0] * 255)))), int(max(0, min(255, round(r["vol"][1] * 255)))),
		int(min(255, r["pitch"][0])), int(min(255, r["pitch"][1])), r["mode"], min(255, r["lvl"]),
		int(r["pos"][0]), int(r["pos"][1]), int(r["pos"][2]), r["first_ref"], len(r["waves"]))
		for r in rules)
	ref_b = b"".join(struct.pack(">H", x) for x in refs)
	ref_b += b"\0" * (-len(ref_b) % 32)
	wave_b = b"".join(struct.pack(">III", ofs, ns, 0) for ofs, ns in wave_recs)
	wave_b += b"\0" * (-len(wave_b) % 32)
	head_len = 32 + len(ent_b) + len(rule_b) + len(ref_b) + len(wave_b)
	pad = b"\0" * (-head_len % 32)
	hdr = struct.pack(">4sHHHHHHII", b"GCSS", 1, len(ent_recs), len(rules), len(wave_recs), len(refs),
		RATE, head_len + len(pad), len(data))
	hdr += b"\0" * (32 - len(hdr))
	args.out.mkdir(parents=True, exist_ok=True)
	path = args.out / f"ss_{args.map.stem}.gcs"
	path.write_bytes(hdr + ent_b + rule_b + ref_b + wave_b + pad + bytes(data))
	secs = sum(ns for _o, ns in wave_recs) / RATE
	print(f"  wrote {path} ({path.stat().st_size} bytes): {len(ent_recs)} entities, {len(rules)} rules, "
		f"{len(wave_recs)} waves, {secs:.0f} s at {RATE} Hz, tables {head_len} bytes")


if __name__ == "__main__":
	main()
