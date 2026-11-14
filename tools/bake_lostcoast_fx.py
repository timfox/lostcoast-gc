#!/usr/bin/env python3
"""Bake Source particle emitters the GX renderer can draw as billboards.

env_smokestack → media/fx_<map>.gcf. The town's black smoke columns
(smoke_canister1-4) are the ones the retail overview shot shows.

  header  8 bytes : "GCfx", u16 version (1), u16 count
  record 32 bytes : f32 origin[3], u8 rgba[4], f32 jet_length,
                    u16 start_size, u16 end_size, u16 speed,
                    u16 spread_speed, u16 base_spread, u8 rate, u8 twist
"""
from __future__ import annotations

import argparse
import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_STACKS = 16


def entities(bsp: bytes):
	off, ln = struct.unpack_from("<ii", bsp, 8)  # lump 0
	text = bsp[off:off + ln].decode("latin-1", "replace")
	for block in re.findall(r"\{[^{}]*\}", text):
		yield dict(re.findall(r'"([^"]*)"\s*"([^"]*)"', block))


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--map", type=Path, default=ROOT / "content/lostcoast/maps/d2_lostcoast.bsp")
	ap.add_argument("--out", type=Path, default=ROOT / "cache/bake")
	args = ap.parse_args()
	recs = []
	for kv in entities(args.map.read_bytes()):
		if kv.get("classname") != "env_smokestack":
			continue
		org = [float(v) for v in kv.get("origin", "0 0 0").split()]
		rgb = [int(float(v)) for v in kv.get("rendercolor", "255 255 255").split()][:3]
		num = lambda k, d: float(kv.get(k, d) or d)
		u16 = lambda v: int(max(0, min(65535, round(v))))
		recs.append((
			num("EndSize", 30),
			struct.pack(
				">3f4BfHHHHHBB", org[0], org[1], org[2],
				rgb[0], rgb[1], rgb[2], int(max(0, min(255, num("renderamt", 255)))),
				num("JetLength", 180),
				u16(num("StartSize", 20)), u16(num("EndSize", 30)), u16(num("Speed", 30)),
				u16(num("SpreadSpeed", 15)), u16(num("BaseSpread", 20)),
				int(max(1, min(255, num("Rate", 20)))), 1 if kv.get("twist", "0") not in ("0", "") else 0,
			),
		))
		print(f"  smokestack {kv.get('targetname', '-')}: at {org} size {kv.get('StartSize')}→{kv.get('EndSize')} jet {kv.get('JetLength')}")
	# Largest first: the engine may stop early.
	recs.sort(key=lambda r: -r[0])
	recs = recs[:MAX_STACKS]
	# Ropes and cables (move_rope / keyframe_rope → NextKey): a 7-point
	# strip per span, sagged as a parabola whose arc length is span + Slack.
	#   media/rp_<map>.gcr : "GCrp", u16 version (1), u16 count,
	#   then 44 bytes each: s16 xyz[7], u8 width, u8 pad
	import math
	all_ents = list(entities(args.map.read_bytes()))
	by_name = {e.get("targetname"): e for e in all_ents if e.get("targetname")}
	ropes = []
	for kv in all_ents:
		if kv.get("classname") not in ("move_rope", "keyframe_rope") or not kv.get("NextKey"):
			continue
		nxt = by_name.get(kv["NextKey"])
		if not nxt:
			continue
		a = [float(v) for v in kv.get("origin", "0 0 0").split()]
		b = [float(v) for v in nxt.get("origin", "0 0 0").split()]
		span = math.dist(a, b)
		if span < 1.0:
			continue
		slack = max(0.0, float(kv.get("Slack", "25") or 25))
		sag = math.sqrt(3.0 * span * slack / 8.0)
		pts = []
		for i in range(7):
			t = i / 6.0
			x = a[0] + (b[0] - a[0]) * t
			y = a[1] + (b[1] - a[1]) * t
			z = a[2] + (b[2] - a[2]) * t - sag * 4.0 * t * (1.0 - t)
			pts += [int(max(-32767, min(32767, round(v)))) for v in (x, y, z)]
		width = int(max(1, min(8, float(kv.get("Width", "2") or 2))))
		ropes.append(struct.pack(">21hBB", *pts, width, 0))
	ropes = ropes[:64]
	rp = args.out / "media" / f"rp_{args.map.stem.lower()}.gcr"
	rp.write_bytes(struct.pack(">4sHH", b"GCrp", 1, len(ropes)) + b"".join(ropes))
	print(f"  wrote {rp} ropes={len(ropes)}")
	out = args.out / "media" / f"fx_{args.map.stem.lower()}.gcf"
	out.write_bytes(struct.pack(">4sHH", b"GCfx", 1, len(recs)) + b"".join(r[1] for r in recs))
	print(f"  wrote {out} stacks={len(recs)}")


if __name__ == "__main__":
	main()
