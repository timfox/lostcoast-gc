#!/usr/bin/env python3
"""Bake the map's auto-exposure limits: media/tm_<map>.bin.

Source's HDR auto-exposure moves a tonemap scale toward a goal measured from
the frame and clamps it to [min, max]. The map sets those limits: its
env_tonemap_controller takes SetAutoExposureMin / SetAutoExposureMax from
trigger volumes as the player walks the level (2 around the cove, up to 10
inside the monastery). Until a trigger fires, the engine defaults apply
(mat_autoexposure_min 0.5, mat_autoexposure_max 2).

Every brush trigger with such outputs becomes one record: its world box,
the limits it sets on entry (OnStartTouch, OnTrigger) and on exit
(OnEndTouch). A trigger_once fires a single time. The engine tests the
camera against the boxes (there is no player).

Not baked: SetBloomScale, and the intro's BlendTonemapScale fade.

  header 24 bytes : "GCtm", u16 version (1), u16 count, f32 baked exposure
                    (bake_worlddl LIGHT_EXPOSURE), f32 min, f32 max at map spawn,
                    f32 sky exposure (SKY_EXPOSURE: the tonemap scale the sky box
                    textures were baked at from the HDR sky; 0 = the sky is
                    screen art and the engine leaves it alone)
  record 44 bytes : f32 mins[3], f32 maxs[3], f32 enter_max, f32 enter_min,
                    f32 leave_max, f32 leave_min (negative = leave unchanged),
                    u32 flags (1 = once)
"""
from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from bake_worlddl import LIGHT_EXPOSURE, Bsp  # noqa: E402

DEFAULT_MIN, DEFAULT_MAX = 0.5, 2.0  # mat_autoexposure_min / _max
# The sky box is baked as gamma( HDR sky x SKY_EXPOSURE ) (scripts/
# bake-lostcoast-sky.sh). Source draws the sky as HDR x tonemap scale; the
# engine multiplies the sky by ( scale / SKY_EXPOSURE )^( 1 / 2.2 ), up to 2.
# At 1 the 8-bit texture clips only where the HDR sky is over 1 (the sun's
# glare), and scales up to 4.6 stay inside the engine's factor of 2.
SKY_EXPOSURE = 1.0
# The world is baked as gamma( light x LIGHT_EXPOSURE ), so the baked picture
# stands for tonemap scale LIGHT_EXPOSURE. It used to be counted as half of
# that (1.4), fitted to one retail still while the engine's exposure left the
# sky out of its measurement and sat near the map's maximum. With the sky
# measured (xash3d-gc exposure-sky) Source's rule puts the cove at the map's
# minimum, 0.9, and against playtests/reference/retail-cove.jpg from the same
# spot the sand is 97 of 255 in retail: ours is 98 with the picture counted
# as 2.8, and was 135 counted as 1.4.
WORLD_LIGHT_UNITS = 1.0


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--map", type=Path, default=ROOT / "content/lostcoast/maps/d2_lostcoast.bsp")
	ap.add_argument("--out", type=Path, default=ROOT / "cache/bake")
	args = ap.parse_args()
	bsp = Bsp(args.map)
	d = bsp.data
	ofs, ln = struct.unpack_from("<ii", d, 8)
	ents = d[ofs:ofs + ln].decode("latin1")
	controllers = set()
	blocks = []
	for blk in re.findall(r"\{([^}]*)\}", ents):
		kv = re.findall(r'"([^"]+)" "([^"]*)"', blk)
		dk = dict(kv)
		if dk.get("classname") == "env_tonemap_controller":
			controllers.add(dk.get("targetname", ""))
		blocks.append((kv, dk))
	recs = []
	for kv, dk in blocks:
		model = dk.get("model", "")
		if not model.startswith("*") or not dk.get("classname", "").startswith("trigger_"):
			continue
		enter = {"max": -1.0, "min": -1.0}
		leave = {"max": -1.0, "min": -1.0}
		hit = False
		for key, val in kv:
			parts = val.split(",")
			if len(parts) < 3 or parts[0] not in controllers:
				continue
			which = {"SetAutoExposureMax": "max", "SetAutoExposureMin": "min"}.get(parts[1])
			if which is None:
				continue
			if key in ("OnStartTouch", "OnTrigger", "OnStartTouchAll"):
				enter[which] = float(parts[2])
				hit = True
			elif key in ("OnEndTouch", "OnEndTouchAll"):
				leave[which] = float(parts[2])
				hit = True
		if not hit:
			continue
		m = bsp.models[int(model[1:])]
		org = [float(x) for x in dk.get("origin", "0 0 0").split()]
		mins = [m[k] + org[k] for k in range(3)]
		maxs = [m[3 + k] + org[k] for k in range(3)]
		once = 1 if dk["classname"] == "trigger_once" else 0
		recs.append(struct.pack(">6f4fI", *mins, *maxs, enter["max"], enter["min"], leave["max"], leave["min"], once))
		print(f"  {dk['classname']} {model}: box {[round(v) for v in mins]} .. {[round(v) for v in maxs]} "
			f"enter max/min {enter['max']}/{enter['min']} leave {leave['max']}/{leave['min']}")
	# The limits the level starts with: what the map sends the controller when
	# it spawns (logic_auto OnMapSpawn, straight or through a logic_relay it
	# triggers). d2_lostcoast: e3_start sets min 0.9, max 3. Without them the
	# engine defaults apply (mat_autoexposure_min / _max).
	start = {"min": DEFAULT_MIN, "max": DEFAULT_MAX}
	by_name: dict[str, list] = {}
	for kv, dk in blocks:
		if dk.get("targetname"):
			by_name.setdefault(dk["targetname"], []).append(kv)

	def apply(outputs, key_names, depth=0):
		for key, val in outputs:
			if key not in key_names:
				continue
			parts = val.split(",")
			if len(parts) < 3:
				continue
			if parts[0] in controllers:
				which = {"SetAutoExposureMax": "max", "SetAutoExposureMin": "min"}.get(parts[1])
				if which:
					start[which] = float(parts[2])
			elif parts[1] == "Trigger" and depth < 4:
				for kv2 in by_name.get(parts[0], []):
					if dict(kv2).get("classname") == "logic_relay":
						apply(kv2, ("OnTrigger",), depth + 1)

	for kv, dk in blocks:
		if dk.get("classname") == "logic_auto":
			apply(kv, ("OnMapSpawn", "OnNewGame"))
	print(f"  limits at map spawn: {start['min']} .. {start['max']}")
	out = args.out / "media" / f"tm_{args.map.stem.lower()}.bin"
	out.write_bytes(struct.pack(">4sHHffff", b"GCtm", 1, len(recs), LIGHT_EXPOSURE / WORLD_LIGHT_UNITS, start["min"], start["max"], SKY_EXPOSURE) + b"".join(recs))
	print(f"  wrote {out} volumes={len(recs)} baked picture = scale {LIGHT_EXPOSURE / WORLD_LIGHT_UNITS}, sky baked at {SKY_EXPOSURE}")


if __name__ == "__main__":
	main()
