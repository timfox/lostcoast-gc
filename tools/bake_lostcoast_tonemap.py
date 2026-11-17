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
                    (bake_worlddl LIGHT_EXPOSURE), f32 default min, f32 default max,
                    4 spare
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
	out = args.out / "media" / f"tm_{args.map.stem.lower()}.bin"
	out.write_bytes(struct.pack(">4sHHfff4x", b"GCtm", 1, len(recs), LIGHT_EXPOSURE, DEFAULT_MIN, DEFAULT_MAX) + b"".join(recs))
	print(f"  wrote {out} volumes={len(recs)} baked exposure {LIGHT_EXPOSURE}")


if __name__ == "__main__":
	main()
