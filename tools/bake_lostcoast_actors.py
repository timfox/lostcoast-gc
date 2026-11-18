#!/usr/bin/env python3
"""Bake what Lost Coast's characters do, from the game's own data.

For the GoldSrc-side NPCs in hlsdk dlls/lostcoast_npcs.cpp:

  * scenes: the map's logic_choreographed_scene .vcd files, read from
    scenes/scenes.image (tools/source_vcd.py), flattened to a small text
    script per scene: speak / lookat / face / moveto / gesture / end lines
    with Source's own timings and targets.
  * voice: every wave the scenes and lead nags speak, resolved through the
    game's soundscripts (scripts/level_voices_lostcoast.txt) and resampled to
    16 kHz 16-bit mono for the GameCube sound store.
  * navigation: maps/graphs/d2_lostcoast.ain, Source's AI node graph, as a
    compact binary 'LCNG' (nodes + human-hull ground links) so the game code
    can route with A* like CAI_Navigator does.

Output: <out>/scripts/lc/*.txt, <out>/scripts/lc/<map>.lcng and
<out>/sound/lostcoast/vo/... (copied onto the disc by build-lostcoastgc.sh).
"""
from __future__ import annotations

import argparse
import io
import re
import struct
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from source_vcd import SceneImage  # noqa: E402
from vpk import open_vpk_roots, read_first  # noqa: E402

NUM_HULLS = 10
HULL_HUMAN = 0
AI_MOVE_GROUND = 0x01
VOICE_RATE = 16000

# Source sound levels (soundflags.h) -> GoldSrc attenuation
SNDLVL = {"SNDLVL_TALKING": 80, "SNDLVL_NORM": 75, "SNDLVL_NONE": 0}


def soundscripts(vpks) -> dict[str, dict]:
	out: dict[str, dict] = {}
	for f in ("scripts/level_voices_lostcoast.txt", "scripts/level_sounds_lostcoast.txt"):
		raw = read_first(vpks, f)
		if not raw:
			continue
		text = re.sub(r"//[^\n]*", "", raw.decode("latin-1"))
		for name, body in re.findall(r'"([^"]+)"\s*\{([^{}]*)\}', text):
			kv = dict(re.findall(r'"([^"]+)"\s+"([^"]*)"', body))
			out[name.lower()] = kv
	return out


def sound_atten(kv: dict) -> float:
	lvl = kv.get("soundlevel", "SNDLVL_TALKING")
	db = SNDLVL.get(lvl)
	if db is None:
		m = re.search(r"(\d+)", lvl)
		db = int(m.group(1)) if m else 75
	# engine soundlevel -> attenuation: SNDLVL_TO_ATTN(a) = 20 / (a - 50)
	return 0.0 if db <= 50 else round(20.0 / (db - 50), 3)


def sound_volume(kv: dict) -> float:
	v = kv.get("volume", "VOL_NORM")
	try:
		return float(v)
	except ValueError:
		return 1.0


def resample_wave(data: bytes, rate: int) -> bytes:
	"""Any PCM wave -> mono 16-bit at rate (polyphase low-pass resample)."""
	from math import gcd

	import numpy as np
	from scipy.signal import resample_poly

	with wave.open(io.BytesIO(data)) as w:
		ch, width, src, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
		raw = w.readframes(n)
	if width == 1:
		x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) * 256.0
	else:
		x = np.frombuffer(raw, "<i2").astype(np.float32)
	if ch > 1:
		x = x.reshape(-1, ch).mean(axis=1)
	if src != rate:
		g = gcd(src, rate)
		x = resample_poly(x, rate // g, src // g)
	pcm = np.clip(np.round(x), -32768, 32767).astype("<i2").tobytes()
	out = io.BytesIO()
	with wave.open(out, "wb") as w:
		w.setnchannels(1)
		w.setsampwidth(2)
		w.setframerate(rate)
		w.writeframes(pcm)
	return out.getvalue()


def wave_of(name: str, scripts: dict) -> tuple[str, float, float] | None:
	kv = scripts.get(name.lower())
	if not kv:
		return None
	w = kv.get("wave", "").lstrip("*#@^)(}<>!?").replace("\\", "/")
	return (w, sound_volume(kv), sound_atten(kv)) if w else None


def bake_scene(img: SceneImage, vcd: str, actor: str, scripts: dict, waves: set[str]) -> str | None:
	sc = img.scene(vcd)
	if not sc:
		return None
	lines = [f"// {vcd} (scenes.image), actor {actor}"]
	end = 0.0
	for e in sc.events:
		if e.type == "stoppoint":
			end = e.start
	for a in sc.actors:
		if a.name.lower() != actor.lower():
			continue
		for c in a.channels:
			if not c.active:
				continue
			for e in c.events:
				end = max(end, e.end if e.end > 0 else e.start)
				if e.type == "speak":
					got = wave_of(e.param1, scripts)
					if not got:
						lines.append(f"// unresolved speak {e.param1}")
						continue
					w, vol, att = got
					waves.add(w)
					lines.append(f"speak {e.start:.2f} {e.end:.2f} {w} {vol:g} {att:g}")
				elif e.type in ("lookat", "face"):
					lines.append(f"{e.type} {e.start:.2f} {e.end:.2f} {e.param1}")
				elif e.type == "moveto":
					lines.append(f"moveto {e.start:.2f} {e.end:.2f} {e.param1} {e.distance:g} {e.param2 or 'Walk'}")
				elif e.type == "gesture" and e.param1:
					# Timing tags: where the scene puts each of the sequence's own
					# marks (apex, accent, loop, end). Pairs of playback fraction
					# and sequence fraction; the cycle runs straight between them
					# (CChoreoEvent::GetOriginalPercentageFromPlaybackPercentage).
					play = dict(e.abs_tags[0]) if len(e.abs_tags) > 1 else {}
					pairs = sorted((play[n], o) for n, o in (e.abs_tags[1] if len(e.abs_tags) > 1 else []) if n in play)[:4]
					tags = "".join(f" {p:.3f} {o:.3f}" for p, o in pairs)
					lines.append(f"gesture {e.start:.2f} {e.end:.2f} {e.param1}{tags}")
				elif e.type == "sequence" and e.param1:
					lines.append(f"sequence {e.start:.2f} {e.end:.2f} {e.param1}")
				elif e.type == "firetrigger":
					lines.append(f"trigger {e.start:.2f} {e.param1}")
	lines.append(f"end {end:.2f}")
	return "\n".join(lines) + "\n"


def bake_nodes(ain: bytes) -> bytes:
	"""Source .ain (AINET_VERSION 37) -> 'LCNG': nodes and human ground links."""
	ver, _mapver, n = struct.unpack_from("<iii", ain, 0)
	if ver != 37:
		raise SystemExit(f"unexpected .ain version {ver}")
	o = 12
	nodes = []
	for _ in range(n):
		x, y, z, _yaw = struct.unpack_from("<4f", ain, o)
		o += 16
		voff = struct.unpack_from(f"<{NUM_HULLS}f", ain, o)
		o += 4 * NUM_HULLS
		ntype = ain[o]
		o += 1
		_info, _zone = struct.unpack_from("<Hh", ain, o)
		o += 4
		# ground nodes sit voffset above the floor for each hull
		nodes.append((x, y, z + voff[HULL_HUMAN], ntype))
	nl = struct.unpack_from("<i", ain, o)[0]
	o += 4
	links = []
	for _ in range(nl):
		a, b = struct.unpack_from("<hh", ain, o)
		moves = ain[o + 4:o + 4 + NUM_HULLS]
		o += 4 + NUM_HULLS
		if moves[HULL_HUMAN] & AI_MOVE_GROUND:
			links.append((a, b))
	out = bytearray(b"LCNG")
	out += struct.pack(">HH", len(nodes), len(links))
	for x, y, z, _t in nodes:
		out += struct.pack(">3f", x, y, z)
	for a, b in links:
		out += struct.pack(">HH", a, b)
	print(f"  nodes: {len(nodes)} nodes, {len(links)} human ground links of {nl}")
	return bytes(out)


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--lostcoast", required=True)
	ap.add_argument("--hl2", required=True)
	ap.add_argument("--map", type=Path, required=True)
	ap.add_argument("--out", type=Path, required=True)
	args = ap.parse_args()
	vpks = open_vpk_roots(sorted(Path(args.lostcoast).glob("*_dir.vpk")) + sorted(Path(args.hl2).glob("*_dir.vpk")))
	mapname = args.map.stem
	out = args.out
	(out / "scripts" / "lc").mkdir(parents=True, exist_ok=True)

	img = SceneImage(read_first(vpks, "scenes/scenes.image"))
	scripts = soundscripts(vpks)
	waves: set[str] = set()

	# Scenes the map plays on the fisherman (logic_choreographed_scene).
	d = args.map.read_bytes()
	off, ln = struct.unpack_from("<ii", d, 8)
	text = d[off:off + ln].decode("latin-1", "replace")
	nscene = 0
	for b in re.findall(r"\{[^{}]*\}", text):
		e = dict(re.findall(r'"([^"]+)"\s+"([^"]*)"', b))
		if e.get("classname") != "logic_choreographed_scene" or "fisherman" not in e.get("SceneFile", ""):
			continue
		vcd = e["SceneFile"].replace("\\", "/")
		s = bake_scene(img, vcd, "fisherman", scripts, waves)
		if s:
			(out / "scripts" / "lc" / (Path(vcd).stem.lower() + ".txt")).write_text(s)
			nscene += 1
	print(f"  scenes: {nscene} fisherman scenes")

	# Talker responses (scripts/talker/npc_fisherman.txt): the scenes each
	# concept plays, e.g. the lead behaviour's ComingBack / Retrieve guides.
	talk = read_first(vpks, "scripts/talker/npc_fisherman.txt")
	if talk:
		t = re.sub(r"//[^\n]*", "", talk.decode("latin-1"))
		responses = {n.strip('"').lower(): re.findall(r'scene\s+"([^"]+)"', body)
			for n, body in re.findall(r'response\s+("?[\w]+"?)\s*\{([^{}]*)\}', t)}
		concepts = dict(re.findall(r'criterion\s+"(\w+)"\s+"Concept"\s+"(\w+)"', t))
		lines = ["// scripts/talker/npc_fisherman.txt: concept -> scenes"]
		for body in re.findall(r'rule\s+"?\w+"?\s*\{([^{}]*)\}', t):
			crit = re.search(r"criteria\s+([^\n]+)", body)
			resp = re.search(r"response\s+\"?(\w+)", body)
			if not crit or not resp:
				continue
			concept = next((concepts.get(c, c.replace("Concept", "")) for c in crit.group(1).split()
				if c.startswith("Concept")), None)
			files = []
			for vcd in responses.get(resp.group(1).lower(), []):
				sc = bake_scene(img, vcd, "fisherman", scripts, waves)
				if sc:
					f = "scripts/lc/" + Path(vcd).stem.lower() + ".txt"
					(out / f).write_text(sc)
					files.append(f)
			if concept and files:
				lines.append(f"{concept} " + " ".join(files))
		(out / "scripts" / "lc" / "fisherman_talker.txt").write_text("\n".join(lines) + "\n")
		print(f"  talker: {len(lines) - 1} concepts")

	ain = read_first(vpks, f"maps/graphs/{mapname}.ain")
	if ain:
		(out / "scripts" / "lc" / f"{mapname}.lcng").write_bytes(bake_nodes(ain))

	total = 0
	for w in sorted(waves):
		raw = read_first(vpks, f"sound/{w}")
		if not raw:
			print(f"  missing sound/{w}")
			continue
		dst = out / "sound" / w
		dst.parent.mkdir(parents=True, exist_ok=True)
		pcm = resample_wave(raw, VOICE_RATE)
		dst.write_bytes(pcm)
		total += len(pcm)
	print(f"  voice: {len(waves)} waves, {total // 1024} KB at {VOICE_RATE} Hz")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
