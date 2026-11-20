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


def lip_chunk(wav: bytes, source: bytes) -> bytes:
	"""Carry the source wave's phoneme timings (Faceposer's VDAT chunk) into
	the baked wave as a "GCLP" chunk ahead of the samples, where the engine's
	voice stream finds it in the sector it reads first: u16 count, then
	count x (u16 phoneme code, u16 start ms, u16 end ms), big-endian."""
	from source_flex import wave_phonemes

	ev = wave_phonemes(source)
	if not ev:
		return wav
	body = struct.pack(">H", len(ev)) + b"".join(
		struct.pack(">HHH", code & 0xFFFF, min(65535, int(round(s * 1000))), min(65535, int(round(e * 1000)))) for code, s, e in ev)
	if len(body) & 1:
		body += b"\0"
	i = wav.index(b"data", 12)
	out = wav[:i] + b"GCLP" + struct.pack("<I", len(body)) + body + wav[i:]
	return out[:4] + struct.pack("<I", len(out) - 8) + out[8:]


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


FACE_HZ = 10.0
FACE_TRACKS: dict[str, bytes] = {}  # scene stem -> .gft file
_FACE = None  # (Flexes, shape keys of <actor>.gcf), loaded once


def _curve(samples, t: float) -> float:
	"""A flex track at event time t: straight between its samples (Faceposer
	draws a spline through them; they sit about half a second apart)."""
	if not samples:
		return 0.0
	if t <= samples[0][0]:
		return samples[0][1]
	for (t0, v0), (t1, v1) in zip(samples, samples[1:]):
		if t <= t1:
			return v0 + (v1 - v0) * (t - t0) / (t1 - t0) if t1 > t0 else v1
	return samples[-1][1]


def face_track(events, face) -> tuple[bytes, float, float] | None:
	"""A scene's flex animation (brows, eyelids, cheeks ...) as weights of the
	actor's face shapes, FACE_HZ times a second: the scene's controller
	curves through the model's flex rules, exactly as the lip-sync bake runs
	phonemes through them. Stereo tracks split by their balance curve
	(0.5 = both sides full).

	.gft, big-endian: "GCFT", u16 version (2), u16 nshape, u16 nsamples,
	u16 hz * 256, then nsamples x nshape x s8 ((weight - resting weight) * 16).
	"""
	fx, keys = face
	names = {c[0] for c in fx.controllers}
	t0 = min(e.start for e in events)
	t1 = max(e.end for e in events)
	n = max(2, int((t1 - t0) * FACE_HZ) + 1)
	rows = bytearray()
	rest = fx.weights({})
	for i in range(n):
		t = t0 + i / FACE_HZ
		vals: dict[str, float] = {}
		for e in events:
			if not (e.start <= t <= e.end):
				continue
			for tr in e.flex:
				if not tr["active"] or not tr["amount"]:
					continue
				v = _curve(tr["amount"], t - e.start)
				if tr["balance"] or ("left_" + tr["name"]) in names:
					b = _curve(tr["balance"], t - e.start) if tr["balance"] else 0.5
					sides = (("left_", 1.0 if b <= 0.5 else 2.0 * (1.0 - b)), ("right_", 1.0 if b >= 0.5 else 2.0 * b))
					for side, k in sides:
						if side + tr["name"] in names:
							vals[side + tr["name"]] = max(vals.get(side + tr["name"], 0.0), v * k)
				elif tr["name"] in names:
					vals[tr["name"]] = max(vals.get(tr["name"], 0.0), v)
		w = fx.weights(vals)
		rows += struct.pack(f">{len(keys)}b", *(max(-127, min(127,
			int(round((fx.ramp(k[1], w[k[0]]) - fx.ramp(k[1], rest[k[0]])) * 16.0)))) for k in keys))
	return struct.pack(">4sHHHH", b"GCFT", 2, len(keys), n, int(FACE_HZ * 256)) + bytes(rows), t0, t1


_MAP_ENTS: dict = {}     # targetname -> entity, of the map being baked
_SEQ_SECONDS: dict = {}  # the actor's sequences: name -> seconds
# What a sequence's script events set swinging (a scripted_sequence's
# OnScriptEventNN exploding an env_physexplosion next to hinged props: the
# harbor gate): name -> [(seconds into it, hinge, degrees a second, stop)].
_SEQ_HINGES: dict = {}


def _hinge_lines(play: str, t: float) -> list[str]:
	return [f"hinge {t + dt:.2f} {t + dt:.2f} @{h[0]:g},{h[1]:g},{h[2]:g},{spin:.1f},{stop:g}"
		for dt, h, spin, stop in _SEQ_HINGES.get(play.lower(), [])]



def bake_scene(img: SceneImage, vcd: str, actor: str, scripts: dict, waves: set[str], scene_outputs: dict | None = None) -> str | None:
	sc = img.scene(vcd)
	if not sc:
		return None
	lines = [f"// {vcd} (scenes.image), actor {actor}"]
	end = 0.0
	face_events = []
	seqs_named: set = set()
	triggers: list = []
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
					# A named target (a scripted_sequence, a path marker) is
					# written as its place and facing: those entities do not
					# exist in the GameCube map.
					tgt = e.param1
					me = _MAP_ENTS.get(tgt.lower())
					if me and "origin" in me:
						o = me["origin"].split()
						yaw = (me.get("angles", "0 0 0").split() + ["0", "0"])[1]
						tgt = f"@{float(o[0]):g},{float(o[1]):g},{float(o[2]):g},{float(yaw):g}"
					lines.append(f"moveto {e.start:.2f} {e.end:.2f} {tgt} {e.distance:g} {e.param2 or 'Walk'}")
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
					lines.append(f"sequence {e.start:.2f} {e.end:.2f} seq_{e.param1}")
					lines += _hinge_lines(e.param1, e.start)
					seqs_named.add(e.param1.lower())
				elif e.type == "firetrigger":
					lines.append(f"trigger {e.start:.2f} {e.param1}")
					triggers.append((e.start, e.param1))
				elif e.type == "flexanimation" and e.flex and e.name.lower() != "head":
					# (the "head" tracks turn the head bone, not the face)
					face_events.append(e)
	# The scene entity's OnTriggerN outputs that begin a scripted_sequence on
	# him: the sequence at the trigger's time, then the entity's post-idle
	# held (SitDown, then SitDown_PostIdle). One the scene already plays as a
	# sequence event of its own (OpenGate) is not played twice.
	for t, n in sorted(triggers):
		for tgt in (scene_outputs or {}).get(f"ontrigger{n}", []):
			ssq = _MAP_ENTS.get(tgt.lower())
			if not ssq or ssq.get("classname") != "scripted_sequence":
				continue
			play, post = ssq.get("m_iszPlay", ""), ssq.get("m_iszPostIdle", "")
			if not play or play.lower() in seqs_named or play.lower() not in _SEQ_SECONDS:
				continue
			name = f"seq_{play}" + (f">seq_{post}:{_SEQ_SECONDS[post.lower()]:.2f}" if post and post.lower() in _SEQ_SECONDS else "")
			lines.append(f"sequence {t:.2f} {t + _SEQ_SECONDS[play.lower()]:.2f} {name}")
			lines += _hinge_lines(play, t)
	if face_events and _FACE:
		got = face_track(face_events, _FACE)
		if got:
			stem = Path(vcd).stem.lower()
			FACE_TRACKS[stem] = got[0]
			lines.append(f"flex {got[1]:.2f} {got[2]:.2f} scripts/lc/{stem}.gft")
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
	# the fisherman's face shapes, as the character bake wrote them
	global _FACE
	keyfile = out / "models" / "lc" / "fisherman.gcf.json"
	if keyfile.exists():
		import json

		import source_flex

		fb = "models/lostcoast/fisherman/fisherman"
		_FACE = (source_flex.Flexes(read_first(vpks, fb + ".mdl"), read_first(vpks, fb + ".vvd")),
			[(d, tuple(t)) for d, t in json.loads(keyfile.read_text())])
	scripts = soundscripts(vpks)
	waves: set[str] = set()

	# Scenes the map plays on the fisherman (logic_choreographed_scene).
	d = args.map.read_bytes()
	off, ln = struct.unpack_from("<ii", d, 8)
	text = d[off:off + ln].decode("latin-1", "replace")
	nscene = 0
	for b in re.findall(r"\{[^{}]*\}", text):
		e = dict(re.findall(r'"([^"]+)"\s+"([^"]*)"', b))
		if e.get("targetname"):
			_MAP_ENTS.setdefault(e["targetname"].lower(), e)
	import source_studio as _ss

	_fa = "models/lostcoast/fisherman/Fisherman_animations"
	_am = _ss.read_studio(read_first(vpks, _fa + ".mdl"), read_first(vpks, _fa + ".ani"))
	for _sq in _am.seqs:
		_an = _am.anims[_sq.anims[0]]
		_SEQ_SECONDS[_sq.label.lower()] = (_an.numframes - 1) / _an.fps
	# Script events of his sequences (mstudioevent_t, event 1003 with the
	# number as its option) and what the map hangs on them.
	import hinge_swing
	from bake_lostcoast_props import _static_props

	_md = read_first(vpks, _fa + ".mdl")
	_ns, _si = struct.unpack_from("<ii", _md, 188)
	_events: dict = {}
	for _i in range(_ns):
		_o = _si + _i * 212
		_lab = _o + struct.unpack_from("<i", _md, _o + 4)[0]
		_ne, _ei = struct.unpack_from("<ii", _md, _o + 24)
		for _k in range(_ne):
			_e = _o + _ei + _k * 80
			_cyc, _ev = struct.unpack_from("<fi", _md, _e)
			if _ev == 1003:
				_events.setdefault(_md[_lab:_md.index(b"\0", _lab)].decode().lower(), {})[
					_md[_e + 12:_e + 76].split(b"\0")[0].decode("latin-1").strip()] = _cyc
	_all = [dict(re.findall(r'"([^"]+)"\s+"([^"]*)"', b)) for b in re.findall(r"\{[^{}]*\}", text)]
	_props = None
	for b in re.findall(r"\{[^{}]*\}", text):
		pairs = re.findall(r'"([^"]+)"\s+"([^"]*)"', b)
		e = dict(pairs)
		play = e.get("m_iszPlay", "").lower()
		if e.get("classname") != "scripted_sequence" or play not in _SEQ_SECONDS:
			continue
		for k, v in pairs:
			m = re.fullmatch(r"onscriptevent0*(\d+)", k.lower())
			parts = [x.strip() for x in v.split(",")]
			if not m or len(parts) < 4 or parts[1].lower() != "explode" or m.group(1) not in _events.get(play, {}):
				continue
			if _props is None:
				_props = _static_props(d)
			at = _events[play][m.group(1)] * _SEQ_SECONDS[play] + float(parts[3] or 0)
			for h, spin, stop in hinge_swing.swings_of(d, _all, parts[0], lambda p: read_first(vpks, p), _props):
				_SEQ_HINGES.setdefault(play, []).append((at, h, spin, stop))
				print(f"  hinge: {play} +{at:.2f} s swings the leaf at {h[0]:g} {h[1]:g}: {spin:.1f} deg/s to {stop:g}")
	for b in re.findall(r"\{[^{}]*\}", text):
		pairs = re.findall(r'"([^"]+)"\s+"([^"]*)"', b)
		e = dict(pairs)
		if e.get("classname") != "logic_choreographed_scene" or "fisherman" not in e.get("SceneFile", ""):
			continue
		vcd = e["SceneFile"].replace("\\", "/")
		outs: dict = {}
		for k, v in pairs:
			parts = [x.strip() for x in v.split(",")]
			if k.lower().startswith("ontrigger") and len(parts) > 1 and parts[1].lower() == "beginsequence":
				outs.setdefault(k.lower(), []).append(parts[0])
		s = bake_scene(img, vcd, "fisherman", scripts, waves, outs)
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
		pcm = lip_chunk(resample_wave(raw, VOICE_RATE), raw)
		dst.write_bytes(pcm)
		total += len(pcm)
	print(f"  voice: {len(waves)} waves, {total // 1024} KB at {VOICE_RATE} Hz")
	for stem, data in sorted(FACE_TRACKS.items()):
		(out / "scripts" / "lc" / f"{stem}.gft").write_bytes(data)
	if FACE_TRACKS:
		print(f"  face tracks: {len(FACE_TRACKS)} scenes, {sum(len(d) for d in FACE_TRACKS.values()) // 1024} KB")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
