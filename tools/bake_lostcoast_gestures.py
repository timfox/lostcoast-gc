#!/usr/bin/env python3
"""Bake the fisherman's choreography gestures into one rotation track each.

A Source gesture (F_G_hatadjust, g_shrug, HG_nod, p_left_shift_smallstep ...)
is an empty wrapper sequence; what it does is its autolayers: delta layers
(studio flags DELTA|POST), absolute "frame"/"loop" arm poses and spine
layers, each with a start/peak/tail/end window over the wrapper's cycle and
its own per-bone weight list. Source runs that stack over whatever the actor
is doing (bone_setup.cpp AddSequenceLayers / SlerpBones).

Flipper has neither the memory nor the time for the stack, so it is run here,
over the Idle01 pose he holds while he talks, and what is kept per bone is
the local rotation it adds: delta = base^-1 * result. The engine multiplies
that onto the bone's current local rotation (r_gx_gesture.c), which is exact
for the delta layers and exact for the absolute layers wherever the base is
that idle pose.

Bones a gesture also moves (the hat lifting off the head, the pelvis in a
posture shift) carry the translation it adds, base to result, in the
parent's space. Not carried: the attachment helper bones.

Output scripts/lc/<name>.gst, big-endian:
  0   "GCGS", u16 version (2), u16 ngest
  8   u16 nbones, u16 0, u32 names_ofs, u32 dir_ofs, pad to 32
  names: nbones x char[32]
  dir:   ngest x 48: char name[28], u16 nframes, u8 nrot, u8 npos,
         f32 seconds, u32 bones_ofs, u32 frames_ofs, u32 0
  bones: nrot x u8, then npos x u8 (indices into names)
  frames: nframes x ( nrot x 3 x s16 (quat x, y, z * 32767, w >= 0),
          npos x 3 x s16 (translation * 256) )
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import source_studio as ss  # noqa: E402
from vpk import open_vpk_roots, read_first  # noqa: E402

D = "models/lostcoast/fisherman/"
HZ = 10.0
MIN_DEG = 1.0
MIN_MOVE = 0.1   # units: bones that move less carry rotation only
MAX_POS = 8
# scripted_sequence animations baked as streamed tracks (see bake()).
SEQUENCES = ("OpenGate", "SitDown", "SitDown_PostIdle")
SKIP_BONES = ("anim_attachment", "valvebiped.forward")
vpks = None

AL_SPLINE, AL_XFADE, AL_NOBLEND = 0x40, 0x80, 0x200
DELTA, POST = 0x4, 0x10


def qn(q):
	l = math.sqrt(sum(c * c for c in q)) or 1.0
	return tuple(c / l for c in q)


def qconj(q):
	return (-q[0], -q[1], -q[2], q[3])


def qslerp(a, b, t):
	d = sum(x * y for x, y in zip(a, b))
	if d < 0:
		b = tuple(-c for c in b); d = -d
	if d > 0.9995:
		return qn(tuple(x + (y - x) * t for x, y in zip(a, b)))
	th = math.acos(d)
	s = math.sin(th)
	wa, wb = math.sin((1 - t) * th) / s, math.sin(t * th) / s
	return tuple(wa * x + wb * y for x, y in zip(a, b))


ID = (0.0, 0.0, 0.0, 1.0)


def qscale(q, s):
	return qslerp(ID, q, s)


class Model:
	def __init__(self, path):
		self.d = read_first(vpks, path + ".mdl")
		self.am = ss.read_studio(self.d, read_first(vpks, path + ".ani"))
		self.by = {s.label.lower(): i for i, s in enumerate(self.am.seqs)}
		self.nb = len(self.am.bones)
		_n, self.seqindex = struct.unpack_from("<ii", self.d, 188)

	def seq_weights(self, i):
		o = self.seqindex + i * 212
		wli = struct.unpack_from("<i", self.d, o + 156)[0]
		return struct.unpack_from(f"<{self.nb}f", self.d, o + wli)

	def layers(self, i):
		o = self.seqindex + i * 212
		nal, ali = struct.unpack_from("<ii", self.d, o + 148)
		return [struct.unpack_from("<hhiffff", self.d, o + ali + k * 24) for k in range(nal)]

	def sample(self, i, cycle):
		"""Local pose of sequence i at cycle: {bone: (pos, quat)} for animated bones."""
		s = self.am.seqs[i]
		a = self.am.anims[s.anims[0]]
		n = a.numframes
		if n <= 1:
			f0 = f1 = 0; t = 0.0
		else:
			if s.flags & 1:
				cycle -= math.floor(cycle)
			cycle = min(max(cycle, 0.0), 1.0)
			f = cycle * (n - 1)
			f0 = min(int(f), n - 1); f1 = min(f0 + 1, n - 1); t = f - f0
		out = {}
		for b, (p0, q0, _d) in a.frames[f0].items():
			p1, q1, _ = a.frames[f1].get(b, (p0, q0, _d))
			out[b] = (tuple(x + (y - x) * t for x, y in zip(p0, p1)), qslerp(q0, q1, t))
		return out


def compose(m: Model, name: str, base_pos, base_q, cycle: float):
	"""Apply the wrapper's autolayers (flWeight 1) onto the base pose, as bone_setup.cpp AddSequenceLayers."""
	pos = list(base_pos); q = list(base_q)
	wi = m.by[name.lower()]
	for iseq, _ipose, fl, start, peak, tail, end in m.layers(wi):
		lc, lw = cycle, 1.0
		if start != end:
			s = 1.0
			if cycle < start or cycle >= end:
				continue
			if cycle < peak and start != peak:
				s = (cycle - start) / (peak - start)
			elif cycle > tail and end != tail:
				s = (end - cycle) / (end - tail)
			if fl & AL_SPLINE:
				s = 3 * s * s - 2 * s * s * s
			lw = s
			lc = (cycle - start) / (end - start)
		if lw <= 0:
			continue
		sq = m.am.seqs[iseq]
		w = m.seq_weights(iseq)
		pose = m.sample(iseq, lc)
		for b in range(m.nb):
			s2 = lw * w[b]
			if s2 <= 0:
				continue
			if sq.flags & DELTA:
				if b not in pose:
					continue
				p2, q2 = pose[b]
				if sq.flags & POST:
					q[b] = qn(ss.quat_mul(q[b], qscale(q2, s2)))
				else:
					q[b] = qn(ss.quat_mul(qscale(q2, s2), q[b]))
				pos[b] = tuple(x + y * s2 for x, y in zip(pos[b], p2))
			else:
				p2, q2 = pose.get(b, (m.am.bones[b].pos, m.am.bones[b].quat))
				q[b] = qslerp(q[b], q2, s2)
				pos[b] = tuple(x + (y - x) * s2 for x, y in zip(pos[b], p2))
	return pos, q


def bake(out: Path, names_wanted: list[str]) -> None:
	anim = Model(D + "Fisherman_animations")
	libs = [Model(D + "fisherman_gestures"), Model(D + "fisherman_postures")]
	bnames = [b.name for b in anim.am.bones]
	for m in libs:
		if [b.name for b in m.am.bones] != bnames:
			raise SystemExit("gesture models do not share the animation skeleton")
	idle = anim.sample(anim.by["idle01"], 0.0)
	bp = [idle.get(b, (anim.am.bones[b].pos, anim.am.bones[b].quat))[0] for b in range(anim.nb)]
	bq = [idle.get(b, (anim.am.bones[b].pos, anim.am.bones[b].quat))[1] for b in range(anim.nb)]
	used: list[int] = []
	gest = []
	for g in names_wanted:
		m = next((x for x in libs if g.lower() in x.by), None)
		if not m or not m.layers(m.by[g.lower()]):
			print(f"  gesture {g}: not a layered sequence, skipped")
			continue
		a = m.am.anims[m.am.seqs[m.by[g.lower()]].anims[0]]
		dur = (a.numframes - 1) / a.fps
		nf = max(2, int(round(dur * HZ)) + 1)
		frames = []
		moves = []
		for f in range(nf):
			pos, q = compose(m, g, bp, bq, f / (nf - 1) * 0.9999)
			frames.append([ss.quat_mul(qconj(bq[b]), q[b]) for b in range(m.nb)])
			moves.append([tuple(pos[b][i] - bp[b][i] for i in range(3)) for b in range(m.nb)])
		keep = []
		for b in range(m.nb):
			if any(s in bnames[b].lower() for s in SKIP_BONES):
				continue
			peak = max(2 * math.degrees(math.acos(min(1.0, abs(fr[b][3])))) for fr in frames)
			if peak >= MIN_DEG:
				keep.append(b)
		if not keep or len(keep) > 255:
			print(f"  gesture {g}: nothing to keep, skipped")
			continue
		far = {b: max(math.sqrt(sum(c * c for c in mv[b])) for mv in moves) for b in range(m.nb)
			if not any(s in bnames[b].lower() for s in SKIP_BONES)}
		pkeep = sorted((b for b in far if far[b] >= MIN_MOVE), key=lambda b: -far[b])[:MAX_POS]
		for b in keep + pkeep:
			if b not in used:
				used.append(b)
		gest.append((g, dur, keep, frames, pkeep, moves, far))
	# Whole-body sequences the map plays through scripted_sequence entities
	# (the fisherman opens the gate, sits down). In the studio model they
	# would cost the studio arena 45 KB and more each; as tracks here they
	# stream from ARAM two frames at a time. Stored like a gesture: what the
	# sequence's pose adds over the talking pose, for the bones it turns most
	# (the engine takes 48 a track) and the bones it moves.
	for sname in SEQUENCES:
		if sname.lower() not in anim.by:
			print(f"  sequence {sname}: not in the model, skipped")
			continue
		si = anim.by[sname.lower()]
		a = anim.am.anims[anim.am.seqs[si].anims[0]]
		dur = (a.numframes - 1) / a.fps
		nf = max(2, int(round(dur * HZ)) + 1)
		frames, moves = [], []
		for f in range(nf):
			pose = anim.sample(si, f / (nf - 1) * 0.9999)
			frames.append([ss.quat_mul(qconj(bq[b]), pose[b][1]) if b in pose else ID for b in range(anim.nb)])
			moves.append([tuple(pose[b][0][i] - bp[b][i] for i in range(3)) if b in pose else (0.0, 0.0, 0.0) for b in range(anim.nb)])
		peak = {b: max(2 * math.degrees(math.acos(min(1.0, abs(fr[b][3])))) for fr in frames) for b in range(anim.nb)
			if not any(x in bnames[b].lower() for x in SKIP_BONES)}
		keep = sorted((b for b in peak if peak[b] >= MIN_DEG), key=lambda b: -peak[b])[:48]
		keep.sort()
		far = {b: max(math.sqrt(sum(c * c for c in mv[b])) for mv in moves) for b in peak}
		pkeep = sorted((b for b in far if far[b] >= MIN_MOVE), key=lambda b: -far[b])[:MAX_POS]
		for b in keep + pkeep:
			if b not in used:
				used.append(b)
		gest.append(("seq_" + sname, dur, keep, frames, pkeep, moves, far))
		print(f"  sequence {sname}: {dur:.1f} s, {nf} frames, {len(keep)} bones turned (of {sum(1 for b in peak if peak[b] >= MIN_DEG)}), "
			f"{len(pkeep)} moved, largest move {max(far.values()):.0f} units")
	# Look-at pose parameters (head_yaw, head_pitch, spine_yaw, body_yaw): each
	# is a three-way blend sequence of delta poses at the parameter's low end,
	# middle and high end. Kept as three-frame tracks "look_<pose>" whose
	# cycle is the parameter's place in its range. A GoldSrc bone controller
	# cannot stand in for these: it adds to one Euler angle of one bone, and
	# these bones rest far from zero (the pelvis at roll 88, yaw 90), so the
	# turn came out about the wrong axis.
	from bake_lostcoast_studio import LOOK_POSES, _pose_params

	params = [p[0] for p in _pose_params(anim.d)]
	for pose in LOOK_POSES:
		if pose not in params:
			continue
		pi = params.index(pose)
		si = next((i for i, q in enumerate(anim.am.seqs) if q.paramindex and q.paramindex[0] == pi and len(q.anims) == 3), None)
		if si is None:
			continue
		w = anim.seq_weights(si)
		frames = []
		for ai in anim.am.seqs[si].anims:
			fr = anim.am.anims[ai].frames[0]
			frames.append([qscale(fr[b][1], w[b]) if b in fr and w[b] > 0 else ID for b in range(anim.nb)])
		keep = [b for b in range(anim.nb)
			if not any(x in bnames[b].lower() for x in SKIP_BONES)
			and max(2 * math.degrees(math.acos(min(1.0, abs(fr[b][3])))) for fr in frames) >= MIN_DEG]
		if not keep:
			continue
		for b in keep:
			if b not in used:
				used.append(b)
		gest.append(("look_" + pose, 0.0, keep, frames, [], [], {}))
	if len(used) > 255:
		raise SystemExit("too many gesture bones")
	names_ofs = 32
	dir_ofs = names_ofs + len(used) * 32
	body = bytearray()
	body_ofs = dir_ofs + len(gest) * 48
	dirb = bytearray()
	for g, dur, keep, frames, pkeep, moves, far in gest:
		bones_ofs = body_ofs + len(body)
		body += bytes(used.index(b) for b in keep) + bytes(used.index(b) for b in pkeep)
		if len(body) & 1:
			body += b"\0"
		frames_ofs = body_ofs + len(body)
		for fi, fr in enumerate(frames):
			for b in keep:
				x, y, z, w = fr[b]
				if w < 0:
					x, y, z = -x, -y, -z
				body += struct.pack(">3h", *(max(-32767, min(32767, int(round(c * 32767)))) for c in (x, y, z)))
			for b in pkeep:
				body += struct.pack(">3h", *(max(-32767, min(32767, int(round(c * 256.0)))) for c in moves[fi][b]))
		dirb += struct.pack(">28sHBBfIII", g.lower().encode()[:27], len(frames), len(keep), len(pkeep), dur, bones_ofs, frames_ofs, 0)
		print(f"  gesture {g}: {dur:.1f} s, {len(frames)} frames, {len(keep)} bones turn, {len(pkeep)} move"
			+ (" (" + ", ".join(f"{bnames[b].split('.')[-1].replace('Bip01_', '')} {far[b]:.1f}" for b in pkeep[:4]) + ")" if pkeep else ""))
	hdr = struct.pack(">4sHHHHII", b"GCGS", 2, len(gest), len(used), 0, names_ofs, dir_ofs).ljust(32, b"\0")
	names = b"".join(bnames[b].encode()[:31].ljust(32, b"\0") for b in used)
	out.parent.mkdir(parents=True, exist_ok=True)
	out.write_bytes(hdr + names + bytes(dirb) + bytes(body))
	print(f"  wrote {out}: {len(gest)} gestures, {len(used)} bones, {out.stat().st_size} bytes")


def main() -> int:
	global vpks
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--lostcoast", required=True)
	ap.add_argument("--hl2", required=True)
	ap.add_argument("--out", type=Path, required=True, help="bake studio dir (holds scripts/lc)")
	args = ap.parse_args()
	vpks = open_vpk_roots(
		sorted(Path(args.lostcoast).glob("*_dir.vpk")) + sorted(Path(args.hl2).glob("*_dir.vpk"))
	)
	wanted: list[str] = []
	for p in sorted((args.out / "scripts" / "lc").glob("fisherman_*.txt")):
		for line in p.read_text().splitlines():
			w = line.split()
			if len(w) >= 4 and w[0] == "gesture" and w[3].lower() not in (x.lower() for x in wanted):
				wanted.append(w[3])
	bake(args.out / "scripts" / "lc" / "fisherman.gst", wanted)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
