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

Not carried: bone translation (the postures move the pelvis by up to 3
units) and the attachment helper bones.

Output scripts/lc/<name>.gst, big-endian:
  0   "GCGS", u16 version (1), u16 ngest
  8   u16 nbones, u16 0, u32 names_ofs, u32 dir_ofs, pad to 32
  names: nbones x char[32]
  dir:   ngest x 48: char name[28], u16 nframes, u8 nrot, u8 0, f32 seconds,
         u32 bones_ofs, u32 frames_ofs, u32 0
  bones: nrot x u8 (index into names)
  frames: nframes x nrot x 3 x s16 (quat x, y, z * 32767, w >= 0)
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
		for f in range(nf):
			_pos, q = compose(m, g, bp, bq, f / (nf - 1) * 0.9999)
			frames.append([ss.quat_mul(qconj(bq[b]), q[b]) for b in range(m.nb)])
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
		for b in keep:
			if b not in used:
				used.append(b)
		gest.append((g, dur, keep, frames))
	if len(used) > 255:
		raise SystemExit("too many gesture bones")
	names_ofs = 32
	dir_ofs = names_ofs + len(used) * 32
	body = bytearray()
	body_ofs = dir_ofs + len(gest) * 48
	dirb = bytearray()
	for g, dur, keep, frames in gest:
		bones_ofs = body_ofs + len(body)
		body += bytes(used.index(b) for b in keep)
		if len(body) & 1:
			body += b"\0"
		frames_ofs = body_ofs + len(body)
		for fr in frames:
			for b in keep:
				x, y, z, w = fr[b]
				if w < 0:
					x, y, z = -x, -y, -z
				body += struct.pack(">3h", *(max(-32767, min(32767, int(round(c * 32767)))) for c in (x, y, z)))
		dirb += struct.pack(">28sHBBfIII", g.lower().encode()[:27], len(frames), len(keep), 0, dur, bones_ofs, frames_ofs, 0)
		print(f"  gesture {g}: {dur:.1f} s, {len(frames)} frames, {len(keep)} bones")
	hdr = struct.pack(">4sHHHHII", b"GCGS", 1, len(gest), len(used), 0, names_ofs, dir_ofs).ljust(32, b"\0")
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
