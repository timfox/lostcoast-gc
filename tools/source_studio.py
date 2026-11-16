#!/usr/bin/env python3
"""Source (HL2, MDL v44-v48) skeleton + animation reader.

Companion to source_mdl.py (mesh only). Reads what the GoldSrc studio
converter (bake_lostcoast_studio.py) needs to put Lost Coast characters on the
engine's shared r_studio.c path:

  * bones: name, parent, bind pos/quat, poseToBone, rot/pos scales
  * local animations (in the .mdl or streamed from the .ani animblocks,
    including v48 sections), decoded to per-frame parent-local pos + quat
  * sequences: label, flags, blend table (animdesc indices), activity
  * per-vertex bone weights + normals from the VVD

Formats follow the public Source SDK 2013 headers (public/studio.h):
mstudiobone_t 216 B, mstudioanimdesc_t 100 B, mstudioseqdesc_t 212 B,
mstudioanim_t 4 B + data, Quaternion48/64, Vector48 (half floats).
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

STUDIO_ANIM_RAWPOS = 0x01
STUDIO_ANIM_RAWROT = 0x02
STUDIO_ANIM_ANIMPOS = 0x04
STUDIO_ANIM_ANIMROT = 0x08
STUDIO_ANIM_DELTA = 0x10
STUDIO_ANIM_RAWROT2 = 0x20

STUDIO_LOOPING = 0x0001
STUDIO_DELTA = 0x0004


def _cstr(d: bytes, o: int) -> str:
	if o <= 0 or o >= len(d):
		return ""
	e = d.find(b"\0", o)
	return d[o:e if e >= 0 else len(d)].decode("latin-1")


def _half(h: int) -> float:
	return struct.unpack("<e", struct.pack("<H", h))[0]


# ---------------------------------------------------------------- math

def quat_mul(a, b):
	ax, ay, az, aw = a
	bx, by, bz, bw = b
	return (
		aw * bx + ax * bw + ay * bz - az * by,
		aw * by - ax * bz + ay * bw + az * bx,
		aw * bz + ax * by - ay * bx + az * bw,
		aw * bw - ax * bx - ay * by - az * bz,
	)


def angle_quat(x: float, y: float, z: float):
	"""Source AngleQuaternion(RadianEuler) == Xash AngleQuaternion(studio)."""
	sr, cr = math.sin(x * 0.5), math.cos(x * 0.5)
	sp, cp = math.sin(y * 0.5), math.cos(y * 0.5)
	sy, cy = math.sin(z * 0.5), math.cos(z * 0.5)
	return (
		sr * cp * cy - cr * sp * sy,
		cr * sp * cy + sr * cp * sy,
		cr * cp * sy - sr * sp * cy,
		cr * cp * cy + sr * sp * sy,
	)


def quat_matrix(q):
	x, y, z, w = q
	return [
		[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
		[2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
		[2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
	]


def quat_angles(q):
	"""Inverse of angle_quat: RadianEuler (x=roll, y=pitch, z=yaw)."""
	m = quat_matrix(q)
	xy = math.hypot(m[0][0], m[1][0])
	if xy > 0.001:
		yaw = math.atan2(m[1][0], m[0][0])
		pitch = math.atan2(-m[2][0], xy)
		roll = math.atan2(m[2][1], m[2][2])
	else:
		yaw = math.atan2(-m[0][1], m[1][1])
		pitch = math.atan2(-m[2][0], xy)
		roll = 0.0
	return roll, pitch, yaw


def mat34_mul(a, b):
	"""3x4 * 3x4 (affine)."""
	r = [[0.0] * 4 for _ in range(3)]
	for i in range(3):
		for j in range(4):
			r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + (a[i][3] if j == 3 else 0.0)
	return r


def mat34_from(q, p):
	m = quat_matrix(q)
	return [m[0] + [p[0]], m[1] + [p[1]], m[2] + [p[2]]]


def mat34_xform(m, v):
	return (
		m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3],
		m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3],
		m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3],
	)


def mat34_rotate(m, v):
	return (
		m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
		m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
		m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
	)


# ---------------------------------------------------------------- data

@dataclass
class Bone:
	name: str
	parent: int
	pos: tuple
	quat: tuple
	rot: tuple
	posscale: tuple
	rotscale: tuple
	pose_to_bone: list  # 3x4
	flags: int


@dataclass
class Anim:
	name: str
	fps: float
	flags: int
	numframes: int
	# frames[f][bone] = (pos, quat) parent-local; None for bones the anim
	# does not touch (caller uses the bind pose).
	frames: list = field(default_factory=list)


@dataclass
class Seq:
	label: str
	activity: str
	flags: int
	groupsize: tuple
	anims: list  # animdesc indices (row major, groupsize[0] fastest)
	paramindex: tuple


@dataclass
class SourceStudio:
	version: int
	name: str
	bones: list
	anims: list
	seqs: list
	includes: list
	flags: int
	eye: tuple
	hull_min: tuple
	hull_max: tuple


def read_bones(d: bytes) -> list[Bone]:
	numbones, boneindex = struct.unpack_from("<ii", d, 156)
	out = []
	for i in range(numbones):
		o = boneindex + i * 216
		nameidx, parent = struct.unpack_from("<ii", d, o)
		pos = struct.unpack_from("<3f", d, o + 32)
		quat = struct.unpack_from("<4f", d, o + 44)
		rot = struct.unpack_from("<3f", d, o + 60)
		posscale = struct.unpack_from("<3f", d, o + 72)
		rotscale = struct.unpack_from("<3f", d, o + 84)
		m = struct.unpack_from("<12f", d, o + 96)
		p2b = [list(m[0:4]), list(m[4:8]), list(m[8:12])]
		flags = struct.unpack_from("<i", d, o + 160)[0]
		out.append(Bone(_cstr(d, o + nameidx), parent, pos, quat, rot, posscale, rotscale, p2b, flags))
	return out


def _anim_value(d: bytes, base: int, frame: int, scale: float) -> float:
	"""ExtractAnimValue: RLE spans {valid, total} then `valid` shorts."""
	if base == 0:
		return 0.0
	p = base
	k = frame
	while True:
		valid, total = d[p], d[p + 1]
		if total == 0:
			return 0.0  # corrupt / past end
		if total > k:
			break
		k -= total
		p += (valid + 1) * 2
	if valid > k:
		v = struct.unpack_from("<h", d, p + (k + 1) * 2)[0]
	else:
		v = struct.unpack_from("<h", d, p + valid * 2)[0]
	return v * scale


def _quat48(d: bytes, o: int):
	x, y, zw = struct.unpack_from("<HHH", d, o)
	fx = (x - 32768) * (1.0 / 32768.0)
	fy = (y - 32768) * (1.0 / 32768.0)
	fz = ((zw & 0x7FFF) - 16384) * (1.0 / 16384.0)
	w = math.sqrt(max(0.0, 1.0 - fx * fx - fy * fy - fz * fz))
	if zw & 0x8000:
		w = -w
	return (fx, fy, fz, w)


def _quat64(d: bytes, o: int):
	v = struct.unpack_from("<Q", d, o)[0]
	x = v & 0x1FFFFF
	y = (v >> 21) & 0x1FFFFF
	z = (v >> 42) & 0x1FFFFF
	wneg = (v >> 63) & 1
	fx = (x - 1048576) * (1.0 / 1048576.5)
	fy = (y - 1048576) * (1.0 / 1048576.5)
	fz = (z - 1048576) * (1.0 / 1048576.5)
	w = math.sqrt(max(0.0, 1.0 - fx * fx - fy * fy - fz * fz))
	if wneg:
		w = -w
	return (fx, fy, fz, w)


def _decode_bone_chain(d: bytes, start: int, frame: int, bones: list[Bone], nbones_map) -> dict:
	"""Decode one mstudioanim_t chain at `start` for local frame `frame`.

	Returns {bone_index_in_this_mdl: (pos, quat, delta)}.
	"""
	out = {}
	p = start
	guard = 0
	while p and guard < 512:
		guard += 1
		bone, flags, nextoff = struct.unpack_from("<BBh", d, p)
		data = p + 4
		b = bones[bone] if bone < len(bones) else None
		if b is not None:
			delta = bool(flags & STUDIO_ANIM_DELTA)
			# rotation
			if flags & STUDIO_ANIM_RAWROT:
				q = _quat48(d, data)
			elif flags & STUDIO_ANIM_RAWROT2:
				q = _quat64(d, data)
			elif flags & STUDIO_ANIM_ANIMROT:
				vp = data
				offs = struct.unpack_from("<3h", d, vp)
				ang = [
					_anim_value(d, vp + offs[j], frame, b.rotscale[j]) if offs[j] else 0.0
					for j in range(3)
				]
				if not delta:
					ang = [ang[j] + b.rot[j] for j in range(3)]
				q = angle_quat(*ang)
			else:
				q = (0.0, 0.0, 0.0, 1.0) if delta else b.quat
			# position
			if flags & STUDIO_ANIM_RAWPOS:
				po = data
				if flags & STUDIO_ANIM_ANIMROT:
					po += 6
				if flags & STUDIO_ANIM_RAWROT:
					po += 6
				if flags & STUDIO_ANIM_RAWROT2:
					po += 8
				hx, hy, hz = struct.unpack_from("<3H", d, po)
				pos = (_half(hx), _half(hy), _half(hz))
			elif flags & STUDIO_ANIM_ANIMPOS:
				vp = data + (6 if flags & STUDIO_ANIM_ANIMROT else 0)
				offs = struct.unpack_from("<3h", d, vp)
				pos = [
					_anim_value(d, vp + offs[j], frame, b.posscale[j]) if offs[j] else 0.0
					for j in range(3)
				]
				if not delta:
					pos = [pos[j] + b.pos[j] for j in range(3)]
				pos = tuple(pos)
			else:
				pos = (0.0, 0.0, 0.0) if delta else b.pos
			out[bone] = (tuple(pos), q, delta)
		if nextoff == 0:
			break
		p += nextoff
	return out


def read_anims(d: bytes, ani: bytes | None, bones: list[Bone], want: set[int] | None = None) -> list[Anim | None]:
	numanims, animindex = struct.unpack_from("<ii", d, 180)
	numblocks, blockindex = struct.unpack_from("<ii", d, 352)
	blocks = [struct.unpack_from("<ii", d, blockindex + i * 8) for i in range(numblocks)] if numblocks else []
	out: list[Anim | None] = []
	for ai in range(numanims):
		o = animindex + ai * 100
		if want is not None and ai not in want:
			out.append(None)
			continue
		(_base, nameidx, fps, flags, numframes) = struct.unpack_from("<iifii", d, o)
		animblock, animoff = struct.unpack_from("<ii", d, o + 52)
		sectionindex, sectionframes = struct.unpack_from("<ii", d, o + 80)
		a = Anim(_cstr(d, o + nameidx), fps, flags, numframes)

		def chain_for(block: int, off: int, rel_base: int):
			if block == 0:
				return d, rel_base + off
			if ani is None or block >= len(blocks):
				return None, 0
			return ani, blocks[block][0] + off

		for f in range(numframes):
			if sectionframes:
				sec = f // sectionframes
				lf = f - sec * sectionframes
				sb, so = struct.unpack_from("<ii", d, o + sectionindex + sec * 8)
				src, start = chain_for(sb, so, o)
			else:
				lf = f
				src, start = chain_for(animblock, animoff, o)
			if src is None:
				a.frames.append({})
				continue
			a.frames.append(_decode_bone_chain(src, start, lf, bones, None))
		out.append(a)
	return out


def read_seqs(d: bytes) -> list[Seq]:
	numseq, seqindex = struct.unpack_from("<ii", d, 188)
	out = []
	for i in range(numseq):
		o = seqindex + i * 212
		_b, lab, actname, flags = struct.unpack_from("<iiii", d, o)
		numblends, animindexindex = struct.unpack_from("<ii", d, o + 56)
		g0, g1 = struct.unpack_from("<ii", d, o + 68)
		p0, p1 = struct.unpack_from("<ii", d, o + 76)
		n = max(1, g0 * g1)
		idx = list(struct.unpack_from(f"<{n}h", d, o + animindexindex))
		out.append(Seq(_cstr(d, o + lab), _cstr(d, o + actname), flags, (g0, g1), idx, (p0, p1)))
	return out


def read_studio(d: bytes, ani: bytes | None = None, decode_anims: bool = True) -> SourceStudio:
	if d[:4] != b"IDST":
		raise ValueError("not a Source MDL")
	ver = struct.unpack_from("<i", d, 4)[0]
	bones = read_bones(d)
	seqs = read_seqs(d)
	anims = read_anims(d, ani, bones) if decode_anims else []
	ninc, incidx = struct.unpack_from("<ii", d, 336)
	includes = []
	for i in range(ninc):
		o = incidx + i * 8
		includes.append(_cstr(d, o + struct.unpack_from("<i", d, o + 4)[0]))
	return SourceStudio(
		version=ver,
		name=_cstr(d, 12) if False else d[12:76].split(b"\0")[0].decode("latin-1"),
		bones=bones,
		anims=anims,
		seqs=seqs,
		includes=includes,
		flags=struct.unpack_from("<i", d, 152)[0],
		eye=struct.unpack_from("<3f", d, 80),
		hull_min=struct.unpack_from("<3f", d, 104),
		hull_max=struct.unpack_from("<3f", d, 116),
	)


def vvd_vertex(vvd: bytes, vertex_data_start: int, index: int):
	"""(weights[3], bones[3], numbones, pos, normal, uv) for one VVD vertex."""
	off = vertex_data_start + index * 48
	w = struct.unpack_from("<3f", vvd, off)
	b = struct.unpack_from("<3b", vvd, off + 12)
	nb = vvd[off + 15]
	pos = struct.unpack_from("<3f", vvd, off + 16)
	nrm = struct.unpack_from("<3f", vvd, off + 28)
	uv = struct.unpack_from("<2f", vvd, off + 40)
	return w, b, nb, pos, nrm, uv


def bone_world(bones: list[Bone], local: list) -> list:
	"""local[i] = (pos, quat) parent-relative -> 3x4 model-space matrices."""
	world = [None] * len(bones)
	for i, b in enumerate(bones):
		m = mat34_from(local[i][1], local[i][0])
		world[i] = m if b.parent < 0 else mat34_mul(world[b.parent], m)
	return world
