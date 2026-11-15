#!/usr/bin/env python3
"""Convert Lost Coast's HL2 characters into engine studio models (MDL v10).

The shared engine's r_studio.c is the character path Day of Defeat and HL1 use
(bones, sequences, interpolation, studio lighting). It reads GoldSrc studio
v10 files. This tool turns a Source model (MDL v44-48 + VVD + VTX + includes /
.ani) into one, so Lost Coast NPCs run on that same path instead of being
frozen, bind-posed meshes in the prop pack.

What carries over:
  * the full skeleton (names, hierarchy, bind pose)
  * selected sequences, decoded from Source's compressed animation (RLE
    euler / Quaternion48 / Quaternion64 / Vector48, .ani blocks, sections)
    and re-encoded as GoldSrc RLE channels (value + anim * scale)
  * geometry at one LOD; each vertex rides its heaviest bone (rigid
    skinning, the GoldSrc model and what Flipper's matrix palette does in
    hardware: one PNMTX index per vertex)
  * skins: $basetexture of each material, 8-bit palettised like any studio
    skin, skin families kept

Output goes to <out>/models/lc/<name>.mdl.
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
from vtf import decode_vtf  # noqa: E402

# HL1 activity ids (dlls/activity.h) for sequences the server looks up.
ACT = {"idle": 1, "walk": 3, "run": 4, "fly": 5, "hop": 7, "land": 10}

# name, source model, anim source models, [(goldsrc label, source seq, act)]
MODELS = {
	"fisherman": (
		"models/lostcoast/fisherman/fisherman",
		["models/lostcoast/fisherman/Fisherman_animations"],
		[
			# Intro_preIdle is what he does while the level waits for you;
			# monster_generic idles on the ACT_IDLE sequence.
			("idle", "Idle01", None),
			("intro_preidle", "Intro_preIdle", "idle"),
			("intro", "Intro", None),
			("walk", "walk_all_Moderate", "walk"),
			# SitDown / SitDown_PostIdle come at the gate much later; they
			# cost ~45 KB of the 512 KB studio arena, so they wait for
			# animation streaming from ARAM.
		],
	),
	"combine_soldier": (
		"models/combine_soldier",
		["models/combine_soldier_anims"],
		[
			("idle", "Idle1_SMG1", "idle"),
			("combatidle", "CombatIdle1_SMG1", None),
			("crouchidle", "CrouchIdle", None),
			("walk", "Walk_all", "walk"),
			("run", "RunALL", "run"),
		],
	),
	"combine_super_soldier": (
		"models/combine_super_soldier",
		["models/combine_soldier_anims"],
		[
			("idle", "Idle1_SMG1", "idle"),
			("combatidle", "CombatIdle1_SMG1", None),
			("walk", "Walk_all", "walk"),
		],
	),
	"seagull": (
		"models/seagull",
		["models/seagull"],
		[
			("idle", "Idle01", "idle"),
			("walk", "Walk", "walk"),
			("fly", "Fly", "fly"),
			("soar", "Soar", None),
			("takeoff", "Takeoff", None),
			("land", "Land", "land"),
			("hop", "Hop", "hop"),
		],
	),
}

# LOD per model: CPU bone setup + skin cost scales with verts on Gekko.
# The engine reads studio files into a 512 KB arena (model.c
# GC_STUDIO_MODEL_SHARE_BYTES) shared with the crowbar and grenade, so the
# cast has to fit beside them: ~1-2k verts each, near a DoD player model.
DEFAULT_LOD = {"fisherman": 5, "combine_soldier": 3, "combine_super_soldier": 4, "seagull": 1}

MAX_FPS = 15.0  # GC: sample animation at 15 Hz, r_studio interpolates frames


def _cstr(d: bytes, o: int) -> str:
	e = d.find(b"\0", o)
	return d[o:e].decode("latin-1")


# ---------------------------------------------------------------- mesh

def load_mesh(mdl: bytes, vvd: bytes, vtx: bytes, lod: int):
	"""Triangles per material at one LOD, keyed to absolute VVD vertex ids."""
	num_lod_verts = struct.unpack_from("<8i", vvd, 16)
	num_fixups, fixup_off, vstart, _tan = struct.unpack_from("<iiii", vvd, 48)
	max_lod = max(i for i in range(8) if num_lod_verts[i] > 0)
	lod = max(0, min(lod, max_lod))
	if num_fixups > 0:
		lod_map: list[int] = []
		for i in range(num_fixups):
			flod, src, num = struct.unpack_from("<iii", vvd, fixup_off + i * 12)
			if flod >= lod:
				lod_map.extend(range(src, src + num))
	else:
		lod_map = list(range(num_lod_verts[0]))

	# MDL side: per bodypart, model 0: vertexindex + mesh vertexoffset/material.
	nbp, bpidx = struct.unpack_from("<ii", mdl, 232)
	mdl_meshes = []  # [bp] -> list of (material, abs_vertex_base)
	for bp in range(nbp):
		b = bpidx + bp * 16
		_n, nmodels, _base, midx = struct.unpack_from("<iiii", mdl, b)
		mh = b + midx  # model 0 only (default bodygroup)
		nmesh, meshidx, _nv, vindex = struct.unpack_from("<iiii", mdl, mh + 72)
		ms = []
		# With VVD fixups a LOD's vertex stream only holds that LOD's verts,
		# so a mesh starts after the previous meshes' numLODVertexes[lod]
		# (mstudio_meshvertexdata_t at +48), not at its LOD0 vertexoffset.
		run = vindex // 48
		for k in range(nmesh):
			msh = mh + meshidx + k * 116
			material, _mi, _nvm, voff = struct.unpack_from("<iiii", mdl, msh)
			if num_fixups > 0:
				ms.append((material, run))
				run += struct.unpack_from("<8i", mdl, msh + 52)[lod]
			else:
				ms.append((material, vindex // 48 + voff))
		mdl_meshes.append(ms)

	_ver, _vc, _a, _b, _c, _chk, _nl, _mr, vbp, vbpo = struct.unpack_from("<iiHHiiiiii", vtx, 0)
	tris = []  # (material, vvd_a, vvd_b, vvd_c)
	for bp in range(min(vbp, len(mdl_meshes))):
		bph = vbpo + bp * 8
		_nm, moff = struct.unpack_from("<ii", vtx, bph)
		mh = bph + moff  # model 0
		nlods, lodoff = struct.unpack_from("<ii", vtx, mh)
		ul = min(lod, nlods - 1)
		lh = mh + lodoff + ul * 12
		nmesh, meshoff, _sw = struct.unpack_from("<iif", vtx, lh)
		for k in range(min(nmesh, len(mdl_meshes[bp]))):
			material, vbase = mdl_meshes[bp][k]
			meh = lh + meshoff + k * 9
			nsg, sgoff, _fl = struct.unpack_from("<iiB", vtx, meh)
			for g in range(nsg):
				sgh = meh + sgoff + g * 25
				nv, voff, ni, ioff, _ns, _so, _sf = struct.unpack_from("<iiiiiiB", vtx, sgh)
				ids = []
				for vi in range(nv):
					orig = struct.unpack_from("<H", vtx, sgh + voff + vi * 9 + 4)[0] + vbase
					ids.append(lod_map[orig] if orig < len(lod_map) else -1)
				for ii in range(0, ni - 2, 3):
					a, b2, c = struct.unpack_from("<3H", vtx, sgh + ioff + ii * 2)
					va, vb, vc = ids[a], ids[b2], ids[c]
					if min(va, vb, vc) < 0 or va == vb or vb == vc or va == vc:
						continue
					tris.append((material, va, vb, vc))
	return tris, vstart, lod


def stripify(tris: list[tuple[int, int, int]]):
	"""Greedy strips. Returns list of vertex lists (strip order, GoldSrc rules:
	odd triangles are wound the other way)."""
	edge_tris: dict[tuple[int, int], list[int]] = {}
	for i, (a, b, c) in enumerate(tris):
		for e in ((a, b), (b, c), (c, a)):
			edge_tris.setdefault(e, []).append(i)
	used = [False] * len(tris)

	def take(e):
		for t in edge_tris.get(e, ()):
			if not used[t]:
				return t
		return -1

	strips = []
	for i in range(len(tris)):
		if used[i]:
			continue
		best = None
		a0, b0, c0 = tris[i]
		for rot in ((a0, b0, c0), (b0, c0, a0), (c0, a0, b0)):
			used[i] = True
			s = list(rot)
			tmp = []
			while True:
				n = len(s)
				# next tri shares edge (s[-2], s[-1]); winding alternates
				if n % 2 == 1:
					e = (s[-1], s[-2])
				else:
					e = (s[-2], s[-1])
				t = take(e)
				if t < 0:
					break
				x, y, z = tris[t]
				third = {x, y, z} - {e[0], e[1]}
				if len(third) != 1:
					break
				used[t] = True
				tmp.append(t)
				s.append(third.pop())
			for t in tmp:
				used[t] = False
			used[i] = False
			if best is None or len(s) > len(best[0]):
				best = (s, tmp)
		s, tmp = best
		used[i] = True
		for t in tmp:
			used[t] = True
		strips.append(s)
	return strips


# ---------------------------------------------------------------- textures

def _vmt_basetexture(raw: bytes | None) -> tuple[str | None, dict]:
	if not raw:
		return None, {}
	import re
	text = raw.decode("latin-1", "replace")
	kv = {k.lower(): v for k, v in re.findall(r'"?(\$\w+)"?\s+"?([^"\n\r]+?)"?\s*$', text, re.M)}
	shader = text.strip().split()[0].strip('"').lower() if text.strip() else ""
	kv["__shader"] = shader
	bt = kv.get("$basetexture") or kv.get("$iris")
	return (bt.replace("\\", "/").strip() if bt else None), kv


def load_skin(vpks, cdpaths: list[str], name: str, max_side: int):
	for cd in cdpaths + [""]:
		base = f"materials/{cd}{name}".replace("\\", "/").replace("//", "/")
		raw = read_first(vpks, base + ".vmt")
		if not raw:
			continue
		bt, kv = _vmt_basetexture(raw)
		for rel in ([f"materials/{bt}.vtf"] if bt else []) + [base + ".vtf"]:
			data = read_first(vpks, rel)
			if not data:
				continue
			img = decode_vtf(data, max_side=max_side)
			return img, kv, rel
	return None, {}, None


def palettise(rgba: bytes, w: int, h: int, masked: bool):
	from PIL import Image
	im = Image.frombytes("RGBA", (w, h), rgba)
	rgb = im.convert("RGB")
	ncol = 255 if masked else 256
	q = rgb.quantize(colors=ncol, method=Image.Quantize.MEDIANCUT)
	idx = bytearray(q.tobytes())
	pal = q.getpalette()[: ncol * 3]
	pal += [0] * (768 - len(pal))
	if masked:
		alpha = im.getchannel("A").tobytes()
		for i, a in enumerate(alpha):
			if a < 128:
				idx[i] = 255
		pal[765:768] = [0, 0, 255]
	return bytes(idx), bytes(pal)


def pow2_fit(w: int, h: int, max_side: int):
	def p2(x):
		n = 1
		while n * 2 <= x:
			n *= 2
		return n
	nw, nh = p2(w), p2(h)
	while nw > max_side or nh > max_side:
		nw = max(8, nw // 2) if nw > max_side or nw >= nh else nw
		nh = max(8, nh // 2) if nh > max_side or nh > nw else nh
	return nw, nh


def resize_rgba(rgba: bytes, w: int, h: int, nw: int, nh: int) -> bytes:
	"""Resize RGB and A separately (Pillow premultiplies RGBA, see memory)."""
	from PIL import Image
	if (w, h) == (nw, nh):
		return rgba
	im = Image.frombytes("RGBA", (w, h), rgba)
	r, g, b, a = im.split()
	rgb = Image.merge("RGB", (r, g, b)).resize((nw, nh), Image.Resampling.BOX)
	a2 = a.resize((nw, nh), Image.Resampling.BOX)
	rgb.putalpha(a2)
	return rgb.tobytes()


# ---------------------------------------------------------------- anim

def pick_blend(seq: ss.Seq, anims: list, bones: list) -> int:
	"""Center blend; for locomotion pick the blend that moves forward most."""
	if len(seq.anims) == 1:
		return seq.anims[0]
	g0, g1 = seq.groupsize
	center = seq.anims[(g1 // 2) * g0 + g0 // 2]
	low = seq.label.lower()
	if not ("walk" in low or "run" in low):
		return center
	best, bestx = center, -1e9
	for ai in set(seq.anims):
		a = anims[ai]
		if a is None or a.numframes < 2:
			continue
		f0 = a.frames[0].get(0)
		f1 = a.frames[-1].get(0)
		if not f0 or not f1:
			continue
		dx = f1[0][0] - f0[0][0]
		if dx > bestx:
			best, bestx = ai, dx
	return best


def sample_anim(a: ss.Anim, bones: list, fps_cap: float):
	"""Full per-bone (pos, quat) frames resampled to <= fps_cap."""
	src_fps = a.fps if a.fps > 0 else 30.0
	n = max(1, a.numframes)
	dur = (n - 1) / src_fps if n > 1 else 0.0
	fps = min(src_fps, fps_cap)
	out_n = max(1, int(round(dur * fps)) + 1) if n > 1 else 1
	frames = []
	for i in range(out_n):
		t = (i / fps) * src_fps if out_n > 1 else 0.0
		k = min(n - 1, int(round(t)))
		fr = a.frames[k] if k < len(a.frames) else {}
		row = []
		for bi, b in enumerate(bones):
			got = fr.get(bi)
			row.append((got[0], got[1]) if got else (b.pos, b.quat))
		frames.append(row)
	fps_out = fps if out_n > 1 else src_fps
	return frames, fps_out


def encode_rle(vals: list[int]) -> bytes:
	out = bytearray()
	s, n = 0, len(vals)
	while s < n:
		e = min(s + 255, n)
		t = e - 1
		while t > s and vals[t - 1] == vals[e - 1]:
			t -= 1
		valid = t - s + 1
		out += struct.pack("<BB", valid, e - s)
		for v in vals[s:s + valid]:
			out += struct.pack("<h", v)
		s = e
	return bytes(out)


# ---------------------------------------------------------------- writer

def convert(vpks, name: str, lod: int, tex_max: int, out_dir: Path, flip: bool) -> Path:
	base, anim_srcs, seq_sel = MODELS[name]
	mdl = read_first(vpks, base + ".mdl")
	vvd = read_first(vpks, base + ".vvd")
	vtx = read_first(vpks, base + ".dx90.vtx")
	if not (mdl and vvd and vtx):
		raise SystemExit(f"{name}: missing source files")
	src = ss.read_studio(mdl, decode_anims=False)
	bones = src.bones
	if len(bones) > 128:
		raise SystemExit(f"{name}: {len(bones)} bones > MAXSTUDIOBONES")

	# ---- sequences
	seq_out = []  # (label, act, flags, fps, frames)
	anim_cache = {}
	for label, sname, act in seq_sel:
		got = None
		for asrc in anim_srcs:
			if asrc not in anim_cache:
				ad = read_first(vpks, asrc + ".mdl")
				ani = read_first(vpks, asrc + ".ani")
				am = ss.read_studio(ad, ani)
				# map anim-model bone index -> main bone index by name
				names = {b.name: i for i, b in enumerate(bones)}
				remap = [names.get(b.name, -1) for b in am.bones]
				anim_cache[asrc] = (am, remap)
			am, remap = anim_cache[asrc]
			for s in am.seqs:
				if s.label.lower() == sname.lower():
					got = (am, remap, s)
					break
			if got:
				break
		if not got:
			print(f"  {name}: sequence {sname} not found, skipped")
			continue
		am, remap, s = got
		ai = pick_blend(s, am.anims, am.bones)
		a = am.anims[ai]
		if a.flags & ss.STUDIO_DELTA:
			print(f"  {name}: {sname} is a delta layer, skipped")
			continue
		# re-key decoded frames to main-model bone order
		rk = ss.Anim(a.name, a.fps, a.flags, a.numframes, [
			{remap[b]: v for b, v in fr.items() if 0 <= b < len(remap) and remap[b] >= 0}
			for fr in a.frames
		])
		frames, fps = sample_anim(rk, bones, MAX_FPS)
		seq_out.append((label, ACT.get(act, 0) if act else 0, s.flags & ss.STUDIO_LOOPING, fps, frames))
	if not seq_out:
		raise SystemExit(f"{name}: no sequences")

	# root motion: strip net XY travel of the root over locomotion cycles
	motions = []
	for idx, (label, act, flags, fps, frames) in enumerate(seq_out):
		lm = (0.0, 0.0, 0.0)
		if act in (ACT["walk"], ACT["run"]) and len(frames) > 1:
			p0 = frames[0][0][0]
			p1 = frames[-1][0][0]
			dx, dy = p1[0] - p0[0], p1[1] - p0[1]
			if math.hypot(dx, dy) > 1.0:
				n = len(frames) - 1
				for f in range(len(frames)):
					p, q = frames[f][0]
					frames[f][0] = ((p[0] - dx * f / n, p[1] - dy * f / n, p[2]), q)
				lm = (dx, dy, 0.0)
		motions.append(lm)

	# ---- geometry
	tris, vstart, used_lod = load_mesh(mdl, vvd, vtx, lod)
	vert_ids = sorted({v for t in tris for v in t[1:]})
	vmap = {v: i for i, v in enumerate(vert_ids)}
	verts, vbone, norms, uvs = [], [], [], []
	for vid in vert_ids:
		w, b, nb, pos, nrm, uv = ss.vvd_vertex(vvd, vstart, vid)
		k = max(range(max(1, nb)), key=lambda j: w[j])
		bi = b[k] if 0 <= b[k] < len(bones) else 0
		p2b = bones[bi].pose_to_bone
		verts.append(ss.mat34_xform(p2b, pos))
		norms.append(ss.mat34_rotate(p2b, nrm))
		vbone.append(bi)
		uvs.append(uv)

	# bind-pose bounds (model space)
	bind_world = ss.bone_world(bones, [(b.pos, b.quat) for b in bones])
	mpos = [ss.mat34_xform(bind_world[vbone[i]], verts[i]) for i in range(len(verts))]
	bbmin = tuple(min(p[a] for p in mpos) for a in range(3))
	bbmax = tuple(max(p[a] for p in mpos) for a in range(3))

	# ---- textures
	numtex, texidx = struct.unpack_from("<ii", mdl, 204)
	ncd, cdidx = struct.unpack_from("<ii", mdl, 212)
	cds = [_cstr(mdl, struct.unpack_from("<i", mdl, cdidx + i * 4)[0]).replace("\\", "/") for i in range(ncd)]
	textures = []  # (name, flags, w, h, idx, pal)
	for t in range(numtex):
		o = texidx + t * 64
		tname = _cstr(mdl, o + struct.unpack_from("<i", mdl, o)[0])
		low = tname.lower()
		side = tex_max
		if any(k in low for k in ("eye", "pupil", "glint", "teeth", "mouth")):
			side = min(tex_max, 32)
		img, kv, rel = load_skin(vpks, cds, tname, max_side=max(side, 256))
		flags = 0
		if img is None:
			print(f"  {name}: skin {tname} not found, grey stand-in")
			w = h = 8
			rgba = bytes([128, 128, 128, 255]) * 64
		else:
			w, h, rgba = img.width, img.height, img.rgba
		if kv.get("$alphatest") == "1" or kv.get("$translucent") == "1":
			flags |= 0x0040  # STUDIO_NF_MASKED
		if "glint" in low or kv.get("$additive") == "1":
			flags |= 0x0020  # STUDIO_NF_ADDITIVE
		nw, nh = pow2_fit(w, h, side)
		rgba = resize_rgba(rgba, w, h, nw, nh)
		idx, pal = palettise(rgba, nw, nh, bool(flags & 0x0040))
		textures.append((tname[-63:], flags, nw, nh, idx, pal))
		print(f"  skin {tname}: {nw}x{nh} from {rel}")

	nskinref, nfam, skinidx = struct.unpack_from("<iii", mdl, 220)
	skin_table = list(struct.unpack_from(f"<{nskinref * nfam}h", mdl, skinidx)) if nskinref * nfam else []

	# ---- per material strips
	by_mat: dict[int, list] = {}
	for m, a, b, c in tris:
		tri = (vmap[a], vmap[b], vmap[c])
		if flip:
			tri = (tri[0], tri[2], tri[1])
		by_mat.setdefault(m, []).append(tri)

	# ================================================= layout
	buf = bytearray(244)

	def align():
		while len(buf) % 4:
			buf.append(0)

	def here():
		align()
		return len(buf)

	# bones
	q_bind = [ss.quat_angles(b.quat) for b in bones]
	# channel defaults + scales across every frame of every sequence
	values = [list(b.pos) + list(q_bind[i]) for i, b in enumerate(bones)]
	chan = []  # per seq: [bone][6] list of floats
	for (label, act, flags, fps, frames) in seq_out:
		per = [[[] for _ in range(6)] for _ in bones]
		for fr in frames:
			for bi, (p, q) in enumerate(fr):
				e = ss.quat_angles(q)
				for j in range(3):
					per[bi][j].append(p[j] - values[bi][j])
					d = e[j] - values[bi][3 + j]
					d = (d + math.pi) % (2 * math.pi) - math.pi
					per[bi][3 + j].append(d)
		chan.append(per)
	scales = []
	for bi in range(len(bones)):
		sc = []
		for j in range(6):
			m = max((abs(x) for per in chan for x in per[bi][j]), default=0.0)
			sc.append(m / 32000.0 if m > 1e-6 else (1.0 / 32.0 if j < 3 else 1.0 / 4096.0))
		scales.append(sc)

	boneindex = here()
	for i, b in enumerate(bones):
		buf += struct.pack(
			"<32sii6i6f6f", b.name.encode("latin-1")[:31], b.parent, 0,
			*([-1] * 6), *values[i], *scales[i],
		)

	# hitbox: one box on the root bone so traces have something to hit
	hitboxindex = here()
	buf += struct.pack("<ii3f3f", 0, 0, *bbmin, *bbmax)

	# animation blocks (seqgroup 0 = this file)
	anim_ofs = []
	for si, (label, act, flags, fps, frames) in enumerate(seq_out):
		per = chan[si]
		looping = bool(flags & 1)
		start = here()
		anim_ofs.append(start)
		buf += bytes(len(bones) * 12)
		for bi in range(len(bones)):
			offs = [0] * 6
			for j in range(6):
				q = [int(round(x / scales[bi][j])) for x in per[bi][j]]
				q = [max(-32767, min(32767, v)) for v in q]
				if not any(q):
					continue
				q.append(q[0] if looping else q[-1])  # k+1 read past the end
				rel = len(buf) - (start + bi * 12)
				if rel > 0xFFFF:
					raise SystemExit(f"{name}: {label} anim data > 64 KB, lower MAX_FPS")
				offs[j] = rel
				buf += encode_rle(q)
				while len(buf) % 2:
					buf.append(0)
			struct.pack_into("<6H", buf, start + bi * 12, *offs)

	seqindex = here()
	for si, (label, act, flags, fps, frames) in enumerate(seq_out):
		lm = motions[si]
		motiontype = 0x0001 | 0x0002 if any(lm) else 0  # STUDIO_X | STUDIO_Y
		buf += struct.pack(
			"<32sf10i3f2i3f3f2i2i2f2f4B3i4Bi",
			label.encode()[:31], fps, flags, act, 1 if act else 0, 0, 0, len(frames),
			0, 0, motiontype, 0, *lm, 0, 0, *bbmin, *bbmax,
			1, anim_ofs[si], 0, 0, 0.0, 0.0, 1.0, 1.0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		)

	seqgroupindex = here()
	buf += struct.pack("<32s64sii", b"default", b"", 0, 0)

	# bodypart + model + mesh + verts
	bodypartindex = here()
	buf += struct.pack("<64siii", b"body", 1, 0, 0)
	modelindex = here()
	struct.pack_into("<i", buf, bodypartindex + 72, modelindex)  # absolute, unlike Source
	buf += bytes(112)
	# GoldSrc lights normals mesh by mesh (each mesh owns a contiguous run of
	# the normal array; tricmd normindex is global), so normals are per mesh.
	mesh_norms = []  # per material: (list of vertex ids in normal order, map)
	for m in sorted(by_mat):
		order, nm = [], {}
		for t in by_mat[m]:
			for vi in t:
				if vi not in nm:
					nm[vi] = len(order)
					order.append(vi)
		mesh_norms.append((order, nm))
	vertinfoindex = here()
	buf += bytes(vbone)
	norminfoindex = here()
	for order, _nm in mesh_norms:
		buf += bytes(vbone[vi] for vi in order)
	vertindex = here()
	for v in verts:
		buf += struct.pack("<3f", *v)
	normindex = here()
	nnorms = 0
	for order, _nm in mesh_norms:
		for vi in order:
			buf += struct.pack("<3f", *norms[vi])
		nnorms += len(order)
	meshes = []
	ntris_total = 0
	nstrips_total = 0
	nbase = 0
	for mi, m in enumerate(sorted(by_mat)):
		order, nm = mesh_norms[mi]
		tex = skin_table[m] if m < len(skin_table) else m
		tw, th = (textures[tex][2], textures[tex][3]) if tex < len(textures) else (8, 8)
		strips = stripify(by_mat[m])
		cmds = bytearray()
		for s in strips:
			cmds += struct.pack("<h", len(s))
			for vi in s:
				u, v = uvs[vi]
				cmds += struct.pack("<4h", vi, nbase + nm[vi], int(round(u * tw)), int(round(v * th)))
		cmds += struct.pack("<h", 0)
		triindex = here()
		buf += cmds
		meshes.append((len(by_mat[m]), triindex, m, len(order)))
		nbase += len(order)
		ntris_total += len(by_mat[m])
		nstrips_total += len(strips)
	meshindex = here()
	for nt, ti, m, nn in meshes:
		buf += struct.pack("<5i", nt, ti, m, nn, 0)
	struct.pack_into(
		"<64sif10i", buf, modelindex, name.encode()[:63], 0, 0.0, len(meshes), meshindex,
		len(verts), vertinfoindex, vertindex, nnorms, norminfoindex, normindex, 0, 0,
	)

	# Skins go in a companion <name>t.mdl (GoldSrc $externaltextures), the
	# layout DoD and HL1 NPCs ship on GameCube: the base file stays small in
	# the studio arena and the GX loader merges the skin headers and uploads
	# the pixels from the companion (mod_studio.c G376 / deferred skins).
	nref = nskinref if skin_table else len(textures)
	fam_rows = max(1, nfam)
	tbuf = bytearray(244)

	def talign():
		while len(tbuf) % 4:
			tbuf.append(0)

	talign()
	t_textureindex = len(tbuf)
	tbuf += bytes(80 * len(textures))
	talign()
	t_skinindex = len(tbuf)
	for v in (skin_table if skin_table else range(len(textures))):
		tbuf += struct.pack("<h", v)
	talign()
	t_texturedataindex = len(tbuf)
	for t, (tname, flags, w, h, idx, pal) in enumerate(textures):
		talign()
		off = len(tbuf)
		tbuf += idx + pal
		struct.pack_into("<64sIiii", tbuf, t_textureindex + t * 80, tname.encode("latin-1"), flags, w, h, off)
	talign()
	tbuf[0:244] = struct.pack(
		"<4si64si3f3f3f3f3fi26i",
		b"IDST", 10, f"lc/{name}t.mdl".encode(), len(tbuf), *([0.0] * 15), 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		len(textures), t_textureindex, t_texturedataindex, nref, fam_rows, t_skinindex,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	)

	align()
	eye = src.eye
	hdr = struct.pack(
		"<4si64si3f3f3f3f3fi26i",
		b"IDST", 10, f"lc/{name}.mdl".encode(), len(buf), *eye,
		*src.hull_min, *src.hull_max, *bbmin, *bbmax, 0,
		len(bones), boneindex, 0, boneindex, 1, hitboxindex,
		len(seq_out), seqindex, 1, seqgroupindex,
		0, 0, 0, 0, 0, 0,
		1, bodypartindex, 0, 0, 0, 0, 0, 0, 0, 0,
	)
	buf[0:244] = hdr
	out = out_dir / "models" / "lc" / f"{name}.mdl"
	out.parent.mkdir(parents=True, exist_ok=True)
	out.write_bytes(bytes(buf))
	(out_dir / "models" / "lc" / f"{name}t.mdl").write_bytes(bytes(tbuf))
	print(
		f"  {name}: {len(buf)//1024} KB + skins {len(tbuf)//1024} KB bones={len(bones)} verts={len(verts)} "
		f"tris={ntris_total} strips={nstrips_total} lod={used_lod} seqs={[s[0] for s in seq_out]}"
	)
	return out


# ---------------------------------------------------------------- entities

def _map_entities(src_bsp: Path) -> list[dict]:
	import re
	d = src_bsp.read_bytes()
	off, ln = struct.unpack_from("<ii", d, 8)
	text = d[off:off + ln].decode("latin-1", "replace")
	return [dict(re.findall(r'"([^"]+)"\s+"([^"]*)"', b)) for b in re.findall(r"\{[^{}]*\}", text)]


def npc_spawns(src_bsp: Path, have: set[str]) -> list[dict]:
	"""Where the map puts its characters when the level starts.

	Template NPCs (named by an npc_template_maker or a point_template) do not
	exist at start and are left out. The fisherman waits at ss_Fisherman_Intro in Intro_preIdle.
	"""
	ents = _map_entities(src_bsp)
	templates = {e.get("TemplateName", "").lower() for e in ents if e.get("classname") == "npc_template_maker"}
	# point_template removes the entities it names when the map loads (unless
	# spawnflag 1, "don't remove template entities") and spawns copies later
	# on a trigger. The cliff and monastery squads are all templates: at the
	# start of the level only soldier_cliff_enticement stands in the map.
	for e in ents:
		if e.get("classname") == "point_template" and not int(e.get("spawnflags", "0") or 0) & 1:
			templates |= {v.lower() for k, v in e.items() if k.lower().startswith("template") and v}
	out = []
	for e in ents:
		cls = e.get("classname", "")
		name = e.get("targetname", "").lower()
		if name and name in templates:
			continue
		model = e.get("model", "").lower().replace("\\", "/")
		lc = None
		seq = "idle"
		org, ang = e.get("origin", "0 0 0"), e.get("angles", "0 0 0")
		if cls == "npc_combine_s":
			lc = "combine_super_soldier" if "super" in model else "combine_soldier"
		elif cls == "npc_seagull":
			lc = "seagull"
		elif cls == "scripted_sequence" and e.get("targetname", "").lower() == "ss_fisherman_intro":
			lc, seq = "fisherman", "intro_preidle"
		if lc and lc in have:
			out.append({"model": lc, "origin": org, "angles": ang, "seq": seq})
	return out


def seq_index(name: str, label: str) -> int:
	labels = [s[0] for s in MODELS[name][2]]
	return labels.index(label) if label in labels else 0


def patch_stub_bsp(stub: Path, spawns: list[dict]) -> None:
	"""Append monster_generic entities to the GoldSrc stub BSP's entity lump.

	monster_generic, not cycler: the GX world route soft-skips cycler as a
	decorative entity (sv_game.c SV_GCCapFacesOptionalEntity). A generic
	monster idles on its ACT_IDLE sequence through the HLSDK animation code.
	"""
	d = bytearray(stub.read_bytes())
	off, ln = struct.unpack_from("<ii", d, 4)
	text = d[off:off + ln].split(b"\0")[0].decode("latin-1")
	# drop entities this tool added before (re-bake is idempotent)
	import re
	text = re.sub(r'\{[^{}]*"lc_studio" "1"[^{}]*\}\n?', "", text)
	for sp in spawns:
		text += (
			"{\n\"classname\" \"monster_generic\"\n\"lc_studio\" \"1\"\n"
			f"\"model\" \"models/lc/{sp['model']}.mdl\"\n"
			f"\"origin\" \"{sp['origin']}\"\n\"angles\" \"{sp['angles']}\"\n"
			f"\"sequence\" \"{seq_index(sp['model'], sp['seq'])}\"\n"
			# the GC generic monster does not think; the client advances the
			# frame from animtime * framerate (R_StudioEstimateFrame)
			"\"framerate\" \"1\"\n}\n"
		)
	blob = text.encode("latin-1") + b"\0"
	while len(d) % 4:
		d.append(0)
	new_off = len(d)
	d += blob
	struct.pack_into("<ii", d, 4, new_off, len(blob))
	stub.write_bytes(bytes(d))
	print(f"  stub bsp: {len(spawns)} studio NPC monster_generic -> {stub}")


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--lostcoast", required=True)
	ap.add_argument("--hl2", required=True)
	ap.add_argument("--out", required=True, help="pack root (models/lc/*.mdl)")
	ap.add_argument("--models", default="fisherman,combine_soldier,combine_super_soldier,seagull")
	ap.add_argument("--lod", type=int, default=-1, help="override DEFAULT_LOD")
	ap.add_argument("--tex-max", type=int, default=128, help="GX decodes world-model skins to <=128 anyway")
	ap.add_argument("--flip", action="store_true", help="reverse triangle winding")
	ap.add_argument("--map", type=Path, help="Source d2_lostcoast.bsp (NPC placements)")
	ap.add_argument("--stub-bsp", type=Path, help="baked GoldSrc stub BSP to receive cycler entities")
	args = ap.parse_args()
	vpks = open_vpk_roots(
		sorted(Path(args.lostcoast).glob("*_dir.vpk")) + sorted(Path(args.hl2).glob("*_dir.vpk"))
	)
	for name in args.models.split(","):
		name = name.strip()
		lod = args.lod if args.lod >= 0 else DEFAULT_LOD.get(name, 1)
		convert(vpks, name, lod, args.tex_max, Path(args.out), args.flip)
	if args.map and args.stub_bsp:
		have = {n.strip() for n in args.models.split(",")}
		patch_stub_bsp(args.stub_bsp, npc_spawns(args.map, have))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
