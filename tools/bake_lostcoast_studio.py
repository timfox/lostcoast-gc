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
import dataclasses
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import source_studio as ss  # noqa: E402
from vpk import open_vpk_roots, read_first  # noqa: E402
from vtf import decode_vtf  # noqa: E402
from bake_worlddl import LIGHT_EXPOSURE  # noqa: E402

# HL1 activity ids (dlls/activity.h) for sequences the server looks up.
ACT = {"idle": 1, "walk": 3, "run": 4, "fly": 5, "hop": 7, "land": 10}

# name, source model, anim source models, [(goldsrc label, source seq, act)]
MODELS = {
	"fisherman": (
		"models/lostcoast/fisherman/fisherman",
		["models/lostcoast/fisherman/Fisherman_animations"],
		[
			# He carries weapon_oldmanharpoon, so npc_fisherman's ACT_IDLE
			# resolves to ACT_IDLE_SUITCASE = Idle01 (Idle01_NH is the
			# no-harpoon idle) and ACT_WALK to walk_all_Moderate.
			("idle", "Idle01", "idle"),
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
# NPC weapons (additionalequipment) whose world model's root bone is one of
# the character's own bones: Source bone-merges them, so they bake straight
# into the character's actor mesh on that bone (one draw, no extra entity).
EQUIPMENT = {"fisherman": [("models/lostcoast/fisherman/harpoon", None, 0x08)],
	# additionalequipment: 18 of the map's 23 soldiers carry weapon_smg1
	# (3 a shotgun, 2 the AR2, the elite's among them); one mesh a model.
	"combine_soldier": [("models/weapons/w_smg1", None, 0x08)],
	"combine_super_soldier": [("models/weapons/w_irifle", None, 0x08)]}
# Props the map parents to one of the character's attachments and turns on
# for a while (prop_dynamic keys on anim_attachment_RH, shown while he
# unlocks the gate): baked the same way on the attachment's bone, drawn only
# while the game says so (BATCH_HELD).
HELD = {"fisherman": [("models/lostcoast/fisherman/keys", "anim_attachment_RH", 0x10)]}

# Characters that look at things through Source's head/spine/body pose params.
LOOK_MODELS = {"fisherman"}
LOOK_RANGES: dict[str, str] = {}  # name -> "pose_lo pose_hi deg_lo deg_hi" per controller

DEFAULT_LOD = {"fisherman": 5, "combine_soldier": 3, "combine_super_soldier": 4, "seagull": 1}

MAX_FPS = 15.0  # GC: sample animation at 15 Hz, r_studio interpolates frames

# Characters drawn by Flipper itself (r_gx_actor.c): skinned on the matrix
# palette from display lists, lit by the transform unit. The CPU only builds
# bone matrices, so they can carry the model's own finer LODs: (near, far)
# Source LODs, switched on the model's own VTX switch points. The MDL keeps
# skeleton, sequences and skins; its mesh moves to <name>.gca. Flipper's
# cost per triangle is small next to the CPU's per-bone work, so one fine
# LOD serves every distance until the studio arena has room for a second.
ACTOR_LODS = {"fisherman": (3,), "combine_soldier": (3,)}


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


# ---------------------------------------------------------------- GX actor

GCA_MAGIC = 0x47436163  # 'GCac'
GCA_VERSION = 2          # 8-bit indices into per-batch array windows
GCA_VTXFMT = 5           # r_gx_actor.c: PNMTXIDX, POS/NRM/TEX0 index16
GX_TRIANGLES = 0x90
GX_TRISTRIP = 0x98
MAX_PALETTE = 10         # Flipper position/normal matrix slots (PNMTX0..9)
BATCH_HALFLAMBERT = 0x01
BATCH_MASKED = 0x02
BATCH_ADDITIVE = 0x04
BATCH_EQUIPMENT = 0x08  # a weapon merged onto the character: not drawn once body bit 0 is set
BATCH_HELD = 0x10       # a prop in his hand for a while: drawn only while body bit 1 is set


def vtx_switch_points(vtx: bytes) -> list[float]:
	_v, _vc, _a, _b, _c, _chk, _nl, _mr, nbp, bpo = struct.unpack_from("<iiHHiiiiii", vtx, 0)
	_nm, moff = struct.unpack_from("<ii", vtx, bpo)
	mh = bpo + moff
	nlods, lodoff = struct.unpack_from("<ii", vtx, mh)
	return [struct.unpack_from("<iif", vtx, mh + lodoff + i * 12)[2] for i in range(nlods)]


def _texture_names(mdl: bytes) -> list[str]:
	numtex, texidx = struct.unpack_from("<ii", mdl, 204)
	return [_cstr(mdl, texidx + t * 64 + struct.unpack_from("<i", mdl, texidx + t * 64)[0]) for t in range(numtex)]


def eye_materials(mdl: bytes) -> set[int]:
	return {i for i, n in enumerate(_texture_names(mdl)) if n.lower().startswith("eyeball")}


def _eyeballs(mdl: bytes) -> list[dict]:
	"""mstudioeyeball_t of body part 0, model 0: bone-space origin, up, forward."""
	_nbp, bpidx = struct.unpack_from("<ii", mdl, 232)
	mh = bpidx + struct.unpack_from("<i", mdl, bpidx + 12)[0]
	ne, ei = struct.unpack_from("<ii", mdl, mh + 100)
	out = []
	for k in range(ne):
		o = mh + ei + k * 172
		out.append({
			"bone": struct.unpack_from("<i", mdl, o + 4)[0],
			"org": struct.unpack_from("<3f", mdl, o + 8),
			"up": struct.unpack_from("<3f", mdl, o + 28),
			"forward": struct.unpack_from("<3f", mdl, o + 40),
			"iris_scale": struct.unpack_from("<f", mdl, o + 60)[0],
		})
	return out


def eye_texture(mdl, vvd, vtx, bones, material: int, w: int, h: int, rgba: bytes, iris) -> bytes:
	"""The "eyes" shader on one texture: $iris projected onto the sclera.

	Source projects the iris from the eyeball's own frame (studiorender
	R_StudioEyeballPosition): u, v = 0.5 -+ (p - org) . right, up * iris_scale
	on the half of the ball that faces forward. Flipper draws the eyeball with
	one texture, so the projection is done here, texel by texel, through the
	eyeball mesh's own texture coordinates. The eyes look straight ahead.
	"""
	tris, vstart, _ = load_mesh(mdl, vvd, vtx, 0)
	tris = [t for t in tris if t[0] == material]
	balls = _eyeballs(mdl)
	if not tris or not balls or iris is None:
		return rgba
	vert = {}
	for t in tris:
		for v in t[1:]:
			if v not in vert:
				_w, _b, _nb, p, _n, uv = ss.vvd_vertex(vvd, vstart, v)
				vert[v] = (p, uv)

	def local(ball, p):
		return ss.mat34_xform(bones[ball["bone"]].pose_to_bone, p)

	cen = [sum(vert[v][0][a] for v in vert) / len(vert) for a in range(3)]
	ball = min(balls, key=lambda e: math.dist(local(e, cen), e["org"]))
	f, up = ball["forward"], ball["up"]
	# The mesh is only the cap that shows; the record's forward points into
	# the head on this model, so face the projection the way the cap faces.
	lc = local(ball, cen)
	if sum((lc[i] - ball["org"][i]) * f[i] for i in range(3)) < 0.0:
		f = tuple(-x for x in f)
	right = (f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0])
	out = bytearray(rgba)
	for _m, a, b, c in tris:
		(pa, ta), (pb, tb), (pc, tc) = vert[a], vert[b], vert[c]
		xs = [t[0] * w for t in (ta, tb, tc)]
		ys = [t[1] * h for t in (ta, tb, tc)]
		den = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
		if abs(den) < 1e-9:
			continue
		for y in range(max(0, int(min(ys))), min(h, int(max(ys)) + 1)):
			for x in range(max(0, int(min(xs))), min(w, int(max(xs)) + 1)):
				px, py = x + 0.5, y + 0.5
				l0 = ((ys[1] - ys[2]) * (px - xs[2]) + (xs[2] - xs[1]) * (py - ys[2])) / den
				l1 = ((ys[2] - ys[0]) * (px - xs[2]) + (xs[0] - xs[2]) * (py - ys[2])) / den
				l2 = 1.0 - l0 - l1
				if min(l0, l1, l2) < -0.02:
					continue
				p = local(ball, tuple(pa[i] * l0 + pb[i] * l1 + pc[i] * l2 for i in range(3)))
				d = tuple(p[i] - ball["org"][i] for i in range(3))
				if sum(d[i] * f[i] for i in range(3)) <= 0.0:
					continue
				iu = 0.5 - sum(d[i] * right[i] for i in range(3)) * ball["iris_scale"]
				iv = 0.5 - sum(d[i] * up[i] for i in range(3)) * ball["iris_scale"]
				if not (0.0 <= iu < 1.0 and 0.0 <= iv < 1.0):
					continue
				io = (int(iv * iris.height) * iris.width + int(iu * iris.width)) * 4
				al = iris.rgba[io + 3] / 255.0
				o = (y * w + x) * 4
				for ch in range(3):
					out[o + ch] = int(out[o + ch] * (1.0 - al) + iris.rgba[io + ch] * al + 0.5)
	return bytes(out)


def lod_geometry(mdl, vvd, vtx, lod, bones):
	"""Tris (material, a, b, c) over local verts; verts in their bone's space."""
	tris, vstart, used = load_mesh(mdl, vvd, vtx, lod)
	# Source drops the eyeball meshes from LOD 3 down and leaves the dark
	# sockets painted on the head. The actor is one LOD for every distance,
	# so it keeps LOD 0's eyeballs (32 triangles each).
	eyes = eye_materials(mdl)
	if eyes and not any(t[0] in eyes for t in tris):
		tris = tris + [t for t in load_mesh(mdl, vvd, vtx, 0)[0] if t[0] in eyes]
	# The same LODs drop the teeth and the inside of the mouth. He opens his
	# mouth to speak now, so they come back from the coarsest LOD that still
	# has them (LOD 2: 242 + 77 triangles).
	mouth = {i for i, n in enumerate(_texture_names(mdl)) if n.lower() in ("teeth", "mouth")}
	if mouth and not any(t[0] in mouth for t in tris):
		for finer in range(lod - 1, -1, -1):
			extra = [t for t in load_mesh(mdl, vvd, vtx, finer)[0] if t[0] in mouth]
			if extra:
				tris = tris + extra
				break
	ids = sorted({v for t in tris for v in t[1:]})
	vmap = {v: i for i, v in enumerate(ids)}
	pos, nrm, uv, vb = [], [], [], []
	for vid in ids:
		w, b, nb, p, n, t = ss.vvd_vertex(vvd, vstart, vid)
		k = max(range(max(1, nb)), key=lambda j: w[j])
		bi = b[k] if 0 <= b[k] < len(bones) else 0
		p2b = bones[bi].pose_to_bone
		pos.append(ss.mat34_xform(p2b, p))
		nrm.append(ss.mat34_rotate(p2b, n))
		uv.append(t)
		vb.append(bi)
	ltris = [(m, vmap[a], vmap[b2], vmap[c]) for m, a, b2, c in tris]
	return ltris, pos, nrm, uv, vb, used, ids


MAX_BATCH_VERTS = 256   # 8-bit vertex indices, one array window per batch


def _palette_batches(tris, vb):
	"""Split triangles into runs that touch at most MAX_PALETTE bones and
	MAX_BATCH_VERTS vertices (so the display list can use 8-bit indices)."""
	order = sorted(tris, key=lambda t: tuple(sorted({vb[t[0]], vb[t[1]], vb[t[2]]})))
	batches, cur, bones, verts = [], [], [], set()
	for t in order:
		need = [b for b in dict.fromkeys((vb[t[0]], vb[t[1]], vb[t[2]])) if b not in bones]
		newv = {v for v in t if v not in verts}
		if len(bones) + len(need) > MAX_PALETTE or len(verts) + len(newv) > MAX_BATCH_VERTS:
			batches.append((bones, cur))
			cur, bones, verts = [], [], set()
			need = list(dict.fromkeys((vb[t[0]], vb[t[1]], vb[t[2]])))
			newv = set(t)
		bones += need
		verts |= newv
		cur.append(t)
	if cur:
		batches.append((bones, cur))
	return batches


def _gamma_half(x: float) -> float:
	"""Linear light -> the world bake's texel (gamma(L * exposure) / 2), 0..1."""
	if x <= 0.0:
		return 0.0
	return min(1.0, 0.5 * (x * LIGHT_EXPOSURE) ** (1.0 / 2.2))


def _fit_curve(f) -> tuple[float, float, float]:
	"""Least-squares a0 + a1 t + a2 t^2 to f on t in [0, 1] (Flipper's spot/spec
	attenuation polynomial), with a0 pinned to f(0)."""
	a0 = f(0.0)
	ts = [i / 32.0 for i in range(33)]
	# solve for a1, a2: minimise sum (a1 t + a2 t^2 - (f(t) - a0))^2
	s11 = sum(t * t for t in ts)
	s12 = sum(t ** 3 for t in ts)
	s22 = sum(t ** 4 for t in ts)
	r1 = sum(t * (f(t) - a0) for t in ts)
	r2 = sum(t * t * (f(t) - a0) for t in ts)
	det = s11 * s22 - s12 * s12
	a1 = (r1 * s22 - r2 * s12) / det
	a2 = (s11 * r2 - s12 * r1) / det
	return (a0, a1, a2)


def placement_light(cube, sun):
	"""Ambient cube and sun at one spot -> Flipper light registers.

	Each cube face is one light with attenuation t^2 (t = N.axis, t >= 0),
	which is exactly Source's n^2-weighted ambient cube. The sun is one more
	light whose attenuation polynomial is fitted to the gamma-encoded curve
	gamma(amb + sun t) - gamma(amb), so the lit side ramps like the PC render
	instead of summing two gamma-encoded terms. Everything is halved for the
	TEV x2 stage, as the world lightmaps are.
	"""
	faces = [tuple(_gamma_half(c) for c in face) for face in cube]
	amb = [sum(face[c] for face in cube) / 6.0 for c in range(3)]
	if sun is None:
		return faces, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
	top = [_gamma_half(amb[c] + sun[c]) - _gamma_half(amb[c]) for c in range(3)]
	lum = [0.3 * amb[0] + 0.59 * amb[1] + 0.11 * amb[2], 0.3 * sun[0] + 0.59 * sun[1] + 0.11 * sun[2]]
	g0 = _gamma_half(lum[0])
	g1 = _gamma_half(lum[0] + lum[1]) - g0 or 1.0

	def lam(t):
		return (_gamma_half(lum[0] + lum[1] * t) - g0) / g1

	def half(t):
		return lam((0.5 * t + 0.5) ** 2)

	return faces, tuple(top), _fit_curve(lam), _fit_curve(half)


def write_face(face, out_path: Path, order, ids, vb, bones, pf: int) -> None:
	"""Mouth shapes for lip sync: <name>.gcf beside the actor mesh.

	face = (Flexes, phoneme table). Every (flex descriptor, ramp) the phonemes
	reach becomes one shape: the offsets it gives the actor's vertices, in
	each vertex's bone space and in the position array's own fixed point.
	A phoneme is one weight per shape. The engine blends the weights of the
	phoneme being spoken and the next, and adds the shapes to the position
	array (r_gx_face.c).

	  0   "GCFA", u16 version (2), u16 nshape
	  8   u16 nslot, u16 nphon, u8 shift, 3 x 0
	  16  u32 slots_ofs, u32 dir_ofs, u32 phon_ofs, u32 size
	  slots: nslot x u16: index into LOD 0's position array
	  dir:   nshape x 8: u32 ofs, u16 count, u16 0
	  entries: count x 6: u16 slot table index, s8 dx, dy, dz (position
	         units >> shift), u8 0
	  phon:  nphon x (u16 code, nshape x s8 (weight - resting weight) * 16),
	         padded to even
	"""
	fx, phon = face
	# shapes: (descriptor that weighs it, ramp) -> {vvd id: [dx, dy, dz]} in model space
	shapes: dict[tuple, dict[int, list[float]]] = {}
	for fd, t, pair, va in fx.flexes:
		for vid, speed, side, d in va:
			# ComputeFlexedVertex: (w1 * (1 - side) + side * w2) * speed, w2 the pair's weight
			parts = [(fd, (1.0 - side) * speed), (pair, side * speed)] if pair else [(fd, speed)]
			for desc, k in parts:
				if k <= 0.0:
					continue
				o = shapes.setdefault((desc, t), {}).setdefault(vid, [0.0, 0.0, 0.0])
				for a in range(3):
					o[a] += d[a] * k
	weights = {code: fx.weights(vals) for code, (_n, vals) in phon.items()}
	# Scenes drive the other controllers too (brows, eyelids, cheeks: the
	# flex animation tracks, bake_lostcoast_actors.py), so a shape any single
	# controller reaches at full strength is kept as well.
	solo = [fx.weights({name: 1.0}) for name, _mn, _mx in fx.controllers]
	reach = list(weights.values()) + solo
	# The rules give some shapes weight with every controller at zero (the
	# eyelids' neutral shapes). The mesh as it stands is that face, so a
	# shape counts by how far it is from that resting weight, either way.
	rest = fx.weights({})

	def rel(k, w):
		return fx.ramp(k[1], w[k[0]]) - fx.ramp(k[1], rest[k[0]])

	keys = [k for k in shapes if any(abs(rel(k, w)) > 0.004 for w in reach)]
	peak = {k: max(abs(rel(k, w)) for w in reach) for k in keys}
	slots_of: dict[int, list[int]] = {}
	local_of = {vid: i for i, vid in enumerate(ids)}
	for si, v in enumerate(order):
		slots_of.setdefault(v, []).append(si)
	scale = float(1 << pf)
	raw = []  # per shape: [(slot, (dx, dy, dz) in position units)]
	biggest = 0.0
	for k in keys:
		ent = []
		for vid, d in shapes[k].items():
			li = local_of.get(vid)
			if li is None or li not in slots_of:
				continue
			if math.sqrt(sum(c * c for c in d)) * peak[k] < 0.03:
				continue  # under a 30th of a unit at this shape's strongest phoneme
			bd = ss.mat34_rotate(bones[vb[li]].pose_to_bone, d)
			fixed = tuple(c * scale for c in bd)
			biggest = max(biggest, max(abs(c) for c in fixed))
			for si in slots_of[li]:
				ent.append((si, fixed))
		raw.append(ent)
	keep = [i for i, e in enumerate(raw) if e]
	keys = [keys[i] for i in keep]
	raw = [raw[i] for i in keep]
	if not keys or len(keys) > 255:
		print(f"  face: no usable shapes ({len(keys)})")
		return
	shift = 0
	while biggest / (1 << shift) > 127.0:
		shift += 1
	slot_list = sorted({si for e in raw for si, _d in e})
	slot_ix = {si: i for i, si in enumerate(slot_list)}
	slots_ofs = 32
	dir_ofs = slots_ofs + len(slot_list) * 2
	body_ofs = dir_ofs + len(keys) * 8
	body = bytearray()
	dirb = bytearray()
	for e in raw:
		dirb += struct.pack(">IHH", body_ofs + len(body), len(e), 0)
		for si, d in sorted(e):
			body += struct.pack(">H3bB", slot_ix[si], *(max(-127, min(127, int(round(c / (1 << shift))))) for c in d), 0)
	phon_ofs = body_ofs + len(body)
	ph = bytearray()
	for code in sorted(weights):
		ph += struct.pack(">H", code)
		ph += struct.pack(f">{len(keys)}b", *(max(-127, min(127, int(round(rel(k, weights[code]) * 16.0)))) for k in keys))
		if len(ph) & 1:
			ph += b"\0"
	size = phon_ofs + len(ph)
	hdr = struct.pack(">4sHHHHB3xIIII", b"GCFA", 2, len(keys), len(slot_list), len(weights), shift, slots_ofs, dir_ofs, phon_ofs, size)
	out_path.write_bytes(hdr + b"".join(struct.pack(">H", si) for si in slot_list) + bytes(dirb) + bytes(body) + bytes(ph))
	# the shapes in file order, for the scene bake to weigh its face tracks by
	import json
	out_path.with_suffix(".gcf.json").write_text(json.dumps([[k[0], list(k[1])] for k in keys]))
	print(f"  {out_path.name}: {len(keys)} shapes, {len(slot_list)} vertex slots, {sum(len(e) for e in raw)} offsets, "
		f"{len(weights)} phonemes, shift {shift}, {size} bytes")


def write_actor(name, mdl, vvd, vtx, bones, mat_flags, out_path: Path, placements, extras=(), face=None):
	sw = vtx_switch_points(vtx)
	lods = []
	for li, lod in enumerate(ACTOR_LODS[name]):
		tris, pos, nrm, uv, vb, used, ids = lod_geometry(mdl, vvd, vtx, lod, bones)
		for etris, epos, enrm, euv, evb in extras:
			base = len(pos)
			tris = tris + [(m, a + base, b + base, c + base) for m, a, b, c in etris]
			pos, nrm, uv, vb = pos + epos, nrm + enrm, uv + euv, vb + evb
		# Source picks LOD i once the metric passes LOD i's switch point; the
		# near LOD stands in for every finer one.
		switch = 0.0 if li == 0 else max(0.0, sw[used] if used < len(sw) else 0.0)
		lods.append((switch, tris, pos, nrm, uv, vb, used, ids))

	buf = bytearray(64)

	def pad32():
		buf.extend(b"\0" * ((-len(buf)) % 32))

	lod_ofs = []
	stats = []
	for switch, tris, pos, nrm, uv, vb, used, ids in lods:
		pad32()
		lh = len(buf)
		lod_ofs.append(lh)
		buf += bytes(32)
		m = max((abs(c) for p in pos for c in p), default=1.0)
		pf = 0
		while pf < 14 and m * (1 << (pf + 1)) < 32000:
			pf += 1
		mu = max((abs(c) for t in uv for c in t), default=1.0)
		# u8 texcoords with 7 fraction bits cover [0, 2) at 1/128: one
		# texel of the 128px skins, at half the bytes of s16.
		tex_u8 = min((c for t in uv for c in t), default=0.0) >= 0.0 and mu < 255.0 / 128.0
		tf = 7 if tex_u8 else 0
		while not tex_u8 and tf < 14 and mu * (1 << (tf + 1)) < 32000:
			tf += 1
		# batches: per material, then per bone palette / 256-vertex window
		by_mat: dict[int, list] = {}
		for t in tris:
			by_mat.setdefault(t[0], []).append((t[1], t[2], t[3]))
		batch_list = []
		for mat in sorted(by_mat):
			flags = mat_flags.get(mat, 0)
			for bl, bt in _palette_batches(by_mat[mat], vb):
				batch_list.append((mat, flags, bl, bt))
		# each batch owns a contiguous run of the arrays (shared verts repeat)
		order, bases = [], []
		for mat, flags, bl, bt in batch_list:
			bases.append(len(order))
			order += list(dict.fromkeys(v for t in bt for v in t))
		pad32()
		pos_ofs = len(buf)
		for v in order:
			buf += struct.pack(">3h", *(max(-32767, min(32767, int(round(c * (1 << pf))))) for c in pos[v]))
		if face is not None and lh == lod_ofs[0]:
			write_face(face, out_path.with_suffix(".gcf"), order, ids, vb, bones, pf)
		pad32()
		nrm_ofs = len(buf)
		for v in order:
			n = nrm[v]
			ln = math.sqrt(sum(c * c for c in n)) or 1.0
			buf += struct.pack(">3b", *(max(-64, min(64, int(round(c / ln * 64)))) for c in n))
		pad32()
		tex_ofs = len(buf)
		for v in order:
			t = uv[v]
			if tex_u8:
				buf += bytes(max(0, min(255, int(round(c * 128.0)))) for c in t)
			else:
				buf += struct.pack(">2h", *(int(round(c * (1 << tf))) for c in t))
		pad32()
		batch_ofs = len(buf)
		buf += bytes(32 * len(batch_list))
		nstrips = 0
		for bi, (mat, flags, bl, bt) in enumerate(batch_list):
			slot = {b: k for k, b in enumerate(bl)}
			local = {v: i for i, v in enumerate(dict.fromkeys(v for t in bt for v in t))}
			strips = stripify(bt)
			pad32()
			dl0 = len(buf)

			def vert(v):
				i = local[v]
				return bytes((slot[vb[v]] * 3, i, i, i))

			loose = [s_ for s_ in strips if len(s_) == 3]
			for s_ in strips:
				if len(s_) == 3:
					continue
				buf += struct.pack(">BH", GX_TRISTRIP | GCA_VTXFMT, len(s_))
				for v in s_:
					buf += vert(v)
				nstrips += 1
			if loose:
				buf += struct.pack(">BH", GX_TRIANGLES | GCA_VTXFMT, 3 * len(loose))
				for s_ in loose:
					for v in s_:
						buf += vert(v)
			pad32()
			dl_size = len(buf) - dl0
			bones12 = bytes(bl) + bytes(12 - len(bl))
			struct.pack_into(">HBB12sIIH6x", buf, batch_ofs + bi * 32, mat, len(bl), flags, bones12,
				dl0, dl_size, bases[bi])
		struct.pack_into(">HHBBHIIIIB7x", buf, lh, len(order), len(batch_list), pf, tf, len(tris),
			pos_ofs, nrm_ofs, tex_ofs, batch_ofs, 1 if tex_u8 else 0)
		stats.append(f"lod{used}: {len(pos)} verts ({len(order)} in batch windows) {len(tris)} tris {len(batch_list)} batches {nstrips} strips")

	pad32()
	place_ofs = len(buf)
	for pl in placements:
		faces, sun, lam, half = placement_light(pl["cube"], pl["sun"])
		buf += struct.pack(">3f", *pl["origin"])
		for f in faces:
			buf += bytes(int(round(255 * c)) for c in f) + b"\xff"
		buf += bytes(int(round(255 * min(1.0, c))) for c in sun) + b"\xff"
		buf += struct.pack(">3f", *pl["sundir"])
		buf += struct.pack(">3f3f", *lam, *half)
		buf += bytes(4)
	pad32()
	switch = [l[0] for l in lods] + [0.0] * (4 - len(lods))
	struct.pack_into(">IHHHHI4f4II", buf, 0, GCA_MAGIC, GCA_VERSION, len(lods), len(bones), len(placements),
		len(buf), *switch, *(lod_ofs + [0] * (4 - len(lod_ofs))), place_ofs)
	out_path.write_bytes(bytes(buf))
	print(f"  {name}.gca: {len(buf) // 1024} KB, {'; '.join(stats)}, {len(placements)} placement(s)")


# ---------------------------------------------------------------- look

# Source's look-at pose parameters, in the order they become GoldSrc bone
# controllers 0..3 (lostcoast_npcs.cpp drives them the way
# CAI_BaseActor::UpdateHeadControl drives the pose parameters).
LOOK_POSES = ("head_yaw", "head_pitch", "spine_yaw", "body_yaw")
STUDIO_XR, STUDIO_YR, STUDIO_ZR = 0x08, 0x10, 0x20


def _pose_params(d: bytes) -> list[tuple[str, float, float]]:
	"""mstudioposeparamdesc_t table (studiohdr numlocalposeparameters @300)."""
	n, idx = struct.unpack_from("<ii", d, 300)
	out = []
	for i in range(max(0, min(n, 32))):
		o = idx + i * 20
		nm = _cstr(d, o + struct.unpack_from("<i", d, o)[0])
		_fl, st, en, _lp = struct.unpack_from("<ifff", d, o + 4)
		out.append((nm, st, en))
	return out


def look_controllers(anim_mdl: bytes, am, remap, bones) -> list[dict]:
	"""One bone controller per look pose: the bone and euler axis its delta
	sequence turns most, with the turn at each end of the pose range."""
	params = _pose_params(anim_mdl)
	names = [p[0] for p in params]
	ctrls = []
	for pose in LOOK_POSES:
		if pose not in names:
			continue
		pi = names.index(pose)
		seq = next((q for q in am.seqs if q.paramindex and q.paramindex[0] == pi and len(q.anims) == 3), None)
		if not seq:
			continue
		lo_a, mid_a, hi_a = (am.anims[i] for i in seq.anims)
		best = None
		for b, val in lo_a.frames[0].items():
			mb = remap[b] if 0 <= b < len(remap) else -1
			if mb < 0 or bones[mb].name.lower().endswith("forward"):
				continue
			lo = ss.quat_angles(val[1])
			hiv = hi_a.frames[0].get(b)
			hi = ss.quat_angles(hiv[1]) if hiv else tuple(-x for x in lo)
			for ax in range(3):
				span = abs(hi[ax] - lo[ax])
				if best is None or span > best[0]:
					best = (span, mb, ax, math.degrees(lo[ax]), math.degrees(hi[ax]))
		if not best or best[0] < math.radians(1.0):
			continue
		_span, mb, ax, dlo, dhi = best
		ctrls.append({"pose": pose, "bone": mb, "axis": ax, "pose_lo": params[pi][1], "pose_hi": params[pi][2],
			"deg_lo": dlo, "deg_hi": dhi})
	return ctrls


# ---------------------------------------------------------------- writer

def convert(vpks, name: str, lod: int, tex_max: int, out_dir: Path, flip: bool, placements=None) -> Path:
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

	look = []
	if name in LOOK_MODELS:
		for asrc, (am_, remap_) in anim_cache.items():
			look = look_controllers(read_first(vpks, asrc + ".mdl"), am_, remap_, bones)
			if look:
				break
		LOOK_RANGES[name] = " ".join(
			f"{c['pose_lo']:g} {c['pose_hi']:g} {c['deg_lo']:.2f} {c['deg_hi']:.2f}" for c in look)
		for k, c in enumerate(look):
			print(f"  look ctrl {k}: {c['pose']} -> {bones[c['bone']].name} axis {'xyz'[c['axis']]} "
				f"{c['deg_lo']:.1f}..{c['deg_hi']:.1f} deg over pose {c['pose_lo']:.1f}..{c['pose_hi']:.1f}")

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
	tex_halflambert = []
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
		tex_halflambert.append(kv.get("$halflambert") == "1" or kv.get("__shader") == "eyes")
		if kv.get("$alphatest") == "1" or kv.get("$translucent") == "1":
			flags |= 0x0040  # STUDIO_NF_MASKED
		if "glint" in low or kv.get("$additive") == "1":
			flags |= 0x0020  # STUDIO_NF_ADDITIVE
		nw, nh = pow2_fit(w, h, side)
		rgba = resize_rgba(rgba, w, h, nw, nh)
		if kv.get("__shader") == "eyes" and kv.get("$iris") and name in ACTOR_LODS:
			iris = read_first(vpks, "materials/" + kv["$iris"].replace("\\", "/").strip() + ".vtf")
			rgba = eye_texture(mdl, vvd, vtx, bones, t, nw, nh, rgba, decode_vtf(iris, max_side=64) if iris else None)
		idx, pal = palettise(rgba, nw, nh, bool(flags & 0x0040))
		textures.append((tname[-63:], flags, nw, nh, idx, pal))
		print(f"  skin {tname}: {nw}x{nh} from {rel}")

	nskinref, nfam, skinidx = struct.unpack_from("<iii", mdl, 220)
	skin_table = list(struct.unpack_from(f"<{nskinref * nfam}h", mdl, skinidx)) if nskinref * nfam else []

	actor = name in ACTOR_LODS
	extras = []
	equip_mats = {}  # materials of what he carries -> batch flag: the game can take it out of his hand
	for eq, attach, flag in (EQUIPMENT.get(name, []) + HELD.get(name, [])) if actor else []:
			emdl, evvd, evtx = (read_first(vpks, eq + x) for x in (".mdl", ".vvd", ".dx90.vtx"))
			esrc = ss.read_studio(emdl, decode_anims=False)
			names = {b.name.lower(): i for i, b in enumerate(bones)}
			bmap = [names.get(b.name.lower(), -1) for b in esrc.bones]
			if attach and emdl and evvd and evtx:
				# On an attachment: the prop's own space is the attachment's,
				# which the character's model places on one of his bones.
				na, ai = struct.unpack_from("<ii", mdl, 240)
				hit = None
				for k in range(na):
					o = ai + k * 92
					if _cstr(mdl, o + struct.unpack_from("<i", mdl, o)[0]).lower() == attach.lower():
						m12 = struct.unpack_from("<12f", mdl, o + 12)
						hit = (struct.unpack_from("<i", mdl, o + 8)[0], [list(m12[0:4]), list(m12[4:8]), list(m12[8:12])])
				if hit is None:
					continue
				ident = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
				flat = [dataclasses.replace(b, pose_to_bone=ident) for b in esrc.bones]
				etris, epos, enrm, euv, evb, _, _ids = lod_geometry(emdl, evvd, evtx, 0, flat)
				epos = [ss.mat34_xform(hit[1], p) for p in epos]
				enrm = [ss.mat34_rotate(hit[1], n) for n in enrm]
				bmap = [hit[0]] * len(esrc.bones)
			elif emdl and evvd and evtx and bmap and bmap[0] >= 0:
				etris, epos, enrm, euv, evb, _, _ids = lod_geometry(emdl, evvd, evtx, 0, esrc.bones)
			elif emdl and evvd and evtx and any(b >= 0 for b in bmap):
				# Bone-merged on the first bone both skeletons share (a
				# soldier's SMG: its root "ValveBiped" and Weapon_bone are not
				# his, its R_Hand is): the whole weapon rides that bone, placed
				# as the weapon model's own bind pose has it.
				mb = next(i for i, b in enumerate(bmap) if b >= 0)
				ident = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
				flat = [dataclasses.replace(b, pose_to_bone=ident) for b in esrc.bones]
				etris, epos, enrm, euv, evb, _, _ids = lod_geometry(emdl, evvd, evtx, 0, flat)
				p2b = esrc.bones[mb].pose_to_bone
				epos = [ss.mat34_xform(p2b, p) for p in epos]
				enrm = [ss.mat34_rotate(p2b, n) for n in enrm]
				bmap = [bmap[mb]] * len(esrc.bones)
			else:
				continue
			if True:
				# its skin becomes one more material column of the skin table
				numtex, texidx = struct.unpack_from("<ii", emdl, 204)
				ncd_e, cdidx_e = struct.unpack_from("<ii", emdl, 212)
				ecds = [_cstr(emdl, struct.unpack_from("<i", emdl, cdidx_e + i * 4)[0]).replace("\\", "/") for i in range(ncd_e)]
				o = texidx
				etname = _cstr(emdl, o + struct.unpack_from("<i", emdl, o)[0])
				img, kv, rel = load_skin(vpks, ecds, etname, max_side=max(tex_max, 256))
				if img is not None:
					# A key ring is five units long: 32 texels a side is already
					# finer than his coat, and the skins file has to load in one
					# piece from a heap that had 16 KB less with the full sheet.
					nw, nh = pow2_fit(img.width, img.height, 32 if attach else tex_max)
					rgba = resize_rgba(img.rgba, img.width, img.height, nw, nh)
					idx, pal = palettise(rgba, nw, nh, False)
					new_tex = len(textures)
					textures.append((etname[-63:], 0, nw, nh, idx, pal))
					tex_halflambert.append(kv.get("$halflambert") == "1")
					mat = nskinref
					equip_mats[mat] = flag
					if skin_table:
						skin_table = [v for f in range(nfam) for v in skin_table[f * nskinref:(f + 1) * nskinref] + [new_tex]]
					nskinref += 1
					extras.append(([(mat, a, b, c) for _m, a, b, c in etris], epos, enrm, euv,
						[bmap[v] if bmap[v] >= 0 else bmap[0] for v in evb]))
					print(f"  {name}: {eq} merged on {bones[bmap[0]].name} ({len(etris)} tris, skin {etname})")
	if actor:
		mat_flags = {}
		for m in range(max(1, nskinref)):
			tex = skin_table[m] if m < len(skin_table) else m
			if tex >= len(textures):
				continue
			f = BATCH_HALFLAMBERT if tex_halflambert[tex] else 0
			if textures[tex][1] & 0x0040:
				f |= BATCH_MASKED
			if textures[tex][1] & 0x0020:
				f |= BATCH_ADDITIVE
			mat_flags[m] = f | equip_mats.get(m, 0)
		out_dir.joinpath("models", "lc").mkdir(parents=True, exist_ok=True)
		face = None
		if struct.unpack_from("<i", mdl, 260)[0] > 0:
			# the model has face shapes: lip sync from the game's phoneme table
			import source_flex

			ptxt = read_first(vpks, "expressions/phonemes.txt")
			if ptxt:
				face = (source_flex.Flexes(mdl, vvd), source_flex.phoneme_table(ptxt.decode("latin-1")))
		write_actor(name, mdl, vvd, vtx, bones, mat_flags,
			out_dir / "models" / "lc" / f"{name}.gca", placements or [], extras, face)
		# Flipper draws the mesh from the .gca; the MDL keeps the skeleton,
		# sequences, bounds and skins.
		tris = []
		verts, vbone, norms, uvs = [], [], [], []

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
		bc = [-1] * 6
		for k, c in enumerate(look):
			if c["bone"] == i:
				bc[3 + c["axis"]] = k
		buf += struct.pack(
			"<32sii6i6f6f", b.name.encode("latin-1")[:31], b.parent, 0,
			*bc, *values[i], *scales[i],
		)
	bonecontrollerindex = here()
	for k, c in enumerate(look):
		buf += struct.pack("<iiffii", c["bone"], (STUDIO_XR, STUDIO_YR, STUDIO_ZR)[c["axis"]],
			min(c["deg_lo"], c["deg_hi"]), max(c["deg_lo"], c["deg_hi"]), 0, k)

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
		len(bones), boneindex, len(look), bonecontrollerindex, 1, hitboxindex,
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
	out = []
	for b in re.findall(r"\{[^{}]*\}", text):
		pairs = re.findall(r'"([^"]+)"\s+"([^"]*)"', b)
		e = dict(pairs)
		e["__outputs"] = [(k, v) for k, v in pairs if k.startswith("On") and "," in v]
		out.append(e)
	return out


def _outputs(e: dict) -> list[tuple[str, str]]:
	"""Entity I/O pairs (output name lower-cased, "target,input,param,delay,times")."""
	return [(k.lower(), v) for k, v in e.get("__outputs", [])]


def _parse3(s: str) -> tuple[float, float, float]:
	v = [float(x) for x in s.split()[:3]]
	return (v + [0.0, 0.0, 0.0])[:3]  # type: ignore[return-value]


def _npc_model(e: dict) -> str | None:
	cls = e.get("classname", "")
	model = e.get("model", "").lower().replace("\\", "/")
	if cls == "npc_fisherman":
		return "fisherman"
	if cls == "npc_combine_s":
		return "combine_super_soldier" if "super" in model else "combine_soldier"
	if cls == "npc_seagull":
		return "seagull"
	return None


def _brush_bounds(src_bsp: Path, e: dict) -> tuple | None:
	"""World-space box of a brush entity (*N model bounds + its origin)."""
	m = e.get("model", "")
	if not m.startswith("*"):
		return None
	d = src_bsp.read_bytes()
	off, _ln = struct.unpack_from("<ii", d, 8 + 14 * 16)
	v = struct.unpack_from("<6f", d, off + int(m[1:]) * 48)
	o = _parse3(e.get("origin", "0 0 0"))
	return tuple(v[i] + o[i % 3] for i in range(6))


def _fisherman_keys(src_bsp: Path, ents: list[dict], by_name: dict, name: str, at) -> dict:
	"""What the map does with the fisherman at the start of the level.

	The trigger_once nearest to where he stands that starts one of his scenes
	(lcs_fisherman_intro, on the jetty), the
	scene itself (baked by bake_lostcoast_actors.py) and the ai_goal_lead its
	OnCompletion activates: goal marker, LeadDistance, RetrieveDistance.
	"""
	keys: dict = {}
	best = None
	scenes = {e.get("targetname", "").lower(): e for e in ents
		if e.get("classname") == "logic_choreographed_scene" and e.get("target1", "").lower() == name}
	for e in ents:
		if e.get("classname") != "trigger_once" or e.get("StartDisabled", "0") == "1":
			continue
		for _k, v in _outputs(e):
			parts = [x.strip() for x in v.split(",")]
			if len(parts) > 1 and parts[1].lower() == "start" and parts[0].lower() in scenes:
				sc = scenes[parts[0].lower()]
				bb = _brush_bounds(src_bsp, e)
				if not bb:
					continue
				dist = math.dist(((bb[0] + bb[3]) / 2, (bb[1] + bb[4]) / 2, (bb[2] + bb[5]) / 2), at)
				if best is not None and dist >= best:
					continue
				best = dist
				keys.clear()
				keys["lc_trigger"] = " ".join(f"{x:g}" for x in bb)
				keys["lc_scene"] = "scripts/lc/" + Path(sc["SceneFile"].replace("\\", "/")).stem.lower() + ".txt"
				for k2, v2 in _outputs(sc):
					p2 = [x.strip() for x in v2.split(",")]
					if k2 != "oncompletion" or len(p2) < 2 or p2[1].lower() != "activate":
						continue
					for g in by_name.get(p2[0].lower(), []):
						if g.get("classname") != "ai_goal_lead":
							continue
						goal = (by_name.get(g.get("goal", "").lower()) or [None])[0]
						if goal:
							o = _parse3(goal.get("origin", "0 0 0"))
							yaw = _parse3(goal.get("angles", "0 0 0"))[1]
							keys["lc_lead_goal"] = f"{o[0]:g} {o[1]:g} {o[2]:g} {yaw:g}"
						keys["lc_lead_dist"] = g.get("LeadDistance", "64")
						keys["lc_retrieve_dist"] = g.get("RetrieveDistance", "96")
	# The scene that begins his scripted_sequences (opening the gate, sitting
	# down) and the trigger that starts it, wherever it is.
	gate_best = None
	# (by scene file: lcs_fisherman_keys names no target1)
	his = {e.get("targetname", "").lower(): e for e in ents
		if e.get("classname") == "logic_choreographed_scene" and "fisherman" in e.get("SceneFile", "").lower()}
	for e in ents:
		if e.get("classname") != "trigger_once":
			continue
		for _k, v in _outputs(e):
			parts = [x.strip() for x in v.split(",")]
			sc = his.get(parts[0].lower()) if len(parts) > 1 and parts[1].lower() == "start" else None
			if not sc or not any(k2.startswith("ontrigger") and "beginsequence" in v2.lower() for k2, v2 in _outputs(sc)):
				continue
			bb = _brush_bounds(src_bsp, e)
			if not bb or "lc_lead_goal" not in keys:
				continue
			# (the one at the end of the lead: lcs_leadtogate, on the rocks,
			# begins sequences too)
			goal = [float(x) for x in keys["lc_lead_goal"].split()[:3]]
			dist = math.dist(((bb[0] + bb[3]) / 2, (bb[1] + bb[4]) / 2, (bb[2] + bb[5]) / 2), goal)
			if gate_best is None or dist < gate_best:
				gate_best = dist
				keys["lc_gate_trigger"] = " ".join(f"{x:g}" for x in bb)
				keys["lc_gate_scene"] = "scripts/lc/" + Path(sc["SceneFile"].replace("\\", "/")).stem.lower() + ".txt"
				# What that scene enables when it is over: a trigger that
				# turns on an env_speaker while the player stands in it
				# (his nag to go through the gate). Box, concept, and
				# "seconds after the scene, shortest and longest gap, the
				# trigger's own wait".
				for k2, v2 in _outputs(sc):
					p2 = [x.strip() for x in v2.split(",")]
					if k2 != "oncompletion" or len(p2) < 4 or p2[1].lower() != "enable":
						continue
					for tg in by_name.get(p2[0].lower(), []):
						nb = _brush_bounds(src_bsp, tg)
						for _k3, v3 in _outputs(tg):
							p3 = [x.strip() for x in v3.split(",")]
							sp = (by_name.get(p3[0].lower()) or [None])[0] if len(p3) > 1 and p3[1].lower() == "turnon" else None
							if nb and sp and sp.get("classname") == "env_speaker" and sp.get("concept"):
								keys["lc_nag_trigger"] = " ".join(f"{x:g}" for x in nb)
								keys["lc_nag_concept"] = sp["concept"]
								keys["lc_nag_delay"] = f"{float(p2[3] or 0):g} {sp.get('delaymin', '15')} {sp.get('delaymax', '135')} {tg.get('wait', '1')}"
	# Every other trigger that starts one of his scenes and nothing more
	# (no scripted_sequence): his shouts up to the player on the cliff path
	# (lcs_fisherman_cliffclimb01/02). A trigger_once is a box; a trigger_look
	# also needs the player to look at its target. lc_cueN = "box scene" or
	# "box scene x y z fov looktime timeout".
	taken = {keys.get("lc_scene"), keys.get("lc_gate_scene")}
	ncue = 0
	for e in ents:
		cls = e.get("classname")
		if cls not in ("trigger_once", "trigger_look") or e.get("StartDisabled", "0") == "1":
			continue
		for _k, v in _outputs(e):
			parts = [x.strip() for x in v.split(",")]
			sc = his.get(parts[0].lower()) if len(parts) > 1 and parts[1].lower() == "start" else None
			if not sc or any("beginsequence" in v2.lower() for _k2, v2 in _outputs(sc)):
				continue
			scene = "scripts/lc/" + Path(sc["SceneFile"].replace("\\", "/")).stem.lower() + ".txt"
			bb = _brush_bounds(src_bsp, e)
			if scene in taken or not bb or ncue >= 4:
				continue
			cue = " ".join(f"{x:g}" for x in bb) + " " + scene
			if cls == "trigger_look":
				tgt = (by_name.get(e.get("target", "").lower()) or [None])[0]
				if not tgt:
					continue
				o = _parse3(tgt.get("origin", "0 0 0"))
				cue += f" {o[0]:g} {o[1]:g} {o[2]:g} {e.get('FieldOfView', '0.9')} {e.get('LookTime', '0.5')} {e.get('Timeout', '0')}"
			keys[f"lc_cue{ncue}"] = cue
			taken.add(scene)
			ncue += 1
	keys["lc_nodes"] = f"scripts/lc/{src_bsp.stem.lower()}.lcng"
	keys["lc_talker"] = "scripts/lc/fisherman_talker.txt"
	return keys


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
	# npc_template_makers told to Spawn when the map starts (logic_auto
	# OnMapSpawn, through logic_relays): their template stands at the
	# maker's origin and angles from frame one. The fisherman is one: the
	# e3_start relay spawns him on the jetty (npcmaker_fisherman), where the
	# retail opening shot shows him. ss_Fisherman_Intro is only used when a
	# script runs it, and nothing in the map does.
	by_name: dict[str, list[dict]] = {}
	for e in ents:
		if e.get("targetname"):
			by_name.setdefault(e["targetname"].lower(), []).append(e)
	start_targets: list[tuple[str, str]] = []
	path_starts: list[str] = []
	todo = [(k, v) for e in ents if e.get("classname") == "logic_auto"
		for k, v in _outputs(e) if k == "onmapspawn"]
	seen_relays: set[str] = set()
	while todo:
		_k, v = todo.pop()
		parts = v.split(",")
		if len(parts) < 2:
			continue
		tgt, inp = parts[0].strip().lower(), parts[1].strip().lower()
		start_targets.append((tgt, inp))
		if inp == "trigger" and tgt not in seen_relays:
			seen_relays.add(tgt)
			for r in by_name.get(tgt, []):
				if r.get("classname") == "logic_relay":
					todo += [(k2, v2) for k2, v2 in _outputs(r) if k2 == "ontrigger"]
	for tgt, inp in start_targets:
		if inp != "spawn":
			continue
		for mk in by_name.get(tgt, []):
			if mk.get("classname") != "npc_template_maker":
				continue
			tname = mk.get("TemplateName", "").lower()
			for t in by_name.get(tname, []):
				lc = _npc_model(t)
				if not lc or lc not in have:
					continue
				mo = _parse3(mk.get("origin", "0 0 0"))
				to = _parse3(t.get("origin", "0 0 0"))
				# The NPC drops to the floor on spawn; the template's own
				# origin is where the mapper stood it on that floor.
				z = to[2] if math.hypot(mo[0] - to[0], mo[1] - to[1]) < 48 else mo[2]
				sp = {"model": lc, "origin": f"{mo[0]:g} {mo[1]:g} {z:g}",
					"angles": mk.get("angles", t.get("angles", "0 0 0")), "seq": "idle"}
				if lc == "fisherman":
					sp["classname"] = "monster_lc_fisherman"
					sp["keys"] = _fisherman_keys(src_bsp, ents, by_name, tname, (mo[0], mo[1], z))
				out.append(sp)
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
		if not (lc and lc in have):
			continue
		sp = {"model": lc, "origin": org, "angles": ang, "seq": seq}
		if lc == "seagull":
			# npc_seagull runs as monster_lc_seagull (lostcoast_npcs.cpp,
			# Valve's npc_crow flight). When the map fires its FireUser1 at
			# spawn, its OnUser1 FlyAway,<path_corner> becomes the target.
			sp["classname"] = "monster_lc_seagull"
			if (name, "fireuser1") in start_targets:
				for k, v in _outputs(e):
					parts = [x.strip() for x in v.split(",")]
					if k == "onuser1" and len(parts) > 2 and parts[1].lower() == "flyaway" and parts[2]:
						sp["keys"] = {"target": parts[2]}
						path_starts.append(parts[2])
		out.append(sp)
	# Soldier squads a point_template spawns when the player walks into a
	# trigger_once or looks at a trigger_look's target (ForceSpawn): the
	# cliff path's first squads. Each soldier stands hidden at his template
	# spot until then (monster_lc_soldier): lc_spawn = "box" or
	# "box x y z fov looktime timeout", as the fisherman's lc_cueN.
	squads_done: set[str] = set()
	for e in ents:
		tcls = e.get("classname")
		if tcls not in ("trigger_once", "trigger_look") or e.get("StartDisabled", "0") == "1":
			continue
		for _k, v in _outputs(e):
			parts = [x.strip() for x in v.split(",")]
			if len(parts) < 2 or parts[1].lower() != "forcespawn":
				continue
			for pt in by_name.get(parts[0].lower(), []):
				if pt.get("classname") != "point_template":
					continue
				bb = _brush_bounds(src_bsp, e)
				if not bb:
					continue
				spawn = " ".join(f"{x:g}" for x in bb)
				if tcls == "trigger_look":
					tgt = (by_name.get(e.get("target", "").lower()) or [None])[0]
					if not tgt:
						continue
					o = _parse3(tgt.get("origin", "0 0 0"))
					spawn += f" {o[0]:g} {o[1]:g} {o[2]:g} {e.get('FieldOfView', '0.9')} {e.get('LookTime', '0.5')} {e.get('Timeout', '0')}"
				names = [tv.lower() for tk, tv in pt.items() if tk.lower().startswith("template") and tv]
				# (rappelling squads come down their ropes on a scripted
				# sequence: not yet, they would stand in the air)
				if any(x.get("classname") in ("move_rope", "scripted_sequence") for tn in names for x in by_name.get(tn, [])):
					continue
				# (a template two triggers spawn: the first one in the map)
				if pt.get("targetname", "").lower() in squads_done:
					continue
				squads_done.add(pt.get("targetname", "").lower())
				for tn in names:
					for t in by_name.get(tn, []):
						if t.get("classname") != "npc_combine_s":
							continue
						lc = "combine_super_soldier" if "super" in t.get("model", "").lower() else "combine_soldier"
						if lc not in have:
							continue
						out.append({"model": lc, "origin": t.get("origin", "0 0 0"), "angles": t.get("angles", "0 0 0"),
							"seq": "idle", "classname": "monster_lc_soldier",
							"keys": {"lc_spawn": spawn, "lc_squad": pt.get("targetname", "")}})
	# path_corner chains the characters follow, as GoldSrc path_corners
	seen: set[str] = set()
	for start in path_starts:
		c = start
		while c and c.lower() not in seen:
			pcs = [p_ for p_ in by_name.get(c.lower(), []) if p_.get("classname") == "path_corner"]
			if not pcs:
				break
			seen.add(c.lower())
			pc = pcs[0]
			out.append({"raw": {"classname": "path_corner", "targetname": pc["targetname"],
				"target": pc.get("target", ""), "origin": pc.get("origin", "0 0 0")}})
			c = pc.get("target", "")
	cannon = _cannon_keys(src_bsp, ents, by_name)
	if cannon:
		out.append({"raw": cannon})
	return out


def _cannon_keys(src_bsp: Path, ents: list[dict], by_name: dict) -> dict | None:
	"""The monastery gun shelling the town, as one lc_cannon entity.

	In the map: a trigger_once in the monastery enables logic_timer
	cannon_fire_timer (a random 30..60 s) and fires one shot at once; each
	tick logic_case cannon_fire_random picks one of six relays, each of which
	force-spawns an env_headcrabcanister template that flies from its
	LaunchPositionName (the gun) to its own origin in the town. A later
	trigger disables the timer. The engine has no entity I/O, so the chain
	is flattened into keys (lostcoast_npcs.cpp CLCCannon).
	"""
	timer = next((e for e in ents if e.get("classname") == "logic_timer" and "cannon_fire_random" in str(e.get("__outputs"))), None)
	if not timer:
		return None
	tname = timer.get("targetname", "")

	def box_of(inp: str):
		for e in ents:
			for _k, v in _outputs(e):
				parts = [x.strip() for x in v.split(",")]
				if len(parts) > 1 and parts[0].lower() == tname.lower() and parts[1].lower() == inp:
					b = _brush_bounds(src_bsp, e)
					if b:
						return e, b
		return None, None

	on_e, on_box = box_of("enable")
	_off_e, off_box = box_of("disable")
	if not on_box:
		return None
	keys = {"classname": "lc_cannon", "origin": timer.get("origin", "0 0 0"),
		"lc_start": " ".join(f"{c:g}" for c in on_box),
		"lc_interval": f"{float(timer.get('LowerRandomBound', 30)):g} {float(timer.get('UpperRandomBound', 60)):g}"}
	if off_box:
		keys["lc_stop"] = " ".join(f"{c:g}" for c in off_box)
	first = 0
	shots = 0
	for n in range(1, 9):
		relay = next(iter(by_name.get(f"cannon_fire_{n}_relay", [])), None)
		if not relay:
			continue
		tmpl = next((v.split(",")[0].strip() for _k, v in _outputs(relay) if "forcespawn" in v.lower()), "")
		pt = next(iter(by_name.get(tmpl.lower(), [])), None)
		can = next(iter(by_name.get((pt or {}).get("Template01", "").lower(), [])), None)
		gun = next(iter(by_name.get((can or {}).get("LaunchPositionName", "").lower(), [])), None)
		if not (can and gun):
			continue
		ang = can.get("angles", "0 0 0").split()
		keys[f"lc_shot{n}"] = (f"{gun.get('origin', '0 0 0')} {can.get('origin', '0 0 0')} {float(ang[0]):g} {float(ang[1]):g} "
			f"{float(can.get('FlightTime', 4)):g} {float(can.get('FlightSpeed', 3000)):g}")
		shots += 1
		if on_e and any(f"cannon_fire_{n}_relay" in v for _k, v in _outputs(on_e)):
			first = n
	if not shots:
		return None
	keys["lc_first"] = str(first)
	print(f"  cannon: {shots} canister shots, timer {keys['lc_interval']} s, first shot {first}")
	return keys


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
	nraw = 0
	for sp in spawns:
		if "raw" in sp:
			text += "{\n\"lc_studio\" \"1\"\n" + "".join(
				f"\"{k}\" \"{v}\"\n" for k, v in sp["raw"].items() if v) + "}\n"
			nraw += 1
			continue
		text += (
			f"{{\n\"classname\" \"{sp.get('classname', 'monster_generic')}\"\n\"lc_studio\" \"1\"\n"
			+ "".join(f"\"{k}\" \"{v}\"\n" for k, v in sp.get("keys", {}).items()) +
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
	print(f"  stub bsp: {len(spawns) - nraw} characters + {nraw} path_corners -> {stub}")


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
	have = {n.strip() for n in args.models.split(",")}
	spawns = npc_spawns(args.map, have) if args.map else []
	light = None
	if args.map and any(n in ACTOR_LODS for n in have):
		from source_lightcache import LightCache, lighting_at
		light = LightCache(args.map)
	for name in args.models.split(","):
		name = name.strip()
		lod = args.lod if args.lod >= 0 else DEFAULT_LOD.get(name, 1)
		places = []
		if light and name in ACTOR_LODS:
			for sp in spawns:
				if sp.get("model") != name:
					continue
				o = tuple(float(x) for x in sp["origin"].split()[:3])
				cube, sun, sdir = lighting_at(light, o)
				places.append({"origin": o, "cube": cube, "sun": sun, "sundir": sdir})
				print(f"  {name} @ {sp['origin']}: sun {'visible' if sun else 'blocked'}")
		convert(vpks, name, lod, args.tex_max, Path(args.out), args.flip, places)
		if name in LOOK_RANGES:
			for sp in spawns:
				if sp.get("model") == name and "keys" in sp:
					sp["keys"]["lc_look"] = LOOK_RANGES[name]
	if args.map and args.stub_bsp:
		patch_stub_bsp(args.stub_bsp, spawns)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
