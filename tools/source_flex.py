"""Source facial animation (flexes) read from a .mdl.

A face shape is a flex descriptor: a set of vertex offsets on the meshes
(mstudioflex_t / mstudiovertanim_t). What drives them is a flex controller
(jaw_drop, left_puckerer ...), through the model's flex rules: small RPN
programs that turn controller values into one weight per descriptor
(studio.h mstudioflexrule_t, bone_setup RunFlexRules). Phonemes are rows of
controller values in expressions/phonemes.txt.
"""
from __future__ import annotations

import re
import struct


def _cs(d: bytes, o: int) -> str:
	return d[o:d.index(b"\0", o)].decode("latin-1")


class Flexes:
	def __init__(self, mdl: bytes, vvd: bytes):
		self.mdl = mdl
		nfd, fdi, nfc, fci, nfr, fri = struct.unpack_from("<6i", mdl, 260)
		self.names = [_cs(mdl, fdi + i * 4 + struct.unpack_from("<i", mdl, fdi + i * 4)[0]) for i in range(nfd)]
		self.controllers = []  # (name, min, max)
		for i in range(nfc):
			o = fci + i * 20
			_ty, nm, _l2g, mn, mx = struct.unpack_from("<iiiff", mdl, o)
			self.controllers.append((_cs(mdl, o + nm), mn, mx))
		self.rules = []
		for i in range(nfr):
			o = fri + i * 12
			flex, nops, opi = struct.unpack_from("<iii", mdl, o)
			self.rules.append((flex, [(struct.unpack_from("<i", mdl, o + opi + k * 8)[0], mdl[o + opi + k * 8 + 4:o + opi + k * 8 + 8])
				for k in range(nops)]))
		# VVD ids of the LOD 0 vertex stream (fixups put it back in order)
		num_lod_verts = struct.unpack_from("<8i", vvd, 16)
		num_fixups, fixup_off, _vstart, _tan = struct.unpack_from("<iiii", vvd, 48)
		if num_fixups:
			lod0: list[int] = []
			for i in range(num_fixups):
				_flod, src, num = struct.unpack_from("<iii", vvd, fixup_off + i * 12)
				lod0.extend(range(src, src + num))
		else:
			lod0 = list(range(num_lod_verts[0]))
		# body part 0, model 0: every mesh's flexes
		_nbp, bpidx = struct.unpack_from("<ii", mdl, 232)
		mh = bpidx + struct.unpack_from("<i", mdl, bpidx + 12)[0]
		nmesh, meshidx, _nv, vindex = struct.unpack_from("<iiii", mdl, mh + 72)
		self.flexes = []  # (flexdesc, (t0..t3), pair, [(vvd id, speed, side, (dx, dy, dz))])
		for k in range(nmesh):
			msh = mh + meshidx + k * 116
			_mat, _mi, _nvm, voff, nflex, flexidx = struct.unpack_from("<6i", mdl, msh)
			for f in range(nflex):
				fo = msh + flexidx + f * 60
				fd, t0, t1, t2, t3, nv, vi, pair = struct.unpack_from("<i4fiii", mdl, fo)
				size = 18 if mdl[fo + 32] == 1 else 16  # wrinkle vertanims carry one more short
				va = []
				for j in range(nv):
					o = fo + vi + j * size
					idx, speed, side = struct.unpack_from("<HBB", mdl, o)
					va.append((lod0[vindex // 48 + voff + idx], speed / 255.0, side / 255.0, struct.unpack_from("<3e", mdl, o + 4)))
				self.flexes.append((fd, (t0, t1, t2, t3), pair, va))

	def weights(self, values: dict[str, float]) -> list[float]:
		"""Controller settings (0..1 of each controller's range) -> one weight per flex descriptor."""
		cv = [mn + values.get(name, 0.0) * (mx - mn) for name, mn, mx in self.controllers]
		w = [0.0] * len(self.names)
		for flex, ops in self.rules:
			st: list[float] = []
			for op, a in ops:
				if op == 1:
					st.append(struct.unpack("<f", a)[0])
				elif op == 2:
					st.append(cv[struct.unpack("<i", a)[0]])
				elif op == 3:
					st.append(w[struct.unpack("<i", a)[0]])
				elif op == 4:
					y = st.pop(); st[-1] += y
				elif op == 5:
					y = st.pop(); st[-1] -= y
				elif op == 6:
					y = st.pop(); st[-1] *= y
				elif op == 7:
					y = st.pop(); st[-1] = st[-1] / y if y > 0.0001 else 0.0
				elif op == 8:
					st[-1] = -st[-1]
				elif op == 13:
					y = st.pop(); st[-1] = max(st[-1], y)
				elif op == 14:
					y = st.pop(); st[-1] = min(st[-1], y)
				else:
					raise SystemExit(f"flex rule op {op} is not handled")
			w[flex] = st[-1] if st else 0.0
		return w

	@staticmethod
	def ramp(t, w: float) -> float:
		"""mstudioflex_t target0..3 (RampFlexWeight)."""
		t0, t1, t2, t3 = t
		if w <= t0 or w >= t3:
			return 0.0
		if w < t1:
			return (w - t0) / (t1 - t0)
		if w > t2:
			return (t3 - w) / (t3 - t2)
		return 1.0


def phoneme_table(text: str) -> dict[int, tuple[str, dict[str, float]]]:
	"""expressions/phonemes.txt: phoneme code -> (name, controller settings)."""
	lines = text.splitlines()
	keys = lines[0].split()[1:]
	out = {}
	for line in lines[1:]:
		m = re.match(r'"([^"]+)"\s+"([^"]+)"\s+(.*?)\s+"', line)
		if not m:
			continue
		vals = [float(x) for x in m.group(3).split()]
		code = int(m.group(2), 16) if m.group(2).lower().startswith("0x") else ord(m.group(2)[0])
		# pairs of (setting, weight): weight 0 leaves the controller alone
		out[code] = (m.group(1), {k: vals[i * 2] for i, k in enumerate(keys) if vals[i * 2 + 1] > 0})
	return out


def wave_phonemes(wav: bytes) -> list[tuple[int, float, float]]:
	"""The VDAT chunk Faceposer leaves in a voice wave: (phoneme code, start s, end s)."""
	i = wav.find(b"VDAT")
	if i < 0:
		return []
	size = struct.unpack_from("<I", wav, i + 4)[0]
	text = wav[i + 8:i + 8 + size].decode("latin-1", "replace")
	return [(int(m.group(1)), float(m.group(2)), float(m.group(3)))
		for m in re.finditer(r"^\s*(\d+)\s+\S+\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\s*$", text, re.M)]
