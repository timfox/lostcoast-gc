"""Reader for Source particle files (.pcf): DMX "binary 2", format pcf 1.

  header line, NUL
  i16 string count, NUL-terminated strings (type and attribute names)
  i32 element count; per element: i16 type, name string, 16-byte id
  per element: i32 attribute count; per attribute: i16 name, u8 type, value

Value types: 1 element index, 2 int, 3 float, 4 bool, 5 string, 6 binary,
7 time (i32, 1/10000 s), 8 colour, 9-11 vec2/3/4, 12 angle, 13 quaternion,
14 matrix; 15-28 are arrays of 1-14 (i32 count first).
"""
from __future__ import annotations

import struct


def parse(data: bytes) -> list[dict]:
	pos = data.index(b"\0") + 1
	head = data[:pos]
	if b"binary 2" not in head or b"pcf" not in head:
		raise ValueError(f"unsupported pcf header {head[:60]!r}")

	def cstr():
		nonlocal pos
		end = data.index(b"\0", pos)
		s = data[pos:end].decode("latin-1")
		pos = end + 1
		return s

	def rd(fmt):
		nonlocal pos
		v = struct.unpack_from("<" + fmt, data, pos)
		pos += struct.calcsize("<" + fmt)
		return v

	strings = [cstr() for _ in range(rd("h")[0])]
	elems = []
	for _ in range(rd("i")[0]):
		t = strings[rd("h")[0]]
		name = cstr()
		pos += 16
		elems.append({"_type": t, "_name": name})

	def value(t):
		nonlocal pos
		if t == 1:
			return ("elem", rd("i")[0])
		if t == 2:
			return rd("i")[0]
		if t == 3:
			return rd("f")[0]
		if t == 4:
			return bool(rd("B")[0])
		if t == 5:
			return cstr()
		if t == 6:
			n = rd("i")[0]
			pos += n
			return None
		if t == 7:
			return rd("i")[0] / 10000.0
		if t == 8:
			return rd("4B")
		if t in (9, 10, 11):
			return rd(f"{t - 7}f")
		if t == 12:
			return rd("3f")
		if t == 13:
			return rd("4f")
		if t == 14:
			return rd("16f")
		raise ValueError(f"pcf attribute type {t}")

	for e in elems:
		for _ in range(rd("i")[0]):
			name = strings[rd("h")[0]]
			t = rd("B")[0]
			e[name] = [value(t - 14) for _ in range(rd("i")[0])] if t > 14 else value(t)
	# Resolve element references.
	for e in elems:
		for k, v in list(e.items()):
			if isinstance(v, tuple) and len(v) == 2 and v[0] == "elem":
				e[k] = elems[v[1]] if v[1] >= 0 else None
			elif isinstance(v, list):
				e[k] = [elems[x[1]] if isinstance(x, tuple) and len(x) == 2 and x[0] == "elem" and x[1] >= 0 else x for x in v]
	return elems


def systems(data: bytes) -> dict[str, dict]:
	return {e["_name"]: e for e in parse(data) if e["_type"] == "DmeParticleSystemDefinition"}
