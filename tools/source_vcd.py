#!/usr/bin/env python3
"""Read compiled choreography scenes (.vcd) from a Source scenes.image.

Layout from Source SDK 2013 (SceneImageFile.h, choreoscene.cpp,
choreoactor.cpp, choreochannel.cpp, choreoevent.cpp): a 'VSIF' image with
a shared string table and one entry per scene, keyed by the CRC32 of
"scenes\\<path>.vcd" (lower case, backslashes). Each scene is an optional
Valve 'LZMA' blob around a 'bvcd' binary scene.
"""
from __future__ import annotations

import lzma
import struct
import sys
import zlib
from dataclasses import dataclass, field

EVENT_TYPES = [
	"unspecified", "section", "expression", "lookat", "moveto", "speak",
	"gesture", "sequence", "face", "firetrigger", "flexanimation", "subscene",
	"loop", "interrupt", "stoppoint", "permit_responses", "generic",
]


@dataclass
class Event:
	type: str
	name: str
	start: float
	end: float
	param1: str
	param2: str
	param3: str
	flags: int
	distance: float = 0.0
	gesture_duration: float = -1.0
	relative_tags: list = field(default_factory=list)
	timing_tags: list = field(default_factory=list)
	abs_tags: list = field(default_factory=list)
	flex: list = field(default_factory=list)
	caption: str = ""


@dataclass
class Channel:
	name: str
	events: list
	active: bool


@dataclass
class Actor:
	name: str
	channels: list
	active: bool


@dataclass
class Scene:
	events: list
	actors: list


class SceneImage:
	def __init__(self, data: bytes):
		if data[:4] != b"VSIF":
			raise ValueError("not a scenes.image")
		_ver, nscenes, nstrings, entry_ofs = struct.unpack_from("<iiii", data, 4)
		self.data = data
		table = struct.unpack_from(f"<{nstrings}I", data, 20)
		self.strings = [data[o:data.index(b"\0", o)].decode("latin-1") for o in table]
		self.entries = {}
		for i in range(nscenes):
			crc, ofs, ln, _summ = struct.unpack_from("<IiiI", data, entry_ofs + i * 16)
			self.entries[crc] = (ofs, ln)

	@staticmethod
	def key(path: str) -> int:
		p = path.lower().replace("/", "\\")
		i = p.find("scenes\\")
		p = p[i:] if i >= 0 else "scenes\\" + p
		return zlib.crc32(p.encode("latin-1")) & 0xFFFFFFFF

	def raw(self, path: str) -> bytes | None:
		e = self.entries.get(self.key(path))
		if not e:
			return None
		blob = self.data[e[0]:e[0] + e[1]]
		if blob[:4] == b"LZMA":
			size, _csize = struct.unpack_from("<II", blob, 4)
			props = blob[12:17]
			hdr = props + struct.pack("<Q", size)
			return lzma.LZMADecompressor(format=lzma.FORMAT_ALONE).decompress(hdr + blob[17:])[:size]
		return blob

	def scene(self, path: str) -> Scene | None:
		b = self.raw(path)
		return _parse(b, self.strings) if b else None


class _Buf:
	def __init__(self, b: bytes):
		self.b, self.o = b, 0

	def take(self, fmt: str):
		v = struct.unpack_from("<" + fmt, self.b, self.o)
		self.o += struct.calcsize("<" + fmt)
		return v if len(v) > 1 else v[0]


def _parse(b: bytes, strings: list[str]) -> Scene:
	if b[:4] != b"bvcd":
		raise ValueError("not a binary vcd")
	r = _Buf(b)
	r.o = 9  # tag, version, crc

	def s():
		i = r.take("h")
		return strings[i] if 0 <= i < len(strings) else ""

	def curve():
		for _ in range(r.take("B")):
			r.take("fB")

	def event() -> Event:
		ty = r.take("b")
		ev = Event(EVENT_TYPES[ty] if 0 <= ty < len(EVENT_TYPES) else str(ty), s(), r.take("f"), r.take("f"),
			s(), s(), s(), 0)
		curve()
		ev.flags = r.take("B")
		ev.distance = r.take("f")
		ev.relative_tags = [(s(), r.take("B") / 255.0) for _ in range(r.take("B"))]
		ev.timing_tags = [(s(), r.take("B") / 255.0) for _ in range(r.take("B"))]
		for _ in range(2):
			ev.abs_tags.append([(s(), r.take("H") / 4096.0) for _ in range(r.take("B"))])
		if ev.type == "gesture":
			ev.gesture_duration = r.take("f")
		if r.take("b"):
			s(), s()
		for _ in range(r.take("B")):  # flex animation tracks
			# one flex controller over the event: samples of (time s, value
			# 0..1 of the track's min..max); a stereo ("combo") track adds
			# left/right balance samples (0 = all left, 1 = all right)
			name = s()
			fl = r.take("B")
			lo, hi = r.take("ff")
			amount = [(t, v / 255.0) for t, v, _curve in (r.take("fBH") for _ in range(r.take("h")))]
			balance = []
			if fl & 2:
				balance = [(t, v / 255.0) for t, v, _curve in (r.take("fBH") for _ in range(r.take("H")))]
			ev.flex.append({"name": name, "active": bool(fl & 1), "min": lo, "max": hi, "amount": amount, "balance": balance})
		if ev.type == "loop":
			r.take("b")
		if ev.type == "speak":
			r.take("b")
			ev.caption = s()
			r.take("b")
		return ev

	events = [event() for _ in range(r.take("B"))]
	actors = []
	for _ in range(r.take("B")):
		an = s()
		chans = []
		for _ in range(r.take("B")):
			cn = s()
			evs = [event() for _ in range(r.take("B"))]
			chans.append(Channel(cn, evs, bool(r.take("b"))))
		actors.append(Actor(an, chans, bool(r.take("b"))))
	return Scene(events, actors)


if __name__ == "__main__":
	img = SceneImage(open(sys.argv[1], "rb").read())
	sc = img.scene(sys.argv[2])
	if not sc:
		raise SystemExit("scene not found")
	for e in sc.events:
		print(f"[scene] {e.type:12s} {e.start:6.2f}-{e.end:6.2f} {e.name!r} {e.param1!r} {e.param2!r}")
	for a in sc.actors:
		for c in a.channels:
			for e in c.events:
				print(f"{a.name}/{c.name}: {e.type:12s} {e.start:6.2f}-{e.end:6.2f} {e.name!r} p1={e.param1!r} p2={e.param2!r} p3={e.param3!r}")
