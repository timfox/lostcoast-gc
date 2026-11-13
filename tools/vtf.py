#!/usr/bin/env python3
"""Decode common Source VTF formats to RGBA8 (mip0)."""
from __future__ import annotations

import struct
from dataclasses import dataclass


IMAGE_FORMATS = {
	0: "RGBA8888",
	1: "ABGR8888",
	2: "RGB888",
	3: "BGR888",
	4: "RGB565",
	5: "I8",
	6: "IA88",
	7: "P8",
	8: "A8",
	9: "RGB888_BLUESCREEN",
	10: "BGR888_BLUESCREEN",
	11: "ARGB8888",
	12: "BGRA8888",
	13: "DXT1",
	14: "DXT3",
	15: "DXT5",
	16: "BGRX8888",
	17: "BGR565",
	18: "BGRX5551",
	19: "BGRA4444",
	20: "DXT1_ONEBITALPHA",
	21: "BGRA5551",
	22: "UV88",
	23: "UVWQ8888",
	24: "RGBA16161616F",
	25: "RGBA16161616",
	26: "UVLX8888",
}


@dataclass
class VTFImage:
	width: int
	height: int
	rgba: bytes  # w*h*4


def _decode_dxt1_block(block: bytes, out: bytearray, x: int, y: int, w: int, h: int, one_bit_alpha: bool = False) -> None:
	c0, c1 = struct.unpack_from("<HH", block, 0)
	bits = struct.unpack_from("<I", block, 4)[0]

	def rgb565(c: int) -> tuple[int, int, int]:
		r = ((c >> 11) & 31) * 255 // 31
		g = ((c >> 5) & 63) * 255 // 63
		b = (c & 31) * 255 // 31
		return r, g, b

	p0 = rgb565(c0)
	p1 = rgb565(c1)
	if c0 > c1:
		palette = [
			p0,
			p1,
			((2 * p0[0] + p1[0]) // 3, (2 * p0[1] + p1[1]) // 3, (2 * p0[2] + p1[2]) // 3),
			((p0[0] + 2 * p1[0]) // 3, (p0[1] + 2 * p1[1]) // 3, (p0[2] + 2 * p1[2]) // 3),
		]
		alphas = [255, 255, 255, 255]
	else:
		palette = [
			p0,
			p1,
			((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2, (p0[2] + p1[2]) // 2),
			(0, 0, 0),
		]
		alphas = [255, 255, 255, 0 if one_bit_alpha or True else 255]
		# Standard DXT1: c0<=c1 → color3 transparent
		alphas[3] = 0
	for row in range(4):
		for col in range(4):
			px = x + col
			py = y + row
			if px >= w or py >= h:
				continue
			idx = (bits >> (2 * (row * 4 + col))) & 3
			r, g, b = palette[idx]
			a = alphas[idx]
			o = (py * w + px) * 4
			out[o : o + 4] = bytes((r, g, b, a))


def _decode_dxt5_block(block: bytes, out: bytearray, x: int, y: int, w: int, h: int) -> None:
	a0, a1 = block[0], block[1]
	abits = int.from_bytes(block[2:8], "little")
	if a0 > a1:
		apal = [a0, a1] + [((6 - i) * a0 + (i + 1) * a1) // 7 for i in range(6)]
	else:
		apal = [a0, a1, (4 * a0 + a1) // 5, (3 * a0 + 2 * a1) // 5, (2 * a0 + 3 * a1) // 5, (a0 + 4 * a1) // 5, 0, 255]
	# color
	tmp = bytearray(16 * 4)
	_decode_dxt1_block(block[8:16], tmp, 0, 0, 4, 4)
	for row in range(4):
		for col in range(4):
			px = x + col
			py = y + row
			if px >= w or py >= h:
				continue
			ai = (abits >> (3 * (row * 4 + col))) & 7
			o = (py * w + px) * 4
			to = (row * 4 + col) * 4
			out[o : o + 3] = tmp[to : to + 3]
			out[o + 3] = apal[ai]


def _decode_dxt(data: bytes, w: int, h: int, fmt: str) -> bytes:
	out = bytearray(w * h * 4)
	bw = (w + 3) // 4
	bh = (h + 3) // 4
	bs = 8 if fmt in ("DXT1", "DXT1_ONEBITALPHA") else 16
	i = 0
	for by in range(bh):
		for bx in range(bw):
			block = data[i : i + bs]
			i += bs
			if len(block) < bs:
				break
			if fmt in ("DXT1", "DXT1_ONEBITALPHA"):
				_decode_dxt1_block(block, out, bx * 4, by * 4, w, h, one_bit_alpha=(fmt == "DXT1_ONEBITALPHA"))
			elif fmt == "DXT3":
				# skip explicit alpha (8 bytes) — treat as opaque DXT1 colour for smoke
				_decode_dxt1_block(block[8:16], out, bx * 4, by * 4, w, h)
				for row in range(4):
					for col in range(4):
						px, py = bx * 4 + col, by * 4 + row
						if px >= w or py >= h:
							continue
						a_nibble = (block[(row * 4 + col) // 2] >> (4 * ((row * 4 + col) & 1))) & 0xF
						out[(py * w + px) * 4 + 3] = a_nibble * 17
			else:  # DXT5
				_decode_dxt5_block(block, out, bx * 4, by * 4, w, h)
	return bytes(out)


def _mip_size(w: int, h: int, fmt: int) -> int:
	name = IMAGE_FORMATS.get(fmt, "")
	if name.startswith("DXT1"):
		return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * 8
	if name in ("DXT3", "DXT5"):
		return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * 16
	bpp = {
		"RGBA8888": 4,
		"ABGR8888": 4,
		"ARGB8888": 4,
		"BGRA8888": 4,
		"BGRX8888": 4,
		"RGB888": 3,
		"BGR888": 3,
		"RGB565": 2,
		"BGR565": 2,
		"I8": 1,
		"A8": 1,
		"IA88": 2,
		"UV88": 2,
		"RGBA16161616F": 8,
		"RGBA16161616": 8,
	}.get(name)
	if bpp is None:
		raise ValueError(f"unsupported VTF format {fmt} ({name})")
	return w * h * bpp


def decode_vtf(data: bytes, max_side: int | None = 256) -> VTFImage:
	if data[:4] != b"VTF\0":
		raise ValueError("not a VTF")
	ver_maj, ver_min, header_size = struct.unpack_from("<III", data, 4)
	width, height = struct.unpack_from("<HH", data, 16)
	flags = struct.unpack_from("<I", data, 20)[0]
	frames, first_frame = struct.unpack_from("<HH", data, 24)
	# reflectivity at 32, bumpscale 48, highres format 52
	high_fmt = struct.unpack_from("<I", data, 52)[0]
	mipcount = data[56]
	low_fmt = struct.unpack_from("<I", data, 57)[0]
	low_w, low_h = data[61], data[62]
	# depth at 63 for v7.2+
	(void_depth := 1)
	if ver_min >= 2:
		depth = struct.unpack_from("<H", data, 63)[0] or 1
	else:
		depth = 1
	del void_depth, flags, first_frame

	# Skip low-res thumbnail then find largest mip of frame 0
	thumb = _mip_size(low_w, low_h, low_fmt) if low_w and low_h else 0
	# Align: thumbnail starts at header_size
	pos = header_size + thumb

	# Mipmaps are stored smallest → largest
	mips = []
	w, h = width, height
	for level in range(mipcount):
		mips.append((max(1, w >> (mipcount - 1 - level)), max(1, h >> (mipcount - 1 - level))))
	# Actually Valve stores from smallest to largest:
	sizes = []
	for i in range(mipcount):
		mw = max(1, width >> (mipcount - 1 - i))
		mh = max(1, height >> (mipcount - 1 - i))
		sizes.append((mw, mh, _mip_size(mw, mh, high_fmt)))

	frame_count = max(1, frames)
	# Skip all smaller mips + other frames: we want last mip (largest) of frame 0
	# Layout: for each mip (small→large): for each frame: for each face/z: image
	offset = pos
	target_w = target_h = high = None
	for mi, (mw, mh, ms) in enumerate(sizes):
		for fr in range(frame_count):
			chunk = data[offset : offset + ms]
			offset += ms * depth  # faces/z ignored for 2D
			if fr == 0 and mi == len(sizes) - 1:
				target_w, target_h, high = mw, mh, chunk

	if high is None:
		raise ValueError("VTF missing high-res mip")

	name = IMAGE_FORMATS.get(high_fmt, str(high_fmt))
	if name.startswith("DXT"):
		rgba = _decode_dxt(high, target_w, target_h, name)
	elif name == "RGBA8888":
		rgba = high if len(high) >= target_w * target_h * 4 else high + b"\x00" * (target_w * target_h * 4 - len(high))
	elif name == "BGRA8888" or name == "BGRX8888":
		rgba = bytearray(target_w * target_h * 4)
		for i in range(target_w * target_h):
			b, g, r, a = high[i * 4 : i * 4 + 4]
			if name == "BGRX8888":
				a = 255
			rgba[i * 4 : i * 4 + 4] = bytes((r, g, b, a))
		rgba = bytes(rgba)
	elif name == "ABGR8888":
		rgba = bytearray(target_w * target_h * 4)
		for i in range(target_w * target_h):
			a, b, g, r = high[i * 4 : i * 4 + 4]
			rgba[i * 4 : i * 4 + 4] = bytes((r, g, b, a))
		rgba = bytes(rgba)
	elif name == "RGB888":
		rgba = bytearray(target_w * target_h * 4)
		for i in range(target_w * target_h):
			r, g, b = high[i * 3 : i * 3 + 3]
			rgba[i * 4 : i * 4 + 4] = bytes((r, g, b, 255))
		rgba = bytes(rgba)
	elif name == "BGR888":
		rgba = bytearray(target_w * target_h * 4)
		for i in range(target_w * target_h):
			b, g, r = high[i * 3 : i * 3 + 3]
			rgba[i * 4 : i * 4 + 4] = bytes((r, g, b, 255))
		rgba = bytes(rgba)
	elif name == "I8":
		rgba = bytearray(target_w * target_h * 4)
		for i, v in enumerate(high[: target_w * target_h]):
			rgba[i * 4 : i * 4 + 4] = bytes((v, v, v, 255))
		rgba = bytes(rgba)
	elif name == "UV88":
		# Two unsigned channels. 128 is the zero offset for dudv.
		rgba = bytearray(target_w * target_h * 4)
		n = target_w * target_h
		if len(high) < n * 2:
			raise ValueError("UV88 truncated")
		for i in range(n):
			u, v = high[i * 2], high[i * 2 + 1]
			rgba[i * 4 : i * 4 + 4] = bytes((u, v, 128, 255))
		rgba = bytes(rgba)
	elif name == "RGBA16161616F":
		# IEEE754 binary16 RGBA — Source HDR skyboxes / lightmaps.
		# Exposure + Reinhard so CapFaces CMPR stays LDR but readable.
		import math

		def f16_to_f32(u: int) -> float:
			sign = (u >> 15) & 1
			exp = (u >> 10) & 0x1F
			mant = u & 0x3FF
			if exp == 0:
				if mant == 0:
					val = 0.0
				else:
					val = math.ldexp(mant / 1024.0, -14)
			elif exp == 31:
				val = float("inf") if mant == 0 else float("nan")
			else:
				val = math.ldexp(1.0 + mant / 1024.0, exp - 15)
			return -val if sign else val

		need = target_w * target_h * 8
		if len(high) < need:
			raise ValueError("RGBA16161616F truncated")

		# Source HDR sky floats are often dark; lift with exposure before tonemap.
		exposure = 6.0

		def tonemap(x: float) -> int:
			if not math.isfinite(x) or x <= 0.0:
				return 0
			x *= exposure
			y = x / (1.0 + x)
			y = pow(y, 1.0 / 2.2)
			return min(255, int(y * 255.0 + 0.5))

		rgba = bytearray(target_w * target_h * 4)
		for i in range(target_w * target_h):
			o = i * 8
			r = f16_to_f32(struct.unpack_from("<H", high, o)[0])
			g = f16_to_f32(struct.unpack_from("<H", high, o + 2)[0])
			b = f16_to_f32(struct.unpack_from("<H", high, o + 4)[0])
			# Force opaque — Source stores unused alpha as 0 in these sky VTFs.
			rgba[i * 4 : i * 4 + 4] = bytes((tonemap(r), tonemap(g), tonemap(b), 255))
		rgba = bytes(rgba)
	else:
		raise ValueError(f"unsupported VTF image format {name}")

	# Cap huge textures for GameCube bake. Additive shafts pass None so
	# their 0–7 core can be lifted before any downsample.
	if max_side and (target_w > max_side or target_h > max_side):
		try:
			from PIL import Image
			import numpy as np

			img = Image.fromarray(np.frombuffer(rgba, dtype=np.uint8).reshape(target_h, target_w, 4), "RGBA")
			scale = min(max_side / target_w, max_side / target_h)
			nw, nh = max(8, int(target_w * scale) & ~7), max(8, int(target_h * scale) & ~7)
			img = img.resize((nw, nh), Image.Resampling.BILINEAR)
			rgba = img.tobytes()
			target_w, target_h = nw, nh
		except Exception:
			pass

	return VTFImage(width=target_w, height=target_h, rgba=rgba)
