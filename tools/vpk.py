#!/usr/bin/env python3
"""Minimal Source VPK1/VPK2 reader (directory + numbered packs)."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path


VPK_SIGNATURE = 0x55AA1234


@dataclass
class VPKEntry:
	path: str
	crc: int
	archive_index: int
	offset: int
	length: int
	preload: bytes


class VPK:
	def __init__(self, dir_path: Path):
		self.dir_path = Path(dir_path)
		self.entries: dict[str, VPKEntry] = {}
		self._load_directory()

	def _archive_path(self, index: int) -> Path:
		# lostcoast_pak_dir.vpk → lostcoast_pak_000.vpk
		name = self.dir_path.name
		if not name.endswith("_dir.vpk"):
			raise ValueError(f"expected *_dir.vpk, got {name}")
		stem = name[: -len("_dir.vpk")]
		if index == 0x7FFF:
			return self.dir_path
		return self.dir_path.with_name(f"{stem}_{index:03d}.vpk")

	def _load_directory(self) -> None:
		data = self.dir_path.read_bytes()
		sig, ver = struct.unpack_from("<II", data, 0)
		if sig != VPK_SIGNATURE:
			raise ValueError(f"{self.dir_path}: bad VPK signature {sig:#x}")
		if ver == 1:
			tree_size = struct.unpack_from("<I", data, 8)[0]
			header_size = 12
		elif ver == 2:
			tree_size = struct.unpack_from("<I", data, 8)[0]
			header_size = 28
		else:
			raise ValueError(f"{self.dir_path}: unsupported VPK version {ver}")

		tree = memoryview(data)[header_size : header_size + tree_size]
		pos = 0

		def read_cstring(buf: memoryview, i: int) -> tuple[str, int]:
			start = i
			while i < len(buf) and buf[i] != 0:
				i += 1
			s = bytes(buf[start:i]).decode("latin-1")
			return s, i + 1

		while True:
			ext, pos = read_cstring(tree, pos)
			if ext == "":
				break
			while True:
				path, pos = read_cstring(tree, pos)
				if path == "":
					break
				while True:
					name, pos = read_cstring(tree, pos)
					if name == "":
						break
					# CRC u32, preload u16, archive u16, offset u32, length u32, term u16 = 18
					crc, preload_len, archive_index, entry_offset, entry_length, terminator = struct.unpack_from(
						"<IHHIIH", tree, pos
					)
					pos += 18
					if terminator != 0xFFFF:
						raise ValueError(
							f"{self.dir_path}: bad VPK entry terminator {terminator:#x} at {pos - 2}"
						)
					preload = bytes(tree[pos : pos + preload_len])
					pos += preload_len
					if path == " ":
						rel = f"{name}.{ext}" if name != " " else f"{ext}"
					else:
						rel = f"{path}/{name}.{ext}"
					rel = rel.replace("\\", "/").lower()
					self.entries[rel] = VPKEntry(
						path=rel,
						crc=crc,
						archive_index=archive_index,
						offset=entry_offset,
						length=entry_length,
						preload=preload,
					)

	def read(self, rel: str) -> bytes | None:
		key = rel.replace("\\", "/").lower().lstrip("/")
		ent = self.entries.get(key)
		if ent is None:
			return None
		out = bytearray(ent.preload)
		need = ent.length
		if need <= 0:
			return bytes(out)
		arch = self._archive_path(ent.archive_index)
		with arch.open("rb") as f:
			# VPK2: entry offset is relative to start of archive data (after header for dir-embedded).
			if ent.archive_index == 0x7FFF:
				# data lives after directory header+tree in the dir file
				data = self.dir_path.read_bytes()
				sig, ver = struct.unpack_from("<II", data, 0)
				tree_size = struct.unpack_from("<I", data, 8)[0]
				header_size = 12 if ver == 1 else 28
				base = header_size + tree_size
				chunk = data[base + ent.offset : base + ent.offset + need]
			else:
				f.seek(ent.offset)
				chunk = f.read(need)
		out.extend(chunk)
		return bytes(out)

	def find_prefix(self, prefix: str) -> list[str]:
		p = prefix.replace("\\", "/").lower().lstrip("/")
		return sorted(k for k in self.entries if k.startswith(p))


def open_vpk_roots(paths: list[Path]) -> list[VPK]:
	vpks = []
	for p in paths:
		p = Path(p)
		if p.is_file() and p.name.endswith("_dir.vpk"):
			vpks.append(VPK(p))
		elif p.is_dir():
			for d in sorted(p.glob("*_dir.vpk")):
				vpks.append(VPK(d))
	return vpks


def read_first(vpks: list[VPK], rel: str) -> bytes | None:
	for v in vpks:
		data = v.read(rel)
		if data is not None:
			return data
	return None
