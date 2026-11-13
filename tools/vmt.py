#!/usr/bin/env python3
"""Minimal VMT parser — resolve $basetexture / $basetexture2."""
from __future__ import annotations

import re
from pathlib import Path


_KEY_RE = re.compile(
	r'^\s*(?:"([^"]+)"|(\S+))\s+(?:"([^"]*)"|(\S+))',
	re.IGNORECASE,
)


def parse_vmt(text: str) -> dict[str, str]:
	"""Return lowercased key→value for the root block (ignores nested braces lightly)."""
	# Strip // comments
	lines = []
	for line in text.splitlines():
		if "//" in line:
			line = line[: line.index("//")]
		lines.append(line)
	blob = "\n".join(lines)
	# Flatten nested blocks by taking only depth-1 keys (good enough for lightmappedgeneric).
	out: dict[str, str] = {}
	depth = 0
	for raw in blob.splitlines():
		line = raw.strip()
		if not line:
			continue
		depth += line.count("{") - line.count("}")
		if depth > 1:
			continue
		m = _KEY_RE.match(line.replace("{", " ").replace("}", " "))
		if not m:
			continue
		key = (m.group(1) or m.group(2) or "").lower()
		val = m.group(3) if m.group(3) is not None else (m.group(4) or "")
		if key and not key.startswith("{"):
			out[key] = val.replace("\\", "/")
	return out


def basetextures(vmt: dict[str, str]) -> list[str]:
	names = []
	for key in ("$basetexture", "$basetexture2", "$bumpmap", "$envmapmask"):
		val = vmt.get(key)
		if not val:
			continue
		# strip materials/ prefix and extension if present
		v = val.replace("\\", "/").lower()
		if v.startswith("materials/"):
			v = v[len("materials/") :]
		if v.endswith(".vtf"):
			v = v[: -len(".vtf")]
		if v and v not in names:
			names.append(v)
	return names


def load_vmt(data: bytes) -> dict[str, str]:
	return parse_vmt(data.decode("latin-1", "replace"))


def material_short_name(mat_path: str, used: dict[str, str]) -> str:
	"""Map a Source material path to a unique ≤15-char GoldSrc-style name."""
	mat = mat_path.replace("\\", "/").lower()
	if mat in used:
		return used[mat]
	base = mat.rsplit("/", 1)[-1]
	base = re.sub(r"[^a-z0-9_]", "", base)[:15] or "mat"
	candidate = base[:15]
	n = 0
	existing = set(used.values())
	while candidate in existing:
		n += 1
		suffix = f"_{n}"
		candidate = (base[: 15 - len(suffix)] + suffix)[:15]
	used[mat] = candidate
	return candidate
