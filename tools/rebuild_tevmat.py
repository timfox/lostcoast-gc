#!/usr/bin/env python3
"""Rebuild media/tevmat.gcm + tevmat-ids.json from extracted/VPK materials.

Used when only the TEV material table changes (e.g. cliff emboss). Keeps the
rest of cache/bake intact; run tools/bake_worlddl.py afterwards so WorldDL
mat.tevmat bytes match the new ids.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import bake_tevmat  # noqa: E402
from vmt import load_vmt  # noqa: E402
from vpk import open_vpk_roots, read_first  # noqa: E402
from vtf import decode_vtf  # noqa: E402


def _canonical(mat: str) -> str:
	m = mat.replace("\\", "/").lower()
	if m.startswith("materials/"):
		m = m[len("materials/") :]
	if m.endswith(".vmt"):
		m = m[: -len(".vmt")]
	# VBSP cubemap patches: maps/<map>/<mat>_x_y_z → <mat>
	if m.startswith("maps/"):
		parts = m.split("/", 2)
		if len(parts) >= 3:
			m = parts[2]
	m = re.sub(r"_-?\d+_-?\d+_-?\d+$", "", m)
	return m


def _load_vmt(vpks, extract: Path, mat: str) -> dict | None:
	rel = _canonical(mat)
	# Extracted copy wins (may include local edits).
	for base in (extract / "materials", extract):
		p = base / f"{rel}.vmt"
		if p.is_file():
			return load_vmt(p.read_bytes())
	raw = read_first(vpks, f"materials/{rel}.vmt")
	if raw:
		return load_vmt(raw)
	# Case variants under extract.
	want = rel.lower() + ".vmt"
	root = extract / "materials"
	if root.is_dir():
		for cand in root.rglob("*.vmt"):
			got = str(cand.relative_to(root)).replace("\\", "/").lower()
			if got == want:
				return load_vmt(cand.read_bytes())
	return None


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--out", type=Path, default=ROOT / "cache" / "bake")
	ap.add_argument("--extract", type=Path, default=ROOT / "cache" / "extracted")
	ap.add_argument(
		"--lostcoast",
		type=Path,
		default=ROOT / "content" / "lostcoast",
	)
	ap.add_argument("--hl2", type=Path, default=ROOT / "content" / "hl2")
	args = ap.parse_args()
	out: Path = args.out
	extract: Path = args.extract

	vpk_paths = sorted(args.lostcoast.glob("*_dir.vpk")) + sorted(args.hl2.glob("*_dir.vpk"))
	if not vpk_paths:
		print("error: no VPKs under", args.lostcoast, args.hl2, file=sys.stderr)
		return 1
	vpks = open_vpk_roots(vpk_paths)

	ids_path = out / "tevmat-ids.json"
	names_path = out / "texture-names.json"
	mats: set[str] = set()
	if ids_path.is_file():
		mats.update(json.loads(ids_path.read_text()))
	if names_path.is_file():
		raw = json.loads(names_path.read_text())
		if isinstance(raw, dict):
			mats.update(raw.keys())
		elif isinstance(raw, list):
			mats.update(raw)

	nature = extract / "materials" / "lostcoast" / "nature"
	if nature.is_dir():
		for p in nature.glob("*.vmt"):
			mats.add(f"lostcoast/nature/{p.stem}")

	feats: dict[str, dict] = {}
	weights: dict[str, float] = {}
	for mat in sorted(mats):
		base = _canonical(mat)
		weights[base] = weights.get(base, 0.0) + 1.0
		if base in feats:
			continue
		vmt = _load_vmt(vpks, extract, base)
		if not vmt:
			continue
		feat = bake_tevmat.classify(vmt)
		if feat:
			feats[base] = feat

	def load_rgba(tex: str):
		key = tex.replace("\\", "/").lower()
		if key.startswith("materials/"):
			key = key[len("materials/") :]
		if key.endswith(".vtf"):
			key = key[: -len(".vtf")]
		raw = read_first(vpks, f"materials/{key}.vtf")
		if raw is None:
			for base in (extract / "materials", extract):
				p = base / f"{key}.vtf"
				if p.is_file():
					raw = p.read_bytes()
					break
		if raw is None:
			return None
		img = decode_vtf(raw)
		return img.rgba, img.width, img.height

	new_ids, blob, log = bake_tevmat.build_table(feats, load_rgba, weights)
	expanded = dict(new_ids)
	for mat in mats:
		base = _canonical(mat)
		if base in new_ids:
			expanded[mat] = new_ids[base]
			expanded[base] = new_ids[base]

	media = out / "media"
	bake_tevmat.write_table(media / "tevmat.gcm", blob)
	ids_path.write_text(json.dumps(expanded, indent=1, sort_keys=True) + "\n")
	(out / "emboss-names.json").write_text(
		json.dumps(bake_tevmat.LAST_EMBOSS, indent=1) + "\n"
	)
	(out / "envmask-names.json").write_text(
		json.dumps(bake_tevmat.LAST_ENVMASKS, indent=1) + "\n"
	)
	for line in log:
		print(line)
	n_emboss = sum(1 for f in feats.values() if f["flags"] & bake_tevmat.F_EMBOSS)
	print(
		f"wrote {media / 'tevmat.gcm'} bytes={len(blob)} "
		f"recipes={len(new_ids)} emboss_mats={n_emboss} "
		f"heights={len(bake_tevmat.LAST_EMBOSS)} envmasks={len(bake_tevmat.LAST_ENVMASKS)}"
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
