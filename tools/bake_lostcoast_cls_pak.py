#!/usr/bin/env python3
"""Pack all CapFaces .cls sidecars into one media/cf_<map>_cls.pak (FST-safe)."""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--map", default="d2_lostcoast")
	ap.add_argument("--media", type=Path, default=ROOT / "cache/bake/media")
	args = ap.parse_args()

	cls_files = sorted(args.media.glob(f"cf_{args.map}*.cls"))
	if not cls_files:
		raise SystemExit(f"no .cls under {args.media}")

	# GCpk | ver u16 | count u16 | entries[count]{name[28], face_n u16, pad u16, ofs u32} | rows
	entries = []
	blob = bytearray()
	for p in cls_files:
		raw = p.read_bytes()
		if raw[:4] != b"GCcl":
			continue
		n = struct.unpack_from(">H", raw, 6)[0]
		payload = raw[8 : 8 + n * 2]
		# Store matching .gcp basename so runtime lookup is 1:1 with CapFaces path.
		gcp_name = p.name[:-4] + ".gcp" if p.name.endswith(".cls") else p.name
		name = gcp_name.encode("ascii")[:28]
		entries.append((name, n, len(blob)))
		blob += payload
		print(f"  + {gcp_name} faces={n}")

	hdr = bytearray(8)
	hdr[0:4] = b"GCpk"
	struct.pack_into(">HH", hdr, 4, 1, len(entries))
	table = bytearray()
	table_bytes = len(entries) * 36
	base = 8 + table_bytes
	for name, n, ofs in entries:
		rec = bytearray(36)
		rec[0 : len(name)] = name
		struct.pack_into(">HH", rec, 28, n, 0)
		struct.pack_into(">I", rec, 32, base + ofs)
		table += rec
	out = args.media / f"cf_{args.map}_cls.pak"
	out.write_bytes(bytes(hdr) + bytes(table) + bytes(blob))
	print(f"wrote {out} packs={len(entries)} bytes={out.stat().st_size}")


if __name__ == "__main__":
	main()
