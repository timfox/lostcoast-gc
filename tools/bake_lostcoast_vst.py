#!/usr/bin/env python3
"""Bake Lost Coast VST camera path + Source PVS for CapFaces cull.

The disc GoldSrc BSP is a 1-room stub (visibility=0). CapFaces densify cannot
cull from that. This tool reads the real Source VBSP and emits:

  media/vis_<map>.gcv     — Source compressed PVS (cluster-indexed)
  media/vst_<map>_path.bin — on-rails camera (ptrack_vst / train / view)
  media/cf_<pack>.cls     — Source cluster per CapFace (sidecar)

Runtime rides the path like HL2's func_tracktrain + point_viewcontrol and
culls CapFaces with CHECKVISBIT against the eye cluster's PVS row.
"""
from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_lump(data: bytes, i: int) -> bytes:
	off, ln, _ver, _fcc = struct.unpack_from("<iiii", data, 8 + i * 16)
	return data[off : off + ln]


def parse_ents(ents: str) -> list[dict[str, str]]:
	out = []
	for m in re.finditer(r"\{([^}]*)\}", ents, re.S):
		kv = dict(re.findall(r'"([^"]*)"\s*"([^"]*)"', m.group(1)))
		if kv:
			out.append(kv)
	return out


def point_in_leaf(origin, nodes, leaves, planes):
	"""Source PointInLeaf — return (leaf_index, cluster)."""
	idx = 0
	depth = 0
	while idx >= 0 and depth < 8192:
		depth += 1
		planenum, c0, c1 = nodes[idx]
		nx, ny, nz, dist, _typ = planes[planenum]
		d = nx * origin[0] + ny * origin[1] + nz * origin[2] - dist
		idx = c0 if d >= 0.0 else c1
	leaf_i = -1 - idx
	if leaf_i < 0 or leaf_i >= len(leaves):
		return 0, -1
	_contents, cluster = leaves[leaf_i]
	return leaf_i, int(cluster)


def load_source_tree(bsp_path: Path):
	data = bsp_path.read_bytes()
	if data[:4] != b"VBSP":
		raise SystemExit(f"not VBSP: {bsp_path}")
	planes_raw = source_lump(data, 1)
	nodes_raw = source_lump(data, 5)
	leaves_raw = source_lump(data, 10)
	vis_raw = source_lump(data, 4)
	ents = source_lump(data, 0).decode("latin-1", "replace")

	planes = [
		struct.unpack_from("<ffffi", planes_raw, i)
		for i in range(0, len(planes_raw), 20)
	]
	nodes = [
		struct.unpack_from("<iii", nodes_raw, i)[:3]  # planenum, c0, c1
		for i in range(0, len(nodes_raw), 32)
	]
	leaves = [
		struct.unpack_from("<ih", leaves_raw, i)  # contents, cluster
		for i in range(0, len(leaves_raw), 32)
	]
	numclusters = struct.unpack_from("<i", vis_raw, 0)[0]
	return {
		"planes": planes,
		"nodes": nodes,
		"leaves": leaves,
		"vis": vis_raw,
		"numclusters": numclusters,
		"ents": parse_ents(ents),
	}


def extract_vst_path(ents: list[dict], tree) -> tuple[float, list[dict]]:
	tracks = {}
	train = None
	view = None
	for kv in ents:
		cls = kv.get("classname", "")
		if cls == "path_track":
			tn = kv.get("targetname", "")
			if tn.startswith("ptrack_vst"):
				o = [float(x) for x in kv.get("origin", "0 0 0").split()]
				a = [float(x) for x in kv.get("angles", "0 0 0").split()]
				tracks[tn] = {
					"name": tn,
					"origin": o,
					"angles": a,
					"target": kv.get("target", ""),
					"speed": float(kv.get("speed") or 0),
				}
		elif cls == "func_tracktrain" and kv.get("targetname") == "train":
			train = kv
		elif cls == "point_viewcontrol" and kv.get("targetname") == "view":
			view = kv

	if not train or not tracks:
		raise SystemExit("vst path_track / train missing from Source ents")

	# View is parented to train; sample local offset vs first path node.
	start = train.get("target", "ptrack_vst_1")
	view_off = [0.0, 0.0, -54.0]
	if view and start in tracks:
		vo = [float(x) for x in view.get("origin", "0 0 0").split()]
		to = tracks[start]["origin"]
		view_off = [vo[0] - to[0], vo[1] - to[1], vo[2] - to[2]]

	ordered = []
	name = start
	seen = set()
	while name and name in tracks and name not in seen:
		seen.add(name)
		t = tracks[name]
		eye = [
			t["origin"][0] + view_off[0],
			t["origin"][1] + view_off[1],
			t["origin"][2] + view_off[2],
		]
		_leaf, cluster = point_in_leaf(eye, tree["nodes"], tree["leaves"], tree["planes"])
		ordered.append(
			{
				"name": t["name"],
				"origin": eye,
				"angles": t["angles"],
				"cluster": cluster,
				"speed": t["speed"],
			}
		)
		name = t["target"]

	speed = float(train.get("speed") or 0) or 101.0
	return speed, ordered


def write_vis_pack(path: Path, vis_raw: bytes, numclusters: int) -> None:
	"""GCvis: BE header + per-cluster compressed PVS rows (Source clusters)."""
	path.parent.mkdir(parents=True, exist_ok=True)
	visbytes = (numclusters + 7) >> 3
	# GCvis\0 | ver u16 | nc u16 | vb u16 | pad u16 | row_ofs[nc] u32 | rows
	hdr = bytearray(16)
	hdr[0:5] = b"GCvis"
	struct.pack_into(">HHH", hdr, 6, 1, numclusters, visbytes)
	payload_ofs = 16 + numclusters * 4
	rows = bytearray()
	row_ofs = []
	for c in range(numclusters):
		src_ofs = struct.unpack_from("<i", vis_raw, 4 + c * 8)[0]  # PVS column
		row = _slice_compressed_row(vis_raw, src_ofs, visbytes)
		row_ofs.append(payload_ofs + len(rows))
		rows += row
	ofs_blob = b"".join(struct.pack(">I", of) for of in row_ofs)
	path.write_bytes(bytes(hdr) + ofs_blob + bytes(rows))
	print(f"  wrote {path} clusters={numclusters} visbytes={visbytes} bytes={path.stat().st_size}")


def _slice_compressed_row(vis: bytes, ofs: int, visbytes: int) -> bytes:
	"""Copy one RLE-compressed PVS row starting at ofs."""
	if ofs < 0 or ofs >= len(vis):
		return bytes([0])
	start = ofs
	filled = 0
	while filled < visbytes and ofs < len(vis):
		b = vis[ofs]
		ofs += 1
		if b == 0:
			if ofs >= len(vis):
				break
			c = vis[ofs]
			ofs += 1
			filled += c
		else:
			filled += 1
	return vis[start:ofs]


def write_vst_path(path: Path, speed: float, nodes: list[dict]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	# GCvst\0 | ver u16 | count u16 | speed f32 | reserved u16
	buf = bytearray(16)
	buf[0:5] = b"GCvst"
	struct.pack_into(">HH", buf, 6, 1, len(nodes))
	struct.pack_into(">f", buf, 10, float(speed))
	for n in nodes:
		o, a = n["origin"], n["angles"]
		# Normalize Source pitch (356 → -4)
		pitch = a[0]
		if pitch > 180.0:
			pitch -= 360.0
		yaw = a[1]
		buf += struct.pack(
			">ffffffhH",
			float(o[0]),
			float(o[1]),
			float(o[2]),
			float(pitch),
			float(yaw),
			float(a[2]),
			int(n["cluster"]),
			0,
		)
	path.write_bytes(buf)
	print(f"  wrote {path} nodes={len(nodes)} speed={speed}")


def parse_capfaces_centroids(gcp: Path) -> list[tuple[float, float, float]]:
	d = gcp.read_bytes()
	if d[:4] != b"GCcf":
		raise SystemExit(f"not GCcf: {gcp}")
	_ver, _flags, _eye_n, face_n = struct.unpack_from(">HHHH", d, 4)
	faces_ofs = struct.unpack_from(">I", d, 24)[0]
	out = []
	for i in range(face_n):
		off = faces_ofs + i * 128
		nv = d[off]
		if nv < 3:
			out.append((0.0, 0.0, 0.0))
			continue
		sx = sy = sz = 0.0
		for v in range(nv):
			x, y, z = struct.unpack_from(">hhh", d, off + 12 + v * 6)
			sx += x
			sy += y
			sz += z
		out.append((sx / nv, sy / nv, sz / nv))
	return out


def write_cls(path: Path, clusters: list[int]) -> None:
	buf = bytearray(b"GCcl")
	buf += struct.pack(">HH", 1, len(clusters))
	for c in clusters:
		buf += struct.pack(">h", int(c))
	path.write_bytes(buf)
	ok = sum(1 for c in clusters if c >= 0)
	print(f"  wrote {path} assigned={ok}/{len(clusters)}")


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--map", default="d2_lostcoast", help="CapFaces / disc map stem")
	ap.add_argument(
		"--source-bsp",
		type=Path,
		default=None,
		help="Source VBSP (default: vst_lostcoast for path, else d2_lostcoast)",
	)
	ap.add_argument("--media", type=Path, default=ROOT / "cache/bake/media")
	args = ap.parse_args()

	src_bsp = args.source_bsp
	if src_bsp is None:
		vst = ROOT / "content/lostcoast/maps/vst_lostcoast.bsp"
		d2 = ROOT / "content/lostcoast/maps/d2_lostcoast.bsp"
		src_bsp = vst if vst.is_file() else d2
	# Vis/tree from same level as CapFaces geo (d2); path ents from vst when present.
	d2_bsp = ROOT / "content/lostcoast/maps/d2_lostcoast.bsp"
	vst_bsp = ROOT / "content/lostcoast/maps/vst_lostcoast.bsp"

	print(f"==> Source tree {d2_bsp.name}")
	tree = load_source_tree(d2_bsp)
	print(f"  nodes={len(tree['nodes'])} leaves={len(tree['leaves'])} clusters={tree['numclusters']}")

	path_ents_bsp = vst_bsp if vst_bsp.is_file() else d2_bsp
	print(f"==> VST ents from {path_ents_bsp.name}")
	path_tree = load_source_tree(path_ents_bsp) if path_ents_bsp != d2_bsp else tree
	# PointInLeaf for path uses d2 tree (same world); ents from vst.
	speed, nodes = extract_vst_path(path_tree["ents"], tree)
	print(f"  path nodes={len(nodes)} speed={speed}")
	for i, n in enumerate(nodes[:3]):
		o = n["origin"]
		print(f"    [{i}] ({o[0]:.0f},{o[1]:.0f},{o[2]:.0f}) cl={n['cluster']} {n['name']}")

	media = args.media
	write_vis_pack(media / f"vis_{args.map}.gcv", tree["vis"], tree["numclusters"])
	write_vst_path(media / f"vst_{args.map}_path.bin", speed, nodes)

	# Dump-eye / info_player_start cluster for CapFaces Source PVS (not VST).
	# Runtime loads media/eye_<map>.gce so dump stills use the real leaf PVS.
	eye_origin = (1655.6, 2212.7, -48.0)
	eye_angles = (8.0, 150.0, 0.0)
	for kv in tree["ents"]:
		if kv.get("classname") == "info_player_start":
			eye_origin = tuple(float(x) for x in kv.get("origin", "1655.6 2212.7 -48").split())
			angs = [float(x) for x in kv.get("angles", "0 150 0").split()]
			while len(angs) < 3:
				angs.append(0.0)
			# HL/Source: player pitch often 0 at start; CapFaces still uses 8,150.
			eye_angles = (8.0, float(angs[1]) if len(angs) > 1 else 150.0, 0.0)
			break
	_leaf, eye_cl = point_in_leaf(eye_origin, tree["nodes"], tree["leaves"], tree["planes"])
	eye_path = media / f"eye_{args.map}.gce"
	eye_buf = bytearray(32)
	eye_buf[0:5] = b"GCeye"
	struct.pack_into(">Hh", eye_buf, 6, 1, int(eye_cl))
	struct.pack_into(
		">fffff",
		eye_buf,
		12,
		float(eye_origin[0]),
		float(eye_origin[1]),
		float(eye_origin[2]),
		float(eye_angles[0]),
		float(eye_angles[1]),
	)
	eye_path.write_bytes(eye_buf)
	print(
		f"  wrote {eye_path} eye=({eye_origin[0]:.1f},{eye_origin[1]:.1f},{eye_origin[2]:.1f}) "
		f"cl={eye_cl} leaf={_leaf}"
	)

	# CapFaces cluster sidecars for primary + spatial packs
	print("==> CapFaces cluster sidecars")
	for gcp in sorted(media.glob(f"cf_{args.map}*.gcp")):
		cents = parse_capfaces_centroids(gcp)
		clusters = []
		for c in cents:
			_leaf, cl = point_in_leaf(c, tree["nodes"], tree["leaves"], tree["planes"])
			clusters.append(cl)
		write_cls(gcp.with_suffix(".cls"), clusters)

	print("done")


if __name__ == "__main__":
	main()
