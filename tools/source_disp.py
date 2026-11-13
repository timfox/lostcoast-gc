#!/usr/bin/env python3
"""Tessellate Source VBSP displacements into CapFaces-ready quads."""
from __future__ import annotations

import struct
from typing import Iterable

# ddispinfo_t is 176 bytes in VBSP v20 (Lost Coast / HL2).
DISPINFO_SIZE = 176
DISPVERT_SIZE = 20  # CDispVert: vec3 + dist + alpha


def _lerp(a: tuple[float, float, float], b: tuple[float, float, float], t: float):
	return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def parse_dispinfos(bsp_data: bytes) -> list[dict]:
	off, sz = struct.unpack_from("<II", bsp_data, 8 + 26 * 16)
	if sz <= 0 or sz % DISPINFO_SIZE:
		return []
	out = []
	for i in range(sz // DISPINFO_SIZE):
		o = off + i * DISPINFO_SIZE
		start = struct.unpack_from("<fff", bsp_data, o)
		vert_start, tri_start, power = struct.unpack_from("<iii", bsp_data, o + 12)
		map_face = struct.unpack_from("<H", bsp_data, o + 36)[0]
		out.append(
			{
				"index": i,
				"start": start,
				"vert_start": vert_start,
				"tri_start": tri_start,
				"power": power,
				"map_face": map_face,
			}
		)
	return out


def _disp_vert(bsp_data: bytes, vert_lump_off: int, index: int):
	o = vert_lump_off + index * DISPVERT_SIZE
	vec = struct.unpack_from("<fff", bsp_data, o)
	dist, alpha = struct.unpack_from("<ff", bsp_data, o + 12)
	return vec, dist, alpha


def _ordered_corners(face_pts: list, start: tuple[float, float, float]) -> list:
	"""Wind face verts so C0 == startPosition, then CCW/CW around the quad."""
	if len(face_pts) < 4:
		return []
	idx = min(
		range(4),
		key=lambda i: (face_pts[i][0] - start[0]) ** 2
		+ (face_pts[i][1] - start[1]) ** 2
		+ (face_pts[i][2] - start[2]) ** 2,
	)
	return [face_pts[(idx + k) % 4] for k in range(4)]


def tessellate_disp(
	bsp_data: bytes,
	disp: dict,
	face_pts: list,
	*,
	max_power: int | None = None,
) -> list[list[tuple[float, float, float]]]:
	"""Return list of 4-vert quads (world space) for one displacement."""
	power = int(disp["power"])
	if max_power is not None:
		power = min(power, max_power)
	n = (1 << power) + 1
	full_n = (1 << int(disp["power"])) + 1
	corners = _ordered_corners(face_pts, disp["start"])
	if len(corners) < 4:
		return []
	vert_off, _ = struct.unpack_from("<II", bsp_data, 8 + 33 * 16)
	# Build grid; when downsampling power, sample the full grid.
	step = max(1, (full_n - 1) // (n - 1))
	grid: list[list[tuple[float, float, float]]] = []
	for j in range(n):
		row = []
		fj = j * step
		v = fj / (full_n - 1)
		for i in range(n):
			fi = i * step
			u = fi / (full_n - 1)
			end = _lerp(_lerp(corners[0], corners[1], u), _lerp(corners[3], corners[2], u), v)
			vid = fj * full_n + fi
			vec, dist, _alpha = _disp_vert(bsp_data, vert_off, disp["vert_start"] + vid)
			row.append((end[0] + vec[0] * dist, end[1] + vec[1] * dist, end[2] + vec[2] * dist))
		grid.append(row)
	quads = []
	for j in range(n - 1):
		for i in range(n - 1):
			quads.append([grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]])
	return quads


def iter_disp_quads(
	bsp: dict,
	bsp_data: bytes,
	*,
	max_power: int | None = 3,
) -> Iterable[dict]:
	"""Yield {pts, mat, texinfo, face, cx,cy,cz, area} for each tessellated quad."""
	from bake_lostcoast_map import face_verts  # local import to avoid cycles at module load

	disps = parse_dispinfos(bsp_data)
	for disp in disps:
		fi = disp["map_face"]
		if fi < 0 or fi >= len(bsp["faces"]):
			continue
		face = bsp["faces"][fi]
		if face["dispinfo"] != disp["index"]:
			# Some compilers store absolute index match; still try mapFace.
			pass
		ti = face["texinfo"]
		if ti < 0 or ti >= len(bsp["texinfos"]):
			continue
		texinfo = bsp["texinfos"][ti]
		td = texinfo["texdata"]
		if td < 0 or td >= len(bsp["tex_names"]):
			continue
		mat = bsp["tex_names"][td]
		if mat.startswith("tools/"):
			continue
		pts4 = face_verts(bsp, face)
		quads = tessellate_disp(bsp_data, disp, pts4, max_power=max_power)
		for q in quads:
			cx = sum(p[0] for p in q) / 4.0
			cy = sum(p[1] for p in q) / 4.0
			cz = sum(p[2] for p in q) / 4.0
			# parallelogram area approx
			e0 = (q[1][0] - q[0][0], q[1][1] - q[0][1], q[1][2] - q[0][2])
			e1 = (q[3][0] - q[0][0], q[3][1] - q[0][1], q[3][2] - q[0][2])
			nx = e0[1] * e1[2] - e0[2] * e1[1]
			ny = e0[2] * e1[0] - e0[0] * e1[2]
			nz = e0[0] * e1[1] - e0[1] * e1[0]
			area = (nx * nx + ny * ny + nz * nz) ** 0.5
			yield {
				"pts": q,
				"mat": mat,
				"texinfo": texinfo,
				"face": face,
				"cx": cx,
				"cy": cy,
				"cz": cz,
				"area": max(1.0, area),
				"n": (nx, ny, nz),
			}
