#!/usr/bin/env python3
"""Minimal Source engine MDL/VVD/VTX (dx90) mesh reader for CapFaces prop bake."""
from __future__ import annotations

import struct
from dataclasses import dataclass


@dataclass
class SourceMesh:
	positions: list[tuple[float, float, float]]
	uvs: list[tuple[float, float]]
	tris: list[tuple[int, int, int]]  # indexes into positions/uvs
	tri_mat: list[int]  # material index, parallel to tris


def _get_vvd_vert(vvd: bytes, vertex_data_start: int, index: int) -> tuple[float, float, float, float, float]:
	off = vertex_data_start + index * 48
	px, py, pz = struct.unpack_from("<fff", vvd, off + 16)
	u, v = struct.unpack_from("<ff", vvd, off + 40)
	return px, py, pz, u, v


def _vvd_lod_map(vvd: bytes, target_lod: int) -> list[int]:
	"""Map VTX origMeshVertID → absolute VVD vertex index for this LOD.

	Character models (fisherman) reorder verts with a fixup table. Props often
	have numFixups=0 and use a prefix of the buffer. Ignoring fixups scrambles
	skinned meshes into a spiked blob.
	"""
	num_lod_verts = struct.unpack_from("<8i", vvd, 16)
	num_fixups, fixup_off, _vstart, _tan = struct.unpack_from("<iiii", vvd, 48)
	if num_fixups <= 0:
		n = num_lod_verts[target_lod] if 0 <= target_lod < 8 else num_lod_verts[0]
		n = max(0, int(n))
		return list(range(n))
	ids: list[int] = []
	for i in range(num_fixups):
		lod, src, num = struct.unpack_from("<iii", vvd, fixup_off + i * 12)
		if lod < target_lod:
			continue
		ids.extend(range(src, src + num))
	return ids


def _mdl_mesh_vertex_offsets(mdl: bytes) -> list[int]:
	"""Per-mesh vertexoffset into the model's VVD stream (studiohdr).

	VTX origMeshVertID is mesh-local; add this offset before the VVD lod map.
	Without it, lower-body meshes (legs) remap onto the head and the figure
	reads as a spiked upper-body blob.
	"""
	if len(mdl) < 240 or mdl[:4] != b"IDST":
		return []
	# studiohdr numbodyparts/bodypartindex sit at 232 for v44–v48 HL2 MDLs.
	numbodyparts, bodypartindex = struct.unpack_from("<ii", mdl, 232)
	if not (1 <= numbodyparts <= 32 and 0 < bodypartindex < len(mdl)):
		return []
	offsets: list[int] = []
	for bp in range(numbodyparts):
		base = bodypartindex + bp * 16
		if base + 16 > len(mdl):
			break
		_nameindex, nummodels, _bbase, modelindex = struct.unpack_from("<iiii", mdl, base)
		if not (1 <= nummodels <= 16):
			continue
		for mi in range(nummodels):
			mh = base + modelindex + mi * 148
			if mh + 88 > len(mdl):
				continue
			_typ, _radius, nummeshes, meshindex, numvertices, _vi, _ti = struct.unpack_from(
				"<ifiiiii", mdl, mh + 64,
			)
			if not (1 <= nummeshes <= 64 and 0 < meshindex < len(mdl)):
				continue
			for mesh_i in range(nummeshes):
				msh = mh + meshindex + mesh_i * 116
				if msh + 16 > len(mdl):
					break
				_material, _modelindex_b, numverts_m, vertexoffset = struct.unpack_from(
					"<iiii", mdl, msh,
				)
				if 0 <= vertexoffset <= numvertices and 0 <= numverts_m <= numvertices:
					offsets.append(vertexoffset)
				else:
					offsets.append(0)
	return offsets


def load_source_mesh(mdl: bytes, vvd: bytes, vtx: bytes, lod: int = 2) -> SourceMesh:
	"""Return triangle mesh for the requested LOD (clamped to available)."""
	if vvd[:4] != b"IDSV":
		raise ValueError("not a VVD")
	num_lod_verts = struct.unpack_from("<8i", vvd, 16)
	_num_fixups, _fixup_off, vertex_data_start, _tan = struct.unpack_from("<iiii", vvd, 48)

	(
		_version,
		_vert_cache,
		_max_bones_strip,
		_max_bones_tri,
		_max_bones_vert,
		_checksum,
		_num_lods_hdr,
		_mat_repl,
		num_body_parts,
		body_part_offset,
	) = struct.unpack_from("<iiHHiiiiii", vtx, 0)

	# Highest available LOD index with verts.
	max_lod = 0
	for i in range(8):
		if num_lod_verts[i] > 0:
			max_lod = i
	target_lod = max(0, min(lod, max_lod))
	lod_map = _vvd_lod_map(vvd, target_lod)
	if not lod_map:
		raise ValueError(f"empty VVD lod map for lod={target_lod}")
	mesh_voff = _mdl_mesh_vertex_offsets(mdl)

	orig_ids: list[int] = []
	indices: list[int] = []
	index_mat: list[int] = []

	for bp in range(num_body_parts):
		bp_base = body_part_offset + bp * 8
		num_models, model_offset = struct.unpack_from("<ii", vtx, bp_base)
		for mi in range(num_models):
			mh = bp_base + model_offset + mi * 8
			m_num_lods, lod_offset = struct.unpack_from("<ii", vtx, mh)
			use_lod = min(target_lod, m_num_lods - 1)
			lodh = mh + lod_offset + use_lod * 12
			num_meshes, mesh_offset, _switch = struct.unpack_from("<iif", vtx, lodh)
			for mesh_i in range(num_meshes):
				meshh = lodh + mesh_offset + mesh_i * 9
				num_strip_groups, strip_group_off, _flags = struct.unpack_from("<iiB", vtx, meshh)
				vbase = mesh_voff[mesh_i] if mesh_i < len(mesh_voff) else 0
				for sg in range(num_strip_groups):
					sgh = meshh + strip_group_off + sg * 25
					(
						num_verts,
						vert_offset,
						num_indices,
						index_offset,
						_num_strips,
						_strip_offset,
						_sgflags,
					) = struct.unpack_from("<iiiiiiB", vtx, sgh)
					base = len(orig_ids)
					ok = True
					for vi in range(num_verts):
						local_orig = struct.unpack_from(
							"<H", vtx, sgh + vert_offset + vi * 9 + 4,
						)[0]
						orig = local_orig + vbase
						if orig >= len(lod_map):
							ok = False
							break
						orig_ids.append(lod_map[orig])
					if not ok:
						del orig_ids[base:]
						continue
					for ii in range(num_indices):
						local = struct.unpack_from("<H", vtx, sgh + index_offset + ii * 2)[0]
						indices.append(base + local)
						index_mat.append(mesh_i)

	# Remap unique VVD verts used by this LOD into a compact array.
	used = sorted(set(orig_ids[i] for i in indices))
	remap = {vid: i for i, vid in enumerate(used)}
	positions: list[tuple[float, float, float]] = []
	uvs: list[tuple[float, float]] = []
	for vid in used:
		px, py, pz, u, v = _get_vvd_vert(vvd, vertex_data_start, vid)
		positions.append((px, py, pz))
		uvs.append((u, v))

	tris: list[tuple[int, int, int]] = []
	tri_mat: list[int] = []
	for i in range(0, len(indices) - 2, 3):
		a, b, c = indices[i : i + 3]
		ia, ib, ic = remap[orig_ids[a]], remap[orig_ids[b]], remap[orig_ids[c]]
		if ia == ib or ib == ic or ia == ic:
			continue
		tris.append((ia, ib, ic))
		tri_mat.append(index_mat[i] if i < len(index_mat) else 0)

	return SourceMesh(positions=positions, uvs=uvs, tris=tris, tri_mat=tri_mat)


def angle_matrix(pitch: float, yaw: float, roll: float) -> list[list[float]]:
	"""Source entity angles → 3x3 (degrees)."""
	import math

	p, y, r = (math.radians(pitch), math.radians(yaw), math.radians(roll))
	sp, cp = math.sin(p), math.cos(p)
	sy, cy = math.sin(y), math.cos(y)
	sr, cr = math.sin(r), math.cos(r)
	# Source AngleMatrix order
	return [
		[cp * cy, sr * sp * cy + cr * -sy, cr * sp * cy + -sr * -sy],
		[cp * sy, sr * sp * sy + cr * cy, cr * sp * sy + -sr * cy],
		[-sp, sr * cp, cr * cp],
	]


def transform_point(
	m: list[list[float]], origin: tuple[float, float, float], p: tuple[float, float, float]
) -> tuple[float, float, float]:
	x = m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + origin[0]
	y = m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + origin[1]
	z = m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + origin[2]
	return x, y, z
