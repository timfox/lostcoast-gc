"""Edge-collapse simplification by quadric error (Garland & Heckbert).

The prop bake's far chains and capped models used to be made by welding
vertices on a grid: every vertex in a cell went to the cell's average, so a
hull came out lumpy and lost triangles wherever a cell swallowed a whole
face. Collapsing the cheapest edge first keeps the silhouette: a vertex only
ever moves onto one of its neighbours (no new positions, so texture
coordinates stay those of the source mesh), flat areas go first, and a
collapse that would turn a face over is refused.

simplify(positions, uvs, tris, mats, target) -> (positions, uvs, tris, mats)
with the same shapes the bake uses: per-vertex position and uv, index
triples, one material per triangle.
"""
from __future__ import annotations

import heapq

import numpy as np


def simplify(positions, uvs, tris, mats, target: int):
	if len(tris) <= target or len(tris) < 4:
		return positions, uvs, tris, mats
	P = np.asarray(positions, dtype=np.float64)
	ext = float(np.ptp(P, axis=0).max()) or 1.0
	# Weld by position: the source splits a vertex at every texture seam.
	weld: dict = {}
	wid = []
	wpos = []
	for p in P:
		k = tuple(np.round(p / ext * 20000.0).astype(np.int64))
		i = weld.get(k)
		if i is None:
			i = weld[k] = len(wpos)
			wpos.append(p)
		wid.append(i)
	wpos = np.asarray(wpos)
	nv = len(wpos)
	# Faces as welded vertex triples, with the source vertex of each corner.
	fv = [[wid[a], wid[b], wid[c]] for a, b, c in tris]
	fc = [list(t) for t in tris]
	alive = [f[0] != f[1] and f[1] != f[2] and f[2] != f[0] for f in fv]
	vfaces: list[set] = [set() for _ in range(nv)]
	for fi, f in enumerate(fv):
		if alive[fi]:
			for v in f:
				vfaces[v].add(fi)

	def normal(a, b, c):
		n = np.cross(wpos[b] - wpos[a], wpos[c] - wpos[a])
		l = float(np.linalg.norm(n))
		return (n / l if l > 1e-20 else n), l

	Q = np.zeros((nv, 4, 4))
	edge_faces: dict = {}
	for fi, f in enumerate(fv):
		if not alive[fi]:
			continue
		n, l = normal(*f)
		if l <= 1e-20:
			continue
		pl = np.append(n, -float(n @ wpos[f[0]]))
		K = np.outer(pl, pl) * (0.5 * l)
		for v in f:
			Q[v] += K
		for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
			edge_faces.setdefault((min(a, b), max(a, b)), []).append(fi)
	# Open edges and material borders are held by a plane through the edge,
	# square to its face: without it a sheet shrinks from its rim and a
	# painted stripe wanders across the hull.
	for (a, b), fl in edge_faces.items():
		if len(fl) == 1 or len({mats[fi] for fi in fl}) > 1:
			for fi in fl:
				n, l = normal(*fv[fi])
				e = wpos[b] - wpos[a]
				el = float(np.linalg.norm(e))
				if l <= 1e-20 or el <= 1e-20:
					continue
				pn = np.cross(e / el, n)
				pl = np.append(pn, -float(pn @ wpos[a]))
				K = np.outer(pl, pl) * (el * el * 4.0)
				Q[a] += K
				Q[b] += K

	stamp = [0] * nv
	gone = [False] * nv

	def cost(u, v):
		"""Error of moving u onto v."""
		p = np.append(wpos[v], 1.0)
		return float(p @ (Q[u] + Q[v]) @ p)

	heap = []

	def push(u, v):
		heapq.heappush(heap, (cost(u, v), u, v, stamp[u], stamp[v]))

	def neighbours(u):
		out = set()
		for fi in vfaces[u]:
			out.update(fv[fi])
		out.discard(u)
		return out

	for (a, b) in edge_faces:
		push(a, b)
		push(b, a)
	live = sum(alive)
	while live > target and heap:
		_c, u, v, su, sv = heapq.heappop(heap)
		if gone[u] or gone[v] or stamp[u] != su or stamp[v] != sv:
			continue
		shared = vfaces[u] & vfaces[v]
		if not shared:
			continue
		# Refuse a collapse that turns a surviving face over or flattens it.
		ok = True
		for fi in vfaces[u] - shared:
			f = fv[fi]
			n0, l0 = normal(*f)
			g = [v if x == u else x for x in f]
			n1, l1 = normal(*g)
			if l1 <= 1e-20 or float(n0 @ n1) < 0.2:
				ok = False
				break
		if not ok:
			continue
		# Corners at u take v's source vertex where they sat on the same
		# side of a seam as a face that had both; the others keep theirs.
		swap = {}
		for fi in shared:
			f, c = fv[fi], fc[fi]
			swap[c[f.index(u)]] = c[f.index(v)]
		for fi in shared:
			alive[fi] = False
			live -= 1
			for x in fv[fi]:
				vfaces[x].discard(fi)
		for fi in list(vfaces[u]):
			f, c = fv[fi], fc[fi]
			k = f.index(u)
			f[k] = v
			c[k] = swap.get(c[k], c[k])
			vfaces[v].add(fi)
		vfaces[u] = set()
		gone[u] = True
		Q[v] += Q[u]
		stamp[v] += 1
		for n in neighbours(v):
			push(n, v)
			push(v, n)
	# Corners keep the texture coordinate of their source vertex, at the
	# position of the welded vertex they ended on.
	out_pos, out_uv, out_tris, out_mats = [], [], [], []
	index: dict = {}
	for fi, f in enumerate(fv):
		if not alive[fi]:
			continue
		tri = []
		for v, c in zip(f, fc[fi]):
			key = (v, c)
			i = index.get(key)
			if i is None:
				i = index[key] = len(out_pos)
				out_pos.append(tuple(float(x) for x in wpos[v]))
				out_uv.append(uvs[c])
			tri.append(i)
		out_tris.append(tuple(tri))
		out_mats.append(mats[fi])
	return out_pos, out_uv, out_tris, out_mats
