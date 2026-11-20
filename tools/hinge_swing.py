"""How a hinged physics prop swings when the map pushes it.

Lost Coast's harbor gate is two prop_physics leaves, each on a phys_hinge,
blown open by an env_physexplosion when the fisherman's OpenGate animation
fires its script event. The port has no rigid-body solver, so the bake works
the swing out from the same entities and hands the game three numbers per
leaf: where the hinge is, how fast the leaf starts turning, and the angle at
which it meets the world.

The push follows the engine's own arithmetic (CPhysExplosion::Explode,
CalculateExplosiveDamageForce):

  radius   = "radius", or magnitude * 2.5 when that is 0
  damage   = magnitude - distance / 2.5      (distance to the leaf's centre)
  impulse  = damage * 75 * 4                 (ImpulseScale( 75 kg, 4 in/s ))
             along explosion -> centre, applied at the explosion's origin

and the hinge turns what is left of it into spin about the vertical axis:

  spin     = (r x impulse).z / I             r from the hinge to that origin
  I        = mass * width^2 / 3              a plate turning about one edge,
                                             mass from the model's .phy

"hingefriction" 1 against that inertia is nothing (a turn in several
minutes), so the leaf swings at that speed until something stops it: the
stop is the first angle at which the leaf, as a plate from the hinge to its
far edge, cuts a world face, a displacement or a static prop's hull.
"""
from __future__ import annotations

import math
import re
import struct

import numpy as np

from source_disp import parse_dispinfos, tessellate_disp


def _lump(d: bytes, i: int) -> bytes:
	o, n = struct.unpack_from("<ii", d, 8 + i * 16)
	return d[o:o + n]


def world_tris(bsp: bytes, centre, reach: float) -> list:
	"""World faces and displacements with a corner within reach of centre."""
	V = np.frombuffer(_lump(bsp, 3), "<f4").reshape(-1, 3)
	E = np.frombuffer(_lump(bsp, 12), "<u2").reshape(-1, 2)
	SE = np.frombuffer(_lump(bsp, 13), "<i4")
	F = _lump(bsp, 7)
	c = np.asarray(centre[:2], np.float64)
	tris = []
	pts_of = {}
	for i in range(len(F) // 56):
		fe, ne = struct.unpack_from("<ih", F, i * 56 + 4)
		disp = struct.unpack_from("<h", F, i * 56 + 12)[0]
		p = V[[E[s][0] if s >= 0 else E[-s][1] for s in SE[fe:fe + ne]]].astype(np.float64)
		if disp != -1:
			pts_of[i] = p
			continue
		if np.all(np.linalg.norm(p[:, :2] - c, axis=1) > reach):
			continue
		for k in range(1, ne - 1):
			tris.append((p[0], p[k], p[k + 1]))
	for disp in parse_dispinfos(bsp):
		p = pts_of.get(disp["map_face"])
		if p is None or np.all(np.linalg.norm(p[:, :2] - c, axis=1) > reach + 512):
			continue
		for q in tessellate_disp(bsp, disp, [tuple(x) for x in p]):
			q = np.asarray(q, np.float64)
			if np.all(np.linalg.norm(q[:, :2] - c, axis=1) > reach):
				continue
			tris.append((q[0], q[1], q[2]))
			tris.append((q[0], q[2], q[3]))
	return tris


def _seg_tri(a, b, t) -> bool:
	e1, e2, dv = t[1] - t[0], t[2] - t[0], b - a
	h = np.cross(dv, e2)
	det = float(e1 @ h)
	if abs(det) < 1e-9:
		return False
	s = a - t[0]
	u = float(s @ h) / det
	if u < 0 or u > 1:
		return False
	q = np.cross(s, e1)
	v = float(dv @ q) / det
	if v < 0 or u + v > 1:
		return False
	return 0 <= float(e2 @ q) / det <= 1


def box_tris(org, angles, lo, hi) -> list:
	"""A model's hull box where a prop stands (yaw only), as triangles."""
	yaw = math.radians(angles[1])
	c, s = math.cos(yaw), math.sin(yaw)
	P = []
	for x in (lo[0], hi[0]):
		for y in (lo[1], hi[1]):
			for z in (lo[2], hi[2]):
				P.append(np.array([org[0] + x * c - y * s, org[1] + x * s + y * c, org[2] + z]))
	quads = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
	return [(P[a], P[b], P[c2]) for a, b, c2, d in quads] + [(P[a], P[c2], P[d]) for a, b, c2, d in quads]


def phy_mass(phy: bytes | None) -> float | None:
	m = re.search(rb'"totalmass"\s+"([0-9.]+)"', phy or b"")
	return float(m.group(1)) if m else None


def swing(bsp: bytes, hinge, centre, zlo: float, zhi: float, mass: float, explosion: dict, solids=()):
	"""(degrees a second, signed; stop angle in degrees) of one leaf, or None.

	hinge: a point on the vertical hinge axis; centre: the leaf's centre;
	zlo..zhi: its height; explosion: the env_physexplosion entity."""
	eo = np.array([float(x) for x in explosion["origin"].split()])
	mag = float(explosion.get("magnitude", 0) or 0)
	radius = float(explosion.get("radius", 0) or 0) or mag * 2.5
	centre = np.asarray(centre, np.float64)
	to = centre - eo
	dist = float(np.linalg.norm(to))
	damage = mag - dist / 2.5
	if dist > radius or damage <= 0 or mass <= 0:
		return None
	imp = to / dist * damage * 75.0 * 4.0
	r = eo[:2] - np.asarray(hinge[:2], np.float64)
	arm = centre[:2] - np.asarray(hinge[:2], np.float64)
	width = 2.0 * float(np.linalg.norm(arm))
	inertia = mass * width * width / 3.0
	spin = math.degrees(float(r[0] * imp[1] - r[1] * imp[0]) / inertia)
	if abs(spin) < 1.0:
		return None
	# The stop: turn the plate a degree at a time until it cuts the world.
	tris = world_tris(bsp, hinge, width + 32.0) + list(solids)
	base = math.atan2(arm[1], arm[0])
	sign = 1.0 if spin > 0 else -1.0
	stop = 180.0

	def cut(deg, among):
		a = base + sign * math.radians(deg)
		ca, sa = math.cos(a), math.sin(a)
		out = []
		# (from a quarter of the way up: the bottom rail sits on the ground)
		for z in np.linspace(zlo + (zhi - zlo) * 0.25, zhi - 2.0, 5):
			p0 = np.array([hinge[0] + 4.0 * ca, hinge[1] + 4.0 * sa, z])
			p1 = np.array([hinge[0] + width * ca, hinge[1] + width * sa, z])
			out += [i for i in among if _seg_tri(p0, p1, tris[i])]
		return out

	# What the leaf already stands in when shut (its own frame, a rock the
	# map leans on its top corner) does not stop it in the game either.
	shut = set(cut(0.0, range(len(tris))))
	free = [i for i in range(len(tris)) if i not in shut]
	for deg in range(2, 181):
		if cut(deg, free):
			stop = float(deg - 1)
			break
	return spin, stop


def swings_of(bsp: bytes, ents: list[dict], explosion_name: str, read, static_props=()) -> list:
	"""Every hinged prop the named env_physexplosion pushes:
	[((x, y, z) of the hinge, degrees a second, stop angle, shut box)].

	ents: the map's entities as dicts; read(path) -> file bytes or None;
	static_props: [(model, origin, angles)], which stop a leaf with their hulls."""
	ex = next((e for e in ents if e.get("classname") == "env_physexplosion"
		and e.get("targetname", "").lower() == explosion_name.lower()), None)
	out = []
	if not ex:
		return out
	eo = [float(x) for x in ex["origin"].split()]
	hinges = [[float(x) for x in h["origin"].split()] for h in ents if h.get("classname") == "phys_hinge"]
	solids = []
	for name, org, ang in static_props:
		if abs(org[0] - eo[0]) > 256 or abs(org[1] - eo[1]) > 256:
			continue
		m = read(name.lower())
		if not m:
			continue
		lo, hi = struct.unpack_from("<3f", m, 104), struct.unpack_from("<3f", m, 116)
		# The frame the leaves hang in is a box around both hinges: a hull
		# box is the wrong shape for it (it is hollow), and it holds the
		# hinge rather than stopping the leaf.
		yaw = math.radians(ang[1])
		inside = False
		for hp in hinges:
			dx, dy = hp[0] - org[0], hp[1] - org[1]
			lx, ly = dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)
			inside |= lo[0] - 1 <= lx <= hi[0] + 1 and lo[1] - 1 <= ly <= hi[1] + 1
		if not inside:
			solids += box_tris(org, ang, lo, hi)
	for h in ents:
		if h.get("classname") != "phys_hinge":
			continue
		prop = next((e for e in ents if e.get("targetname", "").lower() == h.get("attach1", "").lower()
			and e.get("classname", "").startswith("prop_")), None)
		if not prop or "model" not in prop:
			continue
		hp = [float(x) for x in h["origin"].split()]
		ha = [float(x) for x in h.get("hingeaxis", h["origin"]).split()]
		if abs(ha[0] - hp[0]) > 1 or abs(ha[1] - hp[1]) > 1:
			continue  # only upright hinges
		mdl = read(prop["model"].lower())
		if not mdl:
			continue
		lo = np.array(struct.unpack_from("<3f", mdl, 104))
		hi = np.array(struct.unpack_from("<3f", mdl, 116))
		org = np.array([float(x) for x in prop["origin"].split()])
		yaw = math.radians(float(prop.get("angles", "0 0 0").split()[1]))
		c = (lo + hi) * 0.5
		centre = org + np.array([c[0] * math.cos(yaw) - c[1] * math.sin(yaw), c[0] * math.sin(yaw) + c[1] * math.cos(yaw), c[2]])
		mass = phy_mass(read(prop["model"].lower()[:-4] + ".phy"))
		if not mass:
			continue
		got = swing(bsp, hp, centre, org[2] + lo[2], org[2] + hi[2], mass, ex, solids)
		if got:
			# and the leaf's box while it is shut: what stands in the way
			cy, sy = math.cos(yaw), math.sin(yaw)
			pts = [(org[0] + x * cy - y * sy, org[1] + x * sy + y * cy) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]
			box = (min(p[0] for p in pts), min(p[1] for p in pts), org[2] + lo[2],
				max(p[0] for p in pts), max(p[1] for p in pts), org[2] + hi[2])
			out.append((tuple(hp), got[0], got[1], box))
	return out
