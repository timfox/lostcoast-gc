"""Validate GCCM v5 records before binding a controlled GX reflection view."""
import argparse
import json
import struct
from pathlib import Path


def records(data):
    if len(data) < 32:
        raise ValueError("truncated GCCM header")
    magic, version, count, side, k, kup, small, nup = struct.unpack_from(
        ">4s7H", data)
    if magic != b"GCCM" or version != 5:
        raise ValueError("expected GCCM v5")
    if not (0 < small <= side <= 1024 and side % 8 == small % 8 == 0
            and k and kup and nup <= count):
        raise ValueError("invalid GCCM dimensions or projection")
    offset = 32 + ((count * 2 + 31) & ~31)
    if offset > len(data):
        raise ValueError("truncated GCCM directory")
    keys = struct.unpack_from(f">{count}H", data, 32)
    if len(set(keys)) != count or sum(key < 256 for key in keys) != nup:
        raise ValueError("duplicate views or invalid up-view count")
    result = []
    for i, key in enumerate(keys):
        cube, axis = key & 255, key >> 8
        if axis > 16 or (i < nup) != (axis == 0):
            raise ValueError("invalid axis or directory ordering")
        size = side if axis == 0 else small
        length = (size * size // 2 + 31) & ~31
        if offset + length > len(data):
            raise ValueError("truncated CMPR view")
        result.append(dict(cube=cube, axis=axis, side=size, offset=offset,
                           bytes=length, projection_k=(kup if axis == 0 else k) / 256))
        offset += length
    if offset != len(data):
        raise ValueError("unexpected trailing GCCM data")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pack", type=Path)
    ap.add_argument("--cube", type=int)
    ap.add_argument("--axis", type=int)
    args = ap.parse_args()
    views = records(args.pack.read_bytes())
    print(json.dumps([v for v in views if
                     (args.cube is None or v['cube'] == args.cube) and
                     (args.axis is None or v['axis'] == args.axis)], indent=2))
