#!/usr/bin/env python3
"""Replace a GameCube ISO's DOL and rebuild its apploader in a new image.

Preserves the source ISO9660 filesystem and FST. The DOL is appended at a
32-byte boundary and the apploader is rebuilt for its actual section offsets.
This is a diagnostic utility, not a substitute for full disc packaging.
"""
import argparse
import hashlib
import importlib.util
import struct
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--iso', type=Path, required=True)
p.add_argument('--dol', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.resolve() in (a.iso.resolve(), a.dol.resolve()):
    raise SystemExit('output must not overwrite the input ISO or DOL')
path = root / 'engine/scripts/build-gamecube-disc.py'
spec = importlib.util.spec_from_file_location('xash_disc_builder', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
image = bytearray(a.iso.read_bytes())
if len(image) < 0x8000 or struct.unpack_from('>I', image, 0x1c)[0] != module.DISC_MAGIC:
    raise SystemExit('input is not a GameCube disc image')
_, fst_offset, fst_size, _ = struct.unpack_from('>4I', image, 0x420)
if fst_offset + fst_size > len(image):
    raise SystemExit('FST extends beyond input image')
image.extend(bytes((-len(image)) % 32))
dol_offset = len(image)
dol = a.dol.read_bytes()
image.extend(dol)
apploader = module.build_apploader(path.parent / 'gamecube-apploader.c', path.parent / 'gamecube-apploader.ld', a.dol, dol_offset, fst_offset, fst_size)
struct.pack_into('>I', image, 0x420, dol_offset)
struct.pack_into('>I', image, 0x2454, len(apploader))
image[module.APPLOADER_DATA_OFFSET:module.DISC_HEADER_SIZE] = bytes(module.DISC_HEADER_SIZE - module.APPLOADER_DATA_OFFSET)
image[module.APPLOADER_DATA_OFFSET:module.APPLOADER_DATA_OFFSET + len(apploader)] = apploader
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_bytes(image)
print('DOL SHA-256', hashlib.sha256(dol).hexdigest())
print('ISO SHA-256', hashlib.sha256(image).hexdigest())
print(f'DOL offset {dol_offset:#x}; FST offset {fst_offset:#x}; apploader {len(apploader)} bytes')
