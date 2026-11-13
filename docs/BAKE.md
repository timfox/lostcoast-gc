# Source → GX bake contracts

Reuse the **shipped** xash3d-gc pack formats (not the unused CACHE/PK3 fantasy docs).

## World textures — `media/wt_<map>.gct` (GCWT)

From `engine/scripts/build-gamecube-disc.py`:

```
GCWT | entry_count u32 BE | blob_size u32 BE
entries: name[16] | w u16 | h u16 | levels u8 | flags u8 | tag u16 | offset u32
pad to 32 → concatenated CMPR mip chains (32-byte aligned)
```

- CMPR = GX DXT1, 8×8 tiles of 2×2 4×4 blocks, big-endian endpoints.
- Names are **≤15 chars** (GoldSrc / CapFaces hash). Source materials are
  shortened to a unique 15-char token; CapFaces `tex_hash` is FNV1a32 of that token.

## CapFaces — `media/cf_<map>.gcp` (GCcf)

From `engine/scripts/bake_capfaces_pack.py`:

- Magic `GCcf`, version 1, ≤384 faces, ≤8 verts/face, 128×128 RGB565 LM atlas.
- Face record 128 bytes BE; runtime `GC_TryLoadCapFacesPack` streams it.

## Smoke stub BSP

`maps/d2_lostcoast.bsp` on the disc is a **minimal GoldSrc BSP29** whose miptex
directory lists the short texture names so `Mod_GCWorldPackLoad` + CapFaces
hash resolve can bind `#wt/…` packs. Real Source geometry lives in the CapFaces pack.

## Boot

`gamecube.cfg` for the smoke disc:

```
gccapfacessmoke
gcworldrender
map d2_lostcoast
dumpframes
```
