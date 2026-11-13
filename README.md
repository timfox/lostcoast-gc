# lostcoast-gc

Isolated GameCube **smoke** for Half-Life 2: Lost Coast.

Steam Lost Coast / HL2 assets are **host-baked** into the same Flipper-native
packs used by xash3d-gc (`GCWT` / `GCcf` / CMPR). The runtime is a CapFaces
viewer on the xash3d-gc GX stack — **not** a Source engine port.

This repo does **not** overwrite HL1 or DoD artifacts
(`halflife1.dol` / `dodgc.dol` / `xash3d-gc.iso`). Deliverable:
**`out/lostcoastgc.iso`**.

## Layout

| Path | Points at |
|------|-----------|
| `engine/` | xash3d-gc checkout (symlink) |
| `content/lostcoast/` | Steam `Half-Life 2/lostcoast` (symlink) |
| `content/hl2/` | Steam `Half-Life 2/hl2` (symlink) |
| `cache/` | writable extract + bake (never mutate Steam) |
| `tools/` | Source → GX host converters |
| `out/lostcoastgc.iso` | CapFaces smoke disc |
| `playtests/` | screenshots / notes |

See [docs/LAYOUT.md](docs/LAYOUT.md) and [docs/BAKE.md](docs/BAKE.md).

## Setup

```bash
scripts/link-deps.sh
# or with overrides:
scripts/link-deps.sh \
  --engine /run/media/tim/Expansion1/xash3d-gc \
  --steam-hl2 "/home/tim/snap/steam/common/.local/share/Steam/steamapps/common/Half-Life 2"
```

## Bake + smoke disc

```bash
# 1) Extract lean materials + bake CapFaces / WT packs for d2_lostcoast
scripts/bake-lostcoast.sh

# 2) Build CapFaces-smoke DOL + ISO (requires GameCube toolchain)
scripts/build-lostcoastgc.sh

# DOL only / ISO only:
scripts/build-lostcoastgc.sh --dol-only
scripts/build-lostcoastgc.sh --iso-only
```

## Approach

1. Host-read Steam VPKs / VBSP v20 (read-only via symlink).
2. Convert VTF → CMPR (`media/wt_d2_lostcoast.gct`) and Source faces → CapFaces
   (`media/cf_d2_lostcoast.gcp`).
3. Emit a tiny GoldSrc BSP29 stub so the existing `-gcmap` / CapFaces path can
   mount the packs (geometry + materials in the packs are Source-derived).
4. Boot `lostcoastgc.dol` with `-gccapfacessmoke` for a Dolphin CapFaces still.

Milestone 1 success: CapFaces drawn with CMPR binds in Dolphin — not walkable
HL2 gameplay, HDR, or Source physics.
