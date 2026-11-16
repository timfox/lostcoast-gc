# Lost Coast GameCube status — November 14, 2026

## Scope

Lost Coast and Day of Defeat share the Xash3D GameCube engine. Lost Coast-specific Source asset conversion and GX rendering features bridge the map to the hardware; the project is not a general Source Engine implementation. Keep game-specific assets, configuration and behavior separate.

## What is working

- The Lost Coast map is converted into GameCube-oriented world, texture, material and prop packs, and the on-rails VST camera path runs in Dolphin.
- Water has a reflected appearance and a depth-driven shoreline treatment. Coplanar packed-water overlaps were removed offline, but motion has not established that every reported water flicker is fixed.
- GX has lightmapped and vertex-lit world paths, streamed lightmap pages, a material-detail stage, and masked additive lighting derived in part from Source normal-map alpha. These are approximations, not a full Source normal-map or cubemap shader.
- A vertex-light scaling correction reduced the large rectangular dark patch on the tower. Matching displacement edges in the world bake now receive limited lighting reconciliation, reducing the abrupt beach seam. The nearby horizontal rock band and a broader sand transition remain visible.
- The shared engine compiled with the DoD build profile after the lighting change. This was a compile check, not a DoD gameplay test.

## Open work

- Match the startup fade, verify lighting and normal-map highlights against comparable retail camera views, and improve the remaining rock and sand seams.
- Full-resolution texture data can exist in ARAM while the existing fixed-size texture slots select a lower mip. A local renderer candidate attempts to promote large nearby chains into the small hot window; it is experimental until captured and checked. Memory limits mean all textures cannot necessarily remain at their highest mip simultaneously.
- Lightmap cache recycling prevents an abrupt fallback after cache exhaustion but adds a GPU synchronization point. Dolphin measurements are not GameCube hardware performance; measure on hardware before accepting a performance conclusion.
- More of the full VST path, normal gameplay, sound, input, map transitions and real-hardware stability still need verification. Do not describe the port as retail-identical or complete.

## Evidence

- [Prior VST capture and manifest](../playtests/notes/dolphin-probe-20261114-144409/capture-manifest.md)
- [Tower-lighting before/after](../playtests/notes/dolphin-probe-20261114-143917/vertex-light-comparison.jpg)
- [Sand-seam before/after](../playtests/notes/dolphin-probe-20261114-143450/sand-seam-comparison.jpg)
- [Fresh baseline VST and capture manifest](../playtests/notes/dolphin-probe-20261114-191617/capture-manifest.md): reached segment 5 in a bounded Dolphin run. A separate experimental disc failed to mount its DVD filesystem before map load; the cause has not yet been established.

## November 14 normal and disc investigation

The Source displacement vector was being used as the GX shading normal. `tools/bake_worlddl.py` now computes geometric normals from neighboring displaced positions and orients them to the face. Synthetic checks cover a sloped plane, reversed face orientation and degenerate geometry. A candidate world pack was baked under `tmp/normal-seam-candidate`; it has not been promoted or visually verified. Geometric normals correct coarse surface lighting; they do not implement view-dependent, per-pixel Source normal-map reflections.

Freshly packaged discs failed DVD filesystem mounting even when built with the previously verified DOL (`out/normal-seam-packaging-probe.iso`, probe `playtests/notes/dolphin-probe-20261114-193412/`). The prior `out/rock-light-seams-vst.iso` mounts, so the new disc image/packaging path needs investigation before a candidate VST can validate the change. The fresh build also staged DoD-named viewmodel assets; determine why and remove unintended cross-game assets before treating it as a Lost Coast release build.

## November 14 shared geometry and camera investigation

GX world display lists now invalidate the indexed vertex cache after switching chunk position arrays. The previous code omitted this documented invalidation in both the ordinary and skyline world paths. This is a plausible source of intermittent cross-game stray geometry, but the reported white object has not yet been reproduced and verified fixed in HL1 or DoD. Both Lost Coast and DoD profiles compile with the shared change.

VST `ptrack_vst_13` and `ptrack_vst_14` use Source's face-path orientation mode; the previous bake incorrectly used stored entity angles. The path bake now derives their direction from the following track and the shared VST player applies a smoothstep to angles, approximating the train's eased turns. The turn screenshots are in `playtests/notes/dolphin-probe-20261114-194733/`. The Source implementation uses quaternion interpolation, so the current eased Euler interpolation is still an approximation.

The Source prop model reader also now respects individual VTX strip boundaries and triangle-strip winding. Representative boat and fisherman assets parse successfully; it is not established that this was the reported white object. Diagnostic discs are Dolphin-only until the ordinary disc packaging/mount issue is fixed.

## Normal-map investigation

The current world specular pass samples the alpha mask but does not use normal RGB for per-pixel reflection. A fixed-camera test combining directional normal RGB with that mask affected fewer than 1% of image pixels by more than 5/255 and did not provide view-dependent reflection; it was not promoted. A GX per-vertex specular-channel experiment, also with an isolated geometric-normal transfer, was similarly inconclusive at that view and was not promoted. See `playtests/notes/normal-direction-20261114/manifest.md` for frame captures and limits.

`tools/bake_tevmat.py` now exposes `normal_xy_ia8`, which area-filters complete normal maps into GX IA8 tiled X/Y data, preserving the independent alpha mask. This is groundwork only: no renderer consumes it yet. Correct view-dependent reflection needs runtime texture use, surface tangent handling, and suitable environment-map sampling, followed by matched-angle validation and memory/performance checks.

## Cliff tiling investigation

The displayed tower crosses several authored rockwall021 material variants whose base textures have different brightness. Removing the shared cliff detail layer had almost no effect in the opening view. The lightmap baker now resamples atlas cells bilinearly and includes their endpoint samples; the corresponding filtered pages were applied to the current cache after verifying that its page ordering matched the controlled capture. A broader cross-material lighting stitch was tested in Dolphin but not adopted because the main material band remained. Captures and hashes: `playtests/notes/cliff-tiling-20261114/manifest.md`.

## November 15 diagnostic provenance correction

Several November 14 patched diagnostic discs updated the DOL header but kept the original apploader, which actually loaded the original executable. Those captures do not verify the appended DOL or its runtime changes. Data-only comparisons remain useful. New diagnostics must rebuild the apploader and confirm section reads in Dolphin before claiming a renderer build was tested.

## Prop cache experiment, November 15

A bounded 32 KiB prop-chain cache is now enabled only on `d2_lostcoast`, with GPU fencing on cache eviction and map-arena reuse. In three matching opening-path Dolphin intervals, the instrumented total prop stage improved by approximately 0.4–0.5 ms per frame; total frame time changed little. See [capture and validation notes](../playtests/notes/prop-cache-20261115/manifest.md). Lost Coast and DoD profiles compile; map transitions, DoD gameplay, and physical hardware remain untested. The cache uses scarce map-arena memory and must be revisited when new rendering features are added.

## Cliff lightmap atlas investigation, November 15

The bake's 32×32 per-cluster atlas overflows for several large cliff clusters, causing the whole lightmap page to be dropped and the renderer to fall back to vertex lighting. A smaller-cell diagnostic restored three pages, but the current renderer suppresses the authored detail stage for lightmapped chunks; a follow-up test that restored detail on the upper cliff still rendered the lower face too dark and smooth. Neither diagnostic has been promoted. A whole-map retry prototype would exchange 23 of the 512 page assignments and needs broader visual validation. See [cliff atlas investigation](../playtests/notes/cliff-atlas-investigation-20261115/manifest.md).

A follow-up neutral-diffuse control exposed broad rear-cliff shading discontinuities, and a global shadow lift changed the cliff only slightly while also affecting sand. Neither was adopted. Both HDR and LDR lighting lumps exist in the BSP; the sampled cliff-face bytes are nearly identical. See [rear-cliff lighting audit](../playtests/notes/cliff-lighting-audit-20261115/manifest.md).
