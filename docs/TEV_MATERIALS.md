# Source shaders → Flipper TEV recipes

Lost Coast's world geometry uses two Source shaders. The GameCube has no
programmable shaders, so each shader *feature* is ported as a fixed TEV
recipe (stages, texgens, constant registers) selected per material.

Inventory of the materials the bake touches (`cache/extracted/materials`):

| Source shader | Materials | Features on top of diffuse × lightmap |
|---|---|---|
| LightmappedGeneric | 71 | `$detail` 20, `$envmap` 27 (`$basealphaenvmapmask` 20, `$normalmapalphaenvmapmask` 9), `$bumpmap` 27, `$parallaxmap` 5, `$alphatest` 2 |
| WorldVertexTransition | 9 | `$basetexture2` 9, `$bumpmap` 7, `$envmap` 3, `$parallaxmap` 1, `$detail` 1 |

## Pipeline

1. `tools/bake_tevmat.py` classifies each VMT and writes `media/tevmat.gcm`
   (material records + shared detail tiles + optional `GCem` I4 envmask
   trailer). `bake_lostcoast_map.py` stores the material id in CapFaces
   face byte 127; `bake_worlddl.py` stores it on each world-DL mat
   (resolving VBSP `maps/<map>/…_x_y_z` patch names to the base VMT).
2. `vid_gamecube.c` passes CapFaces byte 127 through as `surf->styles[1]`.
3. `ref/gx/r_gx_worlddl.c` loads the table and appends detail / alphatest
   after diffuse × lightmap; `$envmap` materials take a second additive
   pass (sand IndTex, or base-alpha I4 mask × sun disc).

Disable at runtime with `-gcnotevmat`.

## Recipes

| Source feature | Status | TEV recipe | Cost |
|---|---|---|---|
| diffuse × lightmap | done (engine) | s0 `TEXC`, s1 `CPREV × TEXC(lm)` ×1/×2 | 2 stages |
| `$detail` mode 0 (mod2x), `$detailscale`, `$detailblendfactor` | **done** | one stage `CPREV × TEXC(detail) × 2`; UV = diffuse ST through a scale texture matrix (no vertex data); blend factor folded into the I8 texel around 0.5; mips fade to neutral | +1 stage, +1 texgen, 5.4 KB per detail texture (max 4) |
| `$alphatest`, `$alphatestreference` | **done** | CMPR 1-bit alpha, `GX_SetAlphaCompare(GEQUAL, ref)`, Z after texture | 0 stages |
| `$envmap` + `$normalmapalphaenvmapmask` + `$bumpmap` (wet sand) | done (engine, sand only) | sky/sun texture × (mask × `$envmaptint`) added to lit colour, sky ST warped by the normal map through an indirect stage | +2 stages, 1 ind stage |
| `$envmap` + `$basealphaenvmapmask` (metal, plaster, frescos) | **done** | additive pass: sun disc × `$envmaptint` × shared 64² I4 mask from base alpha (CMPR alpha is 1-bit); trailer `GCem` in `tevmat.gcm` (max 4 masks) | +2 stages, 2 KB per mask |
| `$bumpmap` without envmap (rock, cliffs) | next | emboss: `lit × (0.5 + h(uv) − h(uv + d)) × 2` with `h` = normal-map-derived height (I8), `d` = per-face sun direction in tangent space loaded as a texture matrix | +2 stages, +1 texgen |
| WorldVertexTransition `$basetexture2` | **done** | second alpha-blended draw of `$basetexture2` weighted by displacement vertex alpha (world DL v2) | +1 draw on blend batches |
| `$parallaxmap` | not planned | no cheap fixed-function equivalent; visually minor at 480p | — |
| HDR bloom / tonemap | **done** | `r_gx_bloom.c`: EFB → half-res I8 through the copy box filter, 8-tap TEV blur − threshold, × warm tint, additive | 1 EFB copy + 1 fullscreen quad (10 stages), 75 KB BSS |
| Water (`Water` shader: reflect/refract/fresnel) | done (engine) | constant body colour + sky add; see `bake_lostcoast_water.py` | — |

## Budget rules

* 16 TEV stages, 8 texgens, 8 texmaps, 4 indirect stages per draw.
* No heap after map load: every table/texture here is BSS.
* Shared textures only (4 detail textures for 27 materials); never one
  auxiliary texture per material.
