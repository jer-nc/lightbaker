# Files

## Scene (`.lbscene`, version 1)

`LBSC`, version (u32 LE), header length (u32), header (JSON), then each mesh's positions: f32 x y z,
three vertices a triangle (not indexed), each array on a 4-byte boundary, in header order. Y up, metres.

Header (`crates/lightbaker/src/scene.rs`, `js/src/scene.ts`):

| field | |
|---|---|
| `sun` | `direction` (towards the sun), `radiusDeg` (disc), `irradiance` (linear RGB) |
| `sky` | radiance: `zenith`, `horizon` (linear in height between them), `ground` (below the horizon) |
| `materials[]` | `name`, `albedo`, `receives` (laid out and baked), `casts`, `opacity` (share of light stopped) |
| `meshes[]` | `name`, `material`, `triangles` |
| `lightmap` | `texelsPerMeter`, `tileSize`, `chartAngleDeg`, `padding`, `atlasWidth`, `indirectDownscale` (1, 2, 4) |
| `bake` | `sunSamples` (square), `aoRays`, `aoReach`, `aoStrength`, `aoFloor`, `aoContrast`, `sdfRadius`, `giRays`, `giBounces`, `giReach`, `insideShare`, `giFilter` |
| `volume` | `min`, `max`, `step`, `rays` (optional) |
| `user` | copied to the bake untouched |

## Bake (`.lbake`, version 1)

`LBAK`, version, header length, header (JSON), then chunks on 8-byte boundaries (offsets from the
first chunk, which starts 8-aligned after the header).

Header: `baker`, `scene` (blake3 of the scene file), `user`, `atlas` (`width`, `height`,
`texelsPerMeter`, `sdfRadius`, `indirectWidth`, `indirectHeight`, `charts`), `skyIrradiance` (the sky
on an open face looking up), `meshes`, `volume` (`origin`, `step`, `size`, `scale`), `chunks`, `stats`.

| chunk | |
|---|---|
| `direct` | KTX2, BC5, zstd: red the sun's signed distance field (0.5 on a shadow's edge, 0 and 1 `sdfRadius` texels in and out), green ambient occlusion |
| `indirect` | KTX2, RGB9E5, zstd: irradiance from the sky and other surfaces |
| `charts` | raw deflate, f32: two rows (x y z constant) a chart, world point to atlas texel coordinates |
| `triangleCharts` | raw deflate, u32: each triangle's chart, every mesh in order (0xffffffff: none) |
| `validity` | raw deflate, u8 a texel: how it got its light (0 none, 1 its centre, 2 a point across it, 3 pushed off the surface, 4 filled from its chart, 5 the volume's), for tools |
| `volume` | raw deflate, 16 bytes a probe (x fastest, then z, then y): L0 rgb as `scale * (b/255)^2`, sun share, L1 x/y/z rgb as share of L0 times sqrt 3 (`b/255*2-1`), 3 unused |

UVs: `uv = (row0 · (p, 1) / width, row1 · (p, 1) / height)` for each vertex `p` of a triangle with its chart.
