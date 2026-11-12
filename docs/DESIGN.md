# How it bakes

1. **Layout** (`layout.rs`): receiving triangles are grown into flat charts (neighbours within the
   chart angle, all in one tile of the ground plan, a quad's two halves always together), each
   projected on its own plane. Each tile packs its charts (skyline, quarter turns); tiles are shelved
   into the atlas on a 4-texel grid. A tile's texels depend on nothing outside it.
2. **Hashes** (`deps.rs`): a tile's hash covers its own surfaces, every casting triangle within the
   bounces' reach of it or between that and the sun, every setting and `BAKER_VERSION`. A tile found
   in the cache (`cache.rs`) is reused.
3. **Texels** (`raster.rs`): every texel a triangle covers, then, conservatively, every texel a
   triangle only touches, so faces thinner than a texel (a sheet's folds, a plank's edge) get light.
4. **Rays** (`gpu/trace.wgsl`, BVH from `bvh.rs`): per texel, a 4 x 4 grid of hard sun rays (the
   distance field) and soft ones to the sun's disc, and cosine-weighted occlusion rays. A texel whose
   rays meet back faces close by lies inside a solid: it is sampled again across the texel, then
   pushed off the surface (thin geometry), else filled from its chart.
5. **CPU passes** (`filter.rs`): edge-aware smoothing within a chart and facing, the sun's field,
   fills.
6. **Bounces**: gathered at least once in every texel of the smaller indirect atlas each chart covers,
   each pass reading the light every tile left in the atlas; rays that escape take the sky.
7. **Volume** (`volume.rs`): probes in chunks by tile, the same rays over the sphere projected on L1
   harmonics, plus the sun's share. A texel left with no light of its own anywhere takes the
   volume's, so nothing bakes black.
8. **Output** (`output.rs`, `encode.rs`): BC5 and RGB9E5 KTX2 (zstd), deflated arrays, and how each
   texel got its light (`validity`) for debugging tools.

References: Unreal's Lightmass and GPU Lightmass (stationary light shadowmask with distance field
shadows, ambient occlusion fractions, volumetric lightmap), Unity's Progressive Lightmapper
(distance shadowmask), Bakery (texel validity, conservative rasterisation, seams).
