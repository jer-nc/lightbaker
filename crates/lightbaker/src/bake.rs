//! A bake from start to end: the layout, the tiles' hashes, the tiles the cache does not have traced
//! on the GPU (the sun and occlusion, then the bounces, each pass reading the light of every tile from
//! the last), the CPU's texel passes, and the atlas put together from every tile.

use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use glam::Vec3;
use rayon::prelude::*;

use crate::bvh::Bvh;
use crate::cache::{Cache, Pack, Unpack};
use crate::deps::{Aabb, Boxes, global_hash, hash_reach};
use crate::filter::{Surface, fill, smooth, sun_field};
use crate::gpu::{Gpu, Params, Pass};
use crate::layout::{Layout, TileKey};
use crate::raster::{Texel, rasterize};
use crate::scene::Scene;
use crate::output::{VALIDITY_CENTRE, VALIDITY_FILLED, VALIDITY_NONE, VALIDITY_OFFSET, VALIDITY_PUSHED, VALIDITY_VOLUME};
use crate::volume::{Volume, bake_volume};

/// off the surface before a ray leaves it (m)
const LIFT: f32 = 0.01;
/** how far back faces count towards a point being inside a solid (m) */
const INSIDE_REACH: f32 = 0.3;
/** how far a texel inside thin geometry is pushed out to be sampled (m; trace.wgsl PUSH) */
const PUSH: f32 = 0.03;

pub struct Options<'a> {
    /// where baked tiles are kept between bakes (none: everything is baked)
    pub cache: Option<&'a Path>,
    pub log: &'a (dyn Fn(&str) + Sync),
}

pub struct Bake {
    pub layout: Layout,
    /// per atlas texel: the sun's distance field and ambient occlusion (0 … 1)
    pub direct: Vec<[f32; 2]>,
    /// per texel of the smaller indirect atlas: the bounced light and the sky's (irradiance, linear RGB)
    pub indirect: Vec<[f32; 3]>,
    pub indirect_size: (u32, u32),
    /// per atlas texel: how it got its light (output::VALIDITY_*)
    pub validity: Vec<u8>,
    pub volume: Option<Volume>,
    pub stats: Stats,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Stats {
    pub tiles: usize,
    pub tiles_baked: usize,
    pub texels_baked: usize,
    pub chunks: usize,
    pub chunks_baked: usize,
    pub seconds: f64,
}

/// A tile's light, as kept: every array `w * h`, texels with no light of their own filled from their neighbours.
struct TileLight {
    known: Vec<u8>,
    /// how each texel got its light (output::VALIDITY_*)
    state: Vec<u8>,
    field: Vec<f32>,
    ao: Vec<f32>,
    direct: Vec<[f32; 3]>,
    indirect: Vec<[f32; 3]>,
}

impl TileLight {
    fn pack(&self) -> Vec<u8> {
        Pack::default().put(&self.known).put(&self.state).put(&self.field).put(&self.ao).put(&self.direct).put(&self.indirect).0
    }
    fn unpack(b: &[u8], size: usize) -> Option<TileLight> {
        let mut u = Unpack::new(b);
        let t = TileLight { known: u.take()?, state: u.take()?, field: u.take()?, ao: u.take()?, direct: u.take()?, indirect: u.take()? };
        (t.known.len() == size && t.state.len() == size && t.field.len() == size && t.ao.len() == size && t.direct.len() == size && t.indirect.len() == size).then_some(t)
    }
}

/// A tile being baked: its texels, and the per-texel state the passes build up.
struct Work {
    tile: usize,
    texels: Vec<Texel>,
    chart: Vec<u32>,
    normal: Vec<[f32; 3]>,
    /// covered and not inside a solid
    valid: Vec<bool>,
    /// where bounced light is gathered
    gathered: Vec<bool>,
}

pub fn bake(scene: &Scene, gpu: &Gpu, o: &Options) -> Result<Bake> {
    let started = Instant::now();
    let h = &scene.header;
    let (b, lm) = (&h.bake, &h.lightmap);
    let layout = Layout::build(scene)?;
    let (aw, ah) = (layout.width as usize, layout.height as usize);
    let chart_tile = layout.chart_tile();
    let affines: Vec<[[f32; 4]; 2]> = layout.charts.iter().enumerate().map(|(i, c)| c.affine(&layout.tiles[chart_tile[i] as usize], layout.density, layout.padding)).collect();
    let bvh = Bvh::build(scene, |m, t| layout.tri_chart[m][t]);
    let materials: Vec<[f32; 4]> = h.materials.iter().map(|m| [m.albedo[0], m.albedo[1], m.albedo[2], m.opacity]).collect();
    let buffers = gpu.scene(&bvh, &materials, &affines);
    let sun = Vec3::from_array(h.sun.direction).normalize();
    let sun_dir = sun;
    let g = (b.sun_samples as f32).sqrt() as u32;
    let sky_avg = (Vec3::from_array(h.sky.zenith) + Vec3::from_array(h.sky.horizon)) * 0.5 * std::f32::consts::PI;
    let params = Params {
        sun_dir: [sun.x, sun.y, sun.z, h.sun.radius_deg.to_radians().tan()],
        sun_irr: rgb4(h.sun.irradiance),
        zenith: rgb4(h.sky.zenith),
        horizon: rgb4(h.sky.horizon),
        // a point is inside a solid by the back faces close round it, whatever occlusion's reach
        ground: [h.sky.ground[0], h.sky.ground[1], h.sky.ground[2], INSIDE_REACH.min(b.ao_reach)],
        fallback: rgb4(sky_avg.to_array()),
        counts: [g, b.ao_rays, b.gi_rays, 0],
        reach: [b.ao_reach, b.gi_reach, LIFT, 0.0],
        atlas: [layout.width, layout.height, 0, b.inside_share.to_bits()],
    };
    (o.log)(&format!("{} triangles, {} charts in {} tiles, atlas {}x{}, BVH {} nodes ({:.1} s)", scene.triangle_count(), layout.charts.len(), layout.tiles.len(), aw, ah, bvh.nodes.len(), started.elapsed().as_secs_f64()));

    // which tiles the cache has
    let cache = Cache::new(o.cache)?;
    let boxes = Boxes::new(scene);
    let global = global_hash(scene);
    let reach = if b.gi_bounces > 0 { b.gi_reach.max(b.ao_reach) } else { b.ao_reach };
    let hashes: Vec<blake3::Hash> = layout.tiles.iter().map(|t| hash_reach(scene, &boxes, &global, tile_box(scene, &layout, &t.charts), reach, &tile_seed(scene, &layout, t.key, &t.charts))).collect();
    let mut lights: Vec<Option<TileLight>> = layout
        .tiles
        .par_iter()
        .zip(&hashes)
        .map(|(t, k)| cache.get("tile", k).and_then(|bytes| TileLight::unpack(&bytes, (t.w * t.h) as usize)))
        .collect();
    let dirty: Vec<usize> = (0..layout.tiles.len()).filter(|&i| lights[i].is_none()).collect();
    (o.log)(&format!("{} of {} tiles to bake", dirty.len(), layout.tiles.len()));

    // the dirty tiles' texels
    let mut work: Vec<Work> = dirty
        .par_iter()
        .map(|&ti| {
            let texels = rasterize(scene, &layout, &layout.tiles[ti]);
            let chart = texels.iter().map(|t| if t.covered() { t.chart() } else { u32::MAX }).collect();
            let normal = texels.iter().map(|t| t.n().to_array()).collect();
            let valid = texels.iter().map(Texel::covered).collect();
            Work { tile: ti, texels, chart, normal, valid, gathered: Vec::new() }
        })
        .collect();
    let covered: Vec<Texel> = work.iter().flat_map(|w| w.texels.iter().filter(|t| t.covered()).copied()).collect();
    let t0 = Instant::now();
    let direct = gpu.run(Pass::Direct, &buffers, params, &covered, &[])?;
    (o.log)(&format!("sun and occlusion: {} texels, {:.1} s", covered.len(), t0.elapsed().as_secs_f64()));

    // geometry worth a look: materials much of whose surface lies inside solids (faces hidden inside
    // other pieces, pieces sunk into each other, faces turned inside out) cost atlas space for nothing
    {
        let mut per = vec![(0usize, 0usize); h.materials.len()];
        let mut k = 0;
        for w in &work {
            for t in w.texels.iter().filter(|t| t.covered()) {
                let m = h.meshes[layout.charts[t.chart() as usize].mesh as usize].material as usize;
                per[m].0 += 1;
                if direct[k][3] > b.inside_share {
                    per[m].1 += 1;
                }
                k += 1;
            }
        }
        // charts the bake leaves without light of their own: no texel centre on them (smaller than a
        // texel), or every texel inside a solid
        let mut chart = vec![(0u32, 0u32); layout.charts.len()];
        let mut k2 = 0;
        for w in &work {
            for t in w.texels.iter().filter(|t| t.covered()) {
                let c = &mut chart[t.chart() as usize];
                c.0 += 1;
                if direct[k2][3] > b.inside_share {
                    c.1 += 1;
                }
                k2 += 1;
            }
        }
        let baked: std::collections::HashSet<u32> = work.iter().flat_map(|w| layout.tiles[w.tile].charts.iter().copied()).collect();
        let (empty, inside) = baked.iter().fold((0, 0), |(e, i), &c| {
            let (n, ins) = chart[c as usize];
            (e + (n == 0) as usize, i + (n > 0 && ins == n) as usize)
        });
        (o.log)(&format!("charts baked: {}, without a texel: {}, all inside solids: {}", baked.len(), empty, inside));
        for (m, (all, inside)) in per.iter().enumerate() {
            if *all > 2000 && *inside * 3 > *all {
                (o.log)(&format!("warning: {}: {}% of its {} texels lie inside solids (hidden or overlapping faces, or faces turned inside out)", h.materials[m].name, inside * 100 / all, all));
            }
        }
    }

    // the sun's field, occlusion and the direct light, per tile
    let starts = offsets(work.iter().map(|w| w.valid.iter().filter(|&&v| v).count()));
    let made: Vec<TileLight> = work
        .par_iter_mut()
        .zip(&starts)
        .map(|(w, &start)| {
            let t = &layout.tiles[w.tile];
            let size = (t.w * t.h) as usize;
            let mut state = vec![VALIDITY_NONE; size];
            let (mut soft, mut ao, mut hard) = (vec![[0f32; 1]; size], vec![[1f32; 1]; size], vec![0u32; size]);
            let mut k = start;
            for i in 0..size {
                if !w.valid[i] {
                    continue;
                }
                let r = direct[k];
                k += 1;
                if r[3] > b.inside_share {
                    w.valid[i] = false;
                    continue;
                }
                soft[i] = [r[0]];
                let bits = r[1].to_bits();
                hard[i] = bits & 0x00ff_ffff;
                // sampled off its centre (bits 24…): the bounces gather from there too
                let pick = bits >> 24;
                state[i] = match pick {
                    0 => VALIDITY_CENTRE,
                    9 => VALIDITY_PUSHED,
                    _ => VALIDITY_OFFSET,
                };
                if pick == 9 {
                    // pushed out off thin geometry (trace.wgsl)
                    let t = &mut w.texels[i];
                    for c in 0..3 {
                        t.pos[c] += t.normal[c] * PUSH;
                    }
                } else if pick > 0 {
                    let (cx, cy) = offset(pick);
                    let t = &mut w.texels[i];
                    for c in 0..3 {
                        t.pos[c] += t.dx[c] * cx + t.dy[c] * cy;
                    }
                }
                ao[i] = [(1.0 - b.ao_strength * r[2]).max(0.0).powf(b.ao_contrast).max(b.ao_floor)];
            }
            let s = Surface { w: t.w as usize, h: t.h as usize, chart: &w.chart, normal: &w.normal };
            let rings = layout.padding as usize + 4;
            let mut ao = smooth(&s, &ao, &w.valid, 1);
            let soft = smooth(&s, &soft, &w.valid, 1);
            let mut field: Vec<[f32; 1]> = sun_field(&s, &hard, &w.valid, g as usize, b.sdf_radius).into_iter().map(|v| [v]).collect();
            let mut direct: Vec<[f32; 3]> = (0..size)
                .map(|i| {
                    let n = Vec3::from_array(w.normal[i]);
                    (Vec3::from_array(h.sun.irradiance) * n.dot(sun).max(0.0) * soft[i][0]).to_array()
                })
                .collect();
            let mut known = w.valid.clone();
            fill(&s, &mut ao, &mut known.clone(), rings);
            fill(&s, &mut field, &mut known.clone(), rings);
            fill(&s, &mut direct, &mut known, rings);
            for i in 0..size {
                if w.texels[i].covered() && state[i] == VALIDITY_NONE && known[i] {
                    state[i] = VALIDITY_FILLED;
                }
            }
            TileLight {
                state,
                // 2: its own light; 1: filled from its neighbours; 0: none
                known: known.iter().zip(&w.valid).map(|(&k, &v)| if v { 2 } else { k as u8 }).collect(),
                field: field.into_iter().map(|v| v[0]).collect(),
                ao: ao.into_iter().map(|v| v[0]).collect(),
                direct,
                indirect: vec![[0.0; 3]; size],
            }
        })
        .collect();
    for (w, m) in work.iter().zip(made) {
        lights[w.tile] = Some(m);
    }

    // the bounces: each gathers what the last left in the atlas
    let mut light = vec![[0f32; 4]; aw * ah];
    for (ti, l) in lights.iter().enumerate() {
        write_light(&layout, ti, l.as_ref().unwrap(), &mut light);
    }
    // bounced light is gathered where the smaller indirect atlas has its texels (every `ds`-th texel each
    // way: tiles sit on a 4-texel grid, so these are the same in the tile and the atlas), and at least
    // once in every smaller texel each chart covers: a chart narrower than the grid (a sheet's fold, a
    // plank's edge) gets its own all along it, never light filled in from a metre away
    let ds = lm.indirect_downscale as usize;
    for w in &mut work {
        let tw = layout.tiles[w.tile].w as usize;
        let block = |i: usize| ((i / tw) / ds) * tw.div_ceil(ds) + (i % tw) / ds;
        w.gathered = w.valid.iter().enumerate().map(|(i, &v)| v && (i % tw) % ds == 0 && (i / tw) % ds == 0).collect();
        let mut seen: std::collections::HashSet<(usize, u32)> = (0..w.valid.len()).filter(|&i| w.gathered[i]).map(|i| (block(i), w.chart[i])).collect();
        for i in 0..w.valid.len() {
            if w.valid[i] && seen.insert((block(i), w.chart[i])) {
                w.gathered[i] = true;
            }
        }
    }
    let gather: Vec<Texel> = work.iter().flat_map(|w| w.texels.iter().zip(&w.gathered).filter(|(_, g)| **g).map(|(t, _)| *t)).collect();
    let starts = offsets(work.iter().map(|w| w.gathered.iter().filter(|&&g| g).count()));
    for bounce in 0..b.gi_bounces {
        let t0 = Instant::now();
        let res = gpu.run(Pass::Gather, &buffers, params, &gather, &light)?;
        let updated: Vec<Vec<[f32; 3]>> = work
            .par_iter()
            .zip(&starts)
            .map(|(w, &start)| {
                let t = &layout.tiles[w.tile];
                let size = (t.w * t.h) as usize;
                let mut e = vec![[0f32; 3]; size];
                let mut k = start;
                for i in 0..size {
                    if w.gathered[i] {
                        e[i] = [res[k][0], res[k][1], res[k][2]];
                        k += 1;
                    }
                }
                let s = Surface { w: t.w as usize, h: t.h as usize, chart: &w.chart, normal: &w.normal };
                let mut e = smooth(&s, &e, &w.gathered, b.gi_filter as usize * ds);
                // every texel of a chart from the nearest gathered ones (each smaller texel of it has one), and the padding round it
                fill(&s, &mut e, &mut w.gathered.clone(), layout.padding as usize + 2 * ds + 2);
                e
            })
            .collect();
        for (w, e) in work.iter().zip(updated) {
            let l = lights[w.tile].as_mut().unwrap();
            l.indirect = e;
            write_light(&layout, w.tile, l, &mut light);
        }
        (o.log)(&format!("bounce {}: {} texels x {} rays, {:.1} s", bounce + 1, gather.len(), b.gi_rays, t0.elapsed().as_secs_f64()));
    }
    let volume = match &h.volume {
        Some(v) => Some(bake_volume(scene, gpu, &buffers, params, &light, v, &boxes, &global, &cache, o.log)?),
        None => None,
    };
    // texels the bake left without light of their own (a face every texel of which lies inside a
    // solid: wedged between others) take the volume's light where they are, never black
    if let Some(v) = &volume {
        let mut fixed = 0;
        for w in &work {
            let l = lights[w.tile].as_mut().unwrap();
            for (i, t) in w.texels.iter().enumerate() {
                if !t.covered() || l.known[i] != 0 {
                    continue;
                }
                let n = t.n();
                let (e, sun) = v.irradiance(Vec3::from_array([t.pos[0], t.pos[1], t.pos[2]]) + n * 0.15, n);
                l.indirect[i] = e;
                l.direct[i] = (Vec3::from_array(h.sun.irradiance) * n.dot(sun_dir).max(0.0) * sun).to_array();
                l.field[i] = if sun > 0.5 { 1.0 } else { 0.0 };
                l.ao[i] = 1.0;
                l.known[i] = 1;
                l.state[i] = VALIDITY_VOLUME;
                fixed += 1;
            }
        }
        if fixed > 0 {
            (o.log)(&format!("{fixed} texels without light of their own lit from the volume"));
        }
    }
    for w in &work {
        cache.put("tile", &hashes[w.tile], &lights[w.tile].as_ref().unwrap().pack())?;
    }

    // the atlas from every tile
    let (iw, ih) = (aw.div_ceil(ds), ah.div_ceil(ds));
    let mut direct_out = vec![[1f32, 1f32]; aw * ah];
    let mut validity = vec![VALIDITY_NONE; aw * ah];
    let mut sum = vec![[0f32; 4]; iw * ih];
    for (ti, t) in layout.tiles.iter().enumerate() {
        let l = lights[ti].as_ref().unwrap();
        for y in 0..t.h as usize {
            for x in 0..t.w as usize {
                let i = y * t.w as usize + x;
                if l.known[i] == 0 {
                    continue;
                }
                let (ax, ay) = (t.x as usize + x, t.y as usize + y);
                direct_out[ay * aw + ax] = [l.field[i], l.ao[i]];
                validity[ay * aw + ax] = l.state[i];
                // a smaller texel takes its own texels' light: a filled one (under a wall, in padding)
                // counts only where it has nothing else, so light from beyond a wall does not leak in
                let k = if l.known[i] == 2 { 1.0 } else { 1e-3 };
                let s = &mut sum[(ay / ds) * iw + ax / ds];
                for c in 0..3 {
                    s[c] += l.indirect[i][c] * k;
                }
                s[3] += k;
            }
        }
    }
    let mut indirect: Vec<[f32; 3]> = sum.iter().map(|s| if s[3] > 0.0 { [s[0] / s[3], s[1] / s[3], s[2] / s[3]] } else { [0.0; 3] }).collect();
    // the empty smaller texels round the charts take their neighbours' light: bilinear filtering at a
    // chart's edge never reads black
    let mut has: Vec<bool> = sum.iter().map(|s| s[3] > 0.0).collect();
    for _ in 0..2 {
        let (was, src) = (has.clone(), indirect.clone());
        for y in 0..ih {
            for x in 0..iw {
                let i = y * iw + x;
                if was[i] {
                    continue;
                }
                let (mut acc, mut n) = ([0f32; 3], 0);
                for yy in y.saturating_sub(1)..(y + 2).min(ih) {
                    for xx in x.saturating_sub(1)..(x + 2).min(iw) {
                        let j = yy * iw + xx;
                        if was[j] {
                            for c in 0..3 {
                                acc[c] += src[j][c];
                            }
                            n += 1;
                        }
                    }
                }
                if n > 0 {
                    indirect[i] = acc.map(|v| v / n as f32);
                    has[i] = true;
                }
            }
        }
    }

    let stats = Stats {
        tiles: layout.tiles.len(),
        tiles_baked: work.len(),
        texels_baked: covered.len(),
        chunks: volume.as_ref().map_or(0, |v| v.chunks),
        chunks_baked: volume.as_ref().map_or(0, |v| v.chunks_baked),
        seconds: started.elapsed().as_secs_f64(),
    };
    Ok(Bake { layout, direct: direct_out, validity, indirect, indirect_size: (iw as u32, ih as u32), volume, stats })
}

/// What a tile sends into the atlas the bounces read: the light reaching each known texel.
fn write_light(layout: &Layout, ti: usize, l: &TileLight, light: &mut [[f32; 4]]) {
    let t = &layout.tiles[ti];
    for y in 0..t.h as usize {
        for x in 0..t.w as usize {
            let i = y * t.w as usize + x;
            let a = (t.y as usize + y) * layout.width as usize + t.x as usize + x;
            light[a] = if l.known[i] >= 1 { [l.direct[i][0] + l.indirect[i][0], l.direct[i][1] + l.indirect[i][1], l.direct[i][2] + l.indirect[i][2], 1.0] } else { [0.0; 4] };
        }
    }
}

/// The box round a tile's surfaces.
fn tile_box(scene: &Scene, layout: &Layout, charts: &[u32]) -> Aabb {
    let mut b = Aabb::empty();
    for &c in charts {
        let c = &layout.charts[c as usize];
        for &t in &c.tris {
            for k in 0..3 {
                b.add(scene.positions[c.mesh as usize][t as usize * 3 + k]);
            }
        }
    }
    b
}

/// A tile's own content: its key, its surfaces (whether they cast or not) and their materials.
fn tile_seed(scene: &Scene, layout: &Layout, key: TileKey, charts: &[u32]) -> Vec<u8> {
    let mut k = blake3::Hasher::new();
    k.update(&key.0.to_le_bytes());
    k.update(&key.1.to_le_bytes());
    for &c in charts {
        let c = &layout.charts[c as usize];
        k.update(&c.mesh.to_le_bytes());
        k.update(&scene.header.meshes[c.mesh as usize].material.to_le_bytes());
        for &t in &c.tris {
            k.update(&t.to_le_bytes());
            for v in &scene.positions[c.mesh as usize][t as usize * 3..t as usize * 3 + 3] {
                k.update(bytemuck::bytes_of(&v.to_array()));
            }
        }
    }
    k.finalize().as_bytes().to_vec()
}

fn offsets(counts: impl Iterator<Item = usize>) -> Vec<usize> {
    let mut at = 0;
    counts
        .map(|n| {
            let s = at;
            at += n;
            s
        })
        .collect()
}

fn rgb4(c: [f32; 3]) -> [f32; 4] {
    [c[0], c[1], c[2], 0.0]
}

/** The sample point across a texel tried when its centre lies inside a solid (as trace.wgsl's `offset`). */
fn offset(q: u32) -> (f32, f32) {
    match q {
        1..=4 => (if q & 1 == 1 { 0.3 } else { -0.3 }, if q > 2 { 0.3 } else { -0.3 }),
        5 => (-0.45, 0.0),
        6 => (0.45, 0.0),
        7 => (0.0, -0.45),
        _ => (0.0, 0.45),
    }
}
