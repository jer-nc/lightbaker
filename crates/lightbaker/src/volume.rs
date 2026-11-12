//! The irradiance volume: probes on a regular grid over the play area, each the light arriving from
//! every direction (L1 spherical harmonics, linear RGB) and how much of the sun's disc it sees. Baked
//! in chunks of the ground plan's tiles, reused like the lightmap's tiles. Probes inside solids are
//! filled from their neighbours.

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::Result;
use glam::Vec3;

use crate::cache::Cache;
use crate::deps::{Aabb, Boxes, hash_reach};
use crate::gpu::{Gpu, Params, Pass, SceneBuffers};
use crate::layout::{TileKey, tile_of};
use crate::raster::Texel;
use crate::scene::{Scene, VolumeSettings};

/// A probe inside a solid sees mostly back faces.
const INSIDE: f32 = 0.5;

pub struct Volume {
    pub origin: [f32; 3],
    pub step: f32,
    /// probes along x, y and z
    pub size: [u32; 3],
    /// per probe (x fastest, then z, then y): four values, L1 coefficients rgb (L0, L1 x, y, z), the first's w the sun
    pub probes: Vec<[[f32; 4]; 4]>,
    pub chunks: usize,
    pub chunks_baked: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn bake_volume(scene: &Scene, gpu: &Gpu, buffers: &SceneBuffers, mut params: Params, light: &[[f32; 4]], v: &VolumeSettings, boxes: &Boxes, global: &blake3::Hash, cache: &Cache, log: &(dyn Fn(&str) + Sync)) -> Result<Volume> {
    let started = Instant::now();
    let n = |a: usize| ((v.max[a] - v.min[a]) / v.step).round() as u32 + 1;
    let size = [n(0).max(2), n(1).max(2), n(2).max(2)];
    let [nx, ny, nz] = size.map(|s| s as usize);
    let at = |k: usize| Vec3::new(v.min[0] + (k % nx) as f32 * v.step, v.min[1] + (k / (nx * nz)) as f32 * v.step, v.min[2] + ((k / nx) % nz) as f32 * v.step);
    let mut chunks: BTreeMap<TileKey, Vec<u32>> = BTreeMap::new();
    for k in 0..nx * ny * nz {
        chunks.entry(tile_of(at(k), scene.header.lightmap.tile_size)).or_default().push(k as u32);
    }
    let mut probes = vec![[[0f32; 4]; 4]; nx * ny * nz];
    let mut known = vec![false; nx * ny * nz];
    let (mut todo, mut keys) = (Vec::new(), Vec::new());
    let total = chunks.len();
    for (key, ids) in &chunks {
        let mut b = Aabb::empty();
        ids.iter().for_each(|&k| b.add(at(k as usize)));
        let mut seed = Vec::new();
        for x in [key.0, key.1] {
            seed.extend_from_slice(&x.to_le_bytes());
        }
        seed.extend_from_slice(&serde_json::to_vec(v)?);
        let hash = hash_reach(scene, boxes, global, b, scene.header.bake.gi_reach, &seed);
        match cache.get("probes", &hash).map(|b| bytemuck::pod_collect_to_vec::<u8, [[f32; 4]; 4]>(&b)).filter(|p| p.len() == ids.len()) {
            Some(p) => {
                for (&k, p) in ids.iter().zip(p) {
                    probes[k as usize] = p;
                }
            }
            None => {
                todo.extend(ids.iter().copied());
                keys.push((hash, ids.clone()));
            }
        }
    }
    let items: Vec<Texel> = todo
        .iter()
        .map(|&k| {
            let p = at(k as usize);
            Texel { pos: [p.x, p.y, p.z, 1.0], normal: [0.0, 1.0, 0.0, f32::from_bits(k)], ..Default::default() }
        })
        .collect();
    params.counts[2] = v.rays;
    let res = gpu.run(Pass::Volume, buffers, params, &items, light)?;
    for (i, &k) in todo.iter().enumerate() {
        probes[k as usize] = [res[i * 4], res[i * 4 + 1], res[i * 4 + 2], res[i * 4 + 3]];
    }
    for (hash, ids) in &keys {
        let p: Vec<[[f32; 4]; 4]> = ids.iter().map(|&k| probes[k as usize]).collect();
        cache.put("probes", hash, bytemuck::cast_slice(&p))?;
    }
    // inside solids: from the neighbours, a ring at a time
    for (k, p) in probes.iter().enumerate() {
        known[k] = p[1][3] <= INSIDE;
    }
    for _ in 0..8 {
        let (was, src) = (known.clone(), probes.clone());
        for k in 0..nx * ny * nz {
            if was[k] {
                continue;
            }
            let (x, z, y) = (k % nx, (k / nx) % nz, k / (nx * nz));
            let (mut sum, mut c) = ([[0f32; 4]; 4], 0);
            for (dx, dy, dz) in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] {
                let (xx, yy, zz) = (x as i64 + dx, y as i64 + dy, z as i64 + dz);
                if xx < 0 || yy < 0 || zz < 0 || xx >= nx as i64 || yy >= ny as i64 || zz >= nz as i64 {
                    continue;
                }
                let j = (yy as usize * nz + zz as usize) * nx + xx as usize;
                if was[j] {
                    for (a, b) in sum.iter_mut().zip(&src[j]) {
                        for q in 0..4 {
                            a[q] += b[q];
                        }
                    }
                    c += 1;
                }
            }
            if c > 0 {
                probes[k] = sum.map(|a| a.map(|q| q / c as f32));
                known[k] = true;
            }
        }
    }
    log(&format!("volume {nx}x{ny}x{nz}: {} probes baked, {:.1} s", todo.len(), started.elapsed().as_secs_f64()));
    Ok(Volume { origin: v.min, step: v.step, size, probes, chunks: total, chunks_baked: keys.len() })
}

impl Volume {
    /// The irradiance a face at `p` facing `n` receives (the eight probes round it blended, their L1
    /// harmonics convolved with the cosine lobe), and the share of the sun's disc seen there.
    pub fn irradiance(&self, p: Vec3, n: Vec3) -> ([f32; 3], f32) {
        let [nx, ny, nz] = self.size.map(|s| s as usize);
        let f = (p - Vec3::from_array(self.origin)) / self.step;
        let cl = |v: f32, n: usize| v.clamp(0.0, (n - 1) as f32);
        let (fx, fy, fz) = (cl(f.x, nx), cl(f.y, ny), cl(f.z, nz));
        let (x0, y0, z0) = ((fx as usize).min(nx - 2), (fy as usize).min(ny - 2), (fz as usize).min(nz - 2));
        let (tx, ty, tz) = (fx - x0 as f32, fy - y0 as f32, fz - z0 as f32);
        let mut p4 = [[0f32; 4]; 4];
        for k in 0..8 {
            let (dx, dy, dz) = (k & 1, (k >> 1) & 1, (k >> 2) & 1);
            let w = (if dx == 1 { tx } else { 1.0 - tx }) * (if dy == 1 { ty } else { 1.0 - ty }) * (if dz == 1 { tz } else { 1.0 - tz });
            let q = &self.probes[((y0 + dy) * nz + (z0 + dz)) * nx + (x0 + dx)];
            for a in 0..4 {
                for c in 0..4 {
                    p4[a][c] += q[a][c] * w;
                }
            }
        }
        let (a0, a1) = (std::f32::consts::PI * 0.282095, 2.0 * std::f32::consts::PI / 3.0 * 0.488603);
        // half the light from all round, half by the face's facing: L1 alone can cancel to nothing on a face turned from the light
        let e = std::array::from_fn(|c| 0.5 * a0 * p4[0][c] + 0.5 * (a0 * p4[0][c] + a1 * (p4[1][c] * n.x + p4[2][c] * n.y + p4[3][c] * n.z)).max(0.0));
        (e, p4[0][3])
    }
}
