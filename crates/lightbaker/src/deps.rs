//! What a tile's light depends on, as one hash: its own surfaces and every triangle that can shade
//! them (within the reach of occlusion and bounces round it, and anything between that and the sun),
//! with the settings, the sun, the sky and the materials. A tile whose hash is unchanged is reused.

use glam::Vec3;
use rayon::prelude::*;

use crate::scene::Scene;

/// Bump when the baker's results change for the same scene: every tile is baked again.
pub const BAKER_VERSION: u32 = 11;

/// A box, and how far it reaches.
#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn empty() -> Aabb {
        Aabb { min: Vec3::splat(f32::INFINITY), max: Vec3::splat(f32::NEG_INFINITY) }
    }
    pub fn add(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }
    pub fn grow(self, r: f32) -> Aabb {
        Aabb { min: self.min - r, max: self.max + r }
    }
}

/// The triangles of every mesh, as boxes (computed once).
pub struct Boxes {
    boxes: Vec<Vec<Aabb>>,
    top: f32,
}

impl Boxes {
    pub fn new(scene: &Scene) -> Boxes {
        let boxes: Vec<Vec<Aabb>> = scene
            .positions
            .iter()
            .map(|p| {
                p.chunks_exact(3)
                    .map(|t| {
                        let mut b = Aabb::empty();
                        t.iter().for_each(|&v| b.add(v));
                        b
                    })
                    .collect()
            })
            .collect();
        let top = boxes.iter().flatten().fold(f32::NEG_INFINITY, |t, b| t.max(b.max.y));
        Boxes { boxes, top }
    }
}

/// The settings every tile's light depends on (hashed once).
pub fn global_hash(scene: &Scene) -> blake3::Hash {
    let h = &scene.header;
    let mut k = blake3::Hasher::new();
    k.update(&BAKER_VERSION.to_le_bytes());
    for v in [serde_json::to_vec(&h.sun), serde_json::to_vec(&h.sky), serde_json::to_vec(&h.materials), serde_json::to_vec(&h.lightmap), serde_json::to_vec(&h.bake)] {
        k.update(&v.expect("settings serialise"));
    }
    k.finalize()
}

/// The hash of what reaches `own` (a tile's surfaces' box, or a chunk of probes'): every casting
/// triangle in it grown by `reach`, or between that and the sun; plus `seed` (the tile's own content).
pub fn hash_reach(scene: &Scene, boxes: &Boxes, global: &blake3::Hash, own: Aabb, reach: f32, seed: &[u8]) -> blake3::Hash {
    let zone = own.grow(reach);
    let sun = Vec3::from_array(scene.header.sun.direction).normalize();
    let mut k = blake3::Hasher::new();
    k.update(global.as_bytes());
    k.update(seed);
    for (mi, m) in scene.header.meshes.iter().enumerate() {
        if !scene.header.materials[m.material as usize].casts {
            continue;
        }
        let hits: Vec<u32> = boxes.boxes[mi].par_iter().enumerate().filter(|(_, b)| swept(zone, sun, boxes.top, b)).map(|(t, _)| t as u32).collect();
        if hits.is_empty() {
            continue;
        }
        k.update(&(mi as u32).to_le_bytes());
        k.update(&m.material.to_le_bytes());
        for t in hits {
            k.update(&t.to_le_bytes());
            for v in &scene.positions[mi][t as usize * 3..t as usize * 3 + 3] {
                k.update(bytemuck::bytes_of(&v.to_array()));
            }
        }
    }
    k.finalize()
}

/// Does `b` meet `zone` swept towards the sun up to `top`?
fn swept(zone: Aabb, sun: Vec3, top: f32, b: &Aabb) -> bool {
    // the times t >= 0 at which zone + t * sun overlaps b, axis by axis
    let tmax = if sun.y > 1e-4 { ((top - zone.min.y) / sun.y).max(0.0) } else { 0.0 };
    let (mut lo, mut hi) = (0f32, tmax);
    for a in 0..3 {
        let (s, zmin, zmax, bmin, bmax) = (sun[a], zone.min[a], zone.max[a], b.min[a], b.max[a]);
        if s.abs() < 1e-6 {
            if zmin > bmax || zmax < bmin {
                return false;
            }
            continue;
        }
        // zmin + t s <= bmax and zmax + t s >= bmin
        let (t1, t2) = ((bmax - zmin) / s, (bmin - zmax) / s);
        let (a0, a1) = if s > 0.0 { (t2, t1) } else { (t1, t2) };
        lo = lo.max(a0);
        hi = hi.min(a1);
        if lo > hi {
            return false;
        }
    }
    true
}
