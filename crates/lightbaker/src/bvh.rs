//! A bounding volume hierarchy over the scene's casting triangles, built on the CPU (binned surface
//! area heuristic) and flattened for the GPU's traversal (gpu/trace.wgsl).

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::scene::Scene;

const BINS: usize = 16;
const LEAF: usize = 4;

/// A node (32 bytes): its box; an inner node's first child (the second follows it), or a leaf's first triangle and count.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Node {
    pub min: [f32; 3],
    /// inner: the first child; leaf: the first triangle
    pub first: u32,
    pub max: [f32; 3],
    /// 0: inner node
    pub count: u32,
}

/// A triangle (48 bytes) as the rays meet it: a corner and the two edges from it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Tri {
    /// w: the material's index, as bits
    pub v0: [f32; 4],
    /// w: the lightmap chart's index (u32::MAX: none), as bits
    pub e1: [f32; 4],
    pub e2: [f32; 4],
}

pub struct Bvh {
    pub nodes: Vec<Node>,
    pub tris: Vec<Tri>,
}

#[derive(Clone, Copy)]
struct Prim {
    min: Vec3,
    max: Vec3,
    centre: Vec3,
    tri: Tri,
}

impl Bvh {
    /// Over every triangle whose material casts; `chart` gives a triangle's chart (mesh, triangle).
    pub fn build(scene: &Scene, chart: impl Fn(usize, usize) -> u32) -> Bvh {
        let mut prims = Vec::new();
        for (mi, m) in scene.header.meshes.iter().enumerate() {
            if !scene.header.materials[m.material as usize].casts {
                continue;
            }
            let p = &scene.positions[mi];
            for t in 0..p.len() / 3 {
                let (a, b, c) = (p[t * 3], p[t * 3 + 1], p[t * 3 + 2]);
                let (e1, e2) = (b - a, c - a);
                if e1.cross(e2).length_squared() < 1e-14 {
                    continue;
                }
                prims.push(Prim {
                    min: a.min(b).min(c),
                    max: a.max(b).max(c),
                    centre: (a + b + c) / 3.0,
                    tri: Tri {
                        v0: [a.x, a.y, a.z, f32::from_bits(m.material)],
                        e1: [e1.x, e1.y, e1.z, f32::from_bits(chart(mi, t))],
                        e2: [e2.x, e2.y, e2.z, 0.0],
                    },
                });
            }
        }
        let mut nodes = vec![Node::default()];
        if prims.is_empty() {
            // one empty leaf, a box nothing reaches
            nodes[0] = Node { min: [f32::MAX; 3], first: 0, max: [f32::MIN; 3], count: 0 };
            return Bvh { nodes, tris: vec![Tri::default()] };
        }
        // (node, start, end) left to split
        let mut stack = vec![(0usize, 0usize, prims.len())];
        while let Some((ni, start, end)) = stack.pop() {
            let slice = &mut prims[start..end];
            let (bmin, bmax) = bounds(slice.iter().map(|p| (p.min, p.max)));
            let split = if slice.len() > LEAF { best_split(slice) } else { None };
            match split {
                None => nodes[ni] = Node { min: bmin.to_array(), first: start as u32, max: bmax.to_array(), count: (end - start) as u32 },
                Some((axis, at)) => {
                    let mid = start + partition(slice, |p| p.centre[axis] < at);
                    let mid = if mid == start || mid == end { start + (end - start) / 2 } else { mid };
                    if mid - start == 0 || end - mid == 0 {
                        nodes[ni] = Node { min: bmin.to_array(), first: start as u32, max: bmax.to_array(), count: (end - start) as u32 };
                        continue;
                    }
                    let left = nodes.len();
                    nodes.push(Node::default());
                    nodes.push(Node::default());
                    nodes[ni] = Node { min: bmin.to_array(), first: left as u32, max: bmax.to_array(), count: 0 };
                    stack.push((left + 1, mid, end));
                    stack.push((left, start, mid));
                }
            }
        }
        Bvh { nodes, tris: prims.into_iter().map(|p| p.tri).collect() }
    }
}

fn bounds(it: impl Iterator<Item = (Vec3, Vec3)>) -> (Vec3, Vec3) {
    it.fold((Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)), |(a, b), (lo, hi)| (a.min(lo), b.max(hi)))
}

fn area(min: Vec3, max: Vec3) -> f32 {
    let d = (max - min).max(Vec3::ZERO);
    d.x * d.y + d.y * d.z + d.z * d.x
}

/// The axis and plane of the cheapest split of the centres into bins, if any beats a leaf.
fn best_split(prims: &[Prim]) -> Option<(usize, f32)> {
    let (cmin, cmax) = bounds(prims.iter().map(|p| (p.centre, p.centre)));
    let (pmin, pmax) = bounds(prims.iter().map(|p| (p.min, p.max)));
    let leaf_cost = prims.len() as f32 * area(pmin, pmax);
    let mut best: Option<(f32, usize, f32)> = None;
    for axis in 0..3 {
        let extent = cmax[axis] - cmin[axis];
        if extent < 1e-6 {
            continue;
        }
        let mut bins = [(Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY), 0u32); BINS];
        for p in prims {
            let b = (((p.centre[axis] - cmin[axis]) / extent * BINS as f32) as usize).min(BINS - 1);
            bins[b].0 = bins[b].0.min(p.min);
            bins[b].1 = bins[b].1.max(p.max);
            bins[b].2 += 1;
        }
        for s in 1..BINS {
            let (l, r) = (&bins[..s], &bins[s..]);
            let (ln, rn): (u32, u32) = (l.iter().map(|b| b.2).sum(), r.iter().map(|b| b.2).sum());
            if ln == 0 || rn == 0 {
                continue;
            }
            let (lmin, lmax) = bounds(l.iter().map(|b| (b.0, b.1)));
            let (rmin, rmax) = bounds(r.iter().map(|b| (b.0, b.1)));
            let cost = ln as f32 * area(lmin, lmax) + rn as f32 * area(rmin, rmax);
            if best.is_none_or(|b| cost < b.0) {
                best = Some((cost, axis, cmin[axis] + extent * s as f32 / BINS as f32));
            }
        }
    }
    best.filter(|b| b.0 < leaf_cost).map(|b| (b.1, b.2))
}

fn partition<T>(v: &mut [T], pred: impl Fn(&T) -> bool) -> usize {
    let mut i = 0;
    for j in 0..v.len() {
        if pred(&v[j]) {
            v.swap(i, j);
            i += 1;
        }
    }
    i
}
