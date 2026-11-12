//! A tile's texels: for each one a chart's triangle covers (its centre inside it), the point on the
//! surface, the face's normal and how the surface moves from one texel to the next (the rays sample
//! across the texel, not only at its centre).

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};

use crate::layout::{Layout, Tile, face_normal};
use crate::scene::Scene;

/// One texel as the GPU reads it (64 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Texel {
    /// the point (w: 1 where a surface covers the texel, 0 where it is empty)
    pub pos: [f32; 4],
    /// the face's normal (w: the chart's index, as bits)
    pub normal: [f32; 4],
    /// metres from one texel to the next along x and along y
    pub dx: [f32; 4],
    pub dy: [f32; 4],
}

impl Texel {
    pub fn covered(&self) -> bool {
        self.pos[3] > 0.5
    }

    pub fn chart(&self) -> u32 {
        self.normal[3].to_bits()
    }

    pub fn n(&self) -> Vec3 {
        Vec3::new(self.normal[0], self.normal[1], self.normal[2])
    }
}

/// The tile's texels, row by row in its own rectangle (`tile.w` x `tile.h`).
pub fn rasterize(scene: &Scene, layout: &Layout, tile: &Tile) -> Vec<Texel> {
    let (w, h) = (tile.w as i32, tile.h as i32);
    let mut out = vec![Texel::default(); (w * h) as usize];
    // first every texel whose centre a triangle covers; then, conservatively, every texel a triangle
    // only touches (faces thinner or smaller than a texel: a sheet's folds, a bag's facets, a plank's
    // edge), sampled at the triangle's point nearest its centre: no face is left without a texel
    for conservative in [false, true] {
    for &ci in &tile.charts {
        let c = &layout.charts[ci as usize];
        let pos = &scene.positions[c.mesh as usize];
        for &t in &c.tris {
            let t = t as usize;
            let (pa, pb, pc) = (pos[t * 3], pos[t * 3 + 1], pos[t * 3 + 2]);
            let (ua, ub, uc) = (c.local(pa, layout.density, layout.padding), c.local(pb, layout.density, layout.padding), c.local(pc, layout.density, layout.padding));
            let det = (ub.y - uc.y) * (ua.x - uc.x) + (uc.x - ub.x) * (ua.y - uc.y);
            if det.abs() < 1e-12 {
                continue;
            }
            let bary = |q: Vec2| {
                let l1 = ((ub.y - uc.y) * (q.x - uc.x) + (uc.x - ub.x) * (q.y - uc.y)) / det;
                let l2 = ((uc.y - ua.y) * (q.x - uc.x) + (ua.x - uc.x) * (q.y - uc.y)) / det;
                (l1, l2, 1.0 - l1 - l2)
            };
            let at = |q: Vec2| {
                let (l1, l2, l3) = bary(q);
                pa * l1 + pb * l2 + pc * l3
            };
            let n = face_normal(pos, t);
            let o = at(Vec2::ZERO);
            let (dx, dy) = (at(Vec2::X) - o, at(Vec2::Y) - o);
            let lo = ua.min(ub).min(uc).floor().max(Vec2::ZERO);
            let hi = ua.max(ub).max(uc).ceil().min(Vec2::new((w - 1) as f32, (h - 1) as f32));
            for y in lo.y as i32..=hi.y as i32 {
                for x in lo.x as i32..=hi.x as i32 {
                    let centre = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                    let i = (y * w + x) as usize;
                    if out[i].covered() {
                        continue;
                    }
                    let q = if conservative { nearest(centre, ua, ub, uc) } else { centre };
                    // the texel's square touches the triangle where the nearest point is within it
                    if conservative && ((q.x - centre.x).abs() > 0.5 || (q.y - centre.y).abs() > 0.5) {
                        continue;
                    }
                    let (l1, l2, l3) = bary(q);
                    if !conservative && (l1 < -1e-4 || l2 < -1e-4 || l3 < -1e-4) {
                        continue;
                    }
                    let p = at(q);
                    out[i] = Texel {
                        pos: [p.x, p.y, p.z, 1.0],
                        normal: [n.x, n.y, n.z, f32::from_bits(ci)],
                        dx: [dx.x, dx.y, dx.z, 0.0],
                        dy: [dy.x, dy.y, dy.z, 0.0],
                    };
                }
            }
        }
    }
    }
    out
}

/** The point of triangle (a, b, c) nearest `p` (2D). */
fn nearest(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> Vec2 {
    let side = |o: Vec2, e: Vec2, q: Vec2| (e - o).perp_dot(q - o);
    let s = side(a, b, c).signum();
    if side(a, b, p) * s >= 0.0 && side(b, c, p) * s >= 0.0 && side(c, a, p) * s >= 0.0 {
        return p;
    }
    let seg = |o: Vec2, e: Vec2| {
        let d = e - o;
        o + d * ((p - o).dot(d) / d.length_squared().max(1e-12)).clamp(0.0, 1.0)
    };
    [seg(a, b), seg(b, c), seg(c, a)].into_iter().min_by(|x, y| x.distance_squared(p).total_cmp(&y.distance_squared(p))).unwrap()
}
