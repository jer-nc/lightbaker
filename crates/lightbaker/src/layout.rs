//! Where every lit surface sits in the lightmap.
//!
//! The receiving triangles are grown into flat charts (neighbours facing within the chart angle of
//! the first, all in one tile of the ground plan), each projected on its own plane with square
//! texels. A tile packs its own charts (skyline, turned where it fits better) into a rectangle of its
//! own, so its texels depend on nothing outside it; the tiles' rectangles are then shelved into the
//! atlas. A change to the scene moves tiles in the atlas (a copy) but bakes again only the tiles it
//! reaches. Everything is in a fixed order: the same scene always gives the same layout.

use std::collections::{BTreeMap, HashMap};

use anyhow::{Result, bail};
use glam::{Vec2, Vec3};

use crate::scene::Scene;

/// Tiles and atlas rectangles are aligned to this (block-compressed formats: 4 x 4 blocks).
pub const ALIGN: u32 = 4;

#[derive(Clone, Debug)]
pub struct Chart {
    pub mesh: u32,
    /// the mesh's triangles in it, in mesh order
    pub tris: Vec<u32>,
    pub normal: Vec3,
    /// the plane's axes (texel x and y before any turn)
    pub u: Vec3,
    pub v: Vec3,
    pub min: Vec2,
    pub max: Vec2,
    pub tile: TileKey,
    /// turned a quarter in its tile: texel x follows `v`, y follows `u`
    pub turned: bool,
    /// size in texels, padding included
    pub w: u32,
    pub h: u32,
    /// corner in its tile
    pub x: u32,
    pub y: u32,
}

/// A tile of the ground plan: (floor(x / size), floor(z / size)).
pub type TileKey = (i32, i32);

#[derive(Clone, Debug)]
pub struct Tile {
    pub key: TileKey,
    pub charts: Vec<u32>,
    pub w: u32,
    pub h: u32,
    /// corner in the atlas
    pub x: u32,
    pub y: u32,
}

pub struct Layout {
    pub charts: Vec<Chart>,
    /// in key order
    pub tiles: Vec<Tile>,
    pub width: u32,
    pub height: u32,
    pub density: f32,
    pub padding: u32,
    /// each mesh's triangles' chart (u32::MAX: not laid out)
    pub tri_chart: Vec<Vec<u32>>,
}

impl Chart {
    /// The texel coordinates of a point on the chart, in its tile (the texel's centre at whole numbers plus a half).
    pub fn local(&self, p: Vec3, density: f32, padding: u32) -> Vec2 {
        let a = (Vec2::new(p.dot(self.u), p.dot(self.v)) - self.min) * density;
        let a = if self.turned { Vec2::new(a.y, a.x) } else { a };
        Vec2::new(self.x as f32, self.y as f32) + Vec2::splat(padding as f32 + 0.5) + a
    }

    /// The chart's map from a world point to atlas texel coordinates, as two rows (coefficients, constant).
    pub fn affine(&self, tile: &Tile, density: f32, padding: u32) -> [[f32; 4]; 2] {
        let off = Vec2::new((tile.x + self.x) as f32, (tile.y + self.y) as f32) + Vec2::splat(padding as f32 + 0.5);
        let (ax, ay, mx, my) = if self.turned { (self.v, self.u, self.min.y, self.min.x) } else { (self.u, self.v, self.min.x, self.min.y) };
        let row = |axis: Vec3, min: f32, o: f32| {
            let c = axis * density;
            [c.x, c.y, c.z, o - min * density]
        };
        [row(ax, mx, off.x), row(ay, my, off.y)]
    }
}

impl Layout {
    pub fn build(scene: &Scene) -> Result<Layout> {
        let s = &scene.header.lightmap;
        let (density, padding) = (s.texels_per_meter, s.padding);
        let cos = s.chart_angle_deg.to_radians().cos();
        let mut charts = Vec::new();
        let mut tri_chart: Vec<Vec<u32>> = scene.positions.iter().map(|p| vec![u32::MAX; p.len() / 3]).collect();
        for (mi, m) in scene.header.meshes.iter().enumerate() {
            if !scene.header.materials[m.material as usize].receives {
                continue;
            }
            for group in grow(&scene.positions[mi], cos, s.tile_size) {
                let c = chart(&scene.positions[mi], mi as u32, group, s.tile_size, density, padding);
                for &t in &c.tris {
                    tri_chart[mi][t as usize] = charts.len() as u32;
                }
                charts.push(c);
            }
        }
        // the tiles, in key order, each packing its own charts
        let mut by_tile: BTreeMap<TileKey, Vec<u32>> = BTreeMap::new();
        for (i, c) in charts.iter().enumerate() {
            by_tile.entry(c.tile).or_default().push(i as u32);
        }
        let mut tiles: Vec<Tile> = by_tile.into_iter().map(|(key, ids)| pack_tile(key, ids, &mut charts)).collect();
        let height = shelve(&mut tiles, s.atlas_width)?;
        Ok(Layout { charts, tiles, width: s.atlas_width, height, density, padding, tri_chart })
    }

    /// Each chart's tile (by index into `tiles`).
    pub fn chart_tile(&self) -> Vec<u32> {
        let mut out = vec![0; self.charts.len()];
        for (ti, t) in self.tiles.iter().enumerate() {
            for &c in &t.charts {
                out[c as usize] = ti as u32;
            }
        }
        out
    }
}

/// Charts by region growing: from the first triangle not yet taken, across shared vertices, every
/// neighbour facing within `cos` of the seed whose box centre lies in the seed's tile. In triangle order.
fn grow(pos: &[Vec3], cos: f32, tile: f32) -> Vec<Vec<u32>> {
    let n = pos.len() / 3;
    let normals: Vec<Vec3> = (0..n).map(|t| face_normal(pos, t)).collect();
    // by the centre of a triangle's box: the two halves of a quad share it, so they never land in two tiles
    let keys: Vec<TileKey> = (0..n).map(|t| tile_of(box_centre(pos, t), tile)).collect();
    let mut at: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
    let key = |v: Vec3| [(v.x * 1e4).round() as i64, (v.y * 1e4).round() as i64, (v.z * 1e4).round() as i64];
    for t in 0..n {
        for k in 0..3 {
            at.entry(key(pos[t * 3 + k])).or_default().push(t as u32);
        }
    }
    let mut taken = vec![false; n];
    let mut out = Vec::new();
    for seed in 0..n {
        if taken[seed] {
            continue;
        }
        taken[seed] = true;
        let mut group = vec![seed as u32];
        let (sn, sk) = (normals[seed], keys[seed]);
        let mut i = 0;
        while i < group.len() {
            let t = group[i] as usize;
            for k in 0..3 {
                for &o in &at[&key(pos[t * 3 + k])] {
                    let o = o as usize;
                    if taken[o] || keys[o] != sk || normals[o].dot(sn) < cos {
                        continue;
                    }
                    taken[o] = true;
                    group.push(o as u32);
                }
            }
            i += 1;
        }
        group.sort_unstable();
        out.push(group);
    }
    out
}

fn chart(pos: &[Vec3], mesh: u32, tris: Vec<u32>, tile: f32, density: f32, padding: u32) -> Chart {
    // the plane of the chart's area-weighted normal; u horizontal where the plane allows (walls and floors keep square texels on the grid)
    let mut sum = Vec3::ZERO;
    for &t in &tris {
        let t = t as usize;
        sum += (pos[t * 3 + 2] - pos[t * 3 + 1]).cross(pos[t * 3] - pos[t * 3 + 1]);
    }
    let n = sum.try_normalize().unwrap_or(face_normal(pos, tris[0] as usize));
    let u = if n.y.abs() > 0.9 { Vec3::X } else { Vec3::Y.cross(n) };
    let u = (u - n * u.dot(n)).normalize();
    let v = n.cross(u);
    let (mut min, mut max) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
    for &t in &tris {
        for k in 0..3 {
            let p = pos[t as usize * 3 + k];
            let q = Vec2::new(p.dot(u), p.dot(v));
            min = min.min(q);
            max = max.max(q);
        }
    }
    let size = |e: f32| (e * density).ceil() as u32 + 1 + 2 * padding;
    let c = box_centre(pos, tris[0] as usize);
    Chart { mesh, normal: n, u, v, min, max, tile: tile_of(c, tile), turned: false, w: size(max.x - min.x), h: size(max.y - min.y), x: 0, y: 0, tris }
}

/// Packs a tile's charts with a skyline (bottom-left), the tallest first, each turned when that lies
/// lower. The tile's width is what a square of its charts' area needs (or its widest chart).
fn pack_tile(key: TileKey, ids: Vec<u32>, charts: &mut [Chart]) -> Tile {
    // lying down: width at least height
    for &i in &ids {
        let c = &mut charts[i as usize];
        if c.h > c.w {
            c.turned = true;
            std::mem::swap(&mut c.w, &mut c.h);
        }
    }
    let mut order = ids.clone();
    order.sort_by(|&a, &b| {
        let (ca, cb) = (&charts[a as usize], &charts[b as usize]);
        cb.h.cmp(&ca.h).then(cb.w.cmp(&ca.w)).then(a.cmp(&b))
    });
    let area: u64 = ids.iter().map(|&i| charts[i as usize].w as u64 * charts[i as usize].h as u64).sum();
    let widest = ids.iter().map(|&i| charts[i as usize].w).max().unwrap_or(1);
    let width = round_up(widest.max(((area as f64 * 1.1).sqrt()).ceil() as u32), ALIGN);
    let mut sky = Skyline::new(width);
    for &i in &order {
        let c = &mut charts[i as usize];
        let flat = sky.find(c.w, c.h);
        let up = if c.h <= width && c.w != c.h { sky.find(c.h, c.w) } else { None };
        let (x, y, turn) = match (flat, up) {
            (Some(a), Some(b)) if b.1 + c.w < a.1 + c.h => (b.0, b.1, true),
            (Some(a), _) => (a.0, a.1, false),
            (None, Some(b)) => (b.0, b.1, true),
            (None, None) => unreachable!("a chart no wider than its tile always fits"),
        };
        if turn {
            c.turned = !c.turned;
            std::mem::swap(&mut c.w, &mut c.h);
        }
        c.x = x;
        c.y = y;
        sky.place(x, y, c.w, c.h);
    }
    let height = round_up(sky.top().max(1), ALIGN);
    Tile { key, charts: ids, w: width, h: height, x: 0, y: 0 }
}

/// The tiles shelved into the atlas, the tallest first; the atlas' height (a multiple of ALIGN).
fn shelve(tiles: &mut [Tile], width: u32) -> Result<u32> {
    let mut order: Vec<usize> = (0..tiles.len()).collect();
    order.sort_by(|&a, &b| tiles[b].h.cmp(&tiles[a].h).then(tiles[a].key.cmp(&tiles[b].key)));
    let (mut x, mut y, mut row) = (0, 0, 0);
    for i in order {
        let t = &mut tiles[i];
        if t.w > width {
            bail!("tile {:?} is {} texels wide, the atlas {}: lower texelsPerMeter or tileSize, or widen the atlas", t.key, t.w, width);
        }
        if x + t.w > width {
            x = 0;
            y += row;
            row = 0;
        }
        t.x = x;
        t.y = y;
        x += t.w;
        row = row.max(t.h);
    }
    Ok(round_up((y + row).max(ALIGN), ALIGN))
}

/// The heights reached along a rectangle's width, as segments.
struct Skyline {
    /// (x, width, top)
    segs: Vec<(u32, u32, u32)>,
    width: u32,
}

impl Skyline {
    fn new(width: u32) -> Skyline {
        Skyline { segs: vec![(0, width, 0)], width }
    }

    /// The lowest place (then the leftmost) for a w x h rectangle.
    fn find(&self, w: u32, _h: u32) -> Option<(u32, u32)> {
        if w > self.width {
            return None;
        }
        let mut best: Option<(u32, u32)> = None;
        for (i, &(x, _, _)) in self.segs.iter().enumerate() {
            if x + w > self.width {
                break;
            }
            let (mut top, mut reach) = (0, 0);
            for &(_, sw, st) in &self.segs[i..] {
                top = top.max(st);
                reach += sw;
                if reach >= w {
                    break;
                }
            }
            if best.is_none_or(|b| top < b.1) {
                best = Some((x, top));
            }
        }
        best
    }

    fn place(&mut self, x: u32, y: u32, w: u32, h: u32) {
        let mut next = Vec::with_capacity(self.segs.len() + 2);
        for &(sx, sw, st) in &self.segs {
            let end = sx + sw;
            if end <= x || sx >= x + w {
                next.push((sx, sw, st));
                continue;
            }
            if sx < x {
                next.push((sx, x - sx, st));
            }
            if end > x + w {
                next.push((x + w, end - x - w, st));
            }
        }
        next.push((x, w, y + h));
        next.sort_by_key(|s| s.0);
        // neighbours at one height become one segment
        let mut merged: Vec<(u32, u32, u32)> = Vec::with_capacity(next.len());
        for s in next {
            match merged.last_mut() {
                Some(m) if m.2 == s.2 && m.0 + m.1 == s.0 => m.1 += s.1,
                _ => merged.push(s),
            }
        }
        self.segs = merged;
    }

    fn top(&self) -> u32 {
        self.segs.iter().map(|s| s.2).max().unwrap_or(0)
    }
}

pub fn face_normal(pos: &[Vec3], t: usize) -> Vec3 {
    (pos[t * 3 + 2] - pos[t * 3 + 1]).cross(pos[t * 3] - pos[t * 3 + 1]).try_normalize().unwrap_or(Vec3::Y)
}

pub fn centre(pos: &[Vec3], t: usize) -> Vec3 {
    (pos[t * 3] + pos[t * 3 + 1] + pos[t * 3 + 2]) / 3.0
}

/// The centre of a triangle's bounding box.
pub fn box_centre(pos: &[Vec3], t: usize) -> Vec3 {
    let (a, b, c) = (pos[t * 3], pos[t * 3 + 1], pos[t * 3 + 2]);
    (a.min(b).min(c) + a.max(b).max(c)) * 0.5
}

pub fn tile_of(p: Vec3, size: f32) -> TileKey {
    ((p.x / size).floor() as i32, (p.z / size).floor() as i32)
}

fn round_up(v: u32, a: u32) -> u32 {
    v.div_ceil(a) * a
}
