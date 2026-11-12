//! The bake as it ships (`.lbake`): the magic `LBAK`, the format version (u32), the header's length
//! (u32), the header (JSON, [`Header`]), then the chunks it names, each on an 8-byte boundary.
//!
//! - `direct`: KTX2, BC5, the atlas: red the sun's distance field (0.5 on a shadow's edge, 0 and 1
//!   `sdfRadius` texels inside and out), green ambient occlusion
//! - `indirect`: KTX2, RGB9E5, the atlas `indirectDownscale` times smaller: the light arriving from the
//!   sky and from other surfaces (irradiance, linear RGB, in the scene's units)
//! - `charts`: raw-deflated f32, two rows (x y z constant) a chart: a world point to atlas texel
//!   coordinates (divide by the atlas' size for UVs)
//! - `validity`: raw-deflated u8 a texel of the atlas: how it got its light (VALIDITY_*), for tools
//! - `triangleCharts`: raw-deflated u32, every mesh's triangles in scene order: its chart (u32::MAX: none)
//! - `volume`: raw-deflated bytes, 16 a probe (x fastest, then z, then y): L0 rgb as `scale` times the
//!   square of byte / 255, the sun's visibility (byte / 255), then L1 x rgb, y rgb, z rgb each as a share
//!   of its channel's L0 times sqrt 3 (byte / 255 * 2 - 1), three bytes unused (radiance, L1 spherical
//!   harmonics, linear RGB)

use std::io::Write;

use anyhow::Result;
use flate2::{Compression, write::DeflateEncoder};
use serde::Serialize;

use crate::bake::{Bake, Stats};
use crate::encode::{Format, bc5, ktx2, rgb9e5};
use crate::scene::Scene;

pub const BAKE_MAGIC: &[u8; 4] = b"LBAK";

/// How a texel got its light (the `validity` chunk): no surface on it,
pub const VALIDITY_NONE: u8 = 0;
/// its own from its centre,
pub const VALIDITY_CENTRE: u8 = 1;
/// its own from a point across it (its centre inside a solid),
pub const VALIDITY_OFFSET: u8 = 2;
/// its own from a point pushed out off the surface (thin geometry),
pub const VALIDITY_PUSHED: u8 = 3;
/// filled from its chart's neighbours (inside a solid),
pub const VALIDITY_FILLED: u8 = 4;
/// the irradiance volume's (no light of its own anywhere on its chart).
pub const VALIDITY_VOLUME: u8 = 5;
pub const BAKE_VERSION: u32 = 1;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Header {
    pub baker: String,
    /// the scene file's blake3
    pub scene: String,
    pub user: serde_json::Value,
    pub atlas: Atlas,
    /// the irradiance the sky alone gives a surface facing up in the open (linear RGB): what an engine
    /// measures the baked light against (exposure, a floor for closed spaces)
    pub sky_irradiance: [f32; 3],
    pub meshes: Vec<MeshOut>,
    pub volume: Option<VolumeOut>,
    pub chunks: Vec<Chunk>,
    pub stats: Stats,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Atlas {
    pub width: u32,
    pub height: u32,
    pub texels_per_meter: f32,
    pub sdf_radius: f32,
    pub indirect_width: u32,
    pub indirect_height: u32,
    pub charts: u32,
}

#[derive(Serialize)]
pub struct MeshOut {
    pub name: String,
    pub triangles: u32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VolumeOut {
    pub origin: [f32; 3],
    pub step: f32,
    pub size: [u32; 3],
    /// the brightest L0 coefficient: what a byte of 255 stands for
    pub scale: f32,
}

#[derive(Serialize)]
pub struct Chunk {
    pub name: String,
    pub offset: u64,
    pub length: u64,
}

pub fn write(scene: &Scene, bake: &Bake) -> Result<Vec<u8>> {
    let l = &bake.layout;
    let (w, h) = (l.width as usize, l.height as usize);
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let rg: Vec<[u8; 2]> = bake.direct.iter().map(|t| [q(t[0]), q(t[1])]).collect();
    let (iw, ih) = bake.indirect_size;
    let packed: Vec<u32> = bake.indirect.iter().map(|&c| rgb9e5(c)).collect();
    let charts = l.charts.iter().zip(l.chart_tile()).flat_map(|(c, t)| c.affine(&l.tiles[t as usize], l.density, l.padding).into_iter().flatten()).collect::<Vec<f32>>();
    let tri: Vec<u32> = l.tri_chart.iter().flatten().copied().collect();
    let mut chunks: Vec<(&str, Vec<u8>)> = vec![
        ("direct", ktx2(Format::Bc5, w as u32, h as u32, &bc5(&rg, w, h))?),
        ("indirect", ktx2(Format::Rgb9e5, iw, ih, bytemuck::cast_slice(&packed))?),
        ("charts", deflate(bytemuck::cast_slice(&charts))?),
        ("triangleCharts", deflate(bytemuck::cast_slice(&tri))?),
        ("validity", deflate(&bake.validity)?),
    ];
    let scale = bake.volume.as_ref().map_or(1.0, |v| v.probes.iter().flat_map(|p| p[0][..3].iter().copied()).fold(1e-6f32, f32::max));
    if let Some(v) = &bake.volume {
        chunks.push(("volume", deflate(&volume_bytes(&v.probes, scale))?));
    }
    let mut header = Header {
        baker: format!("lightbaker {}", env!("CARGO_PKG_VERSION")),
        scene: scene.hash.to_hex().to_string(),
        user: scene.header.user.clone(),
        atlas: Atlas {
            width: l.width,
            height: l.height,
            texels_per_meter: l.density,
            sdf_radius: scene.header.bake.sdf_radius,
            indirect_width: iw,
            indirect_height: ih,
            charts: l.charts.len() as u32,
        },
        // a sky from the horizon (h) to the zenith (z), linear in height, over a face looking up: pi (h / 3 + 2 z / 3)
        sky_irradiance: std::array::from_fn(|c| std::f32::consts::PI * (scene.header.sky.horizon[c] / 3.0 + 2.0 * scene.header.sky.zenith[c] / 3.0)),
        meshes: scene.header.meshes.iter().map(|m| MeshOut { name: m.name.clone(), triangles: m.triangles }).collect(),
        volume: bake.volume.as_ref().map(|v| VolumeOut { origin: v.origin, step: v.step, size: v.size, scale }),
        chunks: Vec::new(),
        stats: bake.stats.clone(),
    };
    // offsets are from the start of the chunk area, which follows the header (8-aligned)
    let mut at = 0u64;
    for (name, b) in &chunks {
        header.chunks.push(Chunk { name: name.to_string(), offset: at, length: b.len() as u64 });
        at = (at + b.len() as u64).div_ceil(8) * 8;
    }
    let head = serde_json::to_vec(&header)?;
    let mut out = Vec::new();
    out.extend_from_slice(BAKE_MAGIC);
    out.extend_from_slice(&BAKE_VERSION.to_le_bytes());
    out.extend_from_slice(&(head.len() as u32).to_le_bytes());
    out.extend_from_slice(&head);
    out.resize(out.len().div_ceil(8) * 8, 0);
    let base = out.len();
    for ((_, b), c) in chunks.iter_mut().zip(&header.chunks) {
        out.resize(base + c.offset as usize, 0);
        out.extend_from_slice(b);
    }
    Ok(out)
}

fn deflate(b: &[u8]) -> Result<Vec<u8>> {
    let mut e = DeflateEncoder::new(Vec::new(), Compression::best());
    e.write_all(b)?;
    Ok(e.finish()?)
}

/// The probes in 16 bytes each (see the module's doc).
fn volume_bytes(probes: &[[[f32; 4]; 4]], scale: f32) -> Vec<u8> {
    const SQRT3: f32 = 1.732_050_8;
    let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let mut out = Vec::with_capacity(probes.len() * 16);
    for p in probes {
        for c in 0..3 {
            out.push(byte((p[0][c].max(0.0) / scale).sqrt()));
        }
        out.push(byte(p[0][3]));
        for axis in 1..4 {
            for c in 0..3 {
                let r = if p[0][c] > 1e-6 { p[axis][c] / (p[0][c] * SQRT3) } else { 0.0 };
                out.push(byte(r * 0.5 + 0.5));
            }
        }
        out.extend_from_slice(&[0; 3]);
    }
    out
}
