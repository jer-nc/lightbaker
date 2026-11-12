//! The scene a bake reads (`.lbscene`): what an engine exports, and all the baker knows of it.
//!
//! A file is the magic `LBSC`, the format version (u32), the header's length (u32), the header (JSON,
//! [`SceneHeader`]), then the meshes' positions (f32, x y z per vertex, three vertices a triangle, not
//! indexed), each mesh's array starting on a 4-byte boundary in header order. Y is up, units are metres.

use anyhow::{Context, Result, bail, ensure};
use glam::Vec3;
use serde::{Deserialize, Serialize};

pub const SCENE_MAGIC: &[u8; 4] = b"LBSC";
pub const SCENE_VERSION: u32 = 1;

/// Linear RGB.
pub type Rgb = [f32; 3];

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SceneHeader {
    pub sun: Sun,
    pub sky: Sky,
    pub materials: Vec<Material>,
    pub meshes: Vec<MeshInfo>,
    pub lightmap: LightmapSettings,
    pub bake: BakeSettings,
    /// the irradiance volume over the play area; none: no volume baked
    #[serde(default)]
    pub volume: Option<VolumeSettings>,
    /// anything the engine wants back with the result, copied untouched (its own geometry hash, say)
    #[serde(default)]
    pub user: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Sun {
    /// towards the sun (normalised by the baker)
    pub direction: [f32; 3],
    /// the disc's angular radius (degrees): the penumbra's width
    pub radius_deg: f32,
    /// irradiance on a surface square to it (linear RGB, the engine's units)
    pub irradiance: Rgb,
}

/// The sky's radiance, a gradient from the horizon to the zenith, and below the horizon the ground's.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Sky {
    pub zenith: Rgb,
    pub horizon: Rgb,
    pub ground: Rgb,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Material {
    pub name: String,
    /// diffuse reflectance (linear RGB): the colour bounced light takes from it
    pub albedo: Rgb,
    /// laid out in the lightmap and baked
    pub receives: bool,
    /// blocks light (the sun's, the sky's, bounces)
    pub casts: bool,
    /// the share of light it stops where it casts (leaves, nets: below one)
    #[serde(default = "one")]
    pub opacity: f32,
}

fn one() -> f32 {
    1.0
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MeshInfo {
    pub name: String,
    pub material: u32,
    pub triangles: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LightmapSettings {
    pub texels_per_meter: f32,
    /// side of the square tiles the ground plan is cut into (m): what is baked again when something changes
    pub tile_size: f32,
    /// neighbouring triangles join a chart while they face within this of its first (degrees)
    pub chart_angle_deg: f32,
    /// empty texels round each chart
    pub padding: u32,
    /// the atlas' width; its height is what the tiles need
    pub atlas_width: u32,
    /// the bounced light's texture is this many times smaller per side
    pub indirect_downscale: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BakeSettings {
    /// sun rays a texel, a square number (a grid over the texel, each to its own part of the disc)
    pub sun_samples: u32,
    pub ao_rays: u32,
    /// how far ambient occlusion looks (m)
    pub ao_reach: f32,
    pub ao_strength: f32,
    pub ao_floor: f32,
    pub ao_contrast: f32,
    /// the sun's distance field reaches this many texels each side of a shadow's edge
    pub sdf_radius: f32,
    pub gi_rays: u32,
    pub gi_bounces: u32,
    /// how far bounced light is gathered from (m); beyond, the sky
    pub gi_reach: f32,
    /// a texel whose rays meet this share of back faces lies inside a solid
    pub inside_share: f32,
    /// the denoiser's radius on bounced light (texels)
    pub gi_filter: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VolumeSettings {
    pub min: [f32; 3],
    pub max: [f32; 3],
    /// between probes (m)
    pub step: f32,
    pub rays: u32,
}

/// A loaded scene: its header and each mesh's triangles.
pub struct Scene {
    pub header: SceneHeader,
    /// per mesh, three vertices a triangle
    pub positions: Vec<Vec<Vec3>>,
    /// of the whole file: what the bake was made from
    pub hash: blake3::Hash,
}

impl Scene {
    pub fn read(bytes: &[u8]) -> Result<Scene> {
        ensure!(bytes.len() >= 12 && &bytes[0..4] == SCENE_MAGIC, "not a lightbaker scene");
        let version = u32::from_le_bytes(bytes[4..8].try_into()?);
        ensure!(version == SCENE_VERSION, "scene format {version}, this baker reads {SCENE_VERSION}");
        let len = u32::from_le_bytes(bytes[8..12].try_into()?) as usize;
        let header: SceneHeader = serde_json::from_slice(bytes.get(12..12 + len).context("scene header cut short")?).context("scene header")?;
        let mut at = align4(12 + len);
        let mut positions = Vec::with_capacity(header.meshes.len());
        for m in &header.meshes {
            ensure!((m.material as usize) < header.materials.len(), "mesh {}: no material {}", m.name, m.material);
            let n = m.triangles as usize * 9 * 4;
            let raw = bytes.get(at..at + n).with_context(|| format!("mesh {}: positions cut short", m.name))?;
            let floats: &[f32] = bytemuck::try_cast_slice(raw).map_err(|e| anyhow::anyhow!("mesh {}: {e:?}", m.name))?;
            positions.push(floats.chunks_exact(3).map(|p| Vec3::new(p[0], p[1], p[2])).collect());
            at = align4(at + n);
        }
        if at != bytes.len() {
            bail!("scene: {} bytes after the last mesh", bytes.len() - at);
        }
        header.validate()?;
        Ok(Scene { header, positions, hash: blake3::hash(bytes) })
    }

    /// The file's bytes (what an engine writes; the tests build scenes with it).
    pub fn write(header: &SceneHeader, positions: &[Vec<Vec3>]) -> Result<Vec<u8>> {
        ensure!(header.meshes.len() == positions.len(), "one positions array a mesh");
        let head = serde_json::to_vec(header)?;
        let mut out = Vec::new();
        out.extend_from_slice(SCENE_MAGIC);
        out.extend_from_slice(&SCENE_VERSION.to_le_bytes());
        out.extend_from_slice(&(head.len() as u32).to_le_bytes());
        out.extend_from_slice(&head);
        for (m, p) in header.meshes.iter().zip(positions) {
            ensure!(p.len() == m.triangles as usize * 3, "mesh {}: {} vertices for {} triangles", m.name, p.len(), m.triangles);
            out.resize(align4(out.len()), 0);
            for v in p {
                out.extend_from_slice(bytemuck::bytes_of(&v.to_array()));
            }
        }
        out.resize(align4(out.len()), 0);
        Ok(out)
    }

    pub fn triangle_count(&self) -> usize {
        self.positions.iter().map(|p| p.len() / 3).sum()
    }
}

impl SceneHeader {
    fn validate(&self) -> Result<()> {
        let l = &self.lightmap;
        ensure!(l.texels_per_meter > 0.0 && l.tile_size > 0.0, "lightmap: texelsPerMeter and tileSize must be positive");
        ensure!(l.atlas_width >= 64 && l.atlas_width % 4 == 0, "lightmap: atlasWidth must be a multiple of 4, at least 64");
        ensure!(matches!(l.indirect_downscale, 1 | 2 | 4), "lightmap: indirectDownscale 1, 2 or 4 (the tiles' grid)");
        let s = (self.bake.sun_samples as f32).sqrt() as u32;
        ensure!(s * s == self.bake.sun_samples && s >= 1, "bake: sunSamples must be a square number");
        ensure!(self.bake.ao_rays >= 1 && self.bake.gi_rays >= 1, "bake: aoRays and giRays at least one");
        if let Some(v) = &self.volume {
            ensure!(v.step > 0.0 && (0..3).all(|i| v.max[i] >= v.min[i]), "volume: bad bounds or step");
        }
        Ok(())
    }
}

pub fn align4(n: usize) -> usize {
    (n + 3) & !3
}
