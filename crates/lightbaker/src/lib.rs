//! A GPU light baker. It reads a neutral scene (triangles, materials, a sun and a sky, see
//! [`scene`]) and writes what an engine needs to light it ([`output`]): a lightmap atlas (the sun's
//! shadow as a distance field, ambient occlusion, the light bounced between surfaces and from the
//! sky), each chart's projection into it, and an irradiance volume for what moves. It knows nothing
//! of any engine. The ground plan is cut into tiles baked on their own, so a bake reuses every tile
//! a change does not reach ([`deps`], [`cache`]).

pub mod bake;
pub mod bvh;
pub mod cache;
pub mod deps;
pub mod encode;
pub mod filter;
pub mod gpu;
pub mod layout;
pub mod output;
pub mod raster;
pub mod scene;
pub mod volume;

pub use bake::{Bake, Options, bake};
pub use gpu::Gpu;
pub use scene::Scene;
