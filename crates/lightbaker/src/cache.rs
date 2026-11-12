//! Baked tiles and probe chunks kept by their hash (deps.rs): a tile whose hash is found is not
//! baked again. A cache entry is the tile's arrays, zstd-compressed; a missing or unreadable one is a miss.

use std::path::{Path, PathBuf};

use anyhow::Result;
use bytemuck::Pod;

pub struct Cache {
    dir: Option<PathBuf>,
}

impl Cache {
    pub fn new(dir: Option<&Path>) -> Result<Cache> {
        if let Some(d) = dir {
            std::fs::create_dir_all(d)?;
        }
        Ok(Cache { dir: dir.map(Path::to_path_buf) })
    }

    pub fn get(&self, kind: &str, hash: &blake3::Hash) -> Option<Vec<u8>> {
        let f = self.dir.as_ref()?.join(format!("{kind}-{}.bin", hash.to_hex()));
        zstd::decode_all(std::fs::read(f).ok()?.as_slice()).ok()
    }

    pub fn put(&self, kind: &str, hash: &blake3::Hash, bytes: &[u8]) -> Result<()> {
        if let Some(d) = &self.dir {
            let f = d.join(format!("{kind}-{}.bin", hash.to_hex()));
            let tmp = f.with_extension("tmp");
            std::fs::write(&tmp, zstd::encode_all(bytes, 3)?)?;
            std::fs::rename(tmp, f)?;
        }
        Ok(())
    }
}

/// Arrays laid end to end, each preceded by its byte length.
#[derive(Default)]
pub struct Pack(pub Vec<u8>);

impl Pack {
    pub fn put<T: Pod>(mut self, v: &[T]) -> Self {
        let b: &[u8] = bytemuck::cast_slice(v);
        self.0.extend_from_slice(&(b.len() as u64).to_le_bytes());
        self.0.extend_from_slice(b);
        self
    }
}

pub struct Unpack<'a>(&'a [u8]);

impl<'a> Unpack<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Unpack(b)
    }
    pub fn take<T: Pod>(&mut self) -> Option<Vec<T>> {
        let n = u64::from_le_bytes(self.0.get(..8)?.try_into().ok()?) as usize;
        let b = self.0.get(8..8 + n)?;
        self.0 = &self.0[8 + n..];
        Some(bytemuck::pod_collect_to_vec(b))
    }
}
