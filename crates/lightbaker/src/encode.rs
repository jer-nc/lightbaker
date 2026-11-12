//! What the GPU samples, as it ships: block-compressed or packed texels in KTX2 files (zstd
//! supercompressed), read by any KTX2 loader without a transcoder.

use anyhow::Result;

/// BC4: one channel, 4 x 4 texels in 8 bytes (two end values and eight steps between them).
pub fn bc4_block(v: &[u8; 16]) -> [u8; 8] {
    let (hi, lo) = (*v.iter().max().unwrap(), *v.iter().min().unwrap());
    let mut out = [0u8; 8];
    out[0] = hi;
    out[1] = lo;
    if hi == lo {
        return out;
    }
    // hi > lo: index 0 is hi, 1 is lo, 2..7 run from hi to lo in sevenths
    let palette: [f32; 8] = std::array::from_fn(|i| match i {
        0 => hi as f32,
        1 => lo as f32,
        _ => ((8 - i) as f32 * hi as f32 + (i - 1) as f32 * lo as f32) / 7.0,
    });
    let mut bits: u64 = 0;
    for (k, &x) in v.iter().enumerate() {
        let mut best = 0;
        for i in 1..8 {
            if (palette[i] - x as f32).abs() < (palette[best] - x as f32).abs() {
                best = i;
            }
        }
        bits |= (best as u64) << (3 * k);
    }
    out[2..8].copy_from_slice(&bits.to_le_bytes()[0..6]);
    out
}

/// BC5: two channels (each a BC4 block), `w` x `h` (multiples of 4), from interleaved pairs.
pub fn bc5(rg: &[[u8; 2]], w: usize, h: usize) -> Vec<u8> {
    assert!(w % 4 == 0 && h % 4 == 0 && rg.len() == w * h);
    let mut out = Vec::with_capacity(w * h);
    for by in (0..h).step_by(4) {
        for bx in (0..w).step_by(4) {
            for c in 0..2 {
                let block: [u8; 16] = std::array::from_fn(|k| rg[(by + k / 4) * w + bx + k % 4][c]);
                out.extend_from_slice(&bc4_block(&block));
            }
        }
    }
    out
}

/// RGB in a shared exponent (9 bits a channel, 5 of exponent): HDR in four bytes.
pub fn rgb9e5(c: [f32; 3]) -> u32 {
    const MAX: f32 = 65408.0;
    let [r, g, b] = c.map(|v| v.clamp(0.0, MAX));
    let m = r.max(g).max(b);
    if m <= 0.0 {
        return 0;
    }
    let mut e = (m.log2().floor() as i32).max(-16) + 1 + 15;
    let mut scale = 2f32.powi(e - 15 - 9);
    if (m / scale + 0.5).floor() as u32 == 512 {
        e += 1;
        scale *= 2.0;
    }
    let q = |v: f32| ((v / scale + 0.5).floor() as u32).min(511);
    q(r) | (q(g) << 9) | (q(b) << 18) | ((e as u32) << 27)
}

pub fn rgb9e5_decode(p: u32) -> [f32; 3] {
    let s = 2f32.powi((p >> 27) as i32 - 15 - 9);
    [(p & 511) as f32 * s, ((p >> 9) & 511) as f32 * s, ((p >> 18) & 511) as f32 * s]
}

#[derive(Clone, Copy)]
pub enum Format {
    /// two linear channels in BC5
    Bc5,
    /// linear HDR RGB, packed
    Rgb9e5,
}

impl Format {
    fn vk(self) -> u32 {
        match self {
            Format::Bc5 => 141,    // VK_FORMAT_BC5_UNORM_BLOCK
            Format::Rgb9e5 => 123, // VK_FORMAT_E5B9G9R9_UFLOAT_PACK32
        }
    }
    fn type_size(self) -> u32 {
        match self {
            Format::Bc5 => 1,
            Format::Rgb9e5 => 4,
        }
    }
}

/// A one-level 2D KTX2 file around `data`, zstd supercompressed.
pub fn ktx2(format: Format, w: u32, h: u32, data: &[u8]) -> Result<Vec<u8>> {
    let packed = zstd::bulk::compress(data, 19)?;
    let dfd = dfd(format);
    const HEADER: usize = 12 + 9 * 4 + 4 * 4 + 2 * 8;
    const LEVELS: usize = 3 * 8;
    let dfd_at = HEADER + LEVELS;
    let data_at = (dfd_at + dfd.len()).div_ceil(8) * 8;
    let mut out = Vec::with_capacity(data_at + packed.len());
    out.extend_from_slice(&[0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A]);
    for v in [format.vk(), format.type_size(), w, h, 0, 0, 1, 1, 2] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    // the data format descriptor, no key/value data, no supercompression global data
    for v in [dfd_at as u32, dfd.len() as u32, 0, 0] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    for v in [data_at as u64, packed.len() as u64, data.len() as u64] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&dfd);
    out.resize(data_at, 0);
    out.extend_from_slice(&packed);
    Ok(out)
}

/// The basic data format descriptor of a format (Khronos Data Format 1.3), linear, BT.709 primaries.
fn dfd(format: Format) -> Vec<u8> {
    // (colour model, block dimensions - 1, bytes in a block's plane, samples: (bit offset, bit length - 1, channel))
    let (model, dims, bytes, samples): (u8, [u8; 4], u8, &[(u16, u8, u8)]) = match format {
        Format::Bc5 => (133, [3, 3, 0, 0], 16, &[(0, 63, 0), (64, 63, 1)]),
        Format::Rgb9e5 => (1, [0, 0, 0, 0], 4, &[(0, 8, 0), (9, 8, 1), (18, 8, 2)]),
    };
    let block = 24 + 16 * samples.len();
    let mut d = Vec::with_capacity(4 + block);
    d.extend_from_slice(&((4 + block) as u32).to_le_bytes());
    d.extend_from_slice(&0u32.to_le_bytes()); // vendor 0 (Khronos), descriptor type 0
    d.extend_from_slice(&2u16.to_le_bytes()); // version
    d.extend_from_slice(&(block as u16).to_le_bytes());
    d.extend_from_slice(&[model, 1, 1, 0]); // model, BT.709, linear, no flags
    d.extend_from_slice(&dims);
    d.extend_from_slice(&[bytes, 0, 0, 0, 0, 0, 0, 0]);
    for &(offset, len, channel) in samples {
        d.extend_from_slice(&offset.to_le_bytes());
        d.extend_from_slice(&[len, channel]);
        d.extend_from_slice(&[0; 4]); // sample positions
        d.extend_from_slice(&0u32.to_le_bytes()); // lower
        d.extend_from_slice(&u32::MAX.to_le_bytes()); // upper
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bc4_keeps_values_within_a_seventh() {
        let v: [u8; 16] = std::array::from_fn(|i| (i * 13) as u8);
        let b = bc4_block(&v);
        let bits = u64::from_le_bytes([b[2], b[3], b[4], b[5], b[6], b[7], 0, 0]);
        let (hi, lo) = (b[0] as f32, b[1] as f32);
        for (k, &x) in v.iter().enumerate() {
            let i = ((bits >> (3 * k)) & 7) as usize;
            let p = match i {
                0 => hi,
                1 => lo,
                _ => ((8 - i) as f32 * hi + (i - 1) as f32 * lo) / 7.0,
            };
            assert!((p - x as f32).abs() <= (hi - lo) / 14.0 + 0.5);
        }
    }

    #[test]
    fn rgb9e5_round_trips() {
        for c in [[0.0, 0.0, 0.0], [1.0, 0.5, 0.25], [3.2, 0.01, 7.5], [0.001, 0.002, 0.0005]] {
            let d = rgb9e5_decode(rgb9e5(c));
            let m = c.iter().cloned().fold(0.0, f32::max);
            for i in 0..3 {
                assert!((d[i] - c[i]).abs() <= m / 256.0 + 1e-6, "{c:?} -> {d:?}");
            }
        }
    }
}
