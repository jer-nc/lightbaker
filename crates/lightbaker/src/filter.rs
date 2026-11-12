//! The texel passes run on the CPU, a tile at a time: the denoiser (edge-aware, never across a chart
//! or a fold), the sun's distance field, and filling texels without light of their own.

/// A tile's texels' chart (u32::MAX: empty) and face normal: what the passes keep within.
pub struct Surface<'a> {
    pub w: usize,
    pub h: usize,
    pub chart: &'a [u32],
    pub normal: &'a [[f32; 3]],
}

impl Surface<'_> {
    fn same(&self, i: usize, j: usize) -> bool {
        let (a, b) = (self.normal[i], self.normal[j]);
        self.chart[i] == self.chart[j] && a[0] * b[0] + a[1] * b[1] + a[2] * b[2] >= 0.9
    }
}

/// Gaussian-weighted mean of the valid texels round each one on the same surface (`valid[i]`).
pub fn smooth<const N: usize>(s: &Surface, v: &[[f32; N]], valid: &[bool], r: usize) -> Vec<[f32; N]> {
    let mut out = v.to_vec();
    if r == 0 {
        return out;
    }
    let sigma2 = 2.0 * (r as f32 * 0.75).powi(2);
    for y in 0..s.h {
        for x in 0..s.w {
            let i = y * s.w + x;
            if !valid[i] {
                continue;
            }
            let (mut sum, mut wsum) = ([0f32; N], 0f32);
            for yy in y.saturating_sub(r)..(y + r + 1).min(s.h) {
                for xx in x.saturating_sub(r)..(x + r + 1).min(s.w) {
                    let j = yy * s.w + xx;
                    if !valid[j] || !s.same(i, j) {
                        continue;
                    }
                    let (dx, dy) = (xx as f32 - x as f32, yy as f32 - y as f32);
                    let k = (-(dx * dx + dy * dy) / sigma2).exp();
                    for c in 0..N {
                        sum[c] += v[j][c] * k;
                    }
                    wsum += k;
                }
            }
            for c in 0..N {
                out[i][c] = sum[c] / wsum;
            }
        }
    }
    out
}

/// Gives texels without light of their own (`valid` false: inside solids, padding, slivers no texel
/// centre fell in) the mean of their valid neighbours, a ring at a time: a chart's texel from its own
/// chart, an empty one from anything. Each filled texel becomes valid.
pub fn fill<const N: usize>(s: &Surface, v: &mut [[f32; N]], valid: &mut [bool], rings: usize) {
    for _ in 0..rings {
        let was = valid.to_vec();
        let src = v.to_vec();
        for y in 0..s.h {
            for x in 0..s.w {
                let i = y * s.w + x;
                if was[i] {
                    continue;
                }
                let (mut own, mut no, mut any, mut na) = ([0f32; N], 0, [0f32; N], 0);
                for yy in y.saturating_sub(1)..(y + 2).min(s.h) {
                    for xx in x.saturating_sub(1)..(x + 2).min(s.w) {
                        let j = yy * s.w + xx;
                        if !was[j] {
                            continue;
                        }
                        if s.chart[i] != u32::MAX && s.chart[j] == s.chart[i] {
                            add(&mut own, &src[j]);
                            no += 1;
                        } else {
                            add(&mut any, &src[j]);
                            na += 1;
                        }
                    }
                }
                if no > 0 {
                    v[i] = own.map(|c| c / no as f32);
                    valid[i] = true;
                } else if na > 0 && s.chart[i] == u32::MAX {
                    v[i] = any.map(|c| c / na as f32);
                    valid[i] = true;
                }
            }
        }
    }
}

fn add<const N: usize>(a: &mut [f32; N], b: &[f32; N]) {
    for c in 0..N {
        a[c] += b[c];
    }
}

/// The sun's signed distance field: from each texel's centre to the nearest sample of the other
/// state (lit or shadowed) on the same surface, in texels, positive where lit, mapped so that 0.5 is
/// the shadow's edge and 0 and 1 are `radius` texels inside and out. `hard[i]` holds the texel's
/// `g` x `g` sub-samples (bit s: cell s % g, s / g lit).
pub fn sun_field(s: &Surface, hard: &[u32], covered: &[bool], g: usize, radius: f32) -> Vec<f32> {
    let reach = radius.ceil() as usize + 1;
    let at = |i: usize, c: usize| hard[i] >> c & 1 == 1;
    let mut out = vec![-1f32; s.w * s.h];
    for y in 0..s.h {
        for x in 0..s.w {
            let i = y * s.w + x;
            if !covered[i] {
                continue;
            }
            // the texel's own state: most of its samples
            let lit = (hard[i].count_ones() as usize) * 2 >= g * g;
            let mut best = radius;
            for yy in y.saturating_sub(reach)..(y + reach + 1).min(s.h) {
                for xx in x.saturating_sub(reach)..(x + reach + 1).min(s.w) {
                    let j = yy * s.w + xx;
                    if !covered[j] || !s.same(i, j) {
                        continue;
                    }
                    for c in 0..g * g {
                        if at(j, c) == lit {
                            continue;
                        }
                        let px = xx as f32 + ((c % g) as f32 + 0.5) / g as f32 - (x as f32 + 0.5);
                        let py = yy as f32 + ((c / g) as f32 + 0.5) / g as f32 - (y as f32 + 0.5);
                        // the edge lies half way to the other state's sample
                        best = best.min((px * px + py * py).sqrt() - 0.5 / g as f32);
                    }
                }
            }
            let d = best.max(0.0).min(radius);
            out[i] = 0.5 + 0.5 * if lit { d } else { -d } / radius;
        }
    }
    out
}
