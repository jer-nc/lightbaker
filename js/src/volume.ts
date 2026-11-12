/**
 * The irradiance volume read anywhere: the eight probes round a point blended, giving the light
 * arriving there as L1 spherical harmonics (linear RGB radiance) and how much of the sun it sees.
 */

export interface VolumeGrid {
  origin: [number, number, number];
  step: number;
  size: [number, number, number];
  /** 16 floats a probe: L0 rgb, sun, L1x rgb, -, L1y rgb, -, L1z rgb, - */
  data: Float32Array;
}

/** What a point receives: 12 coefficients (L0, L1 x, y, z; rgb each) and the sun's visibility. */
export interface VolumeSample {
  sh: Float32Array;
  sun: number;
}

export function volumeSample(): VolumeSample {
  return { sh: new Float32Array(12), sun: 0 };
}

const SLOTS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14];

export function sampleVolume(v: VolumeGrid, x: number, y: number, z: number, out: VolumeSample): VolumeSample {
  const [nx, ny, nz] = v.size;
  const clamp = (f: number, n: number) => Math.max(0, Math.min(n - 1, f));
  const fx = clamp((x - v.origin[0]) / v.step, nx), fy = clamp((y - v.origin[1]) / v.step, ny), fz = clamp((z - v.origin[2]) / v.step, nz);
  const x0 = Math.min(Math.floor(fx), nx - 2), y0 = Math.min(Math.floor(fy), ny - 2), z0 = Math.min(Math.floor(fz), nz - 2);
  const tx = fx - x0, ty = fy - y0, tz = fz - z0;
  out.sh.fill(0);
  out.sun = 0;
  for (let k = 0; k < 8; k++) {
    const dx = k & 1, dy = (k >> 1) & 1, dz = (k >> 2) & 1;
    const w = (dx ? tx : 1 - tx) * (dy ? ty : 1 - ty) * (dz ? tz : 1 - tz);
    const i = (((y0 + dy) * nz + (z0 + dz)) * nx + (x0 + dx)) * 16;
    for (let s = 0; s < 12; s++) out.sh[s] = out.sh[s]! + w * v.data[i + SLOTS[s]!]!;
    out.sun += w * v.data[i + 3]!;
  }
  return out;
}

/**
 * Irradiance on a surface facing (nx, ny, nz) from L1 radiance coefficients (the cosine lobe's
 * convolution: pi for band 0, 2 pi / 3 for band 1), linear RGB into `out`.
 */
export function shIrradiance(sh: Float32Array, nx: number, ny: number, nz: number, out: Float32Array | number[]): void {
  const a0 = Math.PI * 0.282095, a1 = ((2 * Math.PI) / 3) * 0.488603;
  for (let c = 0; c < 3; c++) out[c] = Math.max(0, a0 * sh[c]! + a1 * (sh[3 + c]! * nx + sh[6 + c]! * ny + sh[9 + c]! * nz));
}
