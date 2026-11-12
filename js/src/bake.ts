/**
 * A bake (`.lbake`, docs/FORMAT.md) read in a browser or Node: its header, the atlas' two KTX2
 * textures as they ship (for the engine's KTX2 loader), each chart's projection and each triangle's
 * chart, and the irradiance volume.
 */

export const BAKE_VERSION = 1;

export interface BakeHeader {
  baker: string;
  /** the scene file's blake3 (hex) */
  scene: string;
  user: unknown;
  atlas: { width: number; height: number; texelsPerMeter: number; sdfRadius: number; indirectWidth: number; indirectHeight: number; charts: number };
  /** the sky's irradiance on an open face looking up (linear RGB) */
  skyIrradiance: [number, number, number];
  meshes: { name: string; triangles: number }[];
  volume: { origin: [number, number, number]; step: number; size: [number, number, number]; scale: number } | null;
  chunks: { name: string; offset: number; length: number }[];
  stats: { tiles: number; tilesBaked: number; texelsBaked: number; chunks: number; chunksBaked: number; seconds: number };
}

export interface LightBake {
  header: BakeHeader;
  /** KTX2, BC5: red the sun's distance field (0.5 on a shadow's edge), green ambient occlusion */
  direct: Uint8Array;
  /** KTX2, RGB9E5: the light from the sky and other surfaces (irradiance, linear) */
  indirect: Uint8Array;
  /** two rows (x y z constant) a chart: world point to atlas texel coordinates */
  charts: Float32Array;
  /** every mesh's triangles, in order: its chart (0xffffffff: none) */
  triangleCharts: Uint32Array;
  /** 16 floats a probe (see docs/FORMAT.md) */
  volume: Float32Array | null;
  /** per atlas texel, how it got its light (VALIDITY), for tools; null in a bake without it */
  validity: Uint8Array | null;
}

/** How a texel got its light (a bake's `validity`). */
export const VALIDITY = ['none', 'centre', 'offset', 'pushed out', 'filled from its chart', 'volume'] as const;

/** Reads a bake (null when it is not one of this version). Deflated chunks are inflated natively. */
export async function readBake(buf: ArrayBuffer): Promise<LightBake | null> {
  if (buf.byteLength < 12) return null;
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== 0x4b41424c || dv.getUint32(4, true) !== BAKE_VERSION) return null;
  const len = dv.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 12, len))) as BakeHeader;
  const base = Math.ceil((12 + len) / 8) * 8;
  const chunk = (name: string): Uint8Array | null => {
    const c = header.chunks.find((k) => k.name === name);
    return c ? new Uint8Array(buf, base + c.offset, c.length) : null;
  };
  const need = (name: string): Uint8Array => {
    const c = chunk(name);
    if (!c) throw new Error(`bake: no ${name} chunk`);
    return c;
  };
  const vol = chunk('volume'), val = chunk('validity');
  const [charts, tris, volume, validity] = await Promise.all([inflate(need('charts')), inflate(need('triangleCharts')), vol ? inflate(vol) : null, val ? inflate(val) : null]);
  return {
    header,
    direct: need('direct'),
    indirect: need('indirect'),
    charts: new Float32Array(charts.buffer, charts.byteOffset, charts.byteLength / 4),
    triangleCharts: new Uint32Array(tris.buffer, tris.byteOffset, tris.byteLength / 4),
    volume: volume ? probes(volume, header.volume!.scale) : null,
    validity,
  };
}

async function inflate(b: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([b as Uint8Array<ArrayBuffer>]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** The volume's bytes (docs/FORMAT.md) as 16 floats a probe: L0 rgb, sun, L1 x rgb, -, L1 y rgb, -, L1 z rgb, -. */
function probes(b: Uint8Array, scale: number): Float32Array {
  const n = b.length / 16, out = new Float32Array(n * 16), SQRT3 = Math.sqrt(3);
  for (let p = 0; p < n; p++) {
    const i = p * 16;
    for (let c = 0; c < 3; c++) {
      const l0 = scale * (b[i + c]! / 255) ** 2;
      out[i + c] = l0;
      for (let a = 0; a < 3; a++) out[i + 4 * (a + 1) + c] = ((b[i + 4 + a * 3 + c]! / 255) * 2 - 1) * SQRT3 * l0;
    }
    out[i + 3] = b[i + 3]! / 255;
  }
  return out;
}
