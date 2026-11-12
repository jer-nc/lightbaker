/**
 * The scene a bake reads (`.lbscene`, docs/FORMAT.md): the magic `LBSC`, the format version, the
 * header's length, the header (JSON), then each mesh's positions (f32, three vertices a triangle, not
 * indexed), each on a 4-byte boundary. Y up, metres.
 */

export const SCENE_VERSION = 1;

export type Rgb = [number, number, number];

export interface SceneHeader {
  sun: { direction: [number, number, number]; radiusDeg: number; irradiance: Rgb };
  /** radiance of the sky from the horizon to the zenith, and of the ground below the horizon */
  sky: { zenith: Rgb; horizon: Rgb; ground: Rgb };
  materials: SceneMaterial[];
  meshes: { name: string; material: number; triangles: number }[];
  lightmap: LightmapSettings;
  bake: BakeSettings;
  volume?: VolumeSettings | null;
  /** anything wanted back with the bake, copied untouched */
  user?: unknown;
}

export interface SceneMaterial {
  name: string;
  /** diffuse reflectance, linear RGB */
  albedo: Rgb;
  receives: boolean;
  casts: boolean;
  /** share of light stopped where it casts (leaves: below one) */
  opacity?: number;
}

export interface LightmapSettings {
  texelsPerMeter: number;
  tileSize: number;
  chartAngleDeg: number;
  padding: number;
  atlasWidth: number;
  indirectDownscale: number;
}

export interface BakeSettings {
  sunSamples: number;
  aoRays: number;
  aoReach: number;
  aoStrength: number;
  aoFloor: number;
  aoContrast: number;
  sdfRadius: number;
  giRays: number;
  giBounces: number;
  giReach: number;
  insideShare: number;
  giFilter: number;
}

export interface VolumeSettings {
  min: [number, number, number];
  max: [number, number, number];
  step: number;
  rays: number;
}

const align4 = (n: number) => (n + 3) & ~3;

/** The file's bytes. `positions[i]` holds mesh i's vertices (x y z, three a triangle). */
export function writeScene(header: SceneHeader, positions: readonly Float32Array[]): Uint8Array {
  if (positions.length !== header.meshes.length) throw new Error('scene: one positions array a mesh');
  header.meshes.forEach((m, i) => {
    if (positions[i]!.length !== m.triangles * 9) throw new Error(`scene: mesh ${m.name} has ${positions[i]!.length / 9} triangles, says ${m.triangles}`);
  });
  const head = new TextEncoder().encode(JSON.stringify(header));
  let size = align4(12 + head.length);
  for (const p of positions) size = align4(size + p.byteLength);
  const out = new Uint8Array(size), dv = new DataView(out.buffer);
  out.set([0x4c, 0x42, 0x53, 0x43]);
  dv.setUint32(4, SCENE_VERSION, true);
  dv.setUint32(8, head.length, true);
  out.set(head, 12);
  let at = align4(12 + head.length);
  for (const p of positions) {
    out.set(new Uint8Array(p.buffer, p.byteOffset, p.byteLength), at);
    at = align4(at + p.byteLength);
  }
  return out;
}
