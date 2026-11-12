/**
 * Lightmap UVs from a bake's charts: each triangle's three vertices through its chart's projection,
 * divided by the atlas' size. Nothing per vertex ships: the engine computes them from its own
 * geometry (the same triangles, in the order the scene gave them).
 */

/** `positions`: one mesh's vertices (x y z, three a triangle); `first`: its first triangle among all the scene's. */
export function lightmapUVs(positions: ArrayLike<number>, first: number, triangleCharts: Uint32Array, charts: Float32Array, width: number, height: number, out = new Float32Array((positions.length / 3) * 2)): Float32Array {
  const tris = positions.length / 9;
  for (let t = 0; t < tris; t++) {
    const c = triangleCharts[first + t]!;
    for (let k = 0; k < 3; k++) {
      const v = t * 3 + k, x = positions[v * 3]!, y = positions[v * 3 + 1]!, z = positions[v * 3 + 2]!;
      if (c === 0xffffffff) {
        out[v * 2] = out[v * 2 + 1] = 0;
        continue;
      }
      const r = c * 8;
      out[v * 2] = (charts[r]! * x + charts[r + 1]! * y + charts[r + 2]! * z + charts[r + 3]!) / width;
      out[v * 2 + 1] = (charts[r + 4]! * x + charts[r + 5]! * y + charts[r + 6]! * z + charts[r + 7]!) / height;
    }
  }
  return out;
}
