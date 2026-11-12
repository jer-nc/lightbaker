/**
 * BC5 decoded on the CPU, for a GPU that cannot sample it (WebGL without
 * EXT_texture_compression_rgtc): the KTX2's blocks back to two bytes a texel.
 */

function bc4(src: Uint8Array, at: number, out: Uint8Array, w: number, bx: number, by: number, channel: number): void {
  const a = src[at]!, b = src[at + 1]!;
  const p = [a, b];
  if (a > b) for (let i = 1; i < 7; i++) p.push(Math.round(((7 - i) * a + i * b) / 7));
  else {
    for (let i = 1; i < 5; i++) p.push(Math.round(((5 - i) * a + i * b) / 5));
    p.push(0, 255);
  }
  // 48 bits of 3-bit indices, as two halves (no BigInt in the hot loop)
  const lo = src[at + 2]! | (src[at + 3]! << 8) | (src[at + 4]! << 16), hi = src[at + 5]! | (src[at + 6]! << 8) | (src[at + 7]! << 16);
  for (let k = 0; k < 16; k++) {
    const idx = k < 8 ? (lo >> (3 * k)) & 7 : (hi >> (3 * (k - 8))) & 7;
    out[((by + (k >> 2)) * w + bx + (k & 3)) * 2 + channel] = p[idx]!;
  }
}

/** BC5 blocks (`w` x `h`, multiples of 4) to interleaved pairs. */
export function decodeBC5(src: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h * 2);
  let at = 0;
  for (let by = 0; by < h; by += 4) {
    for (let bx = 0; bx < w; bx += 4) {
      bc4(src, at, out, w, bx, by, 0);
      bc4(src, at + 8, out, w, bx, by, 1);
      at += 16;
    }
  }
  return out;
}

/** An RGB9E5 texel (shared exponent) as linear RGB. */
export function rgb9e5(p: number): [number, number, number] {
  const s = 2 ** ((p >>> 27) - 24);
  return [(p & 511) * s, ((p >>> 9) & 511) * s, ((p >>> 18) & 511) * s];
}
