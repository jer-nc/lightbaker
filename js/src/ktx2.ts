/**
 * A bake's KTX2 textures read back to their texels (one level, 2D, as the baker writes them): the
 * format, the size and the data, zstd-decompressed by the decoder the engine already has (three's
 * ZSTDDecoder, say), so the engine uploads them as it likes.
 */

/** Vulkan formats the baker writes. */
export const VK_FORMAT_BC5_UNORM_BLOCK = 141;
export const VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 = 123;

export interface Ktx2Level {
  vkFormat: number;
  width: number;
  height: number;
  /** the level's bytes, decompressed */
  data: Uint8Array;
}

/** `zstd(bytes, size)` decompresses a zstd frame to `size` bytes. */
export function readKtx2(bytes: Uint8Array, zstd: (b: Uint8Array, size: number) => Uint8Array): Ktx2Level {
  const id = [0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a];
  if (bytes.length < 104 || id.some((b, i) => bytes[i] !== b)) throw new Error('ktx2: not a KTX2 file');
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const u32 = (at: number) => dv.getUint32(at, true), u64 = (at: number) => Number(dv.getBigUint64(at, true));
  const vkFormat = u32(12), width = u32(20), height = u32(24), levels = u32(40), scheme = u32(44);
  if (levels !== 1) throw new Error(`ktx2: ${levels} levels, one read`);
  const offset = u64(80), length = u64(88), size = u64(96);
  const raw = bytes.subarray(offset, offset + length);
  if (scheme === 0) return { vkFormat, width, height, data: raw };
  if (scheme !== 2) throw new Error(`ktx2: supercompression ${scheme}`);
  return { vkFormat, width, height, data: zstd(raw, size) };
}
