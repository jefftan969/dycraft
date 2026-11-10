// Read a USTAR archive: the format `viewer/mesh_bundle.py` packs a run's meshes in.
//
// Its own module so the loader worker can read the archives without the page's loader chain.

const TAR_BLOCK = 512;

export interface TarEntry {
  name: string;
  data: Uint8Array;
}

/**
 * Walk a USTAR archive.
 *
 * Thirty lines because the format is a 512-byte header with the name at 0 and
 * the size, in octal, at 124 — and because `mesh_bundle.py` deliberately emits
 * nothing else: no long-name extensions, no sparse entries, no PAX headers.
 * The `prefix` field is still honored so this reader is correct for an archive
 * written by anything else.
 */
export function* readTarEntries(buffer: ArrayBuffer): Generator<TarEntry> {
  const bytes = new Uint8Array(buffer);
  const decoder = new TextDecoder();
  const field = (start: number, length: number): string =>
    decoder.decode(bytes.subarray(start, start + length)).replace(/\0.*$/, '');

  let offset = 0;
  while (offset + TAR_BLOCK <= bytes.length) {
    const header = bytes.subarray(offset, offset + TAR_BLOCK);
    // The archive ends with zero blocks; one is enough to stop on.
    if (header.every((byte) => byte === 0)) return;

    const name = field(offset, 100);
    const prefix = field(offset + 345, 155);
    const size = Number.parseInt(field(offset + 124, 12).trim() || '0', 8);
    if (!name || !Number.isFinite(size) || size < 0) return;

    const start = offset + TAR_BLOCK;
    if (start + size > bytes.length) return;    // truncated archive
    yield { name: prefix ? `${prefix}/${name}` : name,
            data: bytes.subarray(start, start + size) };
    offset = start + Math.ceil(size / TAR_BLOCK) * TAR_BLOCK;
  }
}
