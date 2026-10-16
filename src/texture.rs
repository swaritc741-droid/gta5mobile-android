// CPU fallback for desktop BC texture formats unavailable on most phone GPUs.
use anyhow::{ensure, Result};
use std::sync::{Mutex, OnceLock};
use wgpu::TextureFormat as T;

pub fn fallback_format(format: T) -> T {
    match format {
        T::Bc1RgbaUnorm | T::Bc2RgbaUnorm | T::Bc3RgbaUnorm | T::Bc7RgbaUnorm => T::Rgba8Unorm,
        T::Bc1RgbaUnormSrgb | T::Bc2RgbaUnormSrgb | T::Bc3RgbaUnormSrgb | T::Bc7RgbaUnormSrgb => T::Rgba8UnormSrgb,
        T::Bc4RUnorm => T::R8Unorm,
        T::Bc4RSnorm => T::R8Snorm,
        T::Bc5RgUnorm => T::Rg8Unorm,
        T::Bc5RgSnorm => T::Rg8Snorm,
        T::Bc6hRgbUfloat | T::Bc6hRgbFloat => T::Rgba16Float,
        other => other,
    }
}

/// Uploads with fewer decoded pixels than this are decoded on the calling thread: spawning workers
/// costs more than it saves on small mips. This threshold is a starting point, not a measured value.
const PARALLEL_MIN_PIXELS: usize = 512 * 512;
/// Upper bound on decode workers (including the calling thread).
const MAX_DECODE_THREADS: usize = 4;

/// Number of threads a large BC decode may use. `MUGUET_BC_THREADS=1` forces single-threaded decoding,
/// which makes A/B testing on a device possible without rebuilding.
fn decode_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        let auto = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(MAX_DECODE_THREADS);
        std::env::var("MUGUET_BC_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).map(|n| n.clamp(1, 8)).unwrap_or(auto)
    })
}

fn is_supported(format: T) -> bool {
    matches!(
        format,
        T::Bc1RgbaUnorm | T::Bc1RgbaUnormSrgb | T::Bc2RgbaUnorm | T::Bc2RgbaUnormSrgb | T::Bc3RgbaUnorm | T::Bc3RgbaUnormSrgb
            | T::Bc4RUnorm | T::Bc4RSnorm | T::Bc5RgUnorm | T::Bc5RgSnorm | T::Bc6hRgbUfloat | T::Bc6hRgbFloat
            | T::Bc7RgbaUnorm | T::Bc7RgbaUnormSrgb
    )
}

/// Expands an A8 upload (one alpha byte per texel, `bpr` bytes per row) to RGBA8 with RGB = 0.
/// Missing source bytes read as 0, matching the previous per-texel `get(..).unwrap_or(0)` behaviour.
pub fn expand_a8(data: &[u8], w: usize, h: usize, bpr: usize) -> Vec<u8> {
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let mut out = vec![0u8; w * h * 4];
    for (r, dst_row) in out.chunks_exact_mut(w * 4).enumerate() {
        let src = r.checked_mul(bpr).and_then(|at| data.get(at..)).unwrap_or(&[]);
        for (texel, a) in dst_row.chunks_exact_mut(4).zip(&src[..w.min(src.len())]) {
            texel[3] = *a;
        }
    }
    out
}

#[derive(Clone, Copy)]
struct Layout {
    w: usize,
    bx: usize,
    pitch: usize,
    layer_stride: usize,
    block_bytes: usize,
    pixel_bytes: usize,
    out_pitch: usize,
}

/// A run of whole block rows of one layer, as a contiguous slice of the output.
struct Band<'a> {
    z: usize,
    y0: usize,
    out: &'a mut [u8],
}

/// Decodes one 4x4 block into `dst`, whose rows are `pitch` bytes apart (`dst` must cover 3 rows + 4 texels).
fn decode_block(format: T, block: &[u8], dst: &mut [u8], pitch: usize) -> Result<()> {
    match format {
        T::Bc1RgbaUnorm | T::Bc1RgbaUnormSrgb => bcdec_rs::bc1(block, dst, pitch as _),
        T::Bc2RgbaUnorm | T::Bc2RgbaUnormSrgb => bcdec_rs::bc2(block, dst, pitch as _),
        T::Bc3RgbaUnorm | T::Bc3RgbaUnormSrgb => bcdec_rs::bc3(block, dst, pitch as _),
        T::Bc4RUnorm | T::Bc4RSnorm => bcdec_rs::bc4(block, dst, pitch as _, format == T::Bc4RSnorm),
        T::Bc5RgUnorm | T::Bc5RgSnorm => bcdec_rs::bc5(block, dst, pitch as _, format == T::Bc5RgSnorm),
        T::Bc6hRgbUfloat | T::Bc6hRgbFloat => {
            let mut rgb = [0u16; 48];
            bcdec_rs::bc6h_half(block, &mut rgb, 12, format == T::Bc6hRgbFloat);
            for row in 0..4 {
                for col in 0..4 {
                    let (o, s) = (row * pitch + col * 8, (row * 4 + col) * 3);
                    for c in 0..3 {
                        dst[o + c * 2..o + c * 2 + 2].copy_from_slice(&rgb[s + c].to_le_bytes());
                    }
                    dst[o + 6..o + 8].copy_from_slice(&0x3c00u16.to_le_bytes());
                }
            }
        }
        T::Bc7RgbaUnorm | T::Bc7RgbaUnormSrgb => bcdec_rs::bc7(block, dst, pitch as _),
        _ => anyhow::bail!("unsupported BC format {format:?}"),
    }
    Ok(())
}

fn decode_band(format: T, data: &[u8], l: &Layout, band: &mut Band) -> Result<()> {
    let rows = band.out.len() / l.out_pitch;
    for local in 0..rows.div_ceil(4) {
        let row_base = band.z * l.layer_stride + (band.y0 + local) * l.pitch;
        for x in 0..l.bx {
            let at = row_base + x * l.block_bytes;
            let block = &data[at..at + l.block_bytes];
            let (px, py) = (x * 4, local * 4);
            if px + 4 <= l.w && py + 4 <= rows {
                // Whole block inside the image: decode straight into the output, no staging copy.
                decode_block(format, block, &mut band.out[py * l.out_pitch + px * l.pixel_bytes..], l.out_pitch)?;
            } else {
                // Partial block on the right/bottom edge: decode to a scratch block and copy the visible part.
                let mut tmp = [0u8; 128];
                decode_block(format, block, &mut tmp, 4 * l.pixel_bytes)?;
                let n = 4.min(l.w - px) * l.pixel_bytes;
                for row in 0..4.min(rows - py) {
                    let dst = (py + row) * l.out_pitch + px * l.pixel_bytes;
                    band.out[dst..dst + n].copy_from_slice(&tmp[row * 4 * l.pixel_bytes..][..n]);
                }
            }
        }
    }
    Ok(())
}

/// Decodes a BC image (`depth` layers, `pitch` bytes per block row, `rows` block rows per layer; 0 / too-small
/// values fall back to tightly packed) to the matching `fallback_format`. Returns the pixels and their row pitch.
pub fn decode(format: T, data: &[u8], w: u32, h: u32, depth: u32, pitch: u32, rows: u32) -> Result<(Vec<u8>, u32)> {
    decode_impl(format, data, w, h, depth, pitch, rows, decode_threads(), PARALLEL_MIN_PIXELS)
}

#[allow(clippy::too_many_arguments)]
fn decode_impl(format: T, data: &[u8], w: u32, h: u32, depth: u32, pitch: u32, rows: u32, threads: usize, parallel_min_pixels: usize) -> Result<(Vec<u8>, u32)> {
    ensure!(format.is_compressed() && w > 0 && h > 0 && depth > 0, "invalid BC dimensions/format");
    ensure!(is_supported(format), "unsupported BC format {format:?}");
    let (w, h, depth) = (w as usize, h as usize, depth as usize);
    let block_bytes = format.block_copy_size(None).unwrap_or(16) as usize;
    let pixel_bytes = fallback_format(format).block_copy_size(None).unwrap_or(4) as usize;
    let bx = w.div_ceil(4);
    let by = h.div_ceil(4);
    let row_bytes = bx.checked_mul(block_bytes).ok_or_else(|| anyhow::anyhow!("BC pitch overflow"))?;
    let pitch = if pitch == 0 { row_bytes } else { pitch as usize };
    let rows = (rows as usize).max(by);
    ensure!(pitch >= row_bytes, "BC row pitch is too small");
    let layer_stride = pitch.checked_mul(rows).ok_or_else(|| anyhow::anyhow!("BC pitch overflow"))?;
    let required = pitch
        .checked_mul(by - 1)
        .and_then(|last_row| layer_stride.checked_mul(depth - 1)?.checked_add(last_row))
        .and_then(|n| n.checked_add(row_bytes))
        .ok_or_else(|| anyhow::anyhow!("BC size overflow"))?;
    ensure!(data.len() >= required, "truncated BC upload: {} bytes, need {required}", data.len());
    let out_pitch = w.checked_mul(pixel_bytes).ok_or_else(|| anyhow::anyhow!("decoded pitch overflow"))?;
    let out_pitch_u32: u32 = out_pitch.try_into()?;
    let layer_len = out_pitch.checked_mul(h).ok_or_else(|| anyhow::anyhow!("decoded size overflow"))?;
    let len = layer_len.checked_mul(depth).ok_or_else(|| anyhow::anyhow!("decoded size overflow"))?;
    // `data.len() >= required` above already bounds `len` to a small multiple of the input size.
    let mut output = Vec::new();
    output.try_reserve_exact(len)?;
    output.resize(len, 0);

    let layout = Layout { w, bx, pitch, layer_stride, block_bytes, pixel_bytes, out_pitch };
    let threads = threads.max(1);
    let parallel = threads > 1 && len / pixel_bytes >= parallel_min_pixels;
    let band_rows = if parallel { by.div_ceil(threads * 2).max(1) } else { by };
    let mut bands: Vec<Band> = Vec::new();
    for (z, layer) in output.chunks_mut(layer_len).enumerate() {
        let mut y0 = 0;
        for chunk in layer.chunks_mut(band_rows * 4 * out_pitch) {
            let block_rows = (chunk.len() / out_pitch).div_ceil(4);
            bands.push(Band { z, y0, out: chunk });
            y0 += block_rows;
        }
    }

    if !parallel || bands.len() < 2 {
        for band in bands.iter_mut() {
            decode_band(format, data, &layout, band)?;
        }
    } else {
        // Bands are pulled from a shared queue, so a worker that fails to spawn (or a slow little core)
        // only reduces parallelism; the calling thread always drains whatever is left.
        let workers = threads.min(bands.len());
        let queue = Mutex::new(bands);
        let failure: Mutex<Option<anyhow::Error>> = Mutex::new(None);
        let work = || loop {
            let next = queue.lock().unwrap_or_else(|e| e.into_inner()).pop();
            let Some(mut band) = next else { break };
            if let Err(e) = decode_band(format, data, &layout, &mut band) {
                let mut slot = failure.lock().unwrap_or_else(|e| e.into_inner());
                if slot.is_none() {
                    *slot = Some(e);
                }
                queue.lock().unwrap_or_else(|e| e.into_inner()).clear();
                break;
            }
        };
        std::thread::scope(|s| {
            for _ in 1..workers {
                let _ = std::thread::Builder::new().name("bc-decode".into()).spawn_scoped(s, work);
            }
            work();
        });
        if let Some(e) = failure.into_inner().unwrap_or_else(|e| e.into_inner()) {
            return Err(e);
        }
    }
    Ok((output, out_pitch_u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bc1_red_and_partial_block() {
        let block = [0, 0xf8, 0, 0, 0, 0, 0, 0];
        let (rgba, pitch) = decode(T::Bc1RgbaUnorm, &block, 3, 2, 1, 8, 1).unwrap();
        assert_eq!(pitch, 12);
        assert_eq!(rgba, [255, 0, 0, 255].repeat(6));
    }

    #[test]
    fn padded_rows_and_layers() {
        let mut bytes = [0u8; 48];
        for at in [0, 16, 32] { bytes[at + 1] = 0xf8; }
        let (rgba, _) = decode(T::Bc1RgbaUnorm, &bytes, 4, 5, 1, 16, 2).unwrap();
        assert_eq!(rgba, [255, 0, 0, 255].repeat(20));
        let (rgba, _) = decode(T::Bc1RgbaUnorm, &bytes, 4, 4, 2, 16, 2).unwrap();
        assert_eq!(rgba, [255, 0, 0, 255].repeat(32));
    }

    #[test]
    fn rejects_truncated_uploads() {
        assert!(decode(T::Bc3RgbaUnorm, &[0; 8], 4, 4, 1, 16, 1).is_err());
        assert!(decode(T::Bc1RgbaUnorm, &[0; 32], 8, 4, 1, 8, 1).is_err());
    }

    #[test]
    fn hdr_keeps_half_float_alpha() {
        let (rgba, pitch) = decode(T::Bc6hRgbUfloat, &[0x13; 16], 4, 4, 1, 16, 1).unwrap();
        assert_eq!(pitch, 32);
        for p in rgba.chunks_exact(8) { assert_eq!(&p[6..], &0x3c00u16.to_le_bytes()); }
    }

    /// The previous implementation (per-block 128-byte scratch + row copies), kept verbatim as the oracle.
    fn decode_reference(format: T, data: &[u8], w: u32, h: u32, depth: u32, pitch: u32, rows: u32) -> Result<(Vec<u8>, u32)> {
        ensure!(format.is_compressed() && w > 0 && h > 0 && depth > 0, "invalid BC dimensions/format");
        let (w, h, depth) = (w as usize, h as usize, depth as usize);
        let block_bytes = format.block_copy_size(None).unwrap_or(16) as usize;
        let pixel_bytes = fallback_format(format).block_copy_size(None).unwrap_or(4) as usize;
        let bx = w.div_ceil(4);
        let by = h.div_ceil(4);
        let pitch = if pitch == 0 { bx * block_bytes } else { pitch as usize };
        let rows = (rows as usize).max(by);
        ensure!(pitch >= bx * block_bytes, "BC row pitch is too small");
        let layer_stride = pitch * rows;
        let required = layer_stride * (depth - 1) + pitch * (by - 1) + bx * block_bytes;
        ensure!(data.len() >= required, "truncated BC upload");
        let out_pitch = w * pixel_bytes;
        let mut output = vec![0u8; out_pitch * h * depth];
        for z in 0..depth {
            for y in 0..by {
                for x in 0..bx {
                    let at = z * layer_stride + y * pitch + x * block_bytes;
                    let block = &data[at..at + block_bytes];
                    let mut pixels = [0u8; 128];
                    match format {
                        T::Bc1RgbaUnorm | T::Bc1RgbaUnormSrgb => bcdec_rs::bc1(block, &mut pixels, 16),
                        T::Bc2RgbaUnorm | T::Bc2RgbaUnormSrgb => bcdec_rs::bc2(block, &mut pixels, 16),
                        T::Bc3RgbaUnorm | T::Bc3RgbaUnormSrgb => bcdec_rs::bc3(block, &mut pixels, 16),
                        T::Bc4RUnorm | T::Bc4RSnorm => bcdec_rs::bc4(block, &mut pixels, 4, format == T::Bc4RSnorm),
                        T::Bc5RgUnorm | T::Bc5RgSnorm => bcdec_rs::bc5(block, &mut pixels, 8, format == T::Bc5RgSnorm),
                        T::Bc6hRgbUfloat | T::Bc6hRgbFloat => {
                            let mut rgb = [0u16; 48];
                            bcdec_rs::bc6h_half(block, &mut rgb, 12, format == T::Bc6hRgbFloat);
                            for i in 0..16 {
                                for c in 0..3 { pixels[i * 8 + c * 2..i * 8 + c * 2 + 2].copy_from_slice(&rgb[i * 3 + c].to_le_bytes()); }
                                pixels[i * 8 + 6..i * 8 + 8].copy_from_slice(&0x3c00u16.to_le_bytes());
                            }
                        }
                        T::Bc7RgbaUnorm | T::Bc7RgbaUnormSrgb => bcdec_rs::bc7(block, &mut pixels, 16),
                        _ => anyhow::bail!("unsupported BC format {format:?}"),
                    }
                    for row in 0..4.min(h - y * 4) {
                        let n = 4.min(w - x * 4) * pixel_bytes;
                        let dst = (z * h + y * 4 + row) * out_pitch + x * 4 * pixel_bytes;
                        let src = row * 4 * pixel_bytes;
                        output[dst..dst + n].copy_from_slice(&pixels[src..src + n]);
                    }
                }
            }
        }
        Ok((output, out_pitch as u32))
    }

    fn noise(len: usize, seed: u64) -> Vec<u8> {
        let mut s = seed | 1;
        (0..len).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 24) as u8 }).collect()
    }

    const ALL: [T; 14] = [
        T::Bc1RgbaUnorm, T::Bc1RgbaUnormSrgb, T::Bc2RgbaUnorm, T::Bc2RgbaUnormSrgb, T::Bc3RgbaUnorm, T::Bc3RgbaUnormSrgb,
        T::Bc4RUnorm, T::Bc4RSnorm, T::Bc5RgUnorm, T::Bc5RgSnorm, T::Bc6hRgbUfloat, T::Bc6hRgbFloat,
        T::Bc7RgbaUnorm, T::Bc7RgbaUnormSrgb,
    ];

    #[test]
    fn matches_reference_for_every_format_size_pitch_and_layer_count() {
        let sizes = [(1, 1), (3, 2), (4, 4), (5, 7), (9, 13), (16, 16), (17, 33), (64, 5)];
        for (fi, &format) in ALL.iter().enumerate() {
            let bb = format.block_copy_size(None).unwrap() as usize;
            for &(w, h) in &sizes {
                for depth in [1u32, 3] {
                    let (bx, by) = ((w as usize).div_ceil(4), (h as usize).div_ceil(4));
                    // tight, padded pitch, and padded rows-per-layer
                    for (pad_pitch, pad_rows) in [(0usize, 0usize), (bb, 0), (0, 2), (3 * bb, 5)] {
                        let (pitch, rows) = (bx * bb + pad_pitch, by + pad_rows);
                        let data = noise(pitch * rows * depth as usize, 0x9e37 + fi as u64 * 131 + w as u64 * 7 + h as u64);
                        let want = decode_reference(format, &data, w, h, depth, pitch as u32, rows as u32).unwrap();
                        for (threads, min_px) in [(1, usize::MAX), (4, 0), (3, 0)] {
                            let got = decode_impl(format, &data, w, h, depth, pitch as u32, rows as u32, threads, min_px).unwrap();
                            assert_eq!(got, want, "{format:?} {w}x{h}x{depth} pitch={pitch} rows={rows} threads={threads}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn tight_packing_when_pitch_and_rows_are_zero() {
        for &format in &ALL {
            let bb = format.block_copy_size(None).unwrap() as usize;
            let data = noise(bb * 3 * 2 * 2, 42);
            let want = decode_reference(format, &data, 10, 6, 2, 0, 0).unwrap();
            assert_eq!(decode(format, &data, 10, 6, 2, 0, 0).unwrap(), want, "{format:?}");
        }
    }

    #[test]
    fn large_parallel_decode_matches_single_thread() {
        let (w, h) = (515u32, 389u32);
        let bb = 16usize;
        let data = noise(w.div_ceil(4) as usize * h.div_ceil(4) as usize * bb, 7);
        let one = decode_impl(T::Bc7RgbaUnorm, &data, w, h, 1, 0, 0, 1, usize::MAX).unwrap();
        let many = decode_impl(T::Bc7RgbaUnorm, &data, w, h, 1, 0, 0, 4, 0).unwrap();
        assert_eq!(one, many);
        assert_eq!(one.0, decode_reference(T::Bc7RgbaUnorm, &data, w, h, 1, 0, 0).unwrap().0);
    }

    #[test]
    fn decode_pitch_is_the_decoded_row_stride() {
        assert_eq!(decode(T::Bc4RUnorm, &[0; 8], 4, 4, 1, 0, 0).unwrap().1, 4);
        assert_eq!(decode(T::Bc5RgUnorm, &[0; 16], 4, 4, 1, 0, 0).unwrap().1, 8);
        assert_eq!(decode(T::Bc7RgbaUnorm, &[0; 16], 4, 4, 1, 0, 0).unwrap().1, 16);
        assert_eq!(decode(T::Bc6hRgbFloat, &[0; 32], 5, 4, 1, 0, 0).unwrap().1, 40);
    }

    #[test]
    fn rejects_bad_geometry_without_panicking() {
        assert!(decode(T::Bc7RgbaUnorm, &[], u32::MAX, u32::MAX, 1, 0, 0).is_err());
        assert!(decode(T::Bc7RgbaUnorm, &[], 4, 4, u32::MAX, 0, 0).is_err());
        assert!(decode(T::Bc7RgbaUnorm, &[0; 16], 8, 4, 1, 8, 1).is_err(), "pitch smaller than a block row");
        assert!(decode(T::Bc7RgbaUnorm, &[0; 16], 0, 4, 1, 0, 0).is_err());
        assert!(decode(T::Bc7RgbaUnorm, &[0; 16], 4, 4, 0, 0, 0).is_err());
        assert!(decode(T::Rgba8Unorm, &[0; 16], 4, 4, 1, 0, 0).is_err(), "uncompressed format");
        assert!(decode(T::Etc2Rgb8Unorm, &[0; 8], 4, 4, 1, 0, 0).is_err(), "compressed but not BC");
    }

    #[test]
    fn a8_expansion_places_alpha_and_zero_fills_missing_bytes() {
        // 3x2 texels, 4 bytes per source row (1 byte padding)
        let out = expand_a8(&[1, 2, 3, 99, 4, 5, 6, 99], 3, 2, 4);
        let want: Vec<u8> = [1u8, 2, 3, 4, 5, 6].iter().flat_map(|&a| [0, 0, 0, a]).collect();
        assert_eq!(out, want);
        // truncated source: absent texels are 0
        let out = expand_a8(&[7, 8, 9, 0, 1], 3, 2, 4);
        assert_eq!(out.chunks_exact(4).map(|p| p[3]).collect::<Vec<_>>(), [7, 8, 9, 1, 0, 0]);
        assert_eq!(expand_a8(&[], 2, 2, 2), vec![0; 16]);
        assert!(expand_a8(&[1], 0, 2, 2).is_empty());
    }
}
