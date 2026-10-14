//! Shared input packing for the ModernBERT heads: pad each chunk's
//! `input_ids` / `attention_mask` into a plan's row-major `[rows, len]` i64
//! inputs. Generic JIT-input packing, independent of any particular head.

use svod_arch::pipelines::text::Encoding;

/// Pad each chunk's `input_ids` into the `[rows, len]` input, zero-filling past
/// each chunk's length and over unused rows.
pub(crate) fn pack_ids(dst: &mut [i64], batch: &[&Encoding], len: usize) {
    pack(dst, batch.iter().map(|enc| enc.input_ids.as_slice()), len);
}

/// Pad each chunk's `attention_mask` (1 = real token, 0 = pad) into the
/// `[rows, len]` input. Trailing pad positions stay 0. An unused row attends to
/// its first (pad) token alone: a row whose keys are all masked would run its
/// softmax over nothing.
pub(crate) fn pack_mask(dst: &mut [i64], batch: &[&Encoding], len: usize) {
    pack(dst, batch.iter().map(|enc| enc.attention_mask.as_slice()), len);
    for row in dst.chunks_exact_mut(len).skip(batch.len()) {
        row[0] = 1;
    }
}

fn pack<'a>(dst: &mut [i64], rows: impl Iterator<Item = &'a [u32]>, len: usize) {
    dst.fill(0);
    for (row, src) in dst.chunks_exact_mut(len).zip(rows) {
        let take = src.len().min(len);
        for (d, &s) in row[..take].iter_mut().zip(&src[..take]) {
            *d = s.into();
        }
    }
}
