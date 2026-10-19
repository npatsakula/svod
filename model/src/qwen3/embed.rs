//! Token rows → embeddings: pack rows into the prepared plan's shape and run it.
//!
//! Tokenization is the caller's: rows are the ids the reference tokenizer
//! produces, each ending in its end-of-text token. Rows are taken longest
//! first, `max_batch` plan rows at a time, into the smallest length bucket
//! (a multiple of [`SEQUENCE_TILE`], up to `max_len`) that holds the
//! longest of them; the shorter rows that follow fill the space left in each
//! plan row (first fit), so a batch of short texts costs what its tokens do,
//! not what `max_batch` full rows would. Each bucket compiles once, on first
//! use. The plan pools a fixed [`SLOT_TOKENS`]-per-sequence number of slots
//! per row, which bounds how many rows one plan row takes.

use std::collections::BTreeMap;
use std::time::Instant;

use svod_runtime::{RunProfile, StageProfile};
use svod_tensor::PrepareConfig;

use crate::jit::InputSpec;

use super::embedder::Qwen3Embedding;
use super::error::Result;
use super::jit::Qwen3EmbeddingJit;
use super::model::SEQUENCE_TILE;

/// Tokens of row capacity per pooled slot: a plan row of `bucket` tokens holds
/// at most `bucket / SLOT_TOKENS` sequences.
const SLOT_TOKENS: usize = 16;

/// One plan execution: the plan rows, each the input rows (indices into the
/// caller's set) packed into it in order.
struct Batch {
    bucket: usize,
    rows: Vec<Vec<usize>>,
}

/// Pack `lens` (every entry at least 1) into batches of `max_batch` plan rows:
/// longest first, each batch's bucket sized to its longest row, the rest
/// first-fit by tokens and by pooled slots.
fn pack(lens: &[usize], max_batch: usize) -> Vec<Batch> {
    let mut pending: Vec<usize> = (0..lens.len()).collect();
    pending.sort_by_key(|&i| std::cmp::Reverse(lens[i]));
    let mut batches = Vec::new();
    while let Some(&longest) = pending.first() {
        let bucket = lens[longest].next_multiple_of(SEQUENCE_TILE);
        let slots = bucket / SLOT_TOKENS;
        let (mut fill, mut rows) = (vec![0usize; max_batch], vec![Vec::new(); max_batch]);
        pending.retain(|&i| {
            let row = (0..max_batch).find(|&r| fill[r] + lens[i] <= bucket && rows[r].len() < slots);
            match row {
                Some(r) => {
                    fill[r] += lens[i];
                    rows[r].push(i);
                    false
                }
                None => true,
            }
        });
        batches.push(Batch { bucket, rows });
    }
    batches
}

pub struct Qwen3Embedder {
    model: Qwen3Embedding,
    max_batch: usize,
    max_len: usize,
    pad_id: i32,
    plans: BTreeMap<usize, Qwen3EmbeddingJit>,
}

impl Qwen3Embedder {
    /// `max_len` is rounded up to a whole [`SEQUENCE_TILE`]; longer rows
    /// are truncated to it.
    pub fn new(model: Qwen3Embedding, max_batch: usize, max_len: usize) -> Self {
        let max_len = max_len.max(1).next_multiple_of(SEQUENCE_TILE);
        let pad_id = model.model.config.pad_token_id as i32;
        Self { model, max_batch: max_batch.max(1), max_len, pad_id, plans: BTreeMap::new() }
    }

    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// One unit-norm embedding per row, in input order.
    pub fn embed<R: AsRef<[u32]>>(&mut self, rows: &[R]) -> Result<Vec<Vec<f32>>> {
        self.run(rows, None)
    }

    /// [`embed`](Self::embed) with one profiled stage per batch.
    pub fn embed_profiled<R: AsRef<[u32]>>(&mut self, rows: &[R]) -> Result<(Vec<Vec<f32>>, RunProfile)> {
        let mut profile = RunProfile::default();
        let embeddings = self.run(rows, Some(&mut profile))?;
        Ok((embeddings, profile))
    }

    fn run<R: AsRef<[u32]>>(&mut self, rows: &[R], mut profile: Option<&mut RunProfile>) -> Result<Vec<Vec<f32>>> {
        let rows: Vec<&[u32]> = rows.iter().map(|r| &r.as_ref()[..r.as_ref().len().min(self.max_len)]).collect();
        // An empty row embeds as one pad token.
        let lens: Vec<usize> = rows.iter().map(|r| r.len().max(1)).collect();
        let mut out = vec![Vec::new(); rows.len()];
        for batch in pack(&lens, self.max_batch) {
            let (max_batch, pad_id, bucket) = (self.max_batch, self.pad_id, batch.bucket);
            let slots = bucket / SLOT_TOKENS;
            let jit = match self.plans.entry(bucket) {
                std::collections::btree_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::btree_map::Entry::Vacant(e) => {
                    let mut jit = Qwen3EmbeddingJit::new(self.model.clone());
                    jit.prepare_with_config(
                        InputSpec::i32(&[max_batch, bucket]),
                        InputSpec::i32(&[max_batch, bucket]),
                        InputSpec::i32(&[max_batch, bucket]),
                        InputSpec::i32(&[max_batch, slots]),
                        &PrepareConfig::device_local(),
                    )?;
                    e.insert(jit)
                }
            };
            // Padding: pad tokens at position 0, each its own segment (a token
            // always sees itself), pooled nowhere.
            let mut ids = vec![pad_id; max_batch * bucket];
            let mut positions = vec![0i32; max_batch * bucket];
            let mut seg_start: Vec<i32> = (0..max_batch * bucket).map(|j| (j % bucket) as i32).collect();
            let mut pool_idx = vec![0i32; max_batch * slots];
            for (r, packed) in batch.rows.iter().enumerate() {
                let mut cursor = r * bucket;
                for (slot, &i) in packed.iter().enumerate() {
                    for (j, &id) in rows[i].iter().enumerate() {
                        ids[cursor + j] = id as i32;
                        positions[cursor + j] = j as i32;
                    }
                    seg_start[cursor..cursor + lens[i]].fill((cursor % bucket) as i32);
                    cursor += lens[i];
                    pool_idx[r * slots + slot] = ((cursor - 1) % bucket) as i32;
                }
            }
            let fill = |mut view: ndarray::ArrayViewMutD<'_, i32>, data: &[i32]| {
                view.iter_mut().zip(data).for_each(|(d, &s)| *d = s)
            };
            fill(jit.input_ids_view_mut::<i32>()?, &ids);
            fill(jit.positions_view_mut::<i32>()?, &positions);
            fill(jit.seg_start_view_mut::<i32>()?, &seg_start);
            fill(jit.pool_idx_view_mut::<i32>()?, &pool_idx);
            let started = Instant::now();
            match profile.as_deref_mut() {
                Some(profile) => {
                    let kernels = jit.execute_profiled()?;
                    let name = format!("embed[{max_batch}x{bucket}]");
                    profile.stages.push(StageProfile::gpu(name, started.elapsed(), kernels));
                }
                None => jit.execute()?,
            }
            let flat = jit.embeddings_to_vec::<f32>()?;
            let dim = flat.len() / (max_batch * slots);
            for (r, packed) in batch.rows.iter().enumerate() {
                for (slot, &i) in packed.iter().enumerate() {
                    let at = (r * slots + slot) * dim;
                    out[i] = flat[at..at + dim].to_vec();
                }
            }
        }
        Ok(out)
    }
}
