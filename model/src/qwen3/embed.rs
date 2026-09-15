//! Token rows → embeddings: bucket by length, run the prepared plan.
//!
//! Tokenization is the caller's: rows are the ids the reference tokenizer
//! produces, each ending in its end-of-text token. Rows are sorted by length
//! and packed `max_batch` at a time into the smallest length bucket that holds
//! them (multiples of the flash-attention tile, up to `max_len`); each bucket
//! compiles once, on first use.

use std::collections::BTreeMap;
use std::time::Instant;

use svod_runtime::{RunProfile, StageProfile};
use svod_tensor::PrepareConfig;

use crate::jit::InputSpec;

use super::embedder::Qwen3Embedding;
use super::error::Result;
use super::jit::Qwen3EmbeddingJit;

pub struct Qwen3Embedder {
    model: Qwen3Embedding,
    max_batch: usize,
    max_len: usize,
    pad_id: i32,
    plans: BTreeMap<usize, Qwen3EmbeddingJit>,
}

impl Qwen3Embedder {
    /// `max_len` is rounded up to a whole flash-attention tile; longer rows
    /// are truncated to it.
    pub fn new(model: Qwen3Embedding, max_batch: usize, max_len: usize) -> Self {
        let max_len = max_len.max(1).next_multiple_of(svod_tk::FLASH_ATTENTION_SEQUENCE_MULTIPLE);
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
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| rows[i].len());
        let mut out = vec![Vec::new(); rows.len()];
        for batch in order.chunks(self.max_batch) {
            let longest = batch.iter().map(|&i| rows[i].len()).max().unwrap_or(1);
            let bucket = longest.max(1).next_multiple_of(svod_tk::FLASH_ATTENTION_SEQUENCE_MULTIPLE);
            let (max_batch, pad_id) = (self.max_batch, self.pad_id);
            let jit = match self.plans.entry(bucket) {
                std::collections::btree_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::btree_map::Entry::Vacant(e) => {
                    let mut jit = Qwen3EmbeddingJit::new(self.model.clone());
                    jit.prepare_with_config(
                        InputSpec::i32(&[max_batch, bucket]),
                        InputSpec::i32(&[max_batch]),
                        &PrepareConfig::from_env(),
                    )?;
                    e.insert(jit)
                }
            };
            let mut ids = jit.input_ids_view_mut::<i32>()?;
            ids.fill(pad_id);
            for (r, &i) in batch.iter().enumerate() {
                for (j, &id) in rows[i].iter().enumerate() {
                    ids[[r, j]] = id as i32;
                }
            }
            let mut lengths = jit.lengths_view_mut::<i32>()?;
            lengths.fill(1);
            for (r, &i) in batch.iter().enumerate() {
                lengths[[r]] = rows[i].len().max(1) as i32;
            }
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
            let dim = flat.len() / max_batch;
            for (r, &i) in batch.iter().enumerate() {
                out[i] = flat[r * dim..(r + 1) * dim].to_vec();
            }
        }
        Ok(out)
    }
}
