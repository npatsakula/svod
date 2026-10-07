//! Text decoder: token + positional embeddings + self/cross-attention transformer blocks.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Layer, LayerNorm, Linear, Module};

use crate::init::{Bias, fan_in_uniform, layer_norm, linear};
use crate::state::{scope_index, scoped, scoped_index};

use super::attention::MultiHeadAttention;
use super::blocks::linear_forward;
use super::config::ModelDimensions;
use super::error::Result;
use crate::attention::tk_launch_error;

#[derive(Clone, Copy)]
struct StepAttentionConfig {
    custom_self: bool,
    custom_cross: bool,
    cross_splits: Option<usize>,
}

impl Default for StepAttentionConfig {
    fn default() -> Self {
        Self { custom_self: true, custom_cross: true, cross_splits: None }
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum StepAttentionMode {
    Generic,
    CustomSelf,
    CustomCross { split: usize },
    CustomBoth { split: usize },
}

#[cfg(test)]
impl From<StepAttentionMode> for StepAttentionConfig {
    fn from(mode: StepAttentionMode) -> Self {
        match mode {
            StepAttentionMode::Generic => Self { custom_self: false, custom_cross: false, cross_splits: None },
            StepAttentionMode::CustomSelf => Self { custom_self: true, custom_cross: false, cross_splits: None },
            StepAttentionMode::CustomCross { split } => {
                Self { custom_self: false, custom_cross: true, cross_splits: Some(split) }
            }
            StepAttentionMode::CustomBoth { split } => {
                Self { custom_self: true, custom_cross: true, cross_splits: Some(split) }
            }
        }
    }
}

/// Validity mask `[B, key_count]` for a cached decoder step: the cached prefix
/// each lane actually filled, plus the key this step just appended at the end
/// of the cache. `true` = attend, the polarity SDPA's `key_padding_mask` wants.
pub(crate) fn cached_step_mask(key_lens: &Tensor, key_count: usize) -> Result<Tensor> {
    let appended = Tensor::arange(key_count as i64, None, None)?.try_eq(key_count as i64 - 1)?;
    Ok(Tensor::sequence_mask(key_lens, key_count)?.try_bitor(&appended)?)
}

/// Decoder transformer block: self-attn + cross-attn + MLP, all pre-norm.
#[derive(Clone, Module)]
pub struct DecoderBlock {
    pub attn: MultiHeadAttention,
    pub attn_ln: LayerNorm,
    pub cross_attn: MultiHeadAttention,
    pub cross_attn_ln: LayerNorm,
    #[module(key = "mlp.0")]
    pub mlp0: Linear,
    #[module(key = "mlp.2")]
    pub mlp2: Linear,
    pub mlp_ln: LayerNorm,
    pub n_state: usize,
}

impl DecoderBlock {
    pub fn empty(n_state: usize, n_head: usize) -> Self {
        Self::empty_dtype(n_state, n_head, DType::Float32)
    }

    pub fn empty_dtype(n_state: usize, n_head: usize, dtype: DType) -> Self {
        let mlp = n_state * 4;
        Self {
            attn: MultiHeadAttention::empty_dtype(n_state, n_head, dtype.clone()),
            attn_ln: layer_norm(n_state, dtype.clone()),
            cross_attn: MultiHeadAttention::empty_dtype(n_state, n_head, dtype.clone()),
            cross_attn_ln: layer_norm(n_state, dtype.clone()),
            mlp0: linear(n_state, mlp, Bias::FanIn, dtype.clone()),
            mlp2: linear(mlp, n_state, Bias::FanIn, dtype.clone()),
            mlp_ln: layer_norm(n_state, dtype),
            n_state,
        }
    }

    /// Forward with SDPA over raw encoder features `xa`.
    pub fn forward(&self, x: &Tensor, xa: &Tensor, mask: &Tensor) -> Result<Tensor> {
        self.residual(x, |h| self.attn.forward(h, None, Some(mask)), |h| self.cross_attn.forward(h, Some(xa), None))
    }

    /// The pre-norm residual skeleton every decoder entry point shares. Each
    /// closure receives its normalized input and returns the projected sublayer
    /// output; how attention is computed is the caller's business.
    fn residual(
        &self,
        x: &Tensor,
        self_attn: impl FnOnce(&Tensor) -> Result<Tensor>,
        cross_attn: impl FnOnce(&Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let h = scoped("attn_ln", || self.attn_ln.forward(x))?;
        let x = x.try_add(&scoped("attn", || self_attn(&h))?)?;
        let h = scoped("cross_attn_ln", || self.cross_attn_ln.forward(&x))?;
        let x = x.try_add(&scoped("cross_attn", || cross_attn(&h))?)?;
        let h = scoped("mlp_ln", || self.mlp_ln.forward(&x))?;
        Ok(x.try_add(&self.mlp(&h)?)?)
    }

    fn mlp(&self, h: &Tensor) -> Result<Tensor> {
        linear_forward(&self.mlp2, &linear_forward(&self.mlp0, h)?.gelu_exact()?)
    }

    /// SDPA cross-attention over one layer's head-major `[B, H, T, Dh]` cache
    /// slice. Also returns the head-split query, which the aligner scores.
    fn cross_sdpa(&self, h: &Tensor, layer_ck: &Tensor, layer_cv: &Tensor, n_head: usize) -> Result<(Tensor, Tensor)> {
        let query = linear_forward(&self.cross_attn.query, h)?.split_heads(n_head)?;
        let out = query.scaled_dot_product_attention().key(layer_ck).value(layer_cv).is_causal(false).call()?;
        Ok((linear_forward(&self.cross_attn.out, &out.merge_heads()?)?, query))
    }
}

/// Whisper text decoder: token embedding + learned positional embedding +
/// N × DecoderBlock + LayerNorm + tied output projection.
#[derive(Clone, Module)]
pub struct TextDecoder {
    #[module(key = "token_embedding.weight")]
    pub token_embedding: Tensor, // [n_vocab, D]
    pub positional_embedding: Tensor, // [n_text_ctx, D]
    pub blocks: Vec<DecoderBlock>,
    pub ln: LayerNorm,
    pub n_state: usize,
    pub n_head: usize,
    pub n_text_ctx: usize,
    #[module(skip)]
    activation_dtype: DType,
    #[module(skip)]
    cache_dtype: DType,
}

impl TextDecoder {
    pub fn empty(dims: &ModelDimensions) -> Self {
        let n_state = dims.n_text_state;
        let dtype = dims.dtype.clone();
        Self {
            token_embedding: fan_in_uniform(&[dims.n_vocab, n_state], n_state, dtype.clone()),
            positional_embedding: Tensor::zeros(&[dims.n_text_ctx, n_state], dtype.clone()),
            blocks: (0..dims.n_text_layer)
                .map(|_| DecoderBlock::empty_dtype(n_state, dims.n_text_head, dtype.clone()))
                .collect(),
            ln: layer_norm(n_state, dtype.clone()),
            n_state,
            n_head: dims.n_text_head,
            n_text_ctx: dims.n_text_ctx,
            activation_dtype: dtype,
            cache_dtype: dims.cache_dtype(),
        }
    }

    fn d_head(&self) -> usize {
        self.n_state / self.n_head
    }

    /// Token plus positional embedding in the activation dtype.
    fn embed(&self, tokens: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        Ok(self.token_embedding.embedding(tokens)?.try_add(pos_emb)?.cast(self.activation_dtype.clone()))
    }

    /// Final norm and the tied output projection, always read back as f32.
    fn logits(&self, x: &Tensor) -> Result<Tensor> {
        let x = scoped("ln", || self.ln.forward(x))?;
        Ok(x.linear().weight(&self.token_embedding.cast(x.dtype())).call()?.cast(DType::Float32))
    }

    /// Pack per-layer `[B, S, n_state]` projections into the `[B, S, n_layer*H, Dh]`
    /// cache layout. A reshape reads each projection contiguously; splitting
    /// heads first would stride through it.
    fn pack_kv(&self, kvs: &[Tensor]) -> Result<Tensor> {
        let heads = kvs
            .iter()
            .map(|kv| Ok(kv.try_reshape([kv.dim_const(0)?, kv.dim_const(1)?, self.n_head, self.d_head()])?))
            .collect::<Result<Vec<_>>>()?;
        Ok(Tensor::cat(&heads.iter().collect::<Vec<_>>(), 2)?.cast(self.cache_dtype.clone()))
    }

    /// One layer's head-major `[B, H, T, Dh]` slices of the packed cross caches.
    fn layer_cross_kv(&self, cross_k: &Tensor, cross_v: &Tensor, layer: usize) -> Result<(Tensor, Tensor)> {
        let slice = |cache: &Tensor| -> Result<Tensor> {
            Ok(cache.narrow(2, layer * self.n_head, self.n_head)?.try_permute(&[0, 2, 1, 3])?)
        };
        Ok((slice(cross_k)?, slice(cross_v)?))
    }

    /// Project encoder features into the packed cross-attention caches.
    pub fn project_cross_kv(&self, xa: &Tensor) -> Result<(Tensor, Tensor)> {
        let xa = xa.cast(self.activation_dtype.clone());
        let mut cross_ks = Vec::with_capacity(self.blocks.len());
        let mut cross_vs = Vec::with_capacity(self.blocks.len());
        for (index, block) in self.blocks.iter().enumerate() {
            let _origin = scope_index("blocks", index);
            // Keep each GEMM independent from the final layer/head packing.
            let k = scoped("cross_attn", || scoped("key", || linear_forward(&block.cross_attn.key, &xa)))?.contiguous();
            let v =
                scoped("cross_attn", || scoped("value", || linear_forward(&block.cross_attn.value, &xa)))?.contiguous();
            cross_ks.push(k);
            cross_vs.push(v);
        }
        Ok((self.pack_kv(&cross_ks)?, self.pack_kv(&cross_vs)?))
    }

    /// Forward pass producing logits for all positions.
    /// `tokens`: `[B, L]` int tensor. `xa`: `[B, T_enc, D]` encoder output.
    /// `offset`: positional embedding offset (for KV-cached incremental decoding).
    pub fn forward(&self, tokens: &Tensor, xa: &Tensor, offset: usize) -> Result<Tensor> {
        let seq_len = tokens.dim_const(1)?;
        let mut x = self.embed(tokens, &self.positional_embedding.narrow(0, offset, seq_len)?)?;
        let xa = xa.cast(self.activation_dtype.clone());
        let mask = Tensor::causal_mask(seq_len, x.dtype())?;
        for (index, block) in self.blocks.iter().enumerate() {
            x = scoped_index("blocks", index, || block.forward(&x, &xa, &mask))?;
        }
        self.logits(&x)
    }

    /// Teacher-forced decoder pass over packed cross K/V, returning raw scaled
    /// QK scores for the statically selected alignment heads.
    pub fn forward_alignment(
        &self,
        tokens: &Tensor,
        cross_k: &Tensor,
        cross_v: &Tensor,
        alignment_heads: &[(usize, usize)],
    ) -> Result<Tensor> {
        let seq_len = tokens.dim_const(1)?;
        let mut x = self.embed(tokens, &self.positional_embedding.narrow(0, 0usize, seq_len)?)?;
        let cross_k = cross_k.cast(self.activation_dtype.clone());
        let cross_v = cross_v.cast(self.activation_dtype.clone());
        let mask = Tensor::causal_mask(seq_len, x.dtype())?;
        let scale = (self.d_head() as f64).sqrt().recip();

        let mut selected_qk: Vec<Option<Tensor>> = (0..alignment_heads.len()).map(|_| None).collect();
        for (layer, block) in self.blocks.iter().enumerate() {
            let _origin = scope_index("blocks", layer);
            let (layer_ck, layer_cv) = self.layer_cross_kv(&cross_k, &cross_v, layer)?;
            let mut query = None;
            x = block.residual(
                &x,
                |h| block.attn.forward(h, None, Some(&mask)),
                |h| {
                    let (out, q) = block.cross_sdpa(h, &layer_ck, &layer_cv, self.n_head)?;
                    query = Some(q);
                    Ok(out)
                },
            )?;
            let query = query.expect("cross-attention ran");
            for (selected, &(_, head)) in alignment_heads.iter().enumerate().filter(|&(_, &(l, _))| l == layer) {
                let keys = layer_ck.narrow(1, head, 1usize)?.try_transpose(-1, -2)?;
                selected_qk[selected] = Some(query.narrow(1, head, 1usize)?.matmul(&keys)?.try_mul(scale)?);
            }
        }
        let selected_qk = selected_qk
            .into_iter()
            .map(|qk| {
                qk.ok_or_else(|| super::error::Error::Tensor {
                    source: Box::new(
                        svod_tensor::error::ErrorKind::SymbolicShapeUnsupported {
                            operation: "alignment head layer out of range".into(),
                        }
                        .into(),
                    ),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Tensor::cat(&selected_qk.iter().collect::<Vec<_>>(), 1)?.cast(DType::Float32))
    }

    /// Prefill from encoder features: the cross projection runs once per window
    /// and its caches come back alongside the self caches it seeded.
    /// Returns `(logits[B, L, n_vocab], self_k, self_v, cross_k, cross_v)`, the
    /// caches packed `[B, len, n_layer*H, Dh]`.
    pub fn forward_prefill(
        &self,
        tokens: &Tensor,
        audio_features: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let seq_len = tokens.dim_const(1)?;
        let (cross_k, cross_v) = self.project_cross_kv(audio_features)?;
        let ck = cross_k.cast(self.activation_dtype.clone());
        let cv = cross_v.cast(self.activation_dtype.clone());
        let mut x = self.embed(tokens, &self.positional_embedding.narrow(0, offset, seq_len)?)?;
        let mask = Tensor::causal_mask(seq_len, x.dtype())?;

        let mut self_ks = Vec::with_capacity(self.blocks.len());
        let mut self_vs = Vec::with_capacity(self.blocks.len());
        for (layer, block) in self.blocks.iter().enumerate() {
            let _origin = scope_index("blocks", layer);
            let (layer_ck, layer_cv) = self.layer_cross_kv(&ck, &cv, layer)?;
            let mut kv = None;
            x = block.residual(
                &x,
                |h| {
                    let (out, k, v) = block.attn.forward_return_kv(h, None, Some(&mask))?;
                    kv = Some((k, v));
                    Ok(out)
                },
                |h| block.cross_sdpa(h, &layer_ck, &layer_cv, self.n_head).map(|(out, _)| out),
            )?;
            let (k, v) = kv.expect("self-attention ran");
            self_ks.push(k);
            self_vs.push(v);
        }
        Ok((self.logits(&x)?, self.pack_kv(&self_ks)?, self.pack_kv(&self_vs)?, cross_k, cross_v))
    }

    /// Single-token forward with KV cache. Used for incremental decoding.
    /// Works for any batch size B (B=1 for greedy, B=beam_size for beam search).
    ///
    /// - `token`: [B, 1] int32
    /// - `self_k_cache`: [B, max_len, n_layer*H, Dh] self-attn K cache
    /// - `self_v_cache`: [B, max_len, n_layer*H, Dh] self-attn V cache
    /// - `cross_k`: [K, n_audio_ctx, n_layer*H, Dh] cross-attn K (fixed)
    /// - `cross_v`: [K, n_audio_ctx, n_layer*H, Dh] cross-attn V (fixed)
    /// - `self_key_lens`: `[B]` i32 valid cached-key counts, which is also each
    ///   row's position and therefore selects its positional embedding
    /// - `cross_cache_map`: `[B]` i32 cross-cache row each lane reads
    ///
    /// Returns `(logits[B, n_vocab], new_self_k[B, 1, n_layer*H, Dh], new_self_v[...])`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_step(
        &self,
        token: &Tensor,
        self_k_cache: &Tensor,
        self_v_cache: &Tensor,
        cross_k: &Tensor,
        cross_v: &Tensor,
        self_key_lens: &Tensor,
        cross_cache_map: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        self.forward_step_with_config(
            token,
            self_k_cache,
            self_v_cache,
            cross_k,
            cross_v,
            self_key_lens,
            cross_cache_map,
            StepAttentionConfig::default(),
        )
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_step_with_attention_mode(
        &self,
        token: &Tensor,
        self_k_cache: &Tensor,
        self_v_cache: &Tensor,
        cross_k: &Tensor,
        cross_v: &Tensor,
        self_key_lens: &Tensor,
        cross_cache_map: &Tensor,
        mode: StepAttentionMode,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        self.forward_step_with_config(
            token,
            self_k_cache,
            self_v_cache,
            cross_k,
            cross_v,
            self_key_lens,
            cross_cache_map,
            mode.into(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_step_with_config(
        &self,
        token: &Tensor,
        self_k_cache: &Tensor,
        self_v_cache: &Tensor,
        cross_k: &Tensor,
        cross_v: &Tensor,
        self_key_lens: &Tensor,
        cross_cache_map: &Tensor,
        attention: StepAttentionConfig,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (n_head, d_head) = (self.n_head, self.d_head());
        // The step appends into and reads the caches as raw bytes of one type,
        // so a cache stored otherwise is a caller bug worth naming here rather
        // than deep inside a concatenation.
        for (name, cache) in
            [("self_k", self_k_cache), ("self_v", self_v_cache), ("cross_k", cross_k), ("cross_v", cross_v)]
        {
            if cache.dtype() != self.cache_dtype {
                let msg =
                    format!("{name} cache is {:?}, the decoder stores caches as {:?}", cache.dtype(), self.cache_dtype);
                return Err(super::error::Error::Decode { msg });
            }
        }
        let batch = token.dim_const(0)?;
        let self_key_count = self_k_cache.dim_const(1)? + 1;
        // The cross split is the device's unless the mode names one.
        let cross_splits = attention.cross_splits;
        let act = |t: Tensor| t.cast(self.activation_dtype.clone());
        let heads = |t: Tensor| -> Result<Tensor> { Ok(t.try_reshape([batch, 1, n_head, d_head])?) };

        let pos_emb = self.positional_embedding.embedding(self_key_lens)?.try_unsqueeze(1)?;
        let mut x = self.embed(token, &pos_emb)?;
        let mut new_ks = Vec::with_capacity(self.blocks.len());
        let mut new_vs = Vec::with_capacity(self.blocks.len());

        for (layer, block) in self.blocks.iter().enumerate() {
            let _origin = scope_index("blocks", layer);
            let lh_start = layer * n_head;
            x = block.residual(
                &x,
                |h| {
                    // Sequence-major `[B, 1, H, Dh]` projections feed the custom
                    // kernel directly and are already in the cache layout.
                    let q = heads(linear_forward(&block.attn.query, h)?)?;
                    let new_k = heads(linear_forward(&block.attn.key, h)?)?.cast(self.cache_dtype.clone());
                    let new_v = heads(linear_forward(&block.attn.value, h)?)?.cast(self.cache_dtype.clone());

                    // The kernel scores this layer's packed cache prefix and the
                    // row just projected, so nothing is spliced: the concatenation
                    // the generic path needs copies the whole slice every layer.
                    let direct = if attention.custom_self {
                        svod_tk::single_query_attention_packed(
                            &q.cast(DType::Float32),
                            self_k_cache,
                            self_v_cache,
                            lh_start,
                            svod_tk::SqAttentionOpts {
                                key_lens: Some(self_key_lens),
                                appended: Some((&new_k, &new_v)),
                                ..Default::default()
                            },
                        )
                        .map_err(tk_launch_error)?
                    } else {
                        None
                    };
                    let out = match direct {
                        Some(out) => act(out.try_reshape([batch, 1, self.n_state])?),
                        None => {
                            let full = |cache: &Tensor, new: &Tensor| -> Result<Tensor> {
                                let layer = cache.narrow(2, lh_start, n_head)?;
                                Ok(act(Tensor::cat(&[&layer, new], 1)?).try_permute(&[0, 2, 1, 3])?)
                            };
                            let valid = cached_step_mask(self_key_lens, self_key_count)?;
                            q.try_permute(&[0, 2, 1, 3])?
                                .scaled_dot_product_attention()
                                .key(&full(self_k_cache, &new_k)?)
                                .value(&full(self_v_cache, &new_v)?)
                                .key_padding_mask(&valid)
                                .is_causal(false)
                                .call()?
                                .merge_heads()?
                        }
                    };
                    new_ks.push(new_k);
                    new_vs.push(new_v);
                    linear_forward(&block.attn.out, &out)
                },
                |h| {
                    let q = heads(linear_forward(&block.cross_attn.query, h)?)?;
                    let direct = if attention.custom_cross {
                        svod_tk::single_query_attention_packed(
                            &q.cast(DType::Float32),
                            cross_k,
                            cross_v,
                            lh_start,
                            svod_tk::SqAttentionOpts {
                                split: cross_splits,
                                cache_map: Some(cross_cache_map),
                                ..Default::default()
                            },
                        )
                        .map_err(tk_launch_error)?
                    } else {
                        None
                    };
                    let out = match direct {
                        Some(out) => act(out.try_reshape([batch, 1, self.n_state])?),
                        None => {
                            // The cache holds one row per attempt, so a lane reads the
                            // row its attempt owns. The tile kernel does that with an
                            // index load; here it costs a gather, which is why the fast
                            // path exists.
                            let owned = |cache: &Tensor| -> Result<Tensor> {
                                let layer = cache.narrow(2, lh_start, n_head)?.index_select(0, cross_cache_map)?;
                                Ok(act(layer).try_permute(&[0, 2, 1, 3])?)
                            };
                            q.try_permute(&[0, 2, 1, 3])?
                                .scaled_dot_product_attention()
                                .key(&owned(cross_k)?)
                                .value(&owned(cross_v)?)
                                .is_causal(false)
                                .call()?
                                .merge_heads()?
                        }
                    };
                    linear_forward(&block.cross_attn.out, &out)
                },
            )?;
        }

        let n_vocab = self.token_embedding.dim_const(0)?;
        let logits = self.logits(&x)?.try_reshape([batch, n_vocab])?;
        let new_k = Tensor::cat(&new_ks.iter().collect::<Vec<_>>(), 2)?;
        let new_v = Tensor::cat(&new_vs.iter().collect::<Vec<_>>(), 2)?;
        Ok((logits, new_k, new_v))
    }
}
