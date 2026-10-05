//! Streaming GTCRN — frame-by-frame causal speech enhancement.
//!
//! A pure-Rust port of the upstream `StreamGTCRN`
//! (`submodules/gtcrn/stream/gtcrn_stream.py`). Processes one STFT frame
//! (`T = 1`) per call, threading recurrent state (conv caches, GRU hidden
//! states) between calls. Because the time dimension is pinned to 1 at
//! JIT-prepare time, the GRU recurrence unrolls exactly one IR node per call —
//! no graph explosion regardless of total audio length.
//!
//! ## The same network, with causality spelled differently
//!
//! [`GtcrnStream`] is [`Gtcrn`](super::Gtcrn) with two edits, both confined to
//! the six depthwise convs:
//!
//! - the offline block folds a causal `(2d, 0)` T pad into the conv's own
//!   padding; the stream block drops that pad and concatenates a cache of
//!   exactly `(kT - 1)·d` frames in front of the frame instead;
//! - the decoder's depthwise `ConvTranspose2d` becomes a regular `Conv2d` over
//!   the spatially flipped kernel — `StreamConvTranspose2d` Version 2. GTCRN's
//!   depthwise case has `in/groups == out/groups == 1`, so both layouts are
//!   `[16, 1, 3, 3]` and `flip(&[-2, -1])` alone is the whole transform (the
//!   shape-equal branch of `convert.py:27-28`), with no channel permute.
//!
//! Everything else — the checkpoint, ERB, SFE, the pointwise convs, the
//! DPGRNNs — is the offline model's, shared struct for struct. The offline
//! model is itself strictly causal, so streaming and offline agree frame for
//! frame on the same window.
//!
//! ## State
//!
//! Three families of caches, all zero at a cold start and recycled
//! **on-device** by the JIT's `state { .. }` slots — the host never copies
//! recurrent state:
//!
//! | cache | count | per-call shape | what it holds |
//! |-------|-------|----------------|---------------|
//! | conv cache | 6 (3 enc + 3 dec) | `(1, 16, 2·d, 33)` | T-history for one depthwise conv; `d` = dilation |
//! | tra h-state | 6 | `(1, 16)` | GRU hidden state for one TRA attention block |
//! | inter h-state | 2 | `(33, 16)` | GRU hidden state for one DPGRNN's `inter_rnn` |

use std::path::Path;

use snafu::ResultExt;
use svod_macros::jit_wrapper;
use svod_tensor::Tensor;
use svod_tensor::nn::{Conv2d, GruDirection, Layer, Module, RnnLayout, StateDict, prefixed};

use crate::jit::InputSpec;
use crate::state;

use super::blocks::{Conv, Dpgrnn, Erb, GTConvBlock, Grnn, GruWeights, Tra, sfe};
use super::error::{HubSnafu, Result};
use super::{
    C_IN, C_NET, CHECKPOINT, DPGRNN_HIDDEN, DPGRNN_WIDTH, DecoderLayer, EncoderLayer, HUB_REPO, N_FREQ,
    default_decoder, default_encoder, empty_dpgrnn,
};

/// Encoder GTConvBlock dilations; the decoder reverses them.
const DILATIONS: [usize; 3] = [1, 2, 5];

/// The three decoder blocks whose depthwise conv is stored flipped.
const FLIPPED: usize = DILATIONS.len();

/// History frames the block at `forward_stream` position `i` caches:
/// `(kT - 1) · dilation = 2 · dilation`, encoder blocks first.
const fn hist(i: usize) -> usize {
    2 * if i < FLIPPED { DILATIONS[i] } else { DILATIONS[2 * FLIPPED - 1 - i] }
}

/// TRA attention GRU hidden size (`2 · in_channels/2`).
const TRA_HIDDEN: usize = C_NET;

/// One family of per-block caches: the three encoder GT blocks, then the three
/// decoder ones.
type Caches = [Tensor; 2 * FLIPPED];
/// The same family on the way in, borrowed from the JIT's state buffers.
type CacheRefs<'a> = [&'a Tensor; 2 * FLIPPED];
/// What one streaming step produces: the enhanced frame plus every cache, in
/// the order [`GtcrnStream::forward_stream`] takes them.
type Step = (Tensor, Caches, Caches, [Tensor; 2]);

// =========================================================================== //
// GtcrnStream — the streaming model
// =========================================================================== //

/// Runtime tuning for the complex-ratio-mask post-processing. Both knobs bake
/// into the JIT graph at `prepare` time.
///
/// The trained model applies `enhanced = spec * mask` (a complex ratio mask).
/// With `scale` and `blend` the applied transform becomes:
/// ```text
/// crm      = spec * (mask * scale)             // complex multiply
/// enhanced = blend * crm + (1 - blend) * spec  // dry/wet mix
/// ```
///
/// - `scale = 1.0` (default): the mask is applied as-is — exact parity with
///   the trained reference. `scale > 1` amplifies the mask (stronger
///   suppression); `scale < 1` dampens it (closer to the input).
/// - `blend = 1.0` (default): fully enhanced output. `blend = 0` is a pure
///   passthrough; intermediate values mix the enhanced and raw spectrograms,
///   which avoids the over-suppression "musical noise" artifacts an aggressive
///   mask can introduce.
#[derive(Clone, Copy, Debug)]
pub struct MaskConfig {
    pub scale: f32,
    pub blend: f32,
}

impl Default for MaskConfig {
    fn default() -> Self {
        Self { scale: 1.0, blend: 1.0 }
    }
}

/// The streaming GTCRN network — same topology and same `gtcrn.safetensors` as
/// the offline [`Gtcrn`](super::Gtcrn). Construct via
/// [`GtcrnStream::from_hub`] / [`GtcrnStream::from_safetensors`], drive through
/// [`GtcrnStreamJit`].
#[derive(Clone, Module)]
pub struct GtcrnStream {
    pub erb: Erb,
    #[module(key = "encoder.en_convs")]
    encoder: [EncoderLayer; 5],
    dpgrnn1: Dpgrnn,
    dpgrnn2: Dpgrnn,
    #[module(key = "decoder.de_convs")]
    decoder: StreamDecoder,
    /// Mask post-processing (scale + dry/wet blend). Defaults to identity.
    #[module(skip)]
    pub mask: MaskConfig,
}

impl GtcrnStream {
    /// Depthwise conv-cache shapes `(1, C, (kT-1)·d, F)`, in the order
    /// [`forward_stream`](Self::forward_stream) takes them: encoder blocks 2..5
    /// (dilations 1, 2, 5), then decoder blocks 0..3 (5, 2, 1).
    pub const CONV_CACHE: [[usize; 4]; 2 * FLIPPED] = {
        let mut out = [[1, C_NET, 0, DPGRNN_WIDTH]; 2 * FLIPPED];
        let mut i = 0;
        while i < out.len() {
            out[i][2] = hist(i);
            i += 1;
        }
        out
    };

    /// One TRA attention block's GRU hidden state.
    pub const TRA_CACHE: [usize; 2] = [1, TRA_HIDDEN];

    /// One DPGRNN's `inter_rnn` hidden state, batched over `B·F`.
    pub const INTER_CACHE: [usize; 2] = [DPGRNN_WIDTH, DPGRNN_HIDDEN];

    /// One streaming step over a `(1, 257, 1, 2)` frame, returning the enhanced
    /// frame plus the three cache families in their input order.
    ///
    /// A pure graph function: it reads caches and returns new ones. Writing
    /// them back into their own buffers is the JIT's `state { .. }` job.
    pub fn forward_stream(
        &self,
        spec: &Tensor,
        conv: CacheRefs<'_>,
        tra: CacheRefs<'_>,
        inter: [&Tensor; 2],
    ) -> Result<Step> {
        // spec: (B, F, T=1, 2) -> feat (B, 3, T, 257), as `Gtcrn::forward`.
        let spec_real = spec.complex_real()?.try_permute(&[0, 2, 1])?;
        let spec_imag = spec.complex_imag()?.try_permute(&[0, 2, 1])?;
        let spec_mag = spec.magnitude(1e-12)?.try_permute(&[0, 2, 1])?;
        let feat = Tensor::stack(&[&spec_mag, &spec_real, &spec_imag], 1)?;

        let feat = self.erb.bm(&feat)?; // (B,3,T,129)
        let feat = sfe(&feat, 3, C_IN)?; // (B,9,T,129)

        // The cache index advances only over the GT blocks, which is why the
        // encoder's three (positions 2..5) take slots 0..3 and the decoder's
        // three (positions 0..3) take slots 3..6.
        let (mut new_conv, mut new_tra) = (Vec::with_capacity(2 * FLIPPED), Vec::with_capacity(2 * FLIPPED));
        let mut en_outs: Vec<Tensor> = Vec::with_capacity(self.encoder.len());
        let mut x = feat;
        for layer in &self.encoder {
            x = match layer {
                EncoderLayer::Conv(c) => c.forward(&x)?,
                EncoderLayer::Gt(g) => step(g, &x, conv, tra, &mut new_conv, &mut new_tra)?,
            };
            en_outs.push(x.clone());
        }

        let (x, inter0) = self.dpgrnn1.forward_stream(&x, inter[0])?;
        let (mut x, inter1) = self.dpgrnn2.forward_stream(&x, inter[1])?;

        for (skip, layer) in en_outs.iter().rev().zip(&self.decoder.0) {
            let added = x.try_add(skip)?;
            x = match layer {
                DecoderLayer::Gt(g) => step(g, &added, conv, tra, &mut new_conv, &mut new_tra)?,
                DecoderLayer::Conv(c) => c.forward(&added)?,
            };
        }

        // ERB synthesis, then the complex ratio mask in the input's own layout.
        let m = self.erb.bs(&x)?.try_permute(&[0, 3, 2, 1])?; // (B,F,T,2)
        let crm = spec.complex_mul(&scaled(m, self.mask.scale)?)?;
        let enh = match approx_one(self.mask.blend) {
            true => crm,
            false => scaled(crm, self.mask.blend)?.try_add(&scaled(spec.clone(), 1.0 - self.mask.blend)?)?,
        };

        Ok((enh, caches(new_conv), caches(new_tra), [inter0, inter1]))
    }

    // ----------------------------------------------------------------------- //
    // Loaders (the offline gtcrn.safetensors, unmodified)
    // ----------------------------------------------------------------------- //

    /// Download `CHECKPOINT` from [`HUB_REPO`] and load it.
    pub fn from_hub() -> Result<Self> {
        Self::from_hub_with_revision("main")
    }

    pub fn from_hub_with_revision(revision: &str) -> Result<Self> {
        let path = crate::hub::HubRepo::open(HUB_REPO, revision)
            .and_then(|repo| repo.get(CHECKPOINT))
            .context(HubSnafu { context: format!("{HUB_REPO}@{revision}/{CHECKPOINT}") })?;
        Self::from_safetensors(&path)
    }

    pub fn from_safetensors(path: &Path) -> Result<Self> {
        Self::from_state_dict(&state::load_safetensors(path)?)
    }

    pub fn from_state_dict(sd: &StateDict) -> Result<Self> {
        let mut model = Self::with_random_weights();
        model.load_state_dict(sd, "")?;
        Ok(model)
    }

    /// Build with random weights matching the GTCRN layout (for tests/JIT
    /// exercising without a checkpoint).
    pub fn with_random_weights() -> Self {
        Self {
            erb: Erb::empty(),
            encoder: stream_encoder(),
            dpgrnn1: empty_dpgrnn(),
            dpgrnn2: empty_dpgrnn(),
            decoder: StreamDecoder(stream_decoder()),
            mask: MaskConfig::default(),
        }
    }

    /// Set the mask post-processing config. Must be called before JIT
    /// `prepare` — the values bake into the compiled graph.
    pub fn with_mask(mut self, mask: MaskConfig) -> Self {
        self.mask = mask;
        self
    }
}

/// Run one GT block against the next unused cache slot, recording the caches it
/// hands back. `new_conv.len()` *is* the slot index — the encoder's three GT
/// blocks fill 0..3 and the decoder's the rest, matching the reference's
/// `conv_cache[0]` / `conv_cache[1]` split.
fn step(
    block: &GTConvBlock,
    x: &Tensor,
    conv: CacheRefs<'_>,
    tra: CacheRefs<'_>,
    new_conv: &mut Vec<Tensor>,
    new_tra: &mut Vec<Tensor>,
) -> Result<Tensor> {
    let i = new_conv.len();
    let (y, nc, nt) = block.forward_stream(x, conv[i], tra[i])?;
    new_conv.push(nc);
    new_tra.push(nt);
    Ok(y)
}

fn caches(v: Vec<Tensor>) -> Caches {
    assert_eq!(v.len(), 2 * FLIPPED, "every GT block must produce exactly one cache");
    std::array::from_fn(|i| v[i].clone())
}

/// `k == 1.0` to within one ulp — the identity, which costs no graph node.
fn approx_one(k: f32) -> bool {
    (k - 1.0).abs() <= f32::EPSILON
}

fn scaled(t: Tensor, k: f32) -> Result<Tensor> {
    if approx_one(k) { Ok(t) } else { Ok(t.try_mul(k)?) }
}

// =========================================================================== //
// GtcrnStreamJit — the JIT wrapper
// =========================================================================== //

/// `state { conv: [Tensor; 6] }` needs a literal length.
const _: () = assert!(2 * FLIPPED == 6, "the stream JIT's cache slots must track the GT block count");

jit_wrapper! {
    GtcrnStreamJit(GtcrnStream) {
        inputs { spec: Tensor }
        // The caches recycle in place: each new cache is stored into its own
        // input buffer, so `execute()` leaves the state where the next frame
        // reads it. Read-before-write is safe — every cache is read exactly
        // once (the cat / the GRU's initial_h) before its store.
        state { conv: [Tensor; 6], tra: [Tensor; 6], inter: [Tensor; 2] }
        outputs { enh }

        build(spec, conv, tra, inter) {
            model.forward_stream(spec, conv, tra, inter)
        }
    }
}

impl GtcrnStreamJit {
    /// Compile the one-frame plan: `(1, 257, 1, 2)` in and out, every cache
    /// device-local and zeroed. `execute()` then costs one dispatch per frame
    /// and `reset()` starts a new stream.
    pub fn prepared(model: GtcrnStream) -> crate::jit::Result<Self> {
        let mut jit = Self::new(model);
        jit.prepare_with_config(
            InputSpec::f32(&[1, N_FREQ, 1, 2]),
            std::array::from_fn(|i| InputSpec::f32(&GtcrnStream::CONV_CACHE[i])),
            std::array::from_fn(|_| InputSpec::f32(&GtcrnStream::TRA_CACHE)),
            std::array::from_fn(|_| InputSpec::f32(&GtcrnStream::INTER_CACHE)),
            &svod_tensor::PrepareConfig::device_local(),
        )?;
        Ok(jit)
    }
}

// =========================================================================== //
// Streaming forwards — the offline blocks, with state threaded through
// =========================================================================== //

impl GTConvBlock {
    /// `(B, C, T, F)` + conv cache `(B, C, hist, F)` + TRA hidden `(B, 2C)` →
    /// `(out, new_cache, new_hidden)`.
    ///
    /// The cache is concatenated in front on the T axis, so the depthwise conv
    /// sees exactly the frames the offline block's causal pad would have
    /// supplied (`convolution.py:85-93` and, for the flipped decoder conv,
    /// `:232-262`). The new cache is the tail of that concatenation — taken
    /// before any F padding, as the reference does.
    ///
    /// The `kT` dilated taps are gathered by hand and the conv runs undilated
    /// on T (see [`stream_depth_conv`]): `Tensor::pool` computes its fold
    /// factor as `max(1, ceildiv(out·stride - dilation, in))` over unsigned
    /// `SInt`s, so the subtraction panics whenever the T output extent is
    /// smaller than the dilation — which one-frame streaming always is.
    /// Gathering first is also the cheaper spelling: the conv then reads `kT`
    /// rows instead of `(kT-1)·d + 1`.
    pub(super) fn forward_stream(
        &self,
        x: &Tensor,
        conv_cache: &Tensor,
        tra_h: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let half = self.in_channels / 2;
        let halves = x.chunk(2, 1)?;

        let x1 = sfe(&halves[0], 3, half)?;
        let h1 = self.point_conv1.forward(&x1)?;
        let h1 = self.point_bn1.forward(&h1)?;
        let h1 = h1.prelu(&self.point_act)?;

        let inp = Tensor::cat(&[conv_cache, &h1], 2)?;
        let (hist, len) = (conv_cache.dim_const(2)?, inp.dim_const(2)?);
        assert_eq!(len - hist, 1, "the streaming block consumes exactly one frame per call");
        let new_cache = inp.narrow(2, len - hist, hist)?;

        // Frames `0, d, …, (kT-1)·d` of the history, where `hist = (kT-1)·d`.
        let kt = depth_kernel_t(&self.depth_conv)?;
        let taps: Vec<Tensor> =
            (0..kt).map(|k| inp.narrow(2, k * (hist / (kt - 1)), 1usize)).collect::<svod_tensor::error::Result<_>>()?;
        let y = self.depth_conv.forward(&Tensor::cat(&taps.iter().collect::<Vec<_>>(), 2)?)?;

        let h1 = self.depth_bn.forward(&y)?;
        let h1 = h1.prelu(&self.depth_act)?;
        let h1 = self.point_conv2.forward(&h1)?;
        let h1 = self.point_bn2.forward(&h1)?;

        let (h1, new_h) = self.tra.forward_stream(&h1, tra_h)?;
        Ok((Self::shuffle(&h1, &halves[1], half)?, new_cache, new_h))
    }
}

impl Tra {
    /// [`Tra::forward`] with the attention GRU's hidden state threaded:
    /// `(B, C, T, F)` + `h (B, 2C)` → `((B, C, T, F), new_h)`.
    fn forward_stream(&self, x: &Tensor, h_prev: &Tensor) -> Result<(Tensor, Tensor)> {
        let zt = x.square().mean_with().axes(-1isize).keepdim(false).call()?; // (B,C,T)
        let (at, new_h) = self.gru.forward_with_state(&zt.try_permute(&[0, 2, 1])?, h_prev)?;
        let at = self.fc.forward(&at)?.try_permute(&[0, 2, 1])?.sigmoid()?; // (B,C,T)
        Ok((x.try_mul(&at.try_unsqueeze(-1)?)?, new_h))
    }
}

impl Dpgrnn {
    /// [`Dpgrnn::forward`] with the `inter_rnn` hidden state threaded. The
    /// `intra_rnn` runs along F within the frame and is stateless.
    fn forward_stream(&self, x: &Tensor, inter_h: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = x.try_permute(&[0, 2, 3, 1])?; // (B,T,F,C)
        let (b, t, f, c) = (x.dim_const(0)?, x.dim_const(1)?, x.dim_const(2)?, x.dim_const(3)?);

        let intra_x = x.try_reshape([b * t, f, c])?;
        let intra_x = self.intra_rnn.forward(&intra_x)?;
        let intra_x = self.intra_fc.forward(&intra_x)?;
        let intra_x = self.intra_ln.forward(&intra_x.try_reshape([b, t, f, c])?)?;
        let intra_out = x.try_add(&intra_x)?;

        let inter_in = intra_out.try_permute(&[0, 2, 1, 3])?; // (B,F,T,C)
        let (inter_x, new_h) = self.inter_rnn.forward_with_state(&inter_in.try_reshape([b * f, t, c])?, inter_h)?;
        let inter_x = self.inter_fc.forward(&inter_x)?;
        let inter_x = inter_x.try_reshape([b, f, t, c])?.try_permute(&[0, 2, 1, 3])?; // (B,T,F,C)
        let inter_x = self.inter_ln.forward(&inter_x)?;
        let inter_out = intra_out.try_add(&inter_x)?;

        Ok((inter_out.try_permute(&[0, 3, 1, 2])?, new_h)) // (B,C,T,F)
    }
}

impl Grnn {
    /// [`Grnn::forward`] with hidden state. Only the unidirectional `inter_rnn`
    /// threads state, and its merged form needs no reordering at all: the
    /// block-diagonal GRU's hidden state *is* `[h₁ | h₂]`, exactly the
    /// concatenation the split pair carried, so the `(B·F, hidden)` cache
    /// layout is unchanged.
    fn forward_with_state(&self, x: &Tensor, h: &Tensor) -> Result<(Tensor, Tensor)> {
        self.merged_forward()?.forward_with_state(x, h)
    }
}

impl GruWeights {
    /// Run the GRU over `(B, T, input)` from an explicit initial hidden state,
    /// returning `((B, T, H), new_h (B, H))`. Unidirectional by construction —
    /// only the T-axis GRUs (TRA and `inter_rnn`) thread state.
    fn forward_with_state(&self, x: &Tensor, h0: &Tensor) -> Result<(Tensor, Tensor)> {
        let bias = Tensor::cat(&[&self.bias_ih, &self.bias_hh], 0)?;
        let out = x
            .gru()
            .w(&self.weight_ih.try_unsqueeze(0)?)
            .r_weights(&self.weight_hh.try_unsqueeze(0)?)
            .hidden_size(self.hidden_size)
            .bias(&bias.try_unsqueeze(0)?)
            .initial_h(&h0.try_unsqueeze(0)?)
            .direction(GruDirection::Forward)
            .linear_before_reset(true)
            .layout(RnnLayout::BatchFirst)
            .call()?;
        // `output` is PyTorch-shaped [batch, seq, D*hidden] and `h_n`
        // [num_directions, batch, hidden] in both layouts.
        Ok((out.output, out.h_n.try_squeeze(Some(0))?))
    }
}

// =========================================================================== //
// The decoder, whose depthwise kernels live flipped
// =========================================================================== //

/// The decoder stack. Its first three blocks hold their depthwise conv as a
/// regular [`Conv2d`] over a spatially flipped kernel, so the state dict is
/// flipped on the way in **and** on the way out: `flip` is an involution, which
/// keeps the emitted dict byte-identical to the checkpoint and makes
/// `from_state_dict(&m.state_dict(""))` the identity.
#[derive(Clone)]
struct StreamDecoder([DecoderLayer; 5]);

/// `flip(&[-2, -1])` on the depthwise kernel — see the module header.
fn flip_kernel(w: &Tensor) -> svod_tensor::error::Result<Tensor> {
    w.flip(&[-2, -1])
}

impl Module for StreamDecoder {
    fn write_state(&self, prefix: &str, out: &mut StateDict) {
        self.0.write_state(prefix, out);
        for i in 0..FLIPPED {
            let key = prefixed(prefix, &format!("{i}.depth_conv.weight"));
            let w = flip_kernel(out.get(&key).expect("the derive just wrote every decoder key"))
                .expect("a 4-D depthwise kernel flips");
            out.insert(key, w);
        }
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> svod_tensor::error::Result<()> {
        self.0.load_state_dict(sd, prefix)?;
        for layer in &mut self.0[..FLIPPED] {
            if let DecoderLayer::Gt(g) = layer
                && let Conv::Normal(c) = &mut g.depth_conv
            {
                c.weight = flip_kernel(&c.weight)?;
            }
        }
        Ok(())
    }
}

// =========================================================================== //
// Architecture: the offline stacks, with the causal pad traded for a cache
// =========================================================================== //

fn stream_encoder() -> [EncoderLayer; 5] {
    let mut encoder = default_encoder();
    for layer in &mut encoder {
        if let EncoderLayer::Gt(g) = layer {
            g.depth_conv = stream_depth_conv(&g.depth_conv);
        }
    }
    encoder
}

fn stream_decoder() -> [DecoderLayer; 5] {
    let mut decoder = default_decoder();
    for layer in &mut decoder {
        if let DecoderLayer::Gt(g) = layer {
            g.depth_conv = stream_depth_conv(&g.depth_conv);
        }
    }
    decoder
}

/// The T extent of a depthwise kernel — `kT` in `(kT - 1)·dilation`.
fn depth_kernel_t(conv: &Conv) -> Result<usize> {
    Ok(match conv {
        Conv::Normal(c) => c.weight.dim_const(2)?,
        Conv::Transposed(c) => c.weight.dim_const(2)?,
    })
}

/// Rewrite an offline depthwise conv for streaming.
///
/// On T the conv keeps nothing: the cache supplies the left context, so the
/// causal pad goes, and [`GTConvBlock::forward_stream`] gathers the dilated
/// taps itself, so the dilation goes with it.
///
/// A `ConvTranspose2d` additionally becomes a regular [`Conv2d`] over the
/// flipped kernel — `StreamConvTranspose2d` Version 2 — whose F padding is the
/// `(F_size - 1)·F_dilation - F_pad` the reference applies by hand
/// (`convolution.py:258`), replacing the crop the transpose spelling implied.
/// The kernel is flipped when the checkpoint is loaded, not here.
fn stream_depth_conv(conv: &Conv) -> Conv {
    match conv {
        Conv::Normal(c) => Conv::Normal(c.clone().with_padding(((0, 0), c.padding.1)).with_dilation((1, c.dilation.1))),
        Conv::Transposed(c) => {
            let (before, after) = c.padding.1;
            assert_eq!(before, after, "GTCRN pads F symmetrically");
            let kw = c.weight.dim_const(3).expect("a static depthwise kernel") as isize;
            let f_pad = (kw - 1) * c.dilation.1 as isize - before;
            Conv::Normal(
                Conv2d::new(c.weight.clone(), c.bias.clone())
                    .with_groups(c.groups)
                    .with_dilation((1, c.dilation.1))
                    .with_padding(((0, 0), (f_pad, f_pad))),
            )
        }
    }
}
