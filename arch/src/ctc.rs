//! Connectionist Temporal Classification (CTC) decoders.
//!
//! Decoders own their vocabulary and configuration; callers pass only per-run
//! data (`log_probs`, `valid_frames`). This crate operates on plain `&[f32]`
//! log-prob slices — there is no coupling to `svod-tensor`, `svod-device`,
//! or any specific model crate.
//!
//! Two decoders are provided:
//! - [`GreedyDecoder`] — argmax per timestep, collapse runs of the same token,
//!   drop blanks. The fast, default path. `&self` decode, infallible.
//! - [`BeamDecoder`] — log-domain prefix beam search backed by an internal
//!   suffix tree (prefix sharing across beam entries; `Copy` 16-byte
//!   `SearchPoint`s instead of allocated prefix vectors). `&mut self` decode
//!   with reusable scratch state. Returns [`Result<_, DecodeError>`] so NaN
//!   handling can be configured via [`BeamOpts::on_nan`].
//!
//! Use [`CtcDecoder`] when you need a single owning enum (e.g. when the choice
//! between greedy and beam is config-driven). With the `serde` feature, all
//! types deserialize from a `{"type": "greedy" | "beam", "vocabulary": [...],
//! ...}` shape.
//!
//! # Layout convention
//!
//! All decode methods take a row-major `[stride_frames, total_vocab]` slice
//! for a single batch item, where `total_vocab == vocabulary.len() + 1` and
//! the blank token id is appended after the vocabulary. `valid_frames`
//! clamps padding produced by static JIT shapes — only frames
//! `0..valid_frames` are consumed.
//!
//! # Per-token timestamps
//!
//! [`GreedyDecoder::decode_with_timestamps`] and
//! [`BeamDecoder::decode_with_timestamps`] return `(text, frames)` where
//! `frames[i]` is the timestep at which the `i`-th output character was
//! emitted (post collapse-runs and blank-drop). Useful for word-level
//! alignment with the original audio.

use std::cmp::Ordering;

use snafu::Snafu;

#[cfg(feature = "serde")]
use serde::Deserialize;

// ─── Errors ───────────────────────────────────────────────────────────────

/// Failure modes for [`BeamDecoder`] (and the `Beam` arm of [`CtcDecoder`]).
///
/// Greedy decoding is infallible — NaN logits are absorbed by the NaN-safe
/// argmax and don't surface as errors.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum DecodeError {
    /// Beam decode encountered a NaN logit and [`BeamOpts::on_nan`] was
    /// [`NanBehavior::Error`].
    #[snafu(display("NaN logits at frame {frame}"))]
    NanInLogits { frame: usize },
}

// ─── NaN behavior ─────────────────────────────────────────────────────────

/// What [`BeamDecoder`] does when a frame contains NaN logits.
///
/// Default is [`NanBehavior::Skip`] — match the existing decoder behavior so
/// adding `on_nan` to [`BeamOpts`] is a non-breaking change for serialized
/// configs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum NanBehavior {
    /// Skip the frame entirely (no contribution to beam scores). The total
    /// number of skipped frames is reported via `eprintln!` once after the
    /// decode completes. This is the default.
    #[default]
    Skip,
    /// Return [`DecodeError::NanInLogits`] when any frame contains NaN.
    /// Strict mode for production deployments where a NaN in the encoder
    /// output should surface, not silently degrade transcriptions.
    Error,
    /// Pass NaN through to the inner-loop arithmetic. Output is unspecified
    /// and may itself contain NaN-poisoned scores. Mainly useful for encoder
    /// bring-up / debugging.
    Propagate,
}

// ─── Greedy ───────────────────────────────────────────────────────────────

/// Greedy CTC decoder: argmax per timestep, collapse runs, drop blanks.
///
/// Holds the vocabulary; the blank token id is implicit at `vocabulary.len()`.
/// Decode is `&self` and allocation-free apart from the returned `String`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct GreedyDecoder {
    vocabulary: Vec<String>,
}

impl GreedyDecoder {
    pub fn new(vocabulary: Vec<String>) -> Self {
        Self { vocabulary }
    }

    pub fn vocabulary(&self) -> &[String] {
        &self.vocabulary
    }

    pub fn blank_id(&self) -> usize {
        self.vocabulary.len()
    }

    pub fn total_vocab(&self) -> usize {
        self.vocabulary.len() + 1
    }

    /// Decode a single batch item. Empty `vocabulary` or `valid_frames == 0`
    /// yield an empty string. NaN best-logits are counted and reported via
    /// `eprintln!` — they don't abort the decode.
    pub fn decode(&self, log_probs: &[f32], stride_frames: usize, valid_frames: usize) -> String {
        let (text, _frames) = self.decode_inner(log_probs, stride_frames, valid_frames, false);
        text
    }

    /// Decode plus per-output-token frame indices. `frames[i]` is the
    /// timestep at which the `i`-th output character was emitted (post
    /// collapse-runs, post blank-drop). `text.chars().count() == frames.len()`
    /// when `vocabulary` entries are all single Unicode scalar values; for
    /// multi-char tokens, `frames.len()` matches the number of *tokens*, not
    /// chars.
    pub fn decode_with_timestamps(
        &self,
        log_probs: &[f32],
        stride_frames: usize,
        valid_frames: usize,
    ) -> (String, Vec<usize>) {
        self.decode_inner(log_probs, stride_frames, valid_frames, true)
    }

    fn decode_inner(
        &self,
        log_probs: &[f32],
        stride_frames: usize,
        valid_frames: usize,
        keep_frames: bool,
    ) -> (String, Vec<usize>) {
        if self.vocabulary.is_empty() || valid_frames == 0 {
            return (String::new(), Vec::new());
        }
        let blank_id = self.blank_id();
        let total_vocab = self.total_vocab();
        let n_frames = stride_frames.min(valid_frames);

        let mut prev = blank_id;
        let mut text = String::new();
        let mut frames = if keep_frames { Vec::with_capacity(n_frames / 2) } else { Vec::new() };
        let mut nan_frames = 0usize;
        for t in 0..n_frames {
            let base = t * total_vocab;
            let frame = &log_probs[base..base + total_vocab];
            let best = argmax_nan_safe(frame);
            if frame[best].is_nan() {
                nan_frames += 1;
            }
            if best != blank_id && best != prev {
                text.push_str(&self.vocabulary[best]);
                if keep_frames {
                    frames.push(t);
                }
            }
            prev = best;
        }
        if nan_frames > 0 {
            eprintln!("ctc::GreedyDecoder: {nan_frames}/{n_frames} frames with NaN best logit");
        }
        (text, frames)
    }

    /// Decode a `[B, stride_frames, total_vocab]` slab. `valid_frames.len()`
    /// defines the batch size.
    pub fn decode_batch(&self, log_probs: &[f32], stride_frames: usize, valid_frames: &[usize]) -> Vec<String> {
        let item_stride = stride_frames * self.total_vocab();
        valid_frames
            .iter()
            .enumerate()
            .map(|(b, &valid)| {
                let base = b * item_stride;
                self.decode(&log_probs[base..base + item_stride], stride_frames, valid)
            })
            .collect()
    }

    /// Per-frame argmax (no collapsing, no blank-dropping). Powers debug
    /// inspection paths that need raw token streams.
    pub fn argmax_per_frame(&self, log_probs: &[f32], stride_frames: usize, valid_frames: usize) -> Vec<usize> {
        let total_vocab = self.total_vocab();
        let n_frames = stride_frames.min(valid_frames);
        (0..n_frames)
            .map(|t| {
                let base = t * total_vocab;
                argmax_nan_safe(&log_probs[base..base + total_vocab])
            })
            .collect()
    }
}

// ─── Beam options ─────────────────────────────────────────────────────────

/// Tunables for [`BeamDecoder`]. Defaults: `beam_size=100`,
/// `beam_size_token=Some(8)`, `beam_threshold=Some(20.0)`, `on_nan=Skip`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct BeamOpts {
    /// Maximum number of prefixes retained per step.
    #[cfg_attr(feature = "serde", serde(default = "default_beam_size"))]
    pub beam_size: usize,
    /// Per-step top-K vocab prune. `None` keeps all tokens. The default
    /// `Some(8)` cuts the inner-loop fan-out from `beam × V` to `beam × 8`
    /// — for typical ASR vocabularies (V≈30+) the bottom of the per-frame
    /// log-prob distribution contributes negligible mass and is safe to
    /// prune. Set to `None` if you want strict equivalence to a
    /// no-pruning beam search.
    #[cfg_attr(feature = "serde", serde(default = "default_beam_size_token"))]
    pub beam_size_token: Option<usize>,
    /// Drop prefixes whose combined log-prob is more than `threshold` below
    /// the current best. `None` disables threshold pruning. The current best
    /// is always retained, so a too-aggressive threshold cannot empty the
    /// beam — but it can still narrow it severely.
    #[cfg_attr(feature = "serde", serde(default = "default_beam_threshold"))]
    pub beam_threshold: Option<f32>,
    /// What to do when a frame contains NaN logits. See [`NanBehavior`].
    #[cfg_attr(feature = "serde", serde(default))]
    pub on_nan: NanBehavior,
}

#[cfg(feature = "serde")]
fn default_beam_size() -> usize {
    100
}
#[cfg(feature = "serde")]
fn default_beam_size_token() -> Option<usize> {
    Some(8)
}
#[cfg(feature = "serde")]
fn default_beam_threshold() -> Option<f32> {
    Some(20.0)
}

impl Default for BeamOpts {
    fn default() -> Self {
        Self { beam_size: 100, beam_size_token: Some(8), beam_threshold: Some(20.0), on_nan: NanBehavior::Skip }
    }
}

// ─── Beam decoder ─────────────────────────────────────────────────────────

/// 1D prefix beam CTC decoder backed by an internal suffix tree.
///
/// Pure log-space; never exponentiates intermediate values. NaN handling is
/// controlled by [`BeamOpts::on_nan`]. Owns mutable scratch buffers (the
/// suffix tree, beam/next `SearchPoint` vectors, sort buffers) that are
/// reused across [`decode`](Self::decode) calls — after warm-up the hot path
/// is allocation-free.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct BeamDecoder {
    vocabulary: Vec<String>,
    #[cfg_attr(feature = "serde", serde(flatten))]
    opts: BeamOpts,
    #[cfg_attr(feature = "serde", serde(skip))]
    scratch: BeamScratch,
}

#[derive(Debug, Default)]
struct BeamScratch {
    tree: SuffixTree,
    beam: Vec<SearchPoint>,
    next: Vec<SearchPoint>,
    token_buf: Vec<usize>,
    /// Reused across decode calls to collect (label, frame) pairs from the
    /// tree walk without per-call alloc.
    walk_buf: Vec<(u32, u32)>,
}

impl Clone for BeamDecoder {
    fn clone(&self) -> Self {
        Self { vocabulary: self.vocabulary.clone(), opts: self.opts.clone(), scratch: BeamScratch::default() }
    }
}

impl BeamDecoder {
    pub fn new(vocabulary: Vec<String>, opts: BeamOpts) -> Self {
        Self { vocabulary, opts, scratch: BeamScratch::default() }
    }

    pub fn vocabulary(&self) -> &[String] {
        &self.vocabulary
    }

    pub fn blank_id(&self) -> usize {
        self.vocabulary.len()
    }

    pub fn total_vocab(&self) -> usize {
        self.vocabulary.len() + 1
    }

    pub fn opts(&self) -> &BeamOpts {
        &self.opts
    }

    pub fn opts_mut(&mut self) -> &mut BeamOpts {
        &mut self.opts
    }

    /// Decode a single batch item. Returns `Err` if [`BeamOpts::on_nan`] is
    /// [`NanBehavior::Error`] and a NaN logit is encountered. Empty
    /// `vocabulary`, `valid_frames == 0`, or `beam_size == 0` yield an empty
    /// string (`Ok("")`).
    pub fn decode(
        &mut self,
        log_probs: &[f32],
        stride_frames: usize,
        valid_frames: usize,
    ) -> Result<String, DecodeError> {
        let node = self.decode_to_best_node(log_probs, stride_frames, valid_frames)?;
        Ok(self.collect_text(node))
    }

    /// Like [`decode`](Self::decode), but also returns per-output-token
    /// frame indices.
    pub fn decode_with_timestamps(
        &mut self,
        log_probs: &[f32],
        stride_frames: usize,
        valid_frames: usize,
    ) -> Result<(String, Vec<usize>), DecodeError> {
        let node = self.decode_to_best_node(log_probs, stride_frames, valid_frames)?;
        Ok(self.collect_text_and_frames(node))
    }

    /// Decode a `[B, stride_frames, total_vocab]` slab.
    pub fn decode_batch(
        &mut self,
        log_probs: &[f32],
        stride_frames: usize,
        valid_frames: &[usize],
    ) -> Result<Vec<String>, DecodeError> {
        let item_stride = stride_frames * self.total_vocab();
        let mut out = Vec::with_capacity(valid_frames.len());
        for (b, &valid) in valid_frames.iter().enumerate() {
            let base = b * item_stride;
            out.push(self.decode(&log_probs[base..base + item_stride], stride_frames, valid)?);
        }
        Ok(out)
    }

    /// Run the beam search; return the best beam entry's tree node id. The
    /// caller walks `self.scratch.tree.iter_from(node)` to reconstruct the
    /// prefix.
    fn decode_to_best_node(
        &mut self,
        log_probs: &[f32],
        stride_frames: usize,
        valid_frames: usize,
    ) -> Result<i32, DecodeError> {
        if self.vocabulary.is_empty() || valid_frames == 0 || self.opts.beam_size == 0 {
            return Ok(ROOT_NODE);
        }

        let blank_id = self.blank_id();
        let total_vocab = self.total_vocab();
        let n_frames = stride_frames.min(valid_frames);
        let alphabet_size = self.vocabulary.len();

        // Split-borrow: `self.scratch` is mutably borrowed as a whole, but
        // we still need read access to `self.opts` and `self.vocabulary` —
        // those are different fields, so Rust's disjoint-borrow rules
        // permit it.
        let BeamScratch { tree, beam, next, token_buf, .. } = &mut self.scratch;
        tree.reset(alphabet_size);
        beam.clear();
        // Initial state: all probability mass on the empty-prefix blank path.
        beam.push(SearchPoint::new(ROOT_NODE, 0.0, f32::NEG_INFINITY));
        next.clear();

        let opts = &self.opts;
        let mut nan_frames = 0usize;

        for t in 0..n_frames {
            let base = t * total_vocab;
            let frame = &log_probs[base..base + total_vocab];

            if frame.iter().any(|v| v.is_nan()) {
                match opts.on_nan {
                    NanBehavior::Skip => {
                        nan_frames += 1;
                        continue;
                    }
                    NanBehavior::Error => {
                        return NanInLogitsSnafu { frame: t }.fail();
                    }
                    NanBehavior::Propagate => { /* fall through with NaN */ }
                }
            }

            top_k_tokens_into(frame, opts.beam_size_token, blank_id, token_buf);

            next.clear();
            for &SearchPoint { node, p_b, p_nb, log_total: p_total } in beam.iter() {
                let tip_label = tree.label(node);

                for &c in token_buf.iter() {
                    let lp = frame[c];
                    if c == blank_id {
                        // Blank extension: prefix unchanged.
                        next.push(SearchPoint::new(node, p_total + lp, f32::NEG_INFINITY));
                    } else if Some(c as u32) == tip_label {
                        // Same as last token: extend without growth (no-blank
                        // path keeps the run collapsed).
                        next.push(SearchPoint::new(node, f32::NEG_INFINITY, p_nb + lp));
                        // Also extend with growth via blank-collapse, but
                        // only if we have any blank-mass to feed it.
                        if p_b > f32::NEG_INFINITY {
                            let child = tree
                                .get_child(node, c as u32)
                                .unwrap_or_else(|| tree.add_node(node, c as u32, t as u32));
                            next.push(SearchPoint::new(child, f32::NEG_INFINITY, p_b + lp));
                        }
                    } else {
                        // New token: grow the prefix.
                        let child =
                            tree.get_child(node, c as u32).unwrap_or_else(|| tree.add_node(node, c as u32, t as u32));
                        next.push(SearchPoint::new(child, f32::NEG_INFINITY, p_total + lp));
                    }
                }
            }

            std::mem::swap(beam, next);

            // Dedup: sort by node id, then sweep merging consecutive same-
            // node entries. Each merge `log_add_exp`'s both axes and rebuilds
            // the surviving `SearchPoint` via the constructor (which keeps
            // `log_total` consistent). Marked-for-delete entries get
            // `i32::MIN` and are retained-out at the end.
            beam.sort_by_key(|sp| sp.node);
            const DELETE: i32 = i32::MIN;
            let mut last_key = DELETE;
            let mut last_pos = 0usize;
            for i in 0..beam.len() {
                let bi = beam[i];
                if bi.node == last_key {
                    let merged = SearchPoint::new(
                        beam[last_pos].node,
                        log_add_exp(beam[last_pos].p_b, bi.p_b),
                        log_add_exp(beam[last_pos].p_nb, bi.p_nb),
                    );
                    beam[last_pos] = merged;
                    beam[i].node = DELETE;
                } else {
                    last_pos = i;
                    last_key = bi.node;
                }
            }
            beam.retain(|sp| sp.node != DELETE);

            // Sort by combined log-prob (descending), apply threshold, truncate.
            // The comparator now reads the cached `log_total` directly — no
            // `log_add_exp` calls per comparison.
            beam.sort_unstable_by(|a, b| compare_logits(b.log_total, a.log_total));
            if let Some(threshold) = opts.beam_threshold
                && let Some(best) = beam.first()
            {
                let cutoff = best.log_total - threshold;
                beam.retain(|sp| sp.log_total >= cutoff);
            }
            beam.truncate(opts.beam_size);
        }

        if nan_frames > 0 {
            eprintln!("ctc::BeamDecoder: {nan_frames}/{n_frames} frames with NaN logits skipped");
        }

        Ok(beam.first().map(|sp| sp.node).unwrap_or(ROOT_NODE))
    }

    fn collect_text(&mut self, node: i32) -> String {
        let walk = &mut self.scratch.walk_buf;
        walk.clear();
        walk.extend(self.scratch.tree.iter_from(node));
        let mut text = String::new();
        for &(label, _) in walk.iter().rev() {
            let id = label as usize;
            if id < self.vocabulary.len() {
                text.push_str(&self.vocabulary[id]);
            }
        }
        text
    }

    fn collect_text_and_frames(&mut self, node: i32) -> (String, Vec<usize>) {
        let walk = &mut self.scratch.walk_buf;
        walk.clear();
        walk.extend(self.scratch.tree.iter_from(node));
        let mut text = String::new();
        let mut frames = Vec::with_capacity(walk.len());
        for &(label, frame) in walk.iter().rev() {
            let id = label as usize;
            if id < self.vocabulary.len() {
                text.push_str(&self.vocabulary[id]);
                frames.push(frame as usize);
            }
        }
        (text, frames)
    }
}

// ─── CtcDecoder enum ──────────────────────────────────────────────────────

/// Owning enum over the available CTC decoders. Use this when the choice is
/// config-driven (e.g. deserialized from JSON).
///
/// All decode methods return `Result<_, DecodeError>` for a uniform dispatch
/// surface. The `Greedy` arm always succeeds (NaN handling is absorbed by
/// the NaN-safe argmax), but signatures are unified.
///
/// `BeamDecoder` is boxed because its scratch buffers make it ~10× larger
/// than `GreedyDecoder`; without the box the enum would be sized for the
/// largest variant on every value, including `Greedy` ones.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type", rename_all = "snake_case"))]
pub enum CtcDecoder {
    Greedy(GreedyDecoder),
    Beam(Box<BeamDecoder>),
}

impl CtcDecoder {
    pub fn vocabulary(&self) -> &[String] {
        match self {
            Self::Greedy(d) => d.vocabulary(),
            Self::Beam(d) => d.vocabulary(),
        }
    }

    pub fn blank_id(&self) -> usize {
        match self {
            Self::Greedy(d) => d.blank_id(),
            Self::Beam(d) => d.blank_id(),
        }
    }

    pub fn total_vocab(&self) -> usize {
        match self {
            Self::Greedy(d) => d.total_vocab(),
            Self::Beam(d) => d.total_vocab(),
        }
    }

    pub fn decode(
        &mut self,
        log_probs: &[f32],
        stride_frames: usize,
        valid_frames: usize,
    ) -> Result<String, DecodeError> {
        match self {
            Self::Greedy(d) => Ok(d.decode(log_probs, stride_frames, valid_frames)),
            Self::Beam(d) => d.decode(log_probs, stride_frames, valid_frames),
        }
    }

    pub fn decode_with_timestamps(
        &mut self,
        log_probs: &[f32],
        stride_frames: usize,
        valid_frames: usize,
    ) -> Result<(String, Vec<usize>), DecodeError> {
        match self {
            Self::Greedy(d) => Ok(d.decode_with_timestamps(log_probs, stride_frames, valid_frames)),
            Self::Beam(d) => d.decode_with_timestamps(log_probs, stride_frames, valid_frames),
        }
    }

    pub fn decode_batch(
        &mut self,
        log_probs: &[f32],
        stride_frames: usize,
        valid_frames: &[usize],
    ) -> Result<Vec<String>, DecodeError> {
        match self {
            Self::Greedy(d) => Ok(d.decode_batch(log_probs, stride_frames, valid_frames)),
            Self::Beam(d) => d.decode_batch(log_probs, stride_frames, valid_frames),
        }
    }
}

// ─── Internal: SuffixTree ─────────────────────────────────────────────────

const ROOT_NODE: i32 = -1;

#[derive(Clone, Copy, Debug)]
struct LabelNode {
    /// Index into vocabulary (NOT including blank).
    label: u32,
    /// Parent node id; [`ROOT_NODE`] = -1 for direct children of root.
    parent: i32,
    /// First frame at which this node was added.
    frame: u32,
}

/// Trie of partial labellings, with one slot of associated data per node
/// (the frame index at which the node was first added). Inlined from
/// `submodules/fast-ctc-decode/src/tree.rs:1–195` and specialized for our
/// use case (T = u32 frame, fixed alphabet size per decoder).
#[derive(Debug, Default)]
struct SuffixTree {
    nodes: Vec<LabelNode>,
    /// Flat row-major 2D: `children[node * alphabet_size + label]`. -1 means
    /// no child for that (node, label) edge.
    children: Vec<i32>,
    /// Children of the virtual root (which has no associated label/parent).
    /// Length == `alphabet_size`.
    root_children: Vec<i32>,
    alphabet_size: usize,
}

impl SuffixTree {
    fn reset(&mut self, alphabet_size: usize) {
        self.nodes.clear();
        self.children.clear();
        self.alphabet_size = alphabet_size;
        self.root_children.clear();
        self.root_children.resize(alphabet_size, -1);
    }

    fn label(&self, node: i32) -> Option<u32> {
        if node >= 0 { Some(self.nodes[node as usize].label) } else { None }
    }

    fn get_child(&self, node: i32, label: u32) -> Option<i32> {
        let label = label as usize;
        debug_assert!(label < self.alphabet_size);
        let idx = if node == ROOT_NODE {
            self.root_children[label]
        } else {
            self.children[node as usize * self.alphabet_size + label]
        };
        if idx >= 0 { Some(idx) } else { None }
    }

    fn add_node(&mut self, parent: i32, label: u32, frame: u32) -> i32 {
        let label_us = label as usize;
        debug_assert!(label_us < self.alphabet_size);
        let new_idx = self.nodes.len() as i32;
        if parent == ROOT_NODE {
            debug_assert_eq!(self.root_children[label_us], -1);
            self.root_children[label_us] = new_idx;
        } else {
            let slot = parent as usize * self.alphabet_size + label_us;
            debug_assert_eq!(self.children[slot], -1);
            self.children[slot] = new_idx;
        }
        self.nodes.push(LabelNode { label, parent, frame });
        // Reserve a fresh row of `-1` children for the new node.
        self.children.resize(self.children.len() + self.alphabet_size, -1);
        new_idx
    }

    fn iter_from(&self, node: i32) -> SuffixTreeIter<'_> {
        SuffixTreeIter { nodes: &self.nodes, next: node }
    }
}

struct SuffixTreeIter<'a> {
    nodes: &'a [LabelNode],
    next: i32,
}

impl<'a> Iterator for SuffixTreeIter<'a> {
    type Item = (u32, u32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= 0 {
            let node = &self.nodes[self.next as usize];
            self.next = node.parent;
            Some((node.label, node.frame))
        } else {
            None
        }
    }
}

// ─── Internal: SearchPoint + helpers ──────────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct SearchPoint {
    /// Tree node id; [`ROOT_NODE`] = -1 means empty prefix.
    node: i32,
    /// log p_blank (mass on paths ending with a blank).
    p_b: f32,
    /// log p_nonblank (mass on paths ending with the tip label).
    p_nb: f32,
    /// Cached `log_add_exp(p_b, p_nb)`. Maintained as an invariant by
    /// [`SearchPoint::new`] — the only constructor — so sort comparator and
    /// threshold filter read it directly without per-comparison
    /// `log_add_exp` calls.
    log_total: f32,
}

impl SearchPoint {
    /// The only way to construct a `SearchPoint`. Computing `log_total` here
    /// is essentially free: every push site in the inner loop has either
    /// `p_b == -inf` or `p_nb == -inf`, and [`log_add_exp`] short-circuits in
    /// both cases. The dedup merge calls this with both axes finite, paying
    /// one `log_add_exp` per merged group — same cost as before but with the
    /// invariant locked in by the type.
    fn new(node: i32, p_b: f32, p_nb: f32) -> Self {
        Self { node, p_b, p_nb, log_total: log_add_exp(p_b, p_nb) }
    }
}

fn argmax_nan_safe(frame: &[f32]) -> usize {
    let mut best = 0usize;
    for i in 1..frame.len() {
        if compare_logits(frame[i], frame[best]).is_gt() {
            best = i;
        }
    }
    best
}

fn compare_logits(a: f32, b: f32) -> Ordering {
    a.partial_cmp(&b).unwrap_or_else(|| match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => Ordering::Equal,
    })
}

/// `log(exp(a) + exp(b))` without exponentiating. Returns the other operand
/// when one is `-inf`.
fn log_add_exp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    hi + (-(hi - lo)).exp().ln_1p()
}

fn top_k_tokens_into(frame: &[f32], k: Option<usize>, blank_id: usize, out: &mut Vec<usize>) {
    out.clear();
    let v = frame.len();
    let Some(k) = k else {
        out.extend(0..v);
        return;
    };
    if k == 0 {
        // Degenerate config — still keep the blank since the inner loop
        // depends on it for prefix-collapse semantics.
        out.push(blank_id);
        return;
    }
    if k >= v {
        out.extend(0..v);
        return;
    }
    // Partial-sort: place the k-th-largest at index k-1; everything before
    // is the top-K (in unspecified order). Average O(v); avoids the full
    // O(v log v) sort we don't need.
    out.extend(0..v);
    out.select_nth_unstable_by(k - 1, |&i, &j| compare_logits(frame[j], frame[i]));
    out.truncate(k);
    if !out.contains(&blank_id) {
        out.push(blank_id);
    }
}
