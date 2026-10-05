//! Graph rewrite engine implementation.
//!
//! # Algorithm
//!
//! Stack-based 3-stage DFS traversal with waitlist for dependency resolution:
//!
//! - **Stage 0 (PushChildren)**: Apply `bpm` patterns (if present), then push children
//!   - `bpm` patterns see **ORIGINAL** children
//!   - Used for bottom-up patterns that need to transform before descent
//!
//! - **Stage 1 (ApplyPatterns)**: Reconstruct with optimized children, then apply `pm` patterns
//!   - `pm` patterns see **OPTIMIZED** children
//!   - This is the default mode - patterns run after children are processed
//!
//! - **Stage 2 (Link)**: Link original node to final result
//!
//! # API
//!
//! - `graph_rewrite(pm, root, ctx)` - Default: patterns see optimized children (Stage 1)
//! - `graph_rewrite_bottom_up(bpm, root, ctx)` - Patterns see original children (Stage 0)
//!
//! # Pattern Context
//!
//! Context is passed at rewrite-time through `graph_rewrite()`, not captured in
//! closures. This provides compile-time type safety without `Rc<RefCell<>>`
//! boilerplate.
//!
//! ## Example
//!
//! ```ignore
//! use svod_ir::pattern::SimplifiedPatternMatcher;
//!
//! // Create context
//! let mut ctx = MyContext::new();
//!
//! // Create matcher using the patterns! macro
//! let matcher = patterns! {
//!     Add(x, @zero) => x,
//!     Mul(x, @one) => x,
//! };
//!
//! // Pass context at rewrite time - patterns see OPTIMIZED children
//! let result = graph_rewrite(&matcher, root, &mut ctx);
//! ```
//!
//! Patterns that don't need context use `()` as the context type:
//!
//! ```ignore
//! let matcher = patterns! {
//!     Add(x, @zero) => x,
//! };
//! let result = graph_rewrite(&matcher, root, &mut ());
//! ```

use crate::{Op, UOp};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::{SmallVec, smallvec};
use std::sync::Arc;

use crate::ops;
use crate::pattern::{Matcher, RewriteResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TraversalMode {
    Full,
    PreserveCallBodies,
}

/// Borrowed children, split into traversed sources and skipped-but-preserved ones.
/// Borrowing (rather than `Op::sources()`) keeps the per-visit cost at zero
/// allocations and zero refcount traffic — tinygrad reads `n.src` directly.
type Children<'a> = SmallVec<[&'a Arc<UOp>; 4]>;

fn traversal_children<'a>(node: &'a Arc<UOp>, mode: TraversalMode) -> (Children<'a>, Children<'a>) {
    if mode == TraversalMode::Full {
        return (node.op().children(), SmallVec::new());
    }

    match node.op() {
        Op::Call(ops::Call { body, args, .. }) | Op::Function(ops::Function { body, args, .. }) => {
            (args.iter().collect(), smallvec![body])
        }
        // PROGRAM is opaque when preserving call bodies.
        Op::Program(ops::Program { sink, linear, source, binary, .. }) => {
            let mut skipped: Children<'a> = smallvec![sink];
            skipped.extend(linear.iter().chain(source.iter()).chain(binary.iter()));
            (SmallVec::new(), skipped)
        }
        _ => (node.op().children(), SmallVec::new()),
    }
}

/// Maximum stack size before we consider the rewrite to be in an infinite loop.
const REWRITE_STACK_LIMIT: usize = 500_000;

/// Stack entry for the 3-stage rewrite algorithm.
///
/// - `n`: the original node (used as key in `replace` dict)
/// - `stage`: 0 (PushChildren), 1 (ApplyPatterns), or 2 (Link)
/// - `new_n`: the working copy (may differ from `n` after bpm rewrites or reconstruction)
struct Entry {
    n: Arc<UOp>,
    stage: u8,
    /// The working copy when it differs from `n`; `None` means "same as `n`",
    /// which saves a refcount pair on every child push and every unchanged node.
    new_n: Option<Arc<UOp>>,
}

/// Internal rewrite engine.
///
/// Generic over matcher types and context type for compile-time type-safe matching.
/// Supports separate `pm` (top-down) and `bpm` (bottom-up) matchers.
struct RewriteEngine<'a, PM, BPM, C>
where
    PM: Matcher<C>,
    BPM: Matcher<C>,
{
    /// Top-down pattern matcher: applied in Stage 1 (ApplyPatterns).
    /// Patterns see OPTIMIZED children.
    pm: Option<&'a PM>,

    /// Bottom-up pattern matcher: applied in Stage 0 (PushChildren).
    /// Patterns see ORIGINAL children.
    bpm: Option<&'a BPM>,

    /// Mutable reference to context passed through to patterns.
    ctx: &'a mut C,

    /// Traversal mode controlling CALL/FUNCTION/PROGRAM boundary handling.
    traversal_mode: TraversalMode,

    /// Results cache: maps original node id → optimized result.
    ///
    /// Keyed on `UOp::id` with an Fx hasher. The key wrapper this replaced already
    /// compared and hashed by id alone, so the `Arc` it carried only bought a
    /// refcount pair per probe, and SipHash only bought cycles over an integer we
    /// already had.
    replace: FxHashMap<u64, Arc<UOp>>,

    /// BPM result cache: prevents re-running pattern matching on nodes already seen.
    bpm_cache: FxHashMap<u64, Option<Arc<UOp>>>,

    /// Scratch set for the bpm fixed-point loop, reused across nodes.
    bpm_seen: FxHashSet<u64>,
}

impl<'a, PM, BPM, C> RewriteEngine<'a, PM, BPM, C>
where
    PM: Matcher<C>,
    BPM: Matcher<C>,
{
    fn new(pm: Option<&'a PM>, bpm: Option<&'a BPM>, ctx: &'a mut C, traversal_mode: TraversalMode) -> Self {
        Self {
            pm,
            bpm,
            ctx,
            traversal_mode,
            replace: FxHashMap::default(),
            bpm_cache: FxHashMap::default(),
            bpm_seen: FxHashSet::default(),
        }
    }

    /// Single-shot top-down pattern application.
    /// No cache needed: pm_rewrite is called at most once per UOp due to the
    /// replace dict check in the main loop.
    #[inline]
    fn pm_rewrite(&mut self, x: &Arc<UOp>) -> Option<Arc<UOp>> {
        let pm = self.pm.as_ref()?;
        match pm.rewrite(x, self.ctx) {
            RewriteResult::Rewritten(new_node) => {
                debug_assert!(
                    !Arc::ptr_eq(&new_node, x),
                    "PM pattern returned Rewritten but produced the same node (id={}). \
                     This causes infinite loops. Return NoMatch instead.\nOp: {:?}",
                    x.id,
                    x.op().as_ref(),
                );
                Some(new_node)
            }
            RewriteResult::Gate(_) | RewriteResult::NoMatch => None,
        }
    }

    /// Cached bottom-up pattern application.
    /// Cache prevents re-running patterns on nodes already seen during fixed-point.
    /// Gate results are NOT cached — Gate is an exception that bypasses the cache.
    #[inline]
    fn cached_bpm_rewrite(&mut self, x: &Arc<UOp>) -> Result<Option<Arc<UOp>>, Arc<UOp>> {
        let key = x.id;
        if let Some(cached) = self.bpm_cache.get(&key) {
            return match cached {
                Some(node) => Ok(Some(node.clone())),
                None => Ok(None),
            };
        }
        let bpm = self.bpm.as_ref().unwrap();
        match bpm.rewrite(x, self.ctx) {
            RewriteResult::Rewritten(new_node) => {
                debug_assert!(
                    !Arc::ptr_eq(&new_node, x),
                    "BPM pattern returned Rewritten but produced the same node (id={}). \
                     This causes infinite loops. Return NoMatch instead.\nOp: {:?}",
                    x.id,
                    x.op().as_ref(),
                );
                self.bpm_cache.insert(key, Some(new_node.clone()));
                Ok(Some(new_node))
            }
            RewriteResult::Gate(gate_node) => Err(gate_node),
            RewriteResult::NoMatch => {
                self.bpm_cache.insert(key, None);
                Ok(None)
            }
        }
    }

    /// Record a result in the replace map.
    #[inline]
    fn record_replace(&mut self, original_id: u64, result: Arc<UOp>) {
        self.replace.insert(original_id, result);
    }

    /// Main rewrite method — stack-based 3-stage traversal.
    fn rewrite(&mut self, root: Arc<UOp>) -> Arc<UOp> {
        let mut stack: Vec<Entry> = vec![Entry { n: root.clone(), stage: 0, new_n: None }];

        // All UOps either on the stack or in self.replace — don't have to be placed again.
        let mut on_stack: FxHashSet<u64> = FxHashSet::default();
        on_stack.insert(root.id);

        // UOps waiting on a dependency to be in self.replace.
        let mut waitlist: FxHashMap<u64, SmallVec<[Entry; 1]>> = FxHashMap::default();

        while let Some(Entry { n, stage, new_n }) = stack.pop() {
            if stack.len() > REWRITE_STACK_LIMIT {
                panic!(
                    "infinite loop in graph_rewrite (stack too big: {}). results cached: {}",
                    stack.len(),
                    self.replace.len(),
                );
            }

            let n_key = n.id;

            if self.replace.contains_key(&n_key) {
                continue;
            }

            if stage == 0 {
                // Stage 0: PushChildren
                let mut working = new_n;

                if self.bpm.is_some() {
                    // Apply bpm rewrite rules until a fixed point is reached.
                    // The scratch set is reused across nodes rather than freshly
                    // allocated per node — a deliberate improvement over tinygrad's
                    // per-node `seen = set()` (`RewriteContext.unified_rewrite` in
                    // `uop/ops.py`), which costs ~280k allocations on a resnet50
                    // schedule for a set that almost always holds a single element.
                    self.bpm_seen.clear();
                    let mut gated = false;
                    loop {
                        let cur = working.as_ref().unwrap_or(&n);
                        if !self.bpm_seen.insert(cur.id) {
                            panic!(
                                "infinite loop in fixed_point_rewrite: node {:?} (id={}) seen twice",
                                cur.op().as_ref(),
                                cur.id
                            );
                        }
                        match self.cached_bpm_rewrite(cur) {
                            Ok(Some(rewritten)) => working = Some(rewritten),
                            Ok(None) => break,
                            Err(gate_node) => {
                                // Gate: done with this node, don't descend into children
                                self.record_replace(n_key, gate_node);
                                if let Some(entries) = waitlist.remove(&n_key) {
                                    stack.extend(entries);
                                }
                                gated = true;
                                break;
                            }
                        }
                    }
                    if gated {
                        continue;
                    }
                }

                // Children are collected before `n`/`working` move into the stage-1 entry.
                let kids: SmallVec<[Arc<UOp>; 4]> = {
                    let cur = working.as_ref().unwrap_or(&n);
                    let (sources, skipped) = traversal_children(cur, self.traversal_mode);
                    for skipped_child in skipped {
                        self.replace.entry(skipped_child.id).or_insert_with(|| skipped_child.clone());
                    }
                    sources.into_iter().rev().filter(|child| on_stack.insert(child.id)).cloned().collect()
                };
                stack.push(Entry { n, stage: 1, new_n: working });
                stack.extend(kids.into_iter().map(|child| Entry { n: child, stage: 0, new_n: None }));
            } else if stage == 1 {
                // Stage 1: ApplyPatterns. Sources are resolved by reference: the
                // common unchanged case allocates nothing and touches no refcounts.
                let cur = new_n.as_ref().unwrap_or(&n);
                let (rebuilt, missing) = {
                    let sources = cur.op().children();
                    let mut resolved: SmallVec<[&Arc<UOp>; 4]> = SmallVec::with_capacity(sources.len());
                    let mut changed = false;
                    let mut missing = None;
                    for src in &sources {
                        match self.replace.get(&src.id) {
                            Some(rx) => {
                                changed |= !Arc::ptr_eq(rx, src);
                                resolved.push(rx);
                            }
                            None => {
                                missing = Some(src.id);
                                break;
                            }
                        }
                    }
                    // Hash consing may collapse reconstruction to the same node even when
                    // sources logically changed; that is treated as unchanged below.
                    let rebuilt = if missing.is_none() && changed {
                        Some(cur.with_sources(resolved.iter().map(|r| (*r).clone()).collect()))
                    } else {
                        None
                    };
                    (rebuilt, missing)
                };

                if let Some(src_key) = missing {
                    // Source not ready: register in waitlist
                    waitlist.entry(src_key).or_default().push(Entry { n, stage: 1, new_n });
                    continue;
                }

                match rebuilt {
                    Some(node) if !Arc::ptr_eq(&node, cur) => {
                        // Reconstruction produced a new node — process it, then link back
                        stack.push(Entry { n, stage: 2, new_n: Some(node.clone()) });
                        stack.push(Entry { n: node, stage: 0, new_n: None });
                    }
                    _ => {
                        // Sources effectively unchanged: try pm rewrite
                        if let Some(new_src_n) = self.pm_rewrite(cur) {
                            stack.push(Entry { n, stage: 2, new_n: Some(new_src_n.clone()) });
                            stack.push(Entry { n: new_src_n, stage: 0, new_n: None });
                        } else {
                            // No pm match — done with this node
                            let result = match new_n {
                                Some(w) => w,
                                None => n,
                            };
                            self.record_replace(n_key, result);
                            if let Some(entries) = waitlist.remove(&n_key) {
                                stack.extend(entries);
                            }
                        }
                    }
                }
            } else {
                // Stage 2: Link
                let new_n = new_n.expect("stage 2 entries always carry the rewritten node");
                let new_n_key = new_n.id;

                if let Some(replaced_new_n) = self.replace.get(&new_n_key).cloned() {
                    self.record_replace(n_key, replaced_new_n);
                    if let Some(entries) = waitlist.remove(&n_key) {
                        stack.extend(entries);
                    }
                } else {
                    // Not ready: register in waitlist
                    waitlist.entry(new_n_key).or_default().push(Entry { n, stage: 2, new_n: Some(new_n) });
                }
            }
        }

        self.replace.get(&root.id).cloned().unwrap_or(root)
    }

    /// MLIR-style walk pattern rewrite driver — single-pass, no re-traversal.
    ///
    /// Differs from [`Self::rewrite`] in two ways:
    /// 1. `bpm` is applied **once** per node — no fixed-point loop.
    /// 2. When a pattern returns a replacement, the replacement is **not**
    ///    traversed: its children are never visited and `pm` is not re-applied.
    ///
    /// Use this when a replacement contains the original key, e.g.
    /// `Buffer → After(Buffer, [Store(...)])` for view-assign, where
    /// re-traversal would loop or wrap the key multiple times.
    fn walk_rewrite(&mut self, root: Arc<UOp>) -> Arc<UOp> {
        let mut stack: Vec<(Arc<UOp>, bool)> = vec![(root.clone(), false)];

        while let Some((n, processed)) = stack.pop() {
            if stack.len() > REWRITE_STACK_LIMIT {
                panic!(
                    "infinite loop in walk_rewrite (stack too big: {}). results cached: {}",
                    stack.len(),
                    self.replace.len(),
                );
            }

            let n_key = n.id;
            if self.replace.contains_key(&n_key) {
                continue;
            }

            if !processed {
                // Try bpm exactly once on the original node. On match, record the
                // replacement and skip descent — the replacement is treated as a
                // leaf even if it contains the original key.
                if self.bpm.is_some() {
                    match self.cached_bpm_rewrite(&n) {
                        Ok(Some(rewritten)) => {
                            self.record_replace(n.id, rewritten);
                            continue;
                        }
                        Err(gated) => {
                            self.record_replace(n.id, gated);
                            continue;
                        }
                        Ok(None) => {}
                    }
                }

                // No bpm match — process children, then come back to rebuild.
                stack.push((n.clone(), true));

                let (sources, skipped) = traversal_children(&n, self.traversal_mode);
                for skipped_child in skipped {
                    self.replace.entry(skipped_child.id).or_insert_with(|| skipped_child.clone());
                }
                for child in sources.into_iter().rev() {
                    if !self.replace.contains_key(&child.id) {
                        stack.push((child.clone(), false));
                    }
                }
            } else {
                // Rebuild with rewritten sources.
                let sources = n.op().children();
                let new_src: Vec<Arc<UOp>> =
                    sources.iter().map(|s| self.replace.get(&s.id).cloned().unwrap_or_else(|| (*s).clone())).collect();
                let any_changed = new_src.iter().zip(sources.iter()).any(|(a, b)| !Arc::ptr_eq(a, b));
                let rebuilt = if any_changed { n.with_sources(new_src) } else { n.clone() };

                // Apply pm exactly once on the rebuilt node — replacement is used as-is.
                let final_n = if self.pm.is_some() { self.pm_rewrite(&rebuilt).unwrap_or(rebuilt) } else { rebuilt };

                self.record_replace(n.id, final_n);
            }
        }

        self.replace.get(&root.id).cloned().unwrap_or(root)
    }
}

/// Marker type for "no matcher" in generic contexts.
pub struct NoMatcher;

impl<C> Matcher<C> for NoMatcher {
    fn rewrite(&self, _node: &Arc<UOp>, _ctx: &mut C) -> RewriteResult {
        RewriteResult::NoMatch
    }
}

/// Apply graph rewriting. Patterns see **OPTIMIZED** children (Stage 1).
pub fn graph_rewrite<M: Matcher<C>, C>(matcher: &M, root: Arc<UOp>, ctx: &mut C) -> Arc<UOp> {
    RewriteEngine::new(Some(matcher), None::<&NoMatcher>, ctx, TraversalMode::Full).rewrite(root)
}

/// Apply graph rewriting with bottom-up pattern application.
/// Patterns see **ORIGINAL** children (Stage 0).
pub fn graph_rewrite_bottom_up<M: Matcher<C>, C>(matcher: &M, root: Arc<UOp>, ctx: &mut C) -> Arc<UOp> {
    RewriteEngine::new(None::<&NoMatcher>, Some(matcher), ctx, TraversalMode::Full).rewrite(root)
}

/// MLIR-style walk pattern rewrite — single-pass, no re-traversal.
///
/// Use when a replacement may contain the original key (e.g.
/// `Buffer → After(Buffer, [Store(...)])` for view-assign): a re-traversing
/// driver would loop or wrap the key multiple times.
pub fn graph_rewrite_walk<M: Matcher<C>, C>(matcher: &M, root: Arc<UOp>, ctx: &mut C) -> Arc<UOp> {
    RewriteEngine::new(None::<&NoMatcher>, Some(matcher), ctx, TraversalMode::Full).walk_rewrite(root)
}

/// Single-pass walk rewrite that does not enter CALL/FUNCTION bodies or PROGRAM internals.
pub fn graph_rewrite_walk_preserve_calls<M: Matcher<C>, C>(matcher: &M, root: Arc<UOp>, ctx: &mut C) -> Arc<UOp> {
    RewriteEngine::new(None::<&NoMatcher>, Some(matcher), ctx, TraversalMode::PreserveCallBodies).walk_rewrite(root)
}

/// Apply graph rewriting with both top-down and bottom-up patterns.
/// - `bpm` patterns see ORIGINAL children (Stage 0)
/// - `pm` patterns see OPTIMIZED children (Stage 1)
pub fn graph_rewrite_with_bpm<PM, BPM, C>(pm: &PM, bpm: &BPM, root: Arc<UOp>, ctx: &mut C) -> Arc<UOp>
where
    PM: Matcher<C>,
    BPM: Matcher<C>,
{
    RewriteEngine::new(Some(pm), Some(bpm), ctx, TraversalMode::Full).rewrite(root)
}

/// Apply graph rewriting with both top-down and bottom-up patterns while preserving
/// CALL/FUNCTION/PROGRAM boundaries.
/// - `bpm` patterns see ORIGINAL children (Stage 0)
/// - `pm` patterns see OPTIMIZED children (Stage 1)
/// - Traversal skips CALL/FUNCTION bodies and PROGRAM internals
/// - CALL/FUNCTION args are still traversed; PROGRAM has no traversed internals
pub fn graph_rewrite_with_bpm_preserve_calls<PM, BPM, C>(pm: &PM, bpm: &BPM, root: Arc<UOp>, ctx: &mut C) -> Arc<UOp>
where
    PM: Matcher<C>,
    BPM: Matcher<C>,
{
    RewriteEngine::new(Some(pm), Some(bpm), ctx, TraversalMode::PreserveCallBodies).rewrite(root)
}

/// Rewrite graph while preserving CALL/FUNCTION/PROGRAM boundaries.
///
/// CALL/FUNCTION/PROGRAM nodes are still matchable/rewriteable, but traversal
/// does not descend into CALL/FUNCTION bodies or PROGRAM internals.
/// CALL/FUNCTION arguments are still traversed; PROGRAM has no traversed internals.
pub fn graph_rewrite_preserve_calls<PM, C>(pm: &PM, root: Arc<UOp>, ctx: &mut C) -> Arc<UOp>
where
    PM: Matcher<C>,
{
    RewriteEngine::new(Some(pm), None::<&NoMatcher>, ctx, TraversalMode::PreserveCallBodies).rewrite(root)
}

/// Bottom-up rewrite with CALL/FUNCTION/PROGRAM boundary preservation.
///
/// CALL/FUNCTION/PROGRAM nodes are still matchable/rewriteable, but traversal
/// does not descend into CALL/FUNCTION bodies or PROGRAM internals.
/// CALL/FUNCTION arguments are still traversed; PROGRAM has no traversed internals.
pub fn graph_rewrite_bottom_up_preserve_calls<BPM, C>(bpm: &BPM, root: Arc<UOp>, ctx: &mut C) -> Arc<UOp>
where
    BPM: Matcher<C>,
{
    RewriteEngine::new(None::<&NoMatcher>, Some(bpm), ctx, TraversalMode::PreserveCallBodies).rewrite(root)
}
