//! Scoped synchronization hazards on hardware: every test races a long
//! `slow_double` kernel (reads and writes at its end) against a host or
//! copy-lane access and asserts the values, so a missing wait shows up as
//! wrong data rather than as timing.

use std::sync::Arc;
use std::time::{Duration, Instant};

use svod_dtype::DType;

use super::graph::Chain;
use super::{Hardware, cuda_alloc_or_skip, device_ptr, download, load, slow_abi, upload, vadd_abi};
use crate::Buffer;
use crate::allocator::{Allocator, BufferSpec, RawBuffer};
use crate::cuda::{CudaAllocator, CudaDevice, CudaPlanCtx, CudaProgram};
use crate::device::{PlanContext, Program};
use crate::sync::CompletionToken;

const N: usize = 256;
/// Milliseconds a "long" kernel sleeps before touching memory; "did not
/// wait" is asserted as a bound on the caller's latency well under it.
const LONG_MS: i64 = 150;
const NOT_WAITED: Duration = Duration::from_millis(LONG_MS as u64 / 2);

fn base(buffer: &RawBuffer) -> u64 {
    device_ptr(buffer) as u64
}

fn values(seed: f32) -> Vec<f32> {
    (0..N).map(|i| seed + i as f32).collect()
}

fn doubled(input: &[f32]) -> Vec<f32> {
    input.iter().map(|x| 2.0 * x).collect()
}

struct Slow {
    alloc: Hardware<CudaAllocator>,
    program: CudaProgram,
}

impl Slow {
    fn new() -> Option<Self> {
        let alloc = cuda_alloc_or_skip()?;
        let program = load(&alloc.dev, "slow_double", &slow_abi());
        Some(Self { alloc, program })
    }

    fn dev(&self) -> &Arc<CudaDevice> {
        &self.alloc.dev
    }

    /// `out = 2 * input` after `ms` milliseconds, on `ctx`.
    fn dispatch(&self, ctx: &CudaPlanCtx, out: &RawBuffer, input: &RawBuffer, ms: i64) {
        unsafe {
            ctx.dispatch(
                &self.program,
                &[device_ptr(out), device_ptr(input)],
                &[ms],
                Some([1, 1, 1]),
                Some([N, 1, 1]),
                false,
            )
        }
        .unwrap();
    }

    /// The same through the per-call path (device dispatch lane).
    fn execute(&self, out: *mut u8, input: *mut u8, ms: i64) {
        unsafe { self.program.execute(&[out, input], &[ms], Some([1, 1, 1]), Some([N, 1, 1]), false) }.unwrap();
    }

    fn ctx(&self) -> CudaPlanCtx {
        CudaPlanCtx::new(Arc::clone(self.dev())).unwrap()
    }
}

/// Publish `ctx`'s token on `storages`, as the executor does after a plan.
fn publish(dev: &CudaDevice, ctx: &CudaPlanCtx, storages: &[&RawBuffer]) -> Arc<dyn CompletionToken> {
    let token = ctx.completion_token().expect("CUDA contexts hand out tokens");
    for storage in storages {
        dev.record_producer(base(storage), &token);
    }
    token.published();
    token
}

/// A host read waits the lane that produced the storage and nothing else:
/// the short kernel's result is read back while the long kernel on another
/// lane is still running.
#[test]
fn read_after_write_waits_only_the_producing_lane() {
    let Some(slow) = Slow::new() else { return };
    let (input_a, input_b) = (upload(&slow.alloc, &values(1.0)), upload(&slow.alloc, &values(1000.0)));
    let (out_a, out_b) = (upload(&slow.alloc, &vec![0.0; N]), upload(&slow.alloc, &vec![0.0; N]));
    let (ctx_a, ctx_b) = (slow.ctx(), slow.ctx());
    slow.dispatch(&ctx_a, &out_a, &input_a, LONG_MS);
    let token_a = publish(slow.dev(), &ctx_a, &[&out_a, &input_a]);
    slow.dispatch(&ctx_b, &out_b, &input_b, 5);
    publish(slow.dev(), &ctx_b, &[&out_b, &input_b]);
    let started = Instant::now();
    assert_eq!(download(&slow.alloc, &out_b, N), doubled(&values(1000.0)));
    let elapsed = started.elapsed();
    assert!(elapsed < NOT_WAITED, "the read waited the unrelated long kernel ({elapsed:?})");
    assert_eq!(download(&slow.alloc, &out_a, N), doubled(&values(1.0)));
    assert!(token_a.retired());
    assert_eq!(slow.dev().producer_count(base(&out_a)), Some(0), "waited tokens are pruned");
}

/// A host write into a buffer a kernel is still about to read waits that
/// kernel (WAR), so the kernel sees the old contents.
#[test]
fn host_copyin_waits_in_flight_readers() {
    let Some(slow) = Slow::new() else { return };
    let old = values(1.0);
    let (input, out) = (upload(&slow.alloc, &old), upload(&slow.alloc, &vec![0.0; N]));
    let ctx = slow.ctx();
    slow.dispatch(&ctx, &out, &input, LONG_MS);
    let token = publish(slow.dev(), &ctx, &[&out, &input]);
    slow.alloc._copyin(&input, 0, super::f32_bytes(&values(500.0))).unwrap();
    assert!(token.retired(), "copyin returned before the reader retired");
    assert_eq!(download(&slow.alloc, &out, N), doubled(&old));
    assert_eq!(download(&slow.alloc, &input, N), values(500.0));
}

/// A device-to-device transfer whose source is still being written is
/// ordered after the writer on the GPU and returns without a host wait; a
/// kernel launched right after it on another lane sees the copied data.
#[test]
fn transfer_is_ordered_after_the_writer_and_before_later_launches() {
    let Some(slow) = Slow::new() else { return };
    let input = values(3.0);
    let (source, out) = (upload(&slow.alloc, &input), upload(&slow.alloc, &vec![0.0; N]));
    let (copy, chained) = (upload(&slow.alloc, &vec![0.0; N]), upload(&slow.alloc, &vec![0.0; N]));
    let ctx = slow.ctx();
    slow.dispatch(&ctx, &out, &source, LONG_MS);
    publish(slow.dev(), &ctx, &[&out, &source]);
    let started = Instant::now();
    slow.alloc._transfer(&copy, 0, &out, 0, N * 4).unwrap();
    let elapsed = started.elapsed();
    assert!(elapsed < NOT_WAITED, "the transfer waited on the host ({elapsed:?})");
    assert_eq!(slow.dev().producer_count(base(&copy)), Some(1), "the copy is the destination's producer");
    assert_eq!(slow.dev().producer_count(base(&out)), Some(2), "the copy is a reader of the source");
    // Another lane reads the copy immediately: it must follow the copy.
    let reader = slow.ctx();
    slow.dispatch(&reader, &chained, &copy, 0);
    publish(slow.dev(), &reader, &[&chained, &copy]);
    assert_eq!(download(&slow.alloc, &chained, N), doubled(&doubled(&input)));
    assert_eq!(download(&slow.alloc, &copy, N), doubled(&input));
}

/// Same-storage overlapping moves keep memmove semantics while the
/// storage is still being written.
#[test]
fn overlapping_transfer_waits_the_writer() {
    let Some(slow) = Slow::new() else { return };
    let input = values(7.0);
    let (source, out) = (upload(&slow.alloc, &input), upload(&slow.alloc, &vec![0.0; N]));
    let ctx = slow.ctx();
    slow.dispatch(&ctx, &out, &source, LONG_MS);
    publish(slow.dev(), &ctx, &[&out, &source]);
    slow.alloc._transfer(&out, 4, &out, 0, (N - 1) * 4).unwrap();
    let expected: Vec<f32> = std::iter::once(2.0 * input[0]).chain(doubled(&input[..N - 1])).collect();
    assert_eq!(download(&slow.alloc, &out, N), expected);
}

/// A per-call dispatch (`Program::execute`, `wait=false`) and an unpublished
/// plan dispatch have no token on their storages; host reads still see their
/// results through the lanes' unpublished flags, and `Buffer::synchronize`
/// drains everything.
#[test]
fn unpublished_dispatches_are_waited_and_synchronize_drains() {
    let Some(slow) = Slow::new() else { return };
    let input = values(11.0);
    let (source, out) = (upload(&slow.alloc, &input), upload(&slow.alloc, &vec![0.0; N]));
    slow.execute(device_ptr(&out), device_ptr(&source), LONG_MS);
    assert_eq!(download(&slow.alloc, &out, N), doubled(&input));
    let ctx = slow.ctx();
    slow.dispatch(&ctx, &out, &out, LONG_MS);
    assert_eq!(download(&slow.alloc, &out, N), doubled(&doubled(&input)));
    slow.dispatch(&ctx, &out, &out, LONG_MS);
    let alloc: Arc<dyn Allocator> = Arc::new((*slow.alloc).clone());
    let buffer = Buffer::new(alloc, DType::Float32, vec![1], BufferSpec::default());
    buffer.synchronize().unwrap();
    // A token minted after the drain retires at once: nothing was left.
    ctx.completion_token().unwrap().wait(LONG_MS as u64 / 4).expect("synchronize drained the lane");
    assert_eq!(download(&slow.alloc, &out, N), doubled(&doubled(&doubled(&input))));
}

/// A graph replay followed by a host read, with and without the executor
/// publishing the graph's token.
#[test]
fn graph_replay_then_host_read() {
    let Some(chain) = Chain::new() else { return };
    let graph = chain.capture();
    graph.replay(&[], &[]).unwrap();
    assert_eq!(download(&chain.alloc, &chain.out, chain.alloc_len()), chain.expected());
    chain.alloc._copyin(&chain.out, 0, &vec![0u8; chain.alloc_len() * 4]).unwrap();
    graph.replay(&[], &[]).unwrap();
    let token = graph.completion_token().expect("replays record completion");
    chain.alloc.dev.record_producer(base(&chain.out), &token);
    assert_eq!(chain.alloc.dev.producer_count(base(&chain.out)), Some(2), "the copy-in's token and the replay's");
    assert_eq!(download(&chain.alloc, &chain.out, chain.alloc_len()), chain.expected());
    assert!(token.retired());
}

/// Managed memory host views (`as_slice`) wait the storage's producers, and
/// `Buffer::record_completion` publishes onto CUDA storage.
#[test]
fn managed_host_views_wait_the_producer() {
    let Some(slow) = Slow::new() else { return };
    if !slow.dev().limits().managed_memory {
        eprintln!("skipping: device has no coherent managed memory");
        return;
    }
    let alloc: Arc<dyn Allocator> = Arc::new((*slow.alloc).clone());
    let input = values(21.0);
    let mut source = Buffer::new(alloc.clone(), DType::Float32, vec![N], BufferSpec::default());
    source.copyin(super::f32_bytes(&input)).unwrap();
    let out = Buffer::new(alloc, DType::Float32, vec![N], BufferSpec::default());
    out.ensure_allocated().unwrap();
    let ctx = slow.ctx();
    unsafe {
        ctx.dispatch(
            &slow.program,
            &[out.as_raw_ptr(), source.as_raw_ptr()],
            &[LONG_MS],
            Some([1, 1, 1]),
            Some([N, 1, 1]),
            false,
        )
    }
    .unwrap();
    let token = ctx.completion_token().unwrap();
    out.record_completion(&token);
    source.record_completion(&token);
    assert_eq!(slow.dev().producer_count(unsafe { out.as_raw_ptr() } as u64), Some(1));
    assert_eq!(out.as_slice::<f32>().unwrap(), &doubled(&input)[..]);
    assert!(token.retired());
    // The host write side: a view write waits readers too.
    source.as_host_bytes_mut().unwrap()[..4].copy_from_slice(&0f32.to_le_bytes());
}

/// A recycled LRU allocation fences on the previous owner's producers.
#[test]
fn recycled_allocations_wait_the_previous_owner() {
    let Some(slow) = Slow::new() else { return };
    let alloc: Arc<dyn Allocator> = Arc::new(crate::allocator::LruAllocator::new(Box::new((*slow.alloc).clone())));
    let spec = BufferSpec { cpu_access: false, ..BufferSpec::default() };
    let input = upload(&slow.alloc, &values(5.0));
    let first = Buffer::new(alloc.clone(), DType::Float32, vec![N], spec);
    first.ensure_allocated().unwrap();
    let recycled = unsafe { first.as_raw_ptr() };
    let ctx = slow.ctx();
    unsafe {
        ctx.dispatch(
            &slow.program,
            &[recycled, device_ptr(&input)],
            &[LONG_MS],
            Some([1, 1, 1]),
            Some([N, 1, 1]),
            false,
        )
    }
    .unwrap();
    let token = ctx.completion_token().unwrap();
    first.record_completion(&token);
    drop(first);
    let second = Buffer::new_with_zero_init(alloc, DType::Float32, vec![N], spec, true);
    second.ensure_allocated().unwrap();
    assert_eq!(unsafe { second.as_raw_ptr() }, recycled, "expected LRU reuse");
    assert!(token.retired(), "the recycled allocation was handed out while its writer was in flight");
    let mut back = vec![0u8; N * 4];
    second.copyout(&mut back).unwrap();
    assert!(back.iter().all(|byte| *byte == 0));
}

/// A storage the device knows nothing about (unregistered) falls back to
/// the context drain, so an unpublished writer on another lane is still
/// waited.
#[test]
fn unregistered_storage_falls_back_to_the_drain() {
    let Some(slow) = Slow::new() else { return };
    let input = values(9.0);
    let (source, out, other) =
        (upload(&slow.alloc, &input), upload(&slow.alloc, &vec![0.0; N]), upload(&slow.alloc, &vec![0.0; N]));
    let ctx = slow.ctx();
    slow.dispatch(&ctx, &out, &source, LONG_MS);
    // The token is published on an unrelated storage only.
    publish(slow.dev(), &ctx, &[&other]);
    slow.dev().unregister_storage(base(&out));
    assert_eq!(slow.dev().producer_count(base(&out)), None);
    assert_eq!(download(&slow.alloc, &out, N), doubled(&input));
}

/// A registered storage with no recorded producer does not wait: it is the
/// executor's contract to publish on every storage a plan touches.
#[test]
fn producers_are_kept_per_lane() {
    let Some(slow) = Slow::new() else { return };
    let program = load(slow.dev(), "vadd", &vadd_abi());
    let (a, out) = (upload(&slow.alloc, &vec![1.0; N]), upload(&slow.alloc, &vec![0.0; N]));
    let (ctx_a, ctx_b) = (slow.ctx(), slow.ctx());
    for ctx in [&ctx_a, &ctx_a, &ctx_b] {
        unsafe {
            ctx.dispatch(
                &program,
                &[device_ptr(&out), device_ptr(&a), device_ptr(&a)],
                &[],
                Some([1, 1, 1]),
                Some([N, 1, 1]),
                false,
            )
        }
        .unwrap();
        publish(slow.dev(), ctx, &[&out]);
    }
    assert_eq!(slow.dev().producer_count(base(&out)), Some(3), "one token per lane, the upload's copy lane included");
    slow.dev().wait_storage(base(&out)).unwrap();
    assert_eq!(slow.dev().producer_count(base(&out)), Some(0));
}

/// Every scoped path fails fast on a poisoned device (a private handle on
/// the same context, so the shared device stays usable).
#[test]
fn poisoned_device_fails_scoped_waits() {
    let Some(shared) = cuda_alloc_or_skip() else { return };
    let dev = Arc::new(CudaDevice::open_uncached(0).unwrap());
    let alloc = CudaAllocator { dev: Arc::clone(&dev), device_id: 0 };
    let spec = BufferSpec { cpu_access: false, ..BufferSpec::default() };
    let buffer = alloc._alloc(N * 4, &spec, true).unwrap();
    let other = alloc._alloc(N * 4, &spec, false).unwrap();
    dev.poison("simulated sticky error");
    let is_poison =
        |error: crate::Error| matches!(error, crate::Error::Runtime { ref message } if message.contains("simulated"));
    assert!(is_poison(dev.wait_storage(base(&buffer)).unwrap_err()));
    assert!(is_poison(alloc._copyout(&mut [0u8; 4], &buffer, 0).unwrap_err()));
    assert!(is_poison(alloc._copyin(&buffer, 0, &[0u8; 4]).unwrap_err()));
    assert!(is_poison(alloc._transfer(&other, 0, &buffer, 0, 4).unwrap_err()));
    assert!(is_poison(dev.zero(base(&buffer), 4).unwrap_err()));
    assert!(is_poison(alloc._alloc(4, &spec, false).unwrap_err()));
    // Frees quarantine instead of touching the dead context.
    alloc._free(buffer, &spec);
    alloc._free(other, &spec);
    assert!(!shared.dev.is_poisoned());
    let _ = shared;
}

/// A token minted but not yet recorded on its storages still covers the
/// lane: a host read of a storage with no recorded producer drains the lane
/// instead of returning early. Once the token is published, only later
/// submissions keep the lane unpublished.
#[test]
fn unrecorded_token_keeps_the_lane_covered() {
    let Some(slow) = Slow::new() else { return };
    let (input, out) = (upload(&slow.alloc, &values(3.0)), upload(&slow.alloc, &vec![0.0; N]));
    let ctx = slow.ctx();
    slow.dispatch(&ctx, &out, &input, LONG_MS);
    let token = ctx.completion_token().expect("CUDA contexts hand out tokens");
    assert_eq!(download(&slow.alloc, &out, N), doubled(&values(3.0)), "read raced the unpublished launch");
    slow.dev().record_producer(base(&out), &token);
    token.published();
    let later = upload(&slow.alloc, &vec![0.0; N]);
    slow.dispatch(&ctx, &later, &input, LONG_MS);
    assert_eq!(download(&slow.alloc, &later, N), doubled(&values(3.0)), "read raced a launch after publication");
}

/// The same for a graph: its token covers the replay until the executor
/// records it, and a profiled replay is fully retired when it returns.
#[test]
fn unrecorded_graph_token_keeps_the_lane_covered() {
    let Some(chain) = Chain::new() else { return };
    let graph = chain.capture();
    graph.replay(&[], &[]).unwrap();
    let token = graph.completion_token().expect("replays record completion");
    assert_eq!(download(&chain.alloc, &chain.out, chain.alloc_len()), chain.expected());
    chain.alloc.dev.record_producer(base(&chain.out), &token);
    token.published();
    chain.alloc._copyin(&chain.out, 0, &vec![0u8; chain.alloc_len() * 4]).unwrap();
    graph.replay_profiled(&[], &[]).unwrap();
    assert_eq!(download(&chain.alloc, &chain.out, chain.alloc_len()), chain.expected());
}

/// A copy-in below the bounce size is published as the storage's producer,
/// so a later launch on any lane is ordered after its DMA on the GPU.
#[test]
fn small_copyin_is_published_as_the_producer() {
    let Some(slow) = Slow::new() else { return };
    let spec = BufferSpec { cpu_access: false, ..BufferSpec::default() };
    let input = slow.alloc._alloc(N * 4, &spec, false).unwrap();
    slow.alloc._copyin(&input, 0, super::f32_bytes(&values(7.0))).unwrap();
    assert_eq!(slow.dev().producer_count(base(&input)), Some(1));
    let out = upload(&slow.alloc, &vec![0.0; N]);
    let ctx = slow.ctx();
    slow.dispatch(&ctx, &out, &input, 0);
    publish(slow.dev(), &ctx, &[&out, &input]);
    assert_eq!(download(&slow.alloc, &out, N), doubled(&values(7.0)));
    slow.alloc._free(input, &spec);
}
