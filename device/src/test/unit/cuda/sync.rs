use std::sync::Arc;

use super::{cuda_alloc_or_skip, cuda_device_or_skip, device_ptr, download, load, upload, vadd_abi};
use crate::cuda::{CudaPlanCtx, CudaStream};
use crate::device::{PlanContext, Program};
use crate::sync::CompletionToken;

#[test]
fn plan_context_dispatches_in_order_with_timestamps() {
    let Some(alloc) = cuda_alloc_or_skip() else { return };
    let program = load(&alloc.dev, "vadd", &vadd_abi());
    let ctx = program.new_exec_context().unwrap().expect("CUDA mints a plan context");
    const N: usize = 1 << 16;
    let ones = vec![1.0f32; N];
    let (a, mid, out) = (upload(&alloc, &ones), upload(&alloc, &vec![0.0; N]), upload(&alloc, &vec![0.0; N]));
    let mut handles = Vec::new();
    for _ in 0..4 {
        let first = unsafe {
            ctx.dispatch(
                &program,
                &[device_ptr(&mid), device_ptr(&a), device_ptr(&a)],
                &[],
                Some([N / 32, 1, 1]),
                Some([32, 1, 1]),
                true,
            )
        }
        .unwrap()
        .expect("profiled dispatch stamps");
        let second = unsafe {
            ctx.dispatch(
                &program,
                &[device_ptr(&out), device_ptr(&mid), device_ptr(&a)],
                &[],
                Some([N / 32, 1, 1]),
                Some([32, 1, 1]),
                false,
            )
        }
        .unwrap();
        assert!(second.is_none(), "unprofiled dispatch has no handle");
        handles.push(first);
    }
    let token = ctx.completion_token().expect("CUDA plan contexts hand out tokens");
    ctx.synchronize().unwrap();
    assert!(token.retired());
    token.wait(1000).unwrap();
    assert!(download(&alloc, &out, N).iter().all(|value| *value == 3.0));
    let mut previous_end = 0;
    for handle in handles {
        let (start, end) = handle.timestamps_ns().expect("completed dispatch has stamps");
        assert!(start > 0 && end >= start, "{start} {end}");
        assert!(start >= previous_end, "dispatches retire in stream order: {start} < {previous_end}");
        previous_end = end;
    }
}

#[test]
fn timestamps_are_none_until_the_dispatch_retires() {
    let Some(alloc) = cuda_alloc_or_skip() else { return };
    let program = load(&alloc.dev, "vadd", &vadd_abi());
    let ctx = CudaPlanCtx::new(Arc::clone(&alloc.dev)).unwrap();
    const N: usize = 1 << 22;
    let (a, out) = (upload(&alloc, &vec![1.0; N]), upload(&alloc, &vec![0.0; N]));
    let mut pending = 0;
    let mut handles = Vec::new();
    for _ in 0..32 {
        let handle = unsafe {
            ctx.dispatch(
                &program,
                &[device_ptr(&out), device_ptr(&a), device_ptr(&a)],
                &[],
                Some([N / 256, 1, 1]),
                Some([256, 1, 1]),
                true,
            )
        }
        .unwrap()
        .unwrap();
        pending += usize::from(handle.timestamps_ns().is_none());
        handles.push(handle);
    }
    ctx.synchronize().unwrap();
    assert!(handles.iter().all(|handle| handle.timestamps_ns().is_some()));
    // Not asserted strictly (a fast GPU may retire everything), but reported.
    eprintln!("{pending} of 32 dispatches were still in flight when queried");
}

#[test]
fn completion_token_from_a_plain_event() {
    let Some(dev) = cuda_device_or_skip() else { return };
    let stream = CudaStream::new(Arc::clone(&*dev)).unwrap();
    let token = stream.token().unwrap();
    token.wait(0).unwrap();
    assert!(token.retired());
    token.wait(5).unwrap();
}

/// Tier-4 hardware counters around a real dispatch.
///
/// `sm__warps_launched` is exactly derivable from the launch geometry, so this
/// asserts the decoded value rather than merely that something came back — a
/// mis-transcribed params struct would return a plausible-looking number.
/// Skips where counters are gated (`NVreg_RestrictProfilingToAdminUsers=1`).
#[test]
fn plan_context_captures_hardware_counters() {
    use crate::profile::{CudaCounter, PmcCounter};

    let Some(alloc) = cuda_alloc_or_skip() else { return };
    let program = load(&alloc.dev, "vadd", &vadd_abi());
    let ctx = program.new_exec_context().unwrap().expect("CUDA mints a plan context");
    if !ctx.pmc_available() {
        eprintln!("skipping CUDA PMC test: hardware counters are gated on this host");
        return;
    }

    const N: usize = 1 << 16;
    const BLOCK: usize = 32;
    let ones = vec![1.0f32; N];
    let (a, out) = (upload(&alloc, &ones), upload(&alloc, &vec![0.0; N]));

    let counters = ctx.pmc_default();
    assert!(counters.contains(&PmcCounter::Cuda(CudaCounter::SmWarpsLaunched)));
    ctx.set_pmc(&counters).unwrap();
    let handle = unsafe {
        ctx.dispatch(
            &program,
            &[device_ptr(&out), device_ptr(&a), device_ptr(&a)],
            &[],
            Some([N / BLOCK, 1, 1]),
            Some([BLOCK, 1, 1]),
            true,
        )
    }
    .unwrap()
    .expect("profiled dispatch stamps");
    ctx.synchronize().unwrap();

    let set = handle.counters().expect("armed dispatch reports counters");
    let warps = set.values[&PmcCounter::Cuda(CudaCounter::SmWarpsLaunched)];
    assert_eq!(warps, (N / 32) as u64, "one warp per 32 threads of the launch");
    for counter in [CudaCounter::SmCyclesActive, CudaCounter::SmspInstExecuted, CudaCounter::DramBytes] {
        let value = set.values[&PmcCounter::Cuda(counter)];
        assert!(value > 0, "{counter:?} came back zero for a live kernel");
    }
    // The vadd kernel issues no tensor-core work, so its tensor pipe stays idle;
    // a non-zero value there would mean the metrics are mismapped.
    assert_eq!(set.values[&PmcCounter::Cuda(CudaCounter::SmPipeTensorCyclesActive)], 0);

    // Disarming restores timing-only dispatches.
    ctx.set_pmc(&[]).unwrap();
    let plain = unsafe {
        ctx.dispatch(
            &program,
            &[device_ptr(&out), device_ptr(&a), device_ptr(&a)],
            &[],
            Some([N / BLOCK, 1, 1]),
            Some([BLOCK, 1, 1]),
            true,
        )
    }
    .unwrap()
    .expect("profiled dispatch stamps");
    ctx.synchronize().unwrap();
    assert!(plain.counters().is_none(), "a disarmed dispatch reports no counters");
    assert_eq!(download(&alloc, &out, N)[0], 2.0, "the counted kernel still computed");
}
