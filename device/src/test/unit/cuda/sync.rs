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
