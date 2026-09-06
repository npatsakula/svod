use std::sync::Arc;

use test_case::test_case;

use super::{Hardware, cuda_alloc_or_skip, device_ptr, download, load, scale_abi, upload, vadd_abi};
use crate::allocator::{Allocator, RawBuffer};
use crate::cuda::graph::alias_signature;
use crate::cuda::{CudaAllocator, CudaGraph, CudaProgram};
use crate::device::{Graph, GraphKernel, Program};

#[test_case(&[], &[]; "empty")]
#[test_case(&[1, 2, 3], &[0, 1, 2]; "distinct")]
#[test_case(&[1, 1, 2, 1], &[0, 0, 2, 0]; "aliases point at the first")]
#[test_case(&[5, 6, 5, 6], &[0, 1, 0, 1]; "interleaved")]
fn alias_signature_names_the_first_equal_slot(buffers: &[u64], expected: &[usize]) {
    assert_eq!(alias_signature(buffers), expected);
}

/// Distinct-but-renumbered bindings share a signature; a re-aliased one does not.
#[test]
fn alias_signature_ignores_addresses_but_not_aliasing() {
    assert_eq!(alias_signature(&[10, 20, 10]), alias_signature(&[7, 9, 7]));
    assert_ne!(alias_signature(&[10, 20, 10]), alias_signature(&[7, 7, 9]));
}

pub(super) const N: usize = 4096;

pub(super) struct Chain {
    pub(super) alloc: Hardware<CudaAllocator>,
    program: CudaProgram,
    a: RawBuffer,
    b: RawBuffer,
    mid1: RawBuffer,
    mid2: RawBuffer,
    pub(super) out: RawBuffer,
}

impl Chain {
    pub(super) fn new() -> Option<Self> {
        let alloc = cuda_alloc_or_skip()?;
        let program = load(&alloc.dev, "vadd", &vadd_abi());
        let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..N).map(|i| (2 * i) as f32).collect();
        let zeros = vec![0.0f32; N];
        Some(Self {
            a: upload(&alloc, &a),
            b: upload(&alloc, &b),
            mid1: upload(&alloc, &zeros),
            mid2: upload(&alloc, &zeros),
            out: upload(&alloc, &zeros),
            alloc,
            program,
        })
    }

    /// `mid1 = a + b; mid2 = mid1 + b; out = mid2 + mid1` with the hazard
    /// edges the executor would record (each kernel reads its predecessor).
    pub(super) fn kernels(&self) -> Vec<GraphKernel<'_>> {
        let launch = |buffers: Vec<*mut u8>, deps: Vec<usize>| GraphKernel {
            program: &self.program as &dyn Program,
            buffers,
            vals: vec![],
            global_size: Some([N / 32, 1, 1]),
            local_size: Some([32, 1, 1]),
            deps,
        };
        vec![
            launch(vec![device_ptr(&self.mid1), device_ptr(&self.a), device_ptr(&self.b)], vec![]),
            launch(vec![device_ptr(&self.mid2), device_ptr(&self.mid1), device_ptr(&self.b)], vec![0]),
            launch(vec![device_ptr(&self.out), device_ptr(&self.mid2), device_ptr(&self.mid1)], vec![0, 1]),
        ]
    }

    fn flattened(&self) -> Vec<u64> {
        self.kernels().iter().flat_map(|kernel| kernel.buffers.iter().map(|pointer| *pointer as u64)).collect()
    }

    /// `out[i] = (a + b) + b + (a + b) = 2a + 3b`.
    pub(super) fn expected(&self) -> Vec<f32> {
        (0..N).map(|i| (2 * i + 3 * 2 * i) as f32).collect()
    }

    pub(super) fn alloc_len(&self) -> usize {
        N
    }

    pub(super) fn capture(&self) -> Box<dyn Graph> {
        CudaGraph::capture(Arc::clone(&self.alloc.dev), &self.kernels()).unwrap().expect("CUDA chains are graphable")
    }
}

#[test]
fn captured_chain_replays_in_order() {
    let Some(chain) = Chain::new() else { return };
    let graph = chain.capture();
    graph.replay(&[], &[]).unwrap();
    assert_eq!(download(&chain.alloc, &chain.out, N), chain.expected());
    for _ in 0..3 {
        graph.replay(&chain.flattened(), &[]).unwrap();
    }
    let token = graph.completion_token().expect("replays record completion");
    token.wait(5_000).unwrap();
    assert!(token.retired());
    assert_eq!(download(&chain.alloc, &chain.out, N), chain.expected());
}

#[test]
fn replay_matches_per_call_dispatch() {
    let Some(chain) = Chain::new() else { return };
    let graph = chain.capture();
    graph.replay(&[], &[]).unwrap();
    let batched = download(&chain.alloc, &chain.out, N);
    chain.alloc._copyin(&chain.out, 0, &vec![0u8; N * 4]).unwrap();
    for kernel in chain.kernels() {
        unsafe { kernel.program.execute(&kernel.buffers, &[], kernel.global_size, kernel.local_size, false) }.unwrap();
    }
    assert_eq!(download(&chain.alloc, &chain.out, N), batched);
    assert_eq!(batched, chain.expected());
}

#[test]
fn replay_rebinds_changed_buffers() {
    let Some(chain) = Chain::new() else { return };
    let graph = chain.capture();
    graph.replay(&[], &[]).unwrap();
    // Fresh inputs (a' = 10, b' = 1) and another output keep the aliasing
    // pattern, so the DAG replays with only the changed nodes patched.
    let a2 = upload(&chain.alloc, &vec![10.0; N]);
    let b2 = upload(&chain.alloc, &vec![1.0; N]);
    let out2 = upload(&chain.alloc, &vec![0.0; N]);
    let mut buffers = chain.flattened();
    buffers[1] = device_ptr(&a2) as u64;
    buffers[2] = device_ptr(&b2) as u64;
    buffers[5] = device_ptr(&b2) as u64;
    buffers[6] = device_ptr(&out2) as u64;
    graph.replay(&buffers, &[]).unwrap();
    // mid1 = 11, mid2 = 12, out2 = 23; the original output is untouched.
    assert!(download(&chain.alloc, &out2, N).iter().all(|value| *value == 23.0));
    assert_eq!(download(&chain.alloc, &chain.out, N), chain.expected());
    // A sub-buffer view binds through its device offset.
    buffers[6] = unsafe { device_ptr(&out2).add(64 * 4) } as u64;
    graph.replay(&buffers, &[]).unwrap();
    let shifted = download(&chain.alloc, &out2, N);
    assert!(shifted.iter().all(|value| *value == 23.0));
    // Back to the captured bindings.
    graph.replay(&chain.flattened(), &[]).unwrap();
    assert_eq!(download(&chain.alloc, &chain.out, N), chain.expected());
}

/// Binding one buffer where two were captured changes the hazards the DAG
/// was built from; the replay then runs the capture-order chain, which is
/// correct for any aliasing.
#[test]
fn realiased_replay_falls_back_to_the_chain() {
    let Some(chain) = Chain::new() else { return };
    let graph = chain.capture();
    let mut buffers = chain.flattened();
    // Kernel 3 writes `out`; make kernel 2 write `out` too and kernel 3 read it:
    // mid2 := out = mid1 + b; out = out + mid1 = 2a + 3b, same expected value
    // but only if kernel 3 strictly follows kernel 2 in the replay.
    buffers[3] = device_ptr(&chain.out) as u64;
    buffers[7] = device_ptr(&chain.out) as u64;
    for _ in 0..8 {
        chain.alloc._copyin(&chain.out, 0, &vec![0u8; N * 4]).unwrap();
        graph.replay(&buffers, &[]).unwrap();
        assert_eq!(download(&chain.alloc, &chain.out, N), chain.expected());
    }
    graph.replay(&[], &[]).unwrap();
    assert_eq!(download(&chain.alloc, &chain.out, N), chain.expected());
}

#[test]
fn replay_arguments_are_validated() {
    let Some(chain) = Chain::new() else { return };
    let graph = chain.capture();
    let error = graph.replay(&chain.flattened()[..4], &[]).expect_err("buffer count");
    assert!(matches!(error, crate::Error::ProgramAbiMismatch { .. }), "{error:?}");
    let error = graph.replay(&[], &[1]).expect_err("scalars");
    assert!(matches!(error, crate::Error::ProgramAbiMismatch { .. }), "{error:?}");
    graph.replay(&chain.flattened(), &[]).unwrap();
    assert_eq!(download(&chain.alloc, &chain.out, N), chain.expected());
}

#[test]
fn profiled_replay_stamps_every_kernel() {
    let Some(chain) = Chain::new() else { return };
    let graph = chain.capture();
    let mut all_stamps = Vec::new();
    for _ in 0..2 {
        let handles = graph.replay_profiled(&[], &[]).unwrap().expect("CUDA graphs stamp dispatches");
        assert_eq!(handles.len(), 3);
        let mut previous_end = 0;
        for handle in &handles {
            let (start, end) = handle.timestamps_ns().expect("completed replay has GPU stamps");
            assert!(start > 0 && end >= start, "{start} {end}");
            assert!(start >= previous_end, "kernels retire in order: {start} < {previous_end}");
            previous_end = end;
        }
        assert_eq!(download(&chain.alloc, &chain.out, N), chain.expected());
        all_stamps.push(handles.iter().map(|h| h.timestamps_ns().unwrap()).collect::<Vec<_>>());
    }
    // Earlier handles keep their own stamps after the next profiled replay.
    assert!(all_stamps[1][0].0 >= all_stamps[0][2].1, "{all_stamps:?}");
}

/// Scalars are packed into the node's kernarg blob and patched per replay.
#[test]
fn scalar_arguments_are_captured_and_patched() {
    let Some(alloc) = cuda_alloc_or_skip() else { return };
    let program = load(&alloc.dev, "scale", &scale_abi());
    let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let (out, a_buf) = (upload(&alloc, &vec![0.0; 64]), upload(&alloc, &a));
    let kernel = GraphKernel {
        program: &program,
        buffers: vec![device_ptr(&out), device_ptr(&a_buf)],
        vals: vec![3],
        global_size: Some([2, 1, 1]),
        local_size: Some([32, 1, 1]),
        deps: vec![],
    };
    let graph = CudaGraph::capture(Arc::clone(&alloc.dev), &[kernel]).unwrap().expect("scalars are graphable");
    graph.replay(&[], &[]).unwrap();
    assert_eq!(download(&alloc, &out, 64), a.iter().map(|x| x * 3.0).collect::<Vec<_>>());
    graph.replay(&[], &[-2]).unwrap();
    assert_eq!(download(&alloc, &out, 64), a.iter().map(|x| x * -2.0).collect::<Vec<_>>());
    let handles = graph.replay_profiled(&[], &[7]).unwrap().unwrap();
    assert_eq!(handles.len(), 1);
    assert_eq!(download(&alloc, &out, 64), a.iter().map(|x| x * 7.0).collect::<Vec<_>>());
    assert!(CudaGraph::capture(Arc::clone(&alloc.dev), &[]).unwrap().is_none());
}

struct NotCuda;

impl Program for NotCuda {
    unsafe fn execute(
        &self,
        _: &[*mut u8],
        _: &[i64],
        _: Option<[usize; 3]>,
        _: Option<[usize; 3]>,
        _: bool,
    ) -> crate::Result<()> {
        Ok(())
    }

    fn name(&self) -> &str {
        "not_cuda"
    }
}

#[test]
fn foreign_programs_and_bad_deps_are_handled() {
    let Some(chain) = Chain::new() else { return };
    let kernel = GraphKernel {
        program: &NotCuda,
        buffers: vec![],
        vals: vec![],
        global_size: None,
        local_size: None,
        deps: vec![],
    };
    assert!(CudaGraph::capture(Arc::clone(&chain.alloc.dev), &[kernel]).unwrap().is_none());
    let mut kernels = chain.kernels();
    kernels[0].deps = vec![2];
    let Err(error) = CudaGraph::capture(Arc::clone(&chain.alloc.dev), &kernels) else { panic!("forward dependency") };
    assert!(format!("{error}").contains("not earlier"), "{error}");
    let mut kernels = chain.kernels();
    kernels[1].buffers.pop();
    let Err(error) = CudaGraph::capture(Arc::clone(&chain.alloc.dev), &kernels) else { panic!("arity") };
    assert!(matches!(error, crate::Error::ProgramAbiMismatch { .. }), "{error:?}");
}
