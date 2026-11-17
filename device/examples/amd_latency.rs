//! Host-side latency of AMD submissions on device 0, each measured best of N.
//!
//!     cargo run --release -p svod-device --example amd_latency -- <command>
//!
//! Commands:
//!   dispatch [threads]  round trip (dispatch + drain) and asynchronous dispatch
//!                       cost, from 1 up to `threads` threads (default: the lane
//!                       count), one exec context, so one lane, per thread
//!   graph-wait          a long graph's waiting-thread CPU against its wall time,
//!                       then a short graph's replay-to-wake latency
//!
//! Needs a KFD-visible AMD GPU and a host clang with the AMDGPU target.

use std::sync::Arc;
use std::time::{Duration, Instant};

use svod_device::allocator::{Allocator, BufferSpec, RawBuffer};
use svod_device::amd::{AmdAllocator, AmdGraph, AmdProgram, SignalPool};
use svod_device::device::{AbiParamDescriptor, AbiParamKind, GraphKernel, Program};

const USAGE: &str = "usage: amd_latency dispatch [threads] | amd_latency graph-wait";

/// A kernel that stores 0.0 to `buf0[tid]`.
const STORE_ZERO_IR: &str = r#"target triple = "amdgcn-amd-amdhsa"
declare i32 @llvm.amdgcn.workitem.id.x()
define amdgpu_kernel void @store_zero(ptr noalias %buf0) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tid_ext = zext i32 %tid to i64
  %p = getelementptr inbounds float, ptr %buf0, i64 %tid_ext
  store float 0.0, ptr %p
  ret void
}
attributes #0 = { alwaysinline nounwind "amdgpu-flat-work-group-size"="1,64" }
"#;

/// One work-item that sleeps `iterations` times (`s_sleep 127`, a few µs
/// each), then stores 0.0.
fn sleepy_ir(iterations: u32) -> String {
    format!(
        r#"target triple = "amdgcn-amd-amdhsa"
declare void @llvm.amdgcn.s.sleep(i32)
define amdgpu_kernel void @sleepy(ptr noalias %buf0) #0 {{
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  call void @llvm.amdgcn.s.sleep(i32 127)
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, {iterations}
  br i1 %done, label %exit, label %loop
exit:
  store float 0.0, ptr %buf0
  ret void
}}
attributes #0 = {{ nounwind "amdgpu-flat-work-group-size"="1,1" }}
"#
    )
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let command = args.first().map(String::as_str);
    if !matches!(command, Some("dispatch" | "graph-wait")) {
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
    let alloc = match AmdAllocator::new(0) {
        Ok(alloc) => alloc,
        Err(error) => {
            eprintln!("no AMD device: {error}");
            std::process::exit(1);
        }
    };
    let core = alloc.dev.core();
    if core.signal_pool().is_none() {
        core.install_signal_pool(SignalPool::new(&alloc, 64).expect("signal pool"));
    }
    match command {
        Some("dispatch") => {
            let threads = args.get(1).map_or(core.hw_queues(), |n| n.parse().expect("a thread count"));
            dispatch(&alloc, threads);
        }
        _ => graph_wait(&alloc),
    }
}

/// Run `f` `rounds` times and return its results sorted, the best first.
fn best_of<T: Ord>(rounds: usize, f: impl FnMut() -> T) -> Vec<T> {
    let mut samples = std::iter::repeat_with(f).take(rounds).collect::<Vec<_>>();
    samples.sort();
    samples
}

fn summary(samples: &[Duration]) -> String {
    let at = |fraction: f64| samples[((samples.len() - 1) as f64 * fraction) as usize];
    format!("best {:?}, median {:?}, p90 {:?}, max {:?}", samples[0], at(0.5), at(0.9), at(1.0))
}

/// Compile `ir` for the device with host clang.
fn compile(alloc: &AmdAllocator, ir: &str) -> Vec<u8> {
    use std::io::Write;
    let child = std::process::Command::new("clang")
        .args(["-x", "ir", "-c", "-O2", "--target=amdgcn-amd-amdhsa"])
        .arg(format!("-mcpu={}", alloc.dev.arch.mcpu()))
        .args(["-mcumode", "-nogpulib", "-nogpuinc", "-Wno-override-module", "-", "-o", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("host clang");
    child.stdin.as_ref().expect("clang stdin").write_all(ir.as_bytes()).expect("feed clang");
    let out = child.wait_with_output().expect("clang");
    assert!(out.status.success() && out.stdout.starts_with(b"\x7fELF"), "clang has no AMDGPU target");
    out.stdout
}

/// Load the one-buffer kernel `name` from `ir`.
fn load(alloc: &AmdAllocator, ir: &str, name: &str) -> AmdProgram {
    let abi = [AbiParamDescriptor {
        slot: 0,
        kind: AbiParamKind::Storage(svod_dtype::AddrSpace::Global),
        dtype: svod_dtype::DType::Float32,
        name: None,
    }];
    AmdProgram::load(Arc::clone(&alloc.dev), alloc, &compile(alloc, ir), name, &abi).expect("program load")
}

/// A host-visible 64-byte output buffer: its raw handle and GPU address.
fn output(alloc: &AmdAllocator) -> (RawBuffer, *mut u8) {
    let raw = alloc.alloc_uncached(64).expect("output buffer");
    let RawBuffer::AmdDevice { gpu_addr, .. } = &raw else { unreachable!("an AMD allocation") };
    let gpu = *gpu_addr as *mut u8;
    (raw, gpu)
}

/// Per thread count: the round trip of one dispatch and its drain, and the cost
/// of one asynchronous dispatch in a burst drained once.
fn dispatch(alloc: &AmdAllocator, threads: usize) {
    const ROUNDS: usize = 1000;
    const BURST: usize = 64;
    let program = load(alloc, STORE_ZERO_IR, "store_zero");
    let outputs = (0..threads.max(1)).map(|_| output(alloc)).collect::<Vec<_>>();
    println!("{} lanes", alloc.dev.core().hw_queues());
    for count in 1..=outputs.len() {
        let results = std::thread::scope(|scope| {
            let workers = outputs[..count]
                .iter()
                .map(|&(_, gpu)| {
                    let gpu = gpu as usize;
                    let program = &program;
                    scope.spawn(move || {
                        let context = program.new_exec_context().expect("context").expect("an AMD context");
                        let launch = || {
                            let one = Some([1, 1, 1]);
                            unsafe { context.dispatch(program, &[gpu as *mut u8], &[], one, one, false) }
                                .expect("dispatch");
                        };
                        let round_trip = best_of(ROUNDS, || {
                            let start = Instant::now();
                            launch();
                            context.synchronize().expect("drain");
                            start.elapsed()
                        });
                        let burst = best_of(ROUNDS / BURST, || {
                            let start = Instant::now();
                            (0..BURST).for_each(|_| launch());
                            let issued = start.elapsed() / BURST as u32;
                            context.synchronize().expect("drain");
                            issued
                        });
                        (round_trip, burst)
                    })
                })
                .collect::<Vec<_>>();
            workers.into_iter().map(|worker| worker.join().expect("worker")).collect::<Vec<_>>()
        });
        for (thread, (round_trip, burst)) in results.iter().enumerate() {
            println!("{count} threads, #{thread}: round trip {}", summary(round_trip));
            println!("{count} threads, #{thread}: async dispatch {}", summary(burst));
        }
    }
    for (raw, _) in outputs {
        alloc._free(raw, &BufferSpec::default());
    }
}

/// CPU time the calling thread has used.
fn thread_cpu() -> Duration {
    // SAFETY: getrusage writes the struct it is given.
    let usage = unsafe {
        let mut usage = std::mem::zeroed::<libc::rusage>();
        assert_eq!(libc::getrusage(libc::RUSAGE_THREAD, &mut usage), 0);
        usage
    };
    let time = |t: libc::timeval| Duration::new(t.tv_sec as u64, t.tv_usec as u32 * 1000);
    time(usage.ru_utime) + time(usage.ru_stime)
}

/// A wait that sleeps uses a sliver of a core; one kept polling by progress
/// (an AQL queue's read index) uses the whole wait. A short graph's wake
/// latency is what a wait's spin budget trades against that CPU.
fn graph_wait(alloc: &AmdAllocator) {
    let core = alloc.dev.core();
    core.set_pm4_graph(true);
    let (raw, gpu) = output(alloc);
    let graph = |iterations: u32, count: usize| {
        let program = load(alloc, &sleepy_ir(iterations), "sleepy");
        let kernels = (0..count)
            .map(|index| GraphKernel {
                program: &program as &dyn Program,
                buffers: vec![gpu],
                vals: Vec::new(),
                global_size: Some([1, 1, 1]),
                local_size: Some([1, 1, 1]),
                deps: Vec::from_iter(index.checked_sub(1)),
            })
            .collect::<Vec<_>>();
        let graph = AmdGraph::capture(alloc, &kernels).expect("capture").expect("a captured graph");
        (graph, program)
    };
    let replay = |graph: &dyn svod_device::device::Graph| {
        graph.replay(&[], &[]).expect("replay");
        core.synchronize_all().expect("synchronize_all");
    };

    let (long, _program) = graph(800, 400);
    let trials = best_of(3, || {
        let (wall, cpu) = (Instant::now(), thread_cpu());
        replay(&*long);
        (wall.elapsed(), thread_cpu() - cpu)
    });
    for (wall, cpu) in trials {
        println!(
            "long wait: wall {wall:?}, waiter CPU {cpu:?} ({:.1} %)",
            100.0 * cpu.as_secs_f64() / wall.as_secs_f64()
        );
    }

    for iterations in [1, 30, 300] {
        let (short, _program) = graph(iterations, 1);
        let timestamps = short.replay_profiled(&[], &[]).expect("profiled replay").expect("profile support");
        core.synchronize_all().expect("synchronize_all");
        let (start, end) = timestamps[0].timestamps_ns().expect("graph GPU timestamps");
        let span = Duration::from_nanos(end - start);
        let walls = best_of(300, || {
            let wall = Instant::now();
            replay(&*short);
            wall.elapsed()
        });
        println!("short wait, GPU {span:?}: {}", summary(&walls));
    }
    drop(long);
    alloc._free(raw, &BufferSpec::default());
}
