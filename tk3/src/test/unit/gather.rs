//! Row-mapped global views: the interpreter reads each row where its map
//! says and zero where the map rejects it, and the emitter keeps the row
//! decode out of the loop, fills by zero-filling `cp.async` and zeroes a
//! staged run with one select.

use std::sync::Arc;

use svod_dtype::{DType, DeviceSpec, ScalarDType, default_device::default_device};
use svod_ir::{BinaryOp as UBinary, Op, TernaryOp, UOp, ops};
use svod_tensor::Tensor;
use test_case::test_case;

use crate::atoms::{Target, sm86};
use crate::build::*;
use crate::interp::{round_to, run};
use crate::ir::*;
use crate::launch::graph_launch;
use crate::layouts::WarpGrid;
use crate::lower::{Lowering, lower};
use crate::schedule::{Prefetch, Schedule};

fn data(n: usize, seed: &mut u64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            round_to(ScalarDType::BFloat16, ((*seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0)
        })
        .collect()
}

/// Source row and validity of gathered row `i` of an `m`-row operand.
fn source(i: usize, m: usize) -> Option<usize> {
    (!i.is_multiple_of(3)).then_some(i * 7 % m)
}

/// An `[8, 8]` gathered window at column 8 of a `[16, 16]` operand: row `r`
/// is source row `(7·r) % 16` unless `r % 3 = 0`, read into registers either
/// directly or through a shared tile.
#[test_case(false; "into registers")]
#[test_case(true; "through shared memory")]
fn gathered_rows_follow_their_map_and_read_zero_outside_it(staged: bool) {
    let (m, cols) = (16usize, 16usize);
    let mut k = Kernel::new("gather");
    let x = k.param::<F32>("x", ParamKind::In, m * cols);
    let y = k.param::<F32>("y", ParamKind::Out, 8 * 8);
    let r = k.row();
    let rows = r.clone() * 7 % m * cols;
    let valid = Sc::from(0).lt(r % 3);
    let g = k.gather(x, 0, rows, Some(valid), Shape::new(8, 8), None);
    let g = k.at(g, 0, 8);
    let v = if staged {
        let s = k.smem::<F32>("s", 64);
        let t = k.smem_view::<F32>(s, 0, Shape::new(8, 8));
        k.stage(t, g, CopyMode::Sync);
        k.load(t)
    } else {
        k.load(g)
    };
    let out = k.view(y, 0, [8, 1], Shape::new(8, 8), [None, None]);
    k.store(out, v);
    let xs: Vec<f64> = (0..m * cols).map(|i| i as f64 + 1.0).collect();
    let got = run(&k.finish(), vec![xs.clone(), vec![-1.0; 64]], &[]).unwrap();
    for i in 0..8 {
        for j in 0..8 {
            let want = source(i, m).map_or(0.0, |s| xs[s * cols + 8 + j]);
            assert_eq!(got[1][i * 8 + j], want, "y[{i}, {j}]");
        }
    }
}

/// A store through a gathered view writes the mapped rows and drops the rest.
#[test]
fn gathered_stores_drop_rejected_rows() {
    let m = 16usize;
    let mut k = Kernel::new("scatter");
    let x = k.param::<F32>("x", ParamKind::In, m * 8);
    let y = k.param::<F32>("y", ParamKind::Out, m * 8);
    let src = k.view(x, 0, [8, 1], Shape::new(m, 8), [None, None]);
    let v = k.load(src);
    let r = k.row();
    let rows = r.clone() * 7 % m * 8;
    let valid = Sc::from(0).lt(r % 3);
    let dst = k.gather(y, 0, rows, Some(valid), Shape::new(m, 8), None);
    k.store(dst, v);
    let xs: Vec<f64> = (0..m * 8).map(|i| i as f64).collect();
    let got = run(&k.finish(), vec![xs.clone(), vec![-1.0; m * 8]], &[]).unwrap();
    for i in 0..m {
        for j in 0..8 {
            let kept = (0..m).find(|&r| source(r, m) == Some(i));
            let want = kept.map_or(-1.0, |r| xs[r * 8 + j]);
            assert_eq!(got[1][i * 8 + j], want, "y[{i}, {j}]");
        }
    }
}

#[test]
#[should_panic(expected = "moves along its columns only")]
fn a_gathered_view_does_not_move_rows() {
    let mut k = Kernel::new("gather");
    let x = k.param::<F32>("x", ParamKind::In, 64);
    let r = k.row();
    let g = k.gather(x, 0, r * 8, None, Shape::new(8, 8), None);
    k.at(g, 1, 0);
}

/// `c = gather(a) · bᵀ` with gathered row `i` = source row `(7·i) % m` or
/// zero when `i % 3 = 0`: the A operand of an implicit GEMM, decode and
/// padding included, through the pipelined mainloop.
fn gathered_gemm(m: usize, n: usize, kk: usize, tile: [usize; 3], stages: usize, warps: [u32; 2]) -> Program {
    let [bm, bn, bk] = tile;
    let mut k = Kernel::new("gathered_gemm");
    let a = k.param::<BF16>("a", ParamKind::In, m * kk);
    let b = k.param::<BF16>("b", ParamKind::In, n * kk);
    let c = k.param::<F32>("c", ParamKind::Out, m * n);
    k.grid([Sc::from(m / bm), Sc::from(n / bn), Sc::from(1)]);
    k.warps(warps[0] * warps[1]);
    let (row0, col0) = (k.block(0) * bm, k.block(1) * bn);
    let i = row0.clone() + k.row();
    let start = i.clone() * 7 % m * kk;
    let valid = Sc::from(0).lt(i % 3);
    let a_s = k.smem::<BF16>("a_s", stages * bm * bk);
    let b_s = k.smem::<BF16>("b_s", stages * bn * bk);
    let b_view = k.view(b, 0, [kk, 1], Shape::new(bn, bk), [None, None]);
    let b_view = k.at(b_view, col0.clone(), 0);
    let init = k.zeros::<F32>(Shape::new(bm, bn));
    let [acc] = k.pipeline(
        kk / bk,
        stages,
        [init],
        |k, step, slot| {
            let a_g = k.gather(a, step.clone() * bk, start.clone(), Some(valid.clone()), Shape::new(bm, bk), None);
            let a_t = k.smem_slot::<BF16>(a_s, slot.clone(), Shape::new(bm, bk));
            k.stage(a_t, a_g, CopyMode::Async);
            let b_g = k.at(b_view, 0, step * bk);
            let b_t = k.smem_slot::<BF16>(b_s, slot, Shape::new(bn, bk));
            k.stage(b_t, b_g, CopyMode::Async);
        },
        |k, _step, slot, [acc]| {
            let a_t = k.smem_slot::<BF16>(a_s, slot.clone(), Shape::new(bm, bk));
            let b_t = k.smem_slot::<BF16>(b_s, slot, Shape::new(bn, bk));
            [k.mma(acc, a_t, false, b_t, true)]
        },
    );
    let out = k.view(c, 0, [n, 1], Shape::new(bm, bn), [None, None]);
    let out = k.at(out, row0, col0);
    k.store(out, acc);
    k.finish()
}

fn reference(a: &[f64], b: &[f64], m: usize, n: usize, kk: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        let Some(s) = source(i, m) else { continue };
        for j in 0..n {
            c[i * n + j] = (0..kk).map(|t| a[s * kk + t] * b[j * kk + t]).sum();
        }
    }
    c
}

#[test]
fn gathered_gemm_matches_the_reference_in_the_interpreter() {
    let (m, n, kk) = (64usize, 32usize, 96usize);
    let mut seed = 5;
    let (a, b) = (data(m * kk, &mut seed), data(n * kk, &mut seed));
    let prog = gathered_gemm(m, n, kk, [32, 32, 32], 2, [2, 2]);
    let got = run(&prog, vec![a.clone(), b.clone(), vec![0.0; m * n]], &[]).unwrap();
    for (i, (g, w)) in got[2].iter().zip(reference(&a, &b, m, n, kk)).enumerate() {
        assert!((g - w).abs() <= 1e-4 * w.abs().max(1.0), "c[{}, {}] = {g}, want {w}", i / n, i % n);
    }
}

fn lowering(target: Target, prefetch: Prefetch, warps: [u32; 2]) -> Lowering {
    Lowering {
        target,
        schedule: Schedule::Uniform { prefetch, unroll: true },
        grid: WarpGrid { rows: warps[0], cols: warps[1] },
        swizzle: true,
    }
}

/// The emitted instruction list of `prog` on sm_86.
fn emitted(prog: Program, prefetch: Prefetch, warps: [u32; 2]) -> Vec<Arc<UOp>> {
    let params =
        prog.params.iter().enumerate().map(|(i, p)| UOp::param(i, p.elems, DType::Scalar(p.dtype), None)).collect();
    let lowered = lower(prog, &lowering(sm86(), prefetch, warps), params, DeviceSpec::Cuda { device_id: 0 }).unwrap();
    let Op::Program(ops::Program { linear: Some(linear), .. }) = lowered.program.op() else {
        panic!("a linear program")
    };
    let Op::Linear(ops::Linear { ops }) = linear.op() else { panic!("a linear list") };
    ops.to_vec()
}

/// The body of the first loop: the unrolled main loop of the pipeline.
fn loop_body(list: &[Arc<UOp>]) -> &[Arc<UOp>] {
    let start = list.iter().position(|u| matches!(u.op(), Op::Range(..))).expect("a loop");
    let end = start + list[start..].iter().position(|u| matches!(u.op(), Op::End(..))).expect("a loop end");
    &list[start..end]
}

/// Whether `u` depends on the thread index.
fn per_thread(u: &Arc<UOp>) -> bool {
    u.toposort().iter().any(|s| matches!(s.op(), Op::Special(ops::Special { name, .. }) if name == "lidx0"))
}

const PIN: ([usize; 3], usize, [u32; 2]) = ([64, 64, 32], 2, [2, 2]);

/// cp.async: every A chunk is one zero-filling copy, B keeps the plain
/// form, and no per-thread division or remainder (the row decode) is left
/// inside the loop.
#[test]
fn the_row_decode_is_hoisted_and_padding_rows_zero_fill() {
    let (tile, stages, warps) = PIN;
    let list = emitted(gathered_gemm(128, 64, 224, tile, stages, warps), Prefetch::CpAsync, warps);
    let body = loop_body(&list);
    let threads = (warps[0] * warps[1] * 32) as usize;
    let copies = |suffix: &str| {
        body.iter()
            .filter(|u| matches!(u.op(), Op::Custom(ops::Custom { code, .. }) if code.contains(&format!("shared.global.16{suffix}(ptr"))))
            .count()
    };
    let per_step = |rows: usize| rows * tile[2] / 8 / threads;
    assert_eq!(copies(".s"), per_step(tile[0]) * stages, "zero-filling A copies per unrolled loop");
    assert_eq!(copies(""), per_step(tile[1]) * stages, "plain B copies per unrolled loop");
    let decode = |ops: &[Arc<UOp>]| -> Vec<Arc<UOp>> {
        ops.iter()
            .filter(|u| {
                matches!(u.op(), Op::Binary(UBinary::CDiv | UBinary::CMod | UBinary::FloorDiv | UBinary::FloorMod, ..))
            })
            .filter(|u| per_thread(u))
            .cloned()
            .collect()
    };
    let start = list.iter().position(|u| matches!(u.op(), Op::Range(..))).unwrap();
    assert!(!decode(&list[..start]).is_empty(), "the row decode sits in the prologue");
    assert!(decode(body).is_empty(), "per-thread division in the loop: {:?}", decode(body));
}

/// Register staging: every gathered run is an ungated load from a clamped
/// address zeroed by one select, never a gated load.
#[test]
fn a_staged_gathered_run_is_one_select() {
    let (tile, _, warps) = PIN;
    let list = emitted(gathered_gemm(128, 64, 224, tile, 2, warps), Prefetch::RegisterStaged, warps);
    let body = loop_body(&list);
    let reads_a = |u: &Arc<UOp>| {
        u.toposort().iter().any(|s| matches!(s.op(), Op::Param(ops::Param { arg, .. }) if arg.slot == 0))
    };
    let loads: Vec<_> = body.iter().filter(|u| matches!(u.op(), Op::Load(..)) && reads_a(u)).collect();
    assert!(!loads.is_empty(), "the A operand is loaded in the loop");
    assert!(loads.iter().all(|u| matches!(u.op(), Op::Load(ops::Load { gate: None, .. }))), "no gated loads");
    let selected = |load: &Arc<UOp>| {
        body.iter().filter(|u| matches!(u.op(), Op::Ternary(TernaryOp::Where, _, v, _) if Arc::ptr_eq(v, load))).count()
    };
    assert!(loads.iter().all(|l| selected(l) == 1), "one select per loaded run");
}

/// On the attached GPU the gathered GEMM computes what the interpreter does,
/// through the zero-filling copies and through the staged selects.
#[test_case(Prefetch::CpAsync; "cp.async")]
#[test_case(Prefetch::RegisterStaged; "register staged")]
fn gathered_gemm_matches_the_interpreter_on_the_gpu(prefetch: Prefetch) {
    let device = default_device();
    let Some(target) = matches!(device, DeviceSpec::Cuda { .. }).then(|| Target::for_device(&device)).flatten() else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    let (m, n, kk) = (128usize, 64usize, 256usize);
    let (tile, stages, warps) = PIN;
    let mut seed = 9;
    let (a, b) = (data(m * kk, &mut seed), data(n * kk, &mut seed));
    let prog = gathered_gemm(m, n, kk, tile, stages, warps);
    let want = run(&prog, vec![a.clone(), b.clone(), vec![0.0; m * n]], &[]).unwrap();
    let tensor = |v: &[f64]| Tensor::from_slice(v.iter().map(|&x| x as f32).collect::<Vec<_>>()).cast(DType::BFloat16);
    let (a_t, b_t, c_t) = (tensor(&a), tensor(&b), Tensor::empty(&[m * n], DType::Float32));
    let got = graph_launch(prog, &lowering(target, prefetch, warps), &[&a_t, &b_t, &c_t]).unwrap();
    let got = got.to_vec::<f32>().unwrap();
    for (i, (g, w)) in got.iter().zip(&want[2]).enumerate() {
        assert!((*g as f64 - w).abs() <= 1e-3 * w.abs().max(1.0), "c[{}, {}] = {g}, want {w}", i / n, i % n);
    }
}
