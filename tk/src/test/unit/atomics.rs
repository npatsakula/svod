//! The atomic add: its numbers, what the lowering refuses, and what it
//! compiles to. What the builder's types refuse is in its `compile_fail`
//! examples ([`Kernel::accum`], [`Kernel::gather_accum`],
//! [`Kernel::atomic_add`]).

use proptest::prelude::*;
use svod_dtype::GpuArch;
use svod_tensor::Tensor;
use test_case::test_case;

use crate::atoms::Target;
use crate::build::{Added, F32, I32, Kernel, Regs, Sc, Tile};
use crate::interp;
use crate::ir::*;
use crate::kernels::scatter::{ScatterSpec, TILE, scatter_add};
use crate::launch::{self, graph_launch};
use crate::lower::{self, Lowering};
use crate::sweep::{self, declared};

fn lowering(target: Target) -> Lowering {
    Lowering::new(&declared(target))
}

fn target(name: &str) -> Target {
    sweep::targets().into_iter().find(|(n, _)| *n == name).expect("a swept target").1
}

/// The sum a host adds in row order, wrapping.
fn reference(spec: &ScatterSpec, target: &[f64], index: &[f64], values: &[f64]) -> Vec<f64> {
    let mut out = target.to_vec();
    for (p, &at) in index.iter().enumerate() {
        if at >= 0.0 && (at as usize) < spec.rows {
            for c in 0..spec.d {
                let slot = at as usize * spec.d + c;
                out[slot] = f64::from((out[slot] as i32).wrapping_add(values[p * spec.d + c] as i32));
            }
        }
    }
    out
}

proptest! {
    /// Any indices, colliding or out of range, on either side of the
    /// lowering: exactly the host's sum over what the target held, on a
    /// block's worth or several, and a padded last tile adds nothing twice.
    #[test]
    fn an_i32_scatter_is_the_hosts_sum(
        (rows, d, index, seed) in (1usize..40, 1usize..4, 1usize..3 * TILE)
            .prop_flat_map(|(rows, d, pairs)| {
                (Just(rows), Just(d), prop::collection::vec(-2i32..rows as i32 + 2, pairs), any::<i32>())
            }),
        target_name in prop::sample::select(vec!["sm_86", "gfx1201", "gfx942"]),
    ) {
        let spec = ScatterSpec { rows, pairs: index.len(), d };
        let mix = |i: usize, salt: i32| f64::from((i as i32).wrapping_mul(0x9E37_79B1u32 as i32).wrapping_add(seed ^ salt));
        let held: Vec<f64> = (0..rows * d).map(|i| mix(i, 1)).collect();
        let values: Vec<f64> = (0..index.len() * d).map(|i| mix(i, 2)).collect();
        let index: Vec<f64> = index.iter().map(|&i| f64::from(i)).collect();
        let want = reference(&spec, &held, &index, &values);
        let params = vec![held, index.clone(), values.clone()];
        let prog = scatter_add::<I32>(&spec);
        let before = interp::run(&prog, params.clone(), &[]).unwrap();
        prop_assert_eq!(&before[0], &want);
        let low = lower::standalone(prog, &lowering(target(target_name))).unwrap();
        let after = interp::run(&low.tile, params, &[]).unwrap();
        prop_assert_eq!(&after[0], &want);
        prop_assert_eq!(&after[1], &index, "an input is untouched");
        prop_assert_eq!(&after[2], &values, "an input is untouched");
    }
}

/// An `f32` sum in any order stays within the rounding its terms allow.
#[test]
fn an_f32_scatter_is_within_its_rounding() {
    let spec = ScatterSpec { rows: 3, pairs: 600, d: 1 };
    let index: Vec<f64> = (0..600).map(|i| f64::from(i % 3)).collect();
    let values: Vec<f64> = (0..600).map(|i| f64::from(i as f32 * 0.1)).collect();
    let out = interp::run(&scatter_add::<F32>(&spec), vec![vec![1.0; 3], index, values.clone()], &[]).unwrap();
    let exact: Vec<f64> = (0..3).map(|r| 1.0 + values.iter().skip(r).step_by(3).sum::<f64>()).collect();
    super::assert_rel_close("an f32 sum", &out[0], &exact, 600.0 * f64::from(f32::EPSILON));
}

/// A program of one add of `shape`, `body` given the kernel, its target and
/// a tile to add.
fn one_add(
    shape: Shape,
    warps: u32,
    body: impl for<'k> FnOnce(&mut Kernel<'k>, Tile<'k, Added, I32>, Regs<'k, I32>),
) -> Program {
    Kernel::record("one_add", |k| {
        k.warps(warps);
        let p = k.accum::<I32>("target", shape.elems());
        let row = k.row();
        let dst = k.gather_accum(p, 0, row, Sc::from(1), shape);
        let v = k.fill::<I32>(shape, Const::Int(1));
        body(k, dst, v);
    })
}

fn refusal(prog: Program, on: Target) -> String {
    match lower::standalone(prog, &lowering(on)) {
        Err(lower::Error::Unsupported { what }) => what,
        Err(other) => panic!("{other}"),
        Ok(_) => panic!("lowered"),
    }
}

/// Every update lands once, from every warp: a layout replicating an element,
/// a warp's own add and one under a branch on the warp are refused.
#[test_case("sm_86"; "wave32")]
#[test_case("gfx942"; "wave64")]
fn an_add_lands_once(on: &str) {
    let tile = Shape::new(TILE, 1);
    let plain = one_add(tile, 4, |k, dst, v| k.atomic_add(dst, v));
    lower::standalone(plain, &lowering(target(on))).unwrap();

    let small = one_add(Shape::new(1, 1), 4, |k, dst, v| k.atomic_add(dst, v));
    assert!(refusal(small, target(on)).contains("replicates"));

    let warp = one_add(tile, 4, |k, dst, v| {
        let first = k.warp().eq(0);
        k.if_(first, |k| k.atomic_add(dst, v), |_| {});
    });
    assert!(refusal(warp, target(on)).contains("skip"), "a branch on the warp");

    let own = one_add(tile, 4, |k, dst, v| k.each_warp(|k| k.atomic_add(dst, v)));
    assert!(refusal(own, target(on)).contains("skip"), "a warp's own add");
}

/// A target without the add refuses it before anything is rendered.
#[test]
fn a_target_without_the_add_refuses_it() {
    let mut t = target("sm_86");
    t.atomic_add = false;
    let spec = ScatterSpec { rows: 4, pairs: 8, d: 1 };
    assert!(refusal(scatter_add::<I32>(&spec), t.clone()).contains("atomic add on"));
    assert!(refusal(scatter_add::<F32>(&spec), t).contains("atomic add on"));
}

#[test]
fn the_oracle_gates_the_add() {
    let t = target("gfx1201");
    assert!(!crate::oracle::flipped(&t, &[("atomic_add", Some(false))]).atomic_add);
    let absent = crate::oracle::flipped(&t, &[("atomic_add", None)]);
    assert_eq!(absent.atomic_add, t.atomic_add, "no toolchain is no verdict");
}

/// An accumulator shares no storage with another parameter: the same tensor
/// twice and two views of one buffer are refused before any lowering.
#[test]
fn a_launch_keeps_an_accumulator_to_itself() {
    let spec = ScatterSpec { rows: 4, pairs: 4, d: 1 };
    let low = lowering(target("sm_86"));
    let launch = |t: &Tensor, i: &Tensor, v: &Tensor| graph_launch(scatter_add::<I32>(&spec), &low, &[t, i, v]);
    let (buffer, index) = (Tensor::from_slice([0i32; 4]), Tensor::from_slice([0i32; 4]));
    buffer.realize().unwrap();
    let aliased = |r: launch::Result<Tensor>| matches!(r, Err(launch::Error::Aliased { .. }));
    let buffer4 = buffer.try_reshape([4, 1]).unwrap();
    assert!(aliased(launch(&buffer4, &index, &buffer)), "the same buffer");
    let view = buffer.try_reshape([2, 2]).unwrap();
    assert!(aliased(launch(&buffer4, &index, &view)), "a reshape");
}

/// The text states the ordering of every add, the statement with no result,
/// and claims nothing of the accumulator it cannot: it is not `readonly`.
#[test_case("sm_86", "syncscope(\"device\") monotonic"; "sm_86")]
#[test_case("gfx1201", "syncscope(\"agent\") monotonic, !amdgpu.no.fine.grained.memory !{}"; "gfx1201")]
#[test_case("gfx942", "syncscope(\"agent\") monotonic, !amdgpu.no.fine.grained.memory !{}"; "gfx942")]
fn the_text_adds_with_its_ordering(on: &str, ordering: &str) {
    let prog = scatter_add::<I32>(&ScatterSpec { rows: 8, pairs: 512, d: 2 });
    let ir = lower::render(prog, &lowering(target(on))).unwrap();
    assert!(ir.contains("ptr noalias align 32 %data0"), "{ir}");
    let add = ir.lines().find(|l| l.contains("atomicrmw")).expect("an atomic add");
    assert!(add.trim_start().starts_with("atomicrmw add ptr"), "the result goes unused: {add}");
    assert!(add.ends_with(ordering), "{add}");
}

/// No fence and no barrier: dispatch hazards make an add visible, and a
/// kernel with an add has as many as one without.
#[test]
fn an_add_adds_no_synchronization() {
    let prog = scatter_add::<I32>(&ScatterSpec { rows: 8, pairs: 512, d: 2 });
    let tile = lower::standalone(prog, &lowering(target("gfx1201"))).unwrap().tile;
    assert_eq!(tile.walk().filter(|(_, s)| matches!(s, Stmt::Sync(_))).count(), 0);
}

/// What the backend makes of it: the instruction itself, never a
/// compare-and-swap loop, for either type on every family. SASS was read by
/// hand on sm_86: `RED.E.ADD`, the add with no result.
#[test_case("sm_86"; "sm_86")]
#[test_case("sm_90"; "sm_90")]
#[test_case("gfx1151"; "gfx1151")]
#[test_case("gfx1201"; "gfx1201")]
#[test_case("gfx942"; "gfx942")]
#[test_case("gfx950"; "gfx950")]
fn an_add_compiles_to_the_instruction(on: &str) {
    let t = target(on);
    let spec = ScatterSpec { rows: 8, pairs: 512, d: 2 };
    for (ir, want) in [
        (
            lower::render(scatter_add::<I32>(&spec), &lowering(t.clone())).unwrap(),
            [".global.add.u32", "global_atomic_add"],
        ),
        (
            lower::render(scatter_add::<F32>(&spec), &lowering(t.clone())).unwrap(),
            [".global.add.f32", "global_atomic_add_f32"],
        ),
    ] {
        let (asm, want) = match t.arch {
            GpuArch::Cuda(arch) => (svod_runtime::cuda::compile_ir_to_ptx(&ir, arch), want[0]),
            GpuArch::Amd(arch) => (svod_runtime::amd::compile_ir_to_amd_asm(&ir, arch), want[1]),
            GpuArch::Metal(_) => unreachable!("a swept target"),
        };
        let asm = match asm {
            Ok(asm) => String::from_utf8(asm).unwrap(),
            Err(e) if e.to_string().contains("target") || e.to_string().contains("too old") => {
                eprintln!("skipped: {e}");
                return;
            }
            Err(e) => panic!("{e}"),
        };
        assert!(asm.contains(want), "{on}: no {want}");
        assert!(!asm.contains("cmpswap") && !asm.contains(".global.cas"), "{on}: a compare-and-swap loop");
    }
}

fn on_cpu<R>(f: impl FnOnce() -> R) -> R {
    svod_dtype::default_device::with_default_device(svod_dtype::DeviceSpec::Cpu, f)
}

/// The host's sum, launches building on one another on one buffer, an index
/// outside it dropped, what it held kept; values computed from the target read
/// it before the add.
fn the_op_sums_and_chains() {
    let target = Tensor::from_slice([1i32, 2, 3, 4, 5, 6]).try_reshape([3, 2]).unwrap();
    target.realize().unwrap();
    let index = Tensor::from_slice([2i32, 0, 2, -1, 3]);
    let values = Tensor::from_slice([10i32, 20, 30, 40, 50, 60, 70, 80, 90, 100]).try_reshape([5, 2]).unwrap();
    let once = crate::ops::scatter_add(target, &index, &values).unwrap();
    let twice = crate::ops::scatter_add(once, &index, &values).unwrap();
    twice.realize().unwrap();
    assert_eq!(twice.as_vec::<i32>().unwrap(), vec![61, 82, 3, 4, 125, 166]);

    let target = Tensor::from_slice([1i32, 2, 3, 4]).try_reshape([4, 1]).unwrap();
    target.realize().unwrap();
    let index = Tensor::from_slice([1i32, 1, 0, 3]);
    let doubled = target.try_add(&target).unwrap();
    let out = crate::ops::scatter_add(target, &index, &doubled).unwrap();
    out.realize().unwrap();
    assert_eq!(out.as_vec::<i32>().unwrap(), vec![7, 8, 3, 12]);
}

#[test]
fn the_op_sums_and_chains_on_the_cpu() {
    on_cpu(the_op_sums_and_chains);
}

/// The op refuses what no kernel holds: shapes, types, a lazy target, a
/// target sharing its buffer with the values.
#[test]
fn the_op_refuses_what_it_cannot_add() {
    on_cpu(|| {
        let target = || {
            let t = Tensor::from_slice([0i32; 4]).try_reshape([2, 2]).unwrap();
            t.realize().unwrap();
            t
        };
        let index = Tensor::from_slice([0i32, 1]);
        let rows = |n: usize| Tensor::from_slice(vec![1i32; n * 2]).try_reshape([n, 2]).unwrap();
        let add = crate::ops::scatter_add;
        assert!(matches!(add(target(), &index, &rows(3)), Err(crate::ops::Error::Shape { .. })));
        let floats = Tensor::from_slice([1f32; 4]).try_reshape([2, 2]).unwrap();
        assert!(matches!(add(target(), &index, &floats), Err(crate::ops::Error::Dtype { .. })));
        let wide = Tensor::from_slice([0i64, 1]);
        assert!(matches!(add(target(), &wide, &rows(2)), Err(crate::ops::Error::Dtype { .. })));
        let lazy = Tensor::zeros(&[2, 2], svod_dtype::DType::Int32);
        assert!(matches!(add(lazy, &index, &rows(2)), Err(crate::ops::Error::Unrealized { .. })));
        let shared = target();
        assert!(matches!(add(shared.clone(), &index, &shared), Err(crate::ops::Error::Launch { .. })));
    });
}

/// Indices colliding on a small target and past it, and values mixed enough
/// that a lost, repeated or torn add shows in the sum.
fn colliding(spec: &ScatterSpec) -> Vec<Vec<f64>> {
    let &ScatterSpec { rows, pairs, d } = spec;
    let mix = |i: usize, salt: u32| (i as u32).wrapping_mul(0x9E37_79B1).wrapping_add(salt) as i32;
    let index = (0..pairs).map(|p| f64::from((mix(p, 7) >> 8).rem_euclid(rows as i32 + 3) - 2)).collect();
    let held = (0..rows * d).map(|i| f64::from(mix(i, 1))).collect();
    let values = (0..pairs * d).map(|i| f64::from(mix(i, 2))).collect();
    vec![held, index, values]
}

/// On the device an `i32` scatter is the host's sum to the bit, under heavy
/// collisions, across block counts from a padded one to many.
#[test_case(4, 100, 1; "a padded block")]
#[test_case(16, TILE, 2; "one whole block")]
#[test_case(16, 300, 3; "a padded last block")]
#[test_case(4, 40 * TILE, 2; "every thread on four rows")]
#[test_case(1 << 16, 1 << 18, 2; "a target past the cache")]
fn an_i32_scatter_is_exact_on_the_device(rows: usize, pairs: usize, d: usize) {
    let Some(target) = super::device_target() else { return };
    let spec = ScatterSpec { rows, pairs, d };
    let params = colliding(&spec);
    let want = reference(&spec, &params[0], &params[1], &params[2]);
    let got = super::targets::on_device(scatter_add::<I32>(&spec), &lowering(target), &params);
    let got = got[0].as_ref().expect("the target");
    let wrong = got.iter().zip(&want).filter(|(g, w)| g != w).count();
    assert_eq!(wrong, 0, "{wrong} of {} elements differ", want.len());
}

/// On the device an `f32` scatter is within the rounding its terms allow, in
/// whatever order the threads land.
#[test]
fn an_f32_scatter_is_within_its_rounding_on_the_device() {
    let Some(target) = super::device_target() else { return };
    let spec = ScatterSpec { rows: 3, pairs: 3 * TILE, d: 1 };
    let index: Vec<f64> = (0..spec.pairs).map(|i| (i % 3) as f64).collect();
    let values: Vec<f64> = (0..spec.pairs).map(|i| f64::from(i as f32 * 0.1)).collect();
    let params = vec![vec![1.0; 3], index, values.clone()];
    let got = super::targets::on_device(scatter_add::<F32>(&spec), &lowering(target), &params);
    let exact: Vec<f64> = (0..3).map(|r| 1.0 + values.iter().skip(r).step_by(3).sum::<f64>()).collect();
    let bound = spec.pairs as f64 * f64::from(f32::EPSILON);
    super::assert_rel_close("an f32 sum", got[0].as_ref().expect("the target"), &exact, bound);
}

#[test]
fn the_op_sums_and_chains_on_the_device() {
    if super::device_target().is_some() {
        the_op_sums_and_chains();
    }
}
