//! Running a tile program as a lazy graph kernel: placeholders sized at
//! capacity, the lowered PROGRAM as the call body. The first `Out`
//! parameter is the call's output and every other parameter an input;
//! results return in `prog.params()` order.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::sync::{Arc, LazyLock, Mutex};

use snafu::{ResultExt, Snafu};
use svod_dtype::default_device::default_device;
use svod_dtype::{AmdArch, DeviceSpec, GpuArch};
use svod_ir::{CallInfo, UOp};
use svod_tensor::Tensor;

use crate::ir::{ParamKind, Program};
use crate::lower::{self, Lowered, Lowering};

/// Lowered bodies by tile program, lowering, device and placeholders: a model
/// calls an op once per layer, and lowering is milliseconds of instruction
/// emission per call that hash-consing would only deduplicate afterwards.
static LOWERED: LazyLock<Mutex<HashMap<u64, Arc<UOp>>>> = LazyLock::new(Mutex::default);

fn lowered_key(prog: &Program, lowering: &Lowering, device: &DeviceSpec, placeholders: &[Arc<UOp>]) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    format!("{prog:?}{lowering:?}{device:?}").hash(&mut hasher);
    for placeholder in placeholders {
        format!("{:?}{:?}", placeholder.dtype(), placeholder.shape().ok().flatten()).hash(&mut hasher);
    }
    hasher.finish()
}

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("lowering {name}: {source}"))]
    Lower {
        name: String,
        #[snafu(source(from(lower::Error, Box::new)))]
        source: Box<lower::Error>,
    },
    #[snafu(display("graph kernel {name}: {source}"))]
    Graph { name: String, source: svod_tensor::error::Error },
    #[snafu(display("{name} declares {want} parameters, {got} tensors were given"))]
    Arity { name: String, want: usize, got: usize },
    #[snafu(display("{name} writes {a}, which shares storage with {b}"))]
    Aliased { name: String, a: String, b: String },
    #[snafu(display("{name} is lowered for {lowered:?}, which an {device} device does not run"))]
    Arch { name: String, lowered: GpuArch, device: AmdArch },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Launch `prog` over `tensors`, one per declared parameter in order; the
/// first `Out` parameter's tensor is returned as the lazy result.
pub fn graph_launch(prog: Program, lowering: &Lowering, tensors: &[&Tensor]) -> Result<Tensor> {
    graph_launch_ranged(prog, lowering, tensors, &[])
}

/// [`graph_launch`] where some parameters touch only a range of their
/// tensor's last axis (`ranges[i]`, missing or `None`: all of it): a channel
/// window of a buffer another parameter reads or writes another window of.
pub fn graph_launch_ranged(
    prog: Program,
    lowering: &Lowering,
    tensors: &[&Tensor],
    ranges: &[Option<Range<usize>>],
) -> Result<Tensor> {
    let out_at = first_output(&prog);
    Ok(launch_all(prog, lowering, tensors, ranges, true)?.swap_remove(out_at))
}

/// [`graph_launch`] with the body lowered for this launch alone, kept in no
/// memo: a tuner's candidate on scratch buffers, whose placeholders never
/// key an op's call.
pub(crate) fn graph_launch_once(prog: Program, lowering: &Lowering, tensors: &[&Tensor]) -> Result<Tensor> {
    let out_at = first_output(&prog);
    Ok(launch_all(prog, lowering, tensors, &[], false)?.swap_remove(out_at))
}

/// [`graph_launch_all`] of `body`, `prog` lowered already for its target
/// alone ([`lower::standalone`], as [`body`] hands it), kept in no memo: its parameters are
/// positional in `prog.params()` order, so the call binds `tensors` in that
/// order. Where this device or the tensors' placeholders are not the ones it
/// was lowered over, `prog` is lowered again for them.
pub(crate) fn graph_launch_lowered(
    prog: &Program,
    lowering: &Lowering,
    body: Arc<UOp>,
    tensors: &[&Tensor],
) -> Result<Vec<Tensor>> {
    let name = prog.name.clone();
    snafu::ensure!(
        tensors.len() == prog.params.len(),
        AritySnafu { name: name.clone(), want: prog.params.len(), got: tensors.len() }
    );
    let device = default_device();
    if device != lower::standalone_device(lowering) || tensors.is_empty() {
        return launch_all(prog.clone(), lowering, tensors, &[], false);
    }
    runs_on(&name, lowering, &device)?;
    let origin = svod_ir::origin::current();
    let info =
        CallInfo { name: Some(name.clone()), origin, origins: origin.into_iter().collect(), ..CallInfo::default() };
    let mut fits = true;
    let outs = tensors[0].custom_kernel_with(&tensors[1..], info, |ph| {
        let param =
            |(i, p): (usize, &crate::ir::Param)| UOp::param(i, p.elems, svod_dtype::DType::Scalar(p.dtype), None);
        fits = ph.iter().zip(prog.params.iter().enumerate().map(param)).all(|(ph, param)| ph.base() == param);
        if fits { body.without_origins() } else { UOp::noop() }
    });
    if !fits {
        return launch_all(prog.clone(), lowering, tensors, &[], false);
    }
    outs.context(GraphSnafu { name })
}

pub(crate) fn first_output(prog: &Program) -> usize {
    prog.params.iter().position(|p| p.kind != ParamKind::In).expect("a program writes something")
}

/// The bases of `[out, ins...]` placeholders back in parameter order: the
/// one at `out_at` is slot 0, the inputs slots 1.. in order.
fn in_param_order(placeholders: &[Arc<UOp>], out_at: usize) -> Vec<Arc<UOp>> {
    let mut ins = placeholders[1..].iter();
    (0..placeholders.len())
        .map(|i| if i == out_at { &placeholders[0] } else { ins.next().expect("an input") }.base())
        .collect()
}

/// [`graph_launch`] returning every parameter's tensor as it is after the
/// kernel, in `prog.params()` order, so programs with several outputs can be read.
pub fn graph_launch_all(prog: Program, lowering: &Lowering, tensors: &[&Tensor]) -> Result<Vec<Tensor>> {
    launch_all(prog, lowering, tensors, &[], true)
}

fn launch_all(
    prog: Program,
    lowering: &Lowering,
    tensors: &[&Tensor],
    ranges: &[Option<Range<usize>>],
    memo: bool,
) -> Result<Vec<Tensor>> {
    let name = prog.name.clone();
    snafu::ensure!(
        tensors.len() == prog.params.len(),
        AritySnafu { name: name.clone(), want: prog.params.len(), got: tensors.len() }
    );
    aliasing(&prog, tensors, ranges)?;
    let device = default_device();
    runs_on(&name, lowering, &device)?;
    let out_at = first_output(&prog);
    // `custom_kernel` hands placeholders in `[out, ins...]` order.
    let ins: Vec<&Tensor> = tensors.iter().enumerate().filter(|(i, _)| *i != out_at).map(|(_, t)| *t).collect();
    let mut failure = None;
    // The kernel is charged to the call site, never to whichever site first
    // lowered the memoized body.
    let origin = svod_ir::origin::current();
    let info =
        CallInfo { name: Some(name.clone()), origin, origins: origin.into_iter().collect(), ..CallInfo::default() };
    let result = tensors[out_at].custom_kernel_with(&ins, info, |ph| {
        let key = lowered_key(&prog, lowering, &device, &ph);
        if let Some(program) = LOWERED.lock().expect("lowered memo").get(&key).filter(|_| memo) {
            return program.clone();
        }
        match lower::lower(prog, lowering, in_param_order(&ph, out_at), device).and_then(|l| body(&l, lowering)) {
            Ok(program) => {
                if memo {
                    LOWERED.lock().expect("lowered memo").insert(key, program.clone());
                }
                program
            }
            Err(err) => {
                failure = Some(err);
                UOp::noop()
            }
        }
    });
    if let Some(source) = failure {
        return Err(Error::Lower { name, source: Box::new(source) });
    }
    // Back from `[out, ins...]` to parameter order.
    let mut outs = result.context(GraphSnafu { name })?;
    let out = outs.remove(0);
    outs.insert(out_at, out);
    Ok(outs)
}

/// The PROGRAM a launch of `lowered` hands the graph, without origins: on
/// an AMD target with its SOURCE stage ([`Lowered::source`]), so the code
/// object carries its line table and `kernel.tk`; elsewhere bare, for the
/// device to render.
pub(crate) fn body(lowered: &Lowered, lowering: &Lowering) -> lower::Result<Arc<UOp>> {
    let program = match lowering.device.target.arch {
        GpuArch::Amd(arch) => lowered.source(arch)?,
        GpuArch::Cuda(_) | GpuArch::Metal(_) => lowered.program.clone(),
    };
    Ok(program.without_origins())
}

/// An AMD device runs what is lowered for its own arch: a [`body`]'s
/// SOURCE names no arch, so a kernel lowered for another would compile for
/// the device's unnoticed. A device whose arch is unknown here (none
/// attached) is not checked.
fn runs_on(name: &str, lowering: &Lowering, device: &DeviceSpec) -> Result<()> {
    static ARCHS: LazyLock<Mutex<std::collections::HashMap<usize, Option<AmdArch>>>> = LazyLock::new(Default::default);
    let &DeviceSpec::Amd { device_id } = device else { return Ok(()) };
    // Topology is read once per device: a launch would otherwise walk sysfs.
    let arch = *ARCHS
        .lock()
        .expect("device archs")
        .entry(device_id)
        .or_insert_with(|| svod_device::registry::resolve_amd_arch_from_topology(device_id).ok());
    let Some(arch) = arch else { return Ok(()) };
    let lowered = lowering.device.target.arch;
    snafu::ensure!(lowered == GpuArch::Amd(arch), ArchSnafu { name, lowered, device: arch });
    Ok(())
}

/// No parameter the kernel writes shares memory with another: the body
/// declares every buffer `noalias` and every input `readonly`
/// ([`lower::lower`]), which promise only that no element one parameter
/// writes is touched through another. Two tensors share storage as one lazy
/// node, or as realized buffers over one allocation whose windows overlap;
/// parameters sharing it stay apart where `ranges` gives both a range of the
/// last axis and the ranges are disjoint (every row's elements apart).
pub(crate) fn aliasing(prog: &Program, tensors: &[&Tensor], ranges: &[Option<Range<usize>>]) -> Result<()> {
    let shared = |a: &Tensor, b: &Tensor| {
        Arc::ptr_eq(&a.uop(), &b.uop())
            || matches!((a.buffer(), b.buffer()), (Some(x), Some(y))
                if x.storage_id() == y.storage_id()
                    && x.offset() < y.offset() + y.size()
                    && y.offset() < x.offset() + x.size())
    };
    let range = |i: usize| ranges.get(i).cloned().flatten();
    let apart =
        |i: usize, j: usize| matches!((range(i), range(j)), (Some(a), Some(b)) if a.end <= b.start || b.end <= a.start);
    for (i, (a, pa)) in tensors.iter().zip(&prog.params).enumerate() {
        for (j, (b, pb)) in tensors.iter().zip(&prog.params).enumerate().skip(i + 1) {
            let read = pa.kind == ParamKind::In && pb.kind == ParamKind::In;
            let fine = read || apart(i, j) || !shared(a, b);
            snafu::ensure!(fine, AliasedSnafu { name: &prog.name, a: &pa.name, b: &pb.name });
        }
    }
    Ok(())
}
