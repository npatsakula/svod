//! Running a tile program as a lazy graph kernel: placeholders sized at
//! capacity, the lowered PROGRAM as the call body, outputs in `prog.params`
//! order after the inputs.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, LazyLock, Mutex};

use snafu::{ResultExt, Snafu};
use svod_dtype::DeviceSpec;
use svod_dtype::default_device::default_device;
use svod_ir::{CallInfo, UOp};
use svod_tensor::Tensor;

use crate::ir::{ParamKind, Program};
use crate::lower::{self, Lowering};

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
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Launch `prog` over `tensors`, one per declared parameter in order; the
/// first `Out`/`InOut` parameter's tensor is returned as the lazy result.
pub fn graph_launch(prog: Program, lowering: &Lowering, tensors: &[&Tensor]) -> Result<Tensor> {
    let out_at = first_output(&prog);
    Ok(graph_launch_all(prog, lowering, tensors)?.swap_remove(out_at))
}

fn first_output(prog: &Program) -> usize {
    prog.params
        .iter()
        .position(|p| matches!(p.kind, ParamKind::Out | ParamKind::InOut))
        .expect("a program writes something")
}

/// [`graph_launch`] returning every parameter's tensor as it is after the
/// kernel, in `prog.params` order, so programs with several outputs can be read.
pub fn graph_launch_all(prog: Program, lowering: &Lowering, tensors: &[&Tensor]) -> Result<Vec<Tensor>> {
    let name = prog.name.clone();
    snafu::ensure!(
        tensors.len() == prog.params.len(),
        AritySnafu { name: name.clone(), want: prog.params.len(), got: tensors.len() }
    );
    let out_at = first_output(&prog);
    // `custom_kernel` hands placeholders in `[out, ins...]` order.
    let ins: Vec<&Tensor> = tensors.iter().enumerate().filter(|(i, _)| *i != out_at).map(|(_, t)| *t).collect();
    let device = default_device();
    let mut failure = None;
    // The kernel is charged to the call site, never to whichever site first
    // lowered the memoized body.
    let origin = svod_ir::origin::current();
    let info =
        CallInfo { name: Some(name.clone()), origin, origins: origin.into_iter().collect(), ..CallInfo::default() };
    let result = tensors[out_at].custom_kernel_with(&ins, info, |ph| {
        let key = lowered_key(&prog, lowering, &device, &ph);
        if let Some(program) = LOWERED.lock().expect("lowered memo").get(&key) {
            return program.clone();
        }
        let mut params: Vec<Arc<UOp>> = Vec::with_capacity(ph.len());
        let mut rest = ph[1..].iter();
        for i in 0..ph.len() {
            params.push(if i == out_at { ph[0].base() } else { rest.next().expect("an input").base() });
        }
        match lower::lower(prog, lowering, params, device) {
            Ok(lowered) => {
                let program = lowered.program.without_origins();
                LOWERED.lock().expect("lowered memo").insert(key, program.clone());
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
