//! The compile-probe oracle: instruction availability asked of the toolchain
//! that compiles the kernels, per target, cached in the tune store like any
//! probe. The [`Target`](crate::atoms::Target) booleans (`group_mma`,
//! `ldmatrix`, `tr_load`, `dot2_bf16`, the int8 atoms) are hand arch lists
//! today; the oracle is what makes an unseen architecture self-describing —
//! its verdict is the vendor assembler's own accept/reject of a one-instruction
//! module (ptxas: "Instruction 'wgmma.fence' not supported on .target
//! 'sm_86'"; the AMDGPU assembler: "instruction not supported on this GPU").
//!
//! Rollout gate (this phase): the tables still drive everything. Where a
//! device's oracle answers, [`audit`] compares the two and logs a mismatch
//! loudly — on a runnable target a disagreement is a bug in one of the two,
//! and the host test pins them equal for every target the sweep digests.

use std::hash::{Hash, Hasher};

use svod_dtype::{AmdArch, CudaArch, GpuArch, ScalarDType};
use tracing::warn;

use crate::atoms::{Fill, Target};
use crate::tune::{TuneKey, TuneStore};

/// One instruction the target tables gate that gates cost terms or admission:
/// its tune-store op (`oracle:<name>`), the module exercising exactly it, and
/// the hand table's verdict for the target.
struct Capability {
    name: &'static str,
    module: String,
    table: bool,
}

/// The `name` the tune store records a capability's verdict under.
fn op(name: &str) -> &'static str {
    match name {
        "wgmma" => "oracle:wgmma",
        "ldmatrix" => "oracle:ldmatrix",
        "mma_i8" => "oracle:mma_i8",
        "cp_async" => "oracle:cp_async",
        "wmma16" => "oracle:wmma16",
        "tr_load" => "oracle:tr_load",
        "dot2_bf16" => "oracle:dot2_bf16",
        other => unreachable!("no oracle op for {other}"),
    }
}

/// Every capability the target's tables gate, with the one-instruction module
/// that asks the toolchain about each and the table's own answer. The modules
/// are keyed by arch family where an instruction's operand packing differs
/// (RDNA3's WMMA holds its operands in eight registers a lane, RDNA4's in
/// four, CDNA's MFMA in two) — ISA encodings, the same class of fact as the
/// tables they are checked against.
fn capabilities(target: &Target) -> Vec<Capability> {
    match target.arch {
        GpuArch::Cuda(c) => cuda(target, c),
        GpuArch::Amd(a) => amd(target, a),
        GpuArch::Metal(_) => vec![],
    }
}

fn cuda(target: &Target, c: CudaArch) -> Vec<Capability> {
    use svod_codegen::llvm::nvptx::{march, ptx_isa};
    let isa = ptx_isa(c);
    // The skeleton every module shares: the ISA floor and `-march` the build
    // itself compiles with, so the assembler answers for the target kernels
    // are emitted to.
    let module =
        |body: &str| format!(".version {}.{}\n.target {}\n.address_size 64\n{body}\n", isa / 10, isa % 10, march(c));
    let wgmma = module("\n.visible .entry oracle() {\n  wgmma.fence.sync.aligned;\n  ret;\n}");
    let shared = ".shared .align 16 .b8 smem[128];\n";
    let ldmatrix = module(&format!(
        "{shared}\n.visible .entry oracle() {{\n  .reg .b32 %r<5>;\n  .reg .b64 %rd1;\n  mov.u64 %rd1, smem;\n  \
         ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r1, %r2, %r3, %r4}}, [%rd1];\n  ret;\n}}"
    ));
    let mma_i8 = module(
        "\n.visible .entry oracle() {\n  .reg .b32 %r<12>;\n  mov.b32 %r1, 0;\n  \
         mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%r5, %r6, %r7, %r8}, \
         {%r1, %r2, %r3, %r4}, {%r9, %r10}, {%r5, %r6, %r7, %r8};\n  ret;\n}",
    );
    let cp_async = module(&format!(
        "{shared}\n.visible .entry oracle() {{\n  .reg .b64 %rd<3>;\n  mov.u64 %rd1, smem;\n  mov.u64 %rd2, 0;\n  \
         cp.async.cg.shared.global [%rd1], [%rd2], 16;\n  cp.async.commit_group;\n  ret;\n}}"
    ));
    vec![
        Capability { name: "wgmma", module: wgmma, table: !target.group_mma.is_empty() },
        Capability { name: "ldmatrix", module: ldmatrix, table: target.ldmatrix },
        Capability { name: "mma_i8", module: mma_i8, table: target.geometry_of(ScalarDType::Int8).is_some() },
        Capability { name: "cp_async", module: cp_async, table: target.fill == Fill::CpAsync },
    ]
}

fn amd(target: &Target, a: AmdArch) -> Vec<Capability> {
    let (cdna, rdna4) = (a.is_cdna(), a.is_rdna4());
    let module = |body: &str| {
        format!(".text\n.globl oracle\n.p2align 8\n.type oracle,@function\noracle:\n  {body}\n  s_endpgm\n")
    };
    // RDNA3 and RDNA4 name the same WMMA with different operand packing; CDNA
    // has MFMA instead. Verified per family against the assembler (the RDNA3
    // form is what a lowered kernel emits at wave32, eight dwords an operand).
    let wmma16 = match (cdna, rdna4) {
        (true, _) => module("v_mfma_f32_16x16x16_f16 v[0:3], v[4:5], v[6:7], v[0:3]"),
        (false, true) => module("v_wmma_f32_16x16x16_bf16 v[0:7], v[8:11], v[12:15], v[16:23]"),
        (false, false) => module("v_wmma_f32_16x16x16_bf16 v[0:7], v[8:15], v[16:23], v[24:31]"),
    };
    let mma_i8 = match (cdna, rdna4) {
        (true, _) => module("v_mfma_i32_16x16x32_i8 v[0:3], v[4:5], v[6:7], v[0:3]"),
        (false, true) => module("v_wmma_i32_16x16x16_iu8 v[0:7], v[8:9], v[10:11], v[12:19]"),
        (false, false) => module("v_wmma_i32_16x16x16_iu8 v[0:7], v[8:11], v[12:15], v[16:23]"),
    };
    vec![
        Capability { name: "wmma16", module: wmma16, table: target.geometry().is_some() },
        Capability { name: "mma_i8", module: mma_i8, table: target.geometry_of(ScalarDType::Int8).is_some() },
        Capability {
            name: "tr_load",
            module: module("global_load_tr_b128 v[0:3], v[4:5], off"),
            table: target.tr_load,
        },
        Capability { name: "dot2_bf16", module: module("v_dot2_f32_bf16 v0, v1, v2, v3"), table: target.dot2_bf16 },
    ]
}

/// The toolchain's answer for one module: `Some(whether it accepts)`; `None`
/// where no toolchain can answer (no ptxas, no amdgcn clang), which is not a
/// verdict and is never stored.
fn assemble(target: &Target, module: &str) -> Option<bool> {
    match target.arch {
        GpuArch::Cuda(c) => {
            if !svod_runtime::cuda::has_ptxas() {
                return None;
            }
            Some(svod_runtime::cuda::assemble_ptx(module, c).is_ok())
        }
        GpuArch::Amd(a) => {
            if !svod_runtime::amd::has_amdgpu_target() {
                return None;
            }
            Some(svod_runtime::amd::assemble_gcn(module, a).is_ok())
        }
        GpuArch::Metal(_) => None,
    }
}

/// A module's digest, keying its verdict's store line: a module that changes
/// (an operand form corrected, an instruction renamed) measures anew.
fn digest(module: &str) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    module.hash(&mut hasher);
    hasher.finish()
}

/// Every capability with the toolchain's fresh verdict for `target` — no
/// store, no device, an assemble per module: the host gate's comparison
/// against the tables ([`crate::test`]'s `oracle_verdicts_match_the_tables`).
#[cfg(test)]
pub(crate) fn probed(target: &Target) -> Vec<(&'static str, bool, Option<bool>)> {
    capabilities(target).into_iter().map(|c| (c.name, c.table, assemble(target, &c.module))).collect()
}

/// The hand tables' own verdicts for `target`, by capability: what an offline
/// target runs on where no toolchain answered.
pub fn tables(target: &Target) -> Vec<(&'static str, bool)> {
    capabilities(target).into_iter().map(|c| (c.name, c.table)).collect()
}

/// Every capability with its recorded verdict: the store's line where one
/// measured (first-measurement-wins, as any probe), else assembled and stored
/// — never for a target without an SM count, whose device no store names.
fn verdicts(target: &Target) -> Vec<(Capability, Option<bool>)> {
    let store = (target.sms.is_some()).then(TuneStore::global);
    let compiler = crate::tune::compiler(&svod_dtype::default_device::default_device());
    capabilities(target)
        .into_iter()
        .map(|c| {
            let verdict = store.and_then(|store| {
                let key = TuneKey::new(op(c.name), target, compiler, ScalarDType::Int32, &[], &c.module);
                store.value(&key, || digest(&c.module), || assemble(target, &c.module).map(u64::from)).map(|v| v == 1)
            });
            (c, verdict)
        })
        .collect()
}

/// The recorded verdicts of the target's device, one per capability,
/// assembling and storing any that is missing (the store memoizes: one
/// assemble per device, toolchain and module). `None` where the toolchain is
/// absent — a missing verdict, never a false.
pub fn record(target: &Target) -> Vec<(&'static str, Option<bool>)> {
    verdicts(target).into_iter().map(|(c, v)| (c.name, v)).collect()
}

/// The oracle beside the target's tables, at planner build: every recorded
/// verdict is compared with the table's own — equal, the tables drive on and
/// the verdict is evidence; a mismatch does not switch anything but is logged
/// loudly, a bug in one of the two on a runnable target.
pub(crate) fn audit(target: &Target) {
    for (capability, verdict) in verdicts(target) {
        let name = capability.name;
        match verdict {
            Some(v) if v != capability.table => warn!(
                arch = %target.arch.target_name(),
                name,
                oracle = v,
                table = capability.table,
                "the compile-probe oracle disagrees with the target tables: one of the two is wrong"
            ),
            Some(_) => tracing::debug!(target: crate::ops::PLAN, name, "oracle agrees with the tables"),
            None => tracing::debug!(target: crate::ops::PLAN, name, "no oracle: the toolchain is absent"),
        }
    }
}

/// The verdicts a store copied from a card recorded for its device, read
/// never written: `None` per capability the store does not settle.
pub fn stored(dir: &std::path::Path, target: &Target) -> Vec<(&'static str, Option<bool>)> {
    let device = crate::tune::device(target);
    let text = TuneStore::replay(dir.to_path_buf())
        .file(&device)
        .and_then(|file| std::fs::read_to_string(file).ok())
        .unwrap_or_default();
    capabilities(target)
        .into_iter()
        .map(|c| {
            let prefix = format!("value|{}|{device}|", op(c.name));
            let verdict = text.lines().find_map(|line| {
                let (key, value) = line.split_once(' ')?;
                key.starts_with(&prefix).then(|| value.trim().parse::<u64>().ok())?
            });
            (c.name, verdict.map(|v| v == 1))
        })
        .collect()
}
