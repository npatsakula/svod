//! `clang -target amdgcn-amd-amdhsa` driver: lowers AMD LLVM text-IR
//! to an AMDGPU code object (ELF) that the KFD runtime can dispatch.

use svod_dtype::AmdArch;
use tracing::debug;

use crate::clang::{ClangToolchain, dump_ir, path_clang_has_target};
use crate::error::JitResultExt;

/// Compile AMD LLVM IR text into a fully-linked AMDGPU code object.
///
/// `clang --target=amdgcn-amd-amdhsa -mcpu={arch}` produces an `ET_DYN` ELF
/// that already has lld's amdgpu-link step applied (clang invokes lld
/// internally for single-TU compilations), so the output is directly loadable
/// by the KFD runtime — no further link step required.
///
/// # Errors
///
/// - [`crate::Error::JitCompilation`] when `clang` is missing, the AMDGPU
///   target is not enabled in the host LLVM, or compilation fails.
pub fn compile_ir_to_amd_object(ir: &str, arch: AmdArch) -> crate::Result<Vec<u8>> {
    let toolchain = ClangToolchain::discover(None)?;
    compile_ir_to_amd_object_with(&toolchain, ir, arch)
}

/// Clang driver flags for one kernel.
///
/// `-nogpulib` skips the ROCm device-library search entirely, the way tinygrad
/// drives amdgcn straight through LLVM (`runtime/support/compiler_llvm.py:19-24`
/// links no device libs). The renderer emits `@llvm.*` intrinsics for every
/// float unary the AMDGPU backend can select, so the libraries are only needed
/// when the IR still references the f64 `__ocml_*` entry points
/// (`codegen/src/llvm/amd/ops.rs::render_float_unary`). The IR is part of the
/// object-cache key, so keying a flag off it keeps the key sound.
pub(crate) fn amd_object_flags(ir: &str, arch: AmdArch) -> Vec<String> {
    let mut flags: Vec<String> = vec![
        "-x".into(),
        "ir".into(),
        "-c".into(),
        "-O3".into(),
        "--target=amdgcn-amd-amdhsa".into(),
        format!("-mcpu={}", arch.mcpu()),
        "-mcumode".into(),
        "-nogpuinc".into(),
        "-Wno-override-module".into(),
        "-fno-math-errno".into(),
    ];
    if !ir.contains("@__ocml_") {
        flags.push("-nogpulib".into());
    }
    flags.extend(["-", "-o", "-"].map(str::to_string));
    flags
}

pub(crate) fn compile_ir_to_amd_object_with(
    toolchain: &ClangToolchain,
    ir: &str,
    arch: AmdArch,
) -> crate::Result<Vec<u8>> {
    if !toolchain.has_target("amdgcn") {
        return Err(crate::Error::JitCompilation {
            reason: "AMD GPU support requires clang built with the AMDGPU target. \
                     Reinstall clang from your distro or build with \
                     -DLLVM_TARGETS_TO_BUILD='X86;AArch64;AMDGPU'."
                .to_string(),
        });
    }
    dump_ir("SVOD_DUMP_AMD_IR", arch.mcpu(), ir);
    debug!(arch = arch.mcpu(), ir.length = ir.len(), "compiling amdgcn IR via clang");
    toolchain.compile_ir(&amd_object_flags(ir, arch), ir, &format!("amdgcn (mcpu={})", arch.mcpu()))
}

/// Validate both the generic ELF contract and the architecture encoded in
/// AMDGPU `e_flags` before cached bytes reach `AmdProgram`.
pub(crate) fn validate_amd_object(bytes: &[u8], arch: AmdArch, kernel_name: &str) -> crate::Result<()> {
    use object::elf::{ELFCLASS64, ELFDATA2LSB, EM_AMDGPU};
    use object::read::elf::FileHeader;
    use object::read::{Object, ObjectSymbol};
    use object::{BinaryFormat, LittleEndian, ObjectKind};

    let header = object::elf::FileHeader64::<LittleEndian>::parse(bytes).jit("parse cached AMD ELF header")?;
    let endian = header.endian().jit("read cached AMD ELF endian")?;
    if header.e_ident().class != ELFCLASS64
        || header.e_ident().data != ELFDATA2LSB
        || header.e_machine.get(endian) != EM_AMDGPU
        || header.e_flags.get(endian) & 0xff != amd_elf_machine(arch)
    {
        return Err(crate::Error::JitCompilation {
            reason: format!("cached AMD object is not compatible with {}", arch.mcpu()),
        });
    }
    let file = object::File::parse(bytes).jit("parse cached AMD object")?;
    if file.format() != BinaryFormat::Elf || !matches!(file.kind(), ObjectKind::Relocatable | ObjectKind::Dynamic) {
        return Err(crate::Error::JitCompilation { reason: "cached AMD object has invalid ELF format".into() });
    }
    let descriptor = format!("{kernel_name}.kd");
    if !file.symbols().any(|symbol| symbol.is_definition() && symbol.name() == Ok(&descriptor)) {
        return Err(crate::Error::JitCompilation {
            reason: format!("cached AMD object has no kernel descriptor {descriptor:?}"),
        });
    }
    Ok(())
}

fn amd_elf_machine(arch: AmdArch) -> u32 {
    match arch {
        AmdArch::Gfx1100 => 0x041,
        AmdArch::Gfx1101 => 0x046,
        AmdArch::Gfx1102 => 0x047,
        AmdArch::Gfx1200 => 0x048,
        AmdArch::Gfx1151 => 0x04a,
        AmdArch::Gfx942 => 0x04c,
        AmdArch::Gfx1201 => 0x04e,
        AmdArch::Gfx950 => 0x04f,
    }
}

/// Does the host `clang` advertise the `amdgpu` target?
///
/// Cached for the lifetime of the process: clang installation doesn't change
/// during a run, and the subprocess is too slow to do per-call.
pub fn has_amdgpu_target() -> bool {
    path_clang_has_target("amdgcn")
}

#[cfg(test)]
#[path = "../test/unit/amd_compile.rs"]
mod tests;
