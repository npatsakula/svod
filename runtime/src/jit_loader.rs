//! JIT ELF loader: compiles C source via clang stdin→stdout, parses the
//! relocatable ELF with the `object` crate, copies sections into an anonymous
//! mmap, applies relocations, and returns an executable function pointer.
//!
//! Supports x86_64, aarch64, riscv64, loongarch64, and powerpc64le.

use std::collections::HashMap;

use object::read::{Object, ObjectSection, ObjectSymbol};
use object::{Architecture, RelocationFlags, SectionKind};

use crate::dispatch::KernelCif;
use crate::error::JitResultExt;

/// A compiled C kernel loaded via custom ELF relocator + mmap.
pub struct JitKernel {
    _mmap: memmap2::MmapMut,
    fn_ptr: *const (),
    name: String,
    var_names: Vec<String>,
    cif: KernelCif,
}

// SAFETY: Function pointer points to read-only compiled code in mmap'd memory.
// Multiple threads can call it concurrently.
unsafe impl Send for JitKernel {}
unsafe impl Sync for JitKernel {}

impl JitKernel {
    pub fn compile_with_abi(
        src: &str,
        name: &str,
        var_names: Vec<String>,
        abi: &[svod_device::device::AbiParamDescriptor],
    ) -> crate::Result<Self> {
        let buffer_count = abi.iter().filter(|arg| arg.is_storage()).count();
        svod_device::device::validate_abi_descriptors(abi, buffer_count, &var_names)?;
        let toolchain = crate::clang::ClangToolchain::discover(None)?;
        let flags = c_object_flags();
        let object = crate::clang::compile_c_object(&toolchain, src, &flags)?;
        Self::load_object_with_abi(&object, name, var_names, abi)
    }

    pub fn load_object_with_abi(
        object: &[u8],
        name: &str,
        var_names: Vec<String>,
        abi: &[svod_device::device::AbiParamDescriptor],
    ) -> crate::Result<Self> {
        let buffer_count = abi.iter().filter(|arg| arg.is_storage()).count();
        svod_device::device::validate_abi_descriptors(abi, buffer_count, &var_names)?;
        crate::clang::validate_relocatable_object(object, name)?;
        let (fn_ptr, mmap) = jit_load(object, name)?;
        let cif = KernelCif::from_abi(abi)?;
        Ok(Self { _mmap: mmap, fn_ptr, name: name.to_string(), var_names, cif })
    }

    /// Execute the kernel with buffer pointers and variable values.
    ///
    /// # Safety
    ///
    /// Caller must ensure buffer pointers are valid/aligned and `vals` length
    /// matches `var_names`.
    pub unsafe fn execute_with_vals(&self, buffers: &[*mut u8], vals: &[i64]) -> crate::Result<()> {
        unsafe { self.cif.dispatch(self.fn_ptr, buffers, vals, None)? };
        Ok(())
    }

    #[cfg(not(feature = "dlopen-fallback"))]
    pub(crate) fn cif(&self) -> &KernelCif {
        &self.cif
    }

    pub fn fn_ptr(&self) -> *const () {
        self.fn_ptr
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn var_names(&self) -> &[String] {
        &self.var_names
    }
}

// ── Compilation ─────────────────────────────────────────────────────────────

/// Returns the `--target=<arch>-none-unknown-elf` flag for the host architecture.
/// Shared between the C and LLVM IR compilation paths.
pub(crate) fn elf_target_triple() -> String {
    format!("--target={}", elf_triple())
}

/// Bare `<arch>-none-unknown-elf` triple shared by the clang and in-process
/// LLVM object producers.
pub(crate) fn elf_triple() -> String {
    let arch = std::env::consts::ARCH;
    if arch == "powerpc64" && cfg!(target_endian = "little") {
        "powerpc64le-none-unknown-elf".to_string()
    } else {
        format!("{arch}-none-unknown-elf")
    }
}

/// Extra clang flags required for correct JIT code on the host platform.
/// Shared between the C and LLVM IR compilation paths.
pub(crate) fn platform_clang_flags() -> &'static [&'static str] {
    // Reserve x18 only on macOS ARM, where the kernel clobbers it on context
    // switch. Linux ARM treats x18 as a free GPR; Windows ARM is not a target
    // svod currently supports.
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        &["-ffixed-x18"]
    }
    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    {
        &[]
    }
}

pub(crate) fn c_object_flags() -> Vec<String> {
    let arch = std::env::consts::ARCH;

    // Architecture-specific tuning. On ARM, `-march=native` only sets the
    // base ISA family (e.g. `armv8-a`); CPU-specific tuning (Apple-Silicon
    // pipelines, NEON dual-issue scheduling, FP cost model) requires
    // `-mcpu=native`.
    let march = match arch {
        "x86_64" | "loongarch64" => "-march=native",
        "riscv64" => "-march=rv64g",
        _ => "-mcpu=native",
    };

    let target = elf_target_triple();

    let args = vec![
        "-c",
        "-x",
        "c",
        "-O2",
        march,
        "-fPIC",
        "-ffreestanding",
        "-fno-math-errno",
        "-fno-stack-protector",
        "-nostdlib",
        "-fno-ident",
    ];
    let mut flags = args.into_iter().map(str::to_string).collect::<Vec<_>>();
    flags.push(target);
    flags.extend(platform_clang_flags().iter().map(|flag| (*flag).to_string()));
    flags.extend(["-", "-o", "-"].map(str::to_string));
    flags
}

// ── ELF Loading ─────────────────────────────────────────────────────────────

/// Parse an ELF relocatable object, copy loadable sections into an anonymous
/// mmap, apply relocations, mprotect to executable, and return the function
/// pointer for the named symbol.
pub(crate) fn jit_load(obj: &[u8], name: &str) -> crate::Result<(*const (), memmap2::MmapMut)> {
    let elf = object::File::parse(obj).jit("parse ELF")?;

    let arch = elf.architecture();

    // Collect loadable sections and compute layout.
    let mut section_offsets: HashMap<object::SectionIndex, usize> = HashMap::new();
    let mut total_size: usize = 0;

    for section in elf.sections() {
        if matches!(section.kind(), SectionKind::Text | SectionKind::Data | SectionKind::ReadOnlyData) {
            let align = section.align().max(1) as usize;
            total_size = (total_size + align - 1) & !(align - 1);
            section_offsets.insert(section.index(), total_size);
            total_size += section.size() as usize;
        }
    }

    if total_size == 0 {
        return Err(crate::Error::JitCompilation { reason: "No loadable sections in ELF".to_string() });
    }

    // Reserve space for branch veneers (trampolines) after the loadable
    // sections. Direct branches are PC-relative with a limited reach —
    // ±128 MiB for aarch64 CALL26/JUMP26, ±2 GiB for x86-64 call/jmp rel32 —
    // while external symbols (libm etc.) sit wherever the loader put them. A
    // long-lived process fills its mmap area top-down, so an anonymous JIT
    // mapping eventually lands beyond that reach and every direct call must be
    // routed through a trampoline instead.
    let veneer_base = match arch {
        Architecture::Aarch64 | Architecture::X86_64 => {
            let n = count_external_branch_targets(&elf, &section_offsets, arch);
            let base = total_size.next_multiple_of(VENEER_ALIGN);
            total_size = base + n * VENEER_SIZE;
            base
        }
        _ => total_size,
    };

    // Allocate and populate mmap.
    let mut mmap = memmap2::MmapMut::map_anon(total_size).jit("allocate anon mmap")?;

    for section in elf.sections() {
        if let Some(&offset) = section_offsets.get(&section.index()) {
            let data = section.data().jit("read ELF section")?;
            mmap[offset..offset + data.len()].copy_from_slice(data);
        }
    }

    // Build symbol address table.
    let mmap_base = mmap.as_ptr() as u64;
    let mut symbol_addrs: HashMap<object::SymbolIndex, u64> = HashMap::new();

    for symbol in elf.symbols() {
        if let Some(&sec_offset) = symbol.section_index().and_then(|si| section_offsets.get(&si)) {
            symbol_addrs.insert(symbol.index(), mmap_base + sec_offset as u64 + symbol.address());
        }
    }

    // Apply relocations.
    let mut state = RelocState { veneers: VeneerPool::new(veneer_base), ..Default::default() };

    // PPC64 ELFv2: find TOC base for TOC-relative relocations.
    if arch == Architecture::PowerPc64 {
        state.toc_base = elf.symbols().find(|s| s.name() == Ok(".TOC.")).and_then(|s| {
            s.section_index().and_then(|si| section_offsets.get(&si)).map(|&off| mmap_base + off as u64 + s.address())
        });
    }

    for section in elf.sections() {
        if !matches!(section.kind(), SectionKind::Text | SectionKind::Data | SectionKind::ReadOnlyData) {
            continue;
        }
        let Some(&sec_offset) = section_offsets.get(&section.index()) else { continue };

        for (reloc_offset, reloc) in section.relocations() {
            let patch_mmap_offset = sec_offset + reloc_offset as usize;
            let patch_addr = mmap_base + patch_mmap_offset as u64;

            let target_addr = match reloc.target() {
                object::RelocationTarget::Symbol(sym_idx) => {
                    if let Some(&addr) = symbol_addrs.get(&sym_idx) {
                        addr
                    } else {
                        let sym = elf.symbol_by_index(sym_idx).jit("bad symbol index")?;
                        let sym_name = sym.name().jit("bad symbol name")?;
                        let addr = resolve_symbol(sym_name)?;
                        symbol_addrs.insert(sym_idx, addr);
                        addr
                    }
                }
                object::RelocationTarget::Section(sec_idx) => section_offsets
                    .get(&sec_idx)
                    .map(|&off| mmap_base + off as u64)
                    .ok_or_else(|| crate::Error::JitCompilation {
                        reason: format!("Relocation references unloaded section {sec_idx:?}"),
                    })?,
                other => {
                    return Err(crate::Error::JitCompilation {
                        reason: format!("Unsupported relocation target: {other:?}"),
                    });
                }
            };

            let r_type = match reloc.flags() {
                RelocationFlags::Elf { r_type } => r_type,
                other => {
                    return Err(crate::Error::JitCompilation {
                        reason: format!("Non-ELF relocation format: {other:?}"),
                    });
                }
            };

            apply_relocation(
                &mut mmap,
                patch_mmap_offset,
                patch_addr,
                target_addr,
                reloc.addend(),
                r_type,
                arch,
                &mut state,
            )?;
        }
    }

    // Find kernel entry point.
    let fn_offset = find_symbol_offset(&elf, name, &section_offsets)?;

    // mprotect to executable.
    unsafe {
        let ret = libc::mprotect(mmap.as_ptr() as *mut libc::c_void, mmap.len(), libc::PROT_READ | libc::PROT_EXEC);
        if ret != 0 {
            return Err(std::io::Error::last_os_error()).jit("mprotect to executable");
        }
    }

    // Flush instruction cache on architectures with non-coherent I/D caches.
    #[cfg(not(target_arch = "x86_64"))]
    unsafe {
        unsafe extern "C" {
            fn __clear_cache(start: *mut libc::c_void, end: *mut libc::c_void);
        }
        __clear_cache(mmap.as_ptr() as *mut _, mmap.as_ptr().add(mmap.len()) as *mut _);
    }

    Ok(((mmap_base + fn_offset as u64) as *const (), mmap))
}

// ── Relocation dispatch ─────────────────────────────────────────────────────

/// Auxiliary state for multi-instruction relocations.
#[derive(Default)]
struct RelocState {
    /// RISC-V: maps patch address of PCREL_HI20 → full (S + A - P) value
    /// for subsequent LO12 relocations that reference the same label.
    pcrel_hi: HashMap<u64, i64>,
    /// PPC64 ELFv2: TOC base address (.TOC. symbol), needed for TOC16 relocations.
    toc_base: Option<u64>,
    /// Veneer (branch trampoline) pool for direct branches that exceed the
    /// architecture's PC-relative reach.
    veneers: VeneerPool,
}

/// Pool of branch veneers (trampolines) for direct-branch range overflow.
///
/// When the target of a direct branch is out of reach, we emit a small
/// trampoline inside the JIT mapping — always within reach of the call site —
/// that materialises the full 64-bit address and branches indirectly:
///
/// ```text
///   aarch64:            x86-64:
///   LDR X16, [PC, #8]   MOVABS $target, %r11
///   BR  X16             JMP    *%r11
///   .quad <target>
/// ```
///
/// Both scratch registers are call-clobbered and never carry arguments under
/// their respective C ABIs, so the trampoline is transparent to the callee.
#[derive(Default)]
struct VeneerPool {
    /// Next available offset for a new veneer.
    next: usize,
    /// Reuse map: target address → veneer mmap offset (avoid duplicate veneers).
    map: HashMap<u64, usize>,
}

const VENEER_SIZE: usize = 16; // aarch64: LDR X16 + BR X16 + .quad; x86-64: MOVABS + JMP (13B, padded)
const VENEER_ALIGN: usize = 16;
const CALL26_MAX: i64 = (1 << 27) - 4; // ±128 MiB (signed 28-bit range, 4-byte aligned)

impl VeneerPool {
    fn new(base: usize) -> Self {
        Self { next: base, map: HashMap::new() }
    }

    /// Reserve the next veneer slot for `target_addr`, or return the offset of
    /// an existing one. The flag is `true` when the slot is fresh and the
    /// caller still has to write the trampoline into it.
    fn reserve(&mut self, mmap_len: usize, target_addr: u64) -> (usize, bool) {
        if let Some(&off) = self.map.get(&target_addr) {
            return (off, false);
        }
        let off = self.next;
        self.next += VENEER_SIZE;
        debug_assert!(self.next <= mmap_len, "veneer pool overflow");
        self.map.insert(target_addr, off);
        (off, true)
    }

    /// Get or create an aarch64 veneer for `target_addr`, returning its mmap offset.
    fn get_or_create(&mut self, mmap: &mut [u8], target_addr: u64) -> usize {
        let (off, fresh) = self.reserve(mmap.len(), target_addr);
        if fresh {
            // LDR X16, [PC, #8]  →  0x58000050
            mmap[off..off + 4].copy_from_slice(&0x5800_0050u32.to_le_bytes());
            // BR X16              →  0xD61F0200
            mmap[off + 4..off + 8].copy_from_slice(&0xD61F_0200u32.to_le_bytes());
            // .quad target_addr
            mmap[off + 8..off + 16].copy_from_slice(&target_addr.to_le_bytes());
        }
        off
    }

    /// Get or create an x86-64 veneer for `target_addr`, returning its mmap offset.
    fn get_or_create_x86_64(&mut self, mmap: &mut [u8], target_addr: u64) -> usize {
        let (off, fresh) = self.reserve(mmap.len(), target_addr);
        if fresh {
            // MOVABS $target_addr, %r11  →  49 BB <imm64>
            mmap[off..off + 2].copy_from_slice(&[0x49, 0xBB]);
            mmap[off + 2..off + 10].copy_from_slice(&target_addr.to_le_bytes());
            // JMP *%r11                  →  41 FF E3
            mmap[off + 10..off + 13].copy_from_slice(&[0x41, 0xFF, 0xE3]);
        }
        off
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_relocation(
    mmap: &mut memmap2::MmapMut,
    off: usize,
    patch: u64,
    target: u64,
    addend: i64,
    r_type: object::elf::RelocationType,
    arch: Architecture,
    state: &mut RelocState,
) -> crate::Result<()> {
    match arch {
        Architecture::X86_64 => reloc_x86_64(mmap, off, patch, target, addend, r_type, state),
        Architecture::Aarch64 => reloc_aarch64(mmap, off, patch, target, addend, r_type, state),
        Architecture::Riscv64 => reloc_riscv64(mmap, off, patch, target, addend, r_type, state),
        Architecture::LoongArch64 => reloc_loongarch64(mmap, off, patch, target, addend, r_type),
        Architecture::PowerPc64 => reloc_ppc64(mmap, off, patch, target, addend, r_type, state),
        other => Err(unsupported_arch(other)),
    }
}

// ── x86_64 relocations ─────────────────────────────────────────────────────

fn reloc_x86_64(
    mmap: &mut [u8],
    off: usize,
    patch: u64,
    target: u64,
    addend: i64,
    r_type: object::elf::RelocationType,
    state: &mut RelocState,
) -> crate::Result<()> {
    use object::elf::*;
    match r_type {
        // S + A - P, 32-bit PC-relative
        R_X86_64_PC32 | R_X86_64_PLT32 | R_X86_64_GOTPCRELX | R_X86_64_REX_GOTPCRELX => {
            let displacement = target as i64 + addend - patch as i64;
            let v = match i32::try_from(displacement) {
                Ok(v) => v,
                // Beyond ±2 GiB. A direct `call`/`jmp rel32` can still reach the
                // symbol through a veneer emitted inside this mapping; anything
                // else (RIP-relative data, GOT loads) has no such escape and
                // must fail loudly rather than branch into truncated garbage.
                Err(_) => {
                    let is_direct_branch = matches!(r_type, R_X86_64_PC32 | R_X86_64_PLT32)
                        && matches!(off.checked_sub(1).map(|i| mmap[i]), Some(0xE8 | 0xE9));
                    if !is_direct_branch {
                        return Err(out_of_range_reloc("x86_64", r_type, displacement));
                    }
                    // `call`/`jmp rel32` resume at `patch + 4`, so the absolute
                    // destination is `S + A + 4` and the veneer is reached with
                    // the same +4 correction folded back in.
                    let mmap_base = patch - off as u64;
                    let veneer_off = state.veneers.get_or_create_x86_64(mmap, (target as i64 + addend + 4) as u64);
                    let hop = mmap_base as i64 + veneer_off as i64 - 4 - patch as i64;
                    i32::try_from(hop).map_err(|_| out_of_range_reloc("x86_64", r_type, hop))?
                }
            };
            mmap[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        // S + A, signed 32-bit
        R_X86_64_32S => {
            let value = target as i64 + addend;
            let v = i32::try_from(value).map_err(|_| out_of_range_reloc("x86_64", r_type, value))?;
            mmap[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        // S + A, unsigned 32-bit
        R_X86_64_32 => {
            let value = target as i64 + addend;
            let v = u32::try_from(value).map_err(|_| out_of_range_reloc("x86_64", r_type, value))?;
            mmap[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        // S + A, 64-bit
        R_X86_64_64 => {
            let v = (target as i64 + addend) as u64;
            mmap[off..off + 8].copy_from_slice(&v.to_le_bytes());
        }
        _ => return Err(unsupported_reloc("x86_64", r_type)),
    }
    Ok(())
}

// ── aarch64 relocations ────────────────────────────────────────────────────

fn reloc_aarch64(
    mmap: &mut [u8],
    off: usize,
    patch: u64,
    target: u64,
    addend: i64,
    r_type: object::elf::RelocationType,
    state: &mut RelocState,
) -> crate::Result<()> {
    use object::elf::*;
    match r_type {
        // 26-bit PC-relative branch: (S+A-P)>>2, encoded in [25:0]
        // Range: ±128 MiB. Use a veneer when the target is out of range.
        R_AARCH64_CALL26 | R_AARCH64_JUMP26 => {
            let dest = (target as i64).wrapping_add(addend);
            let offset = dest.wrapping_sub(patch as i64);
            let final_offset = if !(-CALL26_MAX..=CALL26_MAX).contains(&offset) {
                let mmap_base = patch - off as u64;
                let veneer_off = state.veneers.get_or_create(mmap, dest as u64);
                mmap_base as i64 + veneer_off as i64 - patch as i64
            } else {
                offset
            };
            let imm26 = (final_offset >> 2) as u32 & 0x03FF_FFFF;
            patch_insn(mmap, off, 0xFC00_0000, imm26);
        }
        // ADRP page-relative: ((S+A) & ~0xFFF) - (P & ~0xFFF), split into immlo[30:29] immhi[23:5]
        R_AARCH64_ADR_PREL_PG_HI21 => {
            let page_delta = ((target as i64 + addend) & !0xFFF) - (patch as i64 & !0xFFF);
            let imm = (page_delta >> 12) as u32;
            patch_insn(mmap, off, 0x9F00_001F, ((imm & 0x3) << 29) | (((imm >> 2) & 0x7FFFF) << 5));
        }
        // Low 12-bit page offset for ADD/LDR/STR: (S+A) & 0xFFF, shifted by access size
        R_AARCH64_ADD_ABS_LO12_NC
        | R_AARCH64_LDST8_ABS_LO12_NC
        | R_AARCH64_LDST16_ABS_LO12_NC
        | R_AARCH64_LDST32_ABS_LO12_NC
        | R_AARCH64_LDST64_ABS_LO12_NC
        | R_AARCH64_LDST128_ABS_LO12_NC => {
            let shift = match r_type {
                R_AARCH64_LDST16_ABS_LO12_NC => 1,
                R_AARCH64_LDST32_ABS_LO12_NC => 2,
                R_AARCH64_LDST64_ABS_LO12_NC => 3,
                R_AARCH64_LDST128_ABS_LO12_NC => 4,
                _ => 0,
            };
            let imm12 = (((target as i64 + addend) as u32) & 0xFFF) >> shift;
            patch_insn(mmap, off, 0xFFC0_03FF, imm12 << 10);
        }
        _ => return Err(unsupported_reloc("aarch64", r_type)),
    }
    Ok(())
}

// ── RISC-V relocations ─────────────────────────────────────────────────────
//
// Instruction formats referenced below:
//   U-type (lui, auipc):  imm[31:12] | rd[11:7]  | opcode[6:0]
//   I-type (jalr, loads):  imm[31:20] | rs1[19:15] | f3[14:12] | rd[11:7]  | opcode[6:0]
//   S-type (stores):  imm[31:25] | rs2[24:20] | rs1[19:15] | f3[14:12] | imm[11:7] | opcode[6:0]
//   B-type (branches): imm[31] | imm[30:25] | rs2[24:20] | rs1[19:15] | f3[14:12] | imm[11:8] | imm[7] | opcode[6:0]
//   J-type (jal):  imm[31] | imm[30:21] | imm[20] | imm[19:12] | rd[11:7] | opcode[6:0]

const RV_U_MASK: u32 = 0x0000_0FFF; // preserve rd + opcode
const RV_I_MASK: u32 = 0x000F_FFFF; // preserve rs1 + f3 + rd + opcode
const RV_S_MASK: u32 = 0x01FF_F07F; // preserve rs2 + rs1 + f3 + opcode

fn reloc_riscv64(
    mmap: &mut [u8],
    off: usize,
    patch: u64,
    target: u64,
    addend: i64,
    r_type: object::elf::RelocationType,
    state: &mut RelocState,
) -> crate::Result<()> {
    use object::elf::*;
    match r_type {
        // auipc+jalr pair: S+A-P split into hi20 (U-type) + lo12 (I-type)
        R_RISCV_CALL | R_RISCV_CALL_PLT => {
            let v = target as i64 + addend - patch as i64;
            let hi = ((v + 0x800) >> 12) as u32;
            let lo = (v as u32) & 0xFFF;
            patch_insn(mmap, off, RV_U_MASK, hi << 12);
            patch_insn(mmap, off + 4, RV_I_MASK, lo << 20);
        }
        // PC-relative high 20 bits (auipc) — store full value for paired LO12
        R_RISCV_PCREL_HI20 => {
            let v = target as i64 + addend - patch as i64;
            let hi = ((v + 0x800) >> 12) as u32;
            patch_insn(mmap, off, RV_U_MASK, hi << 12);
            state.pcrel_hi.insert(patch, v);
        }
        // PC-relative low 12 bits (I-type), paired with PCREL_HI20
        R_RISCV_PCREL_LO12_I => {
            let full = *state.pcrel_hi.get(&target).ok_or_else(|| crate::Error::JitCompilation {
                reason: format!("PCREL_LO12_I: no paired HI20 at {target:#x}"),
            })?;
            patch_insn(mmap, off, RV_I_MASK, ((full as u32) & 0xFFF) << 20);
        }
        // PC-relative low 12 bits (S-type), paired with PCREL_HI20
        R_RISCV_PCREL_LO12_S => {
            let full = *state.pcrel_hi.get(&target).ok_or_else(|| crate::Error::JitCompilation {
                reason: format!("PCREL_LO12_S: no paired HI20 at {target:#x}"),
            })?;
            let lo = (full as u32) & 0xFFF;
            patch_insn(mmap, off, RV_S_MASK, ((lo >> 5) << 25) | ((lo & 0x1F) << 7));
        }
        // Absolute high 20 bits (lui)
        R_RISCV_HI20 => {
            let v = (target as i64 + addend) as u32;
            patch_insn(mmap, off, RV_U_MASK, v.wrapping_add(0x800) & 0xFFFF_F000);
        }
        // Absolute low 12 bits (I-type)
        R_RISCV_LO12_I => {
            let lo = ((target as i64 + addend) as u32) & 0xFFF;
            patch_insn(mmap, off, RV_I_MASK, lo << 20);
        }
        // Absolute low 12 bits (S-type)
        R_RISCV_LO12_S => {
            let lo = ((target as i64 + addend) as u32) & 0xFFF;
            patch_insn(mmap, off, RV_S_MASK, ((lo >> 5) << 25) | ((lo & 0x1F) << 7));
        }
        // 12-bit PC-relative branch (B-type)
        R_RISCV_BRANCH => {
            let v = (target as i64 + addend - patch as i64) as u32;
            let bits = ((v >> 12) & 1) << 31 | ((v >> 5) & 0x3F) << 25 | ((v >> 1) & 0xF) << 8 | ((v >> 11) & 1) << 7;
            patch_insn(mmap, off, RV_S_MASK, bits);
        }
        // 20-bit PC-relative jump (J-type)
        R_RISCV_JAL => {
            let v = (target as i64 + addend - patch as i64) as u32;
            let bits =
                ((v >> 20) & 1) << 31 | ((v >> 1) & 0x3FF) << 21 | ((v >> 11) & 1) << 20 | ((v >> 12) & 0xFF) << 12;
            patch_insn(mmap, off, RV_U_MASK, bits);
        }
        // Data relocations
        R_RISCV_64 => {
            let v = (target as i64 + addend) as u64;
            mmap[off..off + 8].copy_from_slice(&v.to_le_bytes());
        }
        R_RISCV_32 => {
            let v = (target as i64 + addend) as u32;
            mmap[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        // Linker relaxation hint — skip
        R_RISCV_RELAX => {}
        _ => return Err(unsupported_reloc("riscv64", r_type)),
    }
    Ok(())
}

// ── LoongArch relocations ──────────────────────────────────────────────────
//
// Instruction formats referenced below:
//   1RI20 (pcalau12i): opcode[31:25] | si20[24:5] | rd[4:0]
//   2RI12 (addi/ld/st): opcode[31:22] | si12[21:10] | rj[9:5] | rd[4:0]
//   I26   (b/bl):       opcode[31:26] | offs[15:0 in 25:10] | offs[25:16 in 9:0]

fn reloc_loongarch64(
    mmap: &mut [u8],
    off: usize,
    patch: u64,
    target: u64,
    addend: i64,
    r_type: object::elf::RelocationType,
) -> crate::Result<()> {
    use object::elf::*;
    match r_type {
        // 26-bit PC-relative branch (B/BL)
        R_LARCH_B26 => {
            let offs = ((target as i64 + addend - patch as i64) >> 2) as u32;
            patch_insn(mmap, off, 0xFC00_0000, ((offs & 0xFFFF) << 10) | ((offs >> 16) & 0x3FF));
        }
        // PC-aligned page high 20 bits (pcalau12i): si20 in [24:5]
        R_LARCH_PCALA_HI20 => {
            let page_delta = ((target as i64 + addend + 0x800) >> 12) - (patch as i64 >> 12);
            patch_insn(mmap, off, 0xFE00_001F, ((page_delta as u32) & 0xF_FFFF) << 5);
        }
        // Low 12 bits (2RI12 format): si12 in [21:10]
        R_LARCH_PCALA_LO12 => {
            let lo12 = ((target as i64 + addend) as u32) & 0xFFF;
            patch_insn(mmap, off, 0xFFC0_03FF, lo12 << 10);
        }
        // Data relocations
        R_LARCH_64 => {
            let v = (target as i64 + addend) as u64;
            mmap[off..off + 8].copy_from_slice(&v.to_le_bytes());
        }
        R_LARCH_32 => {
            let v = (target as i64 + addend) as u32;
            mmap[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        // Linker relaxation hint — skip
        R_LARCH_RELAX => {}
        _ => return Err(unsupported_reloc("loongarch64", r_type)),
    }
    Ok(())
}

// ── PPC64 relocations ───────────────────────────────────────────────────────
//
// PPC64 ELFv2 ABI (little-endian). Instructions are 32-bit, stored LE on PPC64LE.
//   I-form  (b/bl):    OPCD[31:26] | LI[25:2] | AA[1] | LK[0]
//   D-form  (addi/lwz): OPCD[31:26] | RT[25:21] | RA[20:16] | D[15:0]
//   DS-form (ld/std):   OPCD[31:26] | RT[25:21] | RA[20:16] | DS[15:2] | XO[1:0]

fn reloc_ppc64(
    mmap: &mut [u8],
    off: usize,
    patch: u64,
    target: u64,
    addend: i64,
    r_type: object::elf::RelocationType,
    state: &mut RelocState,
) -> crate::Result<()> {
    use object::elf::*;
    match r_type {
        // 24-bit PC-relative branch (bl): LI in [25:2]
        R_PPC64_REL24 => {
            let li = ((target as i64 + addend - patch as i64) >> 2) as u32 & 0x00FF_FFFF;
            patch_insn(mmap, off, 0xFC00_0003, li << 2);
        }
        // 16-bit TOC-relative, high adjusted: ha16(S + A - .TOC.)
        R_PPC64_TOC16_HA => {
            let toc = toc_base(state)?;
            let v = target as i64 + addend - toc as i64;
            let ha = (((v >> 16) as u32).wrapping_add((v as u32 >> 15) & 1)) & 0xFFFF;
            patch_insn(mmap, off, 0xFFFF_0000, ha);
        }
        // 16-bit TOC-relative, low: lo16(S + A - .TOC.)
        R_PPC64_TOC16_LO => {
            let toc = toc_base(state)?;
            let lo = ((target as i64 + addend - toc as i64) as u32) & 0xFFFF;
            patch_insn(mmap, off, 0xFFFF_0000, lo);
        }
        // 16-bit TOC-relative, low DS-form: lo16(S + A - .TOC.) with bits [1:0] preserved
        R_PPC64_TOC16_LO_DS => {
            let toc = toc_base(state)?;
            let lo = ((target as i64 + addend - toc as i64) as u32) & 0xFFFC;
            patch_insn(mmap, off, 0xFFFF_0003, lo);
        }
        // 32-bit PC-relative
        R_PPC64_REL32 => {
            let v = (target as i64 + addend - patch as i64) as i32;
            mmap[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        // Data relocations
        R_PPC64_ADDR64 => {
            let v = (target as i64 + addend) as u64;
            mmap[off..off + 8].copy_from_slice(&v.to_le_bytes());
        }
        R_PPC64_ADDR32 => {
            let v = (target as i64 + addend) as u32;
            mmap[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        // TOC-relative 16-bit halves: low or high half of (target - TOC base).
        R_PPC64_TOC16 | R_PPC64_TOC16_HI => {
            let toc = toc_base(state)?;
            let v = target as i64 + addend - toc as i64;
            let bits = match r_type {
                R_PPC64_TOC16 => (v as u32) & 0xFFFF,
                R_PPC64_TOC16_HI => ((v >> 16) as u32) & 0xFFFF,
                _ => unreachable!(),
            };
            patch_insn(mmap, off, 0xFFFF_0000, bits);
        }
        _ => return Err(unsupported_reloc("ppc64", r_type)),
    }
    Ok(())
}

fn toc_base(state: &RelocState) -> crate::Result<u64> {
    state.toc_base.ok_or_else(|| crate::Error::JitCompilation {
        reason: "PPC64 TOC relocation but no .TOC. symbol found in ELF".to_string(),
    })
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Read-modify-write a 32-bit LE instruction: `insn = (insn & mask) | bits`.
fn patch_insn(mmap: &mut [u8], off: usize, mask: u32, bits: u32) {
    let insn = u32::from_le_bytes(mmap[off..off + 4].try_into().unwrap());
    mmap[off..off + 4].copy_from_slice(&((insn & mask) | bits).to_le_bytes());
}

fn unsupported_reloc(arch: &str, r_type: object::elf::RelocationType) -> crate::Error {
    crate::Error::JitCompilation { reason: format!("Unsupported {arch} relocation type: {r_type}") }
}

fn out_of_range_reloc(arch: &str, r_type: object::elf::RelocationType, value: i64) -> crate::Error {
    crate::Error::JitCompilation {
        reason: format!("{arch} relocation type {r_type} out of range: {value:#x} does not fit its field"),
    }
}

fn unsupported_arch(arch: Architecture) -> crate::Error {
    crate::Error::JitCompilation { reason: format!("Unsupported ELF architecture: {arch:?}") }
}

/// Look up a symbol's offset within the mmap by name.
fn find_symbol_offset(
    elf: &object::File,
    name: &str,
    section_offsets: &HashMap<object::SectionIndex, usize>,
) -> crate::Result<usize> {
    let prefixed = format!("_{name}");
    for symbol in elf.symbols() {
        let sym_name = symbol.name().unwrap_or("");
        if (sym_name == name || sym_name == prefixed)
            && let Some(&sec_offset) = symbol.section_index().and_then(|si| section_offsets.get(&si))
        {
            return Ok(sec_offset + symbol.address() as usize);
        }
    }
    Err(crate::Error::FunctionNotFound { name: name.to_string() })
}

/// Count unique external symbols reachable only through a direct branch.
/// Used to pre-allocate veneer space before mmap: only symbols resolved outside
/// the mapping can land beyond the branch's PC-relative reach.
fn count_external_branch_targets(
    elf: &object::File,
    section_offsets: &HashMap<object::SectionIndex, usize>,
    arch: Architecture,
) -> usize {
    use std::collections::HashSet;
    let mut external = HashSet::new();
    for section in elf.sections() {
        if !matches!(section.kind(), SectionKind::Text | SectionKind::Data | SectionKind::ReadOnlyData) {
            continue;
        }
        for (_, reloc) in section.relocations() {
            let r_type = match reloc.flags() {
                RelocationFlags::Elf { r_type } => r_type,
                _ => continue,
            };
            let branch = match arch {
                Architecture::Aarch64 => {
                    matches!(r_type, object::elf::R_AARCH64_CALL26 | object::elf::R_AARCH64_JUMP26)
                }
                Architecture::X86_64 => matches!(r_type, object::elf::R_X86_64_PC32 | object::elf::R_X86_64_PLT32),
                _ => false,
            };
            if !branch {
                continue;
            }
            if let object::RelocationTarget::Symbol(sym_idx) = reloc.target()
                && let Ok(sym) = elf.symbol_by_index(sym_idx)
                && sym.section_index().and_then(|si| section_offsets.get(&si)).is_none()
            {
                external.insert(sym_idx);
            }
        }
    }
    external.len()
}

/// Resolve an external symbol (e.g. `sqrtf`, `expf`) via dlsym at runtime.
fn resolve_symbol(name: &str) -> crate::Result<u64> {
    let cname = std::ffi::CString::new(name).jit("symbol name has interior NUL")?;
    let ptr = unsafe { libc::dlsym(libc::RTLD_DEFAULT, cname.as_ptr()) };
    if ptr.is_null() {
        return Err(crate::Error::JitCompilation { reason: format!("Cannot resolve symbol: {name}") });
    }
    Ok(ptr as u64)
}

#[cfg(test)]
#[path = "test/unit/jit_loader.rs"]
mod tests;
