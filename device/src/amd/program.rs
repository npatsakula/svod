//! `AmdProgram`: load an AMDGPU code object + dispatch it via AQL.
//!
//! Construction parses the ELF returned by `compile_ir_to_amd_object`
//! and resolves the kernel descriptor (symbol `<name>.kd`). Execution claims
//! a kernarg slot from the device arena, fills it with buffer GVAs + scalar
//! vals, builds an AQL dispatch packet, and waits on the device timeline
//! signal.

#![cfg(unix)]

use std::sync::Arc;

use object::elf::{ELFCLASS64, ELFDATA2LSB, EM_AMDGPU};
use object::read::elf::FileHeader;
use object::{LittleEndian, Object, ObjectSection, ObjectSymbol, RelocationFlags, RelocationTarget};
use tracing::debug;

use crate::allocator::{Allocator, AmdBufferGuard, BufferSpec, RawBuffer};
use crate::amd::AmdAllocator;
use crate::amd::device::AmdDevice;
use crate::amd::sys::hsa::AmdHsaKernelDescriptor;
use crate::device::Program;
use crate::error::{Error, Result};

// AMDGPU relocation types per LLVM `ELFRelocs/AMDGPU.def`. Only the 64-bit
// kinds that clang emits for our codegen reach us; anything else surfaces as
// a clean `Runtime` error rather than a silent zero-write.
const R_AMDGPU_ABS64: u32 = 3;
const R_AMDGPU_REL32: u32 = 4;
const R_AMDGPU_REL64: u32 = 5;

/// Pre-execution metadata extracted from a single kernel's ELF.
#[derive(Debug, Clone)]
pub struct ParsedKernel {
    /// Bytes of the laid-out code object (PROGBITS sections placed at their
    /// expected runtime offsets).
    pub image: Vec<u8>,
    /// Offset of the `<name>.kd` symbol inside `image`.
    pub kd_offset: u64,
    /// Decoded kernel descriptor.
    pub kd: AmdHsaKernelDescriptor,
}

/// Parse an AMDGPU code-object ELF and resolve the named kernel descriptor.
///
/// PT_LOAD segments stay at their declared file offsets, no-vaddr sections get
/// appended aligned, then R_AMDGPU_REL64 / R_AMDGPU_ABS64 relocations get
/// applied against the symbol table.
pub fn parse_kernel(bytes: &[u8], kernel_name: &str) -> Result<ParsedKernel> {
    // ── 1. Quick header sanity. ──────────────────────────────────────────
    if bytes.len() < 64 || &bytes[..4] != b"\x7fELF" {
        return Err(Error::Runtime { message: "AMD program: input is not an ELF".into() });
    }
    let header = object::elf::FileHeader64::<LittleEndian>::parse(bytes)
        .map_err(|e| Error::Runtime { message: format!("AMD ELF parse: {e}") })?;
    let endian = header.endian().map_err(|e| Error::Runtime { message: format!("AMD ELF endian: {e}") })?;
    if header.e_ident().class != ELFCLASS64 || header.e_ident().data != ELFDATA2LSB {
        return Err(Error::Runtime { message: "AMD ELF must be ELF64 LE".into() });
    }
    if header.e_machine.get(endian) != EM_AMDGPU {
        return Err(Error::Runtime {
            message: format!("AMD ELF e_machine = {} (expected EM_AMDGPU=224)", header.e_machine.get(endian)),
        });
    }

    // ── 2. Build the laid-out image (section-based, handles ET_REL+ET_DYN).
    // SHF_ALLOC sections with a non-zero sh_addr go at their declared
    // virtual address; address-0 sections get appended aligned to the
    // running image end. The high-level object::File walk gives us
    // SectionKind and address + size + data uniformly.
    let file = object::File::parse(bytes).map_err(|e| Error::Runtime { message: format!("AMD ELF object: {e}") })?;
    use object::SectionFlags;
    let mut image: Vec<u8> = Vec::new();
    let mut placements: Vec<(object::SectionIndex, u64, u64)> = Vec::new(); // (idx, addr, size)
    // First pass: place sections with sh_addr != 0 directly.
    for section in file.sections() {
        let alloc =
            matches!(section.flags(), SectionFlags::Elf { sh_flags, .. } if sh_flags.contains(object::elf::SHF_ALLOC));
        if !alloc || section.size() == 0 {
            continue;
        }
        let addr = section.address();
        if addr == 0 {
            continue;
        }
        let end = addr as usize + section.size() as usize;
        if image.len() < end {
            image.resize(end, 0);
        }
        if let Ok(data) = section.data() {
            image[addr as usize..addr as usize + data.len()].copy_from_slice(data);
        }
        placements.push((section.index(), addr, section.size()));
    }
    // Second pass: append address-0 SHF_ALLOC sections aligned by sh_addralign.
    let mut zero_addr_remap: std::collections::HashMap<object::SectionIndex, u64> = std::collections::HashMap::new();
    for section in file.sections() {
        let alloc =
            matches!(section.flags(), SectionFlags::Elf { sh_flags, .. } if sh_flags.contains(object::elf::SHF_ALLOC));
        if !alloc || section.size() == 0 || section.address() != 0 {
            continue;
        }
        let align = section.align().max(1);
        let start = (image.len() as u64).next_multiple_of(align);
        let end = (start + section.size()) as usize;
        if image.len() < end {
            image.resize(end, 0);
        }
        if let Ok(data) = section.data() {
            image[start as usize..start as usize + data.len()].copy_from_slice(data);
        }
        zero_addr_remap.insert(section.index(), start);
        placements.push((section.index(), start, section.size()));
    }
    if image.is_empty() {
        return Err(Error::Runtime { message: "AMD ELF has no SHF_ALLOC sections to load".into() });
    }
    let _ = placements;

    // ── 3. Find the kernel descriptor symbol. ────────────────────────────
    let mut kd_offset = None;
    let kd_name = format!("{kernel_name}.kd");
    for sym in file.symbols() {
        if sym.name().unwrap_or("") != kd_name {
            continue;
        }
        // For section-relative symbols, sym.address() gives the absolute VA
        // assuming the section is at its declared sh_addr. We patch up
        // address-0 sections via `zero_addr_remap`.
        let sec_idx = sym.section_index();
        let base = match sec_idx {
            Some(idx) => zero_addr_remap.get(&idx).copied().unwrap_or(0),
            None => 0,
        };
        kd_offset = Some(base + sym.address());
        break;
    }
    let kd_offset = kd_offset.ok_or_else(|| Error::Runtime {
        message: format!("AMD ELF: kernel descriptor symbol '{kd_name}' not found"),
    })?;

    // ── 4. Apply relocations. ────────────────────────────────────────────
    // Use the high-level object::File API: iterate sections, then per-section
    // relocations. AMDGPU uses RELA (addends are explicit).
    //
    // For ET_REL (clang `-c` amdgcn output, which is what we get), section
    // relocation offsets and symbol addresses are SECTION-RELATIVE. We must
    // remap them to image-absolute offsets using the placement decisions
    // from steps 2a/2b above (sections placed at non-zero sh_addr stay
    // there; address-0 sections were appended via `zero_addr_remap`).
    //
    // Without this remap, the kernel descriptor's relocated entries
    // (e.g. `kernel_code_entry_byte_offset`) get written at the wrong
    // image offsets, the GPU jumps to garbage on dispatch, and the CP
    // stalls in SPI without launching any shader (radeontop: 100% spi,
    // 0% on TA/SH/SX/SMX/CB/DB).
    let image_offset = |idx: object::SectionIndex| -> Option<u64> {
        if let Some(&remapped) = zero_addr_remap.get(&idx) {
            return Some(remapped);
        }
        // Fall back to the section's declared address (already where we
        // placed it during step 2a).
        file.section_by_index(idx).ok().map(|s| s.address())
    };
    for section in file.sections() {
        let section_base = image_offset(section.index()).unwrap_or(0);
        for (sec_off, reloc) in section.relocations() {
            let r_type = match reloc.flags() {
                RelocationFlags::Elf { r_type } => r_type.0,
                _ => continue,
            };
            let sym_value: i64 = match reloc.target() {
                RelocationTarget::Symbol(sym_idx) => match file.symbol_by_index(sym_idx) {
                    Ok(sym) => {
                        let sym_base = sym.section_index().and_then(image_offset).unwrap_or(0);
                        (sym_base + sym.address()) as i64
                    }
                    Err(_) => continue,
                },
                _ => continue,
            };
            let off = (section_base + sec_off) as usize;
            match r_type {
                R_AMDGPU_ABS64 => {
                    if off + 8 > image.len() {
                        return Err(Error::Runtime { message: format!("AMD ELF: reloc out of range at {off:#x}") });
                    }
                    let value: i64 = sym_value + reloc.addend();
                    image[off..off + 8].copy_from_slice(&value.to_le_bytes());
                }
                R_AMDGPU_REL64 => {
                    if off + 8 > image.len() {
                        return Err(Error::Runtime { message: format!("AMD ELF: reloc out of range at {off:#x}") });
                    }
                    let value = sym_value + reloc.addend() - off as i64;
                    image[off..off + 8].copy_from_slice(&value.to_le_bytes());
                }
                R_AMDGPU_REL32 => {
                    if off + 4 > image.len() {
                        return Err(Error::Runtime { message: format!("AMD ELF: reloc out of range at {off:#x}") });
                    }
                    let value = (sym_value + reloc.addend() - off as i64) as i32;
                    image[off..off + 4].copy_from_slice(&value.to_le_bytes());
                }
                _ => {
                    return Err(Error::Runtime {
                        message: format!("AMD ELF: unsupported reloc type {r_type} at offset {off:#x}"),
                    });
                }
            }
        }
    }

    // ── 5. Read the 64-byte descriptor. ──────────────────────────────────
    if kd_offset as usize + std::mem::size_of::<AmdHsaKernelDescriptor>() > image.len() {
        return Err(Error::Runtime { message: "AMD ELF: kernel descriptor out of range".into() });
    }
    let kd_bytes = &image[kd_offset as usize..kd_offset as usize + std::mem::size_of::<AmdHsaKernelDescriptor>()];
    // SAFETY: AmdHsaKernelDescriptor is `#[repr(C, packed)]`, 64 bytes,
    // and we've bounded the slice exactly to that size.
    let kd: AmdHsaKernelDescriptor = unsafe { std::ptr::read_unaligned(kd_bytes.as_ptr() as *const _) };

    Ok(ParsedKernel { image, kd_offset, kd })
}

/// Loaded AMDGPU program: code object resident in VRAM + kernel metadata.
///
/// Owner-agnostic by construction — programs hold only the immutable kernel
/// descriptor (rsrc1/2/3, prog_addr, kd, arities) plus the device handle needed
/// by `Program::execute` trait callers (who don't supply an owner). Plan and
/// graph callers downcast to `AmdProgram` and route through
/// `execute_on(&OwnerCtx, …)` with their OWN owner context — so one cached
/// program safely services any number of plans on the same physical AMD:N.
#[derive(Debug)]
pub(crate) struct CodeObject {
    buffer: RawBuffer,
}

impl Drop for CodeObject {
    fn drop(&mut self) {
        self.buffer.free_amd_device_in_place();
    }
}

pub struct AmdProgram {
    name: String,
    /// Device handle — used by the `Program::execute` trait method to assign an
    /// owner (`PoolQueue`) when the caller doesn't go through
    /// `AmdProgram::execute_on`. Plan/graph callers ignore this.
    dev: Arc<AmdDevice>,
    /// Logical AMD device index (`AMD:N`), captured from the loading
    /// allocator. Lets the trait `execute` rebuild an `AmdAllocator` (cheap —
    /// shared via `DEVICE_CACHE`) to assign an owner when none is supplied.
    device_id: usize,
    /// AQL `kernel_object` field: GPU VA of the kernel descriptor inside the
    /// loaded code object. Used by the AQL kernel-dispatch packet only.
    pub(crate) aql_prog_addr: u64,
    /// PM4 shader entry point: `code_gpu + kd_offset + kernel_code_entry_byte_offset`.
    /// Used by `build_exec_pm4` (the COMPUTE_PGM_LO/HI register
    /// pair carries `prog_addr >> 8`).
    pm4_prog_addr: u64,
    /// COMPUTE_PGM_RSRC1/2/3 values for the PM4 path, derived from the
    /// kernel descriptor at load time. `rsrc1` carries the gfx11 cwsr-priv
    /// bit; `rsrc2` carries the LDS-size patch.
    rsrc1: u32,
    rsrc2: u32,
    rsrc3: u32,
    /// `(kd.kernel_code_properties & 0x400) != 0` — true for wave32 kernels
    /// (RDNA3/4 default). Controls the `cs_w32_en` bit in DISPATCH_INITIATOR.
    wave32: bool,
    /// gfx major version (9, 11, or 12). gfx9 (CDNA) ignores `cs_w32_en`.
    target_major: u32,
    /// `kernel_code_properties & ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER` — kernel
    /// reads a 4-dword scratch descriptor from user SGPRs 0-3. We prepend
    /// `[scratch_lo, scratch_hi|swizzle_bit, 0xffffffff, 0x20c14000]` to the
    /// USER_DATA registers when set.
    enable_private_segment_sgpr: bool,
    /// Decoded kernel descriptor (size of kernarg, LDS, scratch, etc.).
    pub(crate) kd: AmdHsaKernelDescriptor,
    /// Number of buffer arguments the kernel expects.
    buf_count: usize,
    /// Number of scalar (i64) variable arguments.
    var_count: usize,
    abi: Vec<crate::device::AbiParamDescriptor>,
    /// Shared by the program and every captured/in-flight command that can
    /// execute it. The allocation is released only after the last GPU use.
    code: Arc<CodeObject>,
}

impl AmdProgram {
    /// Load `bytes` (an AMDGPU code object from clang) into VRAM and resolve
    /// the named kernel using its complete, ordered PARAM ABI.
    pub fn load(
        device: Arc<AmdDevice>,
        allocator: &AmdAllocator,
        bytes: &[u8],
        kernel_name: &str,
        abi: &[crate::device::AbiParamDescriptor],
    ) -> Result<Self> {
        let (abi, buf_count, var_count) = retain_program_abi(abi)?;
        let parsed = parse_kernel(bytes, kernel_name)?;

        // Scratch is no longer ensured here — there is no shared "default
        // connector" to grow. Every dispatch site ensures scratch on the
        // connector it owns/leases before `execute_on`: `ExecutionPlan`
        // (`execution_plan.rs::execute_kernel`), `AmdGraph::capture`, and the
        // `Program::execute` trait path (which leases a connector and calls
        // `ensure_has_local_memory` on it) — scratch is ensured per-connector.

        // Allocate VRAM for the code object (EXECUTABLE flag is set on every
        // AmdAllocator alloc; clang's amdgcn output runs on the GPU side).
        let size = parsed.image.len().next_multiple_of(0x1000);
        let opts = BufferSpec { cpu_access: true, nolru: true, ..Default::default() };
        let code_buf = AmdBufferGuard::new(allocator.alloc(size, &opts, /*zero=*/ false)?);
        let (code_gpu, code_host) = match code_buf.buffer() {
            RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
            _ => return Err(Error::NotHostVisible { what: "code object" }),
        };
        // SAFETY: code_host points to size bytes we just mmapped exclusively.
        unsafe { std::ptr::copy_nonoverlapping(parsed.image.as_ptr(), code_host.as_ptr(), parsed.image.len()) };

        let aql_prog_addr = code_gpu + parsed.kd_offset;

        // Derive PM4-path fields from the kernel descriptor:
        //   lds_size = round_up(group_segment_fixed_size, 512) / 512 (clamped 9 bits)
        //   target_major: 9 = CDNA, 11/12 = RDNA3/4
        //   rsrc1 |= 1<<20 on gfx11 (cwsr-priv shim)
        //   rsrc2 |= lds_size << 15
        //   wave32 = kd.kernel_code_properties bit 10
        //   pm4_prog_addr = aql_prog_addr + kernel_code_entry_byte_offset
        let group_segment = parsed.kd.group_segment_fixed_size;
        let lds_size: u32 = ((group_segment.saturating_add(511) / 512) as u32) & 0x1FF;
        let lds_limit = device.node.lds_size_in_kb.saturating_mul(1024) / 512;
        if lds_size > lds_limit {
            return Err(Error::GroupSegmentTooLarge {
                requested: lds_size,
                limit: lds_limit,
                lds_kb: device.node.lds_size_in_kb,
            });
        }
        let target_major: u32 = device.arch.gfx_major();
        // Packed struct: copy fields to locals to avoid unaligned-ref warnings.
        let rsrc1_kd = parsed.kd.compute_pgm_rsrc1;
        let rsrc2_kd = parsed.kd.compute_pgm_rsrc2;
        let rsrc3_kd = parsed.kd.compute_pgm_rsrc3;
        let props = parsed.kd.kernel_code_properties;
        let entry = parsed.kd.kernel_code_entry_byte_offset;
        let rsrc1 = rsrc1_kd | (if target_major == 11 { 1u32 << 20 } else { 0 });
        let rsrc2 = rsrc2_kd | (lds_size << 15);
        let rsrc3 = rsrc3_kd;
        let wave32 = (props & 0x400) != 0;
        let pm4_prog_addr = aql_prog_addr.wrapping_add(entry as u64);

        // Decode KCP bits that affect the user-SGPR pre-load layout. We only
        // honour `kernarg_segment_ptr` and `private_segment_buffer` at this
        // point — `dispatch_ptr` etc. require allocating an HSA dispatch
        // packet alongside kernargs, which isn't wired up yet. Fail fast at
        // load if the kernel needs one of the unsupported bits.
        use crate::amd::sys::hsa::{
            amd_kernel_code_properties_t_AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR,
            amd_kernel_code_properties_t_AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER,
        };
        // `props` is the descriptor's u16 field; the generated constants are
        // `amd_kernel_code_properties_t` (c_int), narrowed here.
        let enable_private_segment_sgpr = (props
            & amd_kernel_code_properties_t_AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER as u16)
            != 0;
        let enable_dispatch_ptr =
            (props & amd_kernel_code_properties_t_AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR as u16) != 0;
        if enable_dispatch_ptr {
            return Err(Error::Runtime {
                message: format!(
                    "AmdProgram '{kernel_name}': kernel_code_properties={:#06x} sets \
                     ENABLE_SGPR_DISPATCH_PTR — svod doesn't allocate an HSA dispatch \
                     packet alongside kernargs yet",
                    props
                ),
            });
        }

        let kernarg_size_log = parsed.kd.kernarg_size;
        let private_seg_log = parsed.kd.private_segment_fixed_size;
        let group_seg_log = parsed.kd.group_segment_fixed_size;
        debug!(
            kernel = kernel_name,
            aql_prog_addr = aql_prog_addr,
            pm4_prog_addr = pm4_prog_addr,
            kernarg_size = kernarg_size_log,
            private_segment_fixed_size = private_seg_log,
            group_segment_fixed_size = group_seg_log,
            wave32 = wave32,
            target_major = target_major,
            "AmdProgram loaded"
        );
        if std::env::var("SVOD_DEBUG_DISPATCH").is_ok() {
            let kcp = props;
            let user_sgpr_count = (rsrc2_kd >> 1) & 0x1F;
            eprintln!(
                "[program-load] kernel={} kernarg_size={} private_seg={} group_seg={} \
                 kernel_code_properties={:#06x} user_sgpr_count={} wave32={} \
                 rsrc1_kd={:#x} rsrc2_kd={:#x} rsrc3_kd={:#x}",
                kernel_name,
                kernarg_size_log,
                private_seg_log,
                group_seg_log,
                kcp,
                user_sgpr_count,
                wave32,
                rsrc1_kd,
                rsrc2_kd,
                rsrc3_kd
            );
            // Decode kernel_code_properties bits that affect SGPR pre-load layout.
            // If any of bits 0-6 are set besides bit 3 (kernarg_segment_ptr),
            // the kernel expects additional values in user SGPRs which we
            // currently DO NOT populate — causing the kernel to read garbage
            // pointers and fault at random addresses.
            let bits = [
                (0, "private_segment_buffer"),
                (1, "dispatch_ptr"),
                (2, "queue_ptr"),
                (3, "kernarg_segment_ptr"),
                (4, "dispatch_id"),
                (5, "flat_scratch_init"),
                (6, "private_segment_size"),
                (10, "wavefront_size32"),
            ];
            let set: Vec<&str> = bits.iter().filter(|(b, _)| (kcp & (1u16 << b)) != 0).map(|(_, n)| *n).collect();
            eprintln!("[program-load]   enabled bits: {:?}", set);
            // Diagnostic: confirm the kd relocation produced the right delta.
            // `entry` is the relocated `kernel_code_entry_byte_offset` field —
            // signed i64, expected to be `(text_image_off - rodata_image_off)`.
            // pm4_prog_addr = code_gpu + kd_offset + entry should equal
            // code_gpu + text_image_off (the actual kernel code address).
            // If `entry` is 0 (unrelocated) or any wrong value, the GPU jumps
            // somewhere other than the kernel entry and SGPRs get scrambled.
            eprintln!(
                "[program-load]   relocation check: kd_offset={:#x} entry_byte_offset={} ({:#x}) \
                 code_gpu={:#x} aql_prog_addr={:#x} pm4_prog_addr={:#x} \
                 image_len={} kd_offset+entry={:#x}",
                parsed.kd_offset,
                entry,
                entry as u64,
                code_gpu,
                aql_prog_addr,
                pm4_prog_addr,
                parsed.image.len(),
                (parsed.kd_offset as i64 + entry) as u64,
            );
        }

        Ok(Self {
            name: kernel_name.to_string(),
            device_id: allocator.device_id,
            dev: device,
            aql_prog_addr,
            pm4_prog_addr,
            rsrc1,
            rsrc2,
            rsrc3,
            wave32,
            target_major,
            enable_private_segment_sgpr,
            kd: parsed.kd,
            buf_count,
            var_count,
            abi,
            code: Arc::new(CodeObject { buffer: code_buf.into_inner() }),
        })
    }

    pub(crate) fn kernarg_size(&self) -> usize {
        // KFD-side kernarg_size is the byte count for the entire kernarg
        // record (already includes alignment padding).
        self.kd.kernarg_size as usize
    }
}

pub(crate) fn retain_program_abi(
    abi: &[crate::device::AbiParamDescriptor],
) -> Result<(Vec<crate::device::AbiParamDescriptor>, usize, usize)> {
    let abi = abi.to_vec();
    let buf_count = abi.iter().filter(|arg| arg.is_storage()).count();
    let var_count = abi.len() - buf_count;
    let var_names =
        abi.iter().filter(|arg| !arg.is_storage()).map(|arg| arg.name.clone().unwrap_or_default()).collect::<Vec<_>>();
    crate::device::validate_abi_descriptors(&abi, buf_count, &var_names)?;
    Ok((abi, buf_count, var_count))
}

/// Graph-capture accessors. The AMD graph factory (`amd/graph.rs`) downcasts a
/// `dyn Program` to `AmdProgram` and uses the same metadata as per-call HCQ
/// submission lowering. Program addresses are linked once; invocation and
/// system-owned fields are patched at replay.
impl AmdProgram {
    /// Device handle. Used by the `Program::execute` trait fallback to lease a
    /// connector per call, and by `AmdGraph::capture` to reach the shared
    /// `Arc<AmdDeviceCore>` for the per-graph connector.
    pub fn device(&self) -> &Arc<AmdDevice> {
        &self.dev
    }

    /// `kd.kernarg_size` — byte count of one kernarg record (ABI padded).
    pub fn kernarg_record_size(&self) -> usize {
        self.kernarg_size()
    }

    /// COMPUTE_PGM_RSRC1/2/3 (PM4 path), pre-patched at load.
    pub fn rsrc(&self) -> (u32, u32, u32) {
        (self.rsrc1, self.rsrc2, self.rsrc3)
    }

    /// PM4 shader entry point (`prog_addr`; the LO/HI registers carry `>> 8`).
    pub fn pm4_prog_addr(&self) -> u64 {
        self.pm4_prog_addr
    }

    /// AQL `kernel_object` — GPU VA of the kernel descriptor the AQL packet
    /// processor reads (the dispatch packet's `kernel_object` field).
    pub fn aql_prog_addr(&self) -> u64 {
        self.aql_prog_addr
    }

    /// Required group (LDS) segment size in bytes, from the kernel descriptor
    /// (`kd.group_segment_fixed_size`) — the dispatch packet's
    /// `group_segment_size` field.
    pub fn group_segment_size(&self) -> u32 {
        self.kd.group_segment_fixed_size
    }

    /// `(wave32, target_major)` — drive the `cs_w32_en` DISPATCH_INITIATOR bit.
    pub fn wave32_target(&self) -> (bool, u32) {
        (self.wave32, self.target_major)
    }

    /// Whether the kernel reads a 4-dword scratch descriptor from user SGPRs
    /// 0-3 (prepended to USER_DATA before the kernarg pointer).
    pub fn enable_private_segment_sgpr(&self) -> bool {
        self.enable_private_segment_sgpr
    }

    /// `(buf_count, var_count)` — kernarg layout: `buf_count*8 + var_count*4`.
    pub fn arg_counts(&self) -> (usize, usize) {
        (self.buf_count, self.var_count)
    }

    pub fn abi(&self) -> &[crate::device::AbiParamDescriptor] {
        &self.abi
    }

    pub(crate) fn code_object(&self) -> Arc<CodeObject> {
        Arc::clone(&self.code)
    }

    /// Required private (scratch) segment size in bytes-per-thread, from the
    /// kernel descriptor (`kd.private_segment_fixed_size`). Used by callers
    /// to size the queue's scratch before dispatch
    /// (`PoolQueue::ensure_has_local_memory`).
    pub fn private_segment_size(&self) -> u32 {
        self.kd.private_segment_fixed_size
    }
}

impl std::fmt::Debug for AmdProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmdProgram")
            .field("name", &self.name)
            .field("gpu_id", &self.dev.node.gpu_id)
            .field("aql_prog_addr", &format_args!("{:#x}", self.aql_prog_addr))
            .finish_non_exhaustive()
    }
}

impl AmdProgram {
    /// Lane-scoped dispatch entry point. Reads the queue, kernarg arena, scratch,
    /// and PM4 counter from the exclusively leased `PoolQueue`. The lease spans
    /// kernarg bump, write, and publication, so kernarg order matches ring order.
    /// Callers must size lane scratch before calling.
    ///
    /// # Safety
    ///
    /// Same contract as [`Program::execute`]: `buffers` must point to live GPU
    /// VAs that outlive the dispatch, `vals` must match the kernel's variable
    /// arity, and launch dims must be valid for the kernel descriptor.
    ///
    /// Profiling ownership is delegated to the HCQ queue finalizer. Both AQL and
    /// PM4 own an optional timestamp slot and insert PM4 probes around `Compute`.
    #[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
    pub(crate) unsafe fn execute_on(
        &self,
        owner: &crate::amd::connector::OwnerCtx,
        pool: &crate::amd::connector::PoolQueue,
        buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
        wait: bool,
        profile: bool,
    ) -> Result<Option<Arc<dyn crate::sync::DispatchTimestamps>>> {
        // Device poisoned by an earlier fault: refuse to dispatch (the GPU
        // state and any cached buffer mappings are no longer trustworthy).
        if let Some(err) = pool.core().poison_error() {
            return Err(err);
        }
        if buffers.len() != self.buf_count {
            return Err(Error::ProgramAbiMismatch {
                reason: format!("AmdProgram expected {} compact buffers, got {}", self.buf_count, buffers.len()),
            });
        }
        if vals.len() != self.var_count {
            return Err(Error::ProgramAbiMismatch {
                reason: format!("AmdProgram expected {} compact scalar vals, got {}", self.var_count, vals.len()),
            });
        }
        // Kernarg layout:
        //   - Each buffer argument = 8 bytes (64-bit GPU pointer)
        //   - Each scalar variable = 4 bytes (uint32)
        // Scalars pack as i32 because svod's renderer lowers
        // `Index` → `i32` via `pm_lower_index_dtype`. The kernel descriptor
        // emitted by clang reflects this — a kernel with `(ptr, ptr, ..., i32
        // %v0, i32 %v1)` has `kernarg_size = bufs*8 + vars*4`, NOT bufs*8 +
        // vars*8. Packing each val as 8 bytes here would overflow the
        // descriptor and corrupt the next kernarg slot in the arena.
        let layout = crate::hcq::ClikeKernargLayout::from_abi(&self.abi);
        let needed = layout.packed_size();
        if needed > self.kernarg_size() {
            return Err(Error::Runtime {
                message: format!(
                    "AmdProgram '{}': kernarg layout {} > kd.kernarg_size {} \
                     (buf_count={}, var_count={})",
                    self.name,
                    needed,
                    self.kernarg_size(),
                    self.buf_count,
                    self.var_count,
                ),
            });
        }

        // The caller's non-clone QueueLease is the sole publication authority,
        // so kernarg reservation order is queue submission order.
        let arena = pool.arena();
        let off = arena.bump(self.kernarg_size(), crate::hcq::KERNARG_ALIGN)?;
        // SAFETY: arena returned a valid slot; the exclusive lane serializes
        // writers, so no concurrent writer holds the same offset.
        let host_base = unsafe { arena.host_at(off) };
        let addresses: smallvec::SmallVec<[u64; 8]> = buffers.iter().map(|p| *p as u64).collect();
        // SAFETY: arena returned a writable record of kernarg_size bytes.
        let dst = unsafe { std::slice::from_raw_parts_mut(host_base, self.kernarg_size()) };
        layout.pack(dst, &addresses, vals)?;
        let kernarg_gpu = arena.gpu_at(off);

        // 2. Submit sequence:
        //   PM4: wait(counter, prev) → memory_barrier → exec → signal(counter, next)
        //   AQL: vendor-IB wait/barrier → native dispatch → vendor-IB timeline store
        let g = global_size.unwrap_or([1, 1, 1]);
        let l = local_size.unwrap_or([1, 1, 1]);

        if std::env::var("SVOD_DEBUG_DISPATCH").is_ok() {
            let bufs_str: Vec<String> =
                buffers.iter().enumerate().map(|(i, b)| format!("buf{}={:#x}", i, *b as u64)).collect();
            eprintln!(
                "[dispatch tv={}] kernel={} grid=[{}, {}, {}] local=[{}, {}, {}] is_pm4={} kernarg_gpu={:#x} scratch={:#x} {}",
                pool.pm4_value(),
                self.name,
                g[0],
                g[1],
                g[2],
                l[0],
                l[1],
                l[2],
                pool.queue().is_pm4(),
                kernarg_gpu,
                pool.scratch_gpu_va(),
                bufs_str.join(" "),
            );
        }

        let queue = pool.queue();
        let is_pm4 = queue.is_pm4();
        let pmc_counters = if profile && is_pm4 && self.target_major == 11 { owner.pmc_counters() } else { Vec::new() };
        let pmc = if pmc_counters.is_empty() {
            None
        } else {
            let grid = crate::amd::pmc::PmcGrid::from_node(&self.dev.node);
            let bytes = crate::amd::pmc::readback_bytes(pmc_counters.len(), &grid);
            let buf =
                AmdBufferGuard::new(crate::amd::AmdAllocator::new(self.device_id)?.alloc_uncached(bytes.max(64))?);
            let (gpu, host) = match buf.buffer() {
                crate::allocator::RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
                _ => return Err(Error::Runtime { message: "PMC readback buffer not host-visible".into() }),
            };
            let (start, read) = crate::amd::pmc::build_streams(&pmc_counters, &grid, gpu);
            Some((buf, host, grid, start, read))
        };
        let (pmc_start, pmc_read): (&[u32], &[u32]) = pmc.as_ref().map_or((&[], &[]), |(_, _, _, s, r)| (s, r));

        let dispatch = crate::hcq::ComputeDispatch {
            workgroup_size: [l[0] as u32, l[1] as u32, l[2] as u32],
            grid_size: [(g[0] * l[0]) as u32, (g[1] * l[1]) as u32, (g[2] * l[2]) as u32],
            private_segment_size: self.kd.private_segment_fixed_size,
            group_segment_size: self.kd.group_segment_fixed_size,
            kernel_object: self.aql_prog_addr,
            kernarg_address: kernarg_gpu,
            // Native completion ownership is assigned by the queue finalizer.
            completion_signal: 0,
            barrier: true,
            amd_pm4: Some(crate::hcq::AmdPm4Dispatch {
                rsrc: [self.rsrc1, self.rsrc2, self.rsrc3],
                program_address: self.pm4_prog_addr,
                enable_private_segment_sgpr: self.enable_private_segment_sgpr,
                workgroup_count: [g[0] as u32, g[1] as u32, g[2] as u32],
                wave32: self.wave32,
                target_major: self.target_major,
            }),
        };
        let mut submission = crate::hcq::Submission::new(crate::hcq::QueueKind::Compute(0));
        submission.push(crate::hcq::Command::MemoryBarrier).push(crate::hcq::Command::Compute(dispatch));
        if profile {
            submission.request_profile();
        }
        let result = queue.submit_hcq_dispatch(pool, &submission, pmc_start, pmc_read)?;
        result.finalizer.retain_code(self.code_object());
        pool.register_inflight(Arc::clone(&result.finalizer));
        owner.set_newest(Arc::clone(&result.finalizer));

        if wait {
            if is_pm4 {
                owner.synchronize()?;
            } else if let Err(e) = owner.synchronize() {
                if let Some(code) = queue.inactive_exception() {
                    return Err(Error::Runtime {
                        message: format!(
                            "AQL dispatch '{}' did not complete: queue halted with exception {code:#x}",
                            self.name
                        ),
                    });
                }
                return Err(e);
            }
        }

        if is_pm4 {
            // PM4 single-XCC path: completion via the queue's monotonic PM4
            // counter (RELEASE_MEM); the submission finalizer retains this
            // owner's exact timeline point.
            //
            // The PM4 CP does not auto-stamp dispatches (no ENABLE_PROFILING as
            // on AQL), so for a profiling dispatch the finalizer acquires and
            // resets a timestamp slot, then brackets
            // the dispatch with two GPU-clock RELEASE_MEM probes into its
            // start/end ts fields. `profile` is threaded from the caller and is
            // set ONLY by callers that retain the returned handle until after
            // `synchronize` (the fire-and-forget execute path passes `false`), so
            // the slot can't be reused while the GPU is still writing it.
            // Build the returned handle: a PMC handle (timestamps + counters) when
            // counters were collected, else the bare timestamp signal.
            let handle: Option<Arc<dyn crate::sync::DispatchTimestamps>> = match (result.timestamps, pmc) {
                (Some(sig), Some((buf, host, grid, _, _))) => Some(Arc::new(crate::amd::pmc::PmcHandle::new(
                    sig,
                    Arc::clone(&result.finalizer),
                    buf.into_inner(),
                    host,
                    pmc_counters,
                    grid.instances(),
                ))),
                (Some(sig), None) => Some(sig),
                (None, _) => None,
            };
            Ok(handle)
        } else {
            Ok(result.timestamps.map(|signal| signal as Arc<dyn crate::sync::DispatchTimestamps>))
        }
    }
}

impl Program for AmdProgram {
    unsafe fn execute(
        &self,
        buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
        wait: bool,
    ) -> Result<()> {
        // Fallback callers use a logical context whose direct session acquires
        // one exclusive queue lane.
        let alloc = crate::amd::AmdAllocator::new(self.device_id)?;
        let owner = crate::amd::connector::OwnerCtx::new(Arc::clone(self.dev.core()), alloc);
        // Dispatch and (when waiting) drain through the poisoning owner-local
        // `synchronize`, exactly like the plan/graph paths. A faulting/hung
        // candidate latches the device error (`poison`) and BEAM fast-fails the
        // remaining candidates rather than hanging 30s — or worse, wedging — per
        // candidate. A KFD queue can't be recovered once faulted; the search
        // stays useful by *avoiding* faults (resource caps in the action
        // filter).
        let _ = unsafe {
            crate::device::PlanContext::dispatch(
                &owner,
                self,
                buffers,
                vals,
                global_size,
                local_size,
                /*profile=*/ false,
            )?
        };
        if wait {
            owner.synchronize()?;
        } else {
            crate::device::PlanContext::finish_replay(&owner)?;
            // Fire-and-forget with a throwaway owner: nothing durable records
            // this submission, so park its finalizer in the core's unattributed
            // list for scoped host waits.
            if let Some(token) = owner.completion_token() {
                self.dev.core().record_unattributed(token);
            }
        }
        Ok(())
    }

    /// The dispatch's own span on the GPU's 100 MHz clock, from the two
    /// `release_mem_timestamp` probes `profile=true` brackets it with.
    ///
    /// Without this a tuner times AMD on the host clock, which charges the
    /// kernel for the whole submit path: the PM4 stream rebuilt per call, the
    /// ring memcpy and doorbell, and — unboundedly — a scratch regrow in
    /// `ensure_has_local_memory` or a kernarg-arena wrap that drains the
    /// device. The first of those is driven by the candidate's own
    /// `private_segment_fixed_size`, so it lands on spilling candidates
    /// specifically and no amount of repetition averages it out.
    ///
    /// The probes are not free: each lowers to an end-of-pipe fence, the clock
    /// write, and an `acquire_mem` that invalidates and writes back the whole
    /// cache hierarchy, and the pre-kernel one falls inside the span. So every
    /// timed run here starts cache-cold — what [`BenchmarkConfig::clear_l2`]
    /// only ever approximated from the host — at a fixed cost per dispatch.
    ///
    /// # Safety
    ///
    /// Same contract as [`Program::execute`].
    unsafe fn execute_timed(
        &self,
        buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
    ) -> Result<Option<std::time::Duration>> {
        let alloc = crate::amd::AmdAllocator::new(self.device_id)?;
        let owner = crate::amd::connector::OwnerCtx::new(Arc::clone(self.dev.core()), alloc);
        // `profile=true` obliges us to hold the handle until after the drain:
        // the probes write into scratch it owns. SAFETY: forwarded contract.
        let stamps = unsafe {
            crate::device::PlanContext::dispatch(
                &owner,
                self,
                buffers,
                vals,
                global_size,
                local_size,
                /*profile=*/ true,
            )?
        };
        // Poisons the device and propagates on a faulting candidate, exactly as
        // the untimed path does, so a tuner drops it instead of hanging.
        owner.synchronize()?;
        Ok(stamps
            .and_then(|stamps| stamps.timestamps_ns())
            .map(|(start, end)| std::time::Duration::from_nanos(end - start)))
    }

    fn name(&self) -> &str {
        &self.name
    }

    /// Downcast hook for the AMD graph factory (`amd/graph.rs`): recovers the
    /// concrete `AmdProgram` so it can read rsrc/prog_addr/arena and pre-build
    /// the indirect-buffer dispatch chain. Mirrors `Program::as_any`'s contract.
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Create logical per-plan state. Hardware queue ownership is acquired by
    /// the context at a replay/session boundary.
    fn new_exec_context(&self) -> Result<Option<Box<dyn crate::device::PlanContext>>> {
        let alloc = crate::amd::AmdAllocator::new(self.device_id)?;
        let owner = crate::amd::connector::OwnerCtx::new(Arc::clone(self.dev.core()), alloc);
        Ok(Some(Box::new(owner)))
    }

    /// Decode VGPR/SGPR/LDS/scratch + VGPR-limited occupancy from the kernel
    /// descriptor. Uses the original (un-patched) `compute_pgm_rsrc1` and the
    /// descriptor's exact segment sizes — pure static, no GPU access.
    fn resource_usage(&self) -> Option<crate::profile::KernelResources> {
        Some(crate::amd::occupancy::decode_resources(
            self.kd.compute_pgm_rsrc1,
            self.kd.group_segment_fixed_size,
            self.kd.private_segment_fixed_size,
            if self.wave32 { 32 } else { 64 },
            self.target_major,
        ))
    }
}
