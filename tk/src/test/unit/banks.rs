//! Shared layouts against a target's banks ([`crate::atoms::SharedBanks`]).

use std::collections::BTreeSet;

use proptest::prelude::*;
use svod_dtype::AmdArch;
use test_case::test_case;

use super::attention::probe_spec;
use super::gemm_spec;
use super::targets::amd_asm;
use crate::atoms::{SharedBanks, Target};
use crate::build::BF16;
use crate::ir::*;
use crate::kernels::attention::{AttnMask, AttnSpec, FaCfg, attention};
use crate::kernels::gemm::gemm;
use crate::layouts::SharedSwizzle;
use crate::lower::{Lowered, Lowering, standalone};
use crate::ops::config;
use crate::sweep::assumed;
use crate::sweep::{amd, declared};

proptest! {
    /// The swizzle is an F2 map that leaves a row's columns a permutation of
    /// themselves by construction; what the algebra does not say is that a
    /// row of chunks that is no power of two (unswizzled, its halves still
    /// exchanged) stays within its row, so every tile of whole chunks keeps
    /// each element a place of its own.
    #[test]
    fn a_shared_tile_keeps_each_element_a_place(
        rows in 1usize..=64,
        chunks in 1usize..=20,
        bytes in prop::sample::select(vec![1usize, 2, 4, 8]),
        halves in prop::option::of(0u32..6),
    ) {
        let cols = chunks * (16 / bytes);
        let swizzle = SharedSwizzle::new(cols, bytes, halves);
        let swizzle = &swizzle;
        let at: BTreeSet<_> = (0..rows).flat_map(|r| (0..cols).map(move |c| swizzle.element(r, c))).collect();
        prop_assert_eq!(at, (0..rows * cols).collect());
    }
}

fn gemm_on(arch: AmdArch) -> Lowered {
    let device = declared(amd(arch, 304));
    let cfg = crate::kernels::gemm::GemmCfg {
        schedule: Schedule::ping_pong(16),
        ..config::gemm_cfg([256, 256, 64], 1, [2, 4], false)
    };
    standalone(gemm::<BF16>(&assumed(gemm_spec(512, 512, 256), cfg)), &cfg.lowering(&device)).unwrap()
}

fn attention_on(arch: AmdArch, d: usize) -> Lowered {
    let device = declared(amd(arch, 304));
    let cfg = FaCfg::new(16, 64, 2);
    let spec = assumed(AttnSpec { scale: 0.125, ..probe_spec(1, 1, 64, 256, d, AttnMask::default()) }, cfg);
    standalone(attention::<BF16>(&spec), &Lowering::new(&device)).unwrap()
}

/// The banks each 16 lanes of a shared read miss, over every read of `low`.
fn missed(low: &Lowered, banks: SharedBanks) -> usize {
    let prog = &low.tile;
    let lanes = 64;
    let mut missed = 0;
    for (_, stmt) in prog.walk() {
        let Stmt::Copy { dst, src, .. } = stmt else { continue };
        let (Place::Smem { alloc, .. }, Some(l)) = (&prog.value(*src).place, prog.layouts[dst.index()].as_ref()) else {
            continue;
        };
        let a = &prog.smem[alloc.index()];
        let Storage { along, halves } = a.storage;
        let shape = prog.value(*src).shape;
        let cols = if along == Axis::Row { shape.rows } else { shape.cols };
        let bytes = a.dtype.bytes();
        let width = 16 / bytes as u32;
        let swizzle = SharedSwizzle::new(cols, bytes, halves);
        for (reg, _) in l.runs(lanes, width, along) {
            for group in 0..lanes / banks.count {
                let reached: BTreeSet<_> = (0..banks.count)
                    .map(|i| {
                        let (r, c) = l.coord(0, group * banks.count + i, reg);
                        let (r, c) = if along == Axis::Row { (c, r) } else { (r, c) };
                        let at = swizzle.element(r as usize, c as usize) * bytes;
                        at / banks.bytes as usize % banks.count as usize
                    })
                    .collect();
                missed += banks.count as usize - reached.len();
            }
        }
    }
    missed
}

#[test]
fn gemm_reads_reach_every_bank() {
    let low = gemm_on(AmdArch::Gfx942);
    assert!(low.tile.smem.iter().all(|a| a.storage.halves.is_some()), "{:?}", low.tile.smem);
    assert_eq!(missed(&low, amd(AmdArch::Gfx942, 304).shared_banks.unwrap()), 0);
}

#[test_case(64; "d64")]
#[test_case(128; "d128")]
fn attention_reads_reach_every_bank(d: usize) {
    let low = attention_on(AmdArch::Gfx942, d);
    assert!(low.tile.smem.iter().all(|a| a.storage.halves.is_some()), "{:?}", low.tile.smem);
    assert_eq!(missed(&low, amd(AmdArch::Gfx942, 304).shared_banks.unwrap()), 0);
}

#[test_case(AmdArch::Gfx950; "gfx950")]
#[test_case(AmdArch::Gfx1201; "gfx1201")]
fn a_target_without_banks_keeps_its_chunks(arch: AmdArch) {
    assert!(amd(arch, 304).shared_banks.is_none());
    let low = attention_on(arch, 64);
    assert!(low.tile.smem.iter().all(|a| a.storage.halves.is_none()));
}

#[test]
fn a_store_stays_within_a_half() {
    let low = gemm_on(AmdArch::Gfx942);
    let text = low.text(&amd(AmdArch::Gfx942, 304)).unwrap();
    let Some(asm) = amd_asm(&text, AmdArch::Gfx942) else { return };
    assert!(asm.contains("ds_write") && !asm.contains("ds_write_b128"), "{asm}");
}

fn gemm_text_on(target: Target, m: usize, n: usize, k: usize) -> String {
    let device = declared(target);
    let cfg = crate::kernels::gemm::GemmCfg {
        schedule: Schedule::ping_pong(16),
        ..config::gemm_cfg([256, 256, 64], 1, [2, 4], false)
    };
    crate::lower::render(gemm::<BF16>(&assumed(gemm_spec(m, n, k), cfg)), &cfg.lowering(&device)).unwrap()
}

fn gemm_text(arch: AmdArch, k: usize) -> String {
    gemm_text_on(amd(arch, 304), 512, 512, k)
}

const FAT_POINTER: &str = "@llvm.amdgcn.make.buffer.rsrc.p7.p1";

/// The stride is the largest power of two within the widest input row, or
/// else output row, and past 8 KiB the field's largest.
#[test_case(64, 1024; "only an output row spreads")]
#[test_case(768, 1024; "a row of 1.5 KiB")]
#[test_case(1024, 2048; "a row of 2 KiB")]
#[test_case(1536, 2048; "a row of 3 KiB")]
#[test_case(3072, 4096; "a row of 6 KiB")]
#[test_case(4096, 8192; "a row of 8 KiB")]
#[test_case(6144, 8192; "a row of 12 KiB")]
#[test_case(8192, 8192; "a row of 16 KiB")]
fn a_kernel_takes_the_power_of_two_within_its_widest_row(k: usize, stride: u32) {
    let text = gemm_text(AmdArch::Gfx942, k);
    assert!(text.contains(FAT_POINTER), "{text}");
    let field = format!("i16 {}, i64 -1", stride | 0x4000);
    assert!(text.contains(&field), "{field}:\n{text}");
}

/// A dispatch reading a line under two mappings reads the wrong sector: every
/// parameter holds the one stride, and no global access is left flat.
#[test_case(1024; "a row of 2 KiB")]
#[test_case(4096; "a row of 8 KiB")]
fn every_global_access_of_a_kernel_takes_one_mapping(k: usize) {
    let text = gemm_text(AmdArch::Gfx942, k);
    let fields: Vec<_> = text
        .lines()
        .filter(|l| l.contains("= call ptr addrspace(7)"))
        .map(|l| l.split("i16 ").nth(1).and_then(|f| f.split(',').next()).expect("a stride field"))
        .collect();
    assert_eq!(fields.len(), 3, "A, B and C:\n{text}");
    assert!(fields.iter().all(|f| *f == fields[0]), "{fields:?}");
    let Some(asm) = amd_asm(&text, AmdArch::Gfx942) else { return };
    assert!(asm.contains("buffer_load_dwordx4") && asm.contains("buffer_store"), "{asm}");
    assert!(!asm.contains("global_load") && !asm.contains("global_store"), "{asm}");
    assert!(!asm.contains("flat_load") && !asm.contains("flat_store"), "{asm}");
}

#[test_case(512, 64, 64; "rows already spread over the sets")]
#[test_case(512, 16384, 16384; "rows the stride leaves piled up")]
#[test_case(1 << 19, 512, 4096; "an operand past a 32-bit offset")]
fn a_kernel_outside_the_swizzle_stays_flat(m: usize, n: usize, k: usize) {
    let text = gemm_text_on(amd(AmdArch::Gfx942, 304), m, n, k);
    assert!(!text.contains(FAT_POINTER), "{text}");
}

#[test_case(AmdArch::Gfx950; "gfx950")]
#[test_case(AmdArch::Gfx1201; "gfx1201")]
fn a_target_without_the_swizzle_stays_flat(arch: AmdArch) {
    assert_eq!(amd(arch, 304).cache_swizzle, None);
    let device = declared(amd(arch, 304));
    let cfg = config::gemm_cfg([64, 64, 32], 1, [2, 2], false);
    let text = crate::lower::render(gemm::<BF16>(&assumed(gemm_spec(256, 256, 512), cfg)), &cfg.lowering(&device));
    assert!(!text.unwrap().contains(FAT_POINTER))
}

#[test]
fn a_swizzled_access_needs_no_waterfall() {
    let Some(asm) = amd_asm(&gemm_text(AmdArch::Gfx942, 4096), AmdArch::Gfx942) else { return };
    assert!(asm.contains("buffer_load_dwordx4") && !asm.contains("s_cbranch_execnz"), "{asm}");
}
