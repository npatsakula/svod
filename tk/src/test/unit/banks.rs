//! Shared layouts against a target's banks ([`crate::atoms::SharedBanks`]).

use std::collections::BTreeSet;

use svod_dtype::AmdArch;
use test_case::test_case;

use crate::atoms::{SharedBanks, Target};
use crate::build::BF16;
use crate::ir::*;
use crate::kernels::Batch;
use crate::kernels::attention::{AttnMask, AttnSpec, FaCfg, attention};
use crate::kernels::gemm::{Epilogue, GemmSpec, gemm};
use crate::layouts::shared_element;
use crate::lower::{Lowered, Lowering, standalone};
use crate::ops::config;
use crate::sweep::{amd, declared};

#[test_case(16, 8, 2, false; "one chunk a row")]
#[test_case(16, 8, 2, true; "one chunk a row, halves")]
#[test_case(64, 16, 2, true; "two chunks")]
#[test_case(64, 32, 2, true; "four chunks")]
#[test_case(64, 64, 2, true; "eight chunks")]
#[test_case(32, 256, 2, true; "a row past the widest span")]
#[test_case(32, 24, 2, true; "unswizzled")]
#[test_case(32, 64, 1, true; "bytes")]
#[test_case(32, 16, 4, true; "words")]
fn a_shared_tile_keeps_each_element_a_place(rows: usize, cols: usize, bytes: usize, halves: bool) {
    let at: BTreeSet<_> =
        (0..rows).flat_map(|r| (0..cols).map(move |c| shared_element(r, c, cols, bytes, halves))).collect();
    assert_eq!(at, (0..rows * cols).collect());
}

fn gemm_on(arch: AmdArch) -> Lowered {
    let device = declared(amd(arch, 304));
    let cfg = crate::kernels::gemm::GemmCfg { crews: 2, ..config::gemm_cfg([256, 256, 64], 1, [2, 4], false) };
    let spec = GemmSpec { m: 512, n: 512, k: 256, batch: Batch::Static(1), epilogue: Epilogue::DEFAULT, cfg };
    standalone(gemm::<BF16>(&spec), &cfg.lowering(&device)).unwrap()
}

fn attention_on(arch: AmdArch, d: usize) -> Lowered {
    let device = declared(amd(arch, 304));
    let cfg = FaCfg::new(16, 64, 2);
    let spec = AttnSpec {
        batch: Batch::Static(1),
        t: 64,
        tk: 256,
        heads: 1,
        kv_heads: 1,
        d,
        mask: AttnMask::default(),
        cache: None,
        packed: false,
        scale: 0.125,
        cfg,
    };
    standalone(attention::<BF16>(&spec), &Lowering::new(&device)).unwrap()
}

/// The banks each 16 lanes of a shared read miss, over every read of `low`.
fn missed(low: &Lowered, banks: SharedBanks) -> usize {
    let prog = &low.tile;
    let lanes = 64;
    let mut missed = 0;
    for (_, stmt) in prog.walk() {
        let Stmt::Copy { dst, src, .. } = stmt else { continue };
        let (Place::Smem { alloc, .. }, Some(l)) = (&prog.value(*src).place, low.layouts[dst.index()].as_ref()) else {
            continue;
        };
        let a = &prog.smem[alloc.index()];
        let shape = prog.value(*src).shape;
        let cols = if a.along == Axis::Row { shape.rows } else { shape.cols };
        let bytes = a.dtype.bytes();
        let width = 16 / bytes as u32;
        for (reg, _) in l.runs(lanes, width, a.along) {
            for group in 0..lanes / banks.count {
                let reached: BTreeSet<_> = (0..banks.count)
                    .map(|i| {
                        let (r, c) = l.coord(0, group * banks.count + i, reg);
                        let (r, c) = if a.along == Axis::Row { (c, r) } else { (r, c) };
                        let at = shared_element(r as usize, c as usize, cols, bytes, a.halves) * bytes;
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
    assert!(low.tile.smem.iter().all(|a| a.halves), "{:?}", low.tile.smem);
    assert_eq!(missed(&low, amd(AmdArch::Gfx942, 304).shared_banks.unwrap()), 0);
}

#[test_case(64; "d64")]
#[test_case(128; "d128")]
fn attention_reads_reach_every_bank(d: usize) {
    let low = attention_on(AmdArch::Gfx942, d);
    assert!(low.tile.smem.iter().all(|a| a.halves), "{:?}", low.tile.smem);
    assert_eq!(missed(&low, amd(AmdArch::Gfx942, 304).shared_banks.unwrap()), 0);
}

#[test_case(AmdArch::Gfx950; "gfx950")]
#[test_case(AmdArch::Gfx1201; "gfx1201")]
fn a_target_without_banks_keeps_its_chunks(arch: AmdArch) {
    assert!(amd(arch, 304).shared_banks.is_none());
    let low = attention_on(arch, 64);
    assert!(low.tile.smem.iter().all(|a| !a.halves));
}

#[test]
fn a_store_stays_within_a_half() {
    let low = gemm_on(AmdArch::Gfx942);
    let text = low.text(&amd(AmdArch::Gfx942, 304)).unwrap();
    let asm = match svod_runtime::amd::compile_ir_to_amd_asm(&text, AmdArch::Gfx942) {
        Ok(asm) => String::from_utf8(asm).unwrap(),
        Err(e) if e.to_string().contains("AMDGPU target") || e.to_string().contains("too old") => {
            eprintln!("skipped: clang has no AMDGPU target");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    assert!(asm.contains("ds_write") && !asm.contains("ds_write_b128"), "{asm}");
}

fn gemm_text(arch: AmdArch, k: usize) -> String {
    let device = declared(Target { swizzled_loads: true, ..amd(arch, 304) });
    let cfg = crate::kernels::gemm::GemmCfg { crews: 2, ..config::gemm_cfg([256, 256, 64], 1, [2, 4], false) };
    let spec = GemmSpec { m: 512, n: 512, k, batch: Batch::Static(1), epilogue: Epilogue::DEFAULT, cfg };
    crate::lower::render(gemm::<BF16>(&spec), &cfg.lowering(&device)).unwrap()
}

const BUFFER_LOAD: &str = "call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load";

#[test_case(1024, 2048; "a row of 2 KiB")]
#[test_case(4096, 8192; "a row of 8 KiB")]
fn a_staged_load_carries_its_row_stride_and_the_swizzle(k: usize, row_bytes: u32) {
    let text = gemm_text(AmdArch::Gfx942, k);
    assert!(text.contains(BUFFER_LOAD), "{text}");
    let field = format!("i16 {}, i32 -1", row_bytes | 0x4000);
    assert!(text.contains(&field), "{field}:\n{text}");
}

#[test]
fn a_stride_past_its_field_loads_flat() {
    assert!(!gemm_text(AmdArch::Gfx942, 8192).contains(BUFFER_LOAD));
}

#[test_case(AmdArch::Gfx942; "gfx942")]
#[test_case(AmdArch::Gfx950; "gfx950")]
#[test_case(AmdArch::Gfx1201; "gfx1201")]
fn a_target_without_swizzled_loads_loads_flat(arch: AmdArch) {
    assert!(!amd(arch, 304).swizzled_loads);
    let device = declared(amd(arch, 304));
    let cfg = config::gemm_cfg([64, 64, 32], 1, [2, 2], false);
    let spec = GemmSpec { m: 256, n: 256, k: 512, batch: Batch::Static(1), epilogue: Epilogue::DEFAULT, cfg };
    assert!(!crate::lower::render(gemm::<BF16>(&spec), &cfg.lowering(&device)).unwrap().contains(BUFFER_LOAD));
}

#[test]
fn a_swizzled_load_needs_no_waterfall() {
    let text = gemm_text(AmdArch::Gfx942, 4096);
    let asm = match svod_runtime::amd::compile_ir_to_amd_asm(&text, AmdArch::Gfx942) {
        Ok(asm) => String::from_utf8(asm).unwrap(),
        Err(e) if e.to_string().contains("AMDGPU target") || e.to_string().contains("too old") => {
            eprintln!("skipped: clang has no AMDGPU target");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    assert!(asm.contains("buffer_load_dwordx4") && !asm.contains("s_cbranch_execnz"), "{asm}");
}
