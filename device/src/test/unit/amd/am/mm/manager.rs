use crate::amd::am::mm::manager::*;
use crate::amd::am::mm::pagetable::Geometry;
use crate::amd::am::mm::{AddrSpace, PhysMem};
use std::collections::HashMap;

/// Sparse in-process VRAM for tests: unwritten reads return 0.
#[derive(Default)]
struct FakeVram {
    cells: HashMap<u64, u64>,
}
impl PhysMem for FakeVram {
    fn read_u64(&self, paddr: u64) -> u64 {
        assert_eq!(paddr % 8, 0, "unaligned PTE read");
        *self.cells.get(&paddr).unwrap_or(&0)
    }
    fn write_u64(&mut self, paddr: u64, val: u64) {
        assert_eq!(paddr % 8, 0, "unaligned PTE write");
        self.cells.insert(paddr, val);
    }
    fn zero(&mut self, paddr: u64, size: u64) {
        for off in (0..size).step_by(8) {
            self.cells.remove(&(paddr + off));
        }
    }
}

const VA_BASE: u64 = 0x2000_0000_0000;

fn mm() -> MemoryManager<FakeVram> {
    MemoryManager::new(
        FakeVram::default(),
        (11, 5, 1),
        Geometry::gfx11(),
        /*vram_size=*/ 256 << 20,
        VA_BASE,
        /*va_size=*/ 1 << 40,
        /*reserve_ptable=*/ true,
        gfx11_palloc_ranges(),
    )
}

#[test]
fn map_one_4k_page_resolves() {
    let mut m = mm();
    let va = VA_BASE + 0x10_0000;
    let pa = 0x40_0000u64;
    m.map_range(va, 0x1000, &[(pa, 0x1000)], AddrSpace::Phys, false, false);
    assert_eq!(m.resolve(va), Some(pa));
    assert_eq!(m.resolve(va + 0x800), Some(pa + 0x800)); // offset within page
    assert_eq!(m.resolve(va + 0x1000), None); // next page unmapped
}

#[test]
fn map_multi_page_contiguous() {
    let mut m = mm();
    let va = VA_BASE + 0x20_0000;
    let pa = 0x80_0000u64;
    m.map_range(va, 0x4000, &[(pa, 0x4000)], AddrSpace::Phys, false, false); // 4 pages
    for i in 0..4u64 {
        assert_eq!(m.resolve(va + i * 0x1000), Some(pa + i * 0x1000));
    }
    assert_eq!(m.resolve(va + 0x4000), None);
}

#[test]
fn two_mib_aligned_uses_huge_page() {
    let mut m = mm();
    // 2 MiB-aligned VA + paddr, 2 MiB size → a single 2 MiB leaf at PDB0.
    let va = VA_BASE + (4u64 << 20);
    let pa = 2u64 << 20;
    m.map_range(va, 2 << 20, &[(pa, 2 << 20)], AddrSpace::Phys, false, false);
    // Resolves across the whole 2 MiB.
    assert_eq!(m.resolve(va), Some(pa));
    assert_eq!(m.resolve(va + (1 << 20)), Some(pa + (1 << 20)));
    assert_eq!(m.resolve(va + (2 << 20) - 4), Some(pa + (2 << 20) - 4));
    // The leaf is at PDB0 (huge page), so there is no PTB table for it: the
    // PDB0 entry must carry the huge-page bit.
    let rebased = va - VA_BASE;
    // walk to PDB0 manually
    let pdb2 = m.root_pt();
    let i2 = m.geom.pte_idx(0, rebased);
    let pdb1 = m.pt_child(pdb2, i2);
    let i1 = m.geom.pte_idx(1, rebased);
    let pdb0 = m.pt_child(pdb1, i1);
    let i0 = m.geom.pte_idx(2, rebased);
    assert!(m.pt_is_page(2, pdb0, i0), "2 MiB mapping must be a huge leaf at PDB0");
}

#[test]
fn unmap_reclaims_tables_and_remaps() {
    let mut m = mm();
    let va = VA_BASE + 0x30_0000;
    let pa = 0x10_0000u64;
    m.map_range(va, 0x2000, &[(pa, 0x2000)], AddrSpace::Phys, false, false);
    assert_eq!(m.resolve(va), Some(pa));
    m.unmap_range(va, 0x2000);
    assert_eq!(m.resolve(va), None);
    // After unmap, the same range maps cleanly again (the pass-1 "already
    // mapped" assert would fire if the PTEs weren't cleared / tables freed).
    m.map_range(va, 0x2000, &[(pa, 0x2000)], AddrSpace::Phys, false, false);
    assert_eq!(m.resolve(va), Some(pa));
}

#[test]
fn valloc_maps_and_resolves_then_vfree() {
    let mut m = mm();
    let vm = m.valloc(0x8000, 0x1000, false, false).expect("valloc");
    assert!(vm.va_addr >= VA_BASE);
    assert_eq!(vm.size, 0x8000);
    // Every page resolves to *some* physical page.
    for i in 0..8u64 {
        assert!(m.resolve(vm.va_addr + i * 0x1000).is_some());
    }
    m.vfree(&vm);
    assert_eq!(m.resolve(vm.va_addr), None);
}

/// A zero-byte request maps one page, as KFD's does, in both physical layouts:
/// the VA alignment is read off the rounded size, which must not be zero.
#[test]
fn a_zero_byte_valloc_maps_one_page() {
    let mut m = mm();
    for contiguous in [false, true] {
        let vm = m.valloc(0, 0x1000, false, contiguous).expect("valloc");
        assert_eq!(vm.size, 0x1000);
        assert!(m.resolve(vm.va_addr).is_some());
        m.vfree(&vm);
        assert_eq!(m.resolve(vm.va_addr), None);
    }
}

#[test]
fn valloc_contiguous_is_one_segment() {
    let mut m = mm();
    let vm = m.valloc(0x4000, 0x1000, true, true).expect("valloc contiguous");
    assert_eq!(vm.paddrs.len(), 1);
    assert_eq!(vm.paddrs[0].1, 0x4000);
    m.vfree(&vm);
}

#[test]
fn uncached_sets_mtype_on_leaf() {
    let mut m = mm();
    let va = VA_BASE + 0x50_0000;
    m.map_range(va, 0x1000, &[(0x20_0000, 0x1000)], AddrSpace::Phys, /*uncached=*/ true, false);
    // Read the leaf PTE and check MTYPE_UC (bits 48..=50 == 3).
    let rebased = va - VA_BASE;
    let mut paddr = m.root_pt();
    let mut lv = 0;
    while !m.pt_is_page(lv, paddr, m.geom.pte_idx(lv, rebased)) {
        paddr = m.pt_child(paddr, m.geom.pte_idx(lv, rebased));
        lv += 1;
    }
    let pte = m.entry(paddr, m.geom.pte_idx(lv, rebased));
    assert_eq!((pte >> 48) & 0x7, 3, "uncached leaf must set MTYPE_UC");
}

#[test]
#[should_panic(expected = "page-table walk made no progress")]
fn unmap_subrange_of_huge_leaf_panics() {
    let mut m = mm();
    // A single 2 MiB huge leaf at PDB0 (2 MiB-aligned VA + paddr + size).
    let va = VA_BASE + (4u64 << 20);
    let pa = 2u64 << 20;
    m.map_range(va, 2 << 20, &[(pa, 2 << 20)], AddrSpace::Phys, false, false);
    // Unmapping only the first 4 KiB of a 2 MiB leaf can't be expressed (the
    // leaf can't be split) — the walker must fail loudly, not spin forever.
    m.unmap_range(va, 0x1000);
}
