//! `MemoryManager`: virtual-address allocation + GMMU page-table mapping.
//!
//! GMMU memory manager + page-table traversal, over the injectable [`PhysMem`]
//! backing so it is unit-tested without a GPU. Three TLSF sub-allocators (virtual address
//! space, physical VRAM, page-table pages) plus a 4-level page-table walk that
//! maps/unmaps ranges, picking the largest huge page that fits + is aligned and
//! reclaiming now-empty tables on unmap.

use super::pagetable::{self, Geometry, IpVer, VM_PDB2};
use super::tlsf::TlsfAllocator;
use super::{AddrSpace, PhysMem, VirtMapping};

#[inline]
fn bit_length(x: u64) -> u32 {
    u64::BITS - x.leading_zeros()
}

#[inline]
fn round_up(x: u64, m: u64) -> u64 {
    x.div_ceil(m) * m
}

/// gfx11 physical-allocation chunk sizes `(size, align)`, largest first — so
/// `valloc` grabs the longest contiguous segments it can (fewer PTEs / less TLB
/// pressure). Derived from `va_shifts=[12..39]`: `2 MiB`-aligned for chunks
/// ≥ 2 MiB, else `4 KiB`.
pub fn gfx11_palloc_ranges() -> Vec<(u64, u64)> {
    (0..=27).rev().map(|i| (1u64 << (i + 12), if i >= 9 { 2 << 20 } else { 0x1000 })).collect()
}

/// One frame of the page-table walk stack.
#[derive(Clone, Copy, Debug)]
struct Frame {
    paddr: u64,
    lv: usize,
    pte_idx: usize,
    pte_covers: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WalkMode {
    /// Visit valid ranges; assert nothing is mapped (map pre-check).
    Inspect,
    /// Descend until the level covers + aligns the request, creating tables.
    Create,
    /// Descend to leaves, freeing now-empty tables on the way up.
    Free,
}

/// Walk state (no borrows — driven by `MemoryManager::walk_next`).
struct Walk {
    vaddr: u64, // rebased (minus va_base); advances across yields/segments
    mode: WalkMode,
    stack: Vec<Frame>,
    size: u64, // remaining bytes in the current segment
    off: u64,  // offset within the current segment
    /// Advance (entries, pte_covers) deferred from the last yield — applied at
    /// the START of the next call so the caller's per-entry body runs *before*
    /// `level_up` (which `Free` mode relies on to see emptied tables).
    pending: Option<(u64, u64)>,
}

/// One step of the walk handed to the caller.
struct Step {
    off: u64,
    pt_paddr: u64,
    lv: usize,
    pte_idx: usize,
    entries: u64,
    pte_covers: u64,
    vaddr_at_yield: u64,
}

pub struct MemoryManager<P: PhysMem> {
    phys: P,
    ip_ver: IpVer,
    pub(crate) geom: Geometry,
    va_base: u64,
    first_lv: usize,
    va_alloc: TlsfAllocator,
    pa_alloc: TlsfAllocator,
    ptable_alloc: TlsfAllocator,
    reserve_ptable: bool,
    root_pt: u64,
    palloc_ranges: Vec<(u64, u64)>,
    /// Offset added to VRAM physical addresses when they are written into page-
    /// table entries (the GPU MC / XGMI base), and subtracted when a child-table
    /// pointer is read back for CPU walk. 0 unless [`set_mc_base`] is called.
    /// CPU/BAR access and the TLSF allocators stay in raw-paddr space.
    mc_base: u64,
}

impl<P: PhysMem> MemoryManager<P> {
    /// Build a manager over `phys`. `vram_size` is the physical pool; the
    /// virtual space is `[va_base, va_base + va_size)`. When `reserve_ptable`,
    /// a slice of VRAM is carved for page tables (so they don't fragment the
    /// data pool).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        phys: P,
        ip_ver: IpVer,
        geom: Geometry,
        vram_size: u64,
        va_base: u64,
        va_size: u64,
        reserve_ptable: bool,
        palloc_ranges: Vec<(u64, u64)>,
    ) -> Self {
        let first_lv = VM_PDB2;
        let ptable_size = if reserve_ptable { round_up(vram_size / 512, 1 << 20) } else { 0 };
        let ptable_alloc = TlsfAllocator::with_defaults(ptable_size, 0);
        let pa_alloc = TlsfAllocator::with_defaults(vram_size - ptable_size, ptable_size);
        let va_alloc = TlsfAllocator::with_defaults(va_size, va_base);
        let mut mm = Self {
            phys,
            ip_ver,
            geom,
            va_base,
            first_lv,
            va_alloc,
            pa_alloc,
            ptable_alloc,
            reserve_ptable,
            root_pt: 0,
            palloc_ranges,
            mc_base: 0,
        };
        // Root page table (zeroed so every slot starts invalid).
        mm.root_pt = mm.palloc(0x1000, 0x1000, true, true).expect("root page table");
        mm
    }

    /// Set the GPU MC/XGMI base added to VRAM paddrs in page-table entries.
    /// Call once after construction, before any mapping.
    #[inline]
    pub fn set_mc_base(&mut self, mc_base: u64) {
        self.mc_base = mc_base;
    }

    #[inline]
    pub fn root_pt(&self) -> u64 {
        self.root_pt
    }
    #[inline]
    pub fn phys(&self) -> &P {
        &self.phys
    }

    // ── physical / page-table page allocation ──────────────────────────────
    fn palloc(&mut self, size: u64, align: u64, zero: bool, ptable: bool) -> Option<u64> {
        let size = round_up(size, 0x1000);
        let paddr = {
            let alloc = if self.reserve_ptable && ptable { &mut self.ptable_alloc } else { &mut self.pa_alloc };
            alloc.alloc(size, align)?
        };
        if zero {
            self.phys.zero(paddr, size);
        }
        Some(paddr)
    }

    fn pfree(&mut self, paddr: u64, ptable: bool) {
        let alloc = if self.reserve_ptable && ptable { &mut self.ptable_alloc } else { &mut self.pa_alloc };
        alloc.free(paddr);
    }

    // ── page-table entry primitives (over PhysMem) ─────────────────────────
    #[inline]
    fn pte_covers(&self, lv: usize) -> u64 {
        self.geom.pte_covers[lv]
    }
    #[inline]
    fn pte_cnt(&self, lv: usize) -> u64 {
        self.geom.pte_cnt[lv]
    }
    #[inline]
    pub(crate) fn entry(&self, table_paddr: u64, idx: usize) -> u64 {
        self.phys.read_u64(table_paddr + (idx as u64) * 8)
    }
    #[inline]
    fn pt_valid(&self, table_paddr: u64, idx: usize) -> bool {
        pagetable::entry_valid(self.entry(table_paddr, idx))
    }
    #[inline]
    pub(crate) fn pt_child(&self, table_paddr: u64, idx: usize) -> u64 {
        // Child-table pointers are stored as MC addresses; recover the raw
        // paddr for CPU/BAR walk.
        pagetable::entry_paddr(self.entry(table_paddr, idx)) - self.mc_base
    }
    #[inline]
    pub(crate) fn pt_is_page(&self, lv: usize, table_paddr: u64, idx: usize) -> bool {
        lv == self.geom.level_cnt - 1 || pagetable::is_pte_huge_page(self.ip_ver, lv, self.entry(table_paddr, idx))
    }

    #[allow(clippy::too_many_arguments)]
    fn set_entry(
        &mut self,
        lv: usize,
        table_paddr: u64,
        idx: usize,
        paddr: u64,
        is_table: bool,
        uncached: bool,
        system: bool,
        snooped: bool,
        frag: u32,
        valid: bool,
    ) {
        // Page-table entries hold GPU MC addresses for VRAM (table pointers are
        // always VRAM); host system pages (Sys) are addressed as-is.
        let stored = if system { paddr } else { paddr + self.mc_base };
        let word = pagetable::encode_entry(self.ip_ver, lv, stored, is_table, uncached, system, snooped, frag, valid);
        self.phys.write_u64(table_paddr + (idx as u64) * 8, word);
    }

    // ── the walk ────────────────────────────────────────────────────────────
    fn new_walk(&self, vaddr: u64, mode: WalkMode) -> Walk {
        let rebased = vaddr - self.va_base;
        let lv = self.first_lv;
        let root =
            Frame { paddr: self.root_pt, lv, pte_idx: self.geom.pte_idx(lv, rebased), pte_covers: self.pte_covers(lv) };
        Walk { vaddr: rebased, mode, stack: vec![root], size: 0, off: 0, pending: None }
    }

    fn level_down(&mut self, w: &mut Walk) {
        let top = *w.stack.last().unwrap();
        if !self.pt_valid(top.paddr, top.pte_idx) {
            assert!(w.mode == WalkMode::Create, "not allowed to create a new page table here");
            let child = self.palloc(0x1000, 0x1000, true, true).expect("page-table page");
            self.set_entry(top.lv, top.paddr, top.pte_idx, child, true, false, false, false, 0, true);
        }
        debug_assert!(!self.pt_is_page(top.lv, top.paddr, top.pte_idx), "expected a table, found a page");
        let child_paddr = self.pt_child(top.paddr, top.pte_idx);
        let child_lv = top.lv + 1;
        w.stack.push(Frame {
            paddr: child_paddr,
            lv: child_lv,
            pte_idx: self.geom.pte_idx(child_lv, w.vaddr),
            pte_covers: self.pte_covers(child_lv),
        });
    }

    /// Free the top table if it is now entirely empty (Free mode only), clearing
    /// the parent PDE. Returns whether it freed.
    fn try_free_pt(&mut self, w: &mut Walk) -> bool {
        let top = *w.stack.last().unwrap();
        if w.mode == WalkMode::Free
            && top.paddr != self.root_pt
            && (0..self.pte_cnt(top.lv) as usize).all(|i| !self.pt_valid(top.paddr, i))
        {
            self.pfree(top.paddr, true);
            let parent = w.stack[w.stack.len() - 2];
            self.set_entry(parent.lv, parent.paddr, parent.pte_idx, 0, false, false, false, false, 0, false);
            return true;
        }
        false
    }

    fn level_up(&mut self, w: &mut Walk) {
        loop {
            let freed = self.try_free_pt(w);
            let top = *w.stack.last().unwrap();
            let at_end = top.pte_idx as u64 == self.pte_cnt(top.lv);
            if !(freed || at_end) {
                break;
            }
            let popped = w.stack.pop().unwrap();
            if w.stack.is_empty() {
                break; // unwound past the root (whole space consumed) — no parent to bump
            }
            if popped.pte_idx as u64 == self.pte_cnt(popped.lv) {
                w.stack.last_mut().unwrap().pte_idx += 1;
            }
        }
    }

    fn walk_next(&mut self, w: &mut Walk) -> Option<Step> {
        // Apply the previous yield's advance + level_up now (deferred so the
        // caller's body ran first — Free mode depends on it).
        if let Some((entries, pte_covers)) = w.pending.take() {
            let adv = entries * pte_covers;
            w.size = w.size.saturating_sub(adv);
            w.off += adv;
            w.vaddr += adv;
            w.stack.last_mut().unwrap().pte_idx += entries as usize;
            self.level_up(w);
        }
        if w.size == 0 {
            return None;
        }
        // Descend to the level this step operates at.
        loop {
            let top = *w.stack.last().unwrap();
            let descend = match w.mode {
                // gfx11 supports_huge_page is always true (lv >= PDB2).
                WalkMode::Create => top.pte_covers > w.size || (w.vaddr & (top.pte_covers - 1)) != 0,
                WalkMode::Inspect => {
                    !self.pt_is_page(top.lv, top.paddr, top.pte_idx) && self.pt_valid(top.paddr, top.pte_idx)
                }
                WalkMode::Free => !self.pt_is_page(top.lv, top.paddr, top.pte_idx),
            };
            if descend {
                self.level_down(w);
            } else {
                break;
            }
        }
        let top = *w.stack.last().unwrap();
        let floor = if w.mode == WalkMode::Inspect { 1 } else { 0 };
        let avail = self.pte_cnt(top.lv) - top.pte_idx as u64;
        let entries = (w.size / top.pte_covers).min(avail).max(floor);
        // Parity with tinygrad (support/memory.py:165). In Free mode the descend
        // stops at the first leaf, so a sub-leaf / misaligned unmap (size < a huge
        // leaf's pte_covers) yields entries == 0 — which advances nothing and spins
        // the walk forever. Fail loudly: unmap_range must cover whole leaves (it is
        // only ever called with a full mapping's size). This is `assert!`, not
        // `debug_assert!`: in a release build the latter compiles out and the walk
        // hangs instead of panicking — the exact failure mode this guard exists to
        // prevent.
        assert!(
            entries > 0,
            "page-table walk made no progress: size={:#x} pte_covers={:#x} lv={} mode={:?} \
             (unmap_range must cover whole huge-page leaves)",
            w.size,
            top.pte_covers,
            top.lv,
            w.mode
        );
        w.pending = Some((entries, top.pte_covers));
        Some(Step {
            off: w.off,
            pt_paddr: top.paddr,
            lv: top.lv,
            pte_idx: top.pte_idx,
            entries,
            pte_covers: top.pte_covers,
            vaddr_at_yield: w.vaddr,
        })
    }

    // ── public mapping API ────────────────────────────────────────────────
    /// Map `paddrs` (must sum to `size`) at `vaddr`. Asserts nothing in the
    /// range is already mapped, then writes PTEs (picking 1 GiB/2 MiB/4 KiB
    /// leaves and creating intermediate tables as needed).
    pub fn map_range(
        &mut self,
        vaddr: u64,
        size: u64,
        paddrs: &[(u64, u64)],
        aspace: AddrSpace,
        uncached: bool,
        snooped: bool,
    ) -> VirtMapping {
        assert_eq!(size, paddrs.iter().map(|p| p.1).sum::<u64>(), "size != sum(paddr sizes)");

        // Pass 1: nothing already mapped.
        let mut w = self.new_walk(vaddr, WalkMode::Inspect);
        w.size = size;
        while let Some(s) = self.walk_next(&mut w) {
            for o in 0..s.entries {
                assert!(!self.pt_valid(s.pt_paddr, s.pte_idx + o as usize), "PTE already mapped");
            }
        }

        // Pass 2: create tables + write the leaf PTEs.
        let system = aspace == AddrSpace::Sys;
        let mut w = self.new_walk(vaddr, WalkMode::Create);
        for &(paddr, psize) in paddrs {
            w.size = psize;
            w.off = 0;
            while let Some(s) = self.walk_next(&mut w) {
                let frag = pagetable::frag_size(s.vaddr_at_yield + s.off, s.entries * s.pte_covers);
                for o in 0..s.entries {
                    let p = paddr + s.off + o * s.pte_covers;
                    self.set_entry(
                        s.lv,
                        s.pt_paddr,
                        s.pte_idx + o as usize,
                        p,
                        false,
                        uncached,
                        system,
                        snooped,
                        frag,
                        true,
                    );
                }
            }
        }
        VirtMapping { va_addr: vaddr, size, paddrs: paddrs.to_vec(), aspace, uncached, snooped }
    }

    /// Invalidate every PTE in `[vaddr, vaddr+size)` and reclaim emptied tables.
    ///
    /// `[vaddr, vaddr+size)` must cover whole leaves: a partial unmap of a huge
    /// (2 MiB / 1 GiB) leaf panics, since the walker cannot split a huge page
    /// (neither does tinygrad). In practice this holds — `vfree` always passes a
    /// full mapping's `size`.
    pub fn unmap_range(&mut self, vaddr: u64, size: u64) {
        let mut w = self.new_walk(vaddr, WalkMode::Free);
        w.size = size;
        while let Some(s) = self.walk_next(&mut w) {
            for o in 0..s.entries {
                let idx = s.pte_idx + o as usize;
                assert!(self.pt_valid(s.pt_paddr, idx), "PTE not mapped");
                self.set_entry(s.lv, s.pt_paddr, idx, 0, false, false, false, false, 0, false);
            }
        }
    }

    /// Allocate `size` bytes of VA + backing VRAM and map them. `contiguous`
    /// forces one physical segment; otherwise physical memory is grabbed in
    /// descending `palloc_ranges` chunks. A zero-byte request maps one page, as
    /// KFD's does. Returns `None` on OOM.
    pub fn valloc(&mut self, size: u64, align: u64, uncached: bool, contiguous: bool) -> Option<VirtMapping> {
        let size = round_up(size.max(1), 0x1000);
        let va_align = (1u64 << (bit_length(size) - 1)).max(align.max(1));
        let va = self.va_alloc.alloc(size, va_align)?;

        let paddrs: Vec<(u64, u64)> = if contiguous {
            vec![(self.palloc(size, 0x1000, true, false)?, size)]
        } else {
            let mut rem = size;
            let mut nxt = 0usize;
            let mut paddrs: Vec<(u64, u64)> = Vec::new();
            while rem > 0 {
                while self.palloc_ranges[nxt].0 > rem {
                    nxt += 1;
                }
                let (try_sz, al) = self.palloc_ranges[nxt];
                match self.palloc(try_sz, al, false, false) {
                    Some(p) => {
                        paddrs.push((p, try_sz));
                        rem -= try_sz;
                    }
                    None => {
                        nxt += 1;
                        if nxt == self.palloc_ranges.len() {
                            for (p, _) in &paddrs {
                                self.pfree(*p, false);
                            }
                            self.va_alloc.free(va);
                            return None;
                        }
                    }
                }
            }
            paddrs
        };
        Some(self.map_range(va, size, &paddrs, AddrSpace::Phys, uncached, false))
    }

    /// Unmap + free a [`valloc`](Self::valloc) mapping (VA, physical, tables).
    pub fn vfree(&mut self, vm: &VirtMapping) {
        self.unmap_range(vm.va_addr, vm.size);
        self.va_alloc.free(vm.va_addr);
        for (p, _) in &vm.paddrs {
            self.pfree(*p, false);
        }
    }

    /// Resolve a virtual address to its physical address by walking the live
    /// page table (test/debug helper). `None` if unmapped.
    pub fn resolve(&self, vaddr: u64) -> Option<u64> {
        let va = vaddr - self.va_base;
        let mut paddr = self.root_pt;
        let mut lv = self.first_lv;
        loop {
            let idx = self.geom.pte_idx(lv, va);
            if !self.pt_valid(paddr, idx) {
                return None;
            }
            if self.pt_is_page(lv, paddr, idx) {
                let covers = self.pte_covers(lv);
                return Some(self.pt_child(paddr, idx) + (va & (covers - 1)));
            }
            paddr = self.pt_child(paddr, idx);
            lv += 1;
        }
    }
}
