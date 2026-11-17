//! AMD's stitcher (`stitch/stitch.cpp`, `trie.cpp`): walks each wave's instruction records along a
//! disassembly from the PCs the trace states, matching record categories to instruction lines and
//! following branches, and attributes a PC to every record it matches.

use std::collections::HashMap;

use super::wave::{Instruction, Wave};
use super::{Category, Layout};

/// The `(pc, line)` pairs of `llvm-objdump -d` output: every line whose `//` comment starts with an
/// address, as in `s_endpgm  // 000000001614: BFB00000`. The PCs are the ELF's; add the load address.
pub fn objdump_lines(dump: &str) -> Vec<(u64, &str)> {
    dump.lines()
        .filter_map(|line| {
            let (_, comment) = line.split_once("//")?;
            let (address, _) = comment.trim_start().split_once(':')?;
            Some((u64::from_str_radix(address, 16).ok()?, line))
        })
        .collect()
}

/// A disassembly line's category (AMD's `InstCategory`, gfx9 cases dropped).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    Skip,
    Smem,
    Salu,
    Vmem,
    Flat,
    Lds,
    Valu,
    Immed,
    Bvh,
    DontKnow,
    Branch,
    BarrierWaitExcluded,
    VMovB64,
}

/// `trie.cpp`'s prefixes.
#[rustfmt::skip]
const PREFIXES: &[(&str, Class)] = &[
    ("s_waitcnt", Class::Immed), ("s_wait_idle", Class::Immed), ("s_wait_dep", Class::Skip),
    ("s_wait_kmcnt", Class::Immed), ("s_wait_loadcnt", Class::Immed), ("s_wait_storecnt", Class::Immed),
    ("s_wait_bvhcnt", Class::Immed), ("s_wait_expcnt", Class::Immed), ("s_wait_dscnt", Class::Immed),
    ("s_wait_samplecnt", Class::Immed), ("s_wait_event", Class::Immed), ("s_wait_alu", Class::Skip),
    ("s_wait_xcnt", Class::Immed), ("s_wait_tensor", Class::Immed), ("s_wait_async", Class::Immed),
    ("s_nop", Class::Immed), ("s_sleep", Class::Immed), ("s_wakeup", Class::Immed), ("s_sendmsg", Class::Immed),
    ("s_setprio ", Class::Immed), ("s_setprio_inc", Class::Salu), ("s_set_inst", Class::Immed),
    ("s_inst_prefetch", Class::Immed), ("s_setkill", Class::Immed), ("s_version", Class::Immed),
    ("s_trap", Class::Immed), ("s_barrie", Class::Immed), ("s_endpgm", Class::Immed), ("s_icache_inv", Class::Immed),
    ("s_dcache_inv", Class::Immed), ("s_incperflvl", Class::Immed), ("s_sethalt", Class::Immed),
    ("s_decperflvl", Class::Immed), ("s_setvskip", Class::Immed), ("s_ttrace", Class::Immed),
    ("s_clause", Class::Immed), ("s_call", Class::Salu), ("s_cvt", Class::Salu), ("s_load", Class::Smem),
    ("s_buffer", Class::Smem), ("s_atomic", Class::Smem), ("s_atc", Class::Smem), ("s_scratch", Class::Smem),
    ("s_store", Class::Smem), ("s_dcache", Class::Smem), ("s_getreg", Class::Salu), ("s_setreg", Class::Salu),
    ("s_memrealtime", Class::Smem), ("buffer_l", Class::Vmem), ("buffer_s", Class::Vmem), ("buffer_a", Class::Vmem),
    ("buffer_g", Class::Vmem), ("buffer_i", Class::Vmem), ("buffer_w", Class::Vmem), ("buffer_p", Class::Vmem),
    ("buffer_d", Class::Vmem), ("buffer_nop", Class::Skip), ("tbuffer_", Class::Vmem), ("flat_", Class::Flat),
    ("global_", Class::Vmem), ("scratch_", Class::Vmem), ("cluster_", Class::Vmem), ("tensor_", Class::Vmem),
    ("ds_", Class::Lds), ("lds_", Class::Lds), ("dds_", Class::Vmem), ("s_add", Class::Salu), ("s_sub", Class::Salu),
    ("s_and", Class::Salu), ("s_or", Class::Salu), ("s_pack", Class::Salu), ("s_prefetch", Class::Smem),
    ("s_round", Class::Salu), ("s_denorm", Class::Salu), ("s_ctz", Class::Salu), ("s_cls", Class::Salu),
    ("s_cmp", Class::Salu), ("s_bit", Class::Salu), ("s_set_vgpr", Class::Skip), ("s_mov", Class::Salu),
    ("s_monitor", Class::Immed), ("s_mul", Class::Salu), ("s_ash", Class::Salu), ("s_bf", Class::Salu),
    ("s_lsh", Class::Salu), ("s_min", Class::Salu), ("s_max", Class::Salu), ("s_abs", Class::Salu),
    ("s_cmov", Class::Salu), ("s_not", Class::Salu), ("s_nor", Class::Salu), ("s_wqm", Class::Salu),
    ("s_quadmask", Class::Salu), ("s_brev", Class::Salu), ("s_bcnt", Class::Salu), ("s_f", Class::Salu),
    ("s_sext", Class::Salu), ("s_cselect", Class::Salu), ("s_xor", Class::Salu), ("s_xnor", Class::Salu),
    ("s_nand", Class::Salu), ("s_delay", Class::Skip), ("s_getpc", Class::Salu), ("s_setpc", Class::Salu),
    ("s_swappc", Class::Salu), ("s_get_pc", Class::Salu), ("s_set_pc", Class::Salu), ("s_swap_pc", Class::Salu),
    ("image_bvh", Class::Bvh), ("image_l", Class::Vmem), ("image_s", Class::Vmem), ("image_a", Class::Vmem),
    ("image_m", Class::Vmem), ("image_g", Class::Vmem),
];

/// AMD's prefix trie. A character with no edge leaves the walk where it is, so a mnemonic matches the
/// first prefix its characters spell as a subsequence along the trie.
/// A node's edges `(byte, child)` and the class of the prefix ending there.
type Node = (Vec<(u8, usize)>, Option<Class>);

struct Trie {
    nodes: Vec<Node>,
}

impl Trie {
    fn new() -> Self {
        let mut trie = Self { nodes: vec![(vec![], None)] };
        for &(prefix, class) in PREFIXES {
            let mut node = 0;
            for c in prefix.bytes() {
                node = match trie.nodes[node].0.iter().find(|(e, _)| *e == c) {
                    Some(&(_, child)) => child,
                    None => {
                        trie.nodes.push((vec![], None));
                        let child = trie.nodes.len() - 1;
                        trie.nodes[node].0.push((c, child));
                        child
                    }
                };
            }
            trie.nodes[node].1 = Some(class);
        }
        trie
    }

    fn class(&self, inst: &str, layout: Layout) -> Class {
        if inst.contains("branch") {
            return Class::Branch;
        }
        let rdna4 = layout == Layout::Rdna4;
        if inst.starts_with("v_") {
            return if rdna4 && inst.starts_with("v_mov_b64") { Class::VMovB64 } else { Class::Valu };
        }
        let mut node = 0;
        for c in inst.bytes() {
            if let Some(&(_, child)) = self.nodes[node].0.iter().find(|(e, _)| *e == c) {
                node = child;
            }
            if let Some(class) = self.nodes[node].1 {
                return if rdna4 && class == Class::Lds && inst.starts_with("ds_bvh") { Class::Bvh } else { class };
            }
        }
        if inst.starts_with("s_") { Class::Salu } else { Class::DontKnow }
    }
}

struct Line<'a> {
    pc: u64,
    text: &'a str,
    class: Class,
    target: u64,
    parsed: bool,
    not_taken: u32,
}

/// `getjump_loc`: the branch's `simm16` from the encoding in the line's comment, else its last operand.
fn branch_target(pc: u64, text: &str, comment: Option<&str>) -> u64 {
    let encoded = comment
        .and_then(|c| c.split_once(':'))
        .and_then(|(_, words)| words.split_whitespace().next())
        .and_then(|word| u32::from_str_radix(word, 16).ok());
    let delta = match encoded {
        Some(word) => i64::from(word as u16 as i16),
        None => {
            let last = text.split_whitespace().last().unwrap_or_default();
            let digits = last.find(|c: char| !c.is_ascii_digit() && c != '-').unwrap_or(last.len());
            let delta = last[..digits].parse::<i64>().unwrap_or(0);
            if delta >= 32768 { delta - 65536 } else { delta }
        }
    };
    pc.wrapping_add_signed(4 + 4 * delta)
}

/// The target of the direct branch (`s_branch`, `s_cbranch_*`) an [`objdump_lines`] line at `pc` holds,
/// as the stitcher follows it.
pub fn branch(pc: u64, line: &str) -> Option<u64> {
    let (text, comment) = line.split_once("//").map_or((line, None), |(t, c)| (t, Some(c)));
    let text = text.trim();
    text.contains("branch").then(|| branch_target(pc, text, comment))
}

fn trivial(wave: Category, line: Class) -> bool {
    matches!(
        (wave, line),
        (Category::None, Class::Skip)
            | (Category::Smem, Class::Smem)
            | (Category::Salu, Class::Salu)
            | (Category::Vmem | Category::Flat, Class::Vmem | Class::Flat)
            | (Category::Lds, Class::Lds)
            | (Category::Valu, Class::Valu)
            | (Category::Immed | Category::Msg, Class::Immed)
            | (Category::Bvh, Class::Bvh)
    )
}

fn skippable(line: Class) -> bool {
    matches!(line, Class::Skip | Class::Smem | Class::Vmem | Class::Flat | Class::Lds | Class::Valu | Class::Immed)
}

fn elaborate(wave: Category, line: Class) -> bool {
    trivial(wave, line)
        || match line {
            Class::Branch => wave == Category::Jump,
            _ => wave == Category::Immed && skippable(line),
        }
}

const MAX_FAILED_STITCHES: u32 = 1000;

/// Attributes PCs to waves' instruction records. Lines keep what earlier waves taught the walk (a
/// branch seen not taken, an excluded barrier), so waves are stitched in [`Trace::waves`] order.
///
/// [`Trace::waves`]: super::wave::Trace::waves
pub struct Stitcher<'a> {
    lines: Vec<Line<'a>>,
    index: HashMap<u64, usize>,
    layout: Layout,
}

impl<'a> Stitcher<'a> {
    /// `code`: `(pc, llvm-objdump instruction line)`; a branch's target comes from the encoding the
    /// line's `// address: words` comment shows, else from a decimal last operand.
    pub fn new(code: &[(u64, &'a str)], layout: Layout) -> Self {
        let trie = Trie::new();
        let mut lines: Vec<Line<'a>> = code
            .iter()
            .map(|&(pc, line)| {
                let text = line.split_once("//").map_or(line, |(t, _)| t).trim();
                let class = trie.class(text, layout);
                let target = if class == Class::Branch { branch(pc, line).unwrap_or(0) } else { 0 };
                Line { pc, text, class, target, parsed: false, not_taken: 0 }
            })
            .collect();
        lines.sort_by_key(|line| line.pc);
        lines.dedup_by_key(|line| line.pc);
        let index = lines.iter().enumerate().map(|(i, line)| (line.pc, i)).collect();
        Self { lines, index, layout }
    }

    fn find(&self, pc: u64) -> Option<usize> {
        self.index.get(&pc).copied()
    }

    /// The PC after line `k`: the next line's, or 4 bytes on past the last.
    fn next_pc(&self, k: usize) -> u64 {
        self.lines.get(k + 1).map_or(self.lines[k].pc + 4, |line| line.pc)
    }

    fn next(&self, k: usize) -> Option<usize> {
        self.find(self.next_pc(k))
    }

    fn jump(&self, k: usize) -> Option<usize> {
        self.find(self.lines[k].target)
    }

    /// Stitches one wave and returns how many of its records, in order, the walk matched.
    pub fn stitch(&mut self, wave: &mut Wave) -> usize {
        let (matched, gaps) = self.walk(wave);
        if self.layout == Layout::Rdna4 && wave.exclude_barrier_wait {
            insert_barrier_waits(&mut wave.instructions, &gaps);
        }
        if !wave.complete {
            wave.end = wave.instructions.iter().map(|i| i.time + i.duration + 4).fold(wave.end, i64::max);
        }
        matched
    }

    /// `stitchWave`: the matched prefix and the excluded barrier waits, `(record index, pc)`.
    fn walk(&mut self, wave: &mut Wave) -> (usize, Vec<(usize, u64)>) {
        let insts = &mut wave.instructions;
        let (mut failed, mut i, mut skipped_immed, mut anchor) = (0, 0, 0, 0);
        let (mut next, mut gaps) = (None, vec![]);
        while i < insts.len() && failed < MAX_FAILED_STITCHES {
            while let Some(&(at, pc)) = wave.pcs.get(anchor).filter(|(at, _)| *at <= i) {
                if at == i {
                    if self.find(pc).is_some() {
                        insts[i].pc = pc;
                    } else if let Some(&(skip_to, _)) = wave.pcs.get(anchor + 1) {
                        i = skip_to;
                        if i >= insts.len() {
                            return (i, gaps);
                        }
                    }
                }
                anchor += 1;
            }
            let inst = insts[i];
            if inst.category == Category::Trap {
                i += 1;
                continue;
            }
            if inst.pc != 0 {
                i += 1;
                let Some(mut k) = self.find(inst.pc) else { return (0, gaps) };
                next = loop {
                    if self.lines[k].class != Class::Skip {
                        let jump = self.lines[k].class == Class::Branch && inst.category == Category::Jump;
                        break if jump { self.jump(k) } else { self.next(k) };
                    }
                    insts[i - 1].pc = self.next_pc(k);
                    match self.next(k) {
                        Some(n) => k = n,
                        None => break None,
                    }
                };
                continue;
            }
            let Some(k) = next.take().filter(|&k| !self.lines[k].text.is_empty()) else { break };
            next = self.next(k);
            let (class, text) = (self.lines[k].class, self.lines[k].text);
            if matches!(class, Class::DontKnow | Class::Skip) {
                continue;
            }
            let barrier_wait = class == Class::BarrierWaitExcluded
                || (!self.lines[k].parsed && class == Class::Immed && text.starts_with("s_barrier_wait"));
            if wave.exclude_barrier_wait && barrier_wait && self.barrier_excluded(insts, i, next) {
                gaps.push((i, self.lines[k].pc));
                self.lines[k].class = Class::BarrierWaitExcluded;
                continue;
            }
            let category = inst.category;
            let mut matched = true;
            if trivial(category, class) {
            } else if class == Class::Branch {
                if category == Category::Next {
                    self.lines[k].not_taken += 1;
                } else if category != Category::Jump && !text.starts_with("s_branch") {
                    // AMD counts no taken branches, so an untraced branch is only skipped once seen not taken.
                    if self.lines[k].not_taken == 0 {
                        break;
                    }
                    continue;
                } else {
                    next = self.jump(k);
                }
            } else if self.layout == Layout::Rdna4 && class == Class::VMovB64 && category == Category::Valu {
                insts[i].duration = inst.duration.max(inst.stall + 2);
            } else {
                matched = false;
                if category == Category::Immed && skippable(class) {
                    matched = true;
                    skipped_immed += 1;
                } else if text.starts_with("s_waitcnt") || text.contains("_load_") {
                    if skipped_immed > 0 && text.starts_with("s_waitcnt") {
                        matched = true;
                        skipped_immed -= 1;
                    } else if !text.contains("scratch_") {
                        break;
                    }
                }
            }
            if matched {
                insts[i].pc = self.lines[k].pc;
                i += 1;
                failed = 0;
                self.lines[k].parsed = true;
            } else {
                failed += 1;
            }
        }
        (i, gaps)
    }

    /// An excluded `s_barrier_wait` sends either an `IMMED` or nothing: it sent nothing unless record `i`
    /// is an `IMMED` and the lines after the barrier fail to match the records from `i` on.
    fn barrier_excluded(&self, insts: &[Instruction], i: usize, mut look: Option<usize>) -> bool {
        if insts[i].category != Category::Immed {
            return true;
        }
        let (mut add, mut kind, mut idx) = (true, None, i);
        while idx < insts.len() && add {
            let Some(k) = look else { break };
            let (class, category) = (self.lines[k].class, insts[idx].category);
            look = if class == Class::Branch && category == Category::Jump { self.jump(k) } else { self.next(k) };
            if look.is_none() {
                break;
            }
            if matches!(class, Class::DontKnow | Class::Skip) {
                idx += 1;
                continue;
            }
            add = elaborate(category, class);
            if category != Category::Immed && class != Class::Valu {
                if kind.is_some_and(|kind| kind != class) {
                    break;
                }
                kind = Some(class);
            }
            // AMD's loop steps twice here.
            idx += 2;
        }
        add
    }
}

/// `insert_gfx12_barrier_wait`: an `IMMED` record for each excluded barrier wait, filling the gap
/// between the records around it.
fn insert_barrier_waits(insts: &mut Vec<Instruction>, gaps: &[(usize, u64)]) {
    if insts.is_empty() || gaps.is_empty() {
        return;
    }
    let mut out = Vec::with_capacity(insts.len() + gaps.len());
    let mut prev = 0;
    for &(index, pc) in gaps {
        if index <= prev {
            continue;
        }
        out.extend_from_slice(&insts[prev..index]);
        let time = insts[index - 1].time + insts[index - 1].duration;
        let current = &mut insts[index];
        if current.time == time && current.duration > 1 {
            current.time += 1;
            current.duration -= 1;
        }
        out.push(Instruction { time, category: Category::Immed, duration: current.time - time, stall: 0, pc });
        prev = index;
    }
    out.extend_from_slice(&insts[prev..]);
    *insts = out;
}
