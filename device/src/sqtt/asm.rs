//! What an `llvm-objdump` instruction line does, read from its mnemonic and, for waits, its counter
//! operands: not a disassembler, and no register def/use.

use serde::{Deserialize, Serialize};

/// A hardware wait counter. gfx11's `vmcnt` is [`Load`](Self::Load), `vscnt` [`Store`](Self::Store).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Counter {
    Load,
    Store,
    Sample,
    Bvh,
    /// gfx12 LDS.
    Ds,
    /// gfx12 scalar memory and messages.
    Km,
    /// gfx11 LDS, scalar memory and messages.
    Lgkm,
    Exp,
}

impl Counter {
    pub const ALL: [Self; 8] =
        [Self::Load, Self::Store, Self::Sample, Self::Bvh, Self::Ds, Self::Km, Self::Lgkm, Self::Exp];

    /// The counter an operand or a `s_wait_<name>` names: gfx11's `vmcnt(…)` or gfx12's `loadcnt`.
    fn named(name: &str) -> Option<Self> {
        Some(match name.strip_suffix("_sat").unwrap_or(name) {
            "vmcnt" | "loadcnt" => Self::Load,
            "vscnt" | "storecnt" => Self::Store,
            "samplecnt" => Self::Sample,
            "bvhcnt" => Self::Bvh,
            "dscnt" => Self::Ds,
            "kmcnt" => Self::Km,
            "lgkmcnt" => Self::Lgkm,
            "expcnt" => Self::Exp,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Load => "load",
            Self::Store => "store",
            Self::Sample => "sample",
            Self::Bvh => "bvh",
            Self::Ds => "ds",
            Self::Km => "km",
            Self::Lgkm => "lgkm",
            Self::Exp => "exp",
        }
    }
}

/// The counter values a wait instruction waits down to: until each named counter is at most its value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Wait {
    counts: [Option<u8>; Counter::ALL.len()],
    /// `s_wait_idle`: every counter down to 0.
    pub idle: bool,
}

impl Wait {
    pub fn get(&self, counter: Counter) -> Option<u8> {
        self.counts[counter as usize]
    }

    pub fn counts(&self) -> impl Iterator<Item = (Counter, u8)> + '_ {
        Counter::ALL.into_iter().filter_map(|c| Some((c, self.get(c)?)))
    }

    fn set(mut self, counter: Counter, value: u8) -> Self {
        self.counts[counter as usize] = Some(value);
        self
    }

    fn idle() -> Self {
        Self { counts: [Some(0); Counter::ALL.len()], idle: true }
    }
}

/// Where a memory instruction goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Space {
    Global,
    /// `buffer_*`, `tbuffer_*`.
    Buffer,
    Scratch,
    /// `flat_*`: global or LDS, decided per lane at run time.
    Flat,
    Image,
    /// `ds_*`: LDS (and GDS).
    Ds,
    /// `lds_param_load`, `lds_direct_load`: counted by `expcnt`.
    LdsDirect,
    /// `s_load`, `s_buffer_*`, `s_store`, `s_atomic`, `s_dcache_*`, `s_memtime`, ...
    Smem,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Access {
    Load,
    Store,
    /// `returns`: gfx11's `glc` or gfx12's `TH_ATOMIC_RETURN`; LDS atomics name it (`_rtn`).
    Atomic {
        returns: bool,
    },
    /// An image sample, gather or MSAA load.
    Sample,
    Bvh,
    /// `global_inv`, `buffer_gl0_inv`, `s_dcache_inv`.
    Invalidate,
    /// `global_wb`, `global_wbinv`, `s_dcache_wb`.
    Writeback,
    /// Prefetches return nothing.
    Prefetch,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Memory {
    pub space: Space,
    pub access: Access,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// `v_wmma`, `v_swmmac`, `v_mfma`, `v_smfmac`.
    Matrix,
    Valu,
    Salu,
    /// `s_branch`, `s_cbranch_*`, `s_setpc`, `s_swappc`, `s_call`, `s_rfe`.
    Branch,
    Memory(Memory),
    /// `s_waitcnt*`, `s_wait_<counter>`, `s_wait_idle`.
    Wait(Wait),
    /// `s_barrier*`.
    Barrier,
    /// `s_sendmsg*`: the SQ counts it with scalar memory.
    SendMsg,
    /// `s_endpgm`, `s_trap`, `s_setprio`, `s_sethalt`, `s_ttrace*`, `s_icache_inv`.
    Message,
    /// `s_nop`, `s_clause`, `s_sleep`, `s_wait_event`, ...: an `IMMED` record.
    Immediate,
    /// `s_wait_alu`, `s_waitcnt_depctr`, `s_delay_alu`, `s_set_vgpr_msb`: the trace sends no token.
    NoToken,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Inst<'a> {
    pub mnemonic: &'a str,
    pub kind: Kind,
}

/// Reads one instruction, with or without its `// address: words` comment.
pub fn classify(line: &str) -> Inst<'_> {
    let text = line.split("//").next().unwrap_or_default().trim();
    let (mnemonic, operands) = text.split_once(char::is_whitespace).unwrap_or((text, ""));
    Inst { mnemonic, kind: kind(mnemonic, operands.trim()) }
}

fn kind(m: &str, operands: &str) -> Kind {
    let any = |prefixes: &[&str]| prefixes.iter().any(|p| m.starts_with(p));
    if any(&["v_wmma", "v_swmmac", "v_mfma", "v_smfmac"]) {
        Kind::Matrix
    } else if m.starts_with("v_") {
        Kind::Valu
    } else if let Some(memory) = memory(m, operands) {
        Kind::Memory(memory)
    } else if m.contains("branch") || any(&["s_setpc", "s_swappc", "s_call", "s_rfe"]) {
        Kind::Branch
    } else if any(&["s_wait_alu", "s_waitcnt_depctr", "s_wait_dep", "s_delay_alu", "s_set_vgpr_msb"]) {
        Kind::NoToken
    } else if let Some(wait) = wait(m, operands) {
        Kind::Wait(wait)
    } else if m.starts_with("s_barrier") {
        Kind::Barrier
    } else if m.starts_with("s_sendmsg") {
        Kind::SendMsg
    } else if any(&["s_endpgm", "s_trap", "s_setprio", "s_sethalt", "s_ttrace", "s_icache"]) {
        Kind::Message
    } else if any(&["s_nop", "s_clause", "s_sleep", "s_set_inst", "s_inst_prefetch", "s_version", "s_wait"]) {
        Kind::Immediate
    } else if m.starts_with("s_") {
        Kind::Salu
    } else {
        Kind::Other
    }
}

const VECTOR: [(&str, Space); 7] = [
    ("global_", Space::Global),
    ("buffer_", Space::Buffer),
    ("tbuffer_", Space::Buffer),
    ("scratch_", Space::Scratch),
    ("flat_", Space::Flat),
    ("image_", Space::Image),
    ("ds_", Space::Ds),
];

const SCALAR: [&str; 9] =
    ["s_load", "s_buffer_", "s_store", "s_scratch", "s_atomic", "s_dcache", "s_prefetch", "s_memtime", "s_memrealtime"];

fn memory(m: &str, operands: &str) -> Option<Memory> {
    let space = match VECTOR.iter().find(|(p, _)| m.starts_with(p)) {
        Some(&(_, space)) => space,
        None if m.starts_with("lds_") => Space::LdsDirect,
        None if SCALAR.iter().any(|p| m.starts_with(p)) => Space::Smem,
        None => return None,
    };
    let has = |s: &str| m.contains(s);
    let access = if has("atomic") || (space == Space::Ds && has("_rtn")) {
        let glc = operands.split_whitespace().any(|o| o == "glc");
        Access::Atomic { returns: glc || operands.contains("TH_ATOMIC_RETURN") || has("_rtn") }
    } else if space == Space::Image && has("bvh") {
        Access::Bvh
    } else if space == Space::Image && (has("sample") || has("gather") || has("msaa")) {
        Access::Sample
    } else if has("prefetch") {
        Access::Prefetch
    } else if has("load") || has("memtime") || has("memrealtime") {
        Access::Load
    } else if has("store") {
        Access::Store
    } else if has("_inv") {
        Access::Invalidate
    } else if has("_wb") {
        Access::Writeback
    } else {
        Access::Other
    };
    Some(Memory { space, access })
}

/// `s_waitcnt vmcnt(N) expcnt(N) lgkmcnt(N)` (or its gfx11 encoded immediate), `s_waitcnt_<counter>
/// null, N`, `s_wait_<counter> N`, `s_wait_loadcnt_dscnt N` (load in bits 13:8, ds in 5:0), `s_wait_idle`.
fn wait(m: &str, operands: &str) -> Option<Wait> {
    if m == "s_wait_idle" {
        return Some(Wait::idle());
    }
    if m == "s_waitcnt" {
        return match number(operands) {
            Some(imm) => Some(gfx11_waitcnt(imm)),
            None => operands
                .split(|c: char| c.is_whitespace() || c == '&' || c == ',')
                .filter(|o| !o.is_empty())
                .try_fold(Wait::default(), |w, operand| {
                    let (name, value) = operand.strip_suffix(')')?.split_once('(')?;
                    Some(w.set(Counter::named(name)?, number(value)?.min(63) as u8))
                }),
        };
    }
    let value = || number(operands.rsplit([',', ' ']).next().unwrap_or_default());
    if let Some(name) = m.strip_prefix("s_waitcnt_") {
        return Some(Wait::default().set(Counter::named(name)?, value()?.min(63) as u8));
    }
    let names = m.strip_prefix("s_wait_")?;
    let imm = value()?;
    match names.split_once('_') {
        None => Some(Wait::default().set(Counter::named(names)?, imm.min(63) as u8)),
        Some((first, second)) => Some(
            Wait::default()
                .set(Counter::named(first)?, (imm >> 8 & 0x3f) as u8)
                .set(Counter::named(second)?, (imm & 0x3f) as u8),
        ),
    }
}

/// gfx11 `s_waitcnt simm16`: expcnt in bits 2:0, lgkmcnt in 9:4, vmcnt in 15:10.
fn gfx11_waitcnt(imm: u32) -> Wait {
    Wait::default()
        .set(Counter::Exp, (imm & 7) as u8)
        .set(Counter::Lgkm, (imm >> 4 & 0x3f) as u8)
        .set(Counter::Load, (imm >> 10 & 0x3f) as u8)
}

fn number(s: &str) -> Option<u32> {
    let s = s.trim();
    match s.strip_prefix("0x") {
        Some(hex) => u32::from_str_radix(hex, 16).ok(),
        None => s.parse().ok(),
    }
}
