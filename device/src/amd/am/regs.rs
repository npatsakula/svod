//! AM register access: vendored register tables + the resolver.
//!
//! The tables in `regs_gen.rs` are generated **once** (by
//! `device/tools/gen_am_regs.py`) and committed — the cargo build never depends
//! on the tinygrad submodule. At boot the right module is chosen by the
//! discovered `ip_ver` ([`select`]). A `RegDef` carries the dword offset within its
//! IP segment plus its bitfields; the absolute MMIO address (segment base +
//! offset) is resolved later from the IP-discovery register bases.

/// One register field: an inclusive bit range `[lo, hi]`.
#[derive(Clone, Copy, Debug)]
pub struct RegField {
    pub name: &'static str,
    pub lo: u8,
    pub hi: u8,
}

/// A register definition: dword `offset` within IP `segment`, plus bitfields.
#[derive(Clone, Copy, Debug)]
pub struct RegDef {
    pub name: &'static str,
    pub offset: u32,
    pub segment: u8,
    pub fields: &'static [RegField],
}

/// A vendored register module: `(prefix, (maj, min, rev), table)`.
pub type RegModule = (&'static str, (u8, u8, u8), &'static [RegDef]);

include!("regs_gen.rs");

impl RegDef {
    /// The named field, if present.
    pub fn field(&self, name: &str) -> Option<&RegField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// OR together `value << field.lo` for each named field. Panics on an
    /// unknown field name or a value wider than its field (programming errors).
    pub fn encode(&self, values: &[(&str, u32)]) -> u64 {
        let mut acc = 0u64;
        for (name, v) in values {
            let f = self.field(name).unwrap_or_else(|| panic!("register {} has no field {name}", self.name));
            let width = u32::from(f.hi - f.lo) + 1;
            assert!(width >= 32 || *v >> width == 0, "{}.{name} = {v:#x} overflows its {width} bits", self.name);
            acc |= (*v as u64) << f.lo;
        }
        acc
    }

    /// Read-modify-write: clear each named field's bits in `base` and OR in the
    /// new value (so reserved/ECO bits the PF set are preserved). Panics on an
    /// unknown field name.
    pub fn encode_onto(&self, mut base: u64, values: &[(&str, u32)]) -> u64 {
        for (name, v) in values {
            let f = self.field(name).unwrap_or_else(|| panic!("register {} has no field {name}", self.name));
            let width = (f.hi - f.lo + 1) as u32;
            let mask = if width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
            base = (base & !(mask << f.lo)) | (((*v as u64) & mask) << f.lo);
        }
        base
    }

    /// Extract a field's value from a register word.
    pub fn get(&self, val: u64, name: &str) -> u32 {
        let f = self.field(name).unwrap_or_else(|| panic!("register {} has no field {name}", self.name));
        let width = (f.hi - f.lo + 1) as u32;
        let mask = if width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
        ((val >> f.lo) & mask) as u32
    }
}

impl RegDef {
    /// Absolute MMIO dword index for instance bases from IP discovery.
    pub fn dword_index(&self, bases: &[u64]) -> usize {
        bases[self.segment as usize] as usize + self.offset as usize
    }
}

/// Find a register by name within a module's table (linear scan; not hot —
/// callers resolve a register once at setup).
pub fn find<'a>(regs: &'a [RegDef], name: &str) -> Option<&'a RegDef> {
    regs.iter().find(|r| r.name == name)
}

/// Select the register module for `prefix` (e.g. `"gc"`) whose version is the
/// greatest `<= ip_ver` sharing the same major. `None` if no matching module is
/// vendored (re-run the generator with a wider module list to add one).
pub fn select(prefix: &str, ip_ver: (u8, u8, u8)) -> Option<&'static [RegDef]> {
    AM_REG_MODULES
        .iter()
        .filter(|(p, v, _)| *p == prefix && v.0 == ip_ver.0 && *v <= ip_ver)
        .max_by_key(|(_, v, _)| *v)
        .map(|(_, _, regs)| *regs)
}
