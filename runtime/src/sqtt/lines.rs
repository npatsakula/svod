//! Which tile-program statement each instruction comes from. A kernel built with a line table over
//! its `kernel.tk` listing (the offline twin of a traced one) is read with `llvm-dwarfdump
//! --debug-line`; its rows reach a capture's instructions only where the twin's `.text` is the
//! captured code object's byte for byte.

use std::collections::BTreeMap;
use std::num::NonZeroU32;

use gimli::Reader;
use object::{Object, ObjectSection, ObjectSymbol};
use serde::{Deserialize, Serialize};
use snafu::{ResultExt, Snafu};
use svod_device::sqtt::stitch::objdump_lines;

use super::{Capture, Error, Result, run_tool};

/// A statement's line in `kernel.tk` and its unroll copy (from 1; 0 outside an unrolled loop): the
/// line and column a line table holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Stmt {
    pub line: NonZeroU32,
    pub copy: u32,
}

/// Each instruction's statement by ELF address; an instruction on line 0 is left out.
pub type Lines = BTreeMap<u64, Stmt>;

/// A kernel's statements: each instruction's, and `kernel.tk`, the listing whose lines they name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Statements {
    pub lines: Lines,
    pub listing: String,
}

/// Why a twin's line table does not describe a capture.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum TwinError {
    #[snafu(display("a code object has no .text"))]
    NoText,
    #[snafu(display(".text differs: {captured} vs {twin} bytes, first at {at:#x}"))]
    TextDiffers { captured: usize, twin: usize, at: usize },
    #[snafu(display("the twin has no line table"))]
    NoLineTable,
    #[snafu(display("{source}"))]
    Tool {
        #[snafu(source(from(Error, Box::new)))]
        source: Box<Error>,
    },
}

/// One row of a line table: from `address` on, until the next row, code is `line` and `column`;
/// an `end` row closes its sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineRow {
    pub address: u64,
    pub line: u32,
    pub column: u32,
    pub end: bool,
}

/// The rows of `llvm-dwarfdump --debug-line`: each line opening with a hex address, then the line
/// and column; `end_sequence` among its flags closes a sequence.
pub fn parse_debug_line(dump: &str) -> Vec<LineRow> {
    dump.lines()
        .filter_map(|l| {
            let mut words = l.split_whitespace();
            let address = u64::from_str_radix(words.next()?.strip_prefix("0x")?, 16).ok()?;
            let (line, column) = (words.next()?.parse().ok()?, words.next()?.parse().ok()?);
            Some(LineRow { address, line, column, end: l.split_whitespace().any(|w| w == "end_sequence") })
        })
        .collect()
}

/// The file a tk kernel's line table names, whose DWARF 5 source is the kernel's listing.
pub const LISTING: &str = "kernel.tk";

/// Why a code object's DWARF does not read.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum DwarfError {
    #[snafu(display("a code object does not parse: {source}"))]
    Elf { source: object::Error },
    #[snafu(display("its DWARF does not parse: {source}"))]
    Dwarf { source: gimli::Error },
}

/// A debug section's relocations by their offset in it: the value each resolves to, and whether
/// what the field holds adds to it (an implicit addend). A relocatable code object (what clang
/// makes of a kernel) leaves the addresses and string offsets of its DWARF to these.
#[derive(Debug)]
struct Relocations(BTreeMap<u64, (u64, bool)>);

impl Relocations {
    /// Each of `section`'s relocations as its symbol's value plus its addend; DWARF's are all
    /// absolute (`R_AMDGPU_ABS32`/`ABS64`), and an implicit addend adds what the field holds.
    fn of(file: &object::File, section: &object::Section) -> Result<Self, object::Error> {
        let mut out = BTreeMap::new();
        for (offset, r) in section.relocations() {
            let base = match r.target() {
                object::RelocationTarget::Symbol(i) => file.symbol_by_index(i)?.address(),
                object::RelocationTarget::Section(i) => file.section_by_index(i)?.address(),
                _ => 0,
            };
            out.insert(offset, (base.wrapping_add(r.addend() as u64), r.has_implicit_addend()));
        }
        Ok(Self(out))
    }

    fn apply(&self, offset: usize, value: u64) -> u64 {
        match self.0.get(&(offset as u64)) {
            Some(&(resolved, true)) => value.wrapping_add(resolved),
            Some(&(resolved, false)) => resolved,
            None => value,
        }
    }
}

impl gimli::read::Relocate for &Relocations {
    fn relocate_address(&self, offset: usize, value: u64) -> gimli::Result<u64> {
        Ok(self.apply(offset, value))
    }

    fn relocate_offset(&self, offset: usize, value: usize) -> gimli::Result<usize> {
        usize::try_from(self.apply(offset, value as u64)).map_err(|_| gimli::Error::UnsupportedOffset)
    }
}

/// A code object's line table, and the source its [`LISTING`] file carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineTable {
    pub rows: Vec<LineRow>,
    pub listing: Option<String>,
}

/// A code object's [`LineTable`]; `None` where it has none (a kernel built without one).
pub fn line_table(elf: &[u8]) -> Result<Option<LineTable>, DwarfError> {
    use gimli::{EndianSlice, LittleEndian, RelocateReader};
    static NONE: Relocations = Relocations(BTreeMap::new());
    let file = object::File::parse(elf).context(ElfSnafu)?;
    let sections: BTreeMap<&str, (&[u8], Relocations)> = file
        .sections()
        .filter_map(|s| Some((s.name().ok().filter(|n| n.starts_with(".debug_"))?, s)))
        .map(|(name, s)| Ok((name, (s.data()?, Relocations::of(&file, &s)?))))
        .collect::<Result<_, object::Error>>()
        .context(ElfSnafu)?;
    let dwarf = gimli::Dwarf::load(|id| {
        let (data, relocations) = sections.get(id.name()).map_or((&[][..], &NONE), |(data, r)| (*data, r));
        Ok::<_, gimli::Error>(RelocateReader::new(EndianSlice::new(data, LittleEndian), relocations))
    })
    .context(DwarfSnafu)?;
    let (mut rows, mut listing, mut tabled) = (vec![], None, false);
    let mut units = dwarf.units();
    while let Some(header) = units.next().context(DwarfSnafu)? {
        let unit = dwarf.unit(header).context(DwarfSnafu)?;
        let Some(program) = unit.line_program.clone() else { continue };
        tabled = true;
        let string = |attr| -> Result<String, DwarfError> {
            Ok(dwarf.attr_string(&unit, attr).context(DwarfSnafu)?.to_string_lossy().context(DwarfSnafu)?.into_owned())
        };
        for file in program.header().file_names() {
            if listing.is_none() && string(file.path_name())? == LISTING {
                listing = file.source().map(string).transpose()?;
            }
        }
        let mut program = program.rows();
        while let Some((_, row)) = program.next_row().context(DwarfSnafu)? {
            let column = match row.column() {
                gimli::ColumnType::LeftEdge => 0,
                gimli::ColumnType::Column(c) => c.get(),
            };
            rows.push(LineRow {
                address: row.address(),
                line: row.line().map_or(0, |l| l.get() as u32),
                column: column as u32,
                end: row.end_sequence(),
            });
        }
    }
    Ok(tabled.then_some(LineTable { rows, listing }))
}

/// `llvm-dwarfdump --debug-line` of a code object, the tool found beside clang.
pub fn debug_line(elf: &[u8]) -> Result<String> {
    run_tool("llvm-dwarfdump", &["--debug-line", "-"], elf)
}

/// Each of `pcs` on the row at or before it within its sequence; a PC past every sequence's end, or
/// on line 0, is left out.
pub fn locate(rows: &[LineRow], pcs: impl IntoIterator<Item = u64>) -> Lines {
    let mut sorted = rows.to_vec();
    // A sequence's end and the next one's start may share an address: the start wins.
    sorted.sort_by_key(|r| (r.address, !r.end));
    pcs.into_iter()
        .filter_map(|pc| {
            let row = sorted[..sorted.partition_point(|r| r.address <= pc)].last()?;
            let line = NonZeroU32::new(row.line).filter(|_| !row.end)?;
            Some((pc, Stmt { line, copy: row.column }))
        })
        .collect()
}

/// A code object's `.text`: its address and bytes.
pub(crate) fn text_section(elf: &[u8]) -> Option<(u64, &[u8])> {
    let file = object::File::parse(elf).ok()?;
    let text = file.section_by_name(".text")?;
    Some((text.address(), text.data().ok()?))
}

/// Whether two code objects hold one `.text`; the addresses of the two.
pub fn same_text(captured: &[u8], twin: &[u8]) -> Result<(u64, u64), TwinError> {
    let (Some((at, a)), Some((twin_at, b))) = (text_section(captured), text_section(twin)) else {
        return NoTextSnafu.fail();
    };
    if a != b {
        let at = a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
        return TextDiffersSnafu { captured: a.len(), twin: b.len(), at }.fail();
    }
    Ok((at, twin_at))
}

impl Capture {
    /// The statements of `twin` (a build of the traced kernel with a line table over `listing`) at
    /// the captured instructions, kept as [`Self::statements`] where the two `.text`s are one; else
    /// why not, the capture left as it was.
    pub fn attach_twin(&mut self, twin: &[u8], listing: &str) -> Result<(), TwinError> {
        let (at, twin_at) = same_text(&self.trace.code.elf, twin)?;
        let rows = parse_debug_line(&debug_line(twin).context(ToolSnafu)?);
        snafu::ensure!(!rows.is_empty(), NoLineTableSnafu);
        let pcs = objdump_lines(&self.disassembly).into_iter().map(|(pc, _)| pc - at + twin_at);
        let lines = locate(&rows, pcs).into_iter().map(|(pc, stmt)| (pc - twin_at + at, stmt)).collect();
        self.statements = Some(Statements { lines, listing: listing.to_string() });
        Ok(())
    }
}
