//! GPU assembly read as text, for comparing what kernels issue: SASS
//! (`nvdisasm -c`, `cuobjdump -sass`), PTX, and AMDGCN (`clang -S`,
//! `llvm-objdump -d --symbolize-operands`) alike. [`main_loop`] finds a
//! kernel's main loop and counts its opcodes, [`function`] cuts one kernel
//! out of a text holding several, [`table`] sets main loops side by side.

use std::collections::BTreeMap;

/// One assembly line: a label, or an instruction's opcode with its branch
/// target.
#[derive(Debug, PartialEq, Eq)]
enum Line {
    Label(String),
    Op(String, Option<String>),
}

/// A SASS address (`/*0a30*/`, `0x0a30`) as the label a branch names it by.
fn address(hex: &str) -> Option<String> {
    let hex = hex.strip_prefix("0x").unwrap_or(hex);
    (!hex.is_empty()).then_some(())?;
    u64::from_str_radix(hex, 16).ok().map(|a| format!("{a:#x}"))
}

/// `asm`'s labels and instructions in order. SASS addresses and encodings
/// sit in `/* */` (`cuobjdump -sass` branches to addresses, so an address
/// is a label too), PTX and `llvm-objdump` comments follow `//`, AMDGCN's
/// `;`, SASS and PTX end a statement with `;`, and `llvm-objdump` prints a
/// label as `<address> <name>:`.
fn lines(asm: &str) -> Vec<Line> {
    let mut out = vec![];
    for l in asm.lines() {
        let Some(code) = l.split("//").next() else { continue };
        let trimmed = code.trim_start();
        if let Some(at) = trimmed.strip_prefix("/*").and_then(|r| r.split_once("*/")).and_then(|(a, _)| address(a)) {
            out.push(Line::Label(at));
        }
        let mut s = code.to_string();
        while let (Some(a), Some(b)) = (s.find("/*"), s.find("*/")) {
            s.replace_range(a..b + 2, "");
        }
        let Some(t) = s.split(';').next().map(str::trim) else { continue };
        if let Some(name) = objdump_label(t) {
            out.push(Line::Label(name.to_string()));
            continue;
        }
        if t.ends_with(':') && !t.contains(char::is_whitespace) {
            out.push(Line::Label(t.trim_end_matches(':').to_string()));
            continue;
        }
        if t.is_empty() || t.starts_with(['.', '{', '}', '-']) {
            continue;
        }
        let words: Vec<&str> = t.split_whitespace().filter(|w| !w.starts_with('@')).collect();
        let Some(op) = words.first() else { continue };
        let branch = ["bra", "BRA", "s_branch", "s_cbranch"].iter().any(|b| op.starts_with(b));
        let target = words
            .last()
            .filter(|_| branch)
            .map(|w| w.trim_matches(|c| "`(),".contains(c)))
            .map(|w| if w.starts_with("0x") { address(w).unwrap_or_else(|| w.to_string()) } else { w.to_string() });
        out.push(Line::Op(op.to_string(), target));
    }
    out
}

/// The name of an `llvm-objdump` label line, `0000000000000400 <L1>:`.
fn objdump_label(t: &str) -> Option<&str> {
    let (at, rest) = t.split_once(' ')?;
    at.chars().all(|c| c.is_ascii_hexdigit()).then_some(())?;
    rest.strip_prefix('<')?.strip_suffix(">:")
}

/// Whether `op` is a matrix-core instruction: `HMMA`/`HGMMA`/`UTCHMMA`
/// (SASS), `mma`/`wgmma` (PTX), `v_mfma`/`v_wmma`/`v_smfmac` (AMDGCN).
fn is_matrix(op: &str) -> bool {
    ["MMA", "mma", "mfma"].iter().any(|m| op.contains(m))
}

/// A main loop's opcodes by count, the most issued first (ties by name).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MainLoop {
    pub ops: Vec<(String, usize)>,
}

impl MainLoop {
    /// The instructions of one trip.
    pub fn instructions(&self) -> usize {
        self.ops.iter().map(|o| o.1).sum()
    }

    /// How many of one trip's instructions are `op`.
    pub fn count(&self, op: &str) -> usize {
        self.ops.iter().find(|o| o.0 == op).map_or(0, |o| o.1)
    }
}

/// The opcodes of the innermost loop (a backward branch's span) that issues
/// a matrix instruction: the main loop's step. `None` where no loop does
/// (the reduction unrolled whole, or no matrix core).
pub fn main_loop(asm: &str) -> Option<MainLoop> {
    let lines = lines(asm);
    let at = |name: &str| lines.iter().position(|l| matches!(l, Line::Label(n) if n == name));
    let loops = lines.iter().enumerate().filter_map(|(j, l)| match l {
        Line::Op(_, Some(target)) => at(target).filter(|&i| i < j).map(|i| (i, j)),
        _ => None,
    });
    let body = |(i, j): (usize, usize)| {
        lines[i..=j].iter().filter_map(|l| if let Line::Op(op, _) = l { Some(op.as_str()) } else { None })
    };
    let span = loops.filter(|&span| body(span).any(is_matrix)).min_by_key(|(i, j)| j - i)?;
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    body(span).for_each(|op| *counts.entry(op).or_default() += 1);
    let mut ops: Vec<(String, usize)> = counts.into_iter().map(|(op, n)| (op.to_string(), n)).collect();
    ops.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    Some(MainLoop { ops })
}

/// The function a line starts, by name: `.text.<name>:` (`nvdisasm`),
/// `Function : <name>` (`cuobjdump -sass`), `<address> <name>:`
/// (`llvm-objdump`; its symbolized `<L3>` block labels start none), `.entry
/// <name>(` (PTX), or an undotted label in the first column (AMDGCN).
fn starts(line: &str) -> Option<&str> {
    let t = line.split(';').next()?.trim();
    if let Some(name) = t.strip_prefix(".text.").and_then(|r| r.strip_suffix(':')) {
        return Some(name);
    }
    if let Some(name) = t.strip_prefix("Function : ") {
        return Some(name.trim());
    }
    if let Some(name) = objdump_label(t) {
        let block = name.strip_prefix('L').is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()));
        return (!block).then_some(name);
    }
    if let Some((_, rest)) = t.split_once(".entry ") {
        return rest.split(['(', ' ']).next().filter(|n| !n.is_empty());
    }
    let name = t.strip_suffix(':')?;
    // A YAML key of the metadata (`amdhsa.kernels:`) holds a dot too.
    let local = name.is_empty() || name.contains(['.', '$']) || name.contains(char::is_whitespace);
    (!local && !line.starts_with(char::is_whitespace)).then_some(name)
}

/// The functions `asm` holds, in order (`nvdisasm` labels a function with
/// symbols twice, `<name>:` and `.text.<name>:`).
pub fn functions(asm: &str) -> Vec<&str> {
    let mut names: Vec<&str> = asm.lines().filter_map(starts).collect();
    names.dedup();
    names
}

/// The text of function `name` in `asm`: from the line that starts it to
/// the next function's.
pub fn function<'a>(asm: &'a str, name: &str) -> Option<&'a str> {
    let mut offset = 0;
    let mut begin = None;
    for line in asm.split_inclusive('\n') {
        match (starts(line), begin) {
            (Some(n), None) if n == name => begin = Some(offset),
            (Some(n), Some(b)) if n != name => return Some(&asm[b..offset]),
            _ => {}
        }
        offset += line.len();
    }
    begin.map(|b| &asm[b..])
}

/// Main loops side by side: one row per opcode any of them issues, its
/// count in each column, the most issued first, under a row of
/// instruction counts.
pub fn table(columns: &[(&str, &MainLoop)]) -> String {
    let mut ops: Vec<(&str, usize)> = vec![];
    for (_, l) in columns {
        for (op, _) in &l.ops {
            if !ops.iter().any(|o| o.0 == op) {
                let most = columns.iter().map(|(_, l)| l.count(op)).max().unwrap_or(0);
                ops.push((op, most));
            }
        }
    }
    ops.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let width = ops.iter().map(|o| o.0.len()).chain(["instructions".len()]).max().unwrap_or(0);
    let row = |first: &str, cell: &dyn Fn(&MainLoop) -> String| {
        let cells = columns.iter().map(|(name, l)| format!(" {:>w$}", cell(l), w = name.len().max(5)));
        format!("{first:width$}{}\n", cells.collect::<String>())
    };
    let header = columns.iter().map(|(name, _)| format!(" {name:>w$}", w = name.len().max(5))).collect::<String>();
    let mut out = format!("{:width$}{header}\n", "");
    out += &row("instructions", &|l| l.instructions().to_string());
    for (op, _) in ops {
        out += &row(op, &|l| l.count(op).to_string());
    }
    out
}
