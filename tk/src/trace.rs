//! Where a kernel's time goes on NVIDIA, by tile statement. The host has no
//! thread trace there and the counters are the administrator's, so a traced
//! lowering ([`Lowering::trace`]) stamps the SM's cycle counter instead: lane
//! 0 of each warp of every [`EVERY`]th block adds the cycles since its last
//! stamp to the word of the statement that just ended. A copy, a sync and a
//! product that waits take a stamp each; the register statements before one
//! share a stamp, charged to the last of them. The stamps cost time
//! themselves (a traced kernel is timed beside the bare one), and a
//! statement's time is not split further.

use std::fmt;

use svod_dtype::ScalarDType;

use crate::ir::{Block, Laid, Param, ParamKind, Phase, Program, Stmt, Sync, TileOp};
use crate::lower::{self, Lowering, listing::listing};

/// A block is sampled where its grid indices sum to a multiple of this.
pub const EVERY: i64 = 16;
/// A word holds its cycles below this and its stamps above, a launch's
/// alone: a run's words are read and zeroed before the next.
pub const COUNT: i64 = 1 << 40;

/// A launch's `words` added to `sums` as cycles and stamps.
pub fn tally(sums: &mut [(f64, f64)], words: &[i64]) {
    for (sum, &w) in sums.iter_mut().zip(words) {
        assert!(w >= 0, "a trace word past 2^63: over 2^23 stamps a statement and warp in one launch");
        *sum = (sum.0 + (w % COUNT) as f64, sum.1 + (w / COUNT) as f64);
    }
}

/// The rows of a trace of `lines` statements: the unattributed tail, the
/// statements by line, the sampled blocks' nanoseconds, then [`BARE`] stamps
/// with nothing between them (what a stamp costs).
pub const fn rows(lines: usize) -> usize {
    lines + 3
}

pub const BARE: usize = 4;

/// A program with its trace parameter (the last, zeroed by the launch), the
/// lowering that stamps it and the listing its rows are the lines of.
pub struct Traced {
    pub prog: Program,
    pub lowering: Lowering,
    pub listing: String,
}

pub fn traced(mut prog: Program, lowering: &Lowering) -> lower::Result<Traced> {
    let tile = lower::standalone(prog.clone(), lowering)?.tile;
    let mut shared = vec![];
    shares(&tile.body, &mut shared);
    // A shared stamp's row reads `… statement`: the register statements up to it.
    let mark = |(line, shared): (&str, bool)| {
        let text = line.trim_start();
        format!("{}{}{text}\n", &line[..line.len() - text.len()], if shared { "… " } else { "" })
    };
    let listing: String = listing(&tile).lines().zip(shared).map(mark).collect();
    let elems = rows(listing.lines().count()) * prog.warps as usize;
    prog.params.push(Param { name: "trace".into(), dtype: ScalarDType::Int64, kind: ParamKind::Accum, elems });
    Ok(Traced { prog, lowering: Lowering { trace: true, ..lowering.clone() }, listing })
}

/// Whether `stmt` takes no stamp of its own in a block whose products are
/// `waited` for by a statement of their own: a register statement does not,
/// nor does a product left in flight (its chain's wait holds its time).
pub(crate) fn shared<P: Phase>(stmt: &Stmt<P>, waited: bool) -> bool {
    matches!(stmt, Stmt::Let { op, .. } if waited || !matches!(op, TileOp::Mma { .. }))
}

/// [`shared`] of every statement under `block`, in listing order.
fn shares(block: &Block<Laid>, out: &mut Vec<bool>) {
    let waited = block.0.iter().any(|s| matches!(s, Stmt::Sync(Sync::WaitProducts { .. })));
    for stmt in &block.0 {
        out.push(shared(stmt, waited));
        stmt.blocks().for_each(|b| shares(b, out));
    }
}

/// A statement's time in a sampled block, over the block's warps.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    /// The statement as listed, indented by its depth.
    pub text: String,
    /// Nanoseconds a block: the warps' mean, least and most.
    pub ns: f64,
    pub least: f64,
    pub most: f64,
    /// Stamps a block and warp.
    pub hits: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    /// Sampled blocks.
    pub blocks: u64,
    /// A sampled block's nanoseconds, first stamp to last.
    pub block_ns: f64,
    /// The time after the last statement's stamp.
    pub tail_ns: f64,
    /// What a stamp costs: each row holds this much a stamp of its own.
    pub stamp_ns: f64,
    pub rows: Vec<Row>,
}

impl Report {
    /// The trace parameter's words [tallied](tally) over launches of a
    /// program of `warps` warps listed as `listing`; `None` where no block
    /// was sampled.
    pub fn decode(listing: &str, warps: usize, words: &[(f64, f64)]) -> Option<Self> {
        let split = |row: usize| words[row * warps..][..warps].iter().copied();
        let lines: Vec<&str> = listing.lines().collect();
        assert_eq!(words.len(), rows(lines.len()) * warps, "a trace of another listing");
        let blocks = split(0).next()?.1;
        let bare = lines.len() + 2;
        let cycles: f64 = (0..=lines.len()).chain([bare]).flat_map(split).map(|(c, _)| c).sum();
        let ns: f64 = split(lines.len() + 1).map(|(ns, _)| ns).sum();
        if blocks == 0.0 || cycles == 0.0 {
            return None;
        }
        // A warp's cycles in a row as nanoseconds a block.
        let scale = ns / cycles / blocks;
        let row = |row: usize, text: &str| {
            let of: Vec<f64> = split(row).map(|(c, _)| c * scale).collect();
            Row {
                text: text.to_string(),
                ns: of.iter().sum::<f64>() / warps as f64,
                least: of.iter().copied().fold(f64::INFINITY, f64::min),
                most: of.iter().copied().fold(0.0, f64::max),
                hits: split(row).map(|(_, h)| h).sum::<f64>() / blocks / warps as f64,
            }
        };
        let rows = lines.iter().enumerate().map(|(i, text)| row(i + 1, text)).filter(|r| r.hits > 0.0).collect();
        let (tail_ns, stamp_ns) = (row(0, "").ns, row(bare, "").ns / BARE as f64);
        Some(Self { blocks: blocks as u64, block_ns: ns / blocks / warps as f64, tail_ns, stamp_ns, rows })
    }

    /// A report as [`fmt::Display`] printed it.
    pub fn parse(text: &str) -> Option<Self> {
        let mut lines = text.lines().filter_map(|l| l.strip_prefix("trace|"));
        let nums = |l: &str, n: usize| {
            l.splitn(n + 1, '|').take(n).map(|x| x.trim().parse().ok()).collect::<Option<Vec<f64>>>()
        };
        let head = nums(lines.next()?, 4)?;
        let rows = lines.map(|l| {
            let n = nums(l, 6)?;
            Some(Row { ns: n[0], hits: n[3], least: n[4], most: n[5], text: l.splitn(7, '|').last()?.to_string() })
        });
        let rows = rows.collect::<Option<_>>()?;
        Some(Self { blocks: head[0] as u64, block_ns: head[1], tail_ns: head[2], stamp_ns: head[3], rows })
    }

    /// A row's time less what its stamps cost.
    pub fn net(&self, row: &Row) -> f64 {
        (row.ns - row.hits * self.stamp_ns).max(0.0)
    }

    /// Each statement both reports hold (by its text, the `n`th of that
    /// text) with its time in each, then those only one holds.
    pub fn compare(&self, other: &Self) -> String {
        let keyed = |r: &Self| {
            let mut seen = std::collections::HashMap::new();
            let mut key = |row: &Row| {
                let nth = seen.entry(row.text.trim().to_string()).or_insert(0usize);
                *nth += 1;
                (row.text.trim().to_string(), *nth)
            };
            r.rows.iter().map(|row| (key(row), row.clone())).collect::<Vec<_>>()
        };
        let (a, b) = (keyed(self), keyed(other));
        let mut out = format!("compare|block ns {:.0} → {:.0}\n", self.block_ns, other.block_ns);
        for (key, row) in &a {
            let (from, to) = (self.net(row), b.iter().find(|(k, _)| k == key).map(|(_, r)| other.net(r)));
            let delta = to.map_or("gone".to_string(), |to| format!("{:+.0}", to - from));
            out += &format!("compare|{from:9.0}|{:9.0}|{delta:>8}|{}\n", to.unwrap_or(0.0), row.text);
        }
        for (_, row) in b.iter().filter(|(k, _)| !a.iter().any(|(ka, _)| ka == k)) {
            out += &format!("compare|{:9.0}|{:9.0}|{:>8}|{}\n", 0.0, other.net(row), "new", row.text);
        }
        out
    }
}

impl fmt::Display for Report {
    /// `trace|blocks|ns a block|tail ns|ns a stamp`, then a statement a line:
    /// `trace|ns a block|less its stamps|that as % of the rows'|stamps|least warp|most warp|statement`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "trace|{}|{:.0}|{:.0}|{:.1}", self.blocks, self.block_ns, self.tail_ns, self.stamp_ns)?;
        let all: f64 = self.rows.iter().map(|r| self.net(r)).sum();
        for r in &self.rows {
            let (net, hits) = (self.net(r), r.hits);
            let share = 100.0 * net / all;
            writeln!(
                f,
                "trace|{:9.0}|{net:9.0}|{share:5.1}|{hits:7.1}|{:9.0}|{:9.0}|{}",
                r.ns, r.least, r.most, r.text
            )?;
        }
        Ok(())
    }
}
