//! Digests of the LLVM text every kernel of the refactor sweep renders to,
//! per target, for proving a refactor byte-identical in what it emits:
//! run at the base and at the change, then `--diff` the two.
//!
//! ```text
//! cargo run -p svod-tk3 --release --example tk3_digest -- --out DIR [--targets sm_86,gfx942] [--full]
//! cargo run -p svod-tk3 --release --example tk3_digest -- --diff DIR_A DIR_B
//! ```
//!
//! The kernels are `svod_tk3::sweep::digested` (every family, the planner's
//! shortlists, the attention, convolution, norm, heads and probe variants)
//! under the declared costs, on the sweep's ten targets unless `--targets`
//! names some. Each target gets `DIR/<target>/digests.tsv` (an FNV-1a digest
//! and the kernel's name per line, a kernel that fails to lower or render
//! digested by its error) and `rankings.txt` (the planner's rankings, which
//! move the tuner's choices without moving any kernel's text); `--full`
//! also writes each kernel's text as `DIR/<target>/<line>.ll`. `--diff`
//! names the kernels whose digest changed, appeared or went per target, and
//! whether the rankings changed; it exits 1 when anything did.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use svod_tk3::sweep::{self, Diff};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const INDEX: &str = "digests.tsv";
const RANKINGS: &str = "rankings.txt";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let at = |name: &str| args.iter().position(|a| a == name);
    if let Some(i) = at("--diff") {
        let (a, b) = (args.get(i + 1).ok_or("--diff DIR_A DIR_B")?, args.get(i + 2).ok_or("--diff DIR_A DIR_B")?);
        let same = compare(Path::new(a), Path::new(b))?;
        std::process::exit(if same { 0 } else { 1 });
    }
    let out = PathBuf::from(at("--out").and_then(|i| args.get(i + 1)).ok_or("--out DIR or --diff DIR_A DIR_B")?);
    let wanted: Option<Vec<&str>> = at("--targets").and_then(|i| args.get(i + 1)).map(|t| t.split(',').collect());
    let full = at("--full").is_some();
    let targets = sweep::targets();
    if let Some(unknown) = wanted.iter().flatten().find(|w| !targets.iter().any(|(n, _)| n == *w)) {
        return Err(format!("{unknown}: not one of {:?}", targets.iter().map(|t| t.0).collect::<Vec<_>>()).into());
    }
    for (name, target) in targets.into_iter().filter(|(n, _)| wanted.as_ref().is_none_or(|w| w.contains(n))) {
        let dir = out.join(name);
        std::fs::create_dir_all(&dir)?;
        let start = std::time::Instant::now();
        let launches = sweep::digested(&target);
        // A failure is digested by its message; the panic hook would print each.
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let texts: Vec<(String, String)> = launches
            .into_par_iter()
            .map(|(kernel, prog, lowering)| {
                let text = match catch_unwind(AssertUnwindSafe(|| sweep::try_render(prog, &lowering))) {
                    Ok(Ok(text)) => text,
                    Ok(Err(e)) => format!("FAILED: {e}"),
                    Err(panic) => {
                        let why = panic.downcast_ref::<String>().cloned();
                        format!(
                            "FAILED: {}",
                            why.or(panic.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
                        )
                    }
                };
                (kernel, text)
            })
            .collect();
        std::panic::set_hook(hook);
        let mut index = String::new();
        for (i, (kernel, text)) in texts.iter().enumerate() {
            index += &format!("{:016x}\t{kernel}\n", sweep::digest(text.as_bytes()));
            if full {
                std::fs::write(dir.join(format!("{:04}.ll", i + 1)), text)?;
            }
        }
        std::fs::write(dir.join(INDEX), index)?;
        std::fs::write(dir.join(RANKINGS), sweep::rankings(&target))?;
        let failed = texts.iter().filter(|(_, t)| t.starts_with("FAILED: ")).count();
        println!("{name}: {} kernels, {failed} failed, {:.1} s", texts.len(), start.elapsed().as_secs_f64());
    }
    Ok(())
}

/// The digests of one target's index, in its order.
fn index(dir: &Path) -> Result<Vec<(String, u64)>> {
    let text = std::fs::read_to_string(dir.join(INDEX))?;
    text.lines()
        .map(|l| {
            let (digest, name) = l.split_once('\t').ok_or_else(|| format!("{}: {l}", dir.display()))?;
            Ok((name.to_string(), u64::from_str_radix(digest, 16)?))
        })
        .collect()
}

/// Prints what differs between the two runs per target; whether nothing does.
fn compare(a: &Path, b: &Path) -> Result<bool> {
    let targets = |dir: &Path| -> Result<Vec<String>> {
        let mut names: Vec<String> = std::fs::read_dir(dir)?
            .flatten()
            .filter(|e| e.path().join(INDEX).is_file())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        Ok(names)
    };
    let (in_a, in_b) = (targets(a)?, targets(b)?);
    let mut same = in_a == in_b;
    if !same {
        println!("targets differ: {in_a:?} against {in_b:?}");
    }
    for target in in_a.iter().filter(|t| in_b.contains(t)) {
        let (da, db) = (a.join(target), b.join(target));
        let (ia, ib) = (index(&da)?, index(&db)?);
        let d: Diff = sweep::diff(&ia, &ib);
        let rankings = std::fs::read(da.join(RANKINGS)).ok() == std::fs::read(db.join(RANKINGS)).ok();
        same &= d.is_empty() && rankings;
        println!(
            "{target}: {} kernels, {} changed, {} removed, {} added; rankings {}",
            ia.len(),
            d.changed.len(),
            d.removed.len(),
            d.added.len(),
            if rankings { "same" } else { "changed" }
        );
        for (what, names) in [("changed", &d.changed), ("removed", &d.removed), ("added", &d.added)] {
            names.iter().for_each(|n| println!("  {what}: {n}"));
        }
    }
    println!("{}", if same { "no kernel and no ranking changed" } else { "differences found" });
    Ok(same)
}
