//! The rank harness over a tune store, offline: replays each `choice|` line
//! the store holds — re-deriving the candidates its key digests, under the
//! costs the same store measured — and prints, per family and device, the
//! Spearman ρ of the model's predicted order against the device's measured
//! one, the fraction of lines whose predicted best is the measured winner,
//! and the lines no hypothesis could reproduce. The baseline numbers the
//! config rework's later phases must not regress; see `svod_tk::rank`.
//!
//! ```text
//! cargo run -p svod-tk --release --example tk_rank -- [--store DIR] [--device NAME] [--op gemm,attention] [--check]
//! ```
//!
//! `--store` defaults to the global tune store; `--device` and `--op` filter
//! by substring. `--check` adds the determinism pass: which choice lines of
//! each family were reproduced (the digest in their key asserted), which
//! skipped, and the shapes it happened on.

use std::path::PathBuf;

use svod_tk::rank;
use svod_tk::tune::TuneStore;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let at = |name: &str| args.iter().position(|a| a == name);
    let opt = at("--store").and_then(|i| args.get(i + 1)).cloned();
    let device = at("--device").and_then(|i| args.get(i + 1)).cloned();
    let ops = at("--op").and_then(|i| args.get(i + 1)).map(|o| o.split(',').collect::<Vec<_>>());
    let check = at("--check").is_some();
    let store = opt.map(PathBuf::from).or_else(|| TuneStore::global().root().map(PathBuf::from));
    let Some(store) = store.filter(|s| s.is_dir()) else {
        return Err("no tune store: pass --store DIR (or set SVOD_TK_TUNE_DIR)".into());
    };
    for device in rank::replay(&store, device.as_deref(), ops.as_deref()) {
        println!("{} ({} of its cost fields unmeasured)", device.name, device.unmeasured.len());
        println!(
            "  {:<12} {:>6} {:>11} {:>7} {:>6} {:>8} {:>7}",
            "family", "lines", "candidates", "rho", "top1", "skipped", "short"
        );
        let planner = &device.planner;
        for family in &device.families {
            println!(
                "  {:<12} {:>6} {:>11} {:>7} {:>6} {:>8} {:>7}",
                family.name,
                family.lines.len(),
                family.candidates(),
                family.rho(planner).map_or("-".into(), |r| format!("{r:.3}")),
                family.top1(planner).map_or("-".into(), |t| format!("{t:.2}")),
                family.skipped,
                family.short,
            );
        }
        if check {
            for family in &device.families {
                println!(
                    "  {}: {} lines replayed, {} skipped, {} too short to correlate",
                    family.name,
                    family.lines.len(),
                    family.skipped,
                    family.short
                );
            }
            for (group, lines) in device.skipped_groups.iter().take(200) {
                println!("    skipped {lines:3} of {group}");
            }
        }
        println!();
    }
    Ok(())
}
