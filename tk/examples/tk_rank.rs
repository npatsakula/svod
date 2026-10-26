//! The rank harness over a tune store, offline: replays each `choice|` line
//! the store holds — its own candidates, the codes the line carries beside
//! its indices, priced under the costs the same store measured — and
//! prints, per family and device, the Spearman ρ of the model's predicted
//! order against the device's measured one, the fraction of lines whose
//! predicted best is the measured winner, how many lines replayed from
//! their own candidates (no derivation matching their digest: a ranking
//! that changed since they were written), and the lines no hypothesis
//! admits a candidate of. The baseline numbers the config rework's later
//! phases must not regress; see `svod_tk::rank`.
//!
//! ```text
//! cargo run -p svod-tk --release --example tk_rank -- [--store DIR] [--device NAME] [--op gemm,attention] [--check] [--fit]
//! ```
//!
//! `--store` defaults to the global tune store; `--device` and `--op` filter
//! by substring. `--check` adds the determinism pass: which choice lines of
//! each family were derived (the digest in their key asserted), which
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
    let ablate = at("--ablate").and_then(|i| args.get(i + 1)).cloned();
    let claims = at("--claims").is_some();
    let explain = at("--explain").and_then(|i| args.get(i + 1)).cloned();
    let fit = at("--fit").is_some();
    let store = opt.map(PathBuf::from).or_else(|| TuneStore::global().root().map(PathBuf::from));
    let Some(store) = store.filter(|s| s.is_dir()) else {
        return Err("no tune store: pass --store DIR (or set SVOD_TK_TUNE_DIR)".into());
    };
    for device in rank::replay(&store, device.as_deref(), ops.as_deref()) {
        println!("{} ({} of its cost fields unmeasured)", device.name, device.unmeasured.len());
        println!(
            "  {:<12} {:>6} {:>11} {:>7} {:>6} {:>7} {:>8} {:>7}",
            "family", "lines", "candidates", "rho", "top1", "stored", "skipped", "short"
        );
        let planner = &device.planner;
        for family in &device.families {
            println!(
                "  {:<12} {:>6} {:>11} {:>7} {:>6} {:>7} {:>8} {:>7}",
                family.name,
                family.lines.len(),
                family.candidates(),
                family.rho(planner).map_or("-".into(), |r| format!("{r:.3}")),
                family.top1(planner).map_or("-".into(), |t| format!("{t:.2}")),
                family.stored,
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
        if let Some(family) = explain.as_deref() {
            for wanted in &device.families {
                if !wanted.name.contains(family) {
                    continue;
                }
                let base = wanted.rho(planner).map_or("-".into(), |r| format!("{r:.3}"));
                println!("  {} at its measured costs: rho {base}", wanted.name);
                for (field, rho) in rank::explained(&device, wanted.name, 2.0) {
                    let delta = rho.zip(wanted.rho(planner)).map_or("-".into(), |(r, b)| format!("{:+.3}", r - b));
                    println!("    {:<12} x2: rho {} ({delta})", field, rho.map_or("-".into(), |r| format!("{r:.3}")));
                }
            }
        }
        if let Some("nearest") = ablate.as_deref() {
            for ablated in rank::ablation(&device) {
                let base = ablated.base.map_or("-".into(), |r| format!("{r:.3}"));
                println!("  {} with every cluster at full weight: rho {base}", ablated.family);
                for (cluster, rho) in &ablated.clusters {
                    let delta = rho.zip(ablated.base).map_or("-".into(), |(r, b)| format!("{:+.3}", r - b));
                    println!(
                        "    {:<6} zeroed: rho {} ({delta})",
                        cluster,
                        rho.map_or("-".into(), |r| format!("{r:.3}"))
                    );
                }
            }
        }
        if fit {
            for f in rank::fitted(&device) {
                let (held, prior) = (f.held_out, f.prior);
                let show = |r: Option<f64>| r.map_or("-".into(), |r| format!("{r:.3}"));
                let folds: Vec<String> = f.multipliers.iter().map(|m| format!("{m:.2}")).collect();
                println!(
                    "  {:<12} fit x{:.2} overall; held-out rho {} against the prior's {} (folds x: {})",
                    f.family,
                    f.overall,
                    show(held),
                    show(prior),
                    folds.join(" "),
                );
            }
        }
        if claims {
            for diffused in rank::claims(&svod_tk::atoms::Target::card("rtx3060").expect("a card")) {
                println!("  {} ({} layout regs):", diffused.name, diffused.regs);
                for (what, predicted, program, holds) in &diffused.rows {
                    println!(
                        "    {:<16} predicted {:>10.1}   program {:>10.1}   {}",
                        what,
                        predicted,
                        program,
                        if *holds { "holds" } else { "outside ±15%" }
                    );
                }
            }
        }
        println!();
    }
    Ok(())
}
