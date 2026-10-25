//! One run captures a device's whole model input: its layer-0 facts
//! (`svod_tk::ops::config::Facts` — identity, limits, caches, clocks, queried
//! of the driver or the KFD topology), its cache instances, the costs its
//! store measured (READ-ONLY — a store without them prints null, nothing is
//! ever probed), and the compile-probe oracle's verdicts (assembled and
//! recorded where missing). An offline target (`--card`/`--arch`) prints the
//! tables' own values, marked `"table"`.
//!
//! ```text
//! cargo run -p svod-tk --release --example tk_facts [--card NAME | --arch sm_90|gfx1201 [--sms N] [--xcds N]] [--store DIR] > facts.json
//! ```

use std::path::PathBuf;

use svod_dtype::{AmdArch, CudaArch, GpuArch};
use svod_tk::atoms::{CARDS, Target};
use svod_tk::ops::config::Facts;
use svod_tk::probe;
use svod_tk::tune::TuneStore;

type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let at = |name: &str| args.iter().position(|a| a == name);
    let opt = |name: &str| at(name).and_then(|i| args.get(i + 1)).map(String::as_str);
    let num = |name: &str| -> Result<Option<usize>> {
        opt(name).map(|v| v.parse()).transpose().map_err(|e| format!("--{name}: {e}").into())
    };
    let store = opt("--store").map(PathBuf::from);
    let mut out = String::from("{\n");
    match target(opt("--card"), opt("--arch"), num("--sms")?, num("--xcds")?) {
        // An offline target: the tables' own values, marked as such.
        Some(target) => {
            facts_section(&mut out, &Facts::of(&target));
            costs_section(&mut out, &store, &target);
            oracle_section(&mut out, &target, true);
        }
        None => {
            // The default device, whatever opens: everything queried of it.
            let device = svod_dtype::default_device::default_device();
            let target =
                Target::for_device(&device).ok_or("the default device reports no target; pass --card or --arch")?;
            let facts = Facts::for_device(&device).ok_or("the default device reports no facts (is a GPU selected?)")?;
            facts_section(&mut out, &facts);
            costs_section(&mut out, &store, &target);
            oracle_section(&mut out, &target, false);
        }
    }
    out.push_str("}\n");
    print!("{out}");
    Ok(())
}

/// The target of `--card`, else `--arch` with the SMs and dies of its first
/// listed card; `None` without either flag: the default device's own.
fn target(card: Option<&str>, arch: Option<&str>, sms: Option<usize>, xcds: Option<usize>) -> Option<Target> {
    if let Some(card) = card {
        return Some(Target::card(card).unwrap_or_else(|| panic!("{card}: not one of {:?}", CARDS.map(|c| c.0))));
    }
    let arch = arch?;
    let arch = match AmdArch::parse(arch) {
        Some(a) => GpuArch::Amd(a),
        None => GpuArch::Cuda(arch.trim_end_matches('a').parse::<CudaArch>().expect("an arch like sm_90 or gfx1201")),
    };
    let listed = CARDS.iter().find(|c| c.1 == arch);
    let sms = sms.map(|s| s as u32).or(listed.map(|c| c.2)).expect("--sms: no listed card of the arch");
    Some(Target::offline(arch, sms, xcds.map(|x| x as u32).or(listed.map(|c| c.3)).unwrap_or(1)))
}

fn facts_section(out: &mut String, facts: &Facts) {
    let quoted = |s: &str| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""));
    let mut line = |name: &str, value: String| out.push_str(&format!("  \"{name}\": {value},\n"));
    let number = |v: Option<u64>| v.map(|v| v.to_string()).unwrap_or("null".into());
    line("device", quoted(&facts.name));
    line("arch", quoted(&facts.arch.target_name().to_string()));
    line("table", facts.table.to_string());
    line("sms", facts.sms.to_string());
    line("wave", facts.wave.to_string());
    line("smem_per_block", number(facts.smem_per_block));
    line("smem_per_block_optin", number(facts.smem_per_block_optin));
    line("smem_reserved", number(facts.smem_reserved));
    line("smem_per_sm", number(facts.smem_per_sm));
    line("registers_per_block", number(facts.registers_per_block.map(u64::from)));
    line("registers_per_sm", number(facts.registers_per_sm.map(u64::from)));
    line("threads_per_sm", number(facts.threads_per_sm.map(u64::from)));
    line("blocks_per_sm", number(facts.blocks_per_sm.map(u64::from)));
    line("l2_bytes", number(facts.l2_bytes));
    line("persisting_l2_bytes", number(facts.persisting_l2_bytes));
    line("core_clock_khz", number(facts.core_clock_khz));
    line("memory_clock_khz", number(facts.memory_clock_khz));
    line("engine_clk_fcompute", number(facts.engine_clk_fcompute));
    line("bus_bits", number(facts.bus_bits.map(u64::from)));
    line("copy_engines", number(facts.copy_engines.map(u64::from)));
    line("xcds", facts.xcds.to_string());
    line("memory_bytes", number(facts.memory_bytes));
    out.push_str("  \"caches\": [");
    for (i, cache) in facts.caches.iter().enumerate() {
        out.push_str(&format!(
            "{}{{\"level\": {}, \"size_bytes\": {}, \"line_bytes\": {}, \"shared_by\": {}}}",
            if i == 0 { "" } else { ", " },
            cache.level,
            cache.size_bytes,
            cache.line_bytes,
            cache.shared_by
        ));
    }
    out.push_str("],\n");
}

/// The costs a store holds for the target's device, read never written: every
/// field null where no line of it measured (nothing is probed here).
fn costs_section(out: &mut String, store: &Option<PathBuf>, target: &Target) {
    out.push_str("  \"costs\": {");
    let dir = store.clone().or_else(|| TuneStore::global().root().map(PathBuf::from)).filter(|dir| dir.is_dir());
    let measured = dir.as_deref().and_then(|dir| probe::stored(dir, target));
    let Some(measured) = measured else {
        out.push_str("},\n");
        return;
    };
    let mut costs = measured.costs;
    for (name, value) in costs.fields() {
        let v =
            if measured.unmeasured.contains(&name) || *value == 0.0 { "null".to_string() } else { format!("{value}") };
        out.push_str(&format!("\n    \"{name}\": {v},"));
    }
    let mut l2 = |name: &str, v: f64| {
        let value = if v == 0.0 { "null".to_string() } else { v.to_string() };
        out.push_str(&format!("\n    \"{name}\": {value},"));
    };
    l2("l2_byte", costs.l2_byte);
    let bytes = if costs.l2_bytes == 0.0 { "null".to_string() } else { costs.l2_bytes.to_string() };
    out.push_str(&format!("\n    \"l2_bytes\": {bytes},"));
    let ladder: Vec<String> = costs
        .ladder
        .iter()
        .filter(|&&(f, r)| f > 0 && r > 0.0)
        .map(|&(f, r)| format!("{{\"footprint\": {f}, \"ns_per_byte\": {r}}}"))
        .collect();
    let ladder = if ladder.is_empty() { "null".to_string() } else { ladder.join(", ") };
    out.push_str(&format!("\n    \"ladder\": [{ladder}]\n  }},\n"));
}

/// The oracle's verdicts: the recorded ones for a device (assembling and
/// storing any that is missing), the tables' own — marked — for an offline
/// target where no toolchain of the card answered.
fn oracle_section(out: &mut String, target: &Target, table: bool) {
    let verdicts = if table {
        svod_tk::oracle::tables(target).into_iter().map(|(name, v)| (name, Some(v))).collect::<Vec<_>>()
    } else {
        svod_tk::oracle::record(target)
    };
    let entries = verdicts
        .into_iter()
        .map(|(name, verdict)| {
            let v = verdict.map(|v| v.to_string()).unwrap_or_else(|| "null".into());
            format!("    \"{name}\": {v}")
        })
        .chain(std::iter::once(format!("    \"table\": {table}")));
    out.push_str(&format!("  \"oracle\": {{\n{}\n  }}\n", entries.collect::<Vec<_>>().join(",\n")));
}
