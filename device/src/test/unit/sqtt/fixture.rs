//! The committed captures, decoded and stitched as the runtime does.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::sqtt::stitch::{Stitcher, objdump_lines};
use crate::sqtt::wave::Trace;
use crate::sqtt::{Layout, decode, wave};

/// One fixture's disassembly and the stitched traces of its instruction-traced SEs.
pub struct Stitched {
    /// The code's load address: `code`'s PCs are ELF addresses plus it.
    pub base: u64,
    /// `(pc, llvm-objdump line)`, the line's `// address: words` comment kept.
    pub code: BTreeMap<u64, String>,
    pub layout: Layout,
    /// The instruction-traced SEs, `traces` in their order.
    pub ses: Vec<u32>,
    pub traces: Vec<Trace>,
}

impl Stitched {
    /// The instruction text at `pc`, without its comment.
    pub fn text(&self, pc: u64) -> Option<&str> {
        self.code.get(&pc).map(|line| line.split("//").next().unwrap_or_default().trim())
    }

    pub fn lines(&self) -> Vec<(u64, &str)> {
        self.code.iter().map(|(&pc, line)| (pc, line.as_str())).collect()
    }

    pub fn waves(&self) -> Vec<wave::Wave> {
        self.traces.iter().flat_map(|t| t.waves.iter().cloned()).collect()
    }
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/test/fixtures/sqtt")
}

fn read(dir: &Path, file: &str) -> Vec<u8> {
    std::fs::read(dir.join(file)).unwrap_or_else(|e| panic!("{}: {e}", dir.join(file).display()))
}

/// `name`: `amd/<case>` (AMD's controls), `gfx1100|gfx1200/<example>` (tinygrad's) or `svod/<capture>`.
pub fn stitched(name: &str) -> Stitched {
    let dir = root().join(name);
    let text = |file: &str| String::from_utf8(read(&dir, file)).unwrap();
    let json = |file: &str| serde_json::from_str::<Value>(&text(file)).unwrap();
    let se = |v: &Value| v["se"].as_u64().unwrap() as u32;
    let (base, dump, streams): (u64, String, Vec<(u32, Vec<u8>)>) = match name.split_once('/').unwrap().0 {
        "amd" => {
            let base = u64::from_str_radix(text("base.txt").trim().trim_start_matches("0x"), 16).unwrap();
            (base, text("code_object.s"), vec![(0, read(&dir, "se0.att"))])
        }
        "svod" => {
            let meta = json("meta.json");
            let ses = meta["ses"].as_array().unwrap().iter().filter(|s| s["itrace"].as_bool().unwrap());
            let streams = ses.map(|s| (se(s), read(&dir, &format!("se{}.att", se(s))))).collect();
            (meta["load_base"].as_u64().unwrap(), text("kernel.s"), streams)
        }
        _ => {
            let golden = json("golden.json");
            let events = golden["events"].as_array().unwrap().iter().filter(|e| e["itrace"].as_bool().unwrap());
            let streams = events.map(|e| (se(e), read(&dir, &format!("trace{}.bin", e["index"])))).collect();
            (golden["base"].as_u64().unwrap(), text("kernel.s"), streams)
        }
    };
    let code: Vec<(u64, &str)> = objdump_lines(&dump).into_iter().map(|(pc, line)| (pc + base, line)).collect();
    let mut layout = Layout::Rdna3;
    let ses = streams.iter().map(|(se, _)| *se).collect();
    let traces = streams
        .iter()
        .map(|(_, bytes)| {
            let mut trace = wave::analyze(decode(bytes));
            layout = if trace.header.version == 4 { Layout::Rdna4 } else { Layout::Rdna3 };
            let mut stitcher = Stitcher::new(&code, layout);
            trace.waves.iter_mut().for_each(|w| _ = stitcher.stitch(w));
            trace
        })
        .collect();
    let code = code.into_iter().map(|(pc, line)| (pc, line.to_string())).collect();
    Stitched { base, code, layout, ses, traces }
}
