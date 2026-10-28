//! KFD topology parser: enumerates AMD GPU nodes via sysfs.
//!
//! Reads `/sys/devices/virtual/kfd/kfd/topology/nodes/N/properties` for each
//! numeric subdir (and its `mem_banks/*/properties` memory sizes and
//! `caches/*/properties` cache instances), returning a [`Vec<AmdNode>`].
//! Containers and hosts without `/dev/kfd` get an empty result (never panic).
//!
//! Source format (one key/value per line, whitespace-separated; a cache's
//! `sibling_map` is a comma-separated bitmap):
//! ```text
//! gpu_id 4660
//! cpu_cores_count 0
//! simd_count 4
//! gfx_target_version 110000
//! drm_render_minor 128
//! …
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// A single KFD topology entry (CPU or GPU node).
#[derive(Debug, Clone)]
pub struct AmdNode {
    /// KFD node index (0, 1, 2…), matches the sysfs directory name.
    pub node_id: u32,
    /// KFD's `gpu_id`; 0 for CPU nodes (which we filter out at enumerate-time
    /// for the GPU-only enumerator).
    pub gpu_id: u32,
    /// Minor number for `/dev/dri/renderD{N}`. 0 for CPU nodes.
    pub drm_render_minor: u32,
    /// Raw `gfx_target_version` integer (e.g. `110000` for gfx1100).
    pub gfx_target_version: u32,
    pub simd_count: u32,
    pub array_count: u32,
    /// Shader arrays per shader-engine. Used to derive the SE count for scratch
    /// wave budgeting; defaults to 1 when KFD omits it.
    pub simd_arrays_per_engine: u32,
    pub simd_per_cu: u32,
    pub max_waves_per_simd: u32,
    pub lds_size_in_kb: u32,
    pub wave_front_size: u32,
    /// Number of XCC compute engines. Defaults to 1 when the field is absent.
    pub num_xcc: u32,
    /// Maximum number of independent compute queues KFD will give us on this
    /// device. Defaults to 8 when the field is absent.
    pub num_cp_queues: u32,
    /// Maximum scratch slots per CU. Used to size the default scratch buffer
    /// at device open. KFD's sysfs property `max_slots_scratch_cu`; falls back
    /// to `simd_per_cu * max_waves_per_simd` (matches RDNA scratch slot
    /// budgeting) when absent.
    pub max_slots_scratch_cu: u32,
    /// Bytes of the node's frame-buffer memory banks (`mem_banks/*/properties`
    /// of heap type 1 or 2, public or private): its VRAM, or an APU's
    /// carve-out. 0 when KFD lists none.
    pub memory_bytes: u64,
    /// Peak shader engine clock, KFD's `max_engine_clk_fcompute` as the
    /// kernel prints it (its unit has flipped between kernels: a diagnostic,
    /// never derived from). 0 when the field is absent.
    pub max_engine_clk_fcompute: u64,
    /// Compute units per SIMD array (`cu_per_simd_array`); 0 when absent.
    pub cu_per_simd_array: u32,
    /// SDMA copy engines (`num_sdma_engines`); 0 when absent.
    pub num_sdma_engines: u32,
    /// SDMA engines on XGMI links (`num_sdma_xgmi_engines`); 0 when absent.
    pub num_sdma_xgmi_engines: u32,
    /// The XGMI hive the node belongs to (`hive_id`); 0 alone.
    pub hive_id: u64,
    /// PCI device id (`device_id`); 0 when absent.
    pub device_id: u32,
    /// PCI vendor id (`vendor_id`); 0 when absent.
    pub vendor_id: u32,
    /// The node's cache instances (`caches/*/properties`), empty where KFD
    /// lists none: per-CU/WGP L1s and LDS, per-SE or per-XCD L2 slices, the
    /// memory-side Infinity Cache (LLC) — with which processors share each.
    pub caches: Vec<AmdCache>,
}

/// One cache instance of a KFD node's `caches/` topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmdCache {
    /// The processor the instance hangs off (`processor_id_low`).
    pub processor_id: u32,
    /// 1 for the L1s and LDS, 2 for L2, 3 for the Infinity Cache (LLC).
    pub level: u32,
    /// Bytes (sysfs prints KB).
    pub size_bytes: u64,
    pub line_bytes: u32,
    pub cache_lines_per_tag: u32,
    /// Associativity, 0 where KFD reports none.
    pub association: u32,
    /// Load latency, 0 where KFD reports none.
    pub latency: u32,
    /// The `type` bitmask (bit 0 data, bit 1 instruction, …).
    pub kind: u32,
    /// Which processors share the instance, KFD's `sibling_map` bitmap (one
    /// 64-bit word per comma-separated entry).
    pub sibling_map: Vec<u64>,
}

impl AmdCache {
    /// How many processors the instance is shared by: the set bits of its
    /// sibling map.
    pub fn shared_by(&self) -> u32 {
        self.sibling_map.iter().map(|w| w.count_ones()).sum()
    }
}

/// Sysfs root for KFD topology. Public so tests can override via env.
fn topology_root() -> PathBuf {
    if let Ok(override_path) = std::env::var("SVOD_KFD_TOPOLOGY") {
        return PathBuf::from(override_path);
    }
    PathBuf::from("/sys/devices/virtual/kfd/kfd/topology/nodes")
}

/// Enumerate all KFD GPU nodes on this host.
///
/// Returns `Ok(empty)` when no nodes are present (no `/dev/kfd`, container
/// without sysfs, or only a CPU node) so callers can distinguish "no GPU" from
/// hard failures. Malformed or unreadable nodes are skipped, never fatal.
pub fn enumerate() -> Vec<AmdNode> {
    let root = topology_root();
    let read = match fs::read_dir(&root) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };

    let mut nodes = Vec::new();
    for entry in read.flatten() {
        let name = match entry.file_name().to_str().map(str::to_string) {
            Some(n) => n,
            None => continue,
        };
        let node_id: u32 = match name.parse() {
            Ok(n) => n,
            Err(_) => continue,
        };
        let node_dir = entry.path();
        let props_path = node_dir.join("properties");
        let props = match fs::read_to_string(&props_path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let map = parse_properties(&props);
        // `gpu_id` lives in a sibling file on real KFD; fall back to the
        // properties block for the test stub that crams it inline.
        let gpu_id = fs::read_to_string(node_dir.join("gpu_id"))
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .unwrap_or_else(|| map.get("gpu_id").copied().unwrap_or(0) as u32);
        if gpu_id == 0 {
            // CPU node — skip.
            continue;
        }
        let simd_per_cu = map.get("simd_per_cu").copied().unwrap_or(0) as u32;
        let max_waves_per_simd = map.get("max_waves_per_simd").copied().unwrap_or(0) as u32;
        nodes.push(AmdNode {
            node_id,
            gpu_id,
            drm_render_minor: map.get("drm_render_minor").copied().unwrap_or(0) as u32,
            gfx_target_version: map.get("gfx_target_version").copied().unwrap_or(0) as u32,
            simd_count: map.get("simd_count").copied().unwrap_or(0) as u32,
            array_count: map.get("array_count").copied().unwrap_or(0) as u32,
            simd_arrays_per_engine: (map.get("simd_arrays_per_engine").copied().unwrap_or(1) as u32).max(1),
            simd_per_cu,
            max_waves_per_simd,
            lds_size_in_kb: map.get("lds_size_in_kb").copied().unwrap_or(0) as u32,
            wave_front_size: map.get("wave_front_size").copied().unwrap_or(0) as u32,
            num_xcc: map.get("num_xcc").copied().unwrap_or(1) as u32,
            num_cp_queues: map.get("num_cp_queues").copied().unwrap_or(8) as u32,
            // KFD exposes `max_slots_scratch_cu` directly on some kernels;
            // fall back to simd_per_cu * max_waves_per_simd (the natural
            // scratch slot budget) when absent.
            max_slots_scratch_cu: map
                .get("max_slots_scratch_cu")
                .copied()
                .map(|v| v as u32)
                .unwrap_or_else(|| simd_per_cu.max(1) * max_waves_per_simd.max(1)),
            memory_bytes: frame_buffer_bytes(&node_dir),
            max_engine_clk_fcompute: map.get("max_engine_clk_fcompute").copied().unwrap_or(0),
            cu_per_simd_array: map.get("cu_per_simd_array").copied().unwrap_or(0) as u32,
            num_sdma_engines: map.get("num_sdma_engines").copied().unwrap_or(0) as u32,
            num_sdma_xgmi_engines: map.get("num_sdma_xgmi_engines").copied().unwrap_or(0) as u32,
            hive_id: map.get("hive_id").copied().unwrap_or(0),
            device_id: map.get("device_id").copied().unwrap_or(0) as u32,
            vendor_id: map.get("vendor_id").copied().unwrap_or(0) as u32,
            caches: cache_instances(&node_dir),
        });
    }
    nodes.sort_by_key(|n| n.node_id);
    nodes
}

/// True iff this host has a *supported* AMD GPU — a KFD node whose
/// `gfx_target_version` maps to a known [`AmdArch`](svod_dtype::AmdArch).
///
/// Sysfs-only and side-effect-free (no `/dev/kfd` open, no ioctl), so it's
/// cheap enough to call at device-registry construction to decide whether to
/// register the AMD factory. Mirrors the predicate `DeviceSpec::Amd` resolution
/// uses, so registration and device-open agree on "supported".
pub fn has_devices() -> bool {
    enumerate().iter().any(|n| svod_dtype::AmdArch::from_gfx_target_version(n.gfx_target_version).is_some())
}

/// KFD's frame-buffer heap types (`HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC` and
/// `_PRIVATE`).
const FRAME_BUFFER_HEAPS: [u64; 2] = [1, 2];

/// The bytes of a node's frame-buffer memory banks, 0 where none is listed.
fn frame_buffer_bytes(node_dir: &std::path::Path) -> u64 {
    let Ok(banks) = fs::read_dir(node_dir.join("mem_banks")) else { return 0 };
    banks
        .flatten()
        .filter_map(|bank| fs::read_to_string(bank.path().join("properties")).ok())
        .map(|props| parse_properties(&props))
        .filter(|map| map.get("heap_type").is_some_and(|h| FRAME_BUFFER_HEAPS.contains(h)))
        .filter_map(|map| map.get("size_in_bytes").copied())
        .sum()
}

/// The node's cache instances from `caches/*/properties`, by their numeric
/// directory name, empty where KFD lists none (or every file of it is
/// malformed, which skips that file alone): the per-CU/WGP L1s and LDS, the
/// L2 slices and the memory-side Infinity Cache, each with the processors
/// sharing it.
fn cache_instances(node_dir: &std::path::Path) -> Vec<AmdCache> {
    let Ok(caches) = fs::read_dir(node_dir.join("caches")) else { return Vec::new() };
    let mut parsed: Vec<(u32, AmdCache)> = caches
        .flatten()
        .filter_map(|cache| {
            let at = cache.file_name().to_str()?.parse::<u32>().ok()?;
            Some((at, parse_cache(&cache.path().join("properties"))?))
        })
        .collect();
    parsed.sort_by_key(|(at, _)| *at);
    parsed.into_iter().map(|(_, cache)| cache).collect()
}

/// One `caches/N/properties` file, `None` where unreadable or without a level.
fn parse_cache(path: &std::path::Path) -> Option<AmdCache> {
    let text = fs::read_to_string(path).ok()?;
    let map = parse_properties(&text);
    let level = map.get("level").copied()? as u32;
    let sibling_map = text
        .lines()
        .find_map(|line| line.trim().strip_prefix("sibling_map"))?
        .split(',')
        .map(|word| word.trim().parse::<u64>().ok())
        .collect::<Option<Vec<_>>>()?;
    Some(AmdCache {
        processor_id: map.get("processor_id_low").copied().unwrap_or(0) as u32,
        level,
        size_bytes: map.get("size").copied().unwrap_or(0) * 1024,
        line_bytes: map.get("cache_line_size").copied().unwrap_or(0) as u32,
        cache_lines_per_tag: map.get("cache_lines_per_tag").copied().unwrap_or(0) as u32,
        association: map.get("association").copied().unwrap_or(0) as u32,
        latency: map.get("latency").copied().unwrap_or(0) as u32,
        kind: map.get("type").copied().unwrap_or(0) as u32,
        sibling_map,
    })
}

pub(crate) fn parse_properties(s: &str) -> HashMap<String, u64> {
    let mut map = HashMap::new();
    for line in s.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let key = match it.next() {
            Some(k) => k.to_string(),
            None => continue,
        };
        let val = match it.next().and_then(|v| v.parse::<u64>().ok()) {
            Some(v) => v,
            None => continue,
        };
        map.insert(key, val);
    }
    map
}
