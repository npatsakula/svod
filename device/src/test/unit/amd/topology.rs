use crate::amd::topology::*;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// Serializes the tests that point `SVOD_KFD_TOPOLOGY` at their own tree:
/// the override is process-wide, so two of them at once read each other's.
fn topology_override() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[test]
fn parse_properties_handles_real_kfd_format() {
    let raw = "cpu_cores_count 0\nsimd_count 4\ngfx_target_version 110000\ndrm_render_minor 128\n";
    let map = parse_properties(raw);
    assert_eq!(map["cpu_cores_count"], 0);
    assert_eq!(map["simd_count"], 4);
    assert_eq!(map["gfx_target_version"], 110000);
    assert_eq!(map["drm_render_minor"], 128);
}

#[test]
fn enumerate_returns_empty_when_topology_missing() {
    let temp = tempfile_dir();
    // Point at a non-existent directory.
    unsafe {
        std::env::set_var("SVOD_KFD_TOPOLOGY", temp.join("does_not_exist"));
    }
    let nodes = enumerate();
    unsafe {
        std::env::remove_var("SVOD_KFD_TOPOLOGY");
    }
    assert!(nodes.is_empty());
}

#[test]
fn enumerate_skips_cpu_nodes_and_parses_gpu() {
    let root = tempfile_dir();
    let n0 = root.join("0");
    let n1 = root.join("1");
    std::fs::create_dir_all(&n0).unwrap();
    std::fs::create_dir_all(&n1).unwrap();
    // Node 0: CPU (gpu_id 0).
    let mut f = std::fs::File::create(n0.join("properties")).unwrap();
    write!(f, "gpu_id 0\ncpu_cores_count 32\nsimd_count 0\n").unwrap();
    // Node 1: GPU.
    let mut f = std::fs::File::create(n1.join("properties")).unwrap();
    write!(
            f,
            "gpu_id 5710\nsimd_count 4\narray_count 4\nsimd_arrays_per_engine 2\ngfx_target_version 110000\ndrm_render_minor 128\nwave_front_size 32\nnum_cp_queues 8\n"
        )
        .unwrap();
    // Two frame-buffer banks and a system heap, which is not the GPU's.
    for (bank, heap, bytes) in [(0, 1, 1u64 << 33), (1, 2, 1 << 30), (2, 0, 1 << 34)] {
        let dir = n1.join("mem_banks").join(bank.to_string());
        std::fs::create_dir_all(&dir).unwrap();
        let mut f = std::fs::File::create(dir.join("properties")).unwrap();
        write!(f, "heap_type {heap}\nsize_in_bytes {bytes}\nflags 0\n").unwrap();
    }

    unsafe {
        std::env::set_var("SVOD_KFD_TOPOLOGY", &root);
    }
    let nodes = enumerate();
    unsafe {
        std::env::remove_var("SVOD_KFD_TOPOLOGY");
    }
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].node_id, 1);
    assert_eq!(nodes[0].gpu_id, 5710);
    assert_eq!(nodes[0].gfx_target_version, 110000);
    assert_eq!(nodes[0].drm_render_minor, 128);
    assert_eq!(nodes[0].wave_front_size, 32);
    assert_eq!(nodes[0].num_cp_queues, 8);
    assert_eq!(nodes[0].simd_arrays_per_engine, 2);
    assert_eq!(nodes[0].memory_bytes, (1 << 33) + (1 << 30));
}

fn tempfile_dir() -> PathBuf {
    // Build a fresh per-test tempdir so concurrent tests don't collide on
    // `SVOD_KFD_TOPOLOGY`. We don't pull `tempfile` for one test path.
    let pid = std::process::id();
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("svod-kfd-topo-{pid}-{nonce}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `sibling_map` bitmap text for `shared` processors from processor 0.
fn siblings(shared: u32) -> String {
    let words = (0..shared.div_ceil(64)).map(|w| {
        let bits = shared.saturating_sub(w * 64).min(64);
        if bits == 64 { u64::MAX.to_string() } else { ((1u64 << bits) - 1).to_string() }
    });
    let padding = std::iter::repeat_n("0".to_string(), 32);
    words.chain(padding).collect::<Vec<_>>().join(",")
}

/// Writes `count` cache instances of one level into `node/caches/` from
/// index `at`, each `kb` KB shared by `shared` processors.
fn write_caches(node: &std::path::Path, at: u32, count: u32, level: u32, kb: u64, shared: u32) {
    for cache in at..at + count {
        let dir = node.join("caches").join(cache.to_string());
        std::fs::create_dir_all(&dir).unwrap();
        let mut f = std::fs::File::create(dir.join("properties")).unwrap();
        let map = siblings(shared);
        write!(
            f,
            "processor_id_low {cache}\nlevel {level}\nsize {kb}\ncache_line_size 128\n\
             cache_lines_per_tag 0\nassociation 16\nlatency 0\ntype 9\nsibling_map {map}\n"
        )
        .unwrap();
    }
}

/// A node's facts beyond the geometry: the engine clock, the SDMA engines,
/// the hive and PCI ids, and the cache instances with their sharing.
#[test]
fn enumerate_reads_an_rdna4_like_node_with_its_caches() {
    let _lock = topology_override();
    let root = tempfile_dir();
    let node = root.join("1");
    std::fs::create_dir_all(&node).unwrap();
    let mut f = std::fs::File::create(node.join("properties")).unwrap();
    write!(
        f,
        "gpu_id 4660\ngfx_target_version 120001\nsimd_count 256\nsimd_per_cu 4\narray_count 16\n\
         lds_size_in_kb 64\nwave_front_size 32\nnum_xcc 1\nmax_engine_clk_fcompute 2900\n\
         cu_per_simd_array 8\nnum_sdma_engines 2\nnum_sdma_xgmi_engines 0\nhive_id 0\n\
         device_id 19026\nvendor_id 4098\n"
    )
    .unwrap();
    // A per-WGP data L1 (two CUs share it), two 4 MB L2 slices (8 MB in all),
    // and the 64 MB memory-side Infinity Cache the whole GPU shares.
    write_caches(&node, 0, 32, 1, 128, 2);
    write_caches(&node, 32, 2, 2, 4 * 1024, 32);
    write_caches(&node, 34, 1, 3, 64 * 1024, 64);

    unsafe {
        std::env::set_var("SVOD_KFD_TOPOLOGY", &root);
    }
    let nodes = enumerate();
    unsafe {
        std::env::remove_var("SVOD_KFD_TOPOLOGY");
    }
    let node = &nodes[0];
    assert_eq!(node.max_engine_clk_fcompute, 2900);
    assert_eq!(node.cu_per_simd_array, 8);
    assert_eq!(node.num_sdma_engines, 2);
    assert_eq!(node.num_sdma_xgmi_engines, 0);
    assert_eq!(node.device_id, 19026);
    assert_eq!(node.vendor_id, 4098);
    assert_eq!(node.caches.len(), 35);
    let wgp_l1 = node.caches.iter().find(|c| c.level == 1).unwrap();
    assert_eq!((wgp_l1.size_bytes, wgp_l1.shared_by()), (128 * 1024, 2));
    let l2 = node.caches.iter().filter(|c| c.level == 2).collect::<Vec<_>>();
    assert_eq!(l2.iter().map(|c| c.size_bytes).sum::<u64>(), 8 * 1024 * 1024, "the L2 slices sum");
    assert!(l2.iter().all(|c| c.shared_by() == 32));
    let llc = node.caches.iter().find(|c| c.level == 3).unwrap();
    assert_eq!((llc.size_bytes, llc.shared_by()), (64 * 1024 * 1024, 64));
}

/// MI300X-shaped: eight XCDs, each with its own 4 MB L2 shared by its 38 CUs,
/// one 256 MB Infinity Cache, 64 KB of LDS per CU.
#[test]
fn enumerate_reads_an_mi300x_like_node_with_its_caches() {
    let _lock = topology_override();
    let root = tempfile_dir();
    let node = root.join("3");
    std::fs::create_dir_all(&node).unwrap();
    let mut f = std::fs::File::create(node.join("properties")).unwrap();
    write!(
        f,
        "gpu_id 9241\ngfx_target_version 90402\nsimd_count 1216\nsimd_per_cu 4\narray_count 32\n\
         lds_size_in_kb 64\nwave_front_size 64\nnum_xcc 8\nmax_engine_clk_fcompute 2100\n\
         cu_per_simd_array 10\nnum_sdma_engines 12\nnum_sdma_xgmi_engines 12\n\
         hive_id 1397\ndevice_id 29872\nvendor_id 4098\n"
    )
    .unwrap();
    write_caches(&node, 0, 8, 2, 4 * 1024, 38);
    write_caches(&node, 8, 1, 3, 256 * 1024, 304);

    unsafe {
        std::env::set_var("SVOD_KFD_TOPOLOGY", &root);
    }
    let nodes = enumerate();
    unsafe {
        std::env::remove_var("SVOD_KFD_TOPOLOGY");
    }
    let node = &nodes[0];
    assert_eq!(node.num_xcc, 8);
    assert_eq!(node.hive_id, 1397);
    assert_eq!(node.num_sdma_xgmi_engines, 12);
    assert_eq!(node.lds_size_in_kb, 64);
    let l2: Vec<_> = node.caches.iter().filter(|c| c.level == 2).collect();
    assert_eq!(l2.len(), 8, "one L2 per XCD");
    assert_eq!(l2.iter().map(|c| c.size_bytes).sum::<u64>(), 32 * 1024 * 1024);
    assert!(l2.iter().all(|c| c.shared_by() == 38), "each XCD's 38 CUs share its L2");
    let llc = node.caches.iter().find(|c| c.level == 3).unwrap();
    assert_eq!((llc.size_bytes, llc.shared_by()), (256 * 1024 * 1024, 304));
}

/// A node without the new keys and without a `caches/` directory parses with
/// them absent, never failing the enumeration.
#[test]
fn enumerate_is_absent_safe_without_the_new_topology() {
    let _lock = topology_override();
    let root = tempfile_dir();
    let node = root.join("2");
    std::fs::create_dir_all(&node).unwrap();
    let mut f = std::fs::File::create(node.join("properties")).unwrap();
    write!(f, "gpu_id 77\nsimd_count 4\nsimd_per_cu 2\ngfx_target_version 100306\n").unwrap();

    unsafe {
        std::env::set_var("SVOD_KFD_TOPOLOGY", &root);
    }
    let nodes = enumerate();
    unsafe {
        std::env::remove_var("SVOD_KFD_TOPOLOGY");
    }
    let node = &nodes[0];
    assert_eq!(node.max_engine_clk_fcompute, 0);
    assert_eq!(node.num_sdma_engines, 0);
    assert_eq!(node.hive_id, 0);
    assert!(node.caches.is_empty());
}

/// Reads the real `/sys/devices/virtual/kfd/kfd/topology/nodes/` if it
/// exists on this host. Asserts only that the parser doesn't choke on
/// real-world data; the field values depend on the local hardware.
#[test]
fn enumerate_real_host_topology_does_not_panic() {
    let _lock = topology_override();
    // Ensure no test-suite override leaks in.
    unsafe {
        std::env::remove_var("SVOD_KFD_TOPOLOGY");
    }
    let nodes = enumerate();
    for n in &nodes {
        assert!(n.gpu_id != 0, "enumerate must skip CPU nodes (got gpu_id 0 in {n:?})");
    }
    eprintln!("host has {} KFD GPU node(s): {nodes:?}", nodes.len());
}
