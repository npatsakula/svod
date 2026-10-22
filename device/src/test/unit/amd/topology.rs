use crate::amd::topology::*;
use std::io::Write;
use std::path::PathBuf;

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

/// Reads the real `/sys/devices/virtual/kfd/kfd/topology/nodes/` if it
/// exists on this host. Asserts only that the parser doesn't choke on
/// real-world data; the field values depend on the local hardware.
#[test]
fn enumerate_real_host_topology_does_not_panic() {
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
