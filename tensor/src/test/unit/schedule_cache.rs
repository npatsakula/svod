use std::sync::Arc;

use crate::*;
use svod_ir::ops;

fn cfg() -> PrepareConfig {
    PrepareConfig::from(svod_schedule::OptimizerConfig::default())
}

/// Verify that same-shape computations produce identical content hashes
/// after normalization (the key property that enables kernel caching).
#[test]
fn test_same_shape_same_hash() {
    let _structural = svod_ir::origin::capture_for_thread(false);
    svod_schedule::testing::setup_test_tracing();
    let cfg = cfg();

    let c1 = (&Tensor::from_slice([1.0f32, 2.0, 3.0]) + &Tensor::from_slice([4.0f32, 5.0, 6.0])).unwrap();
    let c2 = (&Tensor::from_slice([10.0f32, 20.0, 30.0]) + &Tensor::from_slice([40.0f32, 50.0, 60.0])).unwrap();

    let (h1, d1) = crate::schedule_cache::cache_key_for(&c1, &cfg).unwrap();
    let (h2, d2) = crate::schedule_cache::cache_key_for(&c2, &cfg).unwrap();
    assert_eq!(h1, h2, "same-shape computations must produce same content hash");
    assert_eq!(d1, d2);
}

/// Verify that different shapes produce different hashes.
#[test]
fn test_different_shape_different_hash() {
    svod_schedule::testing::setup_test_tracing();
    let cfg = cfg();

    let c1 = (&Tensor::from_slice([1.0f32, 2.0, 3.0]) + &Tensor::from_slice([4.0f32, 5.0, 6.0])).unwrap();
    let c2 = (&Tensor::from_slice([1.0f32, 2.0]) + &Tensor::from_slice([3.0f32, 4.0])).unwrap();

    let (h1, _) = crate::schedule_cache::cache_key_for(&c1, &cfg).unwrap();
    let (h2, _) = crate::schedule_cache::cache_key_for(&c2, &cfg).unwrap();
    assert_ne!(h1, h2, "different shapes must produce different hashes");
}

/// Verify that different ops produce different hashes.
#[test]
fn test_different_op_different_hash() {
    svod_schedule::testing::setup_test_tracing();
    let cfg = cfg();

    let add = (&Tensor::from_slice([1.0f32, 2.0, 3.0]) + &Tensor::from_slice([4.0f32, 5.0, 6.0])).unwrap();
    let mul = (&Tensor::from_slice([1.0f32, 2.0, 3.0]) * &Tensor::from_slice([4.0f32, 5.0, 6.0])).unwrap();

    let (h_add, _) = crate::schedule_cache::cache_key_for(&add, &cfg).unwrap();
    let (h_mul, _) = crate::schedule_cache::cache_key_for(&mul, &cfg).unwrap();
    assert_ne!(h_add, h_mul, "different ops must produce different hashes");
}

/// Verify that realizing the same shape twice produces correct results.
/// The second call should reuse cached compiled kernels (not re-optimize/recompile).
#[test]
fn test_repeated_realize_correct() {
    svod_schedule::testing::setup_test_tracing();
    let cfg = cfg();

    let c1 = (&Tensor::from_slice([1.0f32, 2.0, 3.0]) + &Tensor::from_slice([4.0f32, 5.0, 6.0])).unwrap();
    c1.realize_with(&cfg).unwrap();
    assert_eq!(c1.as_vec::<f32>().unwrap(), vec![5.0, 7.0, 9.0]);

    let c2 = (&Tensor::from_slice([10.0f32, 20.0, 30.0]) + &Tensor::from_slice([40.0f32, 50.0, 60.0])).unwrap();
    c2.realize_with(&cfg).unwrap();
    assert_eq!(c2.as_vec::<f32>().unwrap(), vec![50.0, 70.0, 90.0]);
}

#[test]
fn test_prepare_with_reuses_same_schedule_cache_entry() {
    let _structural = svod_ir::origin::capture_for_thread(false);
    crate::test::helpers::test_setup();
    let cfg = cfg();

    let c1 = (&Tensor::from_slice([1.0f32, 2.0, 3.0]) + &Tensor::from_slice([4.0f32, 5.0, 6.0])).unwrap();
    let key1 = crate::schedule_cache::cache_key_for(&c1, &cfg).expect("first key");
    c1.prepare_with(&cfg).expect("first prepare");

    let cache = crate::schedule_cache::schedule_cache();
    let first_entry = {
        let guard = cache.guard();
        cache.get(&key1, &guard).cloned().expect("entry after first prepare")
    };

    let c2 = (&Tensor::from_slice([10.0f32, 20.0, 30.0]) + &Tensor::from_slice([40.0f32, 50.0, 60.0])).unwrap();
    let key2 = crate::schedule_cache::cache_key_for(&c2, &cfg).expect("second key");
    assert_eq!(key1, key2, "same-shape graph should map to same schedule cache key");
    c2.prepare_with(&cfg).expect("second prepare");

    let second_entry = {
        let guard = cache.guard();
        cache.get(&key2, &guard).cloned().expect("entry after second prepare")
    };

    assert!(
        Arc::ptr_eq(&first_entry, &second_entry),
        "same schedule cache key should reuse the cached pre-schedule entry"
    );
}

fn batch_cache_key(tensors: &[&Tensor], cfg: &PrepareConfig) -> (u64, String) {
    let sink = UOp::sink(tensors.iter().map(|t| t.uop().contiguous()).collect());
    let normalized = crate::realize::normalize_for_schedule_cache(&sink).expect("normalize cache key");
    let codegen = crate::realize::resolve_codegen(&normalized.param_buffers, cfg).expect("batch codegen");
    (crate::schedule_cache::content_hash(&normalized.normalized), codegen)
}

#[test]
fn test_resolve_codegen_skips_disk_buffers() {
    let config = cfg();
    let registry = svod_device::registry::registry();
    // A non-disk CPU buffer pins codegen to CPU; all-disk inputs fall back to
    // the active default device (CPU, or AMD under SVOD_DEVICE=AMD:0).
    let expected_cpu = config
        .resolve_device(&svod_dtype::DeviceSpec::Cpu, registry)
        .expect("CPU device should resolve")
        .compiler
        .cache_key()
        .to_string();
    let expected_default = config
        .resolve_device(&svod_dtype::default_device::default_device(), registry)
        .expect("default device should resolve")
        .compiler
        .cache_key()
        .to_string();

    let disk = UOp::new_buffer(
        svod_dtype::DeviceSpec::Disk { path: std::path::PathBuf::from("weights.bin") },
        16,
        DType::Float32,
    );
    let cpu = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 16, DType::Float32);

    let mixed = crate::realize::resolve_codegen(&[(disk.id, disk.clone()), (cpu.id, cpu)], &config)
        .expect("DISK buffers should not be selected for codegen");
    assert_eq!(mixed, expected_cpu);

    let fallback = crate::realize::resolve_codegen(&[(disk.id, disk)], &config)
        .expect("all-DISK inputs should fall back to default-device codegen");
    assert_eq!(fallback, expected_default);
}

#[test]
fn test_prepare_batch_with_reuses_same_schedule_cache_entry() {
    let _structural = svod_ir::origin::capture_for_thread(false);
    crate::test::helpers::test_setup();
    let cfg = cfg();

    let a1 = (&Tensor::from_slice([1.0f32, 2.0, 3.0]) + &Tensor::from_slice([4.0f32, 5.0, 6.0])).unwrap();
    let b1 = (&Tensor::from_slice([7.0f32, 8.0, 9.0]) + &Tensor::from_slice([10.0f32, 11.0, 12.0])).unwrap();
    let key1 = batch_cache_key(&[&a1, &b1], &cfg);

    Tensor::prepare_batch_with([&a1, &b1], &cfg).expect("first batch prepare");

    let cache = crate::schedule_cache::schedule_cache();
    let first_entry = {
        let guard = cache.guard();
        cache.get(&key1, &guard).cloned().expect("entry after first batch prepare")
    };

    let a2 = (&Tensor::from_slice([13.0f32, 14.0, 15.0]) + &Tensor::from_slice([16.0f32, 17.0, 18.0])).unwrap();
    let b2 = (&Tensor::from_slice([19.0f32, 20.0, 21.0]) + &Tensor::from_slice([22.0f32, 23.0, 24.0])).unwrap();
    let key2 = batch_cache_key(&[&a2, &b2], &cfg);
    assert_eq!(key1, key2, "same-structure batch graph should map to same schedule cache key");

    Tensor::prepare_batch_with([&a2, &b2], &cfg).expect("second batch prepare");

    let second_entry = {
        let guard = cache.guard();
        cache.get(&key2, &guard).cloned().expect("entry after second batch prepare")
    };

    assert!(
        Arc::ptr_eq(&first_entry, &second_entry),
        "same batch schedule cache key should reuse cached pre-schedule entry"
    );
}

/// Verify matmul kernel caching produces correct results on repeated calls.
#[test]
fn test_repeated_matmul_correct() {
    svod_schedule::testing::setup_test_tracing();
    let cfg = cfg();

    let a1 = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape([2, 2]).unwrap();
    let b1 = Tensor::from_slice([5.0f32, 6.0, 7.0, 8.0]).try_reshape([2, 2]).unwrap();
    let c1 = a1.matmul(&b1).unwrap();
    c1.realize_with(&cfg).unwrap();
    assert_eq!(c1.as_vec::<f32>().unwrap(), vec![19.0, 22.0, 43.0, 50.0]);

    let a2 = Tensor::from_slice([10.0f32, 20.0, 30.0, 40.0]).try_reshape([2, 2]).unwrap();
    let b2 = Tensor::from_slice([50.0f32, 60.0, 70.0, 80.0]).try_reshape([2, 2]).unwrap();
    let c2 = a2.matmul(&b2).unwrap();
    c2.realize_with(&cfg).unwrap();
    assert_eq!(c2.as_vec::<f32>().unwrap(), vec![1900.0, 2200.0, 4300.0, 5000.0]);
}

/// Verify that different bind values normalize to same schedule cache key.
#[test]
fn test_different_bind_values_same_hash() {
    let _structural = svod_ir::origin::capture_for_thread(false);
    svod_schedule::testing::setup_test_tracing();
    let cfg = cfg();

    let n = Variable::new("N", 1, 32);
    let shape_3 = [n.bind(3).unwrap().as_sint()];
    let shape_7 = [n.bind(7).unwrap().as_sint()];

    let t3 = Tensor::empty_dynamic(&shape_3, DType::Float32);
    let t7 = Tensor::empty_dynamic(&shape_7, DType::Float32);

    let c3 = t3.sum(()).unwrap();
    let c7 = t7.sum(()).unwrap();

    let (h3, d3) = crate::schedule_cache::cache_key_for(&c3, &cfg).unwrap();
    let (h7, d7) = crate::schedule_cache::cache_key_for(&c7, &cfg).unwrap();

    assert_eq!(h3, h7, "same graph with different bind values must share schedule cache key");
    assert_eq!(d3, d7);
}

#[test]
fn test_prepare_with_bind_reuses_same_schedule_cache_entry() {
    let _structural = svod_ir::origin::capture_for_thread(false);
    crate::test::helpers::test_setup();
    let cfg = cfg();

    let n = Variable::new("N", 1, 32);

    let c3 = Tensor::empty_dynamic(&[n.bind(3).unwrap().as_sint()], DType::Float32).sum(()).unwrap();
    let key3 = crate::schedule_cache::cache_key_for(&c3, &cfg).expect("first bind key");
    c3.prepare_with(&cfg).expect("first bind prepare");

    let cache = crate::schedule_cache::schedule_cache();
    let first_entry = {
        let guard = cache.guard();
        cache.get(&key3, &guard).cloned().expect("cache entry after first bind prepare")
    };

    let c7 = Tensor::empty_dynamic(&[n.bind(7).unwrap().as_sint()], DType::Float32).sum(()).unwrap();
    let key7 = crate::schedule_cache::cache_key_for(&c7, &cfg).expect("second bind key");
    assert_eq!(key3, key7, "same bind graph shape should map to same schedule cache key");
    c7.prepare_with(&cfg).expect("second bind prepare");

    let second_entry = {
        let guard = cache.guard();
        cache.get(&key7, &guard).cloned().expect("cache entry after second bind prepare")
    };

    assert!(
        Arc::ptr_eq(&first_entry, &second_entry),
        "bind-backed schedule key should reuse cached pre-schedule entry"
    );
}

#[test]
fn test_rebind_realize_with_cache_hit_keeps_bound_values() {
    let _structural = svod_ir::origin::capture_for_thread(false);
    crate::test::helpers::test_setup();
    let cfg = cfg();
    let n = Variable::new("N", 1, 16);

    let shape_1 = [n.bind(1).unwrap().as_sint()];
    let input_1 = Tensor::full_dynamic(&shape_1, 1.0f32, DType::Float32).expect("create N=1 tensor");
    let sum_1 = input_1.sum(()).unwrap();
    let key_1 = crate::schedule_cache::cache_key_for(&sum_1, &cfg).expect("first bind key");
    sum_1.realize_with(&cfg).expect("realize N=1");
    assert_eq!(sum_1.as_vec::<f32>().expect("N=1 output"), vec![1.0]);

    let shape_3 = [n.bind(3).unwrap().as_sint()];
    let input_3 = Tensor::full_dynamic(&shape_3, 1.0f32, DType::Float32).expect("create N=3 tensor");
    let sum_3 = input_3.sum(()).unwrap();
    let key_3 = crate::schedule_cache::cache_key_for(&sum_3, &cfg).expect("second bind key");
    assert_eq!(key_1, key_3, "rebind graph should hit the same schedule cache key");

    sum_3.realize_with(&cfg).expect("realize N=3");
    assert_eq!(sum_3.as_vec::<f32>().expect("N=3 output"), vec![3.0]);
}

#[test]
fn test_normalize_for_schedule_cache_collects_var_vals_and_strips_bind_values() {
    let n = Variable::new("N", 1, 32);
    let shape = [n.bind(5).unwrap().as_sint()];
    let t = Tensor::empty_dynamic(&shape, DType::Float32);
    let c = t.sum(()).unwrap();
    let sink = UOp::sink(vec![c.uop().contiguous()]);

    let normalized = crate::realize::normalize_for_schedule_cache(&sink).expect("normalize bind graph");
    assert_eq!(normalized.var_vals.get("N"), Some(&5));

    assert!(
        normalized.normalized.toposort().iter().all(|node| !matches!(node.op(), Op::Bind(..))),
        "normalized graph should replace BIND placeholders with PARAM for reversible restore"
    );
    assert!(
        normalized
            .normalized
            .toposort()
            .iter()
            .any(|node| matches!(node.op(), Op::Param(ops::Param { arg, .. }) if arg.device.is_some())),
        "normalized graph should include PARAM placeholders for stripped runtime BIND values"
    );
}

#[test]
fn test_normalize_for_schedule_cache_normalizes_standalone_unique_identity() {
    let sink_a = UOp::sink(vec![UOp::buffer_id(Some(42))]);
    let sink_b = UOp::sink(vec![UOp::buffer_id(Some(777))]);
    let normalized_a = crate::realize::normalize_for_schedule_cache(&sink_a).expect("normalize unique sink A");
    let normalized_b = crate::realize::normalize_for_schedule_cache(&sink_b).expect("normalize unique sink B");

    assert!(
        normalized_a.normalized.toposort().iter().any(|node| matches!(node.op(), Op::LUnique(_))),
        "standalone UNIQUE should normalize to LUNIQUE placeholder for stable cache keys"
    );
    assert!(
        normalized_a.normalized.toposort().iter().all(|node| !matches!(node.op(), Op::Unique(_))),
        "normalized standalone UNIQUE identity should not leak runtime UNIQUE ids"
    );
    assert_eq!(
        crate::schedule_cache::content_hash(&normalized_a.normalized),
        crate::schedule_cache::content_hash(&normalized_b.normalized),
        "different runtime UNIQUE ids should normalize to identical cache-key structure"
    );
    assert!(normalized_a.param_buffers.is_empty());
    assert!(normalized_a.var_vals.is_empty());
}

#[test]
fn test_normalize_for_schedule_cache_preserves_internal_buffers() {
    let global = UOp::buffer(101, 8, DType::Float32, svod_ir::AddrSpace::Global, Some(svod_dtype::DeviceSpec::Cpu));
    let local = UOp::buffer(909, 8, DType::Float32, svod_ir::AddrSpace::Local, None);
    let reg = UOp::buffer(707, 1, DType::Float32, svod_ir::AddrSpace::Reg, None);
    let normalized = crate::realize::normalize_for_schedule_cache(&UOp::sink(vec![global, local, reg]))
        .expect("normalize mixed storage");

    assert_eq!(normalized.param_values.len(), 1, "only global storage is a positional input");
    assert!(normalized.normalized.toposort().iter().any(
        |u| matches!(u.op(), Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_ir::AddrSpace::Local) && arg.slot == 909)
    ));
    assert!(normalized.normalized.toposort().iter().any(
        |u| matches!(u.op(), Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_ir::AddrSpace::Reg) && arg.slot == 707)
    ));
}

#[test]
fn test_post_sched_cache_restore_replaces_params() {
    let c = (&Tensor::from_slice([1.0f32, 2.0, 3.0]) + &Tensor::from_slice([4.0f32, 5.0, 6.0])).unwrap();
    let sink = UOp::sink(vec![c.uop().contiguous()]);

    let normalized = crate::realize::normalize_for_schedule_cache(&sink).expect("normalize param restore sink");
    assert!(normalized.normalized.toposort().iter().any(|n| matches!(n.op(), Op::Param(..))));

    let restored = crate::realize::restore_post_schedule_cache(&normalized.normalized, &normalized);
    assert!(restored.toposort().iter().any(|n| matches!(n.op(), Op::Buffer(..))));
    assert!(
        restored.toposort().iter().all(|n| !matches!(n.op(), Op::Param(..))),
        "restored graph should not retain PARAM placeholders"
    );
}

#[test]
fn test_post_sched_cache_restore_materializes_runtime_buffers() {
    let shape = svod_ir::shape::shape_to_uop(&smallvec::smallvec![svod_ir::SInt::Const(8)]);
    let arg = svod_ir::ParamArg::buffer(
        usize::MAX,
        DType::Float32,
        svod_ir::AddrSpace::Global,
        Some(svod_dtype::DeviceSpec::Cpu),
    );
    let placeholder = UOp::new(Op::Buffer(ops::Buffer { shape, arg: arg.into() }), DType::Float32)
        .with_tag(smallvec::smallvec![svod_ir::uop::canonical::TAG_SCHEDULE_LOCAL_BUFFER]);
    let root = UOp::sink(vec![placeholder]);

    let normalization = crate::realize::ScheduleCacheNormalization {
        normalized: root.clone(),
        param_values: vec![],
        param_buffers: vec![],
        unique_values: vec![],
        var_vals: std::collections::HashMap::new(),
    };

    let restored = crate::realize::restore_post_schedule_cache(&root, &normalization);
    assert!(
        restored
            .toposort()
            .iter()
            .any(|n| matches!(n.op(), Op::Buffer(ops::Buffer { arg, .. }) if arg.slot != usize::MAX))
    );
    assert!(
        restored
            .toposort()
            .iter()
            .all(|n| !matches!(n.op(), Op::Buffer(ops::Buffer { arg, .. }) if arg.slot == usize::MAX))
    );
}

#[test]
fn test_post_sched_cache_restore_keeps_codegen_params_inside_call_body() {
    crate::test::helpers::test_setup();

    let c = (&Tensor::from_slice([1.0f32, 2.0, 3.0]) + &Tensor::from_slice([4.0f32, 5.0, 6.0])).unwrap();
    let sink = UOp::sink(vec![c.uop().contiguous()]);

    let normalization = crate::realize::normalize_for_schedule_cache(&sink).expect("normalize call boundary sink");
    let rangeify = svod_schedule::rangeify_with_map(normalization.normalized.clone()).unwrap();
    let (kernel_graph, _) = svod_schedule::try_get_kernel_graph(rangeify.sink).unwrap();

    let pre_schedule = crate::schedule::create_pre_schedule(kernel_graph).unwrap();
    let restored = crate::realize::restore_post_schedule_pre_schedule(&pre_schedule, &normalization);

    assert!(
        restored.items.iter().flat_map(|item| &item.sources).all(|source| matches!(source.op(), Op::Buffer(..))),
        "CALL arguments must restore to runtime buffers"
    );
    assert!(
        restored.items.iter().all(|item| item.ast.toposort().iter().any(|node| {
            matches!(node.op(), Op::Param(..))
                && node.tag().as_ref().is_some_and(|tags| tags.contains(&svod_ir::uop::canonical::TAG_CODEGEN_PARAM))
        })),
        "callable bodies must retain codegen PARAM formals"
    );
    assert!(
        restored.items.iter().all(|item| item.ast.toposort().iter().all(|node| !matches!(node.op(), Op::Buffer(..)))),
        "codegen AST must not contain runtime BUFFERs"
    );
}

#[test]
fn test_normalize_for_schedule_cache_rejects_conflicting_bind_values() {
    let var = UOp::define_var("N".to_string(), 1, 32);
    let bind_3 = var.bind(UOp::index_const(3));
    let bind_7 = var.bind(UOp::index_const(7));
    let sink = UOp::sink(vec![bind_3, bind_7]);

    let err = match crate::realize::normalize_for_schedule_cache(&sink) {
        Ok(_) => panic!("conflicting bind values must fail"),
        Err(err) => err,
    };
    assert!(format!("{err}").contains("bind mismatch on variable N"), "error should mention bind mismatch");
}

#[test]
fn test_function_symbolic_actuals_participate_in_schedule_cache_identity() {
    let formal_dim = UOp::scalar_param(1, Some("formal".into()), DType::WeakInt, 1, 8);
    let formal = UOp::param_with_shape(0, &smallvec::smallvec![SInt::Symbolic(formal_dim)], DType::Float32, None);

    let make = |name: &str| {
        let dim = UOp::define_var(name.into(), 1, 8);
        let actual = UOp::param_with_shape(7, &smallvec::smallvec![SInt::Symbolic(dim.clone())], DType::Float32, None);
        UOp::sink(vec![formal.function(smallvec::smallvec![actual, dim], CallInfo::default()).try_gettuple(0).unwrap()])
    };

    let a = crate::realize::normalize_for_schedule_cache(&make("a")).unwrap();
    let b = crate::realize::normalize_for_schedule_cache(&make("b")).unwrap();
    assert_ne!(
        crate::schedule_cache::content_hash(&a.normalized),
        crate::schedule_cache::content_hash(&b.normalized),
        "different symbolic actuals must not alias in the schedule cache"
    );
}

#[test]
fn test_schedule_cache_normalization_preserves_opaque_call_body() {
    let hidden = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 8, DType::Float32);
    let body = UOp::sink(vec![hidden]);
    let actual = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 8, DType::Float32);
    let call = body.call(smallvec::smallvec![actual], CallInfo::default());
    let normalized = crate::realize::normalize_for_schedule_cache(&UOp::sink(vec![call])).unwrap();
    let normalized_call = normalized
        .normalized
        .toposort_call_aware(false)
        .into_iter()
        .find(|node| matches!(node.op(), Op::Call(..)))
        .unwrap();
    let Op::Call(ops::Call { body: normalized_body, .. }) = normalized_call.op() else { unreachable!() };
    assert!(Arc::ptr_eq(normalized_body, &body));
}

// ============================================================================
// Cache cold-vs-hit equivalence tests
//
// These exercise the contract that the schedule-cache code path produces
// numerically identical outputs to the cache-cold rangeify/scheduling path.
// They use `PrepareConfig::disable_schedule_cache` to force the cold path
// without mutating the process-global `SVOD_DISABLE_SCHEDULE_CACHE` env var.
// ============================================================================

fn cfg_with_cache_disabled() -> PrepareConfig {
    let mut c = cfg();
    c.disable_schedule_cache = true;
    c
}

#[test]
fn test_cache_disabled_equals_enabled_outputs_static_shape() {
    svod_schedule::testing::setup_test_tracing();

    let lhs = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];

    let warm = (&Tensor::from_slice(lhs) + &Tensor::from_slice(rhs)).unwrap().try_reshape([2, 3]).unwrap();
    warm.realize_with(&cfg()).expect("cache-warm realize");
    let warm_out = warm.as_vec::<f32>().expect("warm output");

    let cold = (&Tensor::from_slice(lhs) + &Tensor::from_slice(rhs)).unwrap().try_reshape([2, 3]).unwrap();
    cold.realize_with(&cfg_with_cache_disabled()).expect("cache-cold realize");
    let cold_out = cold.as_vec::<f32>().expect("cold output");

    assert_eq!(warm_out, cold_out, "cold-path and cache-warm realize must produce identical outputs");
}

#[test]
fn test_cache_disabled_equals_enabled_outputs_dynamic_shape() {
    svod_schedule::testing::setup_test_tracing();

    let n = Variable::new("N", 1, 16);
    let bound = [n.bind(5).unwrap().as_sint()];

    let warm = Tensor::full_dynamic(&bound, 2.0f32, DType::Float32).expect("warm tensor").sum(()).unwrap();
    warm.realize_with(&cfg()).expect("warm realize");
    let warm_out = warm.as_vec::<f32>().expect("warm output");

    let cold = Tensor::full_dynamic(&bound, 2.0f32, DType::Float32).expect("cold tensor").sum(()).unwrap();
    cold.realize_with(&cfg_with_cache_disabled()).expect("cold realize");
    let cold_out = cold.as_vec::<f32>().expect("cold output");

    assert_eq!(
        warm_out, cold_out,
        "cold-path and cache-warm realize on bound dynamic shape must produce identical outputs"
    );
}

#[test]
fn test_cache_hit_order_independence_dynamic_shape() {
    // Bind n=3 then n=7 in one tensor pair; bind n=7 then n=3 in another;
    // verify both produce per-bind-value-correct outputs regardless of which
    // ordering populates the schedule cache first.
    svod_schedule::testing::setup_test_tracing();
    let cfg = cfg();
    let n = Variable::new("N", 1, 16);

    // Path A: small then large
    let a_small = Tensor::full_dynamic(&[n.bind(3).unwrap().as_sint()], 1.0f32, DType::Float32)
        .expect("a small")
        .sum(())
        .unwrap();
    a_small.realize_with(&cfg).expect("a small realize");
    let a_small_out = a_small.as_vec::<f32>().expect("a small output");

    let a_large = Tensor::full_dynamic(&[n.bind(7).unwrap().as_sint()], 1.0f32, DType::Float32)
        .expect("a large")
        .sum(())
        .unwrap();
    a_large.realize_with(&cfg).expect("a large realize");
    let a_large_out = a_large.as_vec::<f32>().expect("a large output");

    // Path B: large then small (cache key is shared, so this exercises a
    // cache-warm path against a fresh tensor — outputs must still be
    // per-bind-value-correct, not stale-from-the-other-path).
    let b_large = Tensor::full_dynamic(&[n.bind(7).unwrap().as_sint()], 1.0f32, DType::Float32)
        .expect("b large")
        .sum(())
        .unwrap();
    b_large.realize_with(&cfg).expect("b large realize");
    let b_large_out = b_large.as_vec::<f32>().expect("b large output");

    let b_small = Tensor::full_dynamic(&[n.bind(3).unwrap().as_sint()], 1.0f32, DType::Float32)
        .expect("b small")
        .sum(())
        .unwrap();
    b_small.realize_with(&cfg).expect("b small realize");
    let b_small_out = b_small.as_vec::<f32>().expect("b small output");

    // Sums of all-ones with the bound dim populated to vmax produce the bind value.
    assert_eq!(a_small_out, vec![3.0]);
    assert_eq!(a_large_out, vec![7.0]);
    assert_eq!(b_small_out, vec![3.0], "small-after-large must still produce per-bind output, not cached stale value");
    assert_eq!(b_large_out, vec![7.0], "large-first must produce correct large output");
    assert_eq!(a_small_out, b_small_out, "ordering must not affect outputs for shared cache key");
    assert_eq!(a_large_out, b_large_out, "ordering must not affect outputs for shared cache key");
}

#[test]
fn test_cache_disabled_matches_enabled_var_names_and_kernel_count() {
    // Beyond numerical output, verify the prepared plan structure matches
    // between cold and warm paths: same kernel count, same buffer counts.
    svod_schedule::testing::setup_test_tracing();
    let lhs = [1.0f32, 2.0, 3.0, 4.0];
    let rhs = [10.0f32, 20.0, 30.0, 40.0];

    let warm_plan = (&Tensor::from_slice(lhs) + &Tensor::from_slice(rhs))
        .unwrap()
        .try_reshape([2, 2])
        .unwrap()
        .prepare_with(&cfg())
        .expect("warm prepare");

    let cold_plan = (&Tensor::from_slice(lhs) + &Tensor::from_slice(rhs))
        .unwrap()
        .try_reshape([2, 2])
        .unwrap()
        .prepare_with(&cfg_with_cache_disabled())
        .expect("cold prepare");

    assert_eq!(
        warm_plan.prepared_kernels().len(),
        cold_plan.prepared_kernels().len(),
        "kernel count must match across cache-warm and cache-cold paths"
    );
    assert_eq!(
        warm_plan.buffers().len(),
        cold_plan.buffers().len(),
        "buffer count must match across cache-warm and cache-cold paths"
    );
    assert_eq!(
        warm_plan.num_outputs(),
        cold_plan.num_outputs(),
        "output count must match across cache-warm and cache-cold paths"
    );
}

/// An ephemeral prepare records nothing: its schedule is not in the cache
/// afterwards and its kernel is compiled again by a plain prepare of the
/// same computation, which a second plain prepare then shares; the
/// ephemeral plan runs all the same.
#[test]
fn an_ephemeral_prepare_records_nothing() {
    svod_schedule::testing::setup_test_tracing();
    let fresh =
        || (&Tensor::from_slice([1.0f32, 2.0, 3.0, 5.0]) * &Tensor::from_slice([7.0f32, 11.0, 13.0, 17.0])).unwrap();
    let plain = cfg();
    let ephemeral = PrepareConfig { ephemeral: true, ..cfg() };
    let kernels = |plan: &svod_runtime::ExecutionPlan| plan.kernels().map(std::ptr::from_ref).collect::<Vec<_>>();
    let key = crate::schedule_cache::cache_key_for(&fresh(), &ephemeral).unwrap();
    let t = fresh();
    let once = t.prepare_with(&ephemeral).unwrap();
    once.execute().unwrap();
    assert_eq!(t.to_vec::<f32>().unwrap(), [7.0, 22.0, 39.0, 85.0]);
    assert!(!crate::schedule_cache::schedule_cache().pin().contains_key(&key));
    let (first, second) = (fresh().prepare_with(&plain).unwrap(), fresh().prepare_with(&plain).unwrap());
    assert!(!kernels(&once).is_empty());
    assert!(kernels(&once).iter().zip(kernels(&first)).all(|(a, b)| !std::ptr::eq(*a, b)), "published");
    assert!(kernels(&first).iter().zip(kernels(&second)).all(|(a, b)| std::ptr::eq(*a, b)), "not cached");
    assert!(crate::schedule_cache::schedule_cache().pin().contains_key(&key));
}
