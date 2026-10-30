use super::*;
use svod_dtype::DType;

fn key(name: &str) -> WeightKey {
    WeightKey {
        path: std::path::PathBuf::from("/fake/checkpoint.safetensors"),
        name: name.to_string(),
        dtype: DType::Float32,
        shape: vec![4],
        device: svod_dtype::DeviceSpec::Cpu,
    }
}

#[test]
fn same_provenance_shares_immutable_storage() {
    let bytes = [1u8; 16];
    let a = shared_weight_buffer(key("w_share"), &bytes);
    let b = shared_weight_buffer(key("w_share"), &bytes);
    assert_eq!(a.storage_id(), b.storage_id(), "same provenance must share storage");
    assert!(a.is_immutable());

    let other = shared_weight_buffer(key("w_other"), &bytes);
    assert_ne!(a.storage_id(), other.storage_id(), "distinct provenance must not share");

    // Teardown frees: dropping every owner kills the entry, and a reload
    // allocates fresh storage instead of resurrecting the dead one.
    let old = a.storage_id();
    drop(a);
    drop(b);
    let again = shared_weight_buffer(key("w_share"), &bytes);
    assert_ne!(again.storage_id(), old);
}

/// A weight's storage goes back to the device when its last owner drops, not
/// into the allocator's reuse pool: there, a model cast to f16 would keep every
/// f32 checkpoint weight until some allocation failed.
#[test]
fn a_dropped_weight_is_not_pooled_for_reuse() {
    // A size no other test allocates, so only the weight can fill its pool slot.
    let bytes = [5u8; 4_099];
    let key = WeightKey { dtype: DType::UInt8, shape: vec![bytes.len()], ..key("w_unpooled") };
    drop(shared_weight_buffer(key, &bytes));

    let allocator = svod_device::registry::cpu().unwrap();
    let next = Buffer::new(allocator, DType::UInt8, vec![bytes.len()], Default::default());
    let mut out = [0u8; 4_099];
    next.copyout(&mut out).unwrap();
    assert_eq!(out, [0u8; 4_099], "a pooled weight is handed out again, contents and all");
}

#[test]
fn shared_weight_refuses_host_writes() {
    let bytes = [2u8; 16];
    let buffer = shared_weight_buffer(key("w_frozen"), &bytes);
    let mut handle = (*buffer).clone();
    assert!(handle.copyin(&[0u8; 16]).is_err(), "copyin into sealed storage must fail");
    assert!(handle.as_array_mut::<f32>().is_err(), "mutable view of sealed storage must fail");
    let mut out = [0u8; 16];
    buffer.copyout(&mut out).unwrap();
    assert_eq!(out, [2u8; 16], "reads stay legal and correct");
}
