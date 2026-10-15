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

/// The registry drops a weight's storage as soon as the weight's BUFFER uop
/// dies, not whenever papaya's collector next retires a batch of removed
/// entries: a thread that stops dropping tensors keeps its last 31 for good,
/// and a model cast to f16 would keep that many f32 checkpoint weights.
#[test]
fn dropping_a_weight_tensor_releases_its_storage() {
    let bytes = [3u8; 16];
    let tensor = crate::Tensor::from_shared_weight(key("w_drop"), &bytes).unwrap();
    let storage = std::sync::Arc::downgrade(&shared_weight_buffer(key("w_drop"), &bytes));
    drop(tensor);
    assert!(storage.upgrade().is_none(), "the storage must die with its last tensor");
}
