//! The attention prologue: the program against a direct f64 reference on the
//! host, and the lowered kernel against the program on the GPU.

use svod_dtype::{DType, DeviceSpec, ScalarDType, default_device::default_device};
use svod_tensor::Tensor;
use test_case::test_case;

use crate::atoms::Target;
use crate::build::BF16;
use crate::interp::{round_to, run};
use crate::kernels::Batch;
use crate::kernels::heads::{HeadsSpec, Rope, heads};
use crate::kernels::rows::NormCfg;
use crate::launch::graph_launch_all;

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
}

fn bf16s(n: usize, seed: &mut u64, f: impl Fn(f64) -> f64) -> Vec<f64> {
    (0..n).map(|_| round_to(ScalarDType::BFloat16, f(lcg(seed)))).collect()
}

fn spec(batch: Batch, t: usize, [heads, kv_heads, d]: [usize; 3], norms: [bool; 2], rope: Option<bool>) -> HeadsSpec {
    HeadsSpec {
        batch,
        t,
        heads,
        kv_heads,
        d,
        q_norm: norms[0],
        k_norm: norms[1],
        eps: 1e-6,
        rope: rope.map(|per_batch| Rope { per_batch }),
        cfg: NormCfg { br: 4 },
    }
}

/// Every parameter in order, outputs zeroed.
fn inputs(s: &HeadsSpec, seed: u64) -> Vec<Vec<f64>> {
    let mut seed = seed;
    let (cap, slots, half) = (s.batch.capacity(), s.heads + 2 * s.kv_heads, s.d / 2);
    let mut p = vec![bf16s(cap * s.t * slots * s.d, &mut seed, |x| 3.0 * x)];
    for on in [s.q_norm, s.k_norm] {
        if on {
            p.push(bf16s(s.d, &mut seed, |x| 1.0 + 0.5 * x));
        }
    }
    if let Some(rope) = s.rope {
        let rows = if rope.per_batch { cap * s.t } else { s.t };
        let angles: Vec<f64> = (0..rows * half).map(|e| (e / half) as f64 * 0.01 * (1.0 + (e % half) as f64)).collect();
        p.push(angles.iter().map(|a| round_to(ScalarDType::BFloat16, a.cos())).collect());
        p.push(angles.iter().map(|a| round_to(ScalarDType::BFloat16, a.sin())).collect());
    }
    p.push(vec![0.0; cap * s.t * s.heads * s.d]);
    p.push(vec![0.0; cap * s.t * s.kv_heads * s.d]);
    p.push(vec![0.0; cap * s.t * s.kv_heads * s.d]);
    p
}

/// `(q, k, v)` in f64 from the bf16 inputs, rounded to bf16 at the store.
fn reference(s: &HeadsSpec, p: &[Vec<f64>]) -> [Vec<f64>; 3] {
    let (cap, slots, half) = (s.batch.capacity(), s.heads + 2 * s.kv_heads, s.d / 2);
    let mut at = 1;
    let mut take = |on: bool| {
        at += usize::from(on);
        on.then(|| &p[at - 1])
    };
    let (q_w, k_w) = (take(s.q_norm), take(s.k_norm));
    let (cos, sin) = (take(s.rope.is_some()), take(s.rope.is_some()));
    let mut out = [
        vec![0.0; cap * s.t * s.heads * s.d],
        vec![0.0; cap * s.t * s.kv_heads * s.d],
        vec![0.0; cap * s.t * s.kv_heads * s.d],
    ];
    for b in 0..cap {
        for i in 0..s.t {
            for slot in 0..slots {
                let mut x = p[0][((b * s.t + i) * slots + slot) * s.d..][..s.d].to_vec();
                let (which, head, w) = match slot {
                    j if j < s.heads => (0, j, q_w),
                    j if j < s.heads + s.kv_heads => (1, j - s.heads, k_w),
                    j => (2, j - s.heads - s.kv_heads, None),
                };
                if let Some(w) = w {
                    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / s.d as f64 + s.eps).sqrt();
                    x = x.iter().zip(w).map(|(v, w)| v * inv * w).collect();
                }
                if let (Some(cos), Some(sin), true) = (cos, sin, which < 2) {
                    let row = if s.rope.expect("rotated").per_batch { b * s.t + i } else { i };
                    let rotated: Vec<f64> = (0..s.d)
                        .map(|c| {
                            let (cs, sn) = (cos[row * half + c % half], sin[row * half + c % half]);
                            if c < half { x[c] * cs - x[c + half] * sn } else { x[c - half] * sn + x[c] * cs }
                        })
                        .collect();
                    x = rotated;
                }
                let width = [s.heads, s.kv_heads, s.kv_heads][which];
                let dst = &mut out[which][((b * s.t + i) * width + head) * s.d..][..s.d];
                for (d, v) in dst.iter_mut().zip(&x) {
                    *d = round_to(ScalarDType::BFloat16, *v);
                }
            }
        }
    }
    out
}

fn assert_close(what: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let worst = got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0.0, f64::max);
    assert!(worst <= tol, "{what}: max abs diff {worst}");
}

const STATIC: Batch = Batch::Static(2);
fn var() -> Batch {
    Batch::Var { name: "b".into(), min: 1, max: 2 }
}

#[test_case(spec(STATIC, 9, [4, 2, 64], [true, true], Some(false)); "normed and rotated, shared tables")]
#[test_case(spec(var(), 10, [4, 2, 128], [true, true], Some(true)); "d 128, tables per batch, batch var")]
#[test_case(spec(STATIC, 5, [2, 2, 64], [false, false], Some(false)); "rotated only")]
#[test_case(spec(STATIC, 8, [4, 1, 64], [true, false], None); "q normed only")]
#[test_case(spec(STATIC, 3, [2, 1, 32], [false, false], None); "a split")]
fn program_matches_the_reference(s: HeadsSpec) {
    let p = inputs(&s, 7);
    let want = reference(&s, &p);
    let got = run(&heads::<BF16>(&s), p, &[("b", 2)]).unwrap();
    let outputs = got.len() - 3;
    for (i, name) in ["q", "k", "v"].into_iter().enumerate() {
        assert_close(name, &got[outputs + i], &want[i], 2e-2);
    }
}

/// Rows of a batch past the live bound are never written.
#[test]
fn only_the_live_batch_runs() {
    let s = spec(var(), 6, [2, 2, 64], [true, true], Some(true));
    let p = inputs(&s, 3);
    let got = run(&heads::<BF16>(&s), p, &[("b", 1)]).unwrap();
    let (q, half) = (&got[got.len() - 3], s.t * s.heads * s.d);
    assert!(q[..half].iter().any(|x| *x != 0.0));
    assert!(q[half..].iter().all(|x| *x == 0.0));
}

#[test_case(spec(STATIC, 37, [4, 2, 64], [true, true], Some(false)); "d 64, partial block")]
#[test_case(spec(var(), 40, [8, 2, 128], [true, true], Some(true)); "d 128, tables per batch, batch var")]
#[test_case(spec(STATIC, 16, [2, 2, 64], [false, false], None); "a split")]
fn kernel_matches_the_program(s: HeadsSpec) {
    let device = default_device();
    let Some(target) = matches!(device, DeviceSpec::Cuda { .. }).then(|| Target::for_device(&device)).flatten() else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    let cap = s.batch.capacity();
    let p = inputs(&s, 11);
    let want = run(&heads::<BF16>(&s), p.clone(), &[("b", cap as i64)]).unwrap();
    let tensors: Vec<Tensor> = p
        .iter()
        .map(|v| Tensor::from_slice(v.iter().map(|&x| x as f32).collect::<Vec<_>>()).cast(DType::BFloat16))
        .collect();
    let refs: Vec<&Tensor> = tensors.iter().collect();
    let outs = graph_launch_all(heads::<BF16>(&s), &s.cfg.lowering(target), &refs).unwrap();
    let outputs = outs.len() - 3;
    for (i, name) in ["q", "k", "v"].into_iter().enumerate() {
        let out = &outs[outputs + i];
        let mut plan = out.prepare().unwrap();
        plan.execute_with_vars(&[("b", cap as i64)]).unwrap();
        let mut bytes = vec![0u8; want[outputs + i].len() * 2];
        out.buffer().unwrap().copyout(&mut bytes).unwrap();
        let got: Vec<f64> = bytes
            .chunks(2)
            .map(|b| f64::from(f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)))
            .collect();
        assert_close(name, &got, &want[outputs + i], 1e-2);
    }
}
