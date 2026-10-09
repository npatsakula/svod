#[path = "/tmp/claude-1000/-home-mrpink-projects-svod/468f0265-6e57-433e-a55d-49d355110355/scratchpad/old_config.rs"]
#[allow(dead_code)]
mod old;

use crate::atoms::{Target, sm86};
use crate::ops::config as new;
use super::targets::hopper;

#[test]
fn cuda_candidates_are_unchanged() {
    let mut n = 0;
    for target in [sm86(), hopper(), Target { smem_bytes: 48 << 10, ..sm86() }, Target { sms: None, ..sm86() }] {
        for &m in &[1usize, 7, 64, 100, 704, 1000, 1500, 4096, 16384] {
            for &nn in &[8usize, 64, 96, 512, 1000, 2048, 4096] {
                for &k in &[16usize, 48, 64, 96, 384, 512, 4096] {
                    for b in [1usize, 4] {
                        for gated in [false, true] {
                            assert_eq!(old::gemm_candidates(&target, b, m, nn, k, gated), new::gemm_candidates(&target, b, m, nn, k, gated));
                            n += 1;
                        }
                    }
                }
            }
        }
        for &d in &[16usize, 48, 64, 96, 128, 256] {
            for &t in &[1usize, 16, 17, 1500] {
                assert_eq!(old::attention_candidates(&target, d, t), new::attention_candidates(&target, d, t));
            }
        }
        for hw in [5usize, 10, 20, 40, 80, 160] {
            for cin in [16usize, 32, 48, 64, 96, 128, 192, 384, 768] {
                for cout in [8usize, 48, 64, 96, 192, 384] {
                    for (k, s, p) in [(1, 1, 0), (3, 1, 1), (3, 2, 1), (5, 1, 2)] {
                        for b in [1usize, 2] {
                            let g = super::conv::geom([hw, hw], cin, cout, k, s, p, 1);
                            let [ho, wo] = g.out_hw();
                            assert_eq!(old::conv_candidates(&target, b, ho * wo, &g), new::conv_candidates(&target, b, ho * wo, &g), "{g:?}");
                            n += 1;
                        }
                    }
                }
            }
        }
    }
    eprintln!("{n} lists compared");
}

#[test]
fn dump_staged_layouts() {
    use crate::ir::*;
    let target = super::targets::rdna(svod_dtype::AmdArch::Gfx1201, 64);
    for (name, prog, lowering) in super::targets::families(&target).into_iter().take(1) {
        let params = prog.params.iter().enumerate().map(|(i, p)| svod_ir::UOp::param(i, p.elems, svod_dtype::DType::Scalar(p.dtype), None)).collect();
        let l = crate::lower::lower(prog, &lowering, params, svod_dtype::DeviceSpec::Amd { device_id: 0 }).unwrap();
        for (_, s) in l.tile.walk() {
            if let Stmt::Copy { dst, src, mode } = s {
                let d = l.tile.value(*dst);
                let lay = l.layouts[dst.index()].as_ref().or(l.layouts[src.index()].as_ref());
                eprintln!("{name}: {mode:?} {:?} {:?} -> {:?}\n   {:?}", l.tile.value(*src).tier(), d.shape, d.tier(), lay);
            }
        }
    }
}

#[test]
fn dump_ir() {
    for (tag, target) in [("gfx1201", super::targets::rdna(svod_dtype::AmdArch::Gfx1201, 64)), ("gfx1151", super::targets::rdna(svod_dtype::AmdArch::Gfx1151, 40)), ("sm90", hopper())] {
        for (i, (name, prog, lowering)) in super::targets::families(&target).into_iter().enumerate() {
            let ir = super::targets::render(prog, &lowering);
            std::fs::write(format!("/tmp/claude-1000/-home-mrpink-projects-svod/468f0265-6e57-433e-a55d-49d355110355/scratchpad/ir/{tag}_{i:02}_{}.ll", name.replace([' ', '[', ']', ','], "_")), ir).unwrap();
        }
    }
}
