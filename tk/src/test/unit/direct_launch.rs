//! The direct launch path (`run_kernel` / `compile_kernel`) binds no runtime
//! variables, so a variable grid extent is refused before anything compiles.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::UOp;
use svod_tensor::Tensor;

use crate::index::cidx;
use crate::launch::{Error, compile_kernel};

#[test]
fn direct_launch_refuses_a_variable_grid_extent() {
    let a = Tensor::from_slice(vec![1.0f32; 8]);
    let mut out = Tensor::empty(&[8], DType::Float32);
    let batch = UOp::define_var("b".to_string(), 1, 4);
    let grid = crate::Grid([cidx(1), cidx(1), batch]);
    let result = compile_kernel("direct", grid, 64, &mut [&mut out], &[&a], |_| UOp::sink(Vec::<Arc<UOp>>::new()));
    match result {
        Err(Error::SymbolicGridExtent { axis: 2, .. }) => {}
        Err(err) => panic!("unexpected error: {err}"),
        Ok(_) => panic!("a variable grid extent compiled on the direct path"),
    }
}
