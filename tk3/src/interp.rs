//! Host execution of a tile program at the logical level: tiles are matrices,
//! every block runs its statements in order, a pipeline is its serial
//! interleaving. Values are rounded to their element type, so the result is
//! what a correct lowering must reproduce up to accumulation order.

use std::collections::HashMap;

use snafu::{Snafu, ensure};
use svod_dtype::ScalarDType;

use crate::ir::*;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("variable {name} is not bound"))]
    UnboundVar { name: String },
    #[snafu(display("parameter {name} has {got} elements, the program declares {want}"))]
    ParamSize { name: String, got: usize, want: usize },
    #[snafu(display("raw statements cannot be interpreted: {code}"))]
    Raw { code: String },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Round `x` to `dtype`'s precision (round to nearest even).
pub fn round_to(dtype: ScalarDType, x: f64) -> f64 {
    match dtype {
        ScalarDType::Float32 => x as f32 as f64,
        ScalarDType::BFloat16 => {
            let bits = (x as f32).to_bits();
            let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
            f32::from_bits(if (x as f32).is_nan() { bits | 0x0040_0000 } else { rounded }) as f64
        }
        ScalarDType::Float16 => f16_round(x as f32) as f64,
        ScalarDType::Int32 | ScalarDType::Int64 | ScalarDType::Int8 | ScalarDType::Int16 => x.trunc(),
        ScalarDType::Bool => f64::from(x != 0.0),
        other => unimplemented!("rounding to {other:?}"),
    }
}

fn f16_round(x: f32) -> f32 {
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    if x.is_nan() || x.is_infinite() {
        return x;
    }
    if exp > 15 {
        return f32::from_bits((bits & 0x8000_0000) | 0x7f80_0000);
    }
    // Keep 10 explicit mantissa bits for normals; fewer for subnormals.
    let drop = if exp >= -14 { 13 } else { (13 + (-14 - exp)).min(24) };
    let mask = (1u32 << drop) - 1;
    let half = 1u32 << (drop - 1);
    let rem = bits & mask;
    let mut out = bits & !mask;
    if rem > half || (rem == half && (out >> drop) & 1 == 1) {
        out += 1 << drop;
    }
    f32::from_bits(out)
}

struct Frame<'p> {
    prog: &'p Program,
    globals: Vec<Vec<f64>>,
    smem: Vec<Vec<f64>>,
    regs: HashMap<ValId, Vec<f64>>,
    induction: HashMap<ScalarId, i64>,
    vars: HashMap<String, i64>,
    block: [i64; 3],
    warp: i64,
}

/// Run `prog` over its whole grid. `params` are the parameter buffers in
/// declaration order (outputs included); `vars` bind the symbolic variables.
/// Returns the buffers after every block ran.
pub fn run(prog: &Program, params: Vec<Vec<f64>>, vars: &[(&str, i64)]) -> Result<Vec<Vec<f64>>> {
    for (p, buf) in prog.params.iter().zip(&params) {
        ensure!(buf.len() == p.elems, ParamSizeSnafu { name: p.name.clone(), got: buf.len(), want: p.elems });
    }
    let mut frame = Frame {
        prog,
        globals: params,
        smem: prog.smem.iter().map(|s| vec![0.0; s.elems]).collect(),
        regs: HashMap::new(),
        induction: HashMap::new(),
        vars: vars.iter().map(|(n, v)| (n.to_string(), *v)).collect(),
        block: [0; 3],
        warp: 0,
    };
    for var in &prog.vars {
        ensure!(frame.vars.contains_key(&var.name), UnboundVarSnafu { name: var.name.clone() });
    }
    let grid = prog.grid.map(|g| frame.scalar(g).unwrap_or(1));
    for z in 0..grid[2] {
        for y in 0..grid[1] {
            for x in 0..grid[0] {
                frame.block = [x, y, z];
                frame.regs.clear();
                frame.block_run(&prog.body)?;
            }
        }
    }
    Ok(frame.globals)
}

impl Frame<'_> {
    fn scalar(&self, id: ScalarId) -> Option<i64> {
        Some(match self.prog.scalar(id) {
            Scalar::Const(v) => *v,
            Scalar::Var(name) => *self.vars.get(name)?,
            Scalar::Special(Special::Block(axis)) => self.block[*axis as usize],
            Scalar::Special(Special::Warp) => self.warp,
            Scalar::Induction => *self.induction.get(&id)?,
            Scalar::Bin(op, a, b) => {
                let (a, b) = (self.scalar(*a)?, self.scalar(*b)?);
                match op {
                    BinOp::Add => a + b,
                    BinOp::Sub => a - b,
                    BinOp::Mul => a * b,
                    BinOp::Div => a.div_euclid(b),
                    BinOp::Rem => a.rem_euclid(b),
                    BinOp::Min => a.min(b),
                    BinOp::Max => a.max(b),
                    BinOp::Lt => i64::from(a < b),
                    BinOp::Le => i64::from(a <= b),
                    BinOp::Eq => i64::from(a == b),
                    BinOp::And => i64::from(a != 0 && b != 0),
                    BinOp::Or => i64::from(a != 0 || b != 0),
                }
            }
            Scalar::Load { param, index } => {
                let index = self.scalar(*index)?;
                self.globals[param.index()].get(index as usize).copied().unwrap_or(0.0) as i64
            }
        })
    }

    fn must(&self, id: ScalarId) -> i64 {
        self.scalar(id).expect("every scalar of a running block is defined")
    }

    fn block_run(&mut self, block: &Block) -> Result<()> {
        for stmt in &block.0 {
            self.stmt(stmt)?;
        }
        Ok(())
    }

    fn stmt(&mut self, stmt: &Stmt) -> Result<()> {
        match stmt {
            Stmt::Let { dst, op } => {
                let out = self.tile_op(*dst, op);
                self.regs.insert(*dst, out);
            }
            Stmt::Copy { dst, src, .. } => {
                let data = self.read(*src);
                self.write(*dst, &data);
            }
            Stmt::Loop(l) => {
                let extent = self.must(l.extent);
                self.enter(&l.carried);
                for i in 0..extent {
                    self.induction.insert(l.iv, i);
                    self.block_run(&l.body)?;
                    self.advance(&l.carried);
                }
            }
            Stmt::Pipeline(p) => {
                let extent = self.must(p.extent);
                self.enter(&p.carried);
                for i in 0..extent {
                    for stage in [&p.produce, &p.consume] {
                        self.induction.insert(stage.step, i);
                        self.induction.insert(stage.slot, i % p.stages as i64);
                        self.block_run(&stage.body)?;
                    }
                    self.advance(&p.carried);
                }
            }
            Stmt::Role { role, body } => {
                let warps = self.prog.roles[role.index()].warps.clone();
                let saved = self.warp;
                self.warp = i64::from(warps.start);
                self.block_run(body)?;
                self.warp = saved;
            }
            Stmt::If { pred, then, otherwise } => {
                let branch = if self.must(*pred) != 0 { then } else { otherwise };
                self.block_run(branch)?;
            }
            Stmt::Sync(_) => {}
            Stmt::Raw(raw) => return RawSnafu { code: raw.code.clone() }.fail(),
        }
        Ok(())
    }

    fn enter(&mut self, carried: &[Carried]) {
        for c in carried {
            let init = self.regs[&c.init].clone();
            self.regs.insert(c.phi, init);
        }
    }

    /// Carried registers take their `next` where the iteration computed one
    /// (a guarded body may skip it, leaving the register as it was).
    fn advance(&mut self, carried: &[Carried]) {
        for c in carried {
            if let Some(next) = self.regs.remove(&c.next) {
                self.regs.insert(c.phi, next);
            }
        }
    }

    /// Element coordinates → flat index of a view, or `None` out of bounds.
    fn address(&self, place: &Place, r: usize, c: usize, cols: usize) -> Option<usize> {
        match place {
            Place::Reg => Some(r * cols + c),
            Place::Smem { offset, .. } => Some(self.must(*offset) as usize + r * cols + c),
            Place::Global { offset, stride, bounds, .. } => {
                let inside = |b: Option<ScalarId>, i: usize| b.is_none_or(|b| (i as i64) < self.must(b));
                (inside(bounds[0], r) && inside(bounds[1], c)).then(|| {
                    (self.must(*offset) + r as i64 * self.must(stride[0]) + c as i64 * self.must(stride[1])) as usize
                })
            }
        }
    }

    fn read(&self, id: ValId) -> Vec<f64> {
        let v = self.prog.value(id);
        let Shape { rows, cols } = v.shape;
        match &v.place {
            Place::Reg => self.regs[&id].clone(),
            Place::Smem { .. } | Place::Global { .. } => {
                let store = match v.place {
                    Place::Smem { alloc, .. } => &self.smem[alloc.index()],
                    Place::Global { param, .. } => &self.globals[param.index()],
                    Place::Reg => unreachable!(),
                };
                let mut out = vec![0.0; rows * cols];
                for r in 0..rows {
                    for c in 0..cols {
                        if let Some(i) = self.address(&v.place, r, c, cols) {
                            out[r * cols + c] = store.get(i).copied().unwrap_or(0.0);
                        }
                    }
                }
                out
            }
        }
    }

    fn write(&mut self, id: ValId, data: &[f64]) {
        let v = self.prog.value(id).clone();
        let Shape { rows, cols } = v.shape;
        let rounded: Vec<f64> = data.iter().map(|&x| round_to(v.dtype, x)).collect();
        match &v.place {
            Place::Reg => {
                self.regs.insert(id, rounded);
            }
            Place::Smem { .. } | Place::Global { .. } => {
                let mut writes = Vec::with_capacity(rows * cols);
                for r in 0..rows {
                    for c in 0..cols {
                        if let Some(i) = self.address(&v.place, r, c, cols) {
                            writes.push((i, rounded[r * cols + c]));
                        }
                    }
                }
                let store = match v.place {
                    Place::Smem { alloc, .. } => &mut self.smem[alloc.index()],
                    Place::Global { param, .. } => &mut self.globals[param.index()],
                    Place::Reg => unreachable!(),
                };
                for (i, x) in writes {
                    if let Some(slot) = store.get_mut(i) {
                        *slot = x;
                    }
                }
            }
        }
    }

    fn tile_op(&self, dst: ValId, op: &TileOp) -> Vec<f64> {
        let out = self.prog.value(dst);
        let Shape { rows, cols } = out.shape;
        let data = match op {
            TileOp::Fill(c) => {
                let v = match c {
                    Const::Int(i) => *i as f64,
                    Const::Float(f) => *f,
                };
                vec![v; rows * cols]
            }
            TileOp::Coord(axis) => (0..rows * cols)
                .map(|i| match axis {
                    Axis::Row => (i / cols) as f64,
                    Axis::Col => (i % cols) as f64,
                })
                .collect(),
            TileOp::Unary { src, f } => self
                .read(*src)
                .into_iter()
                .map(|x| match f {
                    UnaryOp::Neg => -x,
                    UnaryOp::Exp2 => x.exp2(),
                    UnaryOp::Log2 => x.log2(),
                    UnaryOp::Recip => 1.0 / x,
                    UnaryOp::Sqrt => x.sqrt(),
                    UnaryOp::Rsqrt => 1.0 / x.sqrt(),
                    UnaryOp::Abs => x.abs(),
                    UnaryOp::Not => f64::from(x == 0.0),
                })
                .collect(),
            TileOp::Binary { a, b, f } => {
                let (va, vb) = (self.broadcast(*a, out.shape), self.broadcast(*b, out.shape));
                va.iter()
                    .zip(&vb)
                    .map(|(&x, &y)| match f {
                        BinaryOp::Add => x + y,
                        BinaryOp::Sub => x - y,
                        BinaryOp::Mul => x * y,
                        BinaryOp::Div => x / y,
                        BinaryOp::Max => x.max(y),
                        BinaryOp::Min => x.min(y),
                        BinaryOp::Lt => f64::from(x < y),
                        BinaryOp::Le => f64::from(x <= y),
                        BinaryOp::Eq => f64::from(x == y),
                        BinaryOp::Ne => f64::from(x != y),
                        BinaryOp::And => f64::from(x != 0.0 && y != 0.0),
                        BinaryOp::Or => f64::from(x != 0.0 || y != 0.0),
                    })
                    .collect()
            }
            TileOp::Cast { src, .. } | TileOp::Relayout { src } => self.read(*src),
            TileOp::Mma { acc, a, b, a_t, b_t } => {
                let (va, vb) = (self.read(*a), self.read(*b));
                let (sa, sb) = (self.prog.value(*a).shape, self.prog.value(*b).shape);
                let k = if *a_t { sa.rows } else { sa.cols };
                let at = |m: usize, kk: usize| if *a_t { va[kk * sa.cols + m] } else { va[m * sa.cols + kk] };
                let bt = |kk: usize, n: usize| if *b_t { vb[n * sb.cols + kk] } else { vb[kk * sb.cols + n] };
                let mut out = self.read(*acc);
                for m in 0..rows {
                    for n in 0..cols {
                        let mut sum = 0.0;
                        for kk in 0..k {
                            sum += at(m, kk) * bt(kk, n);
                        }
                        out[m * cols + n] = (out[m * cols + n] as f32 + sum as f32) as f64;
                    }
                }
                out
            }
            TileOp::Reduce { src, axis, f } => {
                let v = self.read(*src);
                let s = self.prog.value(*src).shape;
                let fold = |acc: f64, x: f64| match f {
                    ReduceOp::Sum => acc + x,
                    ReduceOp::Max => acc.max(x),
                    ReduceOp::Min => acc.min(x),
                };
                let init = match f {
                    ReduceOp::Sum => 0.0,
                    ReduceOp::Max => f64::NEG_INFINITY,
                    ReduceOp::Min => f64::INFINITY,
                };
                match axis {
                    Axis::Row => (0..s.rows)
                        .map(|r| v[r * s.cols..(r + 1) * s.cols].iter().fold(init, |a, &x| fold(a, x)))
                        .collect(),
                    Axis::Col => (0..s.cols).map(|c| (0..s.rows).map(|r| v[r * s.cols + c]).fold(init, fold)).collect(),
                }
            }
            TileOp::Where { pred, a, b } => {
                let p = self.read(*pred);
                let (va, vb) = (self.broadcast(*a, out.shape), self.broadcast(*b, out.shape));
                p.iter().zip(va.iter().zip(&vb)).map(|(&p, (&x, &y))| if p != 0.0 { x } else { y }).collect()
            }
            TileOp::Transpose { src } => {
                let v = self.read(*src);
                (0..rows * cols).map(|i| v[(i % cols) * rows + i / cols]).collect()
            }
        };
        data.into_iter().map(|x| round_to(out.dtype, x)).collect()
    }

    fn broadcast(&self, id: ValId, to: Shape) -> Vec<f64> {
        let v = self.read(id);
        let s = self.prog.value(id).shape;
        if s == to {
            return v;
        }
        (0..to.rows * to.cols)
            .map(|i| {
                let (r, c) = (i / to.cols, i % to.cols);
                v[(if s.rows == 1 { 0 } else { r }) * s.cols + if s.cols == 1 { 0 } else { c }]
            })
            .collect()
    }
}
