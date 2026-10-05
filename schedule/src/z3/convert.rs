//! UOp → Z3 conversion.
//!
//! Converts Svod IR (UOps) to Z3 expressions for verification.
//! Uses the z3 crate's (0.21) implicit thread-local context.

use std::collections::HashMap;
use std::sync::Arc;

use snafu::{OptionExt, Snafu};
use svod_dtype::DType;
use svod_ir::types::{BinaryOp, ConstValue, TernaryOp, UnaryOp};
use svod_ir::{Op, UOp, UOpKey};
use z3::Solver;
use z3::ast::{Bool, Dynamic, Int};

use crate::z3::alu::{z3_cdiv, z3_cmod};
use svod_ir::ops;

/// Z3 conversion context with solver.
pub struct Z3Context {
    solver: Solver,
    /// Conversion memo, keyed by node identity and shared across every
    /// `convert_uop` call on this context.
    ///
    /// It must outlive the single call: a RANGE and a narrowing CAST each mint an
    /// unconstrained `fresh_const`, so converting the two sides of an equivalence
    /// separately would give the same node two unrelated Z3 variables and refute
    /// even `e == e`. The key holds an `Arc`, so an address cannot be reused.
    cache: HashMap<UOpKey, Dynamic>,
}

impl Z3Context {
    /// Create a new Z3 context with a solver.
    pub fn new() -> Self {
        let solver = Solver::new();
        Self { solver, cache: HashMap::new() }
    }

    /// Get mutable reference to the solver.
    pub fn solver(&mut self) -> &mut Solver {
        &mut self.solver
    }

    /// Convert a UOp graph to Z3 expression.
    ///
    /// Processes UOps in topological order (bottom-up) to ensure dependencies
    /// are converted before they're used.
    pub fn convert_uop(&mut self, uop: &Arc<UOp>) -> Result<Dynamic, ConversionError> {
        self.convert_uop_cached(uop)
    }

    /// Convert UOp with caching to avoid redundant conversion.
    fn convert_uop_cached(&mut self, uop: &Arc<UOp>) -> Result<Dynamic, ConversionError> {
        let key = UOpKey(uop.clone());
        if let Some(z3_expr) = self.cache.get(&key) {
            return Ok(z3_expr.clone());
        }

        // Convert based on operation type
        let z3_expr = match uop.op() {
            Op::Const(cv) => Self::convert_const(&cv.0)?,

            Op::DefineVar(ops::DefineVar { name, min_val, max_val }) => self.convert_var(name, *min_val, *max_val)?,

            Op::Range(ops::Range { end, .. }) => {
                // Range represents loop variable: [0, end)
                let end_z3 = self.convert_uop_cached(end)?;

                // Create a fresh variable for this range
                let range_var = Int::fresh_const("range");
                let zero = Int::from_i64(0);

                // Add constraints: 0 <= range_var < end
                self.solver.assert(range_var.ge(&zero));
                if let Some(end_int) = end_z3.as_int() {
                    self.solver.assert(range_var.lt(end_int));
                }

                Dynamic::from_ast(&range_var)
            }

            Op::Unary(op, src) => {
                let src_z3 = self.convert_uop_cached(src)?;
                Self::convert_unary(*op, &src_z3)?
            }

            Op::Binary(op, lhs, rhs) => {
                let lhs_z3 = self.convert_uop_cached(lhs)?;
                let rhs_z3 = self.convert_uop_cached(rhs)?;
                Self::convert_binary(*op, &lhs_z3, &rhs_z3)?
            }

            Op::Ternary(TernaryOp::Where, cond, true_val, false_val) => {
                let cond_z3 = self.convert_uop_cached(cond)?;
                let true_z3 = self.convert_uop_cached(true_val)?;
                let false_z3 = self.convert_uop_cached(false_val)?;

                if let Some(cond_bool) = cond_z3.as_bool() {
                    if let (Some(true_int), Some(false_int)) = (true_z3.as_int(), false_z3.as_int()) {
                        Dynamic::from_ast(&cond_bool.ite(&true_int, &false_int))
                    } else {
                        return UnsupportedOperationSnafu { detail: "WHERE with non-integer branches" }.fail();
                    }
                } else {
                    return UnsupportedOperationSnafu { detail: "WHERE with non-boolean condition" }.fail();
                }
            }

            Op::Ternary(TernaryOp::MulAcc, a, b, c) => {
                let a_z3 = self.convert_uop_cached(a)?;
                let b_z3 = self.convert_uop_cached(b)?;
                let c_z3 = self.convert_uop_cached(c)?;

                if let (Some(a_int), Some(b_int), Some(c_int)) = (a_z3.as_int(), b_z3.as_int(), c_z3.as_int()) {
                    Dynamic::from_ast(&(a_int * b_int + c_int))
                } else {
                    return UnsupportedOperationSnafu { detail: "MULACC with non-integer operands" }.fail();
                }
            }

            Op::Cast(ops::Cast { src, dtype }) => {
                // Bind the cast result to the source expression so distinct casts of
                // distinct sources stay distinct in the solver. Without an equality
                // constraint, two `Op::Cast` nodes both produce fresh unconstrained
                // vars and z3 can conclude they're equivalent when they aren't.
                //
                // The equality is only sound when the source's static range fits
                // inside the destination dtype's range — for narrowing casts that
                // wrap or truncate, asserting equality could make the solver
                // globally UNSAT and falsely "verify" arbitrary equivalences. Fall
                // back to a fresh bounded var in that case.
                let src_z3 = self.convert_uop_cached(src)?;
                let cast_z3 = self.convert_bounded_from_dtype(dtype.clone())?;
                let (dst_min, dst_max) = dtype_bounds(dtype.clone());
                let src_fits = match (const_value_to_i64(src.vmin()), const_value_to_i64(src.vmax())) {
                    (Some(lo), Some(hi)) => lo >= dst_min && hi <= dst_max,
                    _ => false,
                };
                if src_fits && let (Some(s_int), Some(c_int)) = (src_z3.as_int(), cast_z3.as_int()) {
                    self.solver.assert(c_int.eq(&s_int));
                }
                cast_z3
            }

            _ => {
                return UnsupportedOpSnafu { op: uop.op().as_ref().to_string() }.fail();
            }
        };

        self.cache.insert(key, z3_expr.clone());
        Ok(z3_expr)
    }

    /// Convert a constant value to Z3.
    fn convert_const(cv: &ConstValue) -> Result<Dynamic, ConversionError> {
        match cv {
            ConstValue::Int(v) => Ok(Dynamic::from_ast(&Int::from_i64(*v))),
            ConstValue::UInt(v) => {
                // Represent as signed int; may overflow for very large u64
                Ok(Dynamic::from_ast(&Int::from_u64(*v)))
            }
            ConstValue::Bool(v) => Ok(Dynamic::from_ast(&Bool::from_bool(*v))),
            ConstValue::Float(_) => UnsupportedTypeSnafu { detail: "Float constants not fully supported" }.fail(),
            // The validity marker has no arithmetic value to encode.
            ConstValue::Invalid => UnsupportedTypeSnafu { detail: "Invalid marker has no Z3 encoding" }.fail(),
        }
    }

    /// Convert a variable with bounds to Z3.
    fn convert_var(&mut self, name: &str, min_val: i64, max_val: i64) -> Result<Dynamic, ConversionError> {
        let var = Int::new_const(name);
        let min_z3 = Int::from_i64(min_val);
        let max_z3 = Int::from_i64(max_val);

        // Add constraints: min_val <= var <= max_val
        self.solver.assert(var.ge(&min_z3));
        self.solver.assert(var.le(&max_z3));

        Ok(Dynamic::from_ast(&var))
    }

    /// Create a fresh bounded variable from dtype.
    fn convert_bounded_from_dtype(&mut self, dtype: DType) -> Result<Dynamic, ConversionError> {
        let (min_val, max_val) = dtype_bounds(dtype);
        let var = Int::fresh_const("cast");
        let min_z3 = Int::from_i64(min_val);
        let max_z3 = Int::from_i64(max_val);

        self.solver.assert(var.ge(&min_z3));
        self.solver.assert(var.le(&max_z3));

        Ok(Dynamic::from_ast(&var))
    }

    /// Convert unary operation.
    fn convert_unary(op: UnaryOp, src: &Dynamic) -> Result<Dynamic, ConversionError> {
        let src_int = src.as_int().context(TypeMismatchSnafu { detail: "Expected int for unary op" })?;

        match op {
            UnaryOp::Neg => Ok(Dynamic::from_ast(&-src_int)),
            _ => UnsupportedUnaryOpSnafu { op: op.as_ref().to_string() }.fail(),
        }
    }

    /// Convert binary operation.
    fn convert_binary(op: BinaryOp, lhs: &Dynamic, rhs: &Dynamic) -> Result<Dynamic, ConversionError> {
        match op {
            // Arithmetic operations (require integers)
            BinaryOp::Add => {
                let l = lhs.as_int().context(TypeMismatchSnafu { detail: "ADD: expected int" })?;
                let r = rhs.as_int().context(TypeMismatchSnafu { detail: "ADD: expected int" })?;
                Ok(Dynamic::from_ast(&(l + r)))
            }
            BinaryOp::Sub => {
                let l = lhs.as_int().context(TypeMismatchSnafu { detail: "SUB: expected int" })?;
                let r = rhs.as_int().context(TypeMismatchSnafu { detail: "SUB: expected int" })?;
                Ok(Dynamic::from_ast(&(l - r)))
            }
            BinaryOp::Mul => {
                let l = lhs.as_int().context(TypeMismatchSnafu { detail: "MUL: expected int" })?;
                let r = rhs.as_int().context(TypeMismatchSnafu { detail: "MUL: expected int" })?;
                Ok(Dynamic::from_ast(&(l * r)))
            }
            BinaryOp::FloorDiv => {
                let l = lhs.as_int().context(TypeMismatchSnafu { detail: "FLOORDIV: expected int" })?;
                let r = rhs.as_int().context(TypeMismatchSnafu { detail: "FLOORDIV: expected int" })?;
                let q = z3_cdiv(&l, &r);
                let rem = z3_cmod(&l, &r);
                let zero = Int::from_i64(0);
                let adjust = Bool::and(&[rem.eq(&zero).not(), (&l * &r).lt(&zero)]);
                Ok(Dynamic::from_ast(&adjust.ite(&(q.clone() - 1), &q)))
            }
            BinaryOp::FloorMod => {
                let l = lhs.as_int().context(TypeMismatchSnafu { detail: "FLOORMOD: expected int" })?;
                let r = rhs.as_int().context(TypeMismatchSnafu { detail: "FLOORMOD: expected int" })?;
                let rem = z3_cmod(&l, &r);
                let zero = Int::from_i64(0);
                let adjust = Bool::and(&[rem.eq(&zero).not(), (&l * &r).lt(&zero)]);
                Ok(Dynamic::from_ast(&(rem + adjust.ite(&r, &zero))))
            }
            BinaryOp::CDiv => {
                let l = lhs.as_int().context(TypeMismatchSnafu { detail: "CDIV: expected int" })?;
                let r = rhs.as_int().context(TypeMismatchSnafu { detail: "CDIV: expected int" })?;
                Ok(Dynamic::from_ast(&z3_cdiv(&l, &r)))
            }
            BinaryOp::CMod => {
                let l = lhs.as_int().context(TypeMismatchSnafu { detail: "CMOD: expected int" })?;
                let r = rhs.as_int().context(TypeMismatchSnafu { detail: "CMOD: expected int" })?;
                Ok(Dynamic::from_ast(&z3_cmod(&l, &r)))
            }
            BinaryOp::Max => {
                let l = lhs.as_int().context(TypeMismatchSnafu { detail: "MAX: expected int" })?;
                let r = rhs.as_int().context(TypeMismatchSnafu { detail: "MAX: expected int" })?;
                // max(a, b) = if a > b then a else b
                Ok(Dynamic::from_ast(&l.gt(&r).ite(&l, &r)))
            }

            // Comparison operations (return boolean)
            BinaryOp::Lt => {
                let l = lhs.as_int().context(TypeMismatchSnafu { detail: "LT: expected int" })?;
                let r = rhs.as_int().context(TypeMismatchSnafu { detail: "LT: expected int" })?;
                Ok(Dynamic::from_ast(&l.lt(r)))
            }
            BinaryOp::Eq => {
                // Try int first, then bool
                if let (Some(l), Some(r)) = (lhs.as_int(), rhs.as_int()) {
                    Ok(Dynamic::from_ast(&l.eq(r)))
                } else if let (Some(l), Some(r)) = (lhs.as_bool(), rhs.as_bool()) {
                    Ok(Dynamic::from_ast(&l.eq(r)))
                } else {
                    TypeMismatchSnafu { detail: "EQ: type mismatch" }.fail()
                }
            }
            BinaryOp::Ne => {
                // Try int first, then bool
                if let (Some(l), Some(r)) = (lhs.as_int(), rhs.as_int()) {
                    Ok(Dynamic::from_ast(&l.eq(r).not()))
                } else if let (Some(l), Some(r)) = (lhs.as_bool(), rhs.as_bool()) {
                    Ok(Dynamic::from_ast(&l.eq(r).not()))
                } else {
                    TypeMismatchSnafu { detail: "NE: type mismatch" }.fail()
                }
            }

            // Bitwise operations
            BinaryOp::And => {
                // Can be int (bitwise) or bool (logical)
                if let (Some(l), Some(r)) = (lhs.as_bool(), rhs.as_bool()) {
                    Ok(Dynamic::from_ast(&Bool::and(&[l, r])))
                } else {
                    UnsupportedOperationSnafu { detail: "Bitwise AND not implemented" }.fail()
                }
            }
            BinaryOp::Or => {
                // Can be int (bitwise) or bool (logical)
                if let (Some(l), Some(r)) = (lhs.as_bool(), rhs.as_bool()) {
                    Ok(Dynamic::from_ast(&Bool::or(&[l, r])))
                } else {
                    UnsupportedOperationSnafu { detail: "Bitwise OR not implemented" }.fail()
                }
            }

            _ => UnsupportedBinaryOpSnafu { op: op.as_ref().to_string() }.fail(),
        }
    }
}

impl Default for Z3Context {
    fn default() -> Self {
        Self::new()
    }
}

/// Best-effort conversion of `vmin`/`vmax` ConstValues to a signed i64 for
/// range-fits comparisons. Returns `None` for floats and out-of-range u64.
fn const_value_to_i64(cv: &ConstValue) -> Option<i64> {
    match cv {
        ConstValue::Int(v) => Some(*v),
        ConstValue::UInt(v) => i64::try_from(*v).ok(),
        ConstValue::Bool(b) => Some(*b as i64),
        ConstValue::Float(_) | ConstValue::Invalid => None,
    }
}

/// Get conservative bounds for a dtype.
fn dtype_bounds(dtype: DType) -> (i64, i64) {
    use svod_dtype::ScalarDType;

    match dtype {
        DType::Scalar(sdt) => match sdt {
            ScalarDType::Bool => (0, 1),
            ScalarDType::Int8 => (i8::MIN as i64, i8::MAX as i64),
            ScalarDType::Int16 => (i16::MIN as i64, i16::MAX as i64),
            ScalarDType::Int32 => (i32::MIN as i64, i32::MAX as i64),
            ScalarDType::Int64 => (i64::MIN, i64::MAX),
            ScalarDType::UInt8 => (0, u8::MAX as i64),
            ScalarDType::UInt16 => (0, u16::MAX as i64),
            ScalarDType::UInt32 => (0, u32::MAX as i64),
            ScalarDType::UInt64 => (0, i64::MAX),    // Conservative
            _ => (i32::MIN as i64, i32::MAX as i64), // Float types
        },
        DType::Ptr { .. } => (0, i64::MAX),
        DType::Vector { scalar, .. } => dtype_bounds(DType::Scalar(scalar)),
        DType::Image { .. } => (0, i64::MAX), // Conservative bounds for image types
    }
}

/// Z3 conversion error.
#[derive(Debug, Clone, Snafu)]
#[snafu(visibility(pub))]
pub enum ConversionError {
    #[snafu(display("Unsupported operation: {detail}"))]
    UnsupportedOperation { detail: &'static str },
    #[snafu(display("Unsupported operation: {op}"))]
    UnsupportedOp { op: String },
    #[snafu(display("Unsupported operation: Unary op: {op}"))]
    UnsupportedUnaryOp { op: String },
    #[snafu(display("Unsupported operation: Binary op: {op}"))]
    UnsupportedBinaryOp { op: String },
    #[snafu(display("Unsupported type: {detail}"))]
    UnsupportedType { detail: &'static str },
    #[snafu(display("Type mismatch: {detail}"))]
    TypeMismatch { detail: &'static str },
}

#[cfg(test)]
#[path = "../test/unit/z3/convert_internal.rs"]
mod tests;
