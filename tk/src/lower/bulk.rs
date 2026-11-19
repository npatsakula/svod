//! Which fills one bulk copy moves (`cp.async.bulk.tensor`, Hopper's TMA):
//! a whole tile, global → shared, by one thread, addressed through a tensor
//! map of the parameter ([`svod_ir::TensorMap`]) by the tile's column and
//! row in it.

use snafu::Snafu;
use svod_ir::{TensorMap, TensorMapSwizzle};

use crate::atoms::Core;
use crate::ir::{Phase, Place, Program, Scalar, ScalarId, ValId};
use crate::layouts::Swizzle;

/// Why a fill stays its threads' 16-byte chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Snafu)]
pub enum NotBulk {
    /// Its rows are gathered: no two lie a stride apart.
    #[snafu(display("its rows are gathered"))]
    Gathered,
    /// Its columns repeat or are a stride apart.
    #[snafu(display("its columns are not adjacent"))]
    Columns,
    /// A column bound, past which the chunks land as zeros.
    #[snafu(display("it has a column bound"))]
    ColumnBound,
    /// A row bound read from a parameter: the rows past it are data a bulk
    /// copy would bring in, where the chunks re-read the last valid row.
    #[snafu(display("its row bound is loaded"))]
    LoadedBound,
    /// The rows' stride is not a constant, or no multiple of 16 bytes that
    /// divides the parameter.
    #[snafu(display("its rows are no constant whole 16-byte chunks apart"))]
    Stride,
    /// The tile is no box a tensor map takes: over 256 rows, or rows that
    /// are neither one swizzle span nor panels of the widest.
    #[snafu(display("its tile is no tensor-map box"))]
    Tile,
}

/// A bulk fill: `panels` boxes of `map`'s tile side by side, the rows of
/// the parameter `row` elements apart.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub map: TensorMap,
    pub row: usize,
    pub panels: usize,
}

impl Frame {
    /// The bytes the fill moves.
    pub fn bytes(&self) -> u32 {
        self.panels as u32 * self.map.tile[0] * self.map.tile[1] * self.map.dtype.bytes() as u32
    }
}

/// The bulk fill of shared tile `dst` from global view `src` on `core`,
/// whose shared rows past the widest swizzle span are panels of it.
pub fn frame<P: Phase>(prog: &Program<P>, core: Core, dst: ValId, src: ValId) -> Result<Frame, NotBulk> {
    let (d, s) = (prog.value(dst), prog.value(src));
    let Place::Global { param, stride, bounds, rows, period, .. } = &s.place else { unreachable!("a fill's source") };
    snafu::ensure!(rows.is_none(), GatheredSnafu);
    let constant = |id: ScalarId| match prog.scalar(id) {
        Scalar::Const(v) => usize::try_from(*v).ok(),
        _ => None,
    };
    snafu::ensure!(period.is_none() && constant(stride[1]) == Some(1), ColumnsSnafu);
    snafu::ensure!(bounds[1].is_none(), ColumnBoundSnafu);
    snafu::ensure!(bounds[0].is_none_or(|bound| !loaded(prog, bound)), LoadedBoundSnafu);
    let (bytes, elems) = (s.dtype.bytes(), prog.params[param.index()].elems);
    let row = constant(stride[0]).filter(|row| *row > 0 && elems.is_multiple_of(*row)).ok_or(NotBulk::Stride)?;
    let cols = d.shape.cols;
    let span = Swizzle::B128.span();
    let panel = if core == Core::Group && cols * bytes > span && (cols * bytes).is_multiple_of(span) {
        span / bytes
    } else {
        cols
    };
    let swizzle = match Swizzle::of_row(panel, bytes) {
        Swizzle::None => TensorMapSwizzle::None,
        Swizzle::B32 => TensorMapSwizzle::B32,
        Swizzle::B64 => TensorMapSwizzle::B64,
        Swizzle::B128 => TensorMapSwizzle::B128,
    };
    let extent = [row as u64, (elems / row) as u64];
    let map =
        TensorMap::new(param.index(), s.dtype, extent, [panel as u32, d.shape.rows as u32], swizzle).map_err(|e| {
            match e {
                svod_ir::Error::TensorMapRowStride { .. } | svod_ir::Error::TensorMapExtent { .. } => NotBulk::Stride,
                _ => NotBulk::Tile,
            }
        })?;
    Ok(Frame { map, row, panels: cols / panel })
}

/// Whether scalar `id` reads a parameter.
fn loaded<P: Phase>(prog: &Program<P>, id: ScalarId) -> bool {
    match prog.scalar(id) {
        Scalar::Load { .. } => true,
        Scalar::Bin(_, a, b) => loaded(prog, *a) || loaded(prog, *b),
        _ => false,
    }
}
