//! Which fills and stores one bulk copy moves ([`crate::lower::bulk`]).

use test_case::test_case;

use super::targets::{families, group_families, hopper};
use crate::atoms::Core;
use crate::ir::{Stmt, Tier};
use crate::lower::bulk::{NotBulk, frame, stored};

/// A fill's boxes, `(panels, [columns, rows])`.
type Boxes = (usize, [u32; 2]);

/// A GEMM's operands are a box each (a row past 128 bytes, a box a panel),
/// whatever rows of the tile lie past the matrix; a convolution's gathered
/// rows and an attention's keys under loaded lengths stay their threads'
/// chunks, as a bulk copy would bring in the rows a chunk fill re-reads or
/// zeroes.
#[test_case("group gemm [128, 256, 64] [8, 1]" => vec![Ok((1, [64, 128])), Ok((1, [64, 256]))]; "a warpgroup gemm")]
#[test_case("group gemm [64, 64, 16] [4, 1]" => vec![Ok((1, [16, 64]))]; "a narrow row")]
#[test_case("gemm one slot" => matches boxes if boxes.iter().all(Result::is_ok); "a warp-level gemm")]
#[test_case("conv 3x3" => matches boxes if boxes.contains(&Err(NotBulk::Gathered)); "a gathered operand")]
#[test_case("attention d64 cache" => vec![Err(NotBulk::LoadedBound)]; "keys under loaded lengths")]
fn a_bulk_copy_moves_plain_matrix_tiles(name: &str) -> Vec<Result<Boxes, NotBulk>> {
    let target = hopper();
    let launches = families(&target).into_iter().chain(group_families(&target));
    let (_, prog, _) = launches.into_iter().find(|(family, ..)| family.starts_with(name)).expect("a family");
    let fill = |stmt: &Stmt| match stmt {
        Stmt::Copy { dst, src, .. }
            if (prog.value(*dst).tier(), prog.value(*src).tier()) == (Tier::Smem, Tier::Global) =>
        {
            Some(frame(&prog, Core::Group, *dst, *src).map(|f| (f.panels, f.map.tile)))
        }
        _ => None,
    };
    let mut fills: Vec<_> = prog.walk().filter_map(|(_, stmt)| fill(stmt)).collect();
    fills.dedup();
    fills
}

/// A tile stored in bulk leaves its ring as panels through an unbounded
/// window of the output; the bounded window an edge tile stores through is
/// none a bulk copy takes, which cuts a tile at the parameter's end alone.
#[test]
fn a_bulk_store_takes_an_unbounded_window() {
    let target = hopper();
    let flushed = |(name, ..): &(String, _, _)| name.ends_with("[128, 128, 64] [8, 1] flushed");
    let (_, prog, _) = group_families(&target).into_iter().find(flushed).expect("a family");
    let prog = &prog;
    let copies = |to: Tier, from: Tier| -> Vec<_> {
        let copy = |stmt: &Stmt| match stmt {
            Stmt::Copy { dst, src, .. } if (prog.value(*dst).tier(), prog.value(*src).tier()) == (to, from) => {
                Some((*dst, *src))
            }
            _ => None,
        };
        prog.walk().filter_map(|(_, stmt)| copy(stmt)).collect()
    };
    let [(whole, scratch)] = copies(Tier::Global, Tier::Smem)[..] else { panic!("one bulk store") };
    let boxes = stored(prog, Core::Group, whole, scratch).map(|f| (f.panels, f.map.tile));
    assert_eq!(boxes, Ok((2, [64, 128])));
    let (bounded, _) = copies(Tier::Global, Tier::Reg)[0];
    assert_eq!(stored(prog, Core::Group, bounded, scratch), Err(NotBulk::RowBound));
}
