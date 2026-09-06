//! Single-warp cross-lane shuffle ops built on the arch-lowered gather primitive
//! `shuffle_lane` (`ds_bpermute` on AMD, `shfl.sync` on CUDA): the generic
//! `shuffle`, the butterfly `shuffle_xor`, the `shuffle_down`/`shuffle_up`
//! rotates, and the bitonic `compare_exchange`. One gather per element — no LDS,
//! no barrier.

use std::sync::Arc;

use super::{Group, SwapDir, iadd, iand, imod, ixor};
use crate::index::{cidx, flat_index, load_at};
use crate::tile::RT;
use svod_ir::UOp;

impl<'k> Group<'k> {
    /// Butterfly reduction within each fixed-width power-of-two subgroup. XOR
    /// partners never cross a subgroup boundary, so the result is replicated in
    /// every lane of that subgroup without LDS or a barrier.
    pub fn subgroup_reduce_scalar<F>(&self, mut value: Arc<UOp>, width: usize, op: F) -> Arc<UOp>
    where
        F: Fn(&Arc<UOp>, &Arc<UOp>) -> Arc<UOp>,
    {
        assert_eq!(self.warps, 1, "subgroup_reduce_scalar is a single-warp op");
        assert!(width.is_power_of_two(), "subgroup width must be a power of two");
        assert!(width <= self.ker.caps.wave_size, "subgroup width must not exceed wave size");
        assert!(self.ker.caps.wave_size.is_multiple_of(width), "subgroup width must divide wave size");
        let mut mask = 1i64;
        while mask < width as i64 {
            value = op(&value, &self.shuffle_xor_lane(&value, mask));
            mask *= 2;
        }
        value
    }

    /// Gather one scalar SSA value from `src_lane` into every lane of the wave.
    pub fn broadcast_scalar(&self, value: &Arc<UOp>, src_lane: i64) -> Arc<UOp> {
        assert_eq!(self.warps, 1, "broadcast_scalar is a single-warp op");
        let w = self.ker.caps.wave_size as i64;
        assert!((0..w).contains(&src_lane), "broadcast source lane {src_lane} must be in 0..{w}");
        self.shuffle_lane(value, &crate::index::cidx(src_lane))
    }

    /// Lane index within a fixed-width power-of-two subgroup.
    pub fn subgroup_laneid(&self, width: usize) -> Arc<UOp> {
        assert!(width.is_power_of_two(), "subgroup width must be a power of two");
        assert!(self.ker.caps.wave_size.is_multiple_of(width), "subgroup width must divide wave size");
        iand(&self.laneid(), width as i64 - 1)
    }

    /// Full-wave butterfly reduction of one scalar SSA value. Each step gathers
    /// from `laneid ^ mask`, so the result is replicated in every lane without
    /// LDS or a barrier. `op` must be associative (normally add or max).
    pub fn wave_reduce_scalar<F>(&self, mut value: Arc<UOp>, op: F) -> Arc<UOp>
    where
        F: Fn(&Arc<UOp>, &Arc<UOp>) -> Arc<UOp>,
    {
        assert_eq!(self.warps, 1, "wave_reduce_scalar is a single-warp op");
        let mut mask = 1i64;
        while mask < self.ker.caps.wave_size as i64 {
            value = op(&value, &self.shuffle_xor_lane(&value, mask));
            mask *= 2;
        }
        value
    }

    /// Per-element cross-lane gather (the public face of [`Self::shuffle_lane`]): for
    /// each logical element, `dst` receives `src`'s value at the SAME position but
    /// from lane `src_lane(laneid)`. Single-warp; one `ds_bpermute` per element (no
    /// LDS, no barrier). The shared foundation for `shuffle_xor`/`compare_exchange`
    /// (and, later, scan / arg-reduce). f32 (bitcast) and i32 transports are
    /// supported today; f16/bf16/i64 are a follow-up.
    ///
    /// # Panics
    /// Panics if the group has more than one warp, or if `dst` and `src` have
    /// different shapes.
    pub fn shuffle<F>(&self, dst: RT<'k>, src: &RT<'k>, src_lane: F) -> RT<'k>
    where
        F: Fn(&Arc<UOp>) -> Arc<UOp>,
    {
        assert_eq!(self.warps, 1, "shuffle is a single-warp op");
        assert_eq!(dst.shape(), src.shape(), "shuffle: shape mismatch");
        let sl = src_lane(&self.laneid());
        let (sbuf, sshape) = (self.anchor(src.uop()), src.shape().to_vec());
        let (dbuf, dshape) = (dst.uop().clone(), dst.shape().to_vec());
        let ended = self.elementwise(&dshape.clone(), move |idxs| {
            let v = load_at(&sbuf, &sshape, idxs);
            flat_index(&dbuf, &dshape, idxs).store(self.shuffle_lane(&v, &sl))
        });
        self.finalize_reg(dst, ended)
    }

    /// Butterfly exchange: `dst[pos] = src[pos]` from lane `laneid ^ mask`. Arch-blind
    /// — for any `mask < wave_size` the XOR partner stays in `[0, wave_size)`, so no
    /// modulus is needed (cheaper than [`Self::shuffle_down`]). The sort/reduce primitive.
    ///
    /// # Panics
    /// Panics if `mask` is not in `1..wave_size`, if the group has more than one
    /// warp, or if `dst` and `src` have different shapes.
    pub fn shuffle_xor(&self, dst: RT<'k>, src: &RT<'k>, mask: i64) -> RT<'k> {
        let w = self.ker.caps.wave_size as i64;
        assert!(mask > 0 && mask < w, "shuffle_xor mask {mask} must be in 1..{w}");
        self.shuffle(dst, src, |laneid| ixor(laneid, mask))
    }

    /// Shift down: `dst[L] = src[(L + delta) mod wave_size]`.
    ///
    /// # Panics
    /// Panics if `delta` is not in `1..wave_size`, if the group has more than one
    /// warp, or if `dst` and `src` have different shapes.
    pub fn shuffle_down(&self, dst: RT<'k>, src: &RT<'k>, delta: i64) -> RT<'k> {
        let w = self.ker.caps.wave_size as i64;
        assert!(delta > 0 && delta < w, "shuffle_down delta {delta} must be in 1..{w}");
        self.shuffle(dst, src, move |laneid| imod(&iadd(laneid, &cidx(delta)), w))
    }

    /// Shift up: `dst[L] = src[(L - delta) mod wave_size]` (the scan primitive).
    ///
    /// # Panics
    /// Panics if `delta` is not in `1..wave_size`, if the group has more than one
    /// warp, or if `dst` and `src` have different shapes.
    pub fn shuffle_up(&self, dst: RT<'k>, src: &RT<'k>, delta: i64) -> RT<'k> {
        let w = self.ker.caps.wave_size as i64;
        assert!(delta > 0 && delta < w, "shuffle_up delta {delta} must be in 1..{w}");
        self.shuffle(dst, src, move |laneid| imod(&iadd(laneid, &cidx(w - delta)), w))
    }

    /// One bitonic compare-exchange stage across the butterfly partner `laneid ^
    /// mask`: each lane keeps the min or max of its element and the partner's, per
    /// `dir` — the building block of sorting networks. Per element: one `ds_bpermute`
    /// gather + an ALU min/max select (no LDS, no barrier).
    ///
    /// # Panics
    /// Panics if the group has more than one warp, if `dst` and `src` have
    /// different shapes, or if `mask` is not in `1..wave_size`.
    pub fn compare_exchange(&self, dst: RT<'k>, src: &RT<'k>, mask: i64, dir: SwapDir) -> RT<'k> {
        assert_eq!(self.warps, 1, "compare_exchange is a single-warp op");
        assert_eq!(dst.shape(), src.shape(), "compare_exchange: shape mismatch");
        let w = self.ker.caps.wave_size as i64;
        assert!(mask > 0 && mask < w, "compare_exchange mask {mask} must be in 1..{w}");
        let laneid = self.laneid();
        // `keep_min`: this lane keeps the smaller of the pair (else the larger). The
        // lower-index lane of a pair is `(laneid & mask) == 0`.
        let is_low = iand(&laneid, mask).try_cmpeq(&cidx(0)).expect("ce is_low");
        let keep_min = match dir {
            SwapDir::Ascending => is_low,
            SwapDir::Descending => iand(&laneid, mask).try_cmpne(&cidx(0)).expect("ce desc"),
            // Bitonic merge: ascending where `(laneid & bit) == 0`. Keep min iff the
            // low-lane flag equals the ascending flag.
            SwapDir::ByLaneBit(bit) => {
                let asc = iand(&laneid, bit).try_cmpeq(&cidx(0)).expect("ce dir bit");
                is_low.try_cmpeq(&asc).expect("ce keep_min")
            }
        };
        let (sbuf, sshape) = (self.anchor(src.uop()), src.shape().to_vec());
        let (dbuf, dshape) = (dst.uop().clone(), dst.shape().to_vec());
        let ended = self.elementwise(&dshape.clone(), move |idxs| {
            let v = load_at(&sbuf, &sshape, idxs);
            let p = self.shuffle_xor_lane(&v, mask);
            let lt = v.try_cmplt(&p).expect("ce lt");
            let mn = UOp::try_where(lt, v.clone(), p.clone()).expect("ce min");
            let mx = v.try_max(&p).expect("ce max");
            let out = UOp::try_where(keep_min.clone(), mn, mx).expect("ce select");
            flat_index(&dbuf, &dshape, idxs).store(out)
        });
        self.finalize_reg(dst, ended)
    }
}
