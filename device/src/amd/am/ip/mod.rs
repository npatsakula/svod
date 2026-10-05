//! IP-block bring-up for the AM driver (GMC, GFX, SDMA). VF flavor: the
//! GIM owns PSP/SMU/clocks/L2, so these modules program only the per-VF state
//! (page-table contexts, rings, queues) over the RLCG/direct register access.

pub mod gfx;
pub mod gmc;
pub mod sdma;
