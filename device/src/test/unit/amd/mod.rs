//! AMD test modules — white-box unit + hardware tests over `crate::amd`.
//!
//! Gating helpers in [`test_support`] skip the hardware tests on hosts without
//! a supported GPU. Module gating mirrors `crate::amd` itself: `topology` and
//! `sys` compile everywhere; the rest is `cfg(unix)` (KFD/AM FFI).

mod occupancy;
#[cfg(unix)]
mod pmc;
mod sys;
mod topology;

#[cfg(unix)]
mod test_support;

#[cfg(unix)]
mod allocator;
#[cfg(unix)]
mod am;
#[cfg(unix)]
mod connector;
#[cfg(unix)]
mod device;
#[cfg(unix)]
mod kernarg;
#[cfg(unix)]
mod linked_plan;
#[cfg(unix)]
mod program;
#[cfg(unix)]
mod program_abi;
#[cfg(unix)]
mod queue;
#[cfg(unix)]
mod signal;
#[cfg(unix)]
mod sqtt;
