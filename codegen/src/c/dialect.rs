//! C dialect selection for the C-family renderer (clang C vs Metal Shading
//! Language). Mirrors `crate::llvm::common::LlvmTarget`: one renderer, one
//! emitter, per-dialect string hooks at the points where the two diverge.

use svod_dtype::AddrSpace;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CDialect {
    #[default]
    Clang,
    Metal,
}

impl CDialect {
    pub fn is_metal(self) -> bool {
        matches!(self, Self::Metal)
    }
}

/// MSL pointer types carry their address space; C pointer types do not.
/// Returns `""` for Clang so every emitted C string is unchanged.
pub fn addr_qualifier(dialect: CDialect, addrspace: Option<AddrSpace>) -> &'static str {
    match dialect {
        CDialect::Clang => "",
        CDialect::Metal => match addrspace {
            Some(AddrSpace::Global | AddrSpace::Swizzled { .. }) => "device ",
            Some(AddrSpace::Local) => "threadgroup ",
            Some(AddrSpace::Reg) | None => "thread ",
        },
    }
}
