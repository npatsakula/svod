/// Bridge a `svod-tk` launch error into the tensor error domain. tk's launch
/// `Err` means a structurally invalid request (a caller bug — fallback-worthy
/// conditions come back as `Ok(None)` instead), so it surfaces as an IR
/// construction failure.
pub(crate) fn tk_launch_error(e: impl std::fmt::Display) -> svod_tensor::error::Error {
    svod_tensor::error::ErrorKind::IrConstruction { details: e.to_string() }.into()
}
