mod encoder;
mod rnnt;

/// Pins the threading contract a serving pipeline relies on: each built
/// transcriber and VAD splitter moves to a per-device worker thread.
#[test]
fn pipeline_parts_are_send() {
    fn assert_send<T: Send>() {}
    assert_send::<crate::gigaam::GigaAm>();
    assert_send::<crate::gigaam::GigaAmTranscriber>();
    assert_send::<svod_arch::pipelines::audio::VadSplitter<crate::firered_vad::FireRedVadProbs>>();
}
