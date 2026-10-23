//! DTW word-level alignment via cross-attention weights.
//!
//! Pure CPU implementation matching `whisper/timing.py`. Pipeline:
//! 1. Extract cross-attention QK weights from alignment heads
//! 2. Softmax over time → standardize → median filter (width=7)
//! 3. Average across heads → `matrix[n_tokens, n_audio_frames]`
//! 4. DTW on `-matrix` → backtrace → `(text_idx, time_idx)` path
//! 5. Map jump points to word boundaries

use svod_arch::pipelines::audio::Segment;

/// One word's timing from DTW alignment.
#[derive(Clone, Debug)]
pub struct WordTiming {
    pub word: String,
    pub tokens: Vec<u32>,
    pub start: f32,
    pub end: f32,
    pub probability: f32,
}

/// Classic DTW + backtrace on a cost matrix. Returns `(text_indices, time_indices)`.
///
/// Matches `whisper.timing.dtw_cpu`. Input `x` is `[N, M]` row-major (N=text,
/// M=time). Returns two parallel vectors that together form the warp path.
pub fn dtw(x: &[f32], n_rows: usize, n_cols: usize) -> (Vec<usize>, Vec<usize>) {
    let n = n_rows;
    let m = n_cols;
    let cost_size = (n + 1) * (m + 1);

    // Cost matrix (row-major, (N+1)×(M+1)), initialized to +inf
    let mut cost = vec![f32::INFINITY; cost_size];
    let mut trace = vec![-1i32; cost_size];

    // cost[0,0] = 0
    cost[0] = 0.0;

    // Fill cost matrix
    for j in 1..=m {
        for i in 1..=n {
            let c0 = cost[(i - 1) * (m + 1) + (j - 1)]; // diagonal
            let c1 = cost[(i - 1) * (m + 1) + j]; // up
            let c2 = cost[i * (m + 1) + (j - 1)]; // left

            let (c, t) = if c0 < c1 && c0 < c2 {
                (c0, 0i32)
            } else if c1 < c0 && c1 < c2 {
                (c1, 1i32)
            } else {
                (c2, 2i32)
            };

            cost[i * (m + 1) + j] = x[(i - 1) * m + (j - 1)] + c;
            trace[i * (m + 1) + j] = t;
        }
    }

    // Backtrace setup: trace[0, :] = 2 (first row), trace[:, 0] = 1 (first column)
    // The original Python sets both, but the second overwrites on trace[0,0].
    for slot in &mut trace[..=m] {
        *slot = 2;
    }
    for i in 0..=n {
        trace[i * (m + 1)] = 1; // trace[i, 0] = 1  (first column, row i)
    }

    let mut text_indices = Vec::new();
    let mut time_indices = Vec::new();
    let mut i = n;
    let mut j = m;

    while i > 0 || j > 0 {
        text_indices.push(i - 1);
        time_indices.push(j - 1);

        match trace[i * (m + 1) + j] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => {
                i -= 1;
            }
            2 => {
                j -= 1;
            }
            _ => break,
        }
    }

    text_indices.reverse();
    time_indices.reverse();
    (text_indices, time_indices)
}

/// Median filter along the last axis with reflect padding.
/// Input shape `[.., width]` flattened. `data` is `[n_rows * n_cols]` with
/// `filter_width` applied to the last axis (n_cols).
pub fn median_filter(data: &[f32], n_rows: usize, n_cols: usize, filter_width: usize) -> Vec<f32> {
    assert!(filter_width > 0 && filter_width % 2 == 1, "filter_width must be odd");
    // Fewer than four frames leave no interior for a seven-wide reflect
    // padding: every window folds back onto the same edge values. Narrow the
    // filter to the window instead, as whisper.cpp does.
    let filter_width = match n_cols {
        1 | 2 => 1,
        3 => 3,
        _ => filter_width,
    };
    let pad = filter_width / 2;
    let mut out = vec![0.0f32; n_rows * n_cols];

    // One window buffer for the whole pass: allocating per output cell dominated
    // the filter at ~1M cells.
    let mut window = Vec::with_capacity(filter_width);
    for i in 0..n_rows {
        for j in 0..n_cols {
            window.clear();
            for off in -(pad as isize)..=(pad as isize) {
                let idx = j as isize + off;
                // Reflect padding (single-bounce)
                let idx = if idx < 0 {
                    -idx
                } else if idx >= n_cols as isize {
                    2 * (n_cols as isize - 1) - idx
                } else {
                    idx
                };
                let idx = idx.clamp(0, n_cols as isize - 1) as usize;
                window.push(data[i * n_cols + idx]);
            }
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            out[i * n_cols + j] = window[filter_width / 2];
        }
    }
    out
}

/// Extract an alignment path from statically packed selected-head raw QK
/// scores. Compiled strides are separate from the valid unpadded extents.
#[allow(clippy::too_many_arguments)]
pub fn find_alignment_path_selected(
    qk: &[f32],
    n_heads: usize,
    text_stride: usize,
    audio_stride: usize,
    valid_text: usize,
    valid_audio: usize,
    medfilt_width: usize,
    sot_len: usize,
) -> (Vec<usize>, Vec<usize>) {
    let valid_text = valid_text.min(text_stride);
    let valid_audio = valid_audio.min(audio_stride);
    // The matrix keeps the `<|notimestamps|>` row before the text tokens, so
    // `sot_len + 2` tokens (prompt, `<|notimestamps|>`, EOT) leave no text row.
    if n_heads == 0 || valid_audio == 0 || valid_text <= sot_len + 2 {
        return (Vec::new(), Vec::new());
    }

    let mut weights = vec![0.0f32; n_heads * valid_text * valid_audio];
    for head in 0..n_heads {
        for text in 0..valid_text {
            let src = (head * text_stride + text) * audio_stride;
            let dst = (head * valid_text + text) * valid_audio;
            weights[dst..dst + valid_audio].copy_from_slice(&qk[src..src + valid_audio]);
        }
    }

    for row in weights.chunks_exact_mut(valid_audio) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for value in row.iter_mut() {
            *value = (*value - max).exp();
            sum += *value;
        }
        for value in row.iter_mut() {
            *value /= sum;
        }
    }

    // Standardize each (head, frame) column. The columns are strided by a whole
    // row, so this sweeps text-major and keeps one accumulator per frame rather
    // than walking each column three times.
    let mut mean = vec![0.0f32; valid_audio];
    let mut deviation = vec![0.0f32; valid_audio];
    for head in 0..n_heads {
        let head_base = head * valid_text * valid_audio;
        let rows = || (0..valid_text).map(move |text| head_base + text * valid_audio);
        mean.fill(0.0);
        deviation.fill(0.0);
        for row in rows() {
            for (accumulated, &value) in mean.iter_mut().zip(&weights[row..row + valid_audio]) {
                *accumulated += value;
            }
        }
        for accumulated in mean.iter_mut() {
            *accumulated /= valid_text as f32;
        }
        for row in rows() {
            for ((accumulated, &value), &mean) in
                deviation.iter_mut().zip(&weights[row..row + valid_audio]).zip(mean.iter())
            {
                *accumulated += (value - mean) * (value - mean);
            }
        }
        for accumulated in deviation.iter_mut() {
            *accumulated = (*accumulated / valid_text as f32).sqrt().max(1e-10);
        }
        for row in rows() {
            for ((value, &mean), &std) in
                weights[row..row + valid_audio].iter_mut().zip(mean.iter()).zip(deviation.iter())
            {
                *value = (*value - mean) / std;
            }
        }
    }

    let filtered = median_filter(&weights, n_heads * valid_text, valid_audio, medfilt_width);
    let aligned_text = valid_text - sot_len - 1;
    let mut cost = vec![0.0f32; aligned_text * valid_audio];
    for text in 0..aligned_text {
        for frame in 0..valid_audio {
            let sum = (0..n_heads)
                .map(|head| filtered[(head * valid_text + text + sot_len) * valid_audio + frame])
                .sum::<f32>();
            cost[text * valid_audio + frame] = -(sum / n_heads as f32);
        }
    }
    dtw(&cost, aligned_text, valid_audio)
}

/// Convert a DTW alignment path into word-level timings.
///
/// `text_indices`, `time_indices`: DTW path. `word_boundaries`: cumulative
/// token counts at word starts (including leading 0). `token_probs`: per-token
/// probabilities from the decoder.
pub fn path_to_word_timings(
    text_indices: &[usize],
    time_indices: &[usize],
    word_boundaries: &[usize],
    words: &[String],
    word_token_lists: &[Vec<u32>],
    token_probs: &[f32],
    tokens_per_second: f32,
) -> Vec<WordTiming> {
    let mut timings = raw_word_timings(
        text_indices,
        time_indices,
        word_boundaries,
        words,
        word_token_lists,
        token_probs,
        tokens_per_second,
    );
    clean_up_word_timings(&mut timings);
    timings
}

/// [`path_to_word_timings`] with `add_word_timestamps`' segment-edge repair.
///
/// `segments` are the recogniser's timestamped segments for the aligned text;
/// the first segment's start and the last segment's end bound the edge words.
/// Returns the repaired timings and the `last_speech_timestamp` that a
/// following window carries into the pause clamp.
#[allow(clippy::too_many_arguments)]
pub fn path_to_word_timings_in_segments(
    text_indices: &[usize],
    time_indices: &[usize],
    word_boundaries: &[usize],
    words: &[String],
    word_token_lists: &[Vec<u32>],
    token_probs: &[f32],
    tokens_per_second: f32,
    segments: &[Segment],
    last_speech_timestamp: f32,
) -> (Vec<WordTiming>, f32) {
    let mut timings = raw_word_timings(
        text_indices,
        time_indices,
        word_boundaries,
        words,
        word_token_lists,
        token_probs,
        tokens_per_second,
    );
    let median = clean_up(&mut timings);
    let last_speech_timestamp = repair_segment_edges(&mut timings, segments, median, last_speech_timestamp);
    (timings, last_speech_timestamp)
}

fn raw_word_timings(
    text_indices: &[usize],
    time_indices: &[usize],
    word_boundaries: &[usize],
    words: &[String],
    word_token_lists: &[Vec<u32>],
    token_probs: &[f32],
    tokens_per_second: f32,
) -> Vec<WordTiming> {
    if word_boundaries.len() <= 1 || text_indices.is_empty() || time_indices.is_empty() {
        return Vec::new();
    }

    // Find jump points: positions where text_indices changes
    let mut jumps = vec![false; text_indices.len()];
    jumps[0] = true;
    for i in 1..text_indices.len() {
        jumps[i] = text_indices[i] != text_indices[i - 1];
    }

    // `jump_times[k]` is the first frame of DTW row `k`, and row `k >= 1`
    // aligns text token `k - 1`: the boundary at token `b` starts at
    // `jump_times[b + 1]`. The final path frame ends the last token, which has
    // no row of its own to start a boundary.
    let mut jump_times: Vec<f32> = time_indices
        .iter()
        .zip(&jumps)
        .filter(|&(_, jump)| *jump)
        .map(|(&time, _)| time as f32 / tokens_per_second)
        .collect();
    let final_frame = time_indices.last().copied().unwrap_or(0) + 1;
    jump_times.push(final_frame as f32 / tokens_per_second);

    let n_words = word_boundaries.len() - 1;
    let mut result = Vec::with_capacity(n_words);

    let last_jump = jump_times.last().copied().unwrap_or(0.0);
    for w in 0..n_words {
        let start = jump_times.get(word_boundaries[w] + 1).copied().unwrap_or(last_jump);
        let end = jump_times.get(word_boundaries[w + 1] + 1).copied().unwrap_or(last_jump).max(start);

        // Token probabilities for this word
        let tok_start = word_boundaries[w];
        let tok_end = word_boundaries[w + 1];
        let prob = if tok_end > tok_start {
            let slice = token_probs.get(tok_start..tok_end.min(token_probs.len())).unwrap_or_default();
            if slice.is_empty() { 0.0 } else { slice.iter().sum::<f32>() / slice.len() as f32 }
        } else {
            0.0
        };

        result.push(WordTiming {
            word: words.get(w).cloned().unwrap_or_default(),
            tokens: word_token_lists.get(w).cloned().unwrap_or_default(),
            start,
            end,
            probability: prob,
        });
    }

    result
}

/// Apply OpenAI Whisper's sentence-boundary duration cleanup and punctuation
/// attachment to raw DTW timings.
pub fn clean_up_word_timings(words: &mut [WordTiming]) {
    clean_up(words);
}

/// The reference's median-duration sentence clamp followed by punctuation
/// attachment. Returns the median duration the segment repair clamps against.
fn clean_up(words: &mut [WordTiming]) -> f32 {
    let median = median_word_duration(words);
    let max_duration = median * 2.0;
    if max_duration > 0.0 {
        const SENTENCE_END: &str = ".。!！?？";
        for index in 1..words.len() {
            if words[index].end - words[index].start > max_duration {
                if SENTENCE_END.contains(words[index].word.as_str()) {
                    words[index].end = words[index].start + max_duration;
                } else if SENTENCE_END.contains(words[index - 1].word.as_str()) {
                    words[index].start = words[index].end - max_duration;
                }
            }
        }
    }

    merge_punctuations(words, "\"'“¿([{-", "\"'.。,，!！?？:：”)]}、");
    median
}

/// Median duration of the finite, positive word spans, capped at 0.7 s.
fn median_word_duration(words: &[WordTiming]) -> f32 {
    let mut durations: Vec<f32> = words
        .iter()
        .map(|word| word.end - word.start)
        .filter(|duration| duration.is_finite() && *duration > 0.0)
        .collect();
    durations.sort_by(|a, b| a.total_cmp(b));
    match durations.len() {
        0 => 0.0,
        len if len % 2 == 0 => (durations[len / 2 - 1] + durations[len / 2]) / 2.0,
        len => durations[len / 2],
    }
    .min(0.7)
}

/// `add_word_timestamps`' segment-edge repair: clamp the first and last word to
/// the segment timestamps the recogniser emitted, using the pause since
/// `last_speech_timestamp`. Returns the timestamp the next window carries.
pub(crate) fn repair_segment_edges(
    words: &mut [WordTiming],
    segments: &[Segment],
    median: f32,
    last_speech_timestamp: f32,
) -> f32 {
    let (Some(segment_start), Some(segment_end)) =
        (segments.first().map(|segment| segment.start), segments.last().map(|segment| segment.end))
    else {
        return last_speech_timestamp;
    };
    let Some(first) = words.iter().position(|word| !word.word.is_empty()) else {
        return last_speech_timestamp;
    };
    let last = words.iter().rposition(|word| !word.word.is_empty()).unwrap_or(first);
    let max_duration = median * 2.0;

    // A long pause before the first word leaves its DTW start far behind its
    // end: first halve the following word's span, then clamp this word's start.
    let second = (first + 1..words.len()).find(|&index| !words[index].word.is_empty());
    let pause_clamp = words[first].end - last_speech_timestamp > median * 4.0
        && (words[first].end - words[first].start > max_duration
            || second.is_some_and(|index| words[index].end - words[first].start > max_duration * 2.0));
    if pause_clamp {
        if let Some(index) = second
            && words[index].end - words[index].start > max_duration
        {
            let boundary = (words[index].end / 2.0).max(words[index].end - max_duration);
            words[first].end = boundary;
            words[index].start = boundary;
        }
        words[first].start = (words[first].end - max_duration).max(0.0);
    }

    // Prefer the segment start when DTW put the first word well before it.
    if segment_start < words[first].end && segment_start - 0.5 > words[first].start {
        words[first].start = (words[first].end - median).min(segment_start).max(0.0);
    }

    // Prefer the segment end when DTW put the last word well after it.
    if segment_end > words[last].start && segment_end + 0.5 < words[last].end {
        words[last].end = (words[last].start + median).max(segment_end);
        segment_end
    } else {
        words[last].end
    }
}

fn merge_punctuations(words: &mut [WordTiming], prepended: &str, appended: &str) {
    let mut following = words.len().saturating_sub(1);
    for previous in (0..words.len().saturating_sub(1)).rev() {
        // A whitespace-only fragment trims to the empty string, which every
        // pattern contains: it rides along onto the next word, keeping its spacing.
        if words[previous].word.starts_with(' ') && prepended.contains(words[previous].word.trim()) {
            let prefix = std::mem::take(&mut words[previous].word);
            words[following].word.insert_str(0, &prefix);
            let mut tokens = std::mem::take(&mut words[previous].tokens);
            tokens.append(&mut words[following].tokens);
            words[following].tokens = tokens;
        } else {
            following = previous;
        }
    }

    let mut previous = 0;
    for following in 1..words.len() {
        if !words[previous].word.ends_with(' ') && appended.contains(words[following].word.as_str()) {
            let suffix = std::mem::take(&mut words[following].word);
            words[previous].word.push_str(&suffix);
            let tokens = std::mem::take(&mut words[following].tokens);
            words[previous].tokens.extend(tokens);
        } else {
            previous = following;
        }
    }
}
