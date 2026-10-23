use svod_arch::pipelines::audio::words_to_text;

use crate::whisper::WhisperTokenizer;
use crate::whisper::aligner::words_from_path;

fn aligned_words(text: &str, language: &str) -> Vec<svod_arch::rnnt::Word> {
    let tokenizer = WhisperTokenizer::from_hub(true, 99).unwrap();
    let tokens = tokenizer.encode(text);
    let path: Vec<_> = (0..=tokens.len()).collect();
    words_from_path(&path, &path, &tokens, &vec![1.0; tokens.len()], Some(language), &tokenizer, &[], 0.0).0
}

#[test]
fn word_split_preserves_text_and_tokens() {
    let tokenizer = WhisperTokenizer::from_hub(true, 99).unwrap();
    let tokens = tokenizer.encode(" hello world!");
    let (words, word_tokens) = tokenizer.split_to_word_tokens_for_language(&tokens, Some("en"));
    assert_eq!(words.concat(), " hello world!");
    assert_eq!(word_tokens.concat(), tokens);
    assert_eq!(words, [" hello", " world", "!"]);
}

#[test]
fn no_space_languages_split_on_unicode_boundaries() {
    let tokenizer = WhisperTokenizer::from_hub(true, 99).unwrap();
    let tokens = tokenizer.encode("你好世界");
    let (words, word_tokens) = tokenizer.split_to_word_tokens_for_language(&tokens, Some("zh"));
    assert_eq!(words.concat(), "你好世界");
    assert_eq!(word_tokens.concat(), tokens);
    assert!(words.len() > 1);
}

#[test]
fn timestamps_are_word_boundaries() {
    let tokenizer = WhisperTokenizer::from_hub(true, 99).unwrap();
    let timestamp = tokenizer.timestamp_begin();
    let mut tokens = tokenizer.encode(" hello");
    tokens.push(timestamp + 10);
    tokens.extend(tokenizer.encode("world"));

    let (words, word_tokens) = tokenizer.split_to_word_tokens_for_language(&tokens, Some("en"));
    assert_eq!(word_tokens.concat(), tokens);
    assert_eq!(word_tokens[1], [timestamp + 10]);
    assert_eq!(words[2], "world");
}

#[test]
fn punctuation_is_kept_separate_for_timing_attachment() {
    let tokenizer = WhisperTokenizer::from_hub(true, 99).unwrap();
    let tokens = tokenizer.encode(" hello!");
    let (words, _) = tokenizer.split_to_word_tokens_for_language(&tokens, Some("en"));
    assert_eq!(words, [" hello", "!"]);
}

#[test]
fn aligned_fragments_preserve_tokenizer_spacing() {
    for (text, language, expected) in [
        (" 23-м доме", "ru", "23-м доме"),
        (" когда-то", "ru", "когда-то"),
        (" температура -5", "ru", "температура -5"),
        (" Hello, world!", "en", "Hello, world!"),
        ("你好！", "zh", "你好！"),
    ] {
        let words = aligned_words(text, language);
        assert_eq!(words_to_text(&words), expected, "failed to render {text:?}");
        assert_eq!(words.iter().map(|word| word.text.as_str()).collect::<String>().trim(), expected);
    }
}

#[test]
fn aligned_hyphen_suffix_keeps_join_left_fragment() {
    let words = aligned_words(" 23-м доме", "ru");
    assert_eq!(words.iter().map(|word| word.text.as_str()).collect::<Vec<_>>(), [" 23", "-м", " доме"]);
}

#[test]
fn aligned_fragments_drop_blanks_and_trim_only_outer_boundary() {
    assert!(aligned_words("   ", "en").is_empty());
    let words = aligned_words("  hello  world  ", "en");
    assert_eq!(words_to_text(&words), "hello  world");
    assert!(words.first().unwrap().text.starts_with(' '));
}
