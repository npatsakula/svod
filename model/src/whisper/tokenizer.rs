//! Whisper BPE tokenizer — delegates to `tiktoken-rs` CoreBPE.
//!
//! The tokenizer loads a `.tiktoken` rank file (base64-encoded BPE merges),
//! appends Whisper's special tokens (SOT, language, task, timestamp, …), and
//! provides Whisper-specific helpers on top of the battle-tested CoreBPE.

use std::collections::HashMap;

use tiktoken_rs::CoreBPE;

use super::error::{Error, Result};

// ─── Language codes ─────────────────────────────────────────────────────────

pub const LANGUAGES: &[(&str, &str)] = &[
    ("en", "english"),
    ("zh", "chinese"),
    ("de", "german"),
    ("es", "spanish"),
    ("ru", "russian"),
    ("ko", "korean"),
    ("fr", "french"),
    ("ja", "japanese"),
    ("pt", "portuguese"),
    ("tr", "turkish"),
    ("pl", "polish"),
    ("ca", "catalan"),
    ("nl", "dutch"),
    ("ar", "arabic"),
    ("sv", "swedish"),
    ("it", "italian"),
    ("id", "indonesian"),
    ("hi", "hindi"),
    ("fi", "finnish"),
    ("vi", "vietnamese"),
    ("he", "hebrew"),
    ("uk", "ukrainian"),
    ("el", "greek"),
    ("ms", "malay"),
    ("cs", "czech"),
    ("ro", "romanian"),
    ("da", "danish"),
    ("hu", "hungarian"),
    ("ta", "tamil"),
    ("no", "norwegian"),
    ("th", "thai"),
    ("ur", "urdu"),
    ("hr", "croatian"),
    ("bg", "bulgarian"),
    ("lt", "lithuanian"),
    ("la", "latin"),
    ("mi", "maori"),
    ("ml", "malayalam"),
    ("cy", "welsh"),
    ("sk", "slovak"),
    ("te", "telugu"),
    ("fa", "persian"),
    ("lv", "latvian"),
    ("bn", "bengali"),
    ("sr", "serbian"),
    ("az", "azerbaijani"),
    ("sl", "slovenian"),
    ("kn", "kannada"),
    ("et", "estonian"),
    ("mk", "macedonian"),
    ("br", "breton"),
    ("eu", "basque"),
    ("is", "icelandic"),
    ("hy", "armenian"),
    ("ne", "nepali"),
    ("mn", "mongolian"),
    ("bs", "bosnian"),
    ("kk", "kazakh"),
    ("sq", "albanian"),
    ("sw", "swahili"),
    ("gl", "galician"),
    ("mr", "marathi"),
    ("pa", "punjabi"),
    ("si", "sinhala"),
    ("km", "khmer"),
    ("sn", "shona"),
    ("yo", "yoruba"),
    ("so", "somali"),
    ("af", "afrikaans"),
    ("oc", "occitan"),
    ("ka", "georgian"),
    ("be", "belarusian"),
    ("tg", "tajik"),
    ("sd", "sindhi"),
    ("gu", "gujarati"),
    ("am", "amharic"),
    ("yi", "yiddish"),
    ("lo", "lao"),
    ("uz", "uzbek"),
    ("fo", "faroese"),
    ("ht", "haitian creole"),
    ("ps", "pashto"),
    ("tk", "turkmen"),
    ("nn", "nynorsk"),
    ("mt", "maltese"),
    ("sa", "sanskrit"),
    ("lb", "luxembourgish"),
    ("my", "myanmar"),
    ("bo", "tibetan"),
    ("tl", "tagalog"),
    ("mg", "malagasy"),
    ("as", "assamese"),
    ("tt", "tatar"),
    ("haw", "hawaiian"),
    ("ln", "lingala"),
    ("ha", "hausa"),
    ("ba", "bashkir"),
    ("jw", "javanese"),
    ("su", "sundanese"),
    ("yue", "cantonese"),
];

/// GPT-2 BPE regex pattern used by tiktoken (same as whisper/tokenizer.py).
const GPT2_PAT: &str = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

// ─── Tokenizer ──────────────────────────────────────────────────────────────

pub struct WhisperTokenizer {
    /// The CoreBPE engine (tiktoken-rs) — handles encode/decode.
    bpe: CoreBPE,
    /// Special token strings → ids.
    special_tokens: HashMap<String, u32>,
    /// Whether this is a multilingual model.
    pub multilingual: bool,
    /// Number of languages supported.
    pub num_languages: usize,
    /// Whisper's non-speech set, encoded once: the symbols, brackets and
    /// music marks its decoding suppresses.
    non_speech: Vec<u32>,
    /// The encoding of a single space, suppressed as a first token.
    blank: Vec<u32>,
}

impl WhisperTokenizer {
    /// Build a tokenizer from a `.tiktoken` rank file (base64-encoded BPE
    /// ranks, one per line: `<base64_token> <rank>`).
    pub fn new(tiktoken_data: &str, multilingual: bool, num_languages: usize) -> Result<Self> {
        // Parse the .tiktoken file into a rank map.
        let encoder: rustc_hash::FxHashMap<Vec<u8>, u32> = parse_tiktoken_ranks(tiktoken_data)?;

        let n_vocab_base = encoder.len();

        // Build Whisper special tokens with sequential IDs starting at n_vocab_base.
        let specials = build_special_tokens(num_languages);
        let mut special_tokens: rustc_hash::FxHashMap<String, u32> = rustc_hash::FxHashMap::default();
        for (i, s) in specials.iter().enumerate() {
            special_tokens.insert(s.clone(), (n_vocab_base + i) as u32);
        }

        // Build the CoreBPE — uses FxHashMap internally.
        let bpe = CoreBPE::new(encoder, special_tokens.clone(), GPT2_PAT)
            .map_err(|e| Error::Tokenizer { msg: format!("CoreBPE::new: {e}") })?;

        // Convert special_tokens to std HashMap for our lookups
        let special_tokens: HashMap<String, u32> = special_tokens.into_iter().collect();

        let mut tokenizer =
            Self { bpe, special_tokens, multilingual, num_languages, non_speech: Vec::new(), blank: Vec::new() };
        tokenizer.non_speech = tokenizer.encode_non_speech();
        tokenizer.blank = tokenizer.encode(" ");
        Ok(tokenizer)
    }

    // ─── Tokenizer loading helpers ────────────────────────────────────────────

    /// Load tokenizer for a Whisper model. Uses embedded tiktoken data
    /// (from the `openai/whisper` submodule) — no runtime download.
    pub fn from_hub(multilingual: bool, num_languages: usize) -> Result<Self> {
        let data = if multilingual {
            include_str!("assets/multilingual.tiktoken")
        } else {
            include_str!("assets/gpt2.tiktoken")
        };
        Self::new(data, multilingual, num_languages)
    }

    /// Load the tiktoken data from a local file.
    pub fn from_file(path: &std::path::Path, multilingual: bool, num_languages: usize) -> Result<Self> {
        let data =
            std::fs::read_to_string(path).map_err(|e| Error::Tokenizer { msg: format!("read tiktoken file: {e}") })?;
        Self::new(&data, multilingual, num_languages)
    }

    // ─── Special token accessors ────────────────────────────────────────────

    pub fn eot(&self) -> u32 {
        self.special_tokens["<|endoftext|>"]
    }

    pub fn sot(&self) -> u32 {
        self.special_tokens["<|startoftranscript|>"]
    }

    pub fn transcribe(&self) -> u32 {
        self.special_tokens["<|transcribe|>"]
    }

    pub fn translate(&self) -> u32 {
        self.special_tokens["<|translate|>"]
    }

    pub fn sot_prev(&self) -> u32 {
        self.special_tokens["<|startofprev|>"]
    }

    pub fn sot_lm(&self) -> u32 {
        self.special_tokens["<|startoflm|>"]
    }

    pub fn no_speech(&self) -> Option<u32> {
        self.special_tokens.get("<|nospeech|>").copied()
    }

    pub fn no_timestamps(&self) -> u32 {
        self.special_tokens["<|notimestamps|>"]
    }

    pub fn timestamp_begin(&self) -> u32 {
        self.special_tokens["<|0.00|>"]
    }

    /// All language token IDs.
    pub fn all_language_tokens(&self) -> Vec<u32> {
        LANGUAGES
            .iter()
            .take(self.num_languages)
            .filter_map(|(code, _)| self.special_tokens.get(&format!("<|{code}|>")).copied())
            .collect()
    }

    /// Language codes matching [`all_language_tokens`](Self::all_language_tokens).
    pub fn all_language_codes(&self) -> Vec<String> {
        LANGUAGES
            .iter()
            .take(self.num_languages)
            .filter_map(|(code, _)| self.special_tokens.get(&format!("<|{code}|>")).map(|_| code.to_string()))
            .collect()
    }

    /// Look up the language token for a code string.
    pub fn language_token_for(&self, code: &str) -> Option<u32> {
        self.special_tokens.get(&format!("<|{code}|>")).copied()
    }

    /// Look up the language code for a token id.
    pub fn code_for_token(&self, token: u32) -> Option<String> {
        LANGUAGES
            .iter()
            .take(self.num_languages)
            .find(|(code, _)| self.special_tokens.get(&format!("<|{code}|>")).map(|t| *t == token).unwrap_or(false))
            .map(|(code, _)| code.to_string())
    }

    // ─── Encode / decode (delegated to CoreBPE) ──────────────────────────────

    /// Encode text using BPE (no special tokens).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.bpe.encode_ordinary(text)
    }

    /// Decode token IDs to text, filtering out special/timestamp tokens.
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let timestamp_begin = self.timestamp_begin();
        let filtered: Vec<u32> = token_ids.iter().filter(|&&t| t < timestamp_begin).copied().collect();
        self.bpe.decode(&filtered).unwrap_or_default()
    }

    /// Decode including timestamp tokens (annotated).
    pub fn decode_with_timestamps(&self, token_ids: &[u32]) -> String {
        let ts_begin = self.timestamp_begin();
        let mut result = String::new();
        for &id in token_ids {
            if id >= ts_begin {
                let secs = (id - ts_begin) as f32 / super::config::TOKENS_PER_SECOND;
                result.push_str(&format!("<|{secs:.2}|>"));
            } else {
                // Decode single token via CoreBPE
                if let Ok(s) = self.bpe.decode(&[id]) {
                    result.push_str(&s);
                }
            }
        }
        result
    }

    /// OpenAI-compatible word grouping for a resolved language. Languages
    /// without reliable spaces split at valid Unicode boundaries instead.
    pub fn split_to_word_tokens_for_language(
        &self,
        tokens: &[u32],
        language: Option<&str>,
    ) -> (Vec<String>, Vec<Vec<u32>>) {
        let (subwords, subword_tokens) = self.split_tokens_on_unicode(tokens);
        if matches!(language, Some("zh" | "ja" | "th" | "lo" | "my" | "yue")) {
            return (subwords, subword_tokens);
        }

        let mut words: Vec<String> = Vec::new();
        let mut word_tokens: Vec<Vec<u32>> = Vec::new();
        for (subword, tokens) in subwords.into_iter().zip(subword_tokens) {
            let special = tokens.first().is_some_and(|&token| token >= self.eot());
            let follows_special =
                word_tokens.last().and_then(|tokens| tokens.first()).is_some_and(|&token| token >= self.eot());
            let punctuation =
                !subword.trim().is_empty() && "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~".contains(subword.trim());
            if words.is_empty() || special || follows_special || subword.starts_with(' ') || punctuation {
                words.push(subword);
                word_tokens.push(tokens);
            } else {
                words.last_mut().unwrap().push_str(&subword);
                word_tokens.last_mut().unwrap().extend(tokens);
            }
        }
        (words, word_tokens)
    }

    fn split_tokens_on_unicode(&self, tokens: &[u32]) -> (Vec<String>, Vec<Vec<u32>>) {
        let mut words = Vec::new();
        let mut word_tokens = Vec::new();
        let mut current = Vec::new();
        for &token in tokens {
            current.push(token);
            if let Ok(decoded) = self.bpe.decode(&current) {
                words.push(decoded);
                word_tokens.push(std::mem::take(&mut current));
            }
        }
        if !current.is_empty() {
            let decoded = self.bpe.decode(&current).unwrap_or_default();
            words.push(decoded);
            word_tokens.push(current);
        }
        (words, word_tokens)
    }

    /// Non-speech tokens to suppress (matching whisper/tokenizer.py).
    pub fn non_speech_tokens(&self) -> &[u32] {
        &self.non_speech
    }

    /// The encoding of a single space.
    pub fn blank_tokens(&self) -> &[u32] {
        &self.blank
    }

    fn encode_non_speech(&self) -> Vec<u32> {
        let symbols = "\"#()*+/:;<=>@[\\]^_`{|}~「」『』";
        let extras = [
            "<<", ">>", "<<<", ">>>", "--", "---", "-(", "-[", "('", "(\"", "((", "))", "(((", ")))", "[[", "]]", "{{",
            "}}", "♪♪", "♪♪♪",
        ];
        let misc = "♩♪♫♬♭♮♯";

        let mut result: Vec<u32> = Vec::new();

        for s in symbols.chars() {
            let ids = self.encode(&s.to_string());
            if ids.len() == 1 {
                result.push(ids[0]);
            }
            let ids = self.encode(&format!(" {s}"));
            if ids.len() == 1 {
                result.push(ids[0]);
            }
        }
        for e in &extras {
            let ids = self.encode(e);
            if ids.len() == 1 {
                result.push(ids[0]);
            }
            let ids = self.encode(&format!(" {e}"));
            if ids.len() == 1 {
                result.push(ids[0]);
            }
        }
        for c in misc.chars() {
            let ids = self.encode(&c.to_string());
            if !ids.is_empty() {
                result.push(ids[0]);
            }
        }
        let dash = self.encode(" -");
        if !dash.is_empty() {
            result.push(dash[0]);
        }
        let quote = self.encode(" '");
        if !quote.is_empty() {
            result.push(quote[0]);
        }

        result.sort();
        result.dedup();
        result
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Parse a `.tiktoken` file (base64-encoded token → rank per line).
fn parse_tiktoken_ranks(data: &str) -> Result<rustc_hash::FxHashMap<Vec<u8>, u32>> {
    use base64::Engine;
    let mut ranks = rustc_hash::FxHashMap::default();
    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let token_b64 = parts.next().ok_or_else(|| Error::Tokenizer { msg: "malformed tiktoken line".into() })?;
        let rank_str = parts.next().ok_or_else(|| Error::Tokenizer { msg: "malformed tiktoken line".into() })?;
        let token_bytes = if token_b64 == "=" {
            // Python's base64.b64decode("=") returns empty bytes
            Vec::new()
        } else {
            base64::engine::general_purpose::STANDARD
                .decode(token_b64)
                .map_err(|e| Error::Tokenizer { msg: format!("base64 decode: {e}") })?
        };
        let rank: u32 = rank_str.parse().map_err(|e| Error::Tokenizer { msg: format!("rank parse: {e}") })?;
        ranks.insert(token_bytes, rank);
    }
    Ok(ranks)
}

fn build_special_tokens(num_languages: usize) -> Vec<String> {
    let mut specials = vec!["<|endoftext|>".to_string(), "<|startoftranscript|>".to_string()];
    for (code, _) in LANGUAGES.iter().take(num_languages) {
        specials.push(format!("<|{code}|>"));
    }
    specials.push("<|translate|>".into());
    specials.push("<|transcribe|>".into());
    specials.push("<|startoflm|>".into());
    specials.push("<|startofprev|>".into());
    specials.push("<|nospeech|>".into());
    specials.push("<|notimestamps|>".into());
    for i in 0..=1500 {
        specials.push(format!("<|{:.2}|>", i as f32 * 0.02));
    }
    specials
}
