use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use fancy_regex::Regex;
use rayon::prelude::*;

/// BPE tokenizer compatible with tiktoken (custom 8192-vocab).
///
/// Loads from tokenizer.json (primary) or .tiktoken + config files (fallback).
/// The JSON format is: {"pattern": "...", "mergeable_ranks": [["base64", rank], ...],
///                       "special_tokens": [["name", rank], ...]}
pub struct Tokenizer {
    /// byte sequence -> rank (merge priority)
    encoder: HashMap<Vec<u8>, u32>,
    /// regex for pre-tokenization splitting
    pattern: Regex,
    /// BOS token ID
    bos_token_id: u16,
}

impl Tokenizer {
    /// Load tokenizer from directory.
    /// Tries tokenizer.json first, falls back to tokenizer.tiktoken + tokenizer_config.txt.
    pub fn load(tokenizer_dir: &str) -> Result<Self> {
        let dir = Path::new(tokenizer_dir);
        let json_path = dir.join("tokenizer.json");
        if json_path.exists() {
            return Self::load_json(&json_path);
        }
        Self::load_tiktoken(dir)
    }

    /// Load from tokenizer.json (the format already on disk).
    fn load_json(path: &Path) -> Result<Self> {
        use base64::Engine;
        let data = fs::read_to_string(path).context("read tokenizer.json")?;

        // Minimal JSON parsing — avoid pulling in serde_json just for this.
        // The file is: {"pattern":"...", "mergeable_ranks":[["b64",rank],...], "special_tokens":[["name",rank],...]}
        // We parse it manually since the structure is fixed and simple.

        // Extract pattern string (between first "pattern":" and next unescaped ")
        let pat_start = data
            .find("\"pattern\":")
            .context("missing pattern field")?
            + "\"pattern\":".len();
        let pat_str = extract_json_string(&data[pat_start..]).context("parse pattern")?;

        // Extract mergeable_ranks array
        let mr_start = data
            .find("\"mergeable_ranks\":")
            .context("missing mergeable_ranks")?
            + "\"mergeable_ranks\":".len();
        let mr_slice = &data[mr_start..];

        let mut encoder = HashMap::with_capacity(8192);
        // Parse [[b64, rank], ...] — scan for each inner array
        let mut pos = mr_slice.find('[').context("no array start")? + 1; // skip outer [
        loop {
            // Find next inner [
            match mr_slice[pos..].find('[') {
                None => break,
                Some(offset) => pos += offset + 1,
            }
            // Read b64 string
            let q1 = match mr_slice[pos..].find('"') {
                None => break,
                Some(o) => pos + o + 1,
            };
            let q2 = match mr_slice[q1..].find('"') {
                None => break,
                Some(o) => q1 + o,
            };
            let b64 = &mr_slice[q1..q2];

            // Read rank number
            let comma = match mr_slice[q2..].find(',') {
                None => break,
                Some(o) => q2 + o + 1,
            };
            let bracket = match mr_slice[comma..].find(']') {
                None => break,
                Some(o) => comma + o,
            };
            let rank: u32 = mr_slice[comma..bracket]
                .trim()
                .parse()
                .context("invalid rank")?;

            let token_bytes = base64::engine::general_purpose::STANDARD
                .decode(b64)
                .context("invalid base64")?;
            encoder.insert(token_bytes, rank);

            pos = bracket + 1;

            // Check if we hit the end of the outer array
            let next_relevant = mr_slice[pos..].find(|c: char| c == '[' || c == ']');
            match next_relevant {
                Some(o) if mr_slice.as_bytes()[pos + o] == b']' => break,
                None => break,
                _ => {}
            }
        }

        // Extract special_tokens to find BOS
        let st_start = data
            .find("\"special_tokens\":")
            .context("missing special_tokens")?
            + "\"special_tokens\":".len();
        let st_slice = &data[st_start..];

        let mut bos_id: Option<u16> = None;
        let mut st_pos = st_slice.find('[').context("no array start")? + 1;
        loop {
            match st_slice[st_pos..].find('[') {
                None => break,
                Some(offset) => st_pos += offset + 1,
            }
            let q1 = match st_slice[st_pos..].find('"') {
                None => break,
                Some(o) => st_pos + o + 1,
            };
            let q2 = match st_slice[q1..].find('"') {
                None => break,
                Some(o) => q1 + o,
            };
            let name = &st_slice[q1..q2];

            let comma = match st_slice[q2..].find(',') {
                None => break,
                Some(o) => q2 + o + 1,
            };
            let bracket = match st_slice[comma..].find(']') {
                None => break,
                Some(o) => comma + o,
            };
            let rank: u16 = st_slice[comma..bracket]
                .trim()
                .parse()
                .context("invalid special token rank")?;

            if name == "<|reserved_0|>" {
                bos_id = Some(rank);
            }

            st_pos = bracket + 1;
            let next = st_slice[st_pos..].find(|c: char| c == '[' || c == ']');
            match next {
                Some(o) if st_slice.as_bytes()[st_pos + o] == b']' => break,
                None => break,
                _ => {}
            }
        }

        let bos_token_id = bos_id.context("BOS token <|reserved_0|> not found")?;
        let pattern = Regex::new(&pat_str).context("compile tokenizer regex")?;

        eprintln!(
            "[tokenizer] loaded {} BPE ranks from json, bos_id={bos_token_id}",
            encoder.len()
        );

        Ok(Tokenizer {
            encoder,
            pattern,
            bos_token_id,
        })
    }

    /// Load from .tiktoken + config files (produced by convert_tokenizer.py).
    fn load_tiktoken(dir: &Path) -> Result<Self> {
        use base64::Engine;

        let tiktoken_path = dir.join("tokenizer.tiktoken");
        let tiktoken_data =
            fs::read_to_string(&tiktoken_path).context("read tokenizer.tiktoken")?;
        let mut encoder = HashMap::new();
        for line in tiktoken_data.lines() {
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split(' ');
            let b64 = parts.next().context("missing base64 token")?;
            let rank: u32 = parts
                .next()
                .context("missing rank")?
                .parse()
                .context("invalid rank")?;
            let token_bytes = base64::engine::general_purpose::STANDARD
                .decode(b64)
                .context("invalid base64")?;
            encoder.insert(token_bytes, rank);
        }

        let config_path = dir.join("tokenizer_config.txt");
        let config_data = fs::read_to_string(&config_path).context("read tokenizer_config.txt")?;

        let mut pat_str = String::new();
        let mut bos_id: Option<u16> = None;

        for line in config_data.lines() {
            if let Some(rest) = line.strip_prefix("pat_str:") {
                pat_str = rest.to_string();
            } else if let Some(rest) = line.strip_prefix("bos_id:") {
                bos_id = Some(rest.parse().context("invalid bos_id")?);
            }
        }

        if pat_str.is_empty() {
            bail!("pat_str not found in tokenizer_config.txt");
        }
        let bos_token_id = bos_id.context("bos_id not found in tokenizer_config.txt")?;
        let pattern = Regex::new(&pat_str).context("compile tokenizer regex")?;

        eprintln!(
            "[tokenizer] loaded {} BPE ranks from tiktoken, bos_id={bos_token_id}",
            encoder.len()
        );

        Ok(Tokenizer {
            encoder,
            pattern,
            bos_token_id,
        })
    }

    /// BOS token ID.
    pub fn bos_token_id(&self) -> u16 {
        self.bos_token_id
    }

    /// Encode text to token IDs (no special tokens — equivalent to encode_ordinary).
    pub fn encode(&self, text: &str) -> Vec<u16> {
        let mut tokens = Vec::new();
        let mut start = 0;
        while start < text.len() {
            let search_text = &text[start..];
            match self.pattern.find(search_text) {
                Ok(Some(m)) => {
                    let piece = m.as_str().as_bytes();
                    self.bpe_encode_piece(piece, &mut tokens);
                    start += m.end();
                }
                _ => break,
            }
        }
        tokens
    }

    /// Encode a batch of texts in parallel, prepending BOS to each.
    pub fn encode_batch_with_bos(&self, texts: &[String]) -> Vec<Vec<u16>> {
        texts
            .par_iter()
            .map(|text| {
                let mut tokens = Vec::with_capacity(text.len() / 3);
                tokens.push(self.bos_token_id);
                tokens.extend(self.encode(text));
                tokens
            })
            .collect()
    }

    /// BPE encode a single piece (byte sequence from regex match) into token IDs.
    /// Exact port of tiktoken's _byte_pair_merge algorithm.
    fn bpe_encode_piece(&self, piece: &[u8], out: &mut Vec<u16>) {
        if piece.len() == 1 {
            if let Some(&rank) = self.encoder.get(piece) {
                out.push(rank as u16);
            }
            return;
        }

        // Whole piece is a single token
        if let Some(&rank) = self.encoder.get(piece) {
            out.push(rank as u16);
            return;
        }

        // Exact tiktoken algorithm: parts[i] = (byte_offset, rank_of_merging_with_next).
        // Two sentinels at end. Initial ranks are for 2-byte windows.
        let n = piece.len();
        let mut parts: Vec<(usize, u32)> = Vec::with_capacity(n + 1);

        let mut min_rank: (u32, usize) = (u32::MAX, usize::MAX);
        for i in 0..n - 1 {
            let rank = self
                .encoder
                .get(&piece[i..i + 2])
                .copied()
                .unwrap_or(u32::MAX);
            if rank < min_rank.0 {
                min_rank = (rank, i);
            }
            parts.push((i, rank));
        }
        parts.push((n - 1, u32::MAX));
        parts.push((n, u32::MAX));

        // get_rank: after merging at i, what's the rank of merging the new token with its neighbor?
        // Looks at piece[parts[i].0 .. parts[i+3].0]
        let get_rank = |parts: &Vec<(usize, u32)>, i: usize| -> u32 {
            if i + 3 < parts.len() {
                self.encoder
                    .get(&piece[parts[i].0..parts[i + 3].0])
                    .copied()
                    .unwrap_or(u32::MAX)
            } else {
                u32::MAX
            }
        };

        while min_rank.0 != u32::MAX {
            let i = min_rank.1;
            // Update ranks before removing
            if i > 0 {
                parts[i - 1].1 = get_rank(&parts, i - 1);
            }
            parts[i].1 = get_rank(&parts, i);
            parts.remove(i + 1);

            // Find new minimum
            min_rank = (u32::MAX, usize::MAX);
            for (idx, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
                if rank < min_rank.0 {
                    min_rank = (rank, idx);
                }
            }
        }

        // Emit tokens from final segmentation
        for i in 0..parts.len() - 1 {
            let s = parts[i].0;
            let e = parts[i + 1].0;
            if let Some(&rank) = self.encoder.get(&piece[s..e]) {
                out.push(rank as u16);
            }
        }
    }
}

/// Extract a JSON string value starting from the current position.
/// Handles escaped characters (\", \\, etc).
fn extract_json_string(s: &str) -> Option<String> {
    let start = s.find('"')? + 1;
    let mut result = String::new();
    let bytes = s.as_bytes();
    let mut i = start;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            match bytes[i + 1] {
                b'"' => result.push('"'),
                b'\\' => result.push('\\'),
                b'n' => result.push('\n'),
                b'r' => result.push('\r'),
                b't' => result.push('\t'),
                other => {
                    result.push('\\');
                    result.push(other as char);
                }
            }
            i += 2;
        } else if bytes[i] == b'"' {
            return Some(result);
        } else {
            result.push(bytes[i] as char);
            i += 1;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_tok() -> Tokenizer {
        let home = std::env::var("HOME").unwrap();
        let tokenizer_dir = format!("{home}/.cache/autoresearch/tokenizer");
        Tokenizer::load(&tokenizer_dir).expect("load tokenizer")
    }

    #[test]
    fn test_load() {
        let tok = load_tok();
        assert_eq!(tok.bos_token_id(), 8188);
        assert_eq!(tok.encoder.len(), 8188);
    }

    /// Validate against Python tiktoken output (ground truth).
    #[test]
    fn test_encode_matches_python() {
        let tok = load_tok();
        // (input_text, expected_token_ids) — from Python tiktoken encode_ordinary
        let cases: &[(&str, &[u16])] = &[
            ("Hello", &[72, 462, 111]),
            ("a", &[97]),
            ("", &[]),
            ("Hello world!", &[72, 462, 111, 997, 33]),
            ("Numbers: 123.", &[78, 373, 1567, 58, 32, 1133, 51, 46]),
            (
                "The quick brown fox jumps over the lazy dog.",
                &[488, 1818, 3764, 278, 1532, 527, 4972, 642, 262, 300, 1654, 121, 960, 46],
            ),
            ("  spaces  ", &[32, 5081, 1452]),
            ("line1\nline2\nline3", &[1326, 49, 10, 1326, 50, 10, 1326, 51]),
            (
                "It's a test. I've been here. They're going.",
                &[1300, 387, 257, 1131, 46, 332, 1411, 740, 1454, 46, 886, 894, 1461, 46],
            ),
            (
                "def foo(x):\n    return x * 2",
                &[100, 853, 278, 505, 40, 120, 41, 1114, 6075, 3270, 2879, 4153, 32, 50],
            ),
            (
                "Hello world! Numbers: 123. Unicode: \u{4f60}\u{597d}",
                &[72, 462, 111, 997, 33, 486, 373, 1567, 58, 32, 1133, 51, 46, 1000, 290, 1578, 58, 32, 228, 189, 160, 229, 165, 189],
            ),
        ];

        for (text, expected) in cases {
            let got = tok.encode(text);
            assert_eq!(
                &got, expected,
                "mismatch for {:?}: got {:?}, expected {:?}",
                text, got, expected
            );
        }
    }

    #[test]
    fn test_batch_with_bos() {
        let tok = load_tok();
        let texts = vec!["Hello world".to_string(), "test".to_string()];
        let batch = tok.encode_batch_with_bos(&texts);
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0][0], 8188); // BOS
        assert_eq!(batch[1][0], 8188); // BOS
        // Rest should match encode output
        assert_eq!(&batch[0][1..], &tok.encode("Hello world")[..]);
        assert_eq!(&batch[1][1..], &tok.encode("test")[..]);
    }
}
