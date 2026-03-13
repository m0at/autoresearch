/// Rust replacement for feeder.py — streams packed training rows to stdout.
///
/// Pipeline: ParquetStreamer → tokenize batches → Packer → binary stdout
///
/// Binary protocol (stdout):
///   Each row: 2049 little-endian u16 tokens (4098 bytes)
///   No headers, no framing.

use std::io::{self, BufWriter, Write};
use std::time::Instant;

use autoresearch_brain::feeder::download::ParquetStreamer;
use autoresearch_brain::feeder::packing::{Packer, ROW_CAPACITY};
use autoresearch_brain::feeder::tokenizer::Tokenizer;

const MAX_SHARD: usize = 6542;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut stream = false;
    let mut input_dir: Option<String> = None;
    let mut prefetch: usize = 4;
    let mut tokenizer_dir: Option<String> = None;
    let mut cache_dir = String::from("/tmp/feeder_cache");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--stream" => stream = true,
            "--input" => {
                i += 1;
                input_dir = Some(args[i].clone());
            }
            "--prefetch" => {
                i += 1;
                prefetch = args[i].parse().expect("--prefetch expects a number");
            }
            "--tokenizer-dir" => {
                i += 1;
                tokenizer_dir = Some(args[i].clone());
            }
            "--cache-dir" => {
                i += 1;
                cache_dir = args[i].clone();
            }
            _ => {
                eprintln!("[feeder] unknown arg: {}", args[i]);
            }
        }
        i += 1;
    }

    let tokenizer_dir = tokenizer_dir.unwrap_or_else(|| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
        format!("{home}/.cache/autoresearch/tokenizer")
    });

    // Load tokenizer
    let tokenizer = Tokenizer::load(&tokenizer_dir).expect("failed to load tokenizer");
    eprintln!(
        "[feeder] tokenizer loaded, bos={}",
        tokenizer.bos_token_id()
    );

    // Set up document stream
    let doc_batches: Box<dyn Iterator<Item = Vec<String>>> = if stream {
        let indices: Vec<usize> = (0..MAX_SHARD).collect();
        Box::new(ParquetStreamer::new(indices, &cache_dir, prefetch, 128))
    } else if let Some(ref dir) = input_dir {
        let indices = local_shard_indices(dir);
        Box::new(
            ParquetStreamer::new(indices, dir, prefetch, 128).with_cleanup(false),
        )
    } else {
        eprintln!("ERROR: --stream or --input required");
        std::process::exit(1);
    };

    // Pipeline: text batches → tokenize (yields Vec<Vec<u16>> per batch) → pack → write
    let token_batches = doc_batches.map(move |batch| tokenizer.encode_batch_with_bos(&batch));
    let packer = Packer::new(token_batches);

    // Write binary to stdout
    let stdout = io::stdout().lock();
    let mut writer = BufWriter::with_capacity(4098 * 256, stdout);
    let t0 = Instant::now();
    let mut rows_written: u64 = 0;

    for row in packer {
        // Write 2049 little-endian u16 values
        for &token in row.iter() {
            writer.write_all(&token.to_le_bytes()).unwrap();
        }
        rows_written += 1;
        if rows_written % 10000 == 0 {
            let elapsed = t0.elapsed().as_secs_f64();
            let rate = rows_written as f64 / elapsed;
            eprintln!("[feeder] {rows_written} rows, {rate:.0} rows/s, {elapsed:.0}s");
        }
    }

    writer.flush().unwrap();
    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!("[feeder] done: {rows_written} rows in {elapsed:.1}s");
}

/// List shard indices from a local directory of parquet files.
/// Excludes the val shard (MAX_SHARD = 6542).
fn local_shard_indices(dir: &str) -> Vec<usize> {
    let mut indices = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(rest) = name.strip_prefix("shard_") {
                if let Some(num_str) = rest.strip_suffix(".parquet") {
                    if let Ok(idx) = num_str.parse::<usize>() {
                        if idx != MAX_SHARD {
                            indices.push(idx);
                        }
                    }
                }
            }
        }
    }
    indices.sort();
    indices
}
