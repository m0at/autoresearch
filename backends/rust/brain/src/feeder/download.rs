use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use anyhow::{bail, Context, Result};

const HF_BASE_URL: &str =
    "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main";
const MAX_RETRIES: u32 = 5;

/// Stream document batches from HuggingFace parquet shards.
/// Downloads parquets in parallel (prefetch_workers threads), reads the "text" column,
/// yields batches of text strings. Cleans up downloaded files after reading.
pub struct ParquetStreamer {
    rx: mpsc::Receiver<PathBuf>,
    batch_size: usize,
    /// Leftover texts from a row group that didn't fill a complete batch.
    pending: Vec<String>,
    /// Download thread handle — joined on drop.
    _dl_handle: Option<thread::JoinHandle<()>>,
    /// Whether to delete parquet files after reading.
    cleanup: bool,
}

impl ParquetStreamer {
    /// Create a new streamer for the given shard indices.
    /// cache_dir: where to download parquet files temporarily
    /// prefetch_workers: number of parallel download threads (default 4)
    /// batch_size: number of text documents per yielded batch (default 128)
    pub fn new(
        shard_indices: Vec<usize>,
        cache_dir: &str,
        prefetch_workers: usize,
        batch_size: usize,
    ) -> Self {
        let cache = cache_dir.to_string();
        fs::create_dir_all(&cache).ok();

        // Channel capacity = workers * 2 to limit prefetch ahead
        let (tx, rx) = mpsc::sync_channel::<PathBuf>(prefetch_workers * 2);

        let handle = thread::spawn(move || {
            prefetch_loop(&shard_indices, &cache, prefetch_workers, tx);
        });

        ParquetStreamer {
            rx,
            batch_size,
            pending: Vec::new(),
            _dl_handle: Some(handle),
            cleanup: true,
        }
    }

    /// Set whether to delete parquet files after reading (default: true).
    /// Use `false` for --input mode where files should be preserved.
    pub fn with_cleanup(mut self, cleanup: bool) -> Self {
        self.cleanup = cleanup;
        self
    }
}

impl Iterator for ParquetStreamer {
    type Item = Vec<String>;

    fn next(&mut self) -> Option<Vec<String>> {
        loop {
            // Drain pending into batches
            if self.pending.len() >= self.batch_size {
                let batch: Vec<String> = self.pending.drain(..self.batch_size).collect();
                return Some(batch);
            }

            // Try to get next parquet file
            let filepath = match self.rx.recv() {
                Ok(p) => p,
                Err(_) => {
                    // Channel closed — no more shards. Flush remaining.
                    if self.pending.is_empty() {
                        return None;
                    }
                    return Some(self.pending.drain(..).collect());
                }
            };

            // Read all texts from this parquet
            match read_parquet_texts(&filepath) {
                Ok(texts) => self.pending.extend(texts),
                Err(e) => {
                    eprintln!(
                        "[download] WARN: skipping corrupt parquet {}: {e}",
                        filepath.display()
                    );
                }
            }

            // Cleanup
            if self.cleanup {
                let _ = fs::remove_file(&filepath);
            }
        }
    }
}

/// Download parquets in parallel chunks, send paths on channel in shard order.
fn prefetch_loop(
    indices: &[usize],
    cache_dir: &str,
    num_workers: usize,
    tx: mpsc::SyncSender<PathBuf>,
) {
    let chunk_size = num_workers * 2;
    for chunk in indices.chunks(chunk_size) {
        // Spawn workers for this chunk
        let handles: Vec<_> = chunk
            .iter()
            .map(|&idx| {
                let cache = cache_dir.to_string();
                thread::spawn(move || download_parquet(idx, &cache))
            })
            .collect();

        // Collect results in order (handles vec preserves chunk order)
        let mut results: Vec<Option<PathBuf>> = Vec::with_capacity(handles.len());
        for h in handles {
            match h.join() {
                Ok(Ok(path)) => results.push(Some(path)),
                Ok(Err(e)) => {
                    eprintln!("[download] ERROR: {e}");
                    results.push(None);
                }
                Err(_) => {
                    eprintln!("[download] ERROR: download thread panicked");
                    results.push(None);
                }
            }
        }

        // Send in order, skip failures
        for path in results.into_iter().flatten() {
            if tx.send(path).is_err() {
                return; // receiver dropped
            }
        }
    }
    // tx drops here → channel closes → receiver gets Err
}

/// Download a single parquet shard with retry + exponential backoff.
fn download_parquet(index: usize, cache_dir: &str) -> Result<PathBuf> {
    let filename = format!("shard_{index:05}.parquet");
    let filepath = Path::new(cache_dir).join(&filename);

    // Already cached
    if filepath.exists() {
        return Ok(filepath);
    }

    let url = format!("{HF_BASE_URL}/{filename}");
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .context("failed to build HTTP client")?;

    for attempt in 0..MAX_RETRIES {
        match try_download(&client, &url, &filepath) {
            Ok(()) => return Ok(filepath),
            Err(e) => {
                if attempt == MAX_RETRIES - 1 {
                    bail!("failed to download {filename} after {MAX_RETRIES} attempts: {e}");
                }
                let backoff = Duration::from_secs(1 << attempt);
                eprintln!(
                    "[download] attempt {}/{MAX_RETRIES} for {filename} failed: {e}, retrying in {backoff:?}",
                    attempt + 1
                );
                thread::sleep(backoff);
            }
        }
    }
    unreachable!()
}

/// Single download attempt: stream to .tmp then atomic rename.
fn try_download(client: &reqwest::blocking::Client, url: &str, dest: &Path) -> Result<()> {
    let mut resp = client
        .get(url)
        .send()
        .context("HTTP request failed")?
        .error_for_status()
        .context("HTTP error status")?;

    let tmp = dest.with_extension("parquet.tmp");
    let mut file = fs::File::create(&tmp).context("create tmp file")?;

    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = std::io::Read::read(&mut resp, &mut buf).context("read HTTP body")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).context("write to tmp file")?;
    }
    file.flush()?;
    drop(file);

    fs::rename(&tmp, dest).context("rename tmp → final")?;
    Ok(())
}

/// Read all text values from the "text" column of a parquet file.
fn read_parquet_texts(path: &Path) -> Result<Vec<String>> {
    use parquet::file::reader::FileReader;
    use parquet::file::reader::SerializedFileReader;
    use parquet::record::RowAccessor;

    let file = fs::File::open(path).context("open parquet")?;
    let reader = SerializedFileReader::new(file).context("parse parquet")?;
    let metadata = reader.metadata();

    // Find the "text" column index
    let schema = metadata.file_metadata().schema_descr();
    let text_col = schema
        .columns()
        .iter()
        .position(|c| c.name() == "text")
        .context("no 'text' column in parquet")?;

    let num_row_groups = metadata.num_row_groups();
    let mut texts = Vec::new();

    for rg_idx in 0..num_row_groups {
        let row_group = reader.get_row_group(rg_idx).context("read row group")?;
        let num_rows = row_group.metadata().num_rows() as usize;
        texts.reserve(num_rows);

        // Read via row iterator — simple and correct
        let row_iter = row_group
            .get_row_iter(None)
            .context("get row iterator")?;

        for row_result in row_iter {
            let row = row_result.context("read row")?;
            let val = row.get_string(text_col).context("get text field")?;
            texts.push(val.to_string());
        }
    }

    Ok(texts)
}
