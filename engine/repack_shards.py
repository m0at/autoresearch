"""
Repack parquet data into binary shards using Python's exact best-fit packing.

Reads original parquet files, tokenizes with BOS, best-fit packs into fixed-length
rows (2049 tokens), and emits binary shards in the format the Rust engine expects.

Binary shard format (little-endian):
  Header (20 bytes): magic "TKNS" | version u32 | vocab_size u32 | seq_len u32 | num_rows u32
  Body: num_rows * seq_len * 2 bytes (u16 tokens)

Usage:
  python3 repack_shards.py --input /root/.cache/autoresearch/data --output /root/.cache/autoresearch/shards
  python3 repack_shards.py --input /root/.cache/autoresearch/data --output /root/.cache/autoresearch/shards --rows-per-shard 500
"""

import os
import sys
import struct
import time
import json
import argparse
import pickle

import requests
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Constants — must match Python reference exactly
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048
ROW_CAPACITY = MAX_SEQ_LEN + 1  # 2049 tokens per row (input[0..2048] + target shift)
VOCAB_SIZE = 8192
SHARD_VERSION = 1
SHARD_MAGIC = b"TKNS"
MAX_SHARD = 6542
VAL_SHARD = MAX_SHARD
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
BUFFER_SIZE = 1000  # same as Python's make_dataloader

# ---------------------------------------------------------------------------
# Tokenizer loading (reuses prepare.py's saved tokenizer)
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_dir):
    tokenizer_pkl = os.path.join(tokenizer_dir, "tokenizer.pkl")
    with open(tokenizer_pkl, "rb") as f:
        enc = pickle.load(f)
    bos_token_id = enc.encode_single_token("<|reserved_0|>")
    return enc, bos_token_id


# ---------------------------------------------------------------------------
# Document iterator — yields batches of raw text from parquet files
# ---------------------------------------------------------------------------

HF_BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"


def list_parquet_files(data_dir):
    files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    )
    return [os.path.join(data_dir, f) for f in files]


def download_parquet(index, cache_dir):
    """Download one parquet shard from HuggingFace, return path. Retries on failure."""
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(cache_dir, filename)
    if os.path.exists(filepath):
        return filepath
    url = f"{HF_BASE_URL}/{filename}"
    for attempt in range(5):
        try:
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
            tmp = filepath + ".tmp"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(tmp, filepath)
            return filepath
        except Exception as e:
            if attempt == 4:
                raise RuntimeError(f"Failed to download {filename} after 5 attempts: {e}")
            time.sleep(2 ** attempt)
    return filepath


def document_batches(parquet_paths, tokenizer_batch_size=128):
    """Yield batches of raw text strings from parquet files. Single pass."""
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            batch = rg.column("text").to_pylist()
            for i in range(0, len(batch), tokenizer_batch_size):
                yield batch[i:i + tokenizer_batch_size]


def streaming_document_batches(shard_indices, cache_dir, tokenizer_batch_size=128, cleanup=True):
    """Stream parquets from HuggingFace: download, yield text, delete. One at a time."""
    os.makedirs(cache_dir, exist_ok=True)
    for idx in shard_indices:
        filepath = download_parquet(idx, cache_dir)
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            batch = rg.column("text").to_pylist()
            for i in range(0, len(batch), tokenizer_batch_size):
                yield batch[i:i + tokenizer_batch_size]
        if cleanup:
            try:
                os.remove(filepath)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Best-fit packing — exact replica of Python's make_dataloader logic
# ---------------------------------------------------------------------------

def pack_rows(doc_iter, enc, bos_token_id, max_rows=None):
    """
    Best-fit pack tokenized documents into fixed-length rows.

    Exactly replicates the packing logic in prepare.py make_dataloader:
    1. Maintain a buffer of tokenized docs (each prepended with BOS)
    2. For each row position, find the largest doc that fits remaining space
    3. If no doc fits, crop the shortest doc to fill exactly
    4. 100% utilization (no padding)

    Yields lists of ints (each list has exactly ROW_CAPACITY tokens).
    """
    doc_buffer = []
    rows_emitted = 0

    def refill_buffer():
        try:
            text_batch = next(doc_iter)
        except StopIteration:
            return False
        # Batch tokenize then prepend BOS — matches Python's encode(text, prepend=bos_token)
        token_lists = enc.encode_ordinary_batch(text_batch, num_threads=8)
        for toks in token_lists:
            toks.insert(0, bos_token_id)
        doc_buffer.extend(token_lists)
        return True

    while True:
        if max_rows is not None and rows_emitted >= max_rows:
            return

        row = []
        pos = 0
        while pos < ROW_CAPACITY:
            # Refill to at least BUFFER_SIZE docs
            while len(doc_buffer) < BUFFER_SIZE:
                if not refill_buffer():
                    break
            if len(doc_buffer) == 0:
                # No more documents at all — this shouldn't happen with enough data
                # but handle gracefully: pad remainder with BOS (token 0 equivalent)
                # Actually, just return what we have if pos > 0
                if pos > 0:
                    # Pad with zeros (won't happen with real data)
                    row.extend([0] * (ROW_CAPACITY - pos))
                    pos = ROW_CAPACITY
                    break
                else:
                    return

            remaining = ROW_CAPACITY - pos

            # Find largest doc that fits entirely
            best_idx = -1
            best_len = 0
            for i, doc in enumerate(doc_buffer):
                doc_len = len(doc)
                if doc_len <= remaining and doc_len > best_len:
                    best_idx = i
                    best_len = doc_len

            if best_idx >= 0:
                doc = doc_buffer.pop(best_idx)
                row.extend(doc)
                pos += len(doc)
            else:
                # No doc fits — crop shortest to fill remaining exactly
                shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                doc = doc_buffer.pop(shortest_idx)
                row.extend(doc[:remaining])
                pos += remaining

        assert len(row) == ROW_CAPACITY, f"row length {len(row)} != {ROW_CAPACITY}"
        yield row
        rows_emitted += 1


# ---------------------------------------------------------------------------
# Binary shard writer
# ---------------------------------------------------------------------------

def write_shard(path, rows):
    """Write a list of rows to a binary shard file."""
    num_rows = len(rows)
    if num_rows == 0:
        return
    header = struct.pack("<4sIIII", SHARD_MAGIC, SHARD_VERSION, VOCAB_SIZE, ROW_CAPACITY, num_rows)
    with open(path, "wb") as f:
        f.write(header)
        for row in rows:
            # Pack as little-endian u16
            f.write(struct.pack(f"<{ROW_CAPACITY}H", *row))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Repack parquet data into binary shards with best-fit packing")
    parser.add_argument("--input", default=None, help="Input directory containing shard_*.parquet files (local mode)")
    parser.add_argument("--output", required=True, help="Output directory for binary shard files")
    parser.add_argument("--tokenizer-dir", default=None,
                        help="Tokenizer directory (default: ~/.cache/autoresearch/tokenizer)")
    parser.add_argument("--rows-per-shard", type=int, default=1000,
                        help="Number of rows per output shard (default: 1000)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Max total rows to emit (default: all data)")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both",
                        help="Which split to emit (default: both)")
    parser.add_argument("--stream", action="store_true",
                        help="Stream parquets from HuggingFace (downloads one at a time, deletes after)")
    parser.add_argument("--cache-dir", default="/tmp/repack_cache",
                        help="Temp dir for streaming downloads (default: /tmp/repack_cache)")
    args = parser.parse_args()

    if args.tokenizer_dir is None:
        args.tokenizer_dir = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "tokenizer")

    os.makedirs(args.output, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_dir}")
    enc, bos_token_id = load_tokenizer(args.tokenizer_dir)
    print(f"  vocab_size={enc.n_vocab}, bos_token_id={bos_token_id}")

    if args.stream:
        # Stream mode: download parquets from HuggingFace one at a time
        train_indices = list(range(0, MAX_SHARD))  # 0..6541
        val_indices = [VAL_SHARD]  # 6542

        splits_to_process = []
        if args.split in ("train", "both"):
            splits_to_process.append(("train", train_indices))
        if args.split in ("val", "both"):
            splits_to_process.append(("val", val_indices))

        print(f"Streaming from HuggingFace: {len(train_indices)} train + {len(val_indices)} val parquets")
    else:
        if args.input is None:
            print("ERROR: --input required in local mode (or use --stream)")
            sys.exit(1)
        # Local mode: read from disk
        all_parquets = list_parquet_files(args.input)
        if not all_parquets:
            print(f"ERROR: no parquet files in {args.input}")
            sys.exit(1)

        val_path = os.path.join(args.input, VAL_FILENAME)
        train_parquets = [p for p in all_parquets if p != val_path]
        val_parquets = [val_path] if val_path in all_parquets else []

        print(f"Found {len(all_parquets)} parquet files: {len(train_parquets)} train, {len(val_parquets)} val")

        splits_to_process = []
        if args.split in ("train", "both"):
            splits_to_process.append(("train", train_parquets))
        if args.split in ("val", "both"):
            if val_parquets:
                splits_to_process.append(("val", val_parquets))
            else:
                print(f"WARNING: val parquet {VAL_FILENAME} not found, skipping val split")

    total_train_shards = 0
    total_val_shards = 0
    total_train_rows = 0
    total_val_rows = 0

    for split_entry in splits_to_process:
        split_name = split_entry[0]
        split_data = split_entry[1]

        if args.stream:
            shard_indices = split_data
            print(f"\n--- Streaming {split_name} split ({len(shard_indices)} parquets from HF) ---")
            doc_iter = streaming_document_batches(shard_indices, args.cache_dir)
        else:
            parquet_paths = split_data
            print(f"\n--- Processing {split_name} split ({len(parquet_paths)} parquets) ---")
            doc_iter = document_batches(parquet_paths)

        t0 = time.time()
        row_gen = pack_rows(doc_iter, enc, bos_token_id, max_rows=args.max_rows)

        shard_idx = 0
        total_rows = 0
        shard_rows = []
        shard_paths = []

        for row in row_gen:
            shard_rows.append(row)
            total_rows += 1

            if len(shard_rows) >= args.rows_per_shard:
                shard_name = f"shard_{split_name}_{shard_idx:05d}.bin"
                shard_path = os.path.join(args.output, shard_name)
                write_shard(shard_path, shard_rows)
                shard_paths.append(shard_path)
                if shard_idx % 100 == 0:
                    elapsed = time.time() - t0
                    print(f"  wrote {shard_name} ({len(shard_rows)} rows, {total_rows} total, {elapsed:.1f}s)")
                shard_rows = []
                shard_idx += 1

        # Write remaining rows
        if shard_rows:
            shard_name = f"shard_{split_name}_{shard_idx:05d}.bin"
            shard_path = os.path.join(args.output, shard_name)
            write_shard(shard_path, shard_rows)
            shard_paths.append(shard_path)
            shard_idx += 1

        elapsed = time.time() - t0
        print(f"  {split_name}: {shard_idx} shards, {total_rows} rows, {elapsed:.1f}s")
        tokens = total_rows * ROW_CAPACITY
        print(f"  {split_name}: {tokens:,} tokens ({tokens / 1e6:.1f}M)")

        if split_name == "train":
            total_train_shards = shard_idx
            total_train_rows = total_rows
        else:
            total_val_shards = shard_idx
            total_val_rows = total_rows

    # Now rename: train shards first, then val shards, sequential numbering.
    # The Rust engine splits by index: shards [0..num_train) = train, [num_train..) = val.
    # So we name them shard_00000.bin, shard_00001.bin, ... with train first.
    print(f"\n--- Renaming shards to sequential order ---")
    print(f"  Train: {total_train_shards} shards ({total_train_rows} rows)")
    print(f"  Val:   {total_val_shards} shards ({total_val_rows} rows)")

    # Collect all temp shard files
    train_files = sorted(
        f for f in os.listdir(args.output)
        if f.startswith("shard_train_") and f.endswith(".bin")
    )
    val_files = sorted(
        f for f in os.listdir(args.output)
        if f.startswith("shard_val_") and f.endswith(".bin")
    )

    # Rename train shards to shard_00000.bin, shard_00001.bin, ...
    for i, fname in enumerate(train_files):
        src = os.path.join(args.output, fname)
        dst = os.path.join(args.output, f"shard_{i:05d}.bin")
        os.rename(src, dst)

    # Rename val shards continuing from train count
    for i, fname in enumerate(val_files):
        src = os.path.join(args.output, fname)
        dst = os.path.join(args.output, f"shard_{total_train_shards + i:05d}.bin")
        os.rename(src, dst)

    # Write manifest
    manifest = {
        "version": 1,
        "vocab_size": VOCAB_SIZE,
        "seq_len": ROW_CAPACITY,
        "rows_per_shard": args.rows_per_shard,
        "num_train_shards": total_train_shards,
        "num_val_shards": total_val_shards,
        "total_shards": total_train_shards + total_val_shards,
        "total_train_rows": total_train_rows,
        "total_val_rows": total_val_rows,
    }
    manifest_path = os.path.join(args.output, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to {manifest_path}")
    print(json.dumps(manifest, indent=2))
    print(f"\nTo use with Rust engine, set num_train_shards = {total_train_shards}")
    print("Done.")


if __name__ == "__main__":
    main()
