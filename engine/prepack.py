"""
Pre-pack training data into binary shards for the Rust engine.

Downloads all 6541 train parquets from HuggingFace, tokenizes with BOS,
best-fit packs into 2049-token rows, and writes binary shard files.

Each shard: 20-byte header + 1000 rows × 2049 u16 tokens.
Header: b"TKNS" + version(u32) + vocab_size(u32) + seq_len(u32) + num_rows(u32)

Writes train shards to --output-dir, then copies val shards from --val-dir
and renumbers them to come after train shards. The engine reads train=[0..N),
val=[N..) from the same directory sorted by name.

Usage:
  python3 prepack.py --output-dir /root/.cache/autoresearch/shards_train \
                     --val-dir /root/.cache/autoresearch/shards_packed
"""

import os
import sys
import struct
import time
import json
import shutil
import argparse

# Reuse feeder.py's packing and download logic
from feeder import (
    load_tokenizer, streaming_document_batches, pack_rows,
    ROW_CAPACITY, VOCAB_SIZE, MAX_SHARD,
)

ROWS_PER_SHARD = 1000
SHARD_MAGIC = b"TKNS"
SHARD_VERSION = 1
ROW_STRUCT = struct.Struct(f"<{ROW_CAPACITY}H")


def write_shard(path, rows):
    """Write a binary shard file with header + packed rows."""
    num_rows = len(rows)
    header = struct.pack("<4sIIII", SHARD_MAGIC, SHARD_VERSION, VOCAB_SIZE, ROW_CAPACITY, num_rows)
    with open(path, "wb") as f:
        f.write(header)
        for row in rows:
            f.write(ROW_STRUCT.pack(*row))


def main():
    parser = argparse.ArgumentParser(description="Pre-pack training shards")
    parser.add_argument("--output-dir", required=True, help="Directory to write all shard files")
    parser.add_argument("--val-dir", required=True, help="Directory containing existing val shards")
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--cache-dir", default="/tmp/feeder_cache")
    parser.add_argument("--workers", type=int, default=8, help="Download workers")
    args = parser.parse_args()

    if args.tokenizer_dir is None:
        args.tokenizer_dir = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "tokenizer")

    os.makedirs(args.output_dir, exist_ok=True)

    enc, bos_token_id = load_tokenizer(args.tokenizer_dir)
    print(f"[prepack] tokenizer loaded, vocab={enc.n_vocab}, bos={bos_token_id}", file=sys.stderr)

    # Check for existing train shards to support resume
    existing_train = sorted(
        f for f in os.listdir(args.output_dir)
        if f.startswith("shard_") and f.endswith(".bin")
    )
    # Don't count val shards that might already be copied
    val_bins = sorted(
        f for f in os.listdir(args.val_dir)
        if f.startswith("shard_") and f.endswith(".bin")
    )
    num_val = len(val_bins)
    start_shard = max(0, len(existing_train) - num_val)  # subtract any copied val shards

    if start_shard > 0:
        print(f"[prepack] found {start_shard} existing train shards, resuming from shard {start_shard}", file=sys.stderr)

    # Stream all train parquets (0..6541) in deterministic shuffled order
    import random
    random.seed(42)
    train_indices = list(range(0, MAX_SHARD))
    random.shuffle(train_indices)

    print(f"[prepack] streaming {len(train_indices)} parquets → {args.output_dir}", file=sys.stderr)

    doc_iter = streaming_document_batches(
        train_indices, args.cache_dir,
        prefetch_workers=args.workers, cleanup=True,
    )

    shard_idx = 0
    rows_written = 0
    current_shard_rows = []
    t_start = time.time()

    for row in pack_rows(doc_iter, enc, bos_token_id):
        rows_written += 1

        # Skip rows for resume
        if shard_idx < start_shard:
            current_shard_rows.append(row)
            if len(current_shard_rows) == ROWS_PER_SHARD:
                current_shard_rows = []
                shard_idx += 1
                if shard_idx == start_shard:
                    elapsed = time.time() - t_start
                    print(f"[prepack] skipped to shard {start_shard} ({rows_written} rows, {elapsed:.0f}s)", file=sys.stderr)
            continue

        current_shard_rows.append(row)
        if len(current_shard_rows) == ROWS_PER_SHARD:
            path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.bin")
            write_shard(path, current_shard_rows)
            elapsed = time.time() - t_start
            rate = rows_written / elapsed if elapsed > 0 else 0
            print(f"[prepack] shard {shard_idx:5d} | {rows_written} rows | {rate:.0f} rows/s | {elapsed:.0f}s", file=sys.stderr)
            current_shard_rows = []
            shard_idx += 1

    # Write final partial shard
    if current_shard_rows:
        path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.bin")
        write_shard(path, current_shard_rows)
        elapsed = time.time() - t_start
        print(f"[prepack] shard {shard_idx:5d} (partial, {len(current_shard_rows)} rows) | {elapsed:.0f}s", file=sys.stderr)
        shard_idx += 1

    num_train_shards = shard_idx
    elapsed = time.time() - t_start
    print(f"\n[prepack] train done: {num_train_shards} shards, {rows_written} rows, {elapsed:.0f}s", file=sys.stderr)

    # Copy val shards, renumbered to come after train shards
    print(f"[prepack] copying {num_val} val shards from {args.val_dir}...", file=sys.stderr)
    total_val_rows = 0
    for i, vf in enumerate(val_bins):
        src = os.path.join(args.val_dir, vf)
        dst_idx = num_train_shards + i
        dst = os.path.join(args.output_dir, f"shard_{dst_idx:05d}.bin")
        shutil.copy2(src, dst)
        # Count rows in val shard
        with open(src, "rb") as f:
            f.seek(16)
            nr = struct.unpack("<I", f.read(4))[0]
            total_val_rows += nr
    print(f"[prepack] copied {num_val} val shards ({total_val_rows} rows)", file=sys.stderr)

    # Write manifest
    manifest = {
        "version": 1,
        "vocab_size": VOCAB_SIZE,
        "seq_len": ROW_CAPACITY,
        "rows_per_shard": ROWS_PER_SHARD,
        "num_train_shards": num_train_shards,
        "num_val_shards": num_val,
        "total_shards": num_train_shards + num_val,
        "total_train_rows": rows_written,
        "total_val_rows": total_val_rows,
    }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[prepack] manifest: {json.dumps(manifest, indent=2)}", file=sys.stderr)
    print(f"\n[prepack] ALL DONE. Run with:", file=sys.stderr)
    print(f"  NUM_TRAIN_SHARDS={num_train_shards} cargo run --release --bin autoresearch-engine -- train --data-dir {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
