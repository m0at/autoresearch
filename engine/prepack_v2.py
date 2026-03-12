"""
Pre-pack exactly the training data needed for N training steps, then stop.

Orchestrates the pipeline: quality ordering -> streaming -> dedup -> dense packing
-> shard writing. Packs only --target-rows rows (default 200K for 700 steps),
saving ~55 minutes vs prepack.py which processes all 6542 parquets.

Each shard: 20-byte header + up to 1000 rows x 2049 u16 tokens.
Header: b"TKNS" + version(u32) + vocab_size(u32) + seq_len(u32) + num_rows(u32)

Usage:
  python3 prepack_v2.py --output-dir /path/to/shards --val-dir /path/to/val \
      --dedup --quality-sort --target-rows 200000 --buffer-size 5000 --num-epochs 2
"""

import os
import sys
import struct
import time
import json
import math
import shutil
import random
import argparse

from feeder import (
    load_tokenizer, streaming_document_batches,
    VOCAB_SIZE, ROW_CAPACITY, MAX_SHARD,
)
from dense_pack import dense_pack_rows
from multi_epoch import multi_epoch_doc_iter

ROWS_PER_SHARD = 1000
SHARD_MAGIC = b"TKNS"
SHARD_VERSION = 1
ROW_STRUCT = struct.Struct(f"<{ROW_CAPACITY}H")

# Average packed rows per parquet (empirical: 394 parquets x 2 epochs -> 116K rows = ~147 rows/parquet-pass)
ROWS_PER_PARQUET_ESTIMATE = 147


def write_shard(path, rows):
    """Write a binary shard file with header + packed rows."""
    num_rows = len(rows)
    header = struct.pack("<4sIIII", SHARD_MAGIC, SHARD_VERSION, VOCAB_SIZE, ROW_CAPACITY, num_rows)
    with open(path, "wb") as f:
        f.write(header)
        for row in rows:
            f.write(ROW_STRUCT.pack(*row))


def count_existing_train_shards(output_dir, num_val):
    """Count existing train shards for resume support."""
    if not os.path.exists(output_dir):
        return 0
    existing = sorted(
        f for f in os.listdir(output_dir)
        if f.startswith("shard_") and f.endswith(".bin")
    )
    return max(0, len(existing) - num_val)


def estimate_parquets_needed(target_rows, num_epochs):
    """Estimate how many unique parquets to download."""
    rows_per_epoch = target_rows / max(num_epochs, 1)
    parquets_needed = math.ceil(rows_per_epoch / ROWS_PER_PARQUET_ESTIMATE)
    # Add 20% safety margin for dedup losses
    parquets_needed = math.ceil(parquets_needed * 1.2)
    return min(parquets_needed, MAX_SHARD)


def main():
    parser = argparse.ArgumentParser(description="Pre-pack exactly the training data needed")
    parser.add_argument("--output-dir", required=True, help="Directory to write all shard files")
    parser.add_argument("--val-dir", required=True, help="Directory containing existing val shards")
    parser.add_argument("--target-rows", type=int, default=200000,
                        help="Number of packed rows to produce (default: 200000)")
    parser.add_argument("--buffer-size", type=int, default=5000,
                        help="Dense packing buffer size (default: 5000)")
    parser.add_argument("--num-epochs", type=int, default=2,
                        help="Number of passes over the curated parquet subset (default: 2)")
    parser.add_argument("--dedup", action="store_true", help="Enable near-duplicate filtering")
    parser.add_argument("--quality-sort", action="store_true",
                        help="Rank parquets by quality score, front-load best data")
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--cache-dir", default="/tmp/feeder_cache")
    parser.add_argument("--workers", type=int, default=8, help="Download/scoring workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for parquet ordering")
    args = parser.parse_args()

    if args.tokenizer_dir is None:
        args.tokenizer_dir = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "tokenizer")

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Count val shards
    # -------------------------------------------------------------------------
    val_bins = sorted(
        f for f in os.listdir(args.val_dir)
        if f.startswith("shard_") and f.endswith(".bin")
    )
    num_val = len(val_bins)
    print(f"[prepack_v2] val shards: {num_val} in {args.val_dir}", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Resume support
    # -------------------------------------------------------------------------
    start_shard = count_existing_train_shards(args.output_dir, num_val)
    rows_already = start_shard * ROWS_PER_SHARD
    if rows_already >= args.target_rows:
        print(f"[prepack_v2] already have {rows_already} rows >= target {args.target_rows}, skipping to val copy", file=sys.stderr)
    elif start_shard > 0:
        print(f"[prepack_v2] resuming: {start_shard} existing shards ({rows_already} rows)", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Load tokenizer
    # -------------------------------------------------------------------------
    enc, bos_token_id = load_tokenizer(args.tokenizer_dir)
    print(f"[prepack_v2] tokenizer loaded, vocab={enc.n_vocab}, bos={bos_token_id}", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Step (a): Select and order parquet indices
    # -------------------------------------------------------------------------
    num_parquets = estimate_parquets_needed(args.target_rows, args.num_epochs)

    if args.quality_sort:
        from quality_order import rank_parquets
        print(f"[prepack_v2] quality-sorting all {MAX_SHARD} parquets...", file=sys.stderr)
        all_indices = list(range(MAX_SHARD))
        ranked = rank_parquets(all_indices, args.cache_dir, workers=args.workers)
        parquet_indices = ranked[:num_parquets]
        print(f"[prepack_v2] selected top {num_parquets} parquets by quality", file=sys.stderr)
    else:
        random.seed(args.seed)
        all_indices = list(range(MAX_SHARD))
        random.shuffle(all_indices)
        parquet_indices = all_indices[:num_parquets]
        print(f"[prepack_v2] selected {num_parquets} parquets (shuffled, seed={args.seed})", file=sys.stderr)

    print(f"[prepack_v2] pipeline: {num_parquets} parquets x {args.num_epochs} epochs"
          f" -> {'dedup -> ' if args.dedup else ''}dense_pack(buf={args.buffer_size})"
          f" -> {args.target_rows} rows target", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Step (b)-(f): Build the document -> packed row pipeline
    # -------------------------------------------------------------------------
    if rows_already < args.target_rows:

        base_doc_iter = streaming_document_batches(
            parquet_indices, args.cache_dir,
            prefetch_workers=args.workers, cleanup=(args.num_epochs <= 1),
        )

        # Multi-epoch: collect docs from parquets once, reshuffle per epoch
        if args.num_epochs > 1:
            doc_iter = multi_epoch_doc_iter(base_doc_iter, num_parquets, args.num_epochs)
        else:
            doc_iter = base_doc_iter

        # Optional dedup
        if args.dedup:
            from dedup import exact_dedup_documents
            doc_iter = exact_dedup_documents(doc_iter)
            print(f"[prepack_v2] dedup enabled (exact SHA-256)", file=sys.stderr)

        # Dense packing
        row_iter = dense_pack_rows(doc_iter, enc, bos_token_id, buffer_size=args.buffer_size)

        # -----------------------------------------------------------------
        # Step (g): Write shards, stop at target_rows
        # -----------------------------------------------------------------
        shard_idx = 0
        rows_written = 0
        current_shard_rows = []
        t_start = time.time()

        for row in row_iter:
            # Skip rows for resume
            if shard_idx < start_shard:
                current_shard_rows.append(row)
                if len(current_shard_rows) == ROWS_PER_SHARD:
                    current_shard_rows = []
                    shard_idx += 1
                    if shard_idx == start_shard:
                        elapsed = time.time() - t_start
                        print(f"[prepack_v2] skipped to shard {start_shard} ({elapsed:.0f}s)", file=sys.stderr)
                continue

            rows_written += 1
            current_shard_rows.append(row)

            if len(current_shard_rows) == ROWS_PER_SHARD:
                path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.bin")
                write_shard(path, current_shard_rows)
                elapsed = time.time() - t_start
                rate = rows_written / elapsed if elapsed > 0 else 0
                print(f"[prepack_v2] shard {shard_idx:5d} | {rows_written:>7d}/{args.target_rows} rows"
                      f" | {rate:.0f} rows/s | {elapsed:.0f}s", file=sys.stderr)
                current_shard_rows = []
                shard_idx += 1

            # Stop once we've hit the target
            if rows_written >= args.target_rows:
                print(f"[prepack_v2] reached target {args.target_rows} rows, stopping", file=sys.stderr)
                break

        # Write final partial shard if any
        if current_shard_rows:
            path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.bin")
            write_shard(path, current_shard_rows)
            elapsed = time.time() - t_start
            print(f"[prepack_v2] shard {shard_idx:5d} (partial, {len(current_shard_rows)} rows) | {elapsed:.0f}s",
                  file=sys.stderr)
            shard_idx += 1

        num_train_shards = shard_idx
        elapsed = time.time() - t_start
        print(f"\n[prepack_v2] train done: {num_train_shards} shards, {rows_written} rows, {elapsed:.0f}s",
              file=sys.stderr)
    else:
        # Already have enough shards
        num_train_shards = start_shard
        rows_written = rows_already

    # -------------------------------------------------------------------------
    # Step (h): Copy val shards, renumbered after train shards
    # -------------------------------------------------------------------------
    print(f"[prepack_v2] copying {num_val} val shards from {args.val_dir}...", file=sys.stderr)
    total_val_rows = 0
    for i, vf in enumerate(val_bins):
        src = os.path.join(args.val_dir, vf)
        dst_idx = num_train_shards + i
        dst = os.path.join(args.output_dir, f"shard_{dst_idx:05d}.bin")
        shutil.copy2(src, dst)
        with open(src, "rb") as f:
            f.seek(16)
            nr = struct.unpack("<I", f.read(4))[0]
            total_val_rows += nr
    print(f"[prepack_v2] copied {num_val} val shards ({total_val_rows} rows)", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Write manifest
    # -------------------------------------------------------------------------
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
        "config": {
            "target_rows": args.target_rows,
            "buffer_size": args.buffer_size,
            "num_epochs": args.num_epochs,
            "dedup": args.dedup,
            "quality_sort": args.quality_sort,
            "seed": args.seed,
            "num_parquets": len(parquet_indices),
        },
    }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[prepack_v2] manifest: {json.dumps(manifest, indent=2)}", file=sys.stderr)
    print(f"\n[prepack_v2] ALL DONE. Run with:", file=sys.stderr)
    print(f"  NUM_TRAIN_SHARDS={num_train_shards} cargo run --release --bin autoresearch-engine -- train --data-dir {args.output_dir}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
