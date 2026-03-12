"""
Pre-pack training data with optional "shard rinsing" — strip web junk before packing.

Extends prepack_v2.py with:
  --rinse            Apply strip_document() to each text before packing (default: off)
  --target-shards N  Convenience flag: sets target_rows = N * 1000
  --no-quality-sort  Disable quality ordering (random shuffle, useful for shard count sweeps)

With --rinse --target-shards 150 you get 150 shards of dense, clean data equivalent
to perhaps 200 shards of raw data: strip stats are logged at the end.

Each shard: 20-byte header + up to 1000 rows x 2049 u16 tokens.
Header: b"TKNS" + version(u32) + vocab_size(u32) + seq_len(u32) + num_rows(u32)

Usage:
  python3 prepack_v3.py --output-dir /path/to/shards --val-dir /path/to/val \\
      --dedup --quality-sort --target-shards 150 --buffer-size 5000 --num-epochs 2 --rinse

  # Shard count sweep (no quality sort, no rinse):
  python3 prepack_v3.py --output-dir /path/to/shards --val-dir /path/to/val \\
      --no-quality-sort --target-shards 100 --seed 7
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


def estimate_parquets_needed(target_rows, num_epochs, rinse):
    """Estimate how many unique parquets to download."""
    rows_per_epoch = target_rows / max(num_epochs, 1)
    parquets_needed = math.ceil(rows_per_epoch / ROWS_PER_PARQUET_ESTIMATE)
    # Add 20% safety margin for dedup losses; extra 25% when rinsing (stripped docs yield fewer tokens)
    safety = 1.5 if rinse else 1.2
    parquets_needed = math.ceil(parquets_needed * safety)
    return min(parquets_needed, MAX_SHARD)


def rinse_doc_iter(doc_iter, strip_fn, stats):
    """
    Wrap a document iterator, applying strip_fn to each text.
    stats is a dict with keys: docs_seen, docs_discarded, bytes_before, bytes_after.
    Documents where strip_fn returns None or empty string are discarded.
    """
    for text in doc_iter:
        stats["docs_seen"] += 1
        before = len(text.encode("utf-8"))
        stats["bytes_before"] += before
        rinsed = strip_fn(text)
        if not rinsed:
            stats["docs_discarded"] += 1
            continue
        after = len(rinsed.encode("utf-8"))
        stats["bytes_after"] += after
        yield rinsed


def main():
    parser = argparse.ArgumentParser(description="Pre-pack training data with optional shard rinsing")
    parser.add_argument("--output-dir", required=True, help="Directory to write all shard files")
    parser.add_argument("--val-dir", required=True, help="Directory containing existing val shards")

    # Target size: either --target-rows (v2 compat) or --target-shards (convenience)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--target-rows", type=int, default=None,
                       help="Number of packed rows to produce")
    group.add_argument("--target-shards", type=int, default=None,
                       help="Number of output train shards (sets target_rows = N * 1000)")

    parser.add_argument("--buffer-size", type=int, default=5000,
                        help="Dense packing buffer size (default: 5000)")
    parser.add_argument("--num-epochs", type=int, default=2,
                        help="Number of passes over the curated parquet subset (default: 2)")
    parser.add_argument("--dedup", action="store_true", help="Enable near-duplicate filtering")

    # Quality sort is on by default; --no-quality-sort disables it
    parser.add_argument("--quality-sort", action="store_true",
                        help="Rank parquets by quality score, front-load best data")
    parser.add_argument("--no-quality-sort", action="store_true",
                        help="Disable quality ordering (random shuffle); for shard count sweep experiments")

    # Rinsing
    parser.add_argument("--rinse", action="store_true",
                        help="Apply strip_document() to strip web junk before packing (default: off)")

    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--cache-dir", default="/tmp/feeder_cache")
    parser.add_argument("--workers", type=int, default=8, help="Download/scoring workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for parquet ordering")
    args = parser.parse_args()

    # Resolve target_rows
    if args.target_shards is not None:
        target_rows = args.target_shards * ROWS_PER_SHARD
        print(f"[prepack_v3] --target-shards {args.target_shards} -> target_rows={target_rows}", file=sys.stderr)
    elif args.target_rows is not None:
        target_rows = args.target_rows
    else:
        target_rows = 200000  # default matching v2

    # Resolve quality_sort: --quality-sort wins; --no-quality-sort disables
    use_quality_sort = args.quality_sort and not args.no_quality_sort

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
    print(f"[prepack_v3] val shards: {num_val} in {args.val_dir}", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Resume support
    # -------------------------------------------------------------------------
    start_shard = count_existing_train_shards(args.output_dir, num_val)
    rows_already = start_shard * ROWS_PER_SHARD
    if rows_already >= target_rows:
        print(f"[prepack_v3] already have {rows_already} rows >= target {target_rows}, skipping to val copy",
              file=sys.stderr)
    elif start_shard > 0:
        print(f"[prepack_v3] resuming: {start_shard} existing shards ({rows_already} rows)", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Load tokenizer
    # -------------------------------------------------------------------------
    enc, bos_token_id = load_tokenizer(args.tokenizer_dir)
    print(f"[prepack_v3] tokenizer loaded, vocab={enc.n_vocab}, bos={bos_token_id}", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Load strip_document if rinsing
    # -------------------------------------------------------------------------
    strip_fn = None
    strip_stats = {"docs_seen": 0, "docs_discarded": 0, "bytes_before": 0, "bytes_after": 0}
    if args.rinse:
        from strip_doc import strip_document
        strip_fn = strip_document
        print(f"[prepack_v3] rinse enabled: strip_document() will be applied to each doc", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Step (a): Select and order parquet indices
    # Use ALL parquets for maximum diversity; cap docs-per-parquet to hit target.
    # -------------------------------------------------------------------------
    AVG_DOCS_PER_ROW = 3.5
    target_docs = int(target_rows * AVG_DOCS_PER_ROW * args.num_epochs)

    if use_quality_sort:
        from quality_order import rank_parquets
        print(f"[prepack_v3] quality-sorting all {MAX_SHARD} parquets...", file=sys.stderr)
        parquet_indices = rank_parquets(list(range(MAX_SHARD)), args.cache_dir, workers=args.workers)
        print(f"[prepack_v3] using all {len(parquet_indices)} parquets (quality-sorted)", file=sys.stderr)
    else:
        random.seed(args.seed)
        parquet_indices = list(range(MAX_SHARD))
        random.shuffle(parquet_indices)
        sort_label = "no-quality-sort" if args.no_quality_sort else "shuffled"
        print(f"[prepack_v3] using all {len(parquet_indices)} parquets ({sort_label}, seed={args.seed})", file=sys.stderr)

    # Docs per parquet: spread budget evenly across all parquets
    max_docs_per_parquet = max(1, math.ceil(target_docs / len(parquet_indices)))

    rinse_label = "rinse -> " if args.rinse else ""
    print(f"[prepack_v3] pipeline: {len(parquet_indices)} parquets x {args.num_epochs} epochs"
          f" x {max_docs_per_parquet} docs/parquet"
          f" -> {rinse_label}{'dedup -> ' if args.dedup else ''}dense_pack(buf={args.buffer_size})"
          f" -> {target_rows} rows target", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Step (b)-(f): Build the document -> packed row pipeline
    # -------------------------------------------------------------------------
    if rows_already < target_rows:

        base_doc_iter = streaming_document_batches(
            parquet_indices, args.cache_dir,
            prefetch_workers=args.workers, cleanup=(args.num_epochs <= 1),
            max_docs_per_parquet=max_docs_per_parquet,
        )

        # Multi-epoch: collect docs from parquets once, reshuffle per epoch
        if args.num_epochs > 1:
            doc_iter = multi_epoch_doc_iter(base_doc_iter, len(parquet_indices), args.num_epochs)
        else:
            doc_iter = base_doc_iter

        # Rinse: strip junk from each document before packing
        if args.rinse:
            doc_iter = rinse_doc_iter(doc_iter, strip_fn, strip_stats)

        # Optional dedup (after rinsing so dedup sees clean text)
        if args.dedup:
            from dedup import exact_dedup_documents
            doc_iter = exact_dedup_documents(doc_iter)
            print(f"[prepack_v3] dedup enabled (exact SHA-256)", file=sys.stderr)

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
                        print(f"[prepack_v3] skipped to shard {start_shard} ({elapsed:.0f}s)", file=sys.stderr)
                continue

            rows_written += 1
            current_shard_rows.append(row)

            if len(current_shard_rows) == ROWS_PER_SHARD:
                path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.bin")
                write_shard(path, current_shard_rows)
                elapsed = time.time() - t_start
                rate = rows_written / elapsed if elapsed > 0 else 0
                print(f"[prepack_v3] shard {shard_idx:5d} | {rows_written:>7d}/{target_rows} rows"
                      f" | {rate:.0f} rows/s | {elapsed:.0f}s", file=sys.stderr)
                current_shard_rows = []
                shard_idx += 1

            # Stop once we've hit the target
            if rows_written >= target_rows:
                print(f"[prepack_v3] reached target {target_rows} rows, stopping", file=sys.stderr)
                break

        # Write final partial shard if any
        if current_shard_rows:
            path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.bin")
            write_shard(path, current_shard_rows)
            elapsed = time.time() - t_start
            print(f"[prepack_v3] shard {shard_idx:5d} (partial, {len(current_shard_rows)} rows) | {elapsed:.0f}s",
                  file=sys.stderr)
            shard_idx += 1

        num_train_shards = shard_idx
        elapsed = time.time() - t_start
        print(f"\n[prepack_v3] train done: {num_train_shards} shards, {rows_written} rows, {elapsed:.0f}s",
              file=sys.stderr)

        # -----------------------------------------------------------------
        # Rinse summary
        # -----------------------------------------------------------------
        if args.rinse and strip_stats["docs_seen"] > 0:
            docs_discarded = strip_stats["docs_discarded"]
            bytes_before = strip_stats["bytes_before"]
            bytes_after = strip_stats["bytes_after"]
            bytes_stripped = bytes_before - bytes_after
            # Effective density gain: how much denser the kept+stripped text is
            # compared to raw. If we stripped 20% of bytes, each output token
            # represents 1/(1-0.2) = 1.25x more "intent" from the raw corpus.
            if bytes_before > 0:
                gain = bytes_stripped / bytes_before
            else:
                gain = 0.0
            print(
                f"[prepack_v3] rinsed: {docs_discarded} docs dropped, "
                f"{bytes_stripped / 1e6:.1f}MB stripped, "
                f"effective density gain: {gain:.1%}",
                file=sys.stderr,
            )
    else:
        # Already have enough shards
        num_train_shards = start_shard
        rows_written = rows_already

    # -------------------------------------------------------------------------
    # Step (h): Copy val shards, renumbered after train shards
    # -------------------------------------------------------------------------
    print(f"[prepack_v3] copying {num_val} val shards from {args.val_dir}...", file=sys.stderr)
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
    print(f"[prepack_v3] copied {num_val} val shards ({total_val_rows} rows)", file=sys.stderr)

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
            "target_rows": target_rows,
            "buffer_size": args.buffer_size,
            "num_epochs": args.num_epochs,
            "dedup": args.dedup,
            "quality_sort": use_quality_sort,
            "rinse": args.rinse,
            "seed": args.seed,
            "num_parquets": len(parquet_indices),
            # Strip stats included when rinsing was active
            **({"strip_stats": strip_stats} if args.rinse else {}),
        },
    }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[prepack_v3] manifest: {json.dumps(manifest, indent=2)}", file=sys.stderr)
    print(f"\n[prepack_v3] ALL DONE. Run with:", file=sys.stderr)
    print(f"  NUM_TRAIN_SHARDS={num_train_shards} cargo run --release --bin autoresearch-engine -- train --data-dir {args.output_dir}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
