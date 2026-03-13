"""
Stream packed training rows to stdout in binary format for the Rust engine.

Reads text from HuggingFace parquets, local parquets, or JSONL files,
tokenizes with BOS, best-fit packs into 2049-token rows, and writes raw
binary rows to stdout. The Rust engine reads these via --stream-input.

Binary protocol (stdout):
  Each row: 2049 little-endian u16 tokens (4098 bytes)
  No headers, no framing — just rows back to back.
  EOF signals end of data (training should be step-limited, not data-limited).

Data sources:
  --stream              Stream parquets from HuggingFace (climbmix-400b-shuffle)
  --input DIR           Read local parquet files from directory
  --jsonl FILE [FILE..] Read JSONL files (one JSON object per line, extracts text
                        from "text" field or concatenates "messages" array content)

Usage:
  python3 feeder.py --stream | ./autoresearch-brain train --stream-input
  python3 feeder.py --jsonl data/*.jsonl | ./autoresearch-brain train --stream-input
"""

import os
import sys
import struct
import time
import argparse
import pickle
import json
import threading
import queue

import requests
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Constants — must match Python reference exactly
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048
ROW_CAPACITY = MAX_SEQ_LEN + 1  # 2049 tokens per row
VOCAB_SIZE = 8192
MAX_SHARD = 6542
VAL_SHARD = MAX_SHARD
BUFFER_SIZE = 1000  # same as Python's make_dataloader

HF_BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"

# Pre-compute the struct format for one row
ROW_STRUCT = struct.Struct(f"<{ROW_CAPACITY}H")

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_dir):
    tokenizer_pkl = os.path.join(tokenizer_dir, "tokenizer.pkl")
    with open(tokenizer_pkl, "rb") as f:
        enc = pickle.load(f)
    bos_token_id = enc.encode_single_token("<|reserved_0|>")
    return enc, bos_token_id


# ---------------------------------------------------------------------------
# Parallel parquet downloader with prefetch
# ---------------------------------------------------------------------------

def download_parquet(index, cache_dir):
    """Download one parquet shard from HuggingFace, return path."""
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


def prefetch_parquets(indices, cache_dir, prefetch_queue, num_workers=4):
    """Download parquets in parallel, put paths on queue in order."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Download in chunks to maintain approximate order while parallelizing
    chunk_size = num_workers * 2
    for start in range(0, len(indices), chunk_size):
        chunk = indices[start:start + chunk_size]
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(download_parquet, idx, cache_dir): idx for idx in chunk}
            results = {}
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
        # Put in order
        for idx in chunk:
            prefetch_queue.put(results[idx])
    prefetch_queue.put(None)  # sentinel


def streaming_document_batches(shard_indices, cache_dir, tokenizer_batch_size=128,
                                prefetch_workers=4, cleanup=True, max_docs_per_parquet=None):
    """Stream parquets with prefetch. Yields text batches.

    max_docs_per_parquet: if set, take at most this many docs from each parquet.
    Use this to spread a fixed token budget across all parquets for maximum diversity.
    """
    os.makedirs(cache_dir, exist_ok=True)

    prefetch_q = queue.Queue(maxsize=prefetch_workers * 2)
    dl_thread = threading.Thread(
        target=prefetch_parquets,
        args=(shard_indices, cache_dir, prefetch_q, prefetch_workers),
        daemon=True,
    )
    dl_thread.start()

    while True:
        filepath = prefetch_q.get()
        if filepath is None:
            break
        pf = pq.ParquetFile(filepath)
        docs_yielded = 0
        done = False
        for rg_idx in range(pf.num_row_groups):
            if done:
                break
            rg = pf.read_row_group(rg_idx)
            batch = rg.column("text").to_pylist()
            for i in range(0, len(batch), tokenizer_batch_size):
                chunk = batch[i:i + tokenizer_batch_size]
                if max_docs_per_parquet is not None:
                    remaining = max_docs_per_parquet - docs_yielded
                    if remaining <= 0:
                        done = True
                        break
                    chunk = chunk[:remaining]
                docs_yielded += len(chunk)
                yield chunk
                if max_docs_per_parquet is not None and docs_yielded >= max_docs_per_parquet:
                    done = True
                    break
        if cleanup:
            try:
                os.remove(filepath)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# JSONL input support (e.g. solidSF trajectories)
# ---------------------------------------------------------------------------

def _text_from_message_obj(msg):
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                txt = block.get("text", "")
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt.strip())
        return "\n".join(parts).strip()
    return ""


def _text_from_jsonl_obj(obj):
    # Plain text JSONL row: {"text": "..."}
    txt = obj.get("text")
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    # OpenAI/chat-like rows: {"messages":[{"role":"user","content":"..."}, ...]}
    messages = obj.get("messages")
    if isinstance(messages, list):
        parts = []
        for msg in messages:
            msg_text = _text_from_message_obj(msg)
            if msg_text:
                parts.append(msg_text)
        if parts:
            return "\n".join(parts)

    # solidSF logger rows: {"turns":[{"role":"user","content":"..."}, ...]}
    turns = obj.get("turns")
    if isinstance(turns, list):
        parts = []
        for turn in turns:
            turn_text = _text_from_message_obj(turn)
            if turn_text:
                parts.append(turn_text)
        if parts:
            return "\n".join(parts)

    return ""


def jsonl_document_batches(jsonl_paths, tokenizer_batch_size=128):
    batch = []
    for path in jsonl_paths:
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[feeder] WARN: {path}:{lineno} invalid JSON ({e})", file=sys.stderr)
                    continue
                if not isinstance(obj, dict):
                    continue
                text = _text_from_jsonl_obj(obj)
                if not text:
                    continue
                batch.append(text)
                if len(batch) >= tokenizer_batch_size:
                    yield batch
                    batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Best-fit packing — exact replica of Python's make_dataloader logic
# ---------------------------------------------------------------------------

def pack_rows(doc_iter, enc, bos_token_id):
    """Best-fit pack tokenized documents into 2049-token rows. Infinite if data allows."""
    doc_buffer = []

    def refill_buffer():
        try:
            text_batch = next(doc_iter)
        except StopIteration:
            return False
        token_lists = enc.encode_ordinary_batch(text_batch, num_threads=16)
        for toks in token_lists:
            toks.insert(0, bos_token_id)
        doc_buffer.extend(token_lists)
        return True

    while True:
        row = []
        pos = 0
        while pos < ROW_CAPACITY:
            while len(doc_buffer) < BUFFER_SIZE:
                if not refill_buffer():
                    break
            if len(doc_buffer) == 0:
                if pos > 0:
                    row.extend([0] * (ROW_CAPACITY - pos))
                    pos = ROW_CAPACITY
                    break
                else:
                    return

            remaining = ROW_CAPACITY - pos

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
                shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                doc = doc_buffer.pop(shortest_idx)
                row.extend(doc[:remaining])
                pos += remaining

        assert len(row) == ROW_CAPACITY, f"row length {len(row)} != {ROW_CAPACITY}"
        yield row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stream packed training rows to stdout")
    parser.add_argument("--stream", action="store_true", help="Stream from HuggingFace")
    parser.add_argument("--input", default=None, help="Local parquet directory")
    parser.add_argument("--jsonl", nargs="+", default=None, help="JSONL files to read")
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--cache-dir", default="/tmp/feeder_cache")
    parser.add_argument("--prefetch", type=int, default=32, help="Prefetch workers (default: 32)")
    parser.add_argument("--no-prefetch", action="store_true", help="Disable prefetch queue; write rows synchronously (debug)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for parquet shuffle order (enables reproducibility)")
    args = parser.parse_args()

    if args.tokenizer_dir is None:
        args.tokenizer_dir = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "tokenizer")

    # Load tokenizer
    enc, bos_token_id = load_tokenizer(args.tokenizer_dir)
    print(f"[feeder] tokenizer loaded, vocab={enc.n_vocab}, bos={bos_token_id}", file=sys.stderr)

    # Set up document stream
    if args.stream:
        import random
        train_indices = list(range(0, MAX_SHARD))  # 0..6541
        random.seed(args.seed)
        random.shuffle(train_indices)
        doc_iter = streaming_document_batches(
            train_indices, args.cache_dir,
            prefetch_workers=args.prefetch, cleanup=True,
        )
    elif args.input:
        from repack_shards import list_parquet_files, document_batches
        parquets = list_parquet_files(args.input)
        val_path = os.path.join(args.input, f"shard_{VAL_SHARD:05d}.parquet")
        parquets = [p for p in parquets if p != val_path]
        doc_iter = document_batches(parquets)
    elif args.jsonl:
        doc_iter = jsonl_document_batches(args.jsonl)
    else:
        print("ERROR: --stream, --input, or --jsonl required", file=sys.stderr)
        sys.exit(1)

    # Pack and write to stdout
    stdout = sys.stdout.buffer

    rows_produced = 0
    t_pack_total = 0.0
    t_write_total = 0.0
    t_start = time.time()

    def _print_stats():
        elapsed = time.time() - t_start
        rate = rows_produced / elapsed if elapsed > 0 else 0.0
        other = elapsed - t_pack_total - t_write_total
        print(f"[feeder] rows={rows_produced} elapsed={elapsed:.1f}s rate={rate:.0f} rows/s", file=sys.stderr)
        print(f"[feeder] pack={t_pack_total:.1f}s write={t_write_total:.1f}s other={other:.1f}s", file=sys.stderr)

    if args.no_prefetch:
        # Synchronous path — no prefetch queue, write each row as it is packed
        try:
            for row in pack_rows(doc_iter, enc, bos_token_id):
                t0 = time.time()
                data = ROW_STRUCT.pack(*row)
                t_pack_total += time.time() - t0

                t0 = time.time()
                stdout.write(data)
                stdout.flush()
                t_write_total += time.time() - t0

                rows_produced += 1
                if rows_produced % 10000 == 0:
                    elapsed = time.time() - t_start
                    rate = rows_produced / elapsed
                    print(f"[feeder] {rows_produced} rows, {rate:.0f} rows/s, {elapsed:.0f}s", file=sys.stderr)
        except BrokenPipeError:
            pass
        finally:
            _print_stats()
    else:
        # Prefetch path — pack rows into a queue on a background thread, main thread writes
        row_queue = queue.Queue(maxsize=args.prefetch)

        def _pack_worker():
            try:
                for row in pack_rows(doc_iter, enc, bos_token_id):
                    t0 = time.time()
                    data = ROW_STRUCT.pack(*row)
                    elapsed_pack = time.time() - t0
                    row_queue.put((data, elapsed_pack))
            except Exception as e:
                print(f"[feeder] pack worker error: {e}", file=sys.stderr)
            finally:
                row_queue.put(None)  # sentinel

        worker = threading.Thread(target=_pack_worker, daemon=True)
        worker.start()

        try:
            while True:
                item = row_queue.get()
                if item is None:
                    break
                data, elapsed_pack = item
                t_pack_total += elapsed_pack

                t0 = time.time()
                stdout.write(data)
                stdout.flush()
                t_write_total += time.time() - t0

                rows_produced += 1
                if rows_produced % 10000 == 0:
                    elapsed = time.time() - t_start
                    rate = rows_produced / elapsed
                    print(f"[feeder] {rows_produced} rows, {rate:.0f} rows/s, {elapsed:.0f}s", file=sys.stderr)
        except BrokenPipeError:
            pass
        finally:
            _print_stats()


if __name__ == "__main__":
    main()
