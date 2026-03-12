"""
Score and rank parquets by text quality for front-loading the best data.

Scores each parquet on three axes:
  - Average document length in characters (longer = more coherent, weight 0.4)
  - Vocabulary diversity: unique words / total words (weight 0.3)
  - Sentence length quality: penalizes too-short or too-long sentences (weight 0.3)

Usage:
  python3 quality_order.py --cache-dir /tmp/feeder_cache --workers 8 --sample 50
  python3 quality_order.py --cache-dir /tmp/feeder_cache --top 200  # print top 200 indices
"""

import os
import sys
import json
import math
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import download_parquet from feeder (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feeder import download_parquet, MAX_SHARD

import pyarrow.parquet as pq

SCORE_CACHE_FILE = "parquet_quality_scores.json"

# Sentence splitter: split on .!? followed by whitespace or end of string
_SENT_RE = re.compile(r'[^.!?]+[.!?]*')


def score_parquet(texts: list[str]) -> float:
    """Score a batch of documents. Returns a float in [0, 1]."""
    if not texts:
        return 0.0

    # --- Average document length (chars) ---
    doc_lengths = [len(t) for t in texts]
    avg_len = sum(doc_lengths) / len(doc_lengths)
    # Sigmoid-like normalization: 2000 chars is "good", diminishing returns after
    len_score = 1.0 - math.exp(-avg_len / 2000.0)

    # --- Vocabulary diversity: unique / total words ---
    total_words = 0
    unique_words = set()
    for t in texts:
        words = t.lower().split()
        total_words += len(words)
        unique_words.update(words)
    diversity = len(unique_words) / max(total_words, 1)
    # Typical diversity 0.1-0.5; normalize to [0,1] range
    div_score = min(diversity / 0.5, 1.0)

    # --- Sentence length quality ---
    # Optimal: 15-25 words per sentence
    sent_lengths = []
    for t in texts:
        sentences = _SENT_RE.findall(t)
        for s in sentences:
            wc = len(s.split())
            if wc > 0:
                sent_lengths.append(wc)
    if sent_lengths:
        avg_sent = sum(sent_lengths) / len(sent_lengths)
        # Gaussian-like penalty centered at 20 words, sigma=10
        sent_score = math.exp(-((avg_sent - 20.0) ** 2) / (2 * 10.0 ** 2))
    else:
        sent_score = 0.0

    return 0.4 * len_score + 0.3 * div_score + 0.3 * sent_score


def _score_one(index: int, cache_dir: str, sample_size: int) -> tuple[int, float]:
    """Download one parquet, sample documents, score, then delete to save disk."""
    try:
        filepath = download_parquet(index, cache_dir)
        pf = pq.ParquetFile(filepath)
        texts = []
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            texts.extend(rg.column("text").to_pylist())
            if len(texts) >= sample_size:
                break
        texts = texts[:sample_size]
        score = score_parquet(texts)
        # Clean up to save disk space during scoring
        try:
            os.remove(filepath)
        except OSError:
            pass
        return (index, score)
    except Exception as e:
        print(f"[quality] ERROR scoring parquet {index}: {e}", file=sys.stderr)
        return (index, 0.0)


def rank_parquets(indices: list[int], cache_dir: str, workers: int = 8,
                  sample_size: int = 50) -> list[int]:
    """Score and rank parquets, returning indices sorted best-first.

    Caches scores to a JSON file so re-ranking is instant on rerun.
    """
    os.makedirs(cache_dir, exist_ok=True)
    score_path = os.path.join(cache_dir, SCORE_CACHE_FILE)

    # Load cached scores
    cached: dict[str, float] = {}
    if os.path.exists(score_path):
        with open(score_path, "r") as f:
            cached = json.load(f)
        print(f"[quality] loaded {len(cached)} cached scores from {score_path}", file=sys.stderr)

    # Figure out which indices still need scoring
    to_score = [i for i in indices if str(i) not in cached]
    already = len(indices) - len(to_score)
    if already > 0:
        print(f"[quality] {already}/{len(indices)} already cached, {len(to_score)} to score", file=sys.stderr)

    if to_score:
        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_score_one, idx, cache_dir, sample_size): idx for idx in to_score}
            for fut in as_completed(futures):
                idx, score = fut.result()
                cached[str(idx)] = score
                done += 1
                if done % 50 == 0 or done == len(to_score):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    print(f"[quality] scored {done}/{len(to_score)} ({rate:.0f}/s)", file=sys.stderr)

        # Save cache
        with open(score_path, "w") as f:
            json.dump(cached, f)
        print(f"[quality] saved scores to {score_path}", file=sys.stderr)

    # Sort by score descending
    ranked = sorted(indices, key=lambda i: cached.get(str(i), 0.0), reverse=True)
    return ranked


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Score and rank parquets by quality")
    parser.add_argument("--cache-dir", default="/tmp/feeder_cache")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sample", type=int, default=50, help="Documents to sample per parquet")
    parser.add_argument("--top", type=int, default=0, help="Print only top N indices (0=all)")
    parser.add_argument("--max-index", type=int, default=MAX_SHARD,
                        help=f"Max parquet index to score (default: {MAX_SHARD})")
    args = parser.parse_args()

    indices = list(range(args.max_index))
    ranked = rank_parquets(indices, args.cache_dir, workers=args.workers, sample_size=args.sample)

    n = args.top if args.top > 0 else len(ranked)
    for i, idx in enumerate(ranked[:n]):
        print(idx)


if __name__ == "__main__":
    main()
