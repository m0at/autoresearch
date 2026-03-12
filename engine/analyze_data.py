"""
Analyze climbmix training data for dedup/boilerplate stripping opportunities.

Samples parquets, detects:
1. Near-duplicate documents (MinHash signatures)
2. Repeated boilerplate paragraphs/sentences across docs
3. Per-doc information density (compression ratio)
4. Token waste from low-info content

Usage:
  python3 analyze_data.py --sample 100          # analyze 100 random parquets
  python3 analyze_data.py --sample 50 --verbose  # show example boilerplate
"""

import os
import sys
import hashlib
import random
import time
import struct
import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Reuse feeder's download/tokenizer infrastructure
from feeder import download_parquet, load_tokenizer, MAX_SHARD

try:
    import pyarrow.parquet as pq
except ImportError:
    print("pip install pyarrow", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# MinHash for near-duplicate detection
# ---------------------------------------------------------------------------

NUM_HASHES = 128
MINHASH_SEED = 0xDEADBEEF
MAX_HASH = (1 << 64) - 1

def _hash_shingle(shingle, seed):
    """Hash a shingle (word n-gram) to uint64."""
    h = hashlib.md5(f"{seed}:{shingle}".encode(), usedforsecurity=False).digest()
    return struct.unpack("<Q", h[:8])[0]

def minhash_signature(text, num_hashes=NUM_HASHES, shingle_size=5):
    """Compute MinHash signature from word-level shingles."""
    words = text.lower().split()
    if len(words) < shingle_size:
        # Too short for meaningful shingling
        return None
    shingles = set()
    for i in range(len(words) - shingle_size + 1):
        shingles.add(" ".join(words[i:i + shingle_size]))
    if not shingles:
        return None
    sig = []
    for seed_offset in range(num_hashes):
        min_h = MAX_HASH
        for s in shingles:
            h = _hash_shingle(s, MINHASH_SEED + seed_offset)
            if h < min_h:
                min_h = h
        sig.append(min_h)
    return tuple(sig)

def jaccard_from_minhash(sig1, sig2):
    """Estimate Jaccard similarity from two MinHash signatures."""
    if sig1 is None or sig2 is None:
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


# ---------------------------------------------------------------------------
# LSH (Locality-Sensitive Hashing) for fast near-dup detection
# ---------------------------------------------------------------------------

def lsh_buckets(sig, bands=16):
    """Split MinHash signature into bands for LSH bucketing."""
    rows_per_band = len(sig) // bands
    buckets = []
    for b in range(bands):
        band_slice = sig[b * rows_per_band:(b + 1) * rows_per_band]
        bucket = hashlib.md5(str(band_slice).encode(), usedforsecurity=False).hexdigest()[:16]
        buckets.append((b, bucket))
    return buckets


# ---------------------------------------------------------------------------
# Boilerplate detection
# ---------------------------------------------------------------------------

def extract_paragraphs(text, min_len=50):
    """Split text into paragraphs, filter short ones."""
    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= min_len]
    return paras

def extract_sentences(text, min_len=30):
    """Simple sentence splitting for boilerplate detection."""
    sents = []
    for part in text.replace("! ", ". ").replace("? ", ". ").split(". "):
        s = part.strip()
        if len(s) >= min_len:
            sents.append(s)
    return sents


# ---------------------------------------------------------------------------
# Information density via compression ratio
# ---------------------------------------------------------------------------

import zlib

def compression_ratio(text):
    """Lower ratio = more redundant. Ratio = compressed/original."""
    raw = text.encode("utf-8")
    if len(raw) < 100:
        return 1.0
    compressed = zlib.compress(raw, level=6)
    return len(compressed) / len(raw)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def load_docs_from_parquet(filepath, max_docs=500, seed=42):
    """Load text documents from a parquet file, sampling if large."""
    table = pq.read_table(filepath, columns=["text"])
    docs = table.column("text").to_pylist()
    if len(docs) > max_docs:
        rng = random.Random(seed)
        docs = rng.sample(docs, max_docs)
    return docs


def analyze_parquets(indices, cache_dir, workers=8, verbose=False, docs_per_parquet=500):
    """Full analysis pipeline."""
    print(f"\n[analyze] sampling {len(indices)} parquets...", file=sys.stderr)

    # Download parquets in parallel
    t0 = time.time()
    paths = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(download_parquet, idx, cache_dir): idx for idx in indices}
        for f in as_completed(futs):
            path = f.result()
            if path:
                paths.append((futs[f], path))
    paths.sort()
    print(f"[analyze] downloaded {len(paths)} parquets in {time.time()-t0:.1f}s", file=sys.stderr)

    # Collect all documents
    all_docs = []
    docs_per_parquet_counts = []
    for idx, path in paths:
        docs = load_docs_from_parquet(path, max_docs=docs_per_parquet)
        docs_per_parquet_counts.append(len(docs))
        for doc in docs:
            all_docs.append((idx, doc))

    total_docs = len(all_docs)
    total_chars = sum(len(d) for _, d in all_docs)
    print(f"[analyze] {total_docs} documents, {total_chars/1e6:.1f}M chars", file=sys.stderr)
    print(f"[analyze] avg {total_chars/total_docs:.0f} chars/doc, "
          f"avg {sum(docs_per_parquet_counts)/len(docs_per_parquet_counts):.0f} docs/parquet (after sampling)", file=sys.stderr)

    # -----------------------------------------------------------------------
    # 1. Near-duplicate detection via MinHash + LSH
    # -----------------------------------------------------------------------
    MINHASH_CAP = 50000
    minhash_docs = all_docs
    if total_docs > MINHASH_CAP:
        print(f"[analyze] WARNING: {total_docs} docs exceeds MinHash cap ({MINHASH_CAP}), subsampling...", file=sys.stderr)
        rng = random.Random(42)
        minhash_docs = rng.sample(all_docs, MINHASH_CAP)

    print(f"\n[analyze] computing MinHash signatures for {len(minhash_docs)} docs...", file=sys.stderr)
    t1 = time.time()
    sigs = []
    for i, (idx, doc) in enumerate(minhash_docs):
        sig = minhash_signature(doc)
        sigs.append(sig)
        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{len(minhash_docs)} signatures", file=sys.stderr)
    print(f"  done in {time.time()-t1:.1f}s", file=sys.stderr)

    # LSH bucketing
    print(f"[analyze] LSH bucketing...", file=sys.stderr)
    t2 = time.time()
    lsh_table = defaultdict(list)  # (band, bucket_hash) -> [doc_index]
    for i, sig in enumerate(sigs):
        if sig is None:
            continue
        for band, bucket in lsh_buckets(sig, bands=16):
            lsh_table[(band, bucket)].append(i)

    # Find candidate pairs and verify
    near_dup_pairs = 0
    near_dup_docs = set()
    checked = set()
    for key, doc_ids in lsh_table.items():
        if len(doc_ids) < 2 or len(doc_ids) > 100:  # skip huge buckets (common boilerplate)
            continue
        for i in range(len(doc_ids)):
            for j in range(i + 1, min(len(doc_ids), i + 10)):  # cap comparisons
                pair = (doc_ids[i], doc_ids[j])
                if pair in checked:
                    continue
                checked.add(pair)
                sim = jaccard_from_minhash(sigs[doc_ids[i]], sigs[doc_ids[j]])
                if sim >= 0.8:
                    near_dup_pairs += 1
                    near_dup_docs.add(doc_ids[j])  # mark the later one as dup

    minhash_total = len(minhash_docs)
    print(f"  {near_dup_pairs} near-duplicate pairs found", file=sys.stderr)
    print(f"  {len(near_dup_docs)} docs removable ({100*len(near_dup_docs)/minhash_total:.1f}% of minhash sample)", file=sys.stderr)
    dup_chars = sum(len(minhash_docs[i][1]) for i in near_dup_docs)
    print(f"  {dup_chars/1e6:.1f}M chars in near-duplicates", file=sys.stderr)

    if verbose and near_dup_pairs > 0:
        # Show a few examples
        shown = 0
        for key, doc_ids in lsh_table.items():
            if len(doc_ids) < 2:
                continue
            for i in range(len(doc_ids)):
                for j in range(i + 1, len(doc_ids)):
                    if sigs[doc_ids[i]] and sigs[doc_ids[j]]:
                        sim = jaccard_from_minhash(sigs[doc_ids[i]], sigs[doc_ids[j]])
                        if sim >= 0.8:
                            print(f"\n  === NEAR-DUP (sim={sim:.2f}) ===", file=sys.stderr)
                            print(f"  DOC A (parquet {minhash_docs[doc_ids[i]][0]}): {minhash_docs[doc_ids[i]][1][:200]}...", file=sys.stderr)
                            print(f"  DOC B (parquet {minhash_docs[doc_ids[j]][0]}): {minhash_docs[doc_ids[j]][1][:200]}...", file=sys.stderr)
                            shown += 1
                            if shown >= 3:
                                break
                if shown >= 3:
                    break
            if shown >= 3:
                break

    # -----------------------------------------------------------------------
    # 2. Boilerplate paragraph detection
    # -----------------------------------------------------------------------
    print(f"\n[analyze] scanning for boilerplate paragraphs...", file=sys.stderr)
    para_counts = Counter()
    para_sources = defaultdict(set)  # para_hash -> set of parquet indices
    total_paras = 0
    for i, (idx, doc) in enumerate(all_docs):
        for para in extract_paragraphs(doc, min_len=60):
            # Normalize whitespace for matching
            normalized = " ".join(para.split()).lower()
            h = hashlib.md5(normalized.encode(), usedforsecurity=False).hexdigest()
            para_counts[h] += 1
            para_sources[h].add(idx)
            total_paras += 1

    # Boilerplate = paragraphs appearing in 3+ different parquets
    boilerplate_paras = {h: c for h, c in para_counts.items() if c >= 5 and len(para_sources[h]) >= 3}
    boilerplate_occurrences = sum(boilerplate_paras.values())

    # Reconstruct actual boilerplate text for reporting
    boilerplate_texts = {}
    if boilerplate_paras:
        for i, (idx, doc) in enumerate(all_docs):
            for para in extract_paragraphs(doc, min_len=60):
                normalized = " ".join(para.split()).lower()
                h = hashlib.md5(normalized.encode(), usedforsecurity=False).hexdigest()
                if h in boilerplate_paras and h not in boilerplate_texts:
                    boilerplate_texts[h] = para[:200]
            if len(boilerplate_texts) >= len(boilerplate_paras):
                break

    print(f"  {total_paras} total paragraphs analyzed", file=sys.stderr)
    print(f"  {len(boilerplate_paras)} unique boilerplate paragraphs (appearing 5+ times across 3+ parquets)", file=sys.stderr)
    print(f"  {boilerplate_occurrences} total boilerplate occurrences ({100*boilerplate_occurrences/max(total_paras,1):.1f}% of paragraphs)", file=sys.stderr)

    if verbose:
        top_bp = sorted(boilerplate_paras.items(), key=lambda x: -x[1])[:10]
        for h, count in top_bp:
            text = boilerplate_texts.get(h, "(text not recovered)")
            print(f"\n  [{count}x across {len(para_sources[h])} parquets] {text}...", file=sys.stderr)

    # -----------------------------------------------------------------------
    # 3. Information density (compression ratio distribution)
    # -----------------------------------------------------------------------
    print(f"\n[analyze] computing compression ratios...", file=sys.stderr)
    ratios = []
    for _, doc in all_docs:
        r = compression_ratio(doc)
        ratios.append(r)

    ratios.sort()
    p10 = ratios[int(0.1 * len(ratios))]
    p25 = ratios[int(0.25 * len(ratios))]
    p50 = ratios[int(0.5 * len(ratios))]
    p75 = ratios[int(0.75 * len(ratios))]
    p90 = ratios[int(0.9 * len(ratios))]

    # Docs with ratio < 0.15 are highly redundant (compress to <15% of original)
    low_info = sum(1 for r in ratios if r < 0.15)
    low_info_chars = sum(len(all_docs[i][1]) for i, r in enumerate(ratios) if r < 0.15)

    print(f"  compression ratio distribution (lower = more redundant):", file=sys.stderr)
    print(f"    p10={p10:.3f}  p25={p25:.3f}  p50={p50:.3f}  p75={p75:.3f}  p90={p90:.3f}", file=sys.stderr)
    print(f"  {low_info} docs ({100*low_info/total_docs:.1f}%) with ratio < 0.15 (highly redundant)", file=sys.stderr)
    print(f"  {low_info_chars/1e6:.1f}M chars in highly-redundant docs", file=sys.stderr)

    if verbose and low_info > 0:
        # Show worst offenders
        worst = sorted(enumerate(ratios), key=lambda x: x[1])[:3]
        for i, r in worst:
            idx, doc = all_docs[i]
            print(f"\n  === LOW-INFO (ratio={r:.3f}, parquet {idx}) ===", file=sys.stderr)
            print(f"  {doc[:300]}...", file=sys.stderr)

    # -----------------------------------------------------------------------
    # 4. Document length distribution
    # -----------------------------------------------------------------------
    print(f"\n[analyze] document length distribution:", file=sys.stderr)
    lengths = [len(d) for _, d in all_docs]
    lengths.sort()
    print(f"  p10={lengths[int(0.1*len(lengths))]}  p25={lengths[int(0.25*len(lengths))]}  "
          f"p50={lengths[int(0.5*len(lengths))]}  p75={lengths[int(0.75*len(lengths))]}  "
          f"p90={lengths[int(0.9*len(lengths))]}", file=sys.stderr)
    tiny = sum(1 for l in lengths if l < 100)
    huge = sum(1 for l in lengths if l > 50000)
    print(f"  {tiny} docs < 100 chars ({100*tiny/total_docs:.1f}%), "
          f"{huge} docs > 50K chars ({100*huge/total_docs:.1f}%)", file=sys.stderr)

    # -----------------------------------------------------------------------
    # Summary: estimated savings
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"SUMMARY: data reduction opportunities", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  Total: {total_docs} docs, {total_chars/1e6:.1f}M chars", file=sys.stderr)
    print(f"  Near-duplicates removable: {len(near_dup_docs)} docs ({dup_chars/1e6:.1f}M chars, {100*dup_chars/total_chars:.1f}%)", file=sys.stderr)
    print(f"  Boilerplate strippable: ~{boilerplate_occurrences} paragraph instances", file=sys.stderr)
    print(f"  Low-info docs (ratio<0.15): {low_info} docs ({low_info_chars/1e6:.1f}M chars, {100*low_info_chars/total_chars:.1f}%)", file=sys.stderr)
    est_savings = dup_chars + low_info_chars
    print(f"  Estimated total reduction: {est_savings/1e6:.1f}M chars ({100*est_savings/total_chars:.1f}%)", file=sys.stderr)
    print(f"  → {total_chars/1e6:.1f}M → ~{(total_chars-est_savings)/1e6:.1f}M chars after cleanup", file=sys.stderr)

    # Extrapolate to full dataset
    scale = MAX_SHARD / len(indices)
    print(f"\n  Extrapolated to full {MAX_SHARD} parquets ({scale:.0f}x):", file=sys.stderr)
    print(f"    ~{len(near_dup_docs)*scale:.0f} near-dup docs removable", file=sys.stderr)
    print(f"    ~{est_savings*scale/1e6:.0f}M chars strippable", file=sys.stderr)

    return {
        "total_docs": total_docs,
        "total_chars": total_chars,
        "near_dup_docs": len(near_dup_docs),
        "near_dup_chars": dup_chars,
        "boilerplate_unique": len(boilerplate_paras),
        "boilerplate_occurrences": boilerplate_occurrences,
        "low_info_docs": low_info,
        "low_info_chars": low_info_chars,
        "compression_p50": p50,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze climbmix data for dedup/stripping opportunities")
    parser.add_argument("--sample", type=int, default=50, help="Number of parquets to sample (default: 50)")
    parser.add_argument("--cache-dir", default="/tmp/feeder_cache", help="Parquet cache directory")
    parser.add_argument("--workers", type=int, default=8, help="Download workers")
    parser.add_argument("--verbose", action="store_true", help="Show example boilerplate and near-dupes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--docs-per-parquet", type=int, default=500, help="Max docs to sample per parquet (default: 500)")
    args = parser.parse_args()

    random.seed(args.seed)
    indices = random.sample(range(MAX_SHARD), min(args.sample, MAX_SHARD))
    indices.sort()

    results = analyze_parquets(indices, args.cache_dir, workers=args.workers, verbose=args.verbose, docs_per_parquet=args.docs_per_parquet)

    # Write results as JSON for programmatic use
    json.dump(results, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
