"""
Near-duplicate and exact-duplicate document filtering for the training pipeline.

Wraps a document batch iterator (yields list[str]) and filters out duplicates
before tokenization/packing, so every packed row contains unique content.

Usage in feeder.py:
    from dedup import dedup_documents
    doc_iter = dedup_documents(streaming_document_batches(...), threshold=0.8)
    for row in pack_rows(doc_iter, enc, bos_token_id):
        ...

No external dependencies — stdlib only (hashlib, struct).
"""

import hashlib
import struct
import sys
import time


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

# 128 hash functions parameterized by (a, b) for MinHash.
# We use h(x) = (a * x + b) mod p, with p a large prime.
_NUM_HASHES = 128
_PRIME = (1 << 61) - 1  # Mersenne prime 2^61 - 1

# Deterministic seeds — pack index into bytes, hash to get a/b pairs.
_HASH_A = []
_HASH_B = []
for _i in range(_NUM_HASHES):
    _seed = hashlib.sha256(struct.pack("<I", _i)).digest()
    _a = int.from_bytes(_seed[:8], "little") % _PRIME
    _b = int.from_bytes(_seed[8:16], "little") % _PRIME
    if _a == 0:
        _a = 1
    _HASH_A.append(_a)
    _HASH_B.append(_b)

_HASH_A = tuple(_HASH_A)
_HASH_B = tuple(_HASH_B)


def _shingle_hashes(text, n=5):
    """Return set of 32-bit hashes for word n-grams (shingles)."""
    words = text.split()
    if len(words) < n:
        # For very short docs, use whatever we have
        h = int.from_bytes(hashlib.md5(text.encode("utf-8", errors="replace")).digest()[:4], "little")
        return {h}
    hashes = set()
    for i in range(len(words) - n + 1):
        shingle = " ".join(words[i:i + n])
        h = int.from_bytes(
            hashlib.md5(shingle.encode("utf-8", errors="replace")).digest()[:4],
            "little",
        )
        hashes.add(h)
    return hashes


def _minhash_signature(shingle_set):
    """Compute MinHash signature (tuple of _NUM_HASHES minimum hashes)."""
    sig = [_PRIME] * _NUM_HASHES
    for h in shingle_set:
        for i in range(_NUM_HASHES):
            val = (_HASH_A[i] * h + _HASH_B[i]) % _PRIME
            if val < sig[i]:
                sig[i] = val
    return tuple(sig)


# ---------------------------------------------------------------------------
# LSH banding for approximate nearest-neighbor duplicate detection
# ---------------------------------------------------------------------------

# With 128 hashes, use 16 bands of 8 rows each.
# For threshold t, we want P(candidate) ~ 1 - (1 - t^r)^b where r=rows, b=bands.
# With r=8, b=16: P(0.8 sim) ~ 1 - (1 - 0.8^8)^16 ~ 0.98 (high recall)
# P(0.5 sim) ~ 1 - (1 - 0.5^8)^16 ~ 0.06 (low false positives)
_NUM_BANDS = 16
_ROWS_PER_BAND = _NUM_HASHES // _NUM_BANDS  # 8


def _band_hashes(sig):
    """Return list of band hash values for LSH lookup."""
    bands = []
    for b in range(_NUM_BANDS):
        start = b * _ROWS_PER_BAND
        band_slice = sig[start:start + _ROWS_PER_BAND]
        # Hash the band slice to a single value for bucket lookup
        h = hashlib.md5(struct.pack(f"<{_ROWS_PER_BAND}Q", *band_slice)).digest()[:8]
        bands.append((b, h))
    return bands


def _jaccard_from_sigs(sig_a, sig_b):
    """Estimate Jaccard similarity from two MinHash signatures."""
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / _NUM_HASHES


# ---------------------------------------------------------------------------
# MinHash near-dedup generator
# ---------------------------------------------------------------------------

def dedup_documents(doc_iter, threshold=0.8, shingle_n=5, stats_interval=50000):
    """
    Wrap a document batch iterator, yielding batches with near-duplicates removed.

    Uses MinHash + LSH banding. Memory scales with number of unique documents
    (stores one signature per doc). CPU cost is O(shingles) per doc for hashing,
    plus O(candidates) for LSH lookups — avoids O(n^2) all-pairs comparison.

    Args:
        doc_iter: iterator yielding list[str] batches (from streaming_document_batches)
        threshold: Jaccard similarity threshold for dedup (default 0.8)
        shingle_n: word n-gram size for shingling (default 5)
        stats_interval: print stats every N docs seen

    Yields:
        list[str] batches with duplicates removed (may yield smaller batches)
    """
    # LSH index: band_id -> {bucket_hash -> [sig_index, ...]}
    lsh_buckets = [{} for _ in range(_NUM_BANDS)]
    # All stored signatures (list, index = sig_id)
    signatures = []

    total_seen = 0
    total_dupes = 0
    t_start = time.time()

    for batch in doc_iter:
        kept = []
        for doc in batch:
            total_seen += 1

            # Skip empty/tiny docs
            if len(doc.split()) < shingle_n:
                kept.append(doc)
                continue

            # Compute MinHash signature
            shingles = _shingle_hashes(doc, n=shingle_n)
            sig = _minhash_signature(shingles)
            bands = _band_hashes(sig)

            # LSH lookup: collect candidate indices from matching buckets
            candidates = set()
            for band_id, bucket_key in bands:
                bucket_map = lsh_buckets[band_id]
                if bucket_key in bucket_map:
                    candidates.update(bucket_map[bucket_key])

            # Check candidates for actual similarity
            is_dup = False
            for cand_idx in candidates:
                sim = _jaccard_from_sigs(sig, signatures[cand_idx])
                if sim >= threshold:
                    is_dup = True
                    break

            if is_dup:
                total_dupes += 1
            else:
                # Add to index
                sig_idx = len(signatures)
                signatures.append(sig)
                for band_id, bucket_key in bands:
                    bucket_map = lsh_buckets[band_id]
                    if bucket_key not in bucket_map:
                        bucket_map[bucket_key] = []
                    bucket_map[bucket_key].append(sig_idx)
                kept.append(doc)

            # Stats
            if total_seen % stats_interval == 0:
                elapsed = time.time() - t_start
                rate = total_seen / elapsed if elapsed > 0 else 0
                dedup_pct = 100.0 * total_dupes / total_seen if total_seen > 0 else 0
                print(
                    f"[dedup] seen={total_seen} dupes={total_dupes} "
                    f"({dedup_pct:.1f}%) unique_sigs={len(signatures)} "
                    f"rate={rate:.0f} docs/s elapsed={elapsed:.1f}s",
                    file=sys.stderr,
                )

        if kept:
            yield kept

    # Final stats
    elapsed = time.time() - t_start
    dedup_pct = 100.0 * total_dupes / total_seen if total_seen > 0 else 0
    print(
        f"[dedup] DONE seen={total_seen} dupes={total_dupes} "
        f"({dedup_pct:.1f}%) unique_sigs={len(signatures)} "
        f"elapsed={elapsed:.1f}s",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Exact-match dedup (simpler/faster fallback)
# ---------------------------------------------------------------------------

def exact_dedup_documents(doc_iter, stats_interval=50000):
    """
    Wrap a document batch iterator, removing exact-duplicate documents.

    Uses SHA-256 of the document text. Much faster than MinHash but only
    catches exact duplicates, not near-duplicates.

    Args:
        doc_iter: iterator yielding list[str] batches
        stats_interval: print stats every N docs seen

    Yields:
        list[str] batches with exact duplicates removed
    """
    seen_hashes = set()
    total_seen = 0
    total_dupes = 0
    t_start = time.time()

    for batch in doc_iter:
        kept = []
        for doc in batch:
            total_seen += 1
            h = hashlib.sha256(doc.encode("utf-8", errors="replace")).digest()[:16]
            if h in seen_hashes:
                total_dupes += 1
            else:
                seen_hashes.add(h)
                kept.append(doc)

            if total_seen % stats_interval == 0:
                elapsed = time.time() - t_start
                rate = total_seen / elapsed if elapsed > 0 else 0
                dedup_pct = 100.0 * total_dupes / total_seen if total_seen > 0 else 0
                print(
                    f"[exact-dedup] seen={total_seen} dupes={total_dupes} "
                    f"({dedup_pct:.1f}%) unique={len(seen_hashes)} "
                    f"rate={rate:.0f} docs/s elapsed={elapsed:.1f}s",
                    file=sys.stderr,
                )

        if kept:
            yield kept

    elapsed = time.time() - t_start
    dedup_pct = 100.0 * total_dupes / total_seen if total_seen > 0 else 0
    print(
        f"[exact-dedup] DONE seen={total_seen} dupes={total_dupes} "
        f"({dedup_pct:.1f}%) unique={len(seen_hashes)} "
        f"elapsed={elapsed:.1f}s",
        file=sys.stderr,
    )
