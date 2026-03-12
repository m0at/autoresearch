"""
Analyze raw document content from climbmix parquets.
Samples 50 docs from each of 5 random cached parquets (250 docs total).
Prints per-doc stats and top-20 repeated paragraph patterns.
Saves full output to /root/doc_review.txt.
"""

import sys
sys.path.insert(0, '/root/autoresearch/engine')
from feeder import download_parquet
import pyarrow.parquet as pq
import random
import zlib
import re
from collections import Counter

CACHE_DIR = "/tmp/feeder_cache"
OUTPUT_PATH = "/root/doc_review.txt"
PARQUETS_TO_SAMPLE = 5
DOCS_PER_PARQUET = 50
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# Pick 5 random shard indices from the training range (0..6541)
all_indices = list(range(0, 6541))
selected_indices = random.sample(all_indices, PARQUETS_TO_SAMPLE)
print(f"Selected shard indices: {selected_indices}", flush=True)

def compression_ratio(text):
    encoded = text.encode("utf-8")
    if not encoded:
        return 1.0
    compressed = zlib.compress(encoded, level=6)
    return len(compressed) / len(encoded)

def safe_repr(s, maxlen=200):
    """Return first maxlen chars with non-printable chars escaped."""
    s = s[:maxlen]
    return repr(s)[1:-1]  # strip outer quotes from repr

lines = []
paragraph_counter = Counter()

all_docs = []  # (parquet_idx, doc_idx, text)

for pi, shard_idx in enumerate(selected_indices):
    print(f"Downloading shard {shard_idx}...", flush=True)
    filepath = download_parquet(shard_idx, CACHE_DIR)
    pf = pq.ParquetFile(filepath)

    # Collect all docs from this parquet
    shard_docs = []
    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx)
        batch = rg.column("text").to_pylist()
        shard_docs.extend(batch)

    # Sample up to DOCS_PER_PARQUET
    sample_size = min(DOCS_PER_PARQUET, len(shard_docs))
    sampled = random.sample(shard_docs, sample_size)
    for di, text in enumerate(sampled):
        all_docs.append((pi, di, text))

    print(f"  shard {shard_idx}: {len(shard_docs)} total docs, sampled {sample_size}", flush=True)

print(f"\nTotal docs: {len(all_docs)}", flush=True)

# ---- Per-doc output ----
lines.append("=" * 80)
lines.append(f"DOCUMENT ANALYSIS — {len(all_docs)} docs from {PARQUETS_TO_SAMPLE} parquets")
lines.append("=" * 80)

for pi, di, text in all_docs:
    cr = compression_ratio(text)
    char_len = len(text)
    first200 = text[:200]
    last200 = text[-200:] if len(text) > 200 else text

    lines.append(f"\n--- DOC parquet={pi} doc={di} chars={char_len} compression_ratio={cr:.3f} ---")
    lines.append(f"FIRST 200: {repr(first200)}")
    lines.append(f"LAST  200: {repr(last200)}")

    # Count paragraphs for cross-doc pattern analysis
    paragraphs = text.split("\n\n")
    for para in paragraphs:
        para_stripped = para.strip()
        # Only track short-to-medium paragraphs (likely boilerplate)
        if 10 <= len(para_stripped) <= 300:
            paragraph_counter[para_stripped] += 1

# ---- Top-20 repeated paragraph patterns ----
lines.append("\n")
lines.append("=" * 80)
lines.append("TOP 20 REPEATED PARAGRAPH PATTERNS (appearing in 2+ docs)")
lines.append("=" * 80)

repeated = [(para, cnt) for para, cnt in paragraph_counter.items() if cnt >= 2]
repeated.sort(key=lambda x: -x[1])
top20 = repeated[:20]

for rank, (para, cnt) in enumerate(top20, 1):
    lines.append(f"\n[#{rank}] count={cnt} chars={len(para)}")
    lines.append(repr(para))
    lines.append("--- plain ---")
    lines.append(para)
    lines.append("--- end ---")

if not top20:
    lines.append("(No paragraphs appeared in 2+ docs)")

# ---- Additional: show ALL paragraphs with count >= 3 ----
lines.append("\n")
lines.append("=" * 80)
lines.append("ALL PARAGRAPHS WITH count >= 3")
lines.append("=" * 80)
threshold3 = [(p, c) for p, c in paragraph_counter.items() if c >= 3]
threshold3.sort(key=lambda x: -x[1])
for para, cnt in threshold3:
    lines.append(f"\n[count={cnt}] {repr(para)}")

# ---- Length distribution ----
lines.append("\n")
lines.append("=" * 80)
lines.append("LENGTH DISTRIBUTION")
lines.append("=" * 80)
lengths = [len(text) for _, _, text in all_docs]
lengths.sort()
n = len(lengths)
lines.append(f"min={lengths[0]} p10={lengths[n//10]} p25={lengths[n//4]} median={lengths[n//2]} p75={lengths[3*n//4]} p90={lengths[9*n//10]} max={lengths[-1]}")

# ---- Compression ratio distribution ----
lines.append("\n")
lines.append("=" * 80)
lines.append("COMPRESSION RATIO DISTRIBUTION")
lines.append("=" * 80)
crs = sorted([compression_ratio(text) for _, _, text in all_docs])
lines.append(f"min={crs[0]:.3f} p10={crs[n//10]:.3f} p25={crs[n//4]:.3f} median={crs[n//2]:.3f} p75={crs[3*n//4]:.3f} p90={crs[9*n//10]:.3f} max={crs[-1]:.3f}")

# ---- Docs with very low compression (likely repetitive/junk) ----
lines.append("\n")
lines.append("=" * 80)
lines.append("DOCS WITH COMPRESSION RATIO < 0.20 (potentially repetitive/junk)")
lines.append("=" * 80)
for pi, di, text in all_docs:
    cr = compression_ratio(text)
    if cr < 0.20:
        lines.append(f"\nparquet={pi} doc={di} cr={cr:.3f} chars={len(text)}")
        lines.append(f"FIRST 500: {repr(text[:500])}")

# ---- Docs with very high compression (random/code heavy) ----
lines.append("\n")
lines.append("=" * 80)
lines.append("DOCS WITH COMPRESSION RATIO > 0.80 (high entropy — code/random?)")
lines.append("=" * 80)
for pi, di, text in all_docs:
    cr = compression_ratio(text)
    if cr > 0.80:
        lines.append(f"\nparquet={pi} doc={di} cr={cr:.3f} chars={len(text)}")
        lines.append(f"FIRST 500: {repr(text[:500])}")

# ---- Common line patterns (short lines that appear across many docs) ----
lines.append("\n")
lines.append("=" * 80)
lines.append("COMMON SHORT LINES (len<=80, appearing in 5+ docs)")
lines.append("=" * 80)
line_counter = Counter()
for _, _, text in all_docs:
    seen_in_doc = set()
    for line in text.splitlines():
        ls = line.strip()
        if 5 <= len(ls) <= 80 and ls not in seen_in_doc:
            line_counter[ls] += 1
            seen_in_doc.add(ls)

common_lines = [(l, c) for l, c in line_counter.items() if c >= 5]
common_lines.sort(key=lambda x: -x[1])
for line, cnt in common_lines[:50]:
    lines.append(f"  count={cnt:4d}  {repr(line)}")

# ---- Write output ----
output = "\n".join(lines)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(output)

print(f"\nWrote {len(output)} chars to {OUTPUT_PATH}", flush=True)
print("DONE", flush=True)
