"""
Drop-in replacement for feeder.py's pack_rows with O(log n) best-fit packing.

Uses a sorted buffer + bisect for O(log n) lookup of the largest document
that fits remaining space, instead of O(n) linear scan.
"""

import sys
import bisect
import time

MAX_SEQ_LEN = 2048
ROW_CAPACITY = MAX_SEQ_LEN + 1  # 2049
BUFFER_SIZE_DEFAULT = 5000


def dense_pack_rows(doc_iter, enc, bos_token_id, buffer_size=BUFFER_SIZE_DEFAULT):
    """Best-fit pack tokenized documents into 2049-token rows.

    Same interface and output format as feeder.pack_rows — yields lists of
    exactly 2049 ints. Uses a sorted buffer with binary search for O(log n)
    best-fit lookup instead of linear scan.
    """
    # Sorted buffer: list of (length, doc_tokens) kept sorted by length.
    # We use bisect on the length keys extracted into a parallel list.
    buf_lens = []   # sorted list of lengths (mirrors buf_docs)
    buf_docs = []   # corresponding token lists

    # Stats
    total_tokens_used = 0
    total_docs_used = 0
    rows_produced = 0
    t_start = time.time()

    def refill_buffer():
        try:
            text_batch = next(doc_iter)
        except StopIteration:
            return False
        token_lists = enc.encode_ordinary_batch(text_batch, num_threads=16)
        for toks in token_lists:
            toks.insert(0, bos_token_id)
            length = len(toks)
            idx = bisect.bisect_left(buf_lens, length)
            buf_lens.insert(idx, length)
            buf_docs.insert(idx, toks)
        return True

    while True:
        row = []
        pos = 0
        docs_in_row = 0

        while pos < ROW_CAPACITY:
            while len(buf_lens) < buffer_size:
                if not refill_buffer():
                    break
            if len(buf_lens) == 0:
                if pos > 0:
                    row.extend([0] * (ROW_CAPACITY - pos))
                    pos = ROW_CAPACITY
                    break
                else:
                    return

            remaining = ROW_CAPACITY - pos

            # Binary search: find largest doc with length <= remaining
            idx = bisect.bisect_right(buf_lens, remaining) - 1

            if idx >= 0:
                # Found a doc that fits
                doc = buf_docs.pop(idx)
                buf_lens.pop(idx)
                row.extend(doc)
                pos += len(doc)
                docs_in_row += 1
            else:
                # Nothing fits — truncate the smallest doc that exceeds remaining
                # (minimizes tail waste vs always picking the shortest)
                trunc_idx = bisect.bisect_left(buf_lens, remaining + 1)
                if trunc_idx >= len(buf_lens):
                    trunc_idx = 0  # fallback: all docs are ≤ remaining but nothing fit (shouldn't happen)
                doc = buf_docs.pop(trunc_idx)
                buf_lens.pop(trunc_idx)
                row.extend(doc[:remaining])
                pos += remaining
                docs_in_row += 1

        assert len(row) == ROW_CAPACITY, f"row length {len(row)} != {ROW_CAPACITY}"

        # Update stats
        tokens_used = sum(1 for t in row if t != 0)
        total_tokens_used += tokens_used
        total_docs_used += docs_in_row
        rows_produced += 1

        if rows_produced % 10000 == 0:
            elapsed = time.time() - t_start
            avg_tokens = total_tokens_used / rows_produced
            avg_docs = total_docs_used / rows_produced
            waste = 100.0 * (1.0 - total_tokens_used / (rows_produced * ROW_CAPACITY))
            print(
                f"[dense_pack] {rows_produced} rows, "
                f"avg_tokens={avg_tokens:.1f}/{ROW_CAPACITY}, "
                f"avg_docs/row={avg_docs:.1f}, "
                f"waste={waste:.2f}%, "
                f"{elapsed:.0f}s",
                file=sys.stderr,
            )

        yield row
