/// Best-fit bin packing — exact replica of Python feeder.py pack_rows().
///
/// Maintains a buffer of tokenized documents. For each output row (2049 tokens):
/// find the largest doc that fits the remaining space (best-fit), pop it; if
/// nothing fits, pop the shortest doc and crop it to fill the row. Pad with
/// zeros if the doc source is exhausted mid-row.
///
/// The refill loop pulls whole batches from the source until the buffer has at
/// least BUFFER_SIZE (1000) docs, matching Python's behavior where each
/// refill_buffer() call extends the buffer by an entire batch (~128 docs). This
/// means the buffer can temporarily exceed 1000.

pub const ROW_CAPACITY: usize = 2049; // MAX_SEQ_LEN + 1
const BUFFER_SIZE: usize = 1000;

/// Best-fit bin packer.
///
/// `B` yields batches of tokenized documents (`Vec<Vec<u16>>`).
/// Each call to `next()` produces one packed row of exactly 2049 u16 tokens.
pub struct Packer<B: Iterator<Item = Vec<Vec<u16>>>> {
    batch_source: B,
    doc_buffer: Vec<Vec<u16>>,
    exhausted: bool,
}

impl<B: Iterator<Item = Vec<Vec<u16>>>> Packer<B> {
    pub fn new(batch_source: B) -> Self {
        Self {
            batch_source,
            doc_buffer: Vec::with_capacity(BUFFER_SIZE + 128),
            exhausted: false,
        }
    }

    /// Pull whole batches from batch_source until buffer >= BUFFER_SIZE or source exhausted.
    /// Matches Python's `while len(doc_buffer) < BUFFER_SIZE: refill_buffer()` where each
    /// refill_buffer() extends with an entire batch.
    fn refill_buffer(&mut self) {
        while self.doc_buffer.len() < BUFFER_SIZE && !self.exhausted {
            match self.batch_source.next() {
                Some(batch) => self.doc_buffer.extend(batch),
                None => self.exhausted = true,
            }
        }
    }
}

impl<B: Iterator<Item = Vec<Vec<u16>>>> Iterator for Packer<B> {
    type Item = [u16; ROW_CAPACITY];

    fn next(&mut self) -> Option<[u16; ROW_CAPACITY]> {
        let mut row = [0u16; ROW_CAPACITY];
        let mut pos = 0usize;

        while pos < ROW_CAPACITY {
            // Refill buffer to >= BUFFER_SIZE if possible
            self.refill_buffer();

            if self.doc_buffer.is_empty() {
                if pos > 0 {
                    // Pad remainder with zeros (already zero-initialized)
                    break;
                } else {
                    // No docs at all, nothing started — done
                    return None;
                }
            }

            let remaining = ROW_CAPACITY - pos;

            // Find the LARGEST document that fits in remaining space
            let mut best_idx: Option<usize> = None;
            let mut best_len: usize = 0;
            for (i, doc) in self.doc_buffer.iter().enumerate() {
                let doc_len = doc.len();
                if doc_len <= remaining && doc_len > best_len {
                    best_idx = Some(i);
                    best_len = doc_len;
                }
            }

            if let Some(idx) = best_idx {
                // Pop the best-fit doc (remove to preserve order — matches Python list.pop(i))
                let doc = self.doc_buffer.remove(idx);
                row[pos..pos + doc.len()].copy_from_slice(&doc);
                pos += doc.len();
            } else {
                // No doc fits — pop the SHORTEST doc and crop it
                // Python: min(range(len(buf)), key=lambda i: len(buf[i]))
                // min picks first occurrence on tie — min_by_key does the same
                let shortest_idx = self
                    .doc_buffer
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, d)| d.len())
                    .unwrap()
                    .0;
                let doc = self.doc_buffer.remove(shortest_idx);
                row[pos..pos + remaining].copy_from_slice(&doc[..remaining]);
                pos += remaining;
            }
        }

        // pos == ROW_CAPACITY in normal operation; can be less only when source
        // exhausted mid-row (row is zero-padded via initialization).
        Some(row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: wrap individual docs into single-doc batches for testing.
    fn as_batches(docs: Vec<Vec<u16>>) -> Vec<Vec<Vec<u16>>> {
        docs.into_iter().map(|d| vec![d]).collect()
    }

    #[test]
    fn exact_fit() {
        let batches = as_batches(vec![vec![1u16; ROW_CAPACITY]]);
        let mut packer = Packer::new(batches.into_iter());
        let row = packer.next().unwrap();
        assert!(row.iter().all(|&t| t == 1));
        assert!(packer.next().is_none());
    }

    #[test]
    fn padding_on_exhaustion() {
        let batches = as_batches(vec![vec![42u16; 100]]);
        let mut packer = Packer::new(batches.into_iter());
        let row = packer.next().unwrap();
        assert!(row[..100].iter().all(|&t| t == 42));
        assert!(row[100..].iter().all(|&t| t == 0));
        assert!(packer.next().is_none());
    }

    #[test]
    fn crop_when_nothing_fits() {
        let batches = as_batches(vec![
            vec![1u16; ROW_CAPACITY + 500],
            vec![2u16; ROW_CAPACITY + 100],
        ]);
        let mut packer = Packer::new(batches.into_iter());
        // First row: nothing fits, shortest (doc[1], len 2149) gets cropped
        let row = packer.next().unwrap();
        assert!(row.iter().all(|&t| t == 2));
        // Second row: doc[0] still too big, gets cropped
        let row = packer.next().unwrap();
        assert!(row.iter().all(|&t| t == 1));
        assert!(packer.next().is_none());
    }

    #[test]
    fn best_fit_picks_largest_fitting() {
        // All docs in one batch (like Python where buffer gets 3 docs at once)
        let batches = vec![vec![
            vec![1u16; 100],
            vec![2u16; 1000],
            vec![3u16; 2000],
        ]];
        let mut packer = Packer::new(batches.into_iter());
        let row = packer.next().unwrap();
        // big (3, len 2000) placed first
        assert!(row[..2000].iter().all(|&t| t == 3));
        // remaining = 49. Neither small (100) nor medium (1000) fits.
        // Shortest is small (100), crop to 49.
        assert!(row[2000..2049].iter().all(|&t| t == 1));
    }

    #[test]
    fn empty_source() {
        let batches: Vec<Vec<Vec<u16>>> = vec![];
        let mut packer = Packer::new(batches.into_iter());
        assert!(packer.next().is_none());
    }

    #[test]
    fn multi_doc_packing() {
        // Two small docs that fit together in one row
        let batches = vec![vec![vec![1u16; 1000], vec![2u16; 1049]]];
        let mut packer = Packer::new(batches.into_iter());
        let row = packer.next().unwrap();
        // Best fit picks larger doc (1049) first, then smaller (1000) fits exactly
        assert!(row[..1049].iter().all(|&t| t == 2));
        assert!(row[1049..2049].iter().all(|&t| t == 1));
        assert!(packer.next().is_none());
    }
}
