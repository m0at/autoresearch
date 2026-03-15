use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

const POLL_INTERVAL: Duration = Duration::from_millis(5);
const BARRIER_DIR: &str = "/tmp";

fn barrier_path(name: &str, rank: usize) -> PathBuf {
    PathBuf::from(format!("{}/pipeline_barrier_{}_{}", BARRIER_DIR, name, rank))
}

fn broadcast_path(name: &str) -> PathBuf {
    PathBuf::from(format!("{}/pipeline_{}.bin", BARRIER_DIR, name))
}

fn loss_path(step: usize) -> PathBuf {
    PathBuf::from(format!("{}/pipeline_loss_{}.f32", BARRIER_DIR, step))
}

/// File-based barrier. Each rank writes a sentinel, then polls until all ranks
/// have written theirs. Once all are present, each rank deletes its own file.
pub fn barrier(rank: usize, world_size: usize, name: &str) {
    let my_path = barrier_path(name, rank);
    fs::write(&my_path, &[]).expect("barrier: failed to write sentinel");

    // Poll until every rank's file exists.
    loop {
        let all_present = (0..world_size).all(|r| barrier_path(name, r).exists());
        if all_present {
            break;
        }
        thread::sleep(POLL_INTERVAL);
    }

    // Clean up own file.
    let _ = fs::remove_file(&my_path);
}

/// Rank 0 writes `data` to a shared file; all other ranks poll until the file
/// appears, then read it. A trailing barrier ensures rank 0 doesn't delete the
/// file before others finish reading.
pub fn broadcast_bytes(rank: usize, world_size: usize, data: &[u8], name: &str) -> Vec<u8> {
    let path = broadcast_path(name);

    if rank == 0 {
        fs::write(&path, data).expect("broadcast: rank 0 failed to write");
    }

    // All ranks wait for the file.
    loop {
        if path.exists() {
            break;
        }
        thread::sleep(POLL_INTERVAL);
    }

    let mut buf = Vec::new();
    let mut f = fs::File::open(&path).expect("broadcast: failed to open");
    f.read_to_end(&mut buf).expect("broadcast: failed to read");

    // Barrier so rank 0 doesn't remove the file before others read it.
    barrier(rank, world_size, &format!("{}_done", name));

    if rank == 0 {
        let _ = fs::remove_file(&path);
    }

    buf
}

/// Process 3 (validator) writes a single f32 loss value for a given step.
pub fn write_loss(rank: usize, step: usize, loss: f32) {
    debug_assert_eq!(rank, 3, "write_loss should only be called by rank 3");
    let path = loss_path(step);
    let bytes = loss.to_le_bytes();
    fs::write(&path, &bytes).expect("write_loss: failed to write");
}

/// Process 0 (coordinator) tries to read the loss for a given step.
/// Returns `None` if the file doesn't exist yet.
pub fn read_loss(step: usize) -> Option<f32> {
    let path = loss_path(step);
    match fs::read(&path) {
        Ok(bytes) if bytes.len() == 4 => {
            let _ = fs::remove_file(&path);
            Some(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        }
        _ => None,
    }
}
