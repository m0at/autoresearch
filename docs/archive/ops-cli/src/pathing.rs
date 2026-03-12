use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow, ensure};

const ROOT_ENV: &str = "AUTORESEARCH_OPS_ROOT";

pub fn workspace_root() -> Result<PathBuf> {
    if let Ok(path) = std::env::var(ROOT_ENV) {
        let root = PathBuf::from(path);
        ensure!(
            looks_like_workspace_root(&root),
            "{ROOT_ENV} does not point at an autoresearch repo root: {}",
            root.display()
        );
        return Ok(root);
    }

    let mut dir = std::env::current_dir()?;
    loop {
        if looks_like_workspace_root(&dir) {
            return Ok(dir);
        }
        if !dir.pop() {
            break;
        }
    }

    Err(anyhow!(
        "could not find repo root; run inside the autoresearch repo or set {ROOT_ENV}"
    ))
}

pub fn now_unix_s() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| anyhow!("system clock before unix epoch: {e}"))?
        .as_secs())
}

fn looks_like_workspace_root(dir: &Path) -> bool {
    dir.join("Cargo.toml").is_file() && dir.join("ops").is_dir()
}
