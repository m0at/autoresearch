use std::fs;
use std::path::Path;
use std::process::Command;

use tempfile::tempdir;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_autoresearch-ops")
}

fn write_repo_fixture(root: &Path) {
    fs::write(root.join("Cargo.toml"), "[workspace]\nmembers = []\n").unwrap();
    fs::create_dir_all(root.join("ops").join("tickets")).unwrap();
    fs::write(
        root.join("ops").join("swarm.example.toml"),
        r#"default_wip_limit = 2

[[assignee]]
name = "director"
role = "human_orchestrator"
team = "control"
model = "human"
wip_limit = 2

[[assignee]]
name = "opus-a"
role = "lead"
team = "swarm-a"
model = "claude-opus"
wip_limit = 2

[[assignee]]
name = "sonnet-a1"
role = "worker"
team = "swarm-a"
model = "claude-sonnet"
wip_limit = 1
"#,
    )
    .unwrap();
}

fn run(root: &Path, args: &[&str]) -> (String, String) {
    let output = Command::new(bin())
        .args(args)
        .current_dir(root)
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        output.status.success(),
        "command failed: {:?}\nstdout:\n{}\nstderr:\n{}",
        args,
        stdout,
        stderr
    );
    (stdout, stderr)
}

#[test]
fn ticket_thread_board_smoke_test() {
    let dir = tempdir().unwrap();
    write_repo_fixture(dir.path());

    let (stdout, _) = run(
        dir.path(),
        &["tickets", "new", "--title", "Investigate DEPTH=14 regression"],
    );
    assert!(stdout.contains("AR-0001"));
    assert!(stdout.contains("owner=opus-a"));

    let (stdout, _) = run(
        dir.path(),
        &[
            "board",
            "post",
            "--ticket",
            "AR-0001",
            "--from",
            "director",
            "--to",
            "opus-a",
            "--kind",
            "request",
            "--body",
            "Investigate why DEPTH=14 regressed.",
        ],
    );
    assert!(stdout.contains("AR-0001-MSG-0001"));
    assert!(stdout.contains("expects_reply=true"));

    let (stdout, _) = run(dir.path(), &["board", "inbox", "--agent", "opus-a"]);
    assert!(stdout.contains("AR-0001-MSG-0001"));
    assert!(stdout.contains("Investigate why DEPTH=14 regressed."));

    let (stdout, _) = run(
        dir.path(),
        &[
            "board",
            "reply",
            "--ticket",
            "AR-0001",
            "--from",
            "opus-a",
            "--reply-to",
            "AR-0001-MSG-0001",
            "--kind",
            "plan",
            "--body",
            "I will fan this into worker tickets.",
        ],
    );
    assert!(stdout.contains("AR-0001-MSG-0002"));

    let (stdout, _) = run(dir.path(), &["board", "thread", "--ticket", "AR-0001"]);
    assert!(stdout.contains("AR-0001-MSG-0001"));
    assert!(stdout.contains("AR-0001-MSG-0002"));
    assert!(stdout.contains("I will fan this into worker tickets."));

    let (stdout, _) = run(dir.path(), &["board", "unresolved", "--agent", "opus-a"]);
    assert!(stdout.contains("no unresolved messages for opus-a"));
}
