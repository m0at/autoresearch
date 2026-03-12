# Ticket Store

This directory holds the local agent ticket store for `autoresearch`.

Conventions:

- one ticket per file
- file name equals ticket id, for example `AR-0001.toml`
- state is edited through `cargo run -- tickets ...` first
- direct file edits are acceptable for bulk repair or agent automation

Useful commands:

```bash
cargo run -p autoresearch-ops -- tickets new --title "Add agent pane inspector"
cargo run -p autoresearch-ops -- tickets list
cargo run -p autoresearch-ops -- tickets show AR-0001
cargo run -p autoresearch-ops -- tickets status
```

The summary command is intended to be the fast text orientation command any agent can run before taking on work.
