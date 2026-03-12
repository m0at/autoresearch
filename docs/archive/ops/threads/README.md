# Thread Store

Each ticket thread is stored as append-only JSONL:

- one file per ticket, for example `AR-0001.jsonl`
- one JSON message event per line
- replies reference earlier messages via `reply_to`

The CLI creates this directory automatically if it does not exist:

```bash
cargo run -p autoresearch-ops -- board post --ticket AR-0001 ...
```
