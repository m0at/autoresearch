# Ops Board

`autoresearch` now has a file-backed, ticket-threaded message board.

Storage:

- tickets: `ops/tickets/*.toml`
- thread messages: `ops/threads/*.jsonl`
- swarm identities: `ops/swarm.toml` or `ops/swarm.example.toml`

Use the standalone CLI:

```bash
cargo run -p autoresearch-ops -- tickets summary
cargo run -p autoresearch-ops -- board inbox --agent opus-a
```

## How to communicate with Opus

1. Create or update a ticket assigned to the lead.
2. Post a `request` message on that ticket thread.
3. Wait for an Opus `plan`, `decision`, `result`, or `blocker` reply.

Example:

```bash
cargo run -p autoresearch-ops -- tickets new \
  --title "Investigate DEPTH=14 regression"

cargo run -p autoresearch-ops -- board post \
  --ticket AR-0001 \
  --from director \
  --to opus-a \
  --kind request \
  --body "Investigate why DEPTH=14 regressed and return a ranked hypothesis list."
```

Then Opus replies on the same thread:

```bash
cargo run -p autoresearch-ops -- board reply \
  --ticket AR-0001 \
  --from opus-a \
  --reply-to AR-0001-MSG-0001 \
  --kind plan \
  --body "I will split this into worker tickets and return a ranked synthesis."
```

Check what Opus still needs:

```bash
cargo run -p autoresearch-ops -- board unresolved --agent director
```

Check what is waiting on Opus:

```bash
cargo run -p autoresearch-ops -- board inbox --agent opus-a
```

## Routing rules

- Tickets carry ownership.
- Thread messages carry the conversation.
- Humans and leads may open new requests.
- Workers reply upward to leads.
- Worker-to-worker direct messaging is rejected.
- Delegation must be explicit and must expect a reply.
