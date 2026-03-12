use std::collections::{BTreeMap, HashSet};
use std::fmt::{Display, Formatter};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, bail, ensure};
use clap::{Args, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};

use crate::pathing::{now_unix_s, workspace_root};
use crate::swarm::{AgentRole, SwarmConfig, load_swarm_config};
use crate::tickets::{load_ticket, ticket_dir};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum MessageKind {
    #[value(name = "request")]
    Request,
    #[value(name = "plan")]
    Plan,
    #[value(name = "decision")]
    Decision,
    #[value(name = "delegation")]
    Delegation,
    #[value(name = "result")]
    Result,
    #[value(name = "blocker")]
    Blocker,
    #[value(name = "note")]
    Note,
    #[value(name = "ack")]
    Ack,
}

impl Display for MessageKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Request => write!(f, "request"),
            Self::Plan => write!(f, "plan"),
            Self::Decision => write!(f, "decision"),
            Self::Delegation => write!(f, "delegation"),
            Self::Result => write!(f, "result"),
            Self::Blocker => write!(f, "blocker"),
            Self::Note => write!(f, "note"),
            Self::Ack => write!(f, "ack"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageEvent {
    pub id: String,
    pub ticket_id: String,
    pub from: String,
    pub to: String,
    pub kind: MessageKind,
    pub body: String,
    pub created_unix_s: u64,
    #[serde(default)]
    pub reply_to: Option<String>,
    pub expects_reply: bool,
    #[serde(default)]
    pub artifacts: Vec<String>,
}

#[derive(Debug, Clone, Subcommand)]
pub enum BoardCommand {
    /// Post a new message to a ticket thread
    Post(PostMessageArgs),
    /// Reply to an existing message in a ticket thread
    Reply(ReplyMessageArgs),
    /// Show the full thread for one ticket
    Thread {
        #[arg(long = "ticket")]
        ticket_id: String,
    },
    /// Show unresolved messages addressed to an agent, with thread context
    Inbox {
        #[arg(long)]
        agent: String,
    },
    /// Show one-line unresolved message summaries for an agent
    Unresolved {
        #[arg(long)]
        agent: String,
    },
}

#[derive(Debug, Clone, Args)]
pub struct PostMessageArgs {
    #[arg(long = "ticket")]
    pub ticket_id: String,
    #[arg(long)]
    pub from: String,
    #[arg(long)]
    pub to: String,
    #[arg(long)]
    pub kind: MessageKind,
    #[arg(long)]
    pub body: String,
    #[arg(long, default_value_t = false)]
    pub expects_reply: bool,
    #[arg(long, default_value_t = false)]
    pub no_expects_reply: bool,
    #[arg(long, value_delimiter = ',')]
    pub artifacts: Vec<String>,
}

#[derive(Debug, Clone, Args)]
pub struct ReplyMessageArgs {
    #[arg(long = "ticket")]
    pub ticket_id: String,
    #[arg(long)]
    pub from: String,
    #[arg(long)]
    pub reply_to: String,
    #[arg(long)]
    pub kind: MessageKind,
    #[arg(long)]
    pub body: String,
    #[arg(long)]
    pub to: Option<String>,
    #[arg(long, default_value_t = false)]
    pub expects_reply: bool,
    #[arg(long, default_value_t = false)]
    pub no_expects_reply: bool,
    #[arg(long, value_delimiter = ',')]
    pub artifacts: Vec<String>,
}

pub fn run(command: BoardCommand) -> Result<()> {
    let root = workspace_root()?;
    let swarm = load_swarm_config(&root)?;
    let store = BoardStore::new(root, swarm);

    match command {
        BoardCommand::Post(args) => post_message(&store, args),
        BoardCommand::Reply(args) => reply_message(&store, args),
        BoardCommand::Thread { ticket_id } => show_thread(&store, &ticket_id),
        BoardCommand::Inbox { agent } => show_inbox(&store, &agent),
        BoardCommand::Unresolved { agent } => show_unresolved(&store, &agent),
    }
}

struct BoardStore {
    root: PathBuf,
    threads_dir: PathBuf,
    tickets_dir: PathBuf,
    swarm: SwarmConfig,
}

impl BoardStore {
    fn new(root: PathBuf, swarm: SwarmConfig) -> Self {
        Self {
            threads_dir: root.join("ops").join("threads"),
            tickets_dir: ticket_dir(&root),
            root,
            swarm,
        }
    }

    fn ensure_ticket_exists(&self, ticket_id: &str) -> Result<()> {
        let _ = load_ticket(&self.tickets_dir, ticket_id)?;
        Ok(())
    }

    fn read_thread(&self, ticket_id: &str) -> Result<Vec<MessageEvent>> {
        let path = self.thread_path(ticket_id);
        if !path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let mut messages = Vec::new();
        for (idx, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let message: MessageEvent = serde_json::from_str(&line).with_context(|| {
                format!("failed to parse {} line {}", path.display(), idx + 1)
            })?;
            messages.push(message);
        }
        Ok(messages)
    }

    fn append_message(&self, message: &MessageEvent) -> Result<()> {
        fs::create_dir_all(&self.threads_dir)?;
        let path = self.thread_path(&message.ticket_id);
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(file, "{}", serde_json::to_string(message)?)?;
        Ok(())
    }

    fn next_message_id(&self, ticket_id: &str, thread: &[MessageEvent]) -> String {
        let mut max_seen = 0_u32;
        let prefix = format!("{ticket_id}-MSG-");
        for message in thread {
            if let Some(num) = message
                .id
                .strip_prefix(&prefix)
                .and_then(|rest| rest.parse::<u32>().ok())
            {
                max_seen = max_seen.max(num);
            }
        }
        format!("{ticket_id}-MSG-{:04}", max_seen + 1)
    }

    fn thread_path(&self, ticket_id: &str) -> PathBuf {
        self.threads_dir.join(format!("{ticket_id}.jsonl"))
    }

    fn message_by_id<'a>(&self, thread: &'a [MessageEvent], message_id: &str) -> Option<&'a MessageEvent> {
        thread.iter().find(|message| message.id == message_id)
    }

    fn pending_for_agent(&self, agent: &str) -> Result<Vec<PendingMessage>> {
        let mut pending = Vec::new();
        if !self.threads_dir.exists() {
            return Ok(pending);
        }

        let mut files: Vec<PathBuf> = fs::read_dir(&self.threads_dir)?
            .filter_map(|entry| entry.ok().map(|item| item.path()))
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("jsonl"))
            .collect();
        files.sort();

        for path in files {
            let ticket_id = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .ok_or_else(|| anyhow::anyhow!("invalid thread file name: {}", path.display()))?
                .to_string();
            let thread = self.read_thread(&ticket_id)?;
            let replied_ids: HashSet<&str> = thread
                .iter()
                .filter_map(|message| message.reply_to.as_deref())
                .collect();

            for message in thread.iter().filter(|message| {
                message.to == agent
                    && message.expects_reply
                    && !replied_ids.contains(message.id.as_str())
            }) {
                let ticket = load_ticket(&self.tickets_dir, &ticket_id)?;
                pending.push(PendingMessage {
                    ticket_id: ticket.id,
                    ticket_title: ticket.title,
                    message: message.clone(),
                });
            }
        }

        pending.sort_by(|a, b| {
            a.message
                .created_unix_s
                .cmp(&b.message.created_unix_s)
                .then(a.ticket_id.cmp(&b.ticket_id))
                .then(a.message.id.cmp(&b.message.id))
        });
        Ok(pending)
    }

    fn validate_routing(
        &self,
        from: &str,
        to: &str,
        kind: MessageKind,
        expects_reply: bool,
        explicit_expects_reply: Option<bool>,
    ) -> Result<()> {
        let from_role = self.swarm.role_of(from);
        let to_role = self.swarm.role_of(to);

        if matches!(from_role, AgentRole::Worker) && matches!(to_role, AgentRole::Worker) && from != to
        {
            bail!("worker-to-worker direct messages are not allowed");
        }

        if matches!(from_role, AgentRole::Worker) && !matches!(to_role, AgentRole::Lead) {
            bail!("workers may only send messages to leads");
        }

        if kind == MessageKind::Delegation && matches!(from_role, AgentRole::Worker) {
            bail!("workers may not send delegation messages");
        }

        if kind == MessageKind::Delegation && explicit_expects_reply != Some(true) {
            bail!("delegation messages require --expects-reply");
        }

        if kind == MessageKind::Delegation && !expects_reply {
            bail!("delegation messages must expect a reply");
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
struct PendingMessage {
    ticket_id: String,
    ticket_title: String,
    message: MessageEvent,
}

fn post_message(store: &BoardStore, args: PostMessageArgs) -> Result<()> {
    ensure!(!args.ticket_id.trim().is_empty(), "ticket id must not be empty");
    ensure!(!args.from.trim().is_empty(), "from must not be empty");
    ensure!(!args.to.trim().is_empty(), "to must not be empty");
    ensure!(!args.body.trim().is_empty(), "body must not be empty");

    store.ensure_ticket_exists(&args.ticket_id)?;
    let thread = store.read_thread(&args.ticket_id)?;
    let explicit_expects_reply = tri_state_flag(args.expects_reply, args.no_expects_reply)?;
    let expects_reply = resolve_expects_reply(args.kind, explicit_expects_reply);

    store.validate_routing(
        &args.from,
        &args.to,
        args.kind,
        expects_reply,
        explicit_expects_reply,
    )?;

    let message = MessageEvent {
        id: store.next_message_id(&args.ticket_id, &thread),
        ticket_id: args.ticket_id,
        from: args.from.trim().to_string(),
        to: args.to.trim().to_string(),
        kind: args.kind,
        body: args.body.trim().to_string(),
        created_unix_s: now_unix_s()?,
        reply_to: None,
        expects_reply,
        artifacts: normalize_artifacts(args.artifacts),
    };
    store.append_message(&message)?;
    print_message_line(&message);
    Ok(())
}

fn reply_message(store: &BoardStore, args: ReplyMessageArgs) -> Result<()> {
    ensure!(!args.ticket_id.trim().is_empty(), "ticket id must not be empty");
    ensure!(!args.from.trim().is_empty(), "from must not be empty");
    ensure!(!args.reply_to.trim().is_empty(), "reply_to must not be empty");
    ensure!(!args.body.trim().is_empty(), "body must not be empty");

    store.ensure_ticket_exists(&args.ticket_id)?;
    let thread = store.read_thread(&args.ticket_id)?;
    let original = store
        .message_by_id(&thread, &args.reply_to)
        .ok_or_else(|| anyhow::anyhow!("message {} not found in ticket {}", args.reply_to, args.ticket_id))?;
    let to = args.to.unwrap_or_else(|| original.from.clone());
    let explicit_expects_reply = tri_state_flag(args.expects_reply, args.no_expects_reply)?;
    let expects_reply = resolve_expects_reply(args.kind, explicit_expects_reply);

    store.validate_routing(
        args.from.trim(),
        to.trim(),
        args.kind,
        expects_reply,
        explicit_expects_reply,
    )?;

    let message = MessageEvent {
        id: store.next_message_id(&args.ticket_id, &thread),
        ticket_id: args.ticket_id,
        from: args.from.trim().to_string(),
        to: to.trim().to_string(),
        kind: args.kind,
        body: args.body.trim().to_string(),
        created_unix_s: now_unix_s()?,
        reply_to: Some(args.reply_to),
        expects_reply,
        artifacts: normalize_artifacts(args.artifacts),
    };
    store.append_message(&message)?;
    print_message_line(&message);
    Ok(())
}

fn show_thread(store: &BoardStore, ticket_id: &str) -> Result<()> {
    store.ensure_ticket_exists(ticket_id)?;
    let ticket = load_ticket(&store.tickets_dir, ticket_id)?;
    let thread = store.read_thread(ticket_id)?;
    let replied_ids: HashSet<&str> = thread
        .iter()
        .filter_map(|message| message.reply_to.as_deref())
        .collect();

    println!("ticket: {} [{}]", ticket.id, ticket.title);
    println!("owner: {}", ticket.assignee.as_deref().unwrap_or("unassigned"));
    println!("thread_file: {}", store.thread_path(ticket_id).display());

    if thread.is_empty() {
        println!("no messages");
        return Ok(());
    }

    println!();
    for message in &thread {
        let waiting = if message.expects_reply && !replied_ids.contains(message.id.as_str()) {
            " waiting"
        } else {
            ""
        };
        println!(
            "{} {} {} -> {} created={} expects_reply={}{}",
            message.id,
            message.kind,
            message.from,
            message.to,
            message.created_unix_s,
            message.expects_reply,
            waiting
        );
        if let Some(reply_to) = message.reply_to.as_deref() {
            println!("reply_to: {}", reply_to);
        }
        println!("body: {}", message.body);
        if !message.artifacts.is_empty() {
            println!("artifacts: {}", message.artifacts.join(", "));
        }
        println!();
    }

    Ok(())
}

fn show_inbox(store: &BoardStore, agent: &str) -> Result<()> {
    let pending = store.pending_for_agent(agent)?;
    println!("agent: {}", agent);
    println!("repo: {}", store.root.display());

    if pending.is_empty() {
        println!("no unresolved messages");
        return Ok(());
    }

    let mut grouped: BTreeMap<&str, Vec<&PendingMessage>> = BTreeMap::new();
    for item in &pending {
        grouped.entry(&item.ticket_id).or_default().push(item);
    }

    for (ticket_id, items) in grouped {
        let title = &items[0].ticket_title;
        println!();
        println!("{ticket_id} [{title}]");
        for item in items {
            println!(
                "  {} {} from={} created={}",
                item.message.id, item.message.kind, item.message.from, item.message.created_unix_s
            );
            println!("  body: {}", item.message.body);
            if !item.message.artifacts.is_empty() {
                println!("  artifacts: {}", item.message.artifacts.join(", "));
            }
        }
    }

    Ok(())
}

fn show_unresolved(store: &BoardStore, agent: &str) -> Result<()> {
    let pending = store.pending_for_agent(agent)?;
    if pending.is_empty() {
        println!("no unresolved messages for {}", agent);
        return Ok(());
    }

    for item in pending {
        println!(
            "{} {} {} from={} to={} title={} body={}",
            item.ticket_id,
            item.message.id,
            item.message.kind,
            item.message.from,
            item.message.to,
            item.ticket_title,
            item.message.body
        );
    }
    Ok(())
}

fn resolve_expects_reply(kind: MessageKind, explicit: Option<bool>) -> bool {
    explicit.unwrap_or_else(|| default_expects_reply(kind))
}

fn default_expects_reply(kind: MessageKind) -> bool {
    matches!(kind, MessageKind::Request | MessageKind::Delegation | MessageKind::Blocker)
}

fn tri_state_flag(expect: bool, no_expect: bool) -> Result<Option<bool>> {
    ensure!(
        !(expect && no_expect),
        "--expects-reply and --no-expects-reply cannot be used together"
    );

    Ok(if expect {
        Some(true)
    } else if no_expect {
        Some(false)
    } else {
        None
    })
}

fn normalize_artifacts(values: Vec<String>) -> Vec<String> {
    let mut clean: Vec<String> = values
        .into_iter()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .collect();
    clean.sort();
    clean.dedup();
    clean
}

fn print_message_line(message: &MessageEvent) {
    println!(
        "{} {} {} -> {} ticket={} expects_reply={}",
        message.id, message.kind, message.from, message.to, message.ticket_id, message.expects_reply
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swarm::AssigneeConfig;
    use crate::tickets::{Ticket, TicketPriority, TicketStatus};
    use tempfile::tempdir;

    fn sample_swarm() -> SwarmConfig {
        SwarmConfig {
            default_wip_limit: 2,
            assignee: vec![
                AssigneeConfig {
                    name: "director".to_string(),
                    role: "human_orchestrator".to_string(),
                    team: "control".to_string(),
                    model: "human".to_string(),
                    wip_limit: 2,
                },
                AssigneeConfig {
                    name: "opus-a".to_string(),
                    role: "lead".to_string(),
                    team: "swarm-a".to_string(),
                    model: "claude-opus".to_string(),
                    wip_limit: 2,
                },
                AssigneeConfig {
                    name: "sonnet-a1".to_string(),
                    role: "worker".to_string(),
                    team: "swarm-a".to_string(),
                    model: "claude-sonnet".to_string(),
                    wip_limit: 1,
                },
                AssigneeConfig {
                    name: "sonnet-a2".to_string(),
                    role: "worker".to_string(),
                    team: "swarm-a".to_string(),
                    model: "claude-sonnet".to_string(),
                    wip_limit: 1,
                },
            ],
        }
    }

    fn sample_ticket(id: &str) -> Ticket {
        Ticket {
            id: id.to_string(),
            title: format!("ticket {id}"),
            status: TicketStatus::Open,
            priority: TicketPriority::P2,
            assignee: Some("opus-a".to_string()),
            swarm: Some("swarm-a".to_string()),
            parent: None,
            labels: vec![],
            description: String::new(),
            created_unix_s: 1,
            updated_unix_s: 1,
            started_unix_s: None,
            closed_unix_s: None,
        }
    }

    fn sample_store() -> (tempfile::TempDir, BoardStore) {
        let dir = tempdir().unwrap();
        let root = dir.path().to_path_buf();
        fs::create_dir_all(root.join("ops").join("tickets")).unwrap();
        sample_ticket("AR-0001")
            .save(&root.join("ops").join("tickets"))
            .unwrap();
        let store = BoardStore::new(root, sample_swarm());
        (dir, store)
    }

    #[test]
    fn append_and_read_preserves_message_order() {
        let (_dir, store) = sample_store();
        let first = MessageEvent {
            id: "AR-0001-MSG-0001".to_string(),
            ticket_id: "AR-0001".to_string(),
            from: "director".to_string(),
            to: "opus-a".to_string(),
            kind: MessageKind::Request,
            body: "first".to_string(),
            created_unix_s: 1,
            reply_to: None,
            expects_reply: true,
            artifacts: vec![],
        };
        let second = MessageEvent {
            id: "AR-0001-MSG-0002".to_string(),
            ticket_id: "AR-0001".to_string(),
            from: "opus-a".to_string(),
            to: "director".to_string(),
            kind: MessageKind::Plan,
            body: "second".to_string(),
            created_unix_s: 2,
            reply_to: Some("AR-0001-MSG-0001".to_string()),
            expects_reply: false,
            artifacts: vec![],
        };

        store.append_message(&first).unwrap();
        store.append_message(&second).unwrap();
        let thread = store.read_thread("AR-0001").unwrap();
        let bodies: Vec<&str> = thread.iter().map(|message| message.body.as_str()).collect();
        assert_eq!(bodies, vec!["first", "second"]);
    }

    #[test]
    fn unresolved_request_appears_in_opus_inbox() {
        let (_dir, store) = sample_store();
        store
            .append_message(&MessageEvent {
                id: "AR-0001-MSG-0001".to_string(),
                ticket_id: "AR-0001".to_string(),
                from: "director".to_string(),
                to: "opus-a".to_string(),
                kind: MessageKind::Request,
                body: "investigate".to_string(),
                created_unix_s: 1,
                reply_to: None,
                expects_reply: true,
                artifacts: vec![],
            })
            .unwrap();

        let pending = store.pending_for_agent("opus-a").unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].message.id, "AR-0001-MSG-0001");
    }

    #[test]
    fn reply_clears_unresolved_message() {
        let (_dir, store) = sample_store();
        store
            .append_message(&MessageEvent {
                id: "AR-0001-MSG-0001".to_string(),
                ticket_id: "AR-0001".to_string(),
                from: "director".to_string(),
                to: "opus-a".to_string(),
                kind: MessageKind::Request,
                body: "investigate".to_string(),
                created_unix_s: 1,
                reply_to: None,
                expects_reply: true,
                artifacts: vec![],
            })
            .unwrap();
        store
            .append_message(&MessageEvent {
                id: "AR-0001-MSG-0002".to_string(),
                ticket_id: "AR-0001".to_string(),
                from: "opus-a".to_string(),
                to: "director".to_string(),
                kind: MessageKind::Plan,
                body: "on it".to_string(),
                created_unix_s: 2,
                reply_to: Some("AR-0001-MSG-0001".to_string()),
                expects_reply: false,
                artifacts: vec![],
            })
            .unwrap();

        let pending = store.pending_for_agent("opus-a").unwrap();
        assert!(pending.is_empty());
    }

    #[test]
    fn worker_to_worker_direct_message_is_rejected() {
        let (_dir, store) = sample_store();
        let err = store
            .validate_routing(
                "sonnet-a1",
                "sonnet-a2",
                MessageKind::Result,
                false,
                Some(false),
            )
            .unwrap_err();
        assert!(err.to_string().contains("worker-to-worker"));
    }

    #[test]
    fn delegation_requires_explicit_reply_expectation() {
        let (_dir, store) = sample_store();
        let err = store
            .validate_routing("opus-a", "sonnet-a1", MessageKind::Delegation, true, None)
            .unwrap_err();
        assert!(err.to_string().contains("require --expects-reply"));
    }
}
