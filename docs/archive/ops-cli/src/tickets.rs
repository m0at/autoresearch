use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use clap::{Args, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};

use crate::pathing::{now_unix_s, workspace_root};
use crate::swarm::{SwarmConfig, load_swarm_config};

const TICKET_PREFIX: &str = "AR";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum TicketStatus {
    #[value(name = "open")]
    Open,
    #[value(name = "in_progress")]
    InProgress,
    #[value(name = "blocked")]
    Blocked,
    #[value(name = "closed")]
    Closed,
}

impl Display for TicketStatus {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open => write!(f, "open"),
            Self::InProgress => write!(f, "in_progress"),
            Self::Blocked => write!(f, "blocked"),
            Self::Closed => write!(f, "closed"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum TicketPriority {
    #[value(name = "p0")]
    P0,
    #[value(name = "p1")]
    P1,
    #[value(name = "p2")]
    P2,
    #[value(name = "p3")]
    P3,
}

impl Display for TicketPriority {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::P0 => write!(f, "p0"),
            Self::P1 => write!(f, "p1"),
            Self::P2 => write!(f, "p2"),
            Self::P3 => write!(f, "p3"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticket {
    pub id: String,
    pub title: String,
    pub status: TicketStatus,
    pub priority: TicketPriority,
    #[serde(default)]
    pub assignee: Option<String>,
    #[serde(default)]
    pub swarm: Option<String>,
    #[serde(default)]
    pub parent: Option<String>,
    #[serde(default)]
    pub labels: Vec<String>,
    #[serde(default)]
    pub description: String,
    pub created_unix_s: u64,
    pub updated_unix_s: u64,
    #[serde(default)]
    pub started_unix_s: Option<u64>,
    #[serde(default)]
    pub closed_unix_s: Option<u64>,
}

impl Ticket {
    pub fn save(&self, ticket_dir: &Path) -> Result<()> {
        fs::create_dir_all(ticket_dir)?;
        let text = toml::to_string_pretty(self)?;
        fs::write(ticket_path(ticket_dir, &self.id), text)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Subcommand)]
pub enum TicketCommand {
    /// Create a new ticket
    New(NewTicketArgs),
    /// Update an existing ticket
    Update(UpdateTicketArgs),
    /// Show active tickets in a compact list
    List(ListTicketsArgs),
    /// Show full details for one ticket
    Show {
        /// Ticket id, e.g. AR-0001
        id: String,
    },
    /// Summarize ticket counts, assignee load, and swarm load
    #[command(visible_alias = "status")]
    Summary,
}

#[derive(Debug, Clone, Args)]
pub struct NewTicketArgs {
    #[arg(long)]
    pub title: String,
    #[arg(long)]
    pub assignee: Option<String>,
    #[arg(long)]
    pub swarm: Option<String>,
    #[arg(long)]
    pub parent: Option<String>,
    #[arg(long, value_delimiter = ',')]
    pub labels: Vec<String>,
    #[arg(long, default_value_t = TicketPriority::P2)]
    pub priority: TicketPriority,
    #[arg(long, default_value_t = TicketStatus::Open)]
    pub status: TicketStatus,
    #[arg(long, default_value = "")]
    pub description: String,
}

#[derive(Debug, Clone, Args)]
pub struct UpdateTicketArgs {
    pub id: String,
    #[arg(long)]
    pub title: Option<String>,
    #[arg(long)]
    pub status: Option<TicketStatus>,
    #[arg(long)]
    pub priority: Option<TicketPriority>,
    #[arg(long)]
    pub assignee: Option<String>,
    #[arg(long, default_value_t = false)]
    pub clear_assignee: bool,
    #[arg(long)]
    pub swarm: Option<String>,
    #[arg(long, default_value_t = false)]
    pub clear_swarm: bool,
    #[arg(long)]
    pub parent: Option<String>,
    #[arg(long, default_value_t = false)]
    pub clear_parent: bool,
    #[arg(long)]
    pub description: Option<String>,
    #[arg(long, value_delimiter = ',')]
    pub labels: Option<Vec<String>>,
}

#[derive(Debug, Clone, Args)]
pub struct ListTicketsArgs {
    #[arg(long)]
    pub status: Option<TicketStatus>,
    #[arg(long)]
    pub assignee: Option<String>,
    #[arg(long)]
    pub swarm: Option<String>,
    #[arg(long, default_value_t = false)]
    pub all: bool,
}

#[derive(Debug, Default, Clone, Copy)]
struct StatusCounts {
    open: usize,
    in_progress: usize,
    blocked: usize,
    closed: usize,
}

impl StatusCounts {
    fn record(&mut self, status: TicketStatus) {
        match status {
            TicketStatus::Open => self.open += 1,
            TicketStatus::InProgress => self.in_progress += 1,
            TicketStatus::Blocked => self.blocked += 1,
            TicketStatus::Closed => self.closed += 1,
        }
    }

    fn active(self) -> usize {
        self.open + self.in_progress + self.blocked
    }
}

pub fn run(command: TicketCommand) -> Result<()> {
    let root = workspace_root()?;
    let ticket_dir = ticket_dir(&root);
    let swarm_config = load_swarm_config(&root)?;

    match command {
        TicketCommand::New(args) => new_ticket(&ticket_dir, &swarm_config, args),
        TicketCommand::Update(args) => update_ticket(&ticket_dir, args),
        TicketCommand::List(args) => list_tickets(&ticket_dir, args),
        TicketCommand::Show { id } => show_ticket(&ticket_dir, &id),
        TicketCommand::Summary => summarize_tickets(&ticket_dir, &swarm_config),
    }
}

pub fn ticket_dir(root: &Path) -> PathBuf {
    root.join("ops").join("tickets")
}

pub fn load_ticket(ticket_dir: &Path, id: &str) -> Result<Ticket> {
    let path = ticket_path(ticket_dir, id);
    let text = fs::read_to_string(&path)
        .with_context(|| format!("failed to read ticket {} at {}", id, path.display()))?;
    let ticket: Ticket =
        toml::from_str(&text).with_context(|| format!("failed to parse ticket {}", id))?;
    Ok(ticket)
}

pub fn load_all_tickets(ticket_dir: &Path) -> Result<Vec<Ticket>> {
    if !ticket_dir.exists() {
        return Ok(Vec::new());
    }

    let mut tickets = Vec::new();
    for entry in fs::read_dir(ticket_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("toml") {
            continue;
        }
        let text = fs::read_to_string(&path)?;
        let ticket: Ticket =
            toml::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))?;
        tickets.push(ticket);
    }
    tickets.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(tickets)
}

pub fn next_ticket_id(ticket_dir: &Path) -> Result<String> {
    let mut max_seen = 0_u32;
    for ticket in load_all_tickets(ticket_dir)? {
        if let Some(num) = ticket
            .id
            .strip_prefix(&format!("{TICKET_PREFIX}-"))
            .and_then(|rest| rest.parse::<u32>().ok())
        {
            max_seen = max_seen.max(num);
        }
    }
    Ok(format!("{TICKET_PREFIX}-{:04}", max_seen + 1))
}

fn new_ticket(ticket_dir: &Path, swarm_config: &SwarmConfig, args: NewTicketArgs) -> Result<()> {
    ensure!(!args.title.trim().is_empty(), "title must not be empty");

    let now = now_unix_s()?;
    let id = next_ticket_id(ticket_dir)?;
    let ticket = Ticket {
        id: id.clone(),
        title: args.title.trim().to_string(),
        status: args.status,
        priority: args.priority,
        assignee: normalize_optional(args.assignee)
            .or_else(|| Some(swarm_config.default_lead_assignee())),
        swarm: normalize_optional(args.swarm),
        parent: normalize_optional(args.parent),
        labels: normalize_labels(args.labels),
        description: args.description.trim().to_string(),
        created_unix_s: now,
        updated_unix_s: now,
        started_unix_s: match args.status {
            TicketStatus::InProgress | TicketStatus::Blocked => Some(now),
            _ => None,
        },
        closed_unix_s: if args.status == TicketStatus::Closed {
            Some(now)
        } else {
            None
        },
    };
    ticket.save(ticket_dir)?;
    print_ticket_line(&ticket);
    Ok(())
}

fn update_ticket(ticket_dir: &Path, args: UpdateTicketArgs) -> Result<()> {
    let mut ticket = load_ticket(ticket_dir, &args.id)?;
    let now = now_unix_s()?;

    if let Some(title) = args.title {
        ensure!(!title.trim().is_empty(), "title must not be empty");
        ticket.title = title.trim().to_string();
    }
    if let Some(priority) = args.priority {
        ticket.priority = priority;
    }
    if let Some(status) = args.status {
        if ticket.started_unix_s.is_none()
            && matches!(status, TicketStatus::InProgress | TicketStatus::Blocked)
        {
            ticket.started_unix_s = Some(now);
        }
        ticket.closed_unix_s = if status == TicketStatus::Closed {
            Some(now)
        } else {
            None
        };
        ticket.status = status;
    }
    if let Some(description) = args.description {
        ticket.description = description.trim().to_string();
    }
    if let Some(labels) = args.labels {
        ticket.labels = normalize_labels(labels);
    }

    apply_option_update(&mut ticket.assignee, args.assignee, args.clear_assignee);
    apply_option_update(&mut ticket.swarm, args.swarm, args.clear_swarm);
    apply_option_update(&mut ticket.parent, args.parent, args.clear_parent);

    ticket.updated_unix_s = now;
    ticket.save(ticket_dir)?;
    print_ticket_line(&ticket);
    Ok(())
}

fn list_tickets(ticket_dir: &Path, args: ListTicketsArgs) -> Result<()> {
    let mut tickets = load_all_tickets(ticket_dir)?;
    tickets.retain(|ticket| {
        if !args.all && args.status.is_none() && ticket.status == TicketStatus::Closed {
            return false;
        }
        if let Some(status) = args.status && ticket.status != status {
            return false;
        }
        if let Some(ref assignee) = args.assignee
            && ticket.assignee.as_deref() != Some(assignee.as_str())
        {
            return false;
        }
        if let Some(ref swarm) = args.swarm
            && ticket.swarm.as_deref() != Some(swarm.as_str())
        {
            return false;
        }
        true
    });

    if tickets.is_empty() {
        println!("no tickets");
        return Ok(());
    }

    for ticket in tickets {
        print_ticket_line(&ticket);
    }
    Ok(())
}

fn show_ticket(ticket_dir: &Path, id: &str) -> Result<()> {
    let ticket = load_ticket(ticket_dir, id)?;
    println!("id: {}", ticket.id);
    println!("title: {}", ticket.title);
    println!("status: {}", ticket.status);
    println!("priority: {}", ticket.priority);
    println!(
        "assignee: {}",
        ticket.assignee.as_deref().unwrap_or("unassigned")
    );
    println!("swarm: {}", ticket.swarm.as_deref().unwrap_or("-"));
    println!("parent: {}", ticket.parent.as_deref().unwrap_or("-"));
    println!("labels: {}", join_or_dash(&ticket.labels));
    println!("created_unix_s: {}", ticket.created_unix_s);
    println!("updated_unix_s: {}", ticket.updated_unix_s);
    println!(
        "started_unix_s: {}",
        ticket
            .started_unix_s
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".to_string())
    );
    println!(
        "closed_unix_s: {}",
        ticket
            .closed_unix_s
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".to_string())
    );
    println!("description:");
    if ticket.description.is_empty() {
        println!("-");
    } else {
        println!("{}", ticket.description);
    }
    Ok(())
}

fn summarize_tickets(ticket_dir: &Path, swarm_config: &SwarmConfig) -> Result<()> {
    let tickets = load_all_tickets(ticket_dir)?;
    let mut totals = StatusCounts::default();
    let mut assignees: BTreeMap<String, StatusCounts> = BTreeMap::new();
    let mut swarms: BTreeMap<String, StatusCounts> = BTreeMap::new();

    for ticket in &tickets {
        totals.record(ticket.status);

        let assignee_key = ticket
            .assignee
            .clone()
            .unwrap_or_else(|| "unassigned".to_string());
        assignees
            .entry(assignee_key)
            .or_default()
            .record(ticket.status);

        let swarm_key = ticket
            .swarm
            .clone()
            .unwrap_or_else(|| "unspecified".to_string());
        swarms.entry(swarm_key).or_default().record(ticket.status);
    }

    println!("ticket_store: {}", ticket_dir.display());
    println!("total: {}", tickets.len());
    println!("open: {}", totals.open);
    println!("in_progress: {}", totals.in_progress);
    println!("blocked: {}", totals.blocked);
    println!("closed: {}", totals.closed);
    println!("active: {}", totals.active());

    let flow_share = if totals.active() == 0 {
        0.0
    } else {
        totals.in_progress as f64 / totals.active() as f64
    };
    println!("flow_ratio: {:.1}%", flow_share * 100.0);

    println!();
    println!("assignee_load:");
    if assignees.is_empty() {
        println!("  - none");
    } else {
        for (name, counts) in &assignees {
            let limit = swarm_config.assignee_wip_limit(name);
            let marker = if counts.in_progress > limit || counts.active() > limit + 1 {
                " OVERLOADED"
            } else if counts.active() == 0 {
                " IDLE"
            } else {
                ""
            };
            println!(
                "  - {}: active={} in_progress={} open={} blocked={} closed={} limit={}{}",
                name,
                counts.active(),
                counts.in_progress,
                counts.open,
                counts.blocked,
                counts.closed,
                limit,
                marker
            );
        }
    }

    println!();
    println!("swarm_load:");
    if swarms.is_empty() {
        println!("  - none");
    } else {
        for (name, counts) in &swarms {
            println!(
                "  - {}: active={} in_progress={} open={} blocked={} closed={}",
                name,
                counts.active(),
                counts.in_progress,
                counts.open,
                counts.blocked,
                counts.closed
            );
        }
    }

    let overloaded: Vec<String> = assignees
        .iter()
        .filter_map(|(name, counts)| {
            let limit = swarm_config.assignee_wip_limit(name);
            if counts.in_progress > limit || counts.active() > limit + 1 {
                Some(format!("{name} ({}/{})", counts.in_progress, limit))
            } else {
                None
            }
        })
        .collect();

    println!();
    println!("load_signals:");
    if overloaded.is_empty() {
        println!("  - no overloaded assignees");
    } else {
        println!("  - overloaded: {}", overloaded.join(", "));
    }
    println!(
        "  - recommendation: keep top-level leads at <=2 concurrent tickets and workers at <=1 in_progress ticket"
    );
    Ok(())
}

fn ticket_path(ticket_dir: &Path, id: &str) -> PathBuf {
    ticket_dir.join(format!("{id}.toml"))
}

fn normalize_optional(value: Option<String>) -> Option<String> {
    value.and_then(|v| {
        let trimmed = v.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    })
}

fn normalize_labels(labels: Vec<String>) -> Vec<String> {
    let mut clean: Vec<String> = labels
        .into_iter()
        .map(|label| label.trim().to_string())
        .filter(|label| !label.is_empty())
        .collect();
    clean.sort();
    clean.dedup();
    clean
}

fn apply_option_update(field: &mut Option<String>, value: Option<String>, clear: bool) {
    if clear {
        *field = None;
        return;
    }
    if let Some(value) = normalize_optional(value) {
        *field = Some(value);
    }
}

fn join_or_dash(values: &[String]) -> String {
    if values.is_empty() {
        "-".to_string()
    } else {
        values.join(", ")
    }
}

fn print_ticket_line(ticket: &Ticket) {
    println!(
        "{} {} {} [{}] owner={} swarm={} parent={} labels={}",
        ticket.id,
        ticket.status,
        ticket.priority,
        ticket.title,
        ticket.assignee.as_deref().unwrap_or("unassigned"),
        ticket.swarm.as_deref().unwrap_or("-"),
        ticket.parent.as_deref().unwrap_or("-"),
        join_or_dash(&ticket.labels),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swarm::AssigneeConfig;
    use tempfile::tempdir;

    fn sample_ticket(
        id: &str,
        status: TicketStatus,
        assignee: Option<&str>,
        swarm: Option<&str>,
    ) -> Ticket {
        Ticket {
            id: id.to_string(),
            title: format!("ticket {id}"),
            status,
            priority: TicketPriority::P2,
            assignee: assignee.map(|v| v.to_string()),
            swarm: swarm.map(|v| v.to_string()),
            parent: None,
            labels: vec!["ops".to_string()],
            description: String::new(),
            created_unix_s: 1,
            updated_unix_s: 1,
            started_unix_s: None,
            closed_unix_s: None,
        }
    }

    #[test]
    fn next_ticket_id_increments_existing_files() {
        let dir = tempdir().unwrap();
        let ticket_dir = dir.path().join("tickets");
        sample_ticket("AR-0001", TicketStatus::Open, None, None)
            .save(&ticket_dir)
            .unwrap();
        sample_ticket("AR-0007", TicketStatus::Closed, None, None)
            .save(&ticket_dir)
            .unwrap();

        assert_eq!(next_ticket_id(&ticket_dir).unwrap(), "AR-0008");
    }

    #[test]
    fn normalize_labels_deduplicates_and_sorts() {
        let labels = normalize_labels(vec![
            "agents".to_string(),
            " ops ".to_string(),
            String::new(),
            "agents".to_string(),
        ]);
        assert_eq!(labels, vec!["agents".to_string(), "ops".to_string()]);
    }

    #[test]
    fn load_all_tickets_sorts_by_id() {
        let dir = tempdir().unwrap();
        let ticket_dir = dir.path().join("tickets");
        sample_ticket(
            "AR-0002",
            TicketStatus::InProgress,
            Some("opus-a"),
            Some("swarm-a"),
        )
        .save(&ticket_dir)
        .unwrap();
        sample_ticket(
            "AR-0001",
            TicketStatus::Open,
            Some("sonnet-a1"),
            Some("swarm-a"),
        )
        .save(&ticket_dir)
        .unwrap();

        let tickets = load_all_tickets(&ticket_dir).unwrap();
        let ids: Vec<&str> = tickets.iter().map(|ticket| ticket.id.as_str()).collect();
        assert_eq!(ids, vec!["AR-0001", "AR-0002"]);
    }

    #[test]
    fn new_ticket_defaults_to_configured_lead() {
        let dir = tempdir().unwrap();
        let ticket_dir = dir.path().join("tickets");
        let cfg = SwarmConfig {
            default_wip_limit: 2,
            assignee: vec![AssigneeConfig {
                name: "opus-chief".to_string(),
                wip_limit: 2,
                role: "lead".to_string(),
                team: String::new(),
                model: String::new(),
            }],
        };

        new_ticket(
            &ticket_dir,
            &cfg,
            NewTicketArgs {
                title: "Investigate regression".to_string(),
                assignee: None,
                swarm: None,
                parent: None,
                labels: vec![],
                priority: TicketPriority::P2,
                status: TicketStatus::Open,
                description: String::new(),
            },
        )
        .unwrap();

        let ticket = load_ticket(&ticket_dir, "AR-0001").unwrap();
        assert_eq!(ticket.assignee.as_deref(), Some("opus-chief"));
    }
}
