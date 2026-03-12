use anyhow::Result;
use autoresearch_ops::board::{self, BoardCommand};
use autoresearch_ops::tickets::{self, TicketCommand};
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "autoresearch-ops", about = "File-backed ops and message board CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Manage repo tickets
    Tickets {
        #[command(subcommand)]
        command: TicketCommand,
    },
    /// Manage ticket-threaded board messages
    Board {
        #[command(subcommand)]
        command: BoardCommand,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Tickets { command } => tickets::run(command),
        Command::Board { command } => board::run(command),
    }
}
