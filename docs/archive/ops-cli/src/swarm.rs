use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct SwarmConfig {
    #[serde(default = "default_wip_limit")]
    pub default_wip_limit: usize,
    #[serde(default)]
    pub assignee: Vec<AssigneeConfig>,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            default_wip_limit: default_wip_limit(),
            assignee: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AssigneeConfig {
    pub name: String,
    #[serde(default = "default_wip_limit")]
    pub wip_limit: usize,
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub team: String,
    #[serde(default)]
    pub model: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentRole {
    Director,
    Lead,
    Worker,
    Unknown,
}

pub fn load_swarm_config(root: &Path) -> Result<SwarmConfig> {
    let candidates = [
        root.join("ops").join("swarm.toml"),
        root.join("ops").join("swarm.example.toml"),
    ];

    for path in candidates {
        if path.exists() {
            let text = fs::read_to_string(&path)?;
            let cfg: SwarmConfig = toml::from_str(&text)
                .with_context(|| format!("failed to parse {}", path.display()))?;
            return Ok(cfg);
        }
    }

    Ok(SwarmConfig::default())
}

impl SwarmConfig {
    pub fn assignee_wip_limit(&self, assignee: &str) -> usize {
        self.assignee
            .iter()
            .find(|entry| entry.name == assignee)
            .map(|entry| entry.wip_limit)
            .unwrap_or(self.default_wip_limit)
    }

    pub fn role_of(&self, assignee: &str) -> AgentRole {
        self.assignee
            .iter()
            .find(|entry| entry.name == assignee)
            .map(|entry| role_from_str(&entry.role))
            .unwrap_or(AgentRole::Unknown)
    }

    pub fn default_lead_assignee(&self) -> String {
        self.assignee
            .iter()
            .find(|entry| role_from_str(&entry.role) == AgentRole::Lead)
            .map(|entry| entry.name.clone())
            .unwrap_or_else(|| "opus-a".to_string())
    }
}

fn role_from_str(value: &str) -> AgentRole {
    match value.trim() {
        "human_orchestrator" | "director" => AgentRole::Director,
        "lead" => AgentRole::Lead,
        "worker" => AgentRole::Worker,
        _ => AgentRole::Unknown,
    }
}

fn default_wip_limit() -> usize {
    2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_lead_falls_back_to_opus_a() {
        assert_eq!(SwarmConfig::default().default_lead_assignee(), "opus-a");
    }

    #[test]
    fn configured_lead_overrides_default() {
        let cfg = SwarmConfig {
            default_wip_limit: 2,
            assignee: vec![
                AssigneeConfig {
                    name: "director".to_string(),
                    role: "human_orchestrator".to_string(),
                    team: String::new(),
                    model: String::new(),
                    wip_limit: 2,
                },
                AssigneeConfig {
                    name: "opus-chief".to_string(),
                    role: "lead".to_string(),
                    team: String::new(),
                    model: String::new(),
                    wip_limit: 2,
                },
            ],
        };

        assert_eq!(cfg.default_lead_assignee(), "opus-chief");
        assert_eq!(cfg.role_of("opus-chief"), AgentRole::Lead);
        assert_eq!(cfg.role_of("director"), AgentRole::Director);
    }
}
