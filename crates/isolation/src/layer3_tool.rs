use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ToolKind {
    Python,
    Browser,
    Terminal,
    FileSystem,
    Shell,
    Compiler,
    Network,
    Database,
    Custom(String),
}

impl std::fmt::Display for ToolKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Python => write!(f, "python"),
            Self::Browser => write!(f, "browser"),
            Self::Terminal => write!(f, "terminal"),
            Self::FileSystem => write!(f, "filesystem"),
            Self::Shell => write!(f, "shell"),
            Self::Compiler => write!(f, "compiler"),
            Self::Network => write!(f, "network"),
            Self::Database => write!(f, "database"),
            Self::Custom(s) => write!(f, "custom-{s}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPod {
    pub id: Uuid,
    pub tool_kind: ToolKind,
    pub sandbox: SandboxSpec,
    pub network_access: bool,
    pub filesystem_write: bool,
    pub max_execution_seconds: u64,
    pub allowed_commands: Vec<String>,
    pub denied_commands: Vec<String>,
    pub status: ToolStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxSpec {
    pub enabled: bool,
    pub sandbox_kind: SandboxKind,
    pub seccomp_profile: String,
    pub read_only_root: bool,
    pub dropped_capabilities: Vec<String>,
    pub tmpfs_size_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SandboxKind {
    GVisor,
    Kata,
    Firecracker,
    NamespaceOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ToolStatus {
    Idle,
    Executing { agent_id: Uuid, started_at: chrono::DateTime<chrono::Utc> },
    Blocked,
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolGateway {
    pub id: Uuid,
    pub enabled: bool,
    pub tools: HashMap<ToolKind, ToolPod>,
    pub audit_log: Vec<ToolCallAudit>,
    pub rate_limiter: ToolRateLimiter,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallAudit {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub tool_kind: ToolKind,
    pub command: String,
    pub allowed: bool,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolRateLimiter {
    pub max_calls_per_minute: u32,
    pub max_calls_per_agent: u32,
    pub cooldown_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionRequest {
    pub agent_id: Uuid,
    pub tool_kind: ToolKind,
    pub command: String,
    pub args: Vec<String>,
    pub timeout_seconds: Option<u64>,
    pub env_vars: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionResult {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub execution_time_ms: u64,
    pub sandbox_violations: Vec<String>,
}

pub type SharedToolGateway = Arc<RwLock<ToolIsolationLayer>>;

pub struct ToolIsolationLayer {
    gateways: HashMap<ToolKind, ToolGateway>,
    allowed_tools: HashSet<ToolKind>,
}

impl ToolIsolationLayer {
    pub fn new(allowed: impl IntoIterator<Item = ToolKind>) -> Self {
        Self {
            gateways: HashMap::new(),
            allowed_tools: allowed.into_iter().collect(),
        }
    }

    pub fn register_tool(
        &mut self,
        kind: ToolKind,
        sandbox: SandboxSpec,
        network: bool,
        fs_write: bool,
    ) -> ToolPod {
        let pod = ToolPod {
            id: Uuid::new_v4(),
            tool_kind: kind.clone(),
            sandbox,
            network_access: network,
            filesystem_write: fs_write,
            max_execution_seconds: 300,
            allowed_commands: Vec::new(),
            denied_commands: Vec::new(),
            status: ToolStatus::Idle,
        };

        let gateway = ToolGateway {
            id: Uuid::new_v4(),
            enabled: true,
            tools: HashMap::new(),
            audit_log: Vec::new(),
            rate_limiter: ToolRateLimiter {
                max_calls_per_minute: 60,
                max_calls_per_agent: 30,
                cooldown_seconds: 1,
            },
        };

        self.gateways.entry(kind.clone()).or_insert(gateway).tools.insert(kind, pod.clone());
        pod
    }

    pub async fn request_execution(
        &mut self,
        request: ToolExecutionRequest,
    ) -> Result<ToolExecutionResult, ToolIsolationError> {
        let gateway = self.gateways.get_mut(&request.tool_kind)
            .ok_or(ToolIsolationError::ToolNotRegistered(request.tool_kind.clone()))?;

        if !gateway.enabled {
            return Err(ToolIsolationError::GatewayDisabled(request.tool_kind.clone()));
        }

        let pod = gateway.tools.get_mut(&request.tool_kind)
            .ok_or(ToolIsolationError::ToolPodNotFound(request.tool_kind.clone()))?;

        if !self.allowed_tools.contains(&request.tool_kind) {
            gateway.audit_log.push(ToolCallAudit {
                id: Uuid::new_v4(),
                agent_id: request.agent_id,
                tool_kind: request.tool_kind.clone(),
                command: request.command.clone(),
                allowed: false,
                timestamp: chrono::Utc::now(),
                reason: "Tool not in allowed list".into(),
            });
            return Err(ToolIsolationError::ToolNotAllowed(request.tool_kind.clone()));
        }

        if !pod.allowed_commands.is_empty() && !pod.allowed_commands.contains(&request.command) {
            return Err(ToolIsolationError::CommandNotAllowed(request.command));
        }

        if pod.denied_commands.contains(&request.command) {
            return Err(ToolIsolationError::CommandDenied(request.command));
        }

        let start = std::time::Instant::now();
        pod.status = ToolStatus::Executing {
            agent_id: request.agent_id,
            started_at: chrono::Utc::now(),
        };

        let result = execute_real_command(&request).await;

        pod.status = ToolStatus::Idle;
        gateway.audit_log.push(ToolCallAudit {
            id: Uuid::new_v4(),
            agent_id: request.agent_id,
            tool_kind: request.tool_kind,
            command: request.command,
            allowed: true,
            timestamp: chrono::Utc::now(),
            reason: "executed".into(),
        });

        Ok(result)
    }

    pub fn set_tool_allowed_commands(
        &mut self,
        kind: &ToolKind,
        commands: Vec<String>,
    ) -> Result<(), ToolIsolationError> {
        let gateway = self.gateways.get_mut(kind)
            .ok_or(ToolIsolationError::ToolNotRegistered(kind.clone()))?;
        let pod = gateway.tools.get_mut(kind)
            .ok_or(ToolIsolationError::ToolPodNotFound(kind.clone()))?;
        pod.allowed_commands = commands;
        Ok(())
    }

    pub fn set_tool_denied_commands(
        &mut self,
        kind: &ToolKind,
        commands: Vec<String>,
    ) -> Result<(), ToolIsolationError> {
        let gateway = self.gateways.get_mut(kind)
            .ok_or(ToolIsolationError::ToolNotRegistered(kind.clone()))?;
        let pod = gateway.tools.get_mut(kind)
            .ok_or(ToolIsolationError::ToolPodNotFound(kind.clone()))?;
        pod.denied_commands = commands;
        Ok(())
    }

    pub fn get_audit_log(&self, kind: &ToolKind) -> Vec<&ToolCallAudit> {
        self.gateways.get(kind)
            .map(|g| g.audit_log.iter().collect())
            .unwrap_or_default()
    }

    pub fn disable_tool(&mut self, kind: &ToolKind) -> Result<(), ToolIsolationError> {
        let gateway = self.gateways.get_mut(kind)
            .ok_or(ToolIsolationError::ToolNotRegistered(kind.clone()))?;
        gateway.enabled = false;
        Ok(())
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum ToolIsolationError {
    #[error("Tool not registered: {0}")]
    ToolNotRegistered(ToolKind),
    #[error("Gateway disabled for tool: {0}")]
    GatewayDisabled(ToolKind),
    #[error("Tool pod not found: {0}")]
    ToolPodNotFound(ToolKind),
    #[error("Tool not allowed: {0}")]
    ToolNotAllowed(ToolKind),
    #[error("Command not allowed: {0}")]
    CommandNotAllowed(String),
    #[error("Command denied: {0}")]
    CommandDenied(String),
}

/// Spawn the tool command in a real subprocess.
async fn execute_real_command(request: &ToolExecutionRequest) -> ToolExecutionResult {
    let start = std::time::Instant::now();

    let program = match request.tool_kind {
        ToolKind::Python => "python3",
        ToolKind::Shell | ToolKind::Terminal => "/bin/sh",
        ToolKind::Browser => "chromium",
        ToolKind::FileSystem => "ls",
        ToolKind::Network => "curl",
        _ => &request.command,
    };

    let output = tokio::process::Command::new(program)
        .args(&request.args)
        .env_clear()
        .envs(&request.env_vars)
        .kill_on_drop(true)
        .output()
        .await;

    match output {
        Ok(out) => ToolExecutionResult {
            success: out.status.success(),
            stdout: String::from_utf8_lossy(&out.stdout).to_string(),
            stderr: String::from_utf8_lossy(&out.stderr).to_string(),
            exit_code: out.status.code().unwrap_or(-1),
            execution_time_ms: start.elapsed().as_millis() as u64,
            sandbox_violations: Vec::new(),
        },
        Err(e) => ToolExecutionResult {
            success: false,
            stdout: String::new(),
            stderr: format!("Failed to execute command: {}", e),
            exit_code: -1,
            execution_time_ms: start.elapsed().as_millis() as u64,
            sandbox_violations: Vec::new(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_registration_and_execution() {
        let mut layer = ToolIsolationLayer::new(vec![ToolKind::Python, ToolKind::Browser]);
        let pod = layer.register_tool(
            ToolKind::Python,
            SandboxSpec {
                enabled: true,
                sandbox_kind: SandboxKind::GVisor,
                seccomp_profile: "default.json".into(),
                read_only_root: true,
                dropped_capabilities: vec!["ALL".into()],
                tmpfs_size_mb: 64,
            },
            false,
            false,
        );
        assert_eq!(pod.status, ToolStatus::Idle);

        let result = layer.request_execution(ToolExecutionRequest {
            agent_id: Uuid::new_v4(),
            tool_kind: ToolKind::Python,
            command: "run".into(),
            args: vec!["script.py".into()],
            timeout_seconds: None,
            env_vars: HashMap::new(),
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_denied_tool() {
        let mut layer = ToolIsolationLayer::new(vec![ToolKind::Browser]);
        layer.register_tool(
            ToolKind::Shell,
            SandboxSpec::default_tool(),
            false,
            false,
        );
        let result = layer.request_execution(ToolExecutionRequest {
            agent_id: Uuid::new_v4(),
            tool_kind: ToolKind::Shell,
            command: "rm".into(),
            args: vec!["-rf".into(), "/".into()],
            timeout_seconds: None,
            env_vars: HashMap::new(),
        });
        assert!(result.is_err());
    }
}

impl Default for SandboxSpec {
    fn default() -> Self {
        Self {
            enabled: true,
            sandbox_kind: SandboxKind::GVisor,
            seccomp_profile: "restricted.json".into(),
            read_only_root: true,
            dropped_capabilities: vec!["ALL".into()],
            tmpfs_size_mb: 64,
        }
    }
}

impl SandboxSpec {
    pub fn default_tool() -> Self {
        Self::default()
    }
}
