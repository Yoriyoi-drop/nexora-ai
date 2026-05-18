use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationConfig {
    pub global: GlobalIsolationConfig,
    pub mode: ModeIsolationConfig,
    pub agent: AgentIsolationConfig,
    pub tool: ToolIsolationConfig,
    pub runtime: RuntimeIsolationConfig,
    pub cognitive: CognitiveIsolationConfig,
    pub permission: PermissionConfig,
    pub firewall: FirewallConfig,
    pub kill_switch: KillSwitchConfig,
    pub multi_cluster: MultiClusterConfig,
}

impl Default for IsolationConfig {
    fn default() -> Self {
        Self {
            global: GlobalIsolationConfig::default(),
            mode: ModeIsolationConfig::default(),
            agent: AgentIsolationConfig::default(),
            tool: ToolIsolationConfig::default(),
            runtime: RuntimeIsolationConfig::default(),
            cognitive: CognitiveIsolationConfig::default(),
            permission: PermissionConfig::default(),
            firewall: FirewallConfig::default(),
            kill_switch: KillSwitchConfig::default(),
            multi_cluster: MultiClusterConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalIsolationConfig {
    pub cluster_name: String,
    pub api_gateway_enabled: bool,
    pub orchestrator_enabled: bool,
    pub monitoring_enabled: bool,
    pub storage_isolation: bool,
    pub scheduler_isolation: bool,
    pub security_core_enabled: bool,
    pub service_mesh: ServiceMeshKind,
    pub observability_backend: ObservabilityBackend,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceMeshKind {
    Istio,
    Linkerd,
    Consul,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ObservabilityBackend {
    Prometheus,
    Datadog,
    GrafanaCloud,
    None,
}

impl Default for GlobalIsolationConfig {
    fn default() -> Self {
        Self {
            cluster_name: "nexora-cluster".into(),
            api_gateway_enabled: true,
            orchestrator_enabled: true,
            monitoring_enabled: true,
            storage_isolation: true,
            scheduler_isolation: true,
            security_core_enabled: true,
            service_mesh: ServiceMeshKind::Istio,
            observability_backend: ObservabilityBackend::Prometheus,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeIsolationConfig {
    pub enabled: bool,
    pub default_network_policy: NetworkPolicy,
    pub default_gpu_quota: GpuQuota,
    pub default_memory_quota_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkPolicy {
    DenyAll,
    AllowSameMode,
    AllowSameCluster,
    AllowSpecific { allowed_modes: Vec<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuQuota {
    pub count: u32,
    pub memory_mb: u64,
    pub share: bool,
}

impl Default for GpuQuota {
    fn default() -> Self {
        Self { count: 1, memory_mb: 8192, share: false }
    }
}

impl Default for ModeIsolationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_network_policy: NetworkPolicy::DenyAll,
            default_gpu_quota: GpuQuota::default(),
            default_memory_quota_mb: 16384,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentIsolationConfig {
    pub enabled: bool,
    pub separate_pod_per_agent: bool,
    pub dedicated_memory_buffer: bool,
    pub dedicated_runtime: bool,
    pub max_agents_per_mode: u32,
    pub agent_communication: AgentCommKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentCommKind {
    DirectMessageBus,
    GatewayOnly,
    FirewallWithAudit,
}

impl Default for AgentIsolationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            separate_pod_per_agent: true,
            dedicated_memory_buffer: true,
            dedicated_runtime: true,
            max_agents_per_mode: 50,
            agent_communication: AgentCommKind::FirewallWithAudit,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolIsolationConfig {
    pub enabled: bool,
    pub sandbox_per_tool: bool,
    pub tool_gateway_enabled: bool,
    pub allowed_tools: Vec<String>,
    pub max_tool_execution_seconds: u64,
    pub tool_network_access: bool,
}

impl Default for ToolIsolationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sandbox_per_tool: true,
            tool_gateway_enabled: true,
            allowed_tools: vec!["python".into(), "browser".into(), "terminal".into(), "filesystem".into()],
            max_tool_execution_seconds: 300,
            tool_network_access: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeIsolationConfig {
    pub enabled: bool,
    pub sandbox_kind: SandboxKind,
    pub restricted_linux: bool,
    pub seccomp_profile: String,
    pub read_only_root: bool,
    pub drop_capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SandboxKind {
    GVisor,
    KataContainers,
    Firecracker,
    NamespaceOnly,
}

impl Default for SandboxKind {
    fn default() -> Self { Self::GVisor }
}

impl Default for RuntimeIsolationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sandbox_kind: SandboxKind::GVisor,
            restricted_linux: true,
            seccomp_profile: "restricted.json".into(),
            read_only_root: true,
            drop_capabilities: vec!["ALL".into()],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveIsolationConfig {
    pub enabled: bool,
    pub isolate_memory_per_agent: bool,
    pub isolate_reasoning_chains: bool,
    pub max_chain_isolation_depth: u32,
    pub auto_quarantine_on_anomaly: bool,
    pub hallucination_spread_prevention: bool,
}

impl Default for CognitiveIsolationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            isolate_memory_per_agent: true,
            isolate_reasoning_chains: true,
            max_chain_isolation_depth: 10,
            auto_quarantine_on_anomaly: true,
            hallucination_spread_prevention: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionConfig {
    pub enabled: bool,
    pub rbac_enabled: bool,
    pub capability_based_access: bool,
    pub default_action: PermissionAction,
    pub audit_all_access: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PermissionAction {
    Allow,
    Deny,
    Audit,
}

impl Default for PermissionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rbac_enabled: true,
            capability_based_access: true,
            default_action: PermissionAction::Deny,
            audit_all_access: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallConfig {
    pub enabled: bool,
    pub default_egress: FirewallRule,
    pub default_ingress: FirewallRule,
    pub rate_limit_per_agent: u32,
    pub max_message_size_bytes: u64,
    pub audit_all_messages: bool,
    pub filter_suspicious: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FirewallRule {
    Allow,
    Deny,
    AuditAllow,
    AuditDeny,
}

impl Default for FirewallConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_egress: FirewallRule::AuditDeny,
            default_ingress: FirewallRule::Deny,
            rate_limit_per_agent: 100,
            max_message_size_bytes: 1048576,
            audit_all_messages: true,
            filter_suspicious: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillSwitchConfig {
    pub enabled: bool,
    pub confirm_before_kill: bool,
    pub grace_period_seconds: u64,
    pub auto_kill_on_anomaly: bool,
    pub kill_scope: KillScope,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KillScope {
    SingleAgent,
    EntireMode,
    SpecificMode(String),
    AllExcept(Vec<String>),
}

impl Default for KillSwitchConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            confirm_before_kill: true,
            grace_period_seconds: 30,
            auto_kill_on_anomaly: false,
            kill_scope: KillScope::SingleAgent,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiClusterConfig {
    pub enabled: bool,
    pub max_regional_clusters: u32,
    pub max_mode_clusters: u32,
    pub max_agent_clusters: u32,
    pub max_micro_vms: u32,
    pub auto_scaling: bool,
    pub cross_cluster_sync: bool,
}

impl Default for MultiClusterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_regional_clusters: 10,
            max_mode_clusters: 100,
            max_agent_clusters: 1000,
            max_micro_vms: 10000,
            auto_scaling: true,
            cross_cluster_sync: true,
        }
    }
}
