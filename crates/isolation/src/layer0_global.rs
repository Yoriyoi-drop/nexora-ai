use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalCluster {
    pub id: Uuid,
    pub name: String,
    pub version: String,
    pub api_gateway: ApiGatewaySpec,
    pub orchestrator: OrchestratorSpec,
    pub monitoring: MonitoringSpec,
    pub storage: StorageSpec,
    pub scheduler: SchedulerSpec,
    pub security_core: SecurityCoreSpec,
    pub service_mesh: ServiceMeshSpec,
    pub health: ClusterHealth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiGatewaySpec {
    pub enabled: bool,
    pub port: u16,
    pub tls_enabled: bool,
    pub rate_limit_global: u32,
    pub auth_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorSpec {
    pub enabled: bool,
    pub max_concurrent_modes: u32,
    pub scheduling_policy: SchedulingPolicy,
    pub heartbeat_interval_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulingPolicy {
    RoundRobin,
    PriorityFirst,
    ResourceAware,
    LeastLoaded,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringSpec {
    pub enabled: bool,
    pub metrics_export_interval_secs: u64,
    pub alert_channel: AlertChannel,
    pub retention_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertChannel {
    Prometheus,
    Webhook(String),
    Email(String),
    Slack(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageSpec {
    pub isolated: bool,
    pub encryption_at_rest: bool,
    pub backup_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerSpec {
    pub isolated: bool,
    pub preemptive_scheduling: bool,
    pub resource_quotas: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityCoreSpec {
    pub enabled: bool,
    pub audit_logging: bool,
    pub intrusion_detection: bool,
    pub policy_enforcement: PolicyEnforcement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyEnforcement {
    Enforce,
    AuditOnly,
    Permissive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMeshSpec {
    pub kind: String,
    pub mtls_enabled: bool,
    pub traffic_policy: TrafficPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrafficPolicy {
    AllowAll,
    DenyAll,
    IstioAuthorization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterHealth {
    pub status: ClusterStatus,
    pub active_modes: u32,
    pub total_agents: u32,
    pub cpu_usage_pct: f64,
    pub memory_usage_pct: f64,
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ClusterStatus {
    Initializing,
    Running,
    Degraded,
    Maintenance,
    Down,
}

pub type SharedGlobalCluster = Arc<RwLock<GlobalCluster>>;

pub struct GlobalSystemIsolation {
    cluster: SharedGlobalCluster,
    #[allow(dead_code)] // Reserved for future implementation
    mode_registry: Arc<RwLock<HashMap<Uuid, String>>>,
}

impl GlobalSystemIsolation {
    pub fn new(name: &str) -> Self {
        Self {
            cluster: Arc::new(RwLock::new(GlobalCluster {
                id: Uuid::new_v4(),
                name: name.to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                api_gateway: ApiGatewaySpec {
                    enabled: true,
                    port: 8080,
                    tls_enabled: true,
                    rate_limit_global: 10000,
                    auth_required: true,
                },
                orchestrator: OrchestratorSpec {
                    enabled: true,
                    max_concurrent_modes: 10,
                    scheduling_policy: SchedulingPolicy::ResourceAware,
                    heartbeat_interval_secs: 15,
                },
                monitoring: MonitoringSpec {
                    enabled: true,
                    metrics_export_interval_secs: 30,
                    alert_channel: AlertChannel::Prometheus,
                    retention_days: 30,
                },
                storage: StorageSpec {
                    isolated: true,
                    encryption_at_rest: true,
                    backup_enabled: true,
                },
                scheduler: SchedulerSpec {
                    isolated: true,
                    preemptive_scheduling: true,
                    resource_quotas: true,
                },
                security_core: SecurityCoreSpec {
                    enabled: true,
                    audit_logging: true,
                    intrusion_detection: true,
                    policy_enforcement: PolicyEnforcement::Enforce,
                },
                service_mesh: ServiceMeshSpec {
                    kind: "istio".into(),
                    mtls_enabled: true,
                    traffic_policy: TrafficPolicy::IstioAuthorization,
                },
                health: ClusterHealth {
                    status: ClusterStatus::Initializing,
                    active_modes: 0,
                    total_agents: 0,
                    cpu_usage_pct: 0.0,
                    memory_usage_pct: 0.0,
                    last_heartbeat: chrono::Utc::now(),
                },
            })),
            mode_registry: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn cluster(&self) -> &SharedGlobalCluster {
        &self.cluster
    }

    pub fn register_mode(&self, mode_id: Uuid, mode_name: &str) {
        let mut registry = self.mode_registry.write();
        registry.insert(mode_id, mode_name.to_string());
        let mut cluster = self.cluster.write();
        cluster.health.active_modes = registry.len() as u32;
    }

    pub fn unregister_mode(&self, mode_id: Uuid) {
        let mut registry = self.mode_registry.write();
        registry.remove(&mode_id);
        let mut cluster = self.cluster.write();
        cluster.health.active_modes = registry.len() as u32;
    }

    #[allow(dead_code)] // Reserved for future implementation
    pub fn heartbeat(&self) {
        let mut cluster = self.cluster.write();
        cluster.health.last_heartbeat = chrono::Utc::now();
    }

    pub fn set_health(&self, status: ClusterStatus) {
        let mut cluster = self.cluster.write();
        cluster.health.status = status;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_cluster_creation() {
        let iso = GlobalSystemIsolation::new("nexora-main");
        let cluster = iso.cluster.read();
        assert_eq!(cluster.name, "nexora-main");
        assert!(cluster.api_gateway.enabled);
        assert!(cluster.security_core.audit_logging);
    }

    #[test]
    fn test_mode_registration() {
        let iso = GlobalSystemIsolation::new("test-cluster");
        let mode_id = Uuid::new_v4();
        iso.register_mode(mode_id, "research");
        assert_eq!(iso.cluster.read().health.active_modes, 1);
        iso.unregister_mode(mode_id);
        assert_eq!(iso.cluster.read().health.active_modes, 0);
    }
}
