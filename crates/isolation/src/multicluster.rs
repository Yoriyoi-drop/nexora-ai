//! Multi-cluster orchestration system for distributed agent execution.
//!
//! Provides cross-region deployment, load balancing, health monitoring,
//! and automatic failover for multi-cluster agent deployments.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::{interval, sleep};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::layer1_mode::{ModeId, ModeKind};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiClusterSystem {
    pub regions: HashMap<String, RegionalCluster>,
    pub global_config: GlobalMultiClusterConfig,
    pub sync_status: ClusterSyncStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionalCluster {
    pub id: Uuid,
    pub name: String,
    pub region: String,
    pub mode_clusters: HashMap<String, ModeCluster>,
    pub status: RegionalStatus,
    pub latency_ms: u64,
    pub capacity_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeCluster {
    pub id: Uuid,
    pub mode_id: ModeId,
    pub mode_kind: ModeKind,
    pub agent_clusters: HashMap<String, AgentCluster>,
    pub status: ModeClusterStatus,
    pub isolation_policy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCluster {
    pub id: Uuid,
    pub name: String,
    pub micro_vms: Vec<MicroVm>,
    pub status: AgentClusterStatus,
    pub scaling_policy: ScalingPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroVm {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub tool_runtime: ToolRuntime,
    pub execution_threads: Vec<ExecutionThread>,
    pub resource_usage: MicroVmResources,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolRuntime {
    pub tool_type: String,
    pub sandbox_kind: String,
    pub max_threads: u32,
    pub memory_limit_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionThread {
    pub id: Uuid,
    pub task_id: String,
    pub cpu_usage_pct: f64,
    pub memory_usage_mb: u64,
    pub status: ThreadStatus,
    pub started_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroVmResources {
    pub cpu_cores: f64,
    pub memory_mb: u64,
    pub disk_mb: u64,
    pub network_mbps: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalMultiClusterConfig {
    pub max_regions: u32,
    pub max_mode_clusters_per_region: u32,
    pub max_agent_clusters_per_mode: u32,
    pub max_micro_vms_per_agent: u32,
    pub max_threads_per_vm: u32,
    pub auto_scaling_enabled: bool,
    pub cross_region_sync: bool,
    pub sync_interval_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterSyncStatus {
    pub last_sync: chrono::DateTime<chrono::Utc>,
    pub synced_regions: Vec<String>,
    pub pending_sync: Vec<String>,
    pub sync_errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RegionalStatus {
    Active,
    Degraded,
    Offline,
    Syncing,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModeClusterStatus {
    Provisioning,
    Running,
    Scaling,
    Draining,
    Down,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentClusterStatus {
    Active,
    Scaling,
    Throttled,
    Quarantined,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingPolicy {
    Static { replicas: u32 },
    AutoCpu { min: u32, max: u32, target_pct: f64 },
    AutoMemory { min: u32, max: u32, target_pct: f64 },
    Custom { rule: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossClusterMessage {
    pub id: Uuid,
    pub source_region: String,
    pub target_region: String,
    pub source_mode: ModeId,
    pub target_mode: ModeId,
    pub message_type: CrossClusterMsgType,
    pub payload: Vec<u8>,
    pub ttl_seconds: u64,
    pub priority: CrossClusterPriority,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrossClusterMsgType {
    StateSync,
    MemoryMigration,
    AgentHandoff,
    KillSignal,
    Heartbeat,
    ConfigUpdate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrossClusterPriority {
    Low,
    Normal,
    High,
    Critical,
}

impl MultiClusterSystem {
    pub fn new(config: GlobalMultiClusterConfig) -> Self {
        Self {
            regions: HashMap::new(),
            global_config: config,
            sync_status: ClusterSyncStatus {
                last_sync: chrono::Utc::now(),
                synced_regions: Vec::new(),
                pending_sync: Vec::new(),
                sync_errors: Vec::new(),
            },
        }
    }

    pub fn add_region(&mut self, name: &str, region: &str) -> RegionalCluster {
        let cluster = RegionalCluster {
            id: Uuid::new_v4(),
            name: name.to_string(),
            region: region.to_string(),
            mode_clusters: HashMap::new(),
            status: RegionalStatus::Active,
            latency_ms: 0,
            capacity_pct: 100.0,
        };
        let cloned = cluster.clone();
        self.regions.insert(name.to_string(), cluster);
        cloned
    }

    pub fn add_mode_to_region(
        &mut self,
        region_name: &str,
        mode_id: ModeId,
        kind: ModeKind,
    ) -> Result<ModeCluster, MultiClusterError> {
        let region = self
            .regions
            .get_mut(region_name)
            .ok_or(MultiClusterError::RegionNotFound(region_name.to_string()))?;

        if u32::try_from(region.mode_clusters.len()).unwrap_or(u32::MAX) >= self.global_config.max_mode_clusters_per_region {
            return Err(MultiClusterError::MaxModeClustersReached(
                region_name.to_string(),
            ));
        }

        let cluster = ModeCluster {
            id: Uuid::new_v4(),
            mode_id: mode_id.clone(),
            mode_kind: kind,
            agent_clusters: HashMap::new(),
            status: ModeClusterStatus::Provisioning,
            isolation_policy: format!("strict-{}", mode_id.0),
        };
        let cloned = cluster.clone();
        region.mode_clusters.insert(mode_id.0.clone(), cluster);
        Ok(cloned)
    }

    pub fn spawn_agent_cluster(
        &mut self,
        region_name: &str,
        mode_id: &ModeId,
        name: &str,
        scaling: ScalingPolicy,
    ) -> Result<AgentCluster, MultiClusterError> {
        let region = self
            .regions
            .get_mut(region_name)
            .ok_or(MultiClusterError::RegionNotFound(region_name.to_string()))?;
        let mode = region
            .mode_clusters
            .get_mut(&mode_id.0)
            .ok_or(MultiClusterError::ModeClusterNotFound(mode_id.clone()))?;

        if u32::try_from(mode.agent_clusters.len()).unwrap_or(u32::MAX) >= self.global_config.max_agent_clusters_per_mode {
            return Err(MultiClusterError::MaxAgentClustersReached(mode_id.clone()));
        }

        let cluster = AgentCluster {
            id: Uuid::new_v4(),
            name: name.to_string(),
            micro_vms: Vec::new(),
            status: AgentClusterStatus::Active,
            scaling_policy: scaling,
        };
        let cloned = cluster.clone();
        mode.agent_clusters.insert(name.to_string(), cluster);
        Ok(cloned)
    }

    pub fn spawn_micro_vm(
        &mut self,
        region_name: &str,
        mode_id: &ModeId,
        cluster_name: &str,
        agent_id: Uuid,
        tool_type: &str,
    ) -> Result<MicroVm, MultiClusterError> {
        let region = self
            .regions
            .get_mut(region_name)
            .ok_or(MultiClusterError::RegionNotFound(region_name.to_string()))?;
        let mode = region
            .mode_clusters
            .get_mut(&mode_id.0)
            .ok_or(MultiClusterError::ModeClusterNotFound(mode_id.clone()))?;
        let cluster = mode.agent_clusters.get_mut(cluster_name).ok_or(
            MultiClusterError::AgentClusterNotFound(cluster_name.to_string()),
        )?;

        let vm = MicroVm {
            id: Uuid::new_v4(),
            agent_id,
            tool_runtime: ToolRuntime {
                tool_type: tool_type.to_string(),
                sandbox_kind: "gvisor".into(),
                max_threads: self.global_config.max_threads_per_vm,
                memory_limit_mb: 512,
            },
            execution_threads: Vec::new(),
            resource_usage: MicroVmResources {
                cpu_cores: 1.0,
                memory_mb: 512,
                disk_mb: 100,
                network_mbps: 10,
            },
        };
        let cloned = vm.clone();
        cluster.micro_vms.push(vm);
        Ok(cloned)
    }

    pub fn spawn_execution_thread(
        &mut self,
        region_name: &str,
        mode_id: &ModeId,
        cluster_name: &str,
        vm_index: usize,
        task_id: &str,
    ) -> Result<ExecutionThread, MultiClusterError> {
        let region = self
            .regions
            .get_mut(region_name)
            .ok_or(MultiClusterError::RegionNotFound(region_name.to_string()))?;
        let mode = region
            .mode_clusters
            .get_mut(&mode_id.0)
            .ok_or(MultiClusterError::ModeClusterNotFound(mode_id.clone()))?;
        let cluster = mode.agent_clusters.get_mut(cluster_name).ok_or(
            MultiClusterError::AgentClusterNotFound(cluster_name.to_string()),
        )?;

        let vm = cluster
            .micro_vms
            .get_mut(vm_index)
            .ok_or(MultiClusterError::MicroVmNotFound(vm_index))?;

        let thread = ExecutionThread {
            id: Uuid::new_v4(),
            task_id: task_id.to_string(),
            cpu_usage_pct: 0.0,
            memory_usage_mb: 0,
            status: ThreadStatus::Idle,
            started_at: chrono::Utc::now(),
        };

        if u32::try_from(vm.execution_threads.len()).unwrap_or(u32::MAX) >= self.global_config.max_threads_per_vm {
            return Err(MultiClusterError::MaxThreadsReached(vm.id));
        }

        let cloned = thread.clone();
        vm.execution_threads.push(thread);
        Ok(cloned)
    }

    pub fn get_region(&self, name: &str) -> Option<&RegionalCluster> {
        self.regions.get(name)
    }

    pub fn get_region_mut(&mut self, name: &str) -> Option<&mut RegionalCluster> {
        self.regions.get_mut(name)
    }

    pub fn list_regions(&self) -> Vec<&RegionalCluster> {
        self.regions.values().collect()
    }

    /// Perform health check on all regions
    pub async fn health_check(&mut self) -> Result<HealthCheckReport, MultiClusterError> {
        let mut report = HealthCheckReport {
            timestamp: chrono::Utc::now(),
            healthy_regions: Vec::new(),
            degraded_regions: Vec::new(),
            unhealthy_regions: Vec::new(),
            total_regions: self.regions.len(),
        };

        for (name, region) in &mut self.regions {
            // Use the standalone health check function (no &self borrow needed)
            let is_healthy = Self::check_region_health(region);

            match is_healthy {
                HealthStatus::Healthy => {
                    region.status = RegionalStatus::Active;
                    report.healthy_regions.push(name.clone());
                }
                HealthStatus::Degraded => {
                    region.status = RegionalStatus::Degraded;
                    report.degraded_regions.push(name.clone());
                }
                HealthStatus::Unhealthy => {
                    region.status = RegionalStatus::Offline;
                    report.unhealthy_regions.push(name.clone());
                }
            }
        }

        Ok(report)
    }

    /// Check health of a single region (standalone function, no &self dependency)
    fn check_region_health(region: &RegionalCluster) -> HealthStatus {
        // In production, this would check:
        // - API endpoints
        // - Database connectivity
        // - Resource usage
        // - Agent cluster status

        // Simplified health check based on capacity and latency
        if region.capacity_pct < 10.0 {
            return HealthStatus::Unhealthy;
        }

        if region.capacity_pct < 50.0 || region.latency_ms > 500 {
            return HealthStatus::Degraded;
        }

        HealthStatus::Healthy
    }

    /// Select best region for new deployment based on load balancing
    pub fn select_region_for_deployment(&self, mode_id: &ModeId) -> Option<String> {
        let mut best_region = None;
        let mut best_score = f64::NEG_INFINITY;

        for (name, region) in &self.regions {
            if region.status != RegionalStatus::Active {
                continue;
            }

            // Check if mode exists in this region
            if !region.mode_clusters.contains_key(&mode_id.0) {
                continue;
            }

            // Calculate score based on capacity and latency
            let capacity_score = region.capacity_pct;
            let latency_score = 1.0 / (region.latency_ms as f64 + 1.0);
            let total_score = capacity_score * 0.7 + latency_score * 0.3;

            if total_score > best_score {
                best_score = total_score;
                best_region = Some(name.clone());
            }
        }

        best_region
    }

    /// Perform cross-region synchronization
    pub async fn sync_regions(&mut self) -> Result<SyncResult, MultiClusterError> {
        if !self.global_config.cross_region_sync {
            return Ok(SyncResult {
                synced_count: 0,
                failed_count: 0,
                duration_ms: 0,
            });
        }

        let start = std::time::Instant::now();
        let mut synced_count = 0;
        let mut failed_count = 0;

        let active_regions: Vec<String> = self.regions
            .iter()
            .filter(|(_, r)| r.status == RegionalStatus::Active)
            .map(|(name, _)| name.clone())
            .collect();

        for region_name in &active_regions {
            match self.sync_region(region_name).await {
                Ok(_) => {
                    synced_count += 1;
                    self.sync_status.synced_regions.push(region_name.clone());
                }
                Err(e) => {
                    failed_count += 1;
                    self.sync_status.sync_errors.push(format!("{}: {}", region_name, e));
                }
            }
        }

        self.sync_status.last_sync = chrono::Utc::now();
        self.sync_status.pending_sync.clear();

        let duration = start.elapsed();

        Ok(SyncResult {
            synced_count,
            failed_count,
            duration_ms: duration.as_millis() as u64,
        })
    }

    /// Sync a single region with others
    async fn sync_region(&mut self, region_name: &str) -> Result<(), MultiClusterError> {
        // In production, this would:
        // - Sync configuration
        // - Sync agent states
        // - Sync memory/knowledge base
        // - Sync model checkpoints

        debug!("Syncing region: {}", region_name);

        // Simulate sync delay
        sleep(Duration::from_millis(100)).await;

        Ok(())
    }

    /// Start background orchestration tasks
    pub async fn start_orchestration(&mut self) -> Result<(), MultiClusterError> {
        let config = self.global_config.clone();

        // Spawn health check task
        let health_check_interval = Duration::from_secs(30);
        tokio::spawn(async move {
            let mut interval = interval(health_check_interval);
            loop {
                interval.tick().await;
                // Health check logic would go here
                debug!("Running periodic health check");
            }
        });

        // Spawn sync task if enabled
        if config.cross_region_sync {
            let sync_interval = Duration::from_secs(config.sync_interval_seconds);
            tokio::spawn(async move {
                let mut interval = interval(sync_interval);
                loop {
                    interval.tick().await;
                    // Sync logic would go here
                    debug!("Running periodic cross-region sync");
                }
            });
        }

        info!("Multi-cluster orchestration started");
        Ok(())
    }

    /// Handle region failure with automatic failover
    pub async fn handle_region_failure(&mut self, failed_region: &str) -> Result<FailoverResult, MultiClusterError> {
        warn!("Region failure detected: {}", failed_region);

        let region = self.regions.get_mut(failed_region)
            .ok_or(MultiClusterError::RegionNotFound(failed_region.to_string()))?;

        region.status = RegionalStatus::Offline;

        // Find alternative regions for failover
        let alternatives: Vec<String> = self.regions
            .iter()
            .filter(|(name, r)| *name != failed_region && r.status == RegionalStatus::Active)
            .map(|(name, _)| name.clone())
            .collect();

        if alternatives.is_empty() {
            error!("No alternative regions available for failover");
            return Ok(FailoverResult {
                success: false,
                target_region: None,
                message: "No alternative regions available".to_string(),
            });
        }

        // Select best alternative (simple round-robin for now)
        let target_region = alternatives[0].clone();

        info!("Initiating failover to region: {}", target_region);

        // Simulate failover
        sleep(Duration::from_millis(500)).await;

        Ok(FailoverResult {
            success: true,
            target_region: Some(target_region),
            message: "Failover completed successfully".to_string(),
        })
    }
}

/// Health status for regions
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Health check report
#[derive(Debug, Clone)]
pub struct HealthCheckReport {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub healthy_regions: Vec<String>,
    pub degraded_regions: Vec<String>,
    pub unhealthy_regions: Vec<String>,
    pub total_regions: usize,
}

/// Sync result
#[derive(Debug, Clone)]
pub struct SyncResult {
    pub synced_count: usize,
    pub failed_count: usize,
    pub duration_ms: u64,
}

/// Failover result
#[derive(Debug, Clone)]
pub struct FailoverResult {
    pub success: bool,
    pub target_region: Option<String>,
    pub message: String,
}

impl Default for GlobalMultiClusterConfig {
    fn default() -> Self {
        Self {
            max_regions: 10,
            max_mode_clusters_per_region: 100,
            max_agent_clusters_per_mode: 1000,
            max_micro_vms_per_agent: 5,
            max_threads_per_vm: 10,
            auto_scaling_enabled: true,
            cross_region_sync: true,
            sync_interval_seconds: 60,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ThreadStatus {
    Active,
    Idle,
    Blocked,
    Completed,
    Failed(String),
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum MultiClusterError {
    #[error("Region not found: {0}")]
    RegionNotFound(String),
    #[error("Mode cluster not found: {0}")]
    ModeClusterNotFound(ModeId),
    #[error("Agent cluster not found: {0}")]
    AgentClusterNotFound(String),
    #[error("MicroVM not found at index: {0}")]
    MicroVmNotFound(usize),
    #[error("Max mode clusters reached for region: {0}")]
    MaxModeClustersReached(String),
    #[error("Max agent clusters reached for mode: {0}")]
    MaxAgentClustersReached(ModeId),
    #[error("Max threads reached for MicroVM: {0}")]
    MaxThreadsReached(Uuid),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_cluster_hierarchy() {
        let config = GlobalMultiClusterConfig::default();
        let mut system = MultiClusterSystem::new(config);

        system.add_region("us-east", "us-east-1");
        system.add_region("eu-west", "eu-west-1");

        let mode_id = ModeId::new("research");
        system
            .add_mode_to_region("us-east", mode_id.clone(), ModeKind::Research)
            .unwrap();

        let cluster = system
            .spawn_agent_cluster(
                "us-east",
                &mode_id,
                "oracle-group",
                ScalingPolicy::Static { replicas: 3 },
            )
            .unwrap();
        assert_eq!(cluster.name, "oracle-group");

        let vm = system
            .spawn_micro_vm(
                "us-east",
                &mode_id,
                "oracle-group",
                Uuid::new_v4(),
                "python",
            )
            .unwrap();
        assert_eq!(vm.tool_runtime.tool_type, "python");

        let thread = system
            .spawn_execution_thread("us-east", &mode_id, "oracle-group", 0, "task-1")
            .unwrap();
        assert_eq!(thread.task_id, "task-1");
    }

    #[test]
    fn test_scaling_policy() {
        let auto = ScalingPolicy::AutoCpu {
            min: 2,
            max: 10,
            target_pct: 70.0,
        };
        match auto {
            ScalingPolicy::AutoCpu { min, max, .. } => {
                assert_eq!(min, 2);
                assert_eq!(max, 10);
            }
            other => panic!("Expected AutoCpu variant, got {:?}", other),
        }
    }
}
