use uuid::Uuid;

use nexora_isolation::config::IsolationConfig;
use nexora_isolation::firewall::FirewallAction;
use nexora_isolation::killswitch::{KillTarget, KillTrigger, KillStatus};
use nexora_isolation::multicluster::ScalingPolicy;
use nexora_isolation::layer0_global::ClusterStatus;
use nexora_isolation::layer1_mode::{ModeId, ModeKind, ModeStatus};
use nexora_isolation::layer2_agent::{AgentRuntimeSpec, AgentType, PodStatus};
use nexora_isolation::layer3_tool::{SandboxSpec, ToolKind, ToolStatus};
use nexora_isolation::layer4_runtime::RuntimeIsolationSpec;
use nexora_isolation::layer6_permission::{AgentRole, Capability};
use nexora_isolation::multicluster::{GlobalMultiClusterConfig, MultiClusterSystem};
use nexora_isolation::quarantine::{QuarantineReason, QuarantineSeverity, QuarantineStatus};
use nexora_isolation::IsolationOrchestrator;

#[test]
fn test_full_isolation_pipeline() {
    let config = IsolationConfig::default();
    let orch = IsolationOrchestrator::new(config);

    let agent_id = Uuid::new_v4();

    // L0: Global isolation — context routing
    {
        let global = orch.global.read();
        assert_eq!(global.name, "nexora-cluster");
        assert!(global.api_gateway.enabled);
        assert!(global.security_core.enabled);
        assert_eq!(global.health.status, ClusterStatus::Initializing);
    }

    // L1: Mode isolation — create, activate, enforce mode isolation
    {
        let mut mode = orch.mode.write();
        let research = ModeId::new("research");
        let defense = ModeId::new("defense");
        mode.create_mode(research.clone(), ModeKind::Research);
        mode.create_mode(defense.clone(), ModeKind::Defense);
        mode.activate_mode(&research).unwrap();

        let m = mode.get_mode(&research).unwrap();
        assert_eq!(m.status, ModeStatus::Active);
        assert!(!mode.can_communicate(&research, &defense));
    }

    // L2: Agent isolation — spawn pod in mode, verify isolation properties
    {
        let mut agent = orch.agent.write();
        let mode_id = ModeId::new("research");
        let group = agent.create_group(mode_id, "oracle-group");
        let pod = agent
            .spawn_pod(
                group.id,
                "oracle",
                AgentType::Oracle,
                AgentRuntimeSpec {
                    cpu_limit: 2.0,
                    memory_limit_mb: 4096,
                    network_access: false,
                    filesystem_access: false,
                    execution_timeout_seconds: 300,
                },
            )
            .unwrap();
        assert_eq!(pod.status, PodStatus::Pending);
        assert_eq!(pod.mode_id, ModeId::new("research"));
        assert!(pod.isolation_label.contains("oracle"));
    }

    // L3: Tool isolation — register tool, set access control, verify restriction
    {
        let mut tool = orch.tool.write();
        let pod = tool.register_tool(
            ToolKind::Python,
            SandboxSpec::default_tool(),
            false,
            false,
        );
        assert_eq!(pod.status, ToolStatus::Idle);

        tool.set_tool_allowed_commands(&ToolKind::Python, vec!["run".into()])
            .unwrap();

        let log = tool.get_audit_log(&ToolKind::Python);
        assert_eq!(log.len(), 0);
    }

    // L4: Runtime constraints — validate spec, reject privileged containers
    {
        let runtime = RuntimeIsolationSpec::gvisor_default();
        let violations = runtime.validate();
        assert!(violations.is_empty());

        let mut bad = RuntimeIsolationSpec::gvisor_default();
        bad.container.privileged = true;
        let v = bad.validate();
        assert!(v.iter().any(|s| s.contains("Privileged")));
    }

    // L5: Cognitive isolation — memory regions are isolated between agents
    {
        let mut cognitive = orch.cognitive.write();
        let region_id = cognitive.create_memory_region(agent_id);
        cognitive
            .write_memory(region_id, agent_id, "secret", b"sensitive data".to_vec())
            .unwrap();
        let read = cognitive
            .read_memory(region_id, agent_id, "secret")
            .unwrap();
        assert_eq!(read, Some(b"sensitive data".to_vec()));

        let other = Uuid::new_v4();
        let result = cognitive.read_memory(region_id, other, "secret");
        assert!(result.is_err());
    }

    // L6: Permission enforcement — agent needs capability to pass check
    {
        let mut perm = orch.permission.write();
        perm.register_agent(agent_id, "oracle", AgentRole::Specialist);
        perm.grant_capability(agent_id, Capability::ModelInference("default".into()))
            .unwrap();
        assert!(perm.check_capability(agent_id, &Capability::ModelInference("default".into())));
        assert!(!perm.check_capability(agent_id, &Capability::ShellAccess));
    }

    // Full pipeline: pre_inference_check passes with proper setup
    assert!(orch.pre_inference_check(agent_id).is_ok());

    // Revoke capability — pipeline now blocks
    orch.permission
        .write()
        .revoke_capability(agent_id, &Capability::ModelInference("default".into()))
        .unwrap();
    assert!(orch.pre_inference_check(agent_id).is_err());
}

#[test]
fn test_kill_switch_blocks_requests() {
    let config = IsolationConfig::default();
    let orch = IsolationOrchestrator::new(config);
    let agent_id = Uuid::new_v4();

    {
        let mut perm = orch.permission.write();
        perm.register_agent(agent_id, "target", AgentRole::Restricted);
        perm.grant_capability(agent_id, Capability::ModelInference("default".into()))
            .unwrap();
    }
    assert!(orch.pre_inference_check(agent_id).is_ok());

    // Trigger kill switch on the agent
    let event = orch
        .trigger_kill_switch(
            KillTarget::Agent(agent_id),
            "security violation: anomalous memory access pattern",
            KillTrigger::AutoQuarantine {
                anomaly_score: 0.95,
            },
        )
        .unwrap();
    assert_eq!(event.target, KillTarget::Agent(agent_id));

    // Block agent in the firewall as kill-switch consequence
    orch.firewall.write().block_agent(agent_id);

    // Verify kill event is recorded in history
    let ks_guard = orch.kill_switch.read();
    let recent = ks_guard.get_recent_kills(5);
    let matching: Vec<_> = recent
        .iter()
        .filter(|e| matches!(&e.target, KillTarget::Agent(id) if *id == agent_id))
        .collect();
    assert!(!matching.is_empty());
    assert!(matches!(
        &matching[0].triggered_by,
        KillTrigger::AutoQuarantine { .. }
    ));

    // Verify agent communication is now denied (firewall blocks)
    let result = orch.check_agent_communication(
        agent_id,
        "agent:target",
        Uuid::new_v4(),
        "agent:other",
        "query",
        b"hello",
    );
    assert!(result.is_err());

    // Complete the kill
    let ks = &mut *orch.kill_switch.write();
    ks.complete_kill(event.id, vec![agent_id], 42);
    let completed = ks.get_recent_kills(1);
    assert!(matches!(completed[0].status, KillStatus::Completed));
}

#[test]
fn test_quarantine_malicious_pattern() {
    let config = IsolationConfig::default();
    let orch = IsolationOrchestrator::new(config);

    let src = Uuid::new_v4();
    let dst = Uuid::new_v4();

    // Firewall detects suspicious pattern in payload
    let action = orch.firewall.write().evaluate(
        src,
        "agent:suspicious",
        dst,
        "agent:victim",
        "query",
        b"overwrite memory now",
    );
    assert!(matches!(action, FirewallAction::Deny));

    // Quarantine the source agent through the quarantine manager
    {
        let mut q = orch.quarantine.write();
        let quarantine = q.quarantine_agent(
            src,
            QuarantineReason::SecurityViolation {
                capability: "memory".into(),
                action: "attempted memory overwrite via inter-agent bus".into(),
            },
            QuarantineSeverity::High,
            vec![],
        );
        assert_eq!(quarantine.status, QuarantineStatus::Active);
        assert!(q.is_agent_quarantined(src));
    }

    // Verify pre_inference_check blocks quarantined agents
    {
        let mut perm = orch.permission.write();
        perm.register_agent(src, "suspicious", AgentRole::Restricted);
        perm.grant_capability(src, Capability::ModelInference("default".into()))
            .unwrap();
    }
    assert!(orch.pre_inference_check(src).is_err());

    // Resolve quarantine — agent can pass checks again
    {
        let q_lock = &mut *orch.quarantine.write();
        let active_ids: Vec<_> = q_lock.get_active_for_agent(src).iter().map(|q| q.id).collect();
        for id in active_ids {
            q_lock.resolve_quarantine(id).unwrap();
        }
        assert!(!q_lock.is_agent_quarantined(src));
    }

    // Verify another unrelated agent is unaffected
    let clean = Uuid::new_v4();
    {
        let mut perm = orch.permission.write();
        perm.register_agent(clean, "clean", AgentRole::Restricted);
        perm.grant_capability(clean, Capability::ModelInference("default".into()))
            .unwrap();
    }
    assert!(orch.pre_inference_check(clean).is_ok());
}

#[test]
fn test_multi_cluster_isolation() {
    let mode_a = ModeId::new("research-alpha");
    let mode_b = ModeId::new("research-beta");

    // Cluster A: us-east, research-alpha, cross-region sync OFF
    let mut cluster_a = MultiClusterSystem::new(GlobalMultiClusterConfig {
        max_regions: 10,
        max_mode_clusters_per_region: 100,
        max_agent_clusters_per_mode: 1000,
        max_micro_vms_per_agent: 5,
        max_threads_per_vm: 10,
        auto_scaling_enabled: true,
        cross_region_sync: false,
        sync_interval_seconds: 60,
    });
    cluster_a.add_region("us-east", "us-east-1");
    cluster_a
        .add_mode_to_region("us-east", mode_a.clone(), ModeKind::Research)
        .unwrap();
    cluster_a
        .spawn_agent_cluster(
            "us-east",
            &mode_a,
            "oracle-group",
            ScalingPolicy::Static { replicas: 3 },
        )
        .unwrap();

    // Cluster B: eu-west, research-beta, cross-region sync ON
    let mut cluster_b = MultiClusterSystem::new(GlobalMultiClusterConfig::default());
    cluster_b.add_region("eu-west", "eu-west-1");
    cluster_b
        .add_mode_to_region("eu-west", mode_b.clone(), ModeKind::Research)
        .unwrap();
    cluster_b
        .spawn_agent_cluster(
            "eu-west",
            &mode_b,
            "defense-group",
            ScalingPolicy::AutoCpu {
                min: 2,
                max: 10,
                target_pct: 70.0,
            },
        )
        .unwrap();

    // Cluster A cannot see cluster B's regions
    assert!(cluster_a.get_region("eu-west").is_none());
    assert!(cluster_a.get_region("us-east").is_some());

    // Cluster B cannot see cluster A's regions
    assert!(cluster_b.get_region("us-east").is_none());
    assert!(cluster_b.get_region("eu-west").is_some());

    // Cluster A's mode cluster is independent
    let region_a = cluster_a.get_region("us-east").unwrap();
    assert!(region_a.mode_clusters.contains_key("research-alpha"));
    assert!(!region_a.mode_clusters.contains_key("research-beta"));

    // Cluster B's mode cluster is independent
    let region_b = cluster_b.get_region("eu-west").unwrap();
    assert!(region_b.mode_clusters.contains_key("research-beta"));
    assert!(!region_b.mode_clusters.contains_key("research-alpha"));

    // Cluster configs are independent
    assert!(!cluster_a.global_config.cross_region_sync);
    assert!(cluster_b.global_config.cross_region_sync);

    // Deploying to one cluster doesn't affect the other
    let a_regions_before = cluster_a.list_regions().len();
    let b_regions_before = cluster_b.list_regions().len();

    cluster_a.add_region("ap-southeast", "ap-southeast-1");

    assert_eq!(cluster_a.list_regions().len(), a_regions_before + 1);
    assert_eq!(cluster_b.list_regions().len(), b_regions_before);
}

#[test]
fn test_isolated_orchestrators_dont_interfere() {
    let config_a = IsolationConfig::default();
    let config_b = IsolationConfig::default();
    let orch_a = IsolationOrchestrator::new(config_a);
    let orch_b = IsolationOrchestrator::new(config_b);

    let agent_a = Uuid::new_v4();
    let agent_b = Uuid::new_v4();

    // Register agent_a in orch_a with inference capability
    {
        let mut perm = orch_a.permission.write();
        perm.register_agent(agent_a, "agent-a", AgentRole::Specialist);
        perm.grant_capability(agent_a, Capability::ModelInference("default".into()))
            .unwrap();
    }

    // Register agent_b in orch_b with inference capability
    {
        let mut perm = orch_b.permission.write();
        perm.register_agent(agent_b, "agent-b", AgentRole::Specialist);
        perm.grant_capability(agent_b, Capability::ModelInference("default".into()))
            .unwrap();
    }

    // Each orchestrator's agent passes their own check
    assert!(orch_a.pre_inference_check(agent_a).is_ok());
    assert!(orch_b.pre_inference_check(agent_b).is_ok());

    // agent_a is not known to orch_b
    assert!(orch_b.pre_inference_check(agent_a).is_err());

    // agent_b is not known to orch_a
    assert!(orch_a.pre_inference_check(agent_b).is_err());

    // Memory isolation across orchestrators
    let mem_a = orch_a.cognitive.write().create_memory_region(agent_a);
    orch_a
        .cognitive
        .write()
        .write_memory(mem_a, agent_a, "key", b"orchestrator-a-data".to_vec())
        .unwrap();

    let read = orch_a
        .cognitive
        .write()
        .read_memory(mem_a, agent_a, "key")
        .unwrap();
    assert_eq!(read, Some(b"orchestrator-a-data".to_vec()));

    // orch_b has no regions at all
    assert!(orch_b
        .cognitive
        .write()
        .read_memory(mem_a, agent_b, "key")
        .is_err());
}

#[test]
fn test_kill_switch_protected_mode_rejected() {
    let config = IsolationConfig::default();
    let orch = IsolationOrchestrator::new(config);

    let system_mode = ModeId::new("system");
    let result = orch.trigger_kill_switch(
        KillTarget::Mode(system_mode),
        "attempted kill on protected mode",
        KillTrigger::Manual {
            user: "attacker".into(),
        },
    );

    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Kill switch error"));
}
