//! NXR-NEXUM Agents Module
//!
//! Core agent implementations for multi-agent orchestration.
//! Provides 6 specialized agents: SWARM-CTL, DELEGATE-X, CONSENSUS-AI,
//! TASK-ROUTER, MERGE-SYNTH, and PRIORITY-GOD.

use std::collections::HashMap;
use std::time::Instant;

pub mod orchestrator_prime;
pub mod consensus_builder;
pub mod alignment_arbiter;
pub mod resource_optimizer;

pub use orchestrator_prime::*;
pub use consensus_builder::*;
pub use alignment_arbiter::*;
pub use resource_optimizer::*;

// ---------------------------------------------------------------------------
// Agent type identifiers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum AgentKind {
    SwarmCtl,
    DelegateX,
    ConsensusAi,
    TaskRouter,
    MergeSynth,
    PriorityGod,
}

impl AgentKind {
    pub fn name(&self) -> &'static str {
        match self {
            AgentKind::SwarmCtl => "SWARM-CTL",
            AgentKind::DelegateX => "DELEGATE-X",
            AgentKind::ConsensusAi => "CONSENSUS-AI",
            AgentKind::TaskRouter => "TASK-ROUTER",
            AgentKind::MergeSynth => "MERGE-SYNTH",
            AgentKind::PriorityGod => "PRIORITY-GOD",
        }
    }
}

// ---------------------------------------------------------------------------
// 1. SWARM-CTL — Swarm Intelligence Controller
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SwarmCtlAgent {
    pub id: String,
    pub max_parallel_agents: u32,
    pub active_agents: u32,
    pub coordination_overhead_ms: f64,
    pub emergence_threshold: f32,
}

impl Default for SwarmCtlAgent {
    fn default() -> Self {
        Self {
            id: "swarm-ctl-01".into(),
            max_parallel_agents: 256,
            active_agents: 0,
            coordination_overhead_ms: 0.8,
            emergence_threshold: 0.75,
        }
    }
}

impl SwarmCtlAgent {
    pub fn kind(&self) -> AgentKind {
        AgentKind::SwarmCtl
    }

    pub fn spawn_swarm(&self, count: u32) -> SwarmDeployment {
        let actual = count.min(self.max_parallel_agents);
        SwarmDeployment {
            agent_id: self.id.clone(),
            total_spawned: actual,
            partitions: ((actual as f64).sqrt().ceil() as u32).max(1),
            overhead_ns: (self.coordination_overhead_ms * 1_000_000.0) as u64,
        }
    }

    pub fn adjust_emergence(&mut self, signal: f32) {
        self.emergence_threshold = (self.emergence_threshold + signal * 0.1).clamp(0.0, 1.0);
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SwarmDeployment {
    pub agent_id: String,
    pub total_spawned: u32,
    pub partitions: u32,
    pub overhead_ns: u64,
}

// ---------------------------------------------------------------------------
// 2. DELEGATE-X — Capability-based Delegation
// ---------------------------------------------------------------------------

pub type CapabilityProfile = HashMap<String, f32>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DelegateXAgent {
    pub id: String,
    pub registry: HashMap<String, CapabilityProfile>,
    pub confidence_threshold: f32,
    pub delegation_history: Vec<DelegationRecord>,
}

impl Default for DelegateXAgent {
    fn default() -> Self {
        Self {
            id: "delegate-x-01".into(),
            registry: HashMap::new(),
            confidence_threshold: 0.6,
            delegation_history: Vec::new(),
        }
    }
}

impl DelegateXAgent {
    pub fn kind(&self) -> AgentKind {
        AgentKind::DelegateX
    }

    pub fn register_agent(&mut self, name: &str, capabilities: CapabilityProfile) {
        self.registry.insert(name.to_string(), capabilities);
    }

    pub fn find_best_match(&self, task_caps: &CapabilityProfile) -> Option<(String, f32)> {
        self.registry
            .iter()
            .filter_map(|(name, profile)| {
                let score = capability_similarity(profile, task_caps);
                if score >= self.confidence_threshold {
                    Some((name.clone(), score))
                } else {
                    None
                }
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    pub fn delegate(&mut self, task: &str, task_caps: CapabilityProfile) -> Option<String> {
        let (agent, confidence) = self.find_best_match(&task_caps)?;
        self.delegation_history.push(DelegationRecord {
            task: task.to_string(),
            delegated_to: agent.clone(),
            confidence,
            timestamp: chrono::Utc::now(),
        });
        Some(agent)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DelegationRecord {
    pub task: String,
    pub delegated_to: String,
    pub confidence: f32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

fn capability_similarity(a: &CapabilityProfile, b: &CapabilityProfile) -> f32 {
    let mut num: f32 = 0.0;
    let mut den_a: f32 = 0.0;
    let mut den_b: f32 = 0.0;
    for (key, av) in a {
        if let Some(bv) = b.get(key) {
            num += av * bv;
        }
        den_a += av * av;
    }
    for bv in b.values() {
        den_b += bv * bv;
    }
    let den = den_a.sqrt() * den_b.sqrt();
    if den == 0.0 { 0.0 } else { num / den }
}

// ---------------------------------------------------------------------------
// 3. CONSENSUS-AI — Consensus Builder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConsensusAiAgent {
    pub id: String,
    pub min_agreement: f32,
    pub max_rounds: u32,
    pub weight_by_confidence: bool,
}

impl Default for ConsensusAiAgent {
    fn default() -> Self {
        Self {
            id: "consensus-ai-01".into(),
            min_agreement: 0.7,
            max_rounds: 5,
            weight_by_confidence: true,
        }
    }
}

impl ConsensusAiAgent {
    pub fn kind(&self) -> AgentKind {
        AgentKind::ConsensusAi
    }

    pub fn evaluate(&self, opinions: &[AgentOpinion]) -> ConsensusVerdict {
        let total_weight: f32 = if self.weight_by_confidence {
            opinions.iter().map(|o| o.confidence).sum::<f32>().max(1e-8)
        } else {
            opinions.len() as f32
        };

        let mut counts: HashMap<String, (f32, Vec<&AgentOpinion>)> = HashMap::new();
        for opinion in opinions {
            let entry = counts.entry(opinion.position.clone()).or_default();
            entry.0 += if self.weight_by_confidence {
                opinion.confidence
            } else {
                1.0
            };
            entry.1.push(opinion);
        }

        let mut best: Option<(String, f32, Vec<&AgentOpinion>)> = None;
        for (position, (weight, supporters)) in counts {
            let share = weight / total_weight;
            if share >= best.as_ref().map_or(0.0, |b: &(String, f32, Vec<&AgentOpinion>)| b.1) {
                best = Some((position.clone(), share, supporters));
            }
        }

        match best {
            Some((position, agreement, supporters)) => ConsensusVerdict {
                position,
                agreement_level: agreement,
                consensus_reached: agreement >= self.min_agreement,
                rounds_used: 1,
                supporters: supporters.into_iter().map(|o| o.agent_id.clone()).collect(),
            },
            None => ConsensusVerdict {
                position: "no-clear-majority".into(),
                agreement_level: 0.0,
                consensus_reached: false,
                rounds_used: 1,
                supporters: vec![],
            },
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentOpinion {
    pub agent_id: String,
    pub position: String,
    pub confidence: f32,
    pub reasoning: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConsensusVerdict {
    pub position: String,
    pub agreement_level: f32,
    pub consensus_reached: bool,
    pub rounds_used: u32,
    pub supporters: Vec<String>,
}

// ---------------------------------------------------------------------------
// 4. TASK-ROUTER — Optimal Pipeline Router
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaskPipeline {
    pub name: String,
    pub tags: Vec<String>,
    pub estimated_latency_ms: u64,
    pub throughput: f32,
    pub is_active: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaskRouterAgent {
    pub id: String,
    pub pipelines: Vec<TaskPipeline>,
    pub routing_history: Vec<RoutedTask>,
}

impl Default for TaskRouterAgent {
    fn default() -> Self {
        Self {
            id: "task-router-01".into(),
            pipelines: Vec::new(),
            routing_history: Vec::new(),
        }
    }
}

impl TaskRouterAgent {
    pub fn kind(&self) -> AgentKind {
        AgentKind::TaskRouter
    }

    pub fn register_pipeline(&mut self, pipeline: TaskPipeline) {
        self.pipelines.push(pipeline);
    }

    pub fn route(&mut self, task: &str, tags: &[String]) -> Option<&TaskPipeline> {
        let best = self
            .pipelines
            .iter()
            .filter(|p| p.is_active && tags.iter().any(|t| p.tags.contains(t)))
            .min_by_key(|p| p.estimated_latency_ms);

        if let Some(pipeline) = best {
            self.routing_history.push(RoutedTask {
                task: task.to_string(),
                pipeline: pipeline.name.clone(),
                routed_at: chrono::Utc::now(),
            });
        }
        best
    }

    pub fn route_with_throughput(&mut self, task: &str, tags: &[String]) -> Option<&TaskPipeline> {
        let best = self
            .pipelines
            .iter()
            .filter(|p| p.is_active && tags.iter().any(|t| p.tags.contains(t)))
            .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap_or(std::cmp::Ordering::Equal));

        if let Some(pipeline) = best {
            self.routing_history.push(RoutedTask {
                task: task.to_string(),
                pipeline: pipeline.name.clone(),
                routed_at: chrono::Utc::now(),
            });
        }
        best
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RoutedTask {
    pub task: String,
    pub pipeline: String,
    pub routed_at: chrono::DateTime<chrono::Utc>,
}

// ---------------------------------------------------------------------------
// 5. MERGE-SYNTH — Parallel Execution Merger
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MergeSynthAgent {
    pub id: String,
    pub strategy: MergeStrategy,
    #[serde(skip)]
    pub conflict_resolution: ConflictMode,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum MergeStrategy {
    FirstWin,
    WeightedAverage,
    LatestTimestamp,
    MajorityVote,
    Custom(String),
}

impl Default for MergeStrategy {
    fn default() -> Self {
        MergeStrategy::WeightedAverage
    }
}

impl Default for MergeSynthAgent {
    fn default() -> Self {
        Self {
            id: String::default(),
            strategy: MergeStrategy::default(),
            conflict_resolution: ConflictMode::Discard,
        }
    }
}

pub enum ConflictMode {
    Discard,
    KeepAll,
    ResolveWith(fn(&[MergeInput]) -> MergeOutput),
}

impl std::fmt::Debug for ConflictMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConflictMode::Discard => write!(f, "Discard"),
            ConflictMode::KeepAll => write!(f, "KeepAll"),
            ConflictMode::ResolveWith(_) => write!(f, "ResolveWith(<fn>)"),
        }
    }
}

impl Clone for ConflictMode {
    fn clone(&self) -> Self {
        match self {
            ConflictMode::Discard => ConflictMode::Discard,
            ConflictMode::KeepAll => ConflictMode::KeepAll,
            ConflictMode::ResolveWith(_) => ConflictMode::KeepAll,
        }
    }
}

impl Default for ConflictMode {
    fn default() -> Self {
        ConflictMode::KeepAll
    }
}

impl MergeSynthAgent {
    pub fn kind(&self) -> AgentKind {
        AgentKind::MergeSynth
    }

    pub fn merge(&self, inputs: &[MergeInput]) -> MergeOutput {
        match self.strategy {
            MergeStrategy::FirstWin => inputs
                .first()
                .map(|i| MergeOutput {
                    result: i.data.clone(),
                    source: i.source.clone(),
                    confidence: i.confidence,
                    merged_count: 1,
                })
                .unwrap_or_default(),
            MergeStrategy::WeightedAverage => {
                let total: f32 = inputs.iter().map(|i| i.confidence).sum::<f32>().max(1e-8);
                let mut merged = String::new();
                let mut max_conf = 0.0f32;
                let mut best_src = String::new();
                for input in inputs {
                    let weight = input.confidence / total;
                    if !merged.is_empty() {
                        merged.push('\n');
                    }
                    merged.push_str(&format!("[w={:.2}] {}", weight, input.data));
                    if input.confidence > max_conf {
                        max_conf = input.confidence;
                        best_src = input.source.clone();
                    }
                }
                MergeOutput {
                    result: merged,
                    source: best_src,
                    confidence: max_conf,
                    merged_count: inputs.len() as u32,
                }
            }
            MergeStrategy::LatestTimestamp => inputs
                .iter()
                .max_by_key(|i| i.timestamp)
                .map(|i| MergeOutput {
                    result: i.data.clone(),
                    source: i.source.clone(),
                    confidence: i.confidence,
                    merged_count: 1,
                })
                .unwrap_or_default(),
            MergeStrategy::MajorityVote => {
                let mut counts: HashMap<&str, u32> = HashMap::new();
                for input in inputs {
                    *counts.entry(&input.data).or_default() += 1;
                }
                let best = counts
                    .into_iter()
                    .max_by_key(|&(_, c)| c)
                    .and_then(|(data, _)| inputs.iter().find(|i| i.data == data));
                match best {
                    Some(i) => MergeOutput {
                        result: i.data.clone(),
                        source: i.source.clone(),
                        confidence: i.confidence,
                        merged_count: inputs.len() as u32,
                    },
                    None => MergeOutput::default(),
                }
            }
            MergeStrategy::Custom(_) => inputs
                .first()
                .map(|i| MergeOutput {
                    result: i.data.clone(),
                    source: i.source.clone(),
                    confidence: i.confidence,
                    merged_count: 1,
                })
                .unwrap_or_default(),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MergeInput {
    pub source: String,
    pub data: String,
    pub confidence: f32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MergeOutput {
    pub result: String,
    pub source: String,
    pub confidence: f32,
    pub merged_count: u32,
}

impl Default for MergeOutput {
    fn default() -> Self {
        Self {
            result: String::new(),
            source: String::new(),
            confidence: 0.0,
            merged_count: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// 6. PRIORITY-GOD — Absolute Priority Manager
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PriorityGodAgent {
    pub id: String,
    pub base_priority: PriorityLevel,
    pub preemption_enabled: bool,
    pub queue: Vec<PrioritizedTask>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum PriorityLevel {
    Critical = 0,
    High = 1,
    Normal = 2,
    Low = 3,
    Background = 4,
}

impl PriorityLevel {
    pub fn label(&self) -> &'static str {
        match self {
            PriorityLevel::Critical => "CRITICAL",
            PriorityLevel::High => "HIGH",
            PriorityLevel::Normal => "NORMAL",
            PriorityLevel::Low => "LOW",
            PriorityLevel::Background => "BACKGROUND",
        }
    }
}

impl Default for PriorityGodAgent {
    fn default() -> Self {
        Self {
            id: "priority-god-01".into(),
            base_priority: PriorityLevel::Normal,
            preemption_enabled: true,
            queue: Vec::new(),
        }
    }
}

impl PriorityGodAgent {
    pub fn kind(&self) -> AgentKind {
        AgentKind::PriorityGod
    }

    pub fn enqueue(&mut self, task: PrioritizedTask) {
        self.queue.push(task);
    }

    pub fn dequeue(&mut self) -> Option<PrioritizedTask> {
        if self.queue.is_empty() {
            return None;
        }
        let mut best_idx = 0;
        let mut best = self.queue[0].clone();
        for (i, t) in self.queue.iter().enumerate().skip(1) {
            if t.effective_priority() < best.effective_priority() {
                best = t.clone();
                best_idx = i;
            }
        }
        Some(self.queue.swap_remove(best_idx))
    }

    pub fn peek(&self) -> Option<&PrioritizedTask> {
        self.queue
            .iter()
            .min_by_key(|t| t.effective_priority())
    }

    pub fn preempt(&mut self, threshold: PriorityLevel) -> Vec<PrioritizedTask> {
        let mut preempted = Vec::new();
        self.queue.retain(|t| {
            if t.effective_priority() > threshold as isize {
                preempted.push(t.clone());
                false
            } else {
                true
            }
        });
        preempted
    }

    pub fn demote(&mut self, task_id: &str, new_priority: PriorityLevel) -> bool {
        if let Some(task) = self.queue.iter_mut().find(|t| t.id == task_id) {
            task.overridden_priority = Some(new_priority);
            true
        } else {
            false
        }
    }

    pub fn promote(&mut self, task_id: &str, new_priority: PriorityLevel) -> bool {
        self.demote(task_id, new_priority)
    }

    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PrioritizedTask {
    pub id: String,
    pub name: String,
    pub base_priority: PriorityLevel,
    pub overridden_priority: Option<PriorityLevel>,
    pub submitted_at: chrono::DateTime<chrono::Utc>,
    pub deadline: Option<chrono::DateTime<chrono::Utc>>,
    pub tags: Vec<String>,
}

impl PrioritizedTask {
    pub fn effective_priority(&self) -> isize {
        let p = self.overridden_priority.unwrap_or(self.base_priority);
        p as isize
    }

    pub fn priority_label(&self) -> &'static str {
        let p = self.overridden_priority.unwrap_or(self.base_priority);
        p.label()
    }
}

// ---------------------------------------------------------------------------
// NexumAgents — Top-level container for all 6 agents
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NexumAgents {
    pub swarm_ctl: SwarmCtlAgent,
    pub delegate_x: DelegateXAgent,
    pub consensus_ai: ConsensusAiAgent,
    pub task_router: TaskRouterAgent,
    pub merge_synth: MergeSynthAgent,
    pub priority_god: PriorityGodAgent,
}

impl Default for NexumAgents {
    fn default() -> Self {
        Self::new()
    }
}

impl NexumAgents {
    pub fn new() -> Self {
        Self {
            swarm_ctl: SwarmCtlAgent::default(),
            delegate_x: DelegateXAgent::default(),
            consensus_ai: ConsensusAiAgent::default(),
            task_router: TaskRouterAgent::default(),
            merge_synth: MergeSynthAgent::default(),
            priority_god: PriorityGodAgent::default(),
        }
    }

    pub fn swarm_ctl(&self) -> &SwarmCtlAgent {
        &self.swarm_ctl
    }

    pub fn swarm_ctl_mut(&mut self) -> &mut SwarmCtlAgent {
        &mut self.swarm_ctl
    }

    pub fn delegate_x(&self) -> &DelegateXAgent {
        &self.delegate_x
    }

    pub fn delegate_x_mut(&mut self) -> &mut DelegateXAgent {
        &mut self.delegate_x
    }

    pub fn consensus_ai(&self) -> &ConsensusAiAgent {
        &self.consensus_ai
    }

    pub fn consensus_ai_mut(&mut self) -> &mut ConsensusAiAgent {
        &mut self.consensus_ai
    }

    pub fn task_router(&self) -> &TaskRouterAgent {
        &self.task_router
    }

    pub fn task_router_mut(&mut self) -> &mut TaskRouterAgent {
        &mut self.task_router
    }

    pub fn merge_synth(&self) -> &MergeSynthAgent {
        &self.merge_synth
    }

    pub fn merge_synth_mut(&mut self) -> &mut MergeSynthAgent {
        &mut self.merge_synth
    }

    pub fn priority_god(&self) -> &PriorityGodAgent {
        &self.priority_god
    }

    pub fn priority_god_mut(&mut self) -> &mut PriorityGodAgent {
        &mut self.priority_god
    }

    pub fn get(&self, kind: AgentKind) -> AgentRef {
        match kind {
            AgentKind::SwarmCtl => AgentRef::SwarmCtl(&self.swarm_ctl),
            AgentKind::DelegateX => AgentRef::DelegateX(&self.delegate_x),
            AgentKind::ConsensusAi => AgentRef::ConsensusAi(&self.consensus_ai),
            AgentKind::TaskRouter => AgentRef::TaskRouter(&self.task_router),
            AgentKind::MergeSynth => AgentRef::MergeSynth(&self.merge_synth),
            AgentKind::PriorityGod => AgentRef::PriorityGod(&self.priority_god),
        }
    }

    pub fn get_mut(&mut self, kind: AgentKind) -> AgentRefMut {
        match kind {
            AgentKind::SwarmCtl => AgentRefMut::SwarmCtl(&mut self.swarm_ctl),
            AgentKind::DelegateX => AgentRefMut::DelegateX(&mut self.delegate_x),
            AgentKind::ConsensusAi => AgentRefMut::ConsensusAi(&mut self.consensus_ai),
            AgentKind::TaskRouter => AgentRefMut::TaskRouter(&mut self.task_router),
            AgentKind::MergeSynth => AgentRefMut::MergeSynth(&mut self.merge_synth),
            AgentKind::PriorityGod => AgentRefMut::PriorityGod(&mut self.priority_god),
        }
    }

    pub fn all_kinds(&self) -> Vec<AgentKind> {
        vec![
            AgentKind::SwarmCtl,
            AgentKind::DelegateX,
            AgentKind::ConsensusAi,
            AgentKind::TaskRouter,
            AgentKind::MergeSynth,
            AgentKind::PriorityGod,
        ]
    }
}

// ---------------------------------------------------------------------------
// Polymorphic references (enum dispatch)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum AgentRef<'a> {
    SwarmCtl(&'a SwarmCtlAgent),
    DelegateX(&'a DelegateXAgent),
    ConsensusAi(&'a ConsensusAiAgent),
    TaskRouter(&'a TaskRouterAgent),
    MergeSynth(&'a MergeSynthAgent),
    PriorityGod(&'a PriorityGodAgent),
}

pub enum AgentRefMut<'a> {
    SwarmCtl(&'a mut SwarmCtlAgent),
    DelegateX(&'a mut DelegateXAgent),
    ConsensusAi(&'a mut ConsensusAiAgent),
    TaskRouter(&'a mut TaskRouterAgent),
    MergeSynth(&'a mut MergeSynthAgent),
    PriorityGod(&'a mut PriorityGodAgent),
}

impl<'a> AgentRef<'a> {
    pub fn kind(&self) -> AgentKind {
        match self {
            AgentRef::SwarmCtl(_) => AgentKind::SwarmCtl,
            AgentRef::DelegateX(_) => AgentKind::DelegateX,
            AgentRef::ConsensusAi(_) => AgentKind::ConsensusAi,
            AgentRef::TaskRouter(_) => AgentKind::TaskRouter,
            AgentRef::MergeSynth(_) => AgentKind::MergeSynth,
            AgentRef::PriorityGod(_) => AgentKind::PriorityGod,
        }
    }

    pub fn name(&self) -> &'static str {
        self.kind().name()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swarm_ctl_spawn() {
        let agent = SwarmCtlAgent::default();
        let dep = agent.spawn_swarm(100);
        assert_eq!(dep.total_spawned, 100);
        assert!(dep.partitions >= 1);
    }

    #[test]
    fn test_delegate_x_register_and_find() {
        let mut agent = DelegateXAgent::default();
        let mut caps = CapabilityProfile::new();
        caps.insert("nlp".into(), 0.9);
        caps.insert("vision".into(), 0.3);
        agent.register_agent("alpha", caps);

        let mut query = CapabilityProfile::new();
        query.insert("nlp".into(), 0.8);
        let result = agent.find_best_match(&query);
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, "alpha");
    }

    #[test]
    fn test_consensus_ai_majority() {
        let agent = ConsensusAiAgent::default();
        let opinions = vec![
            AgentOpinion {
                agent_id: "a1".into(),
                position: "yes".into(),
                confidence: 0.9,
                reasoning: "".into(),
            },
            AgentOpinion {
                agent_id: "a2".into(),
                position: "yes".into(),
                confidence: 0.8,
                reasoning: "".into(),
            },
            AgentOpinion {
                agent_id: "a3".into(),
                position: "no".into(),
                confidence: 0.6,
                reasoning: "".into(),
            },
        ];
        let verdict = agent.evaluate(&opinions);
        assert!(verdict.consensus_reached);
        assert_eq!(verdict.position, "yes");
    }

    #[test]
    fn test_task_router_routing() {
        let mut agent = TaskRouterAgent::default();
        agent.register_pipeline(TaskPipeline {
            name: "fast-lane".into(),
            tags: vec!["urgent".into()],
            estimated_latency_ms: 5,
            throughput: 1000.0,
            is_active: true,
        });
        let result = agent.route("test", &["urgent".into()]);
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "fast-lane");
    }

    #[test]
    fn test_merge_synth_weighted() {
        let agent = MergeSynthAgent {
            strategy: MergeStrategy::WeightedAverage,
            ..Default::default()
        };
        let now = chrono::Utc::now();
        let inputs = vec![
            MergeInput {
                source: "a".into(),
                data: "result a".into(),
                confidence: 0.9,
                timestamp: now,
            },
            MergeInput {
                source: "b".into(),
                data: "result b".into(),
                confidence: 0.5,
                timestamp: now,
            },
        ];
        let output = agent.merge(&inputs);
        assert_eq!(output.merged_count, 2);
        assert!(output.result.contains("result a"));
    }

    #[test]
    fn test_priority_god_queue() {
        let mut agent = PriorityGodAgent::default();
        agent.enqueue(PrioritizedTask {
            id: "t1".into(),
            name: "low task".into(),
            base_priority: PriorityLevel::Low,
            overridden_priority: None,
            submitted_at: chrono::Utc::now(),
            deadline: None,
            tags: vec![],
        });
        agent.enqueue(PrioritizedTask {
            id: "t2".into(),
            name: "critical task".into(),
            base_priority: PriorityLevel::Critical,
            overridden_priority: None,
            submitted_at: chrono::Utc::now(),
            deadline: None,
            tags: vec![],
        });
        let next = agent.dequeue().unwrap();
        assert_eq!(next.id, "t2");
    }

    #[test]
    fn test_nexum_agents_container() {
        let agents = NexumAgents::new();
        assert_eq!(agents.swarm_ctl().kind(), AgentKind::SwarmCtl);
        assert_eq!(agents.delegate_x().kind(), AgentKind::DelegateX);
        assert_eq!(agents.consensus_ai().kind(), AgentKind::ConsensusAi);
        assert_eq!(agents.task_router().kind(), AgentKind::TaskRouter);
        assert_eq!(agents.merge_synth().kind(), AgentKind::MergeSynth);
        assert_eq!(agents.priority_god().kind(), AgentKind::PriorityGod);
    }

    #[test]
    fn test_agent_kind_names() {
        assert_eq!(AgentKind::SwarmCtl.name(), "SWARM-CTL");
        assert_eq!(AgentKind::DelegateX.name(), "DELEGATE-X");
        assert_eq!(AgentKind::ConsensusAi.name(), "CONSENSUS-AI");
        assert_eq!(AgentKind::TaskRouter.name(), "TASK-ROUTER");
        assert_eq!(AgentKind::MergeSynth.name(), "MERGE-SYNTH");
        assert_eq!(AgentKind::PriorityGod.name(), "PRIORITY-GOD");
    }
}
