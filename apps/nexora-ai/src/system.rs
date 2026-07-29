//! Nexora System — integration hub that wires all new subsystems together
//!
//! EventBus → scheduler, GPU, agent, cache, cost optimizer, observability
//! MemoryManager → pools for tensor, embedding, KV cache, buffer
//! DAGScheduler → dependency-aware parallel execution
//! CostOptimizer → cascade routing (regex → rule → small → medium → large)
//! ObservabilityCollector → 33 metrics, dashboard data, prometheus output
//! AgentAutoscaler → dynamic pool (3–49 agents)

use nexora_cost_optimizer::CostOptimizer;
use nexora_eventbus::{EventBus, Event};
use nexora_memory::pool::UnifiedMemoryManager;
use nexora_memory::hybrid_cache::HybridCacheManager;
use nexora_monitoring::observability::{
    ObservabilityCollector, ObservabilityMetrics, MetricsReporter,
};
use nexora_runtime::scheduler_v2::DagScheduler;
use nexora_agent::scaling::AgentAutoscaler;
use nexora_runtime::gpu_runtime::scheduler::GpuScheduler;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::info;

/// System-wide integration — bootstrap semua subsystem baru
#[derive(Clone)]
pub struct NexoraSystem {
    pub event_bus: Arc<EventBus>,
    pub memory_manager: Arc<UnifiedMemoryManager>,
    pub hybrid_cache: Arc<HybridCacheManager>,
    pub dag_scheduler: Option<Arc<DagScheduler>>,
    pub cost_optimizer: Arc<CostOptimizer>,
    pub observability: Arc<ObservabilityCollector>,
    pub metrics_reporter: Arc<MetricsReporter>,
    pub agent_autoscaler: Option<Arc<AgentAutoscaler>>,
    pub gpu_scheduler: Option<Arc<GpuScheduler>>,
    running: Arc<AtomicBool>,
}

impl NexoraSystem {
    pub async fn new() -> Self {
        info!("NexoraSystem initializing...");
        let event_bus = Arc::new(EventBus::new());
        let memory_manager = Arc::new(UnifiedMemoryManager::new());
        let hybrid_cache = Arc::new(HybridCacheManager::new());
        let cost_optimizer = Arc::new(CostOptimizer::new());
        let observability = Arc::new(ObservabilityCollector::new());
        let metrics_reporter = Arc::new(MetricsReporter::new(observability.clone()));

        // Emit startup event
        event_bus.emit_system_event("system.init", "NexoraSystem starting");

        Self {
            event_bus,
            memory_manager,
            hybrid_cache,
            dag_scheduler: None,
            cost_optimizer,
            observability,
            metrics_reporter,
            agent_autoscaler: None,
            gpu_scheduler: None,
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Init DAG scheduler with worker count
    pub fn with_dag_scheduler(mut self, workers: usize, gpus: usize) -> Self {
        let scheduler = Arc::new(DagScheduler::new(workers, gpus));
        self.dag_scheduler = Some(scheduler);
        self
    }

    /// Init GPU scheduler
    pub fn with_gpu_scheduler(mut self, gpu_count: usize) -> Self {
        self.gpu_scheduler = Some(Arc::new(GpuScheduler::new(gpu_count)));
        self
    }

    /// Init agent autoscaler
    pub fn with_agent_autoscaler(mut self) -> Self {
        let config = nexora_agent::scaling::AgentScalingConfig::default();
        let autoscaler = Arc::new(AgentAutoscaler::new(config));
        self.agent_autoscaler = Some(autoscaler);
        self
    }

    /// Start all background services
    pub async fn start(&self) {
        self.running.store(true, Ordering::SeqCst);

        // Observability background collection (every 10s)
        self.observability.start_background_collection(10);

        // Start auto-scaler if configured
        if let Some(ref autoscaler) = self.agent_autoscaler {
            tokio::spawn({
                let a = autoscaler.clone();
                async move { a.start().await }
            });
        }

        // Start DAG scheduler if configured
        if let Some(ref scheduler) = self.dag_scheduler {
            let sched = scheduler.clone();
            tokio::spawn(async move { sched.start().await });
        }

        info!("NexoraSystem started");
        self.event_bus.emit_system_event("system.ready", "NexoraSystem ready");
    }

    /// Route request through cost optimizer → report metrics
    pub async fn optimize_request(&self, input: &str) -> (bool, f64) {
        let result = self.cost_optimizer.optimize(input);
        self.observability.record_cost(
            result.estimated_cost,
            result.savings_vs_large,
        );
        (result.use_large_model, result.estimated_cost)
    }

    /// Cache hit/miss recording
    pub fn record_cache_hit(&self) {
        self.observability.record_cache_hit();
    }

    pub fn record_cache_miss(&self) {
        self.observability.record_cache_miss();
    }

    /// Latency tracking
    pub fn record_latency(&self, ms: u64) {
        self.observability.record_latency(ms);
    }

    /// Get dashboard JSON
    pub fn dashboard_json(&self) -> String {
        let snapshot = self.observability.snapshot();
        let uptime = std::time::Instant::now().elapsed().as_secs();
        let data = nexora_monitoring::observability::DashboardData::from_metrics(
            &snapshot, uptime,
        );
        data.to_json()
    }

    /// Get prometheus metrics
    pub fn prometheus_metrics(&self) -> String {
        self.metrics_reporter.as_prometheus()
    }

    pub async fn shutdown(&self) {
        self.running.store(false, Ordering::SeqCst);
        self.event_bus.emit_system_event("system.shutdown", "NexoraSystem stopping");
        self.event_bus.shutdown().await;
        self.observability.shutdown();
        info!("NexoraSystem shutdown");
    }
}
