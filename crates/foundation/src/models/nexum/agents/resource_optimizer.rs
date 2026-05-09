//! Resource Optimizer Agent
//! 
//! Optimizes resource allocation and utilization across the multi-agent system

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// Resource Optimizer Agent - Optimizes resource allocation and utilization
#[derive(Debug, Clone)]
pub struct ResourceOptimizerAgent {
    /// Agent configuration
    pub config: ResourceOptimizerConfig,
    /// Optimization capabilities
    pub optimization_capabilities: OptimizationCapabilities,
    /// Resource management
    pub resource_management: ResourceManagement,
    /// Performance monitoring
    pub performance_monitoring: PerformanceMonitoring,
    /// Agent status
    status: AgentStatus,
    /// Agent metrics
    metrics: AgentMetrics,
}

/// Resource Optimizer Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceOptimizerConfig {
    /// Base agent configuration
    pub base_config: BaseAgentConfig,
    /// Optimization strategy
    pub optimization_strategy: OptimizationStrategy,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
    /// Performance targets
    pub performance_targets: PerformanceTargets,
    /// Optimization algorithms
    pub optimization_algorithms: Vec<OptimizationAlgorithm>,
}

/// Optimization Strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    /// Cost minimization
    CostMinimization,
    /// Performance maximization
    PerformanceMaximization,
    /// Balanced optimization
    BalancedOptimization { weights: OptimizationWeights },
    /// Adaptive optimization
    AdaptiveOptimization,
    /// Multi-objective optimization
    MultiObjectiveOptimization { objectives: Vec<OptimizationObjective> },
    /// Real-time optimization
    RealTimeOptimization,
}

/// Optimization Weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationWeights {
    /// Performance weight
    pub performance_weight: f32,
    /// Cost weight
    pub cost_weight: f32,
    /// Efficiency weight
    pub efficiency_weight: f32,
    /// Reliability weight
    pub reliability_weight: f32,
    /// Scalability weight
    pub scalability_weight: f32,
}

/// Optimization Objective
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationObjective {
    /// Objective name
    pub name: String,
    /// Objective type
    pub objective_type: ObjectiveType,
    /// Target value
    pub target: f32,
    /// Priority
    pub priority: u8,
    /// Constraints
    pub constraints: Vec<String>,
}

/// Objective Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ObjectiveType {
    /// Minimize objective
    Minimize,
    /// Maximize objective
    Maximize,
    /// Target objective
    Target,
    /// Range objective
    Range { min: f32, max: f32 },
}

/// Resource Constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// CPU constraints
    pub cpu_constraints: CpuConstraints,
    /// Memory constraints
    pub memory_constraints: MemoryConstraints,
    /// Network constraints
    pub network_constraints: NetworkConstraints,
    /// Storage constraints
    pub storage_constraints: StorageConstraints,
    /// Budget constraints
    pub budget_constraints: BudgetConstraints,
}

/// Cpu Constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuConstraints {
    /// Maximum CPU usage
    pub max_cpu_usage: f32,
    /// CPU cores available
    pub cpu_cores: u32,
    /// CPU architecture
    pub cpu_architecture: String,
    /// Specialized hardware
    pub specialized_hardware: Vec<String>,
}

/// Memory Constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConstraints {
    /// Maximum memory usage
    pub max_memory_gb: f32,
    /// Memory type
    pub memory_type: MemoryType,
    /// Memory bandwidth
    pub memory_bandwidth_gbps: f32,
    /// Swap space
    pub swap_space_gb: f32,
}

/// Memory Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryType {
    /// DDR4
    DDR4,
    /// DDR5
    DDR5,
    /// HBM
    HBM,
    /// GDDR
    GDDR,
    /// LPDDR
    LPDDR,
}

/// Network Constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConstraints {
    /// Maximum bandwidth
    pub max_bandwidth_mbps: f32,
    /// Latency requirements
    pub latency_requirements: LatencyRequirements,
    /// Connection limits
    pub connection_limits: ConnectionLimits,
    /// Network protocols
    pub supported_protocols: Vec<String>,
}

/// Latency Requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyRequirements {
    /// Maximum latency
    pub max_latency_ms: f32,
    /// Average latency target
    pub avg_latency_target_ms: f32,
    /// Latency percentile
    pub latency_percentile: f32,
    /// Jitter tolerance
    pub jitter_tolerance_ms: f32,
}

/// Connection Limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionLimits {
    /// Maximum concurrent connections
    pub max_concurrent_connections: u32,
    /// Connection rate limit
    pub connection_rate_limit: u32,
    /// Timeout settings
    pub timeout_settings: TimeoutSettings,
}

/// Timeout Settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutSettings {
    /// Connection timeout
    pub connection_timeout_ms: u64,
    /// Read timeout
    pub read_timeout_ms: u64,
    /// Write timeout
    pub write_timeout_ms: u64,
    /// Idle timeout
    pub idle_timeout_ms: u64,
}

/// Storage Constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConstraints {
    /// Maximum storage usage
    pub max_storage_gb: f32,
    /// Storage type
    pub storage_type: StorageType,
    /// IOPS requirements
    pub iops_requirements: IopsRequirements,
    /// Throughput requirements
    pub throughput_requirements: ThroughputRequirements,
}

/// Storage Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageType {
    /// SSD
    SSD,
    /// HDD
    HDD,
    /// NVMe
    NVMe,
    /// Cloud storage
    CloudStorage { provider: String, region: String },
    /// Hybrid storage
    HybridStorage,
}

/// Iops Requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IopsRequirements {
    /// Read IOPS
    pub read_iops: u32,
    /// Write IOPS
    pub write_iops: u32,
    /// Mixed IOPS
    pub mixed_iops: u32,
    /// IOPS consistency
    pub iops_consistency: f32,
}

/// Throughput Requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputRequirements {
    /// Read throughput MB/s
    pub read_throughput_mbps: f32,
    /// Write throughput MB/s
    pub write_throughput_mbps: f32,
    /// Mixed throughput MB/s
    pub mixed_throughput_mbps: f32,
    /// Sustained duration
    pub sustained_duration_seconds: u64,
}

/// Budget Constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConstraints {
    /// Maximum hourly cost
    pub max_hourly_cost: f32,
    /// Maximum daily cost
    pub max_daily_cost: f32,
    /// Maximum monthly cost
    pub max_monthly_cost: f32,
    /// Cost optimization level
    pub cost_optimization_level: CostOptimizationLevel,
}

/// Cost Optimization Level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CostOptimizationLevel {
    /// Aggressive optimization
    Aggressive,
    /// Moderate optimization
    Moderate,
    /// Conservative optimization
    Conservative,
    /// No optimization
    None,
}

/// Performance Targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    /// Response time targets
    pub response_time_targets: ResponseTimeTargets,
    /// Throughput targets
    pub throughput_targets: ThroughputTargets,
    /// Availability targets
    pub availability_targets: AvailabilityTargets,
    /// Quality targets
    pub quality_targets: QualityTargets,
}

/// Response Time Targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTimeTargets {
    /// P50 response time
    pub p50_response_time_ms: f32,
    /// P95 response time
    pub p95_response_time_ms: f32,
    /// P99 response time
    pub p99_response_time_ms: f32,
    /// Maximum response time
    pub max_response_time_ms: f32,
}

/// Throughput Targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputTargets {
    /// Requests per second
    pub requests_per_second: u32,
    /// Concurrent users
    pub concurrent_users: u32,
    /// Data processing rate
    pub data_processing_rate_mbps: f32,
    /// Peak throughput
    pub peak_throughput_factor: f32,
}

/// Availability Targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailabilityTargets {
    /// Uptime percentage
    pub uptime_percentage: f32,
    /// Mean time between failures
    pub mtbf_hours: f32,
    /// Mean time to recovery
    pub mttr_minutes: f32,
    /// Disaster recovery time
    pub disaster_recovery_time_hours: f32,
}

/// Quality Targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityTargets {
    /// Error rate
    pub error_rate: f32,
    /// Success rate
    pub success_rate: f32,
    /// Customer satisfaction
    pub customer_satisfaction: f32,
    /// System reliability
    pub system_reliability: f32,
}

/// Optimization Algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationAlgorithm {
    /// Linear programming
    LinearProgramming,
    /// Genetic algorithm
    GeneticAlgorithm { population_size: u32, generations: u32 },
    /// Simulated annealing
    SimulatedAnnealing { temperature: f32, cooling_rate: f32 },
    /// Particle swarm optimization
    ParticleSwarmOptimization { particles: u32, iterations: u32 },
    /// Gradient descent
    GradientDescent { learning_rate: f32, momentum: f32 },
    /// Reinforcement learning
    ReinforcementLearning { algorithm: String, exploration_rate: f32 },
}

/// Optimization Capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationCapabilities {
    /// Resource allocation optimization
    pub resource_allocation_optimization: bool,
    /// Performance tuning
    pub performance_tuning: bool,
    /// Cost optimization
    pub cost_optimization: bool,
    /// Load balancing
    pub load_balancing: bool,
    /// Auto scaling
    pub auto_scaling: bool,
    /// Predictive optimization
    pub predictive_optimization: bool,
}

/// Resource Management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceManagement {
    /// Resource pools
    pub resource_pools: HashMap<String, ResourcePool>,
    /// Allocation strategies
    pub allocation_strategies: Vec<AllocationStrategy>,
    /// Resource scheduling
    pub resource_scheduling: ResourceScheduling,
    /// Resource monitoring
    pub resource_monitoring: ResourceMonitoring,
}

/// Resource Pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePool {
    /// Pool ID
    pub pool_id: String,
    /// Pool type
    pub pool_type: ResourceType,
    /// Total capacity
    pub total_capacity: f32,
    /// Available capacity
    pub available_capacity: f32,
    /// Allocated capacity
    pub allocated_capacity: f32,
    /// Reserved capacity
    pub reserved_capacity: f32,
    /// Priority level
    pub priority_level: u8,
}

/// ResourceType
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceType {
    /// CPU
    CPU,
    /// Memory
    Memory,
    /// Storage
    Storage,
    /// Network
    Network,
    /// GPU
    GPU,
    /// TPU
    TPU,
    /// Custom resource
    Custom { name: String, unit: String },
}

/// Allocation Strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// First fit
    FirstFit,
    /// Best fit
    BestFit,
    /// Worst fit
    WorstFit,
    /// Round robin
    RoundRobin,
    /// Priority based
    PriorityBased,
    /// Load balanced
    LoadBalanced,
    /// Predictive allocation
    PredictiveAllocation,
}

/// Resource Scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceScheduling {
    /// Scheduling algorithms
    pub algorithms: Vec<SchedulingAlgorithm>,
    /// Queue management
    pub queue_management: QueueManagement,
    /// Priority handling
    pub priority_handling: PriorityHandling,
    /// Preemption policies
    pub preemption_policies: Vec<PreemptionPolicy>,
}

/// Scheduling Algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulingAlgorithm {
    /// First come first serve
    FirstComeFirstServe,
    /// Shortest job first
    ShortestJobFirst,
    /// Priority scheduling
    PriorityScheduling,
    /// Fair share scheduling
    FairShareScheduling,
    /// Real-time scheduling
    RealTimeScheduling,
    /// Deadline scheduling
    DeadlineScheduling,
}

/// Queue Management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueManagement {
    /// Queue types
    pub queue_types: Vec<QueueType>,
    /// Queue priorities
    pub queue_priorities: HashMap<String, u8>,
    /// Queue limits
    pub queue_limits: HashMap<String, u32>,
    /// Queue overflow handling
    pub queue_overflow_handling: OverflowHandling,
}

/// QueueType
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueueType {
    /// FIFO queue
    FIFO,
    /// Priority queue
    Priority,
    /// Fair queue
    FairQueue,
    /// Weighted fair queue
    WeightedFairQueue,
    /// Deadline queue
    DeadlineQueue,
}

/// OverflowHandling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OverflowHandling {
    /// Drop tail
    DropTail,
    /// Drop head
    DropHead,
    /// Random drop
    RandomDrop,
    /// Priority drop
    PriorityDrop,
    /// Queue expansion
    QueueExpansion,
}

/// PriorityHandling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityHandling {
    /// Priority levels
    pub priority_levels: u8,
    /// Priority inheritance
    pub priority_inheritance: bool,
    /// Priority inversion handling
    pub priority_inversion_handling: bool,
    /// Dynamic priority adjustment
    pub dynamic_priority_adjustment: bool,
}

/// Preemption Policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreemptionPolicy {
    /// No preemption
    NoPreemption,
    /// Priority preemption
    PriorityPreemption,
    /// Time slice preemption
    TimeSlicePreemption { time_slice_ms: u64 },
    /// Resource preemption
    ResourcePreemption { resources: Vec<ResourceType> },
    /// Cost based preemption
    CostBasedPreemption { cost_threshold: f32 },
}

/// Resource Monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMonitoring {
    /// Monitoring metrics
    pub metrics: Vec<ResourceMetric>,
    /// Collection intervals
    pub collection_intervals: HashMap<String, u64>,
    /// Alert thresholds
    pub alert_thresholds: HashMap<String, f32>,
    /// Historical data retention
    pub historical_data_retention_days: u32,
}

/// Resource Metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceMetric {
    /// CPU utilization
    CpuUtilization,
    /// Memory usage
    MemoryUsage,
    /// Disk I/O
    DiskIO,
    /// Network I/O
    NetworkIO,
    /// Response time
    ResponseTime,
    /// Throughput
    Throughput,
    /// Error rate
    ErrorRate,
    /// Queue depth
    QueueDepth,
}

/// Performance Monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMonitoring {
    /// Performance metrics
    pub performance_metrics: Vec<PerformanceMetric>,
    /// Benchmarking
    pub benchmarking: Benchmarking,
    /// Anomaly detection
    pub anomaly_detection: AnomalyDetection,
    /// Performance prediction
    pub performance_prediction: PerformancePrediction,
}

/// Performance Metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceMetric {
    /// Latency metrics
    LatencyMetrics,
    /// Throughput metrics
    ThroughputMetrics,
    /// Resource efficiency
    ResourceEfficiency,
    /// Cost efficiency
    CostEfficiency,
    /// Quality metrics
    QualityMetrics,
    /// User experience metrics
    UserExperienceMetrics,
}

/// Benchmarking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Benchmarking {
    /// Benchmark types
    pub benchmark_types: Vec<BenchmarkType>,
    /// Benchmark frequency
    pub benchmark_frequency_hours: u32,
    /// Baseline establishment
    pub baseline_establishment: bool,
    /// Regression detection
    pub regression_detection: bool,
}

/// BenchmarkType
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BenchmarkType {
    /// Performance benchmark
    PerformanceBenchmark,
    /// Load benchmark
    LoadBenchmark,
    /// Stress benchmark
    StressBenchmark,
    /// Capacity benchmark
    CapacityBenchmark,
    /// Cost benchmark
    CostBenchmark,
}

/// Anomaly Detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetection {
    /// Detection algorithms
    pub algorithms: Vec<AnomalyDetectionAlgorithm>,
    /// Sensitivity level
    pub sensitivity_level: f32,
    /// False positive tolerance
    pub false_positive_tolerance: f32,
    /// Alert mechanisms
    pub alert_mechanisms: Vec<AlertMechanism>,
}

/// Anomaly Detection Algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyDetectionAlgorithm {
    /// Statistical outlier
    StatisticalOutlier,
    /// Machine learning based
    MachineLearningBased { model_type: String },
    /// Rule based
    RuleBased,
    /// Time series analysis
    TimeSeriesAnalysis,
    /// Pattern recognition
    PatternRecognition,
}

/// AlertMechanism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertMechanism {
    /// Email alert
    EmailAlert,
    /// SMS alert
    SMSAlert,
    /// Webhook alert
    WebhookAlert { url: String },
    /// Dashboard alert
    DashboardAlert,
    /// Log alert
    LogAlert,
}

/// Performance Prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePrediction {
    /// Prediction models
    pub prediction_models: Vec<PredictionModel>,
    /// Prediction horizon
    pub prediction_horizon_hours: u32,
    /// Confidence thresholds
    pub confidence_thresholds: HashMap<String, f32>,
    /// Model accuracy requirements
    pub model_accuracy_requirements: f32,
}

/// PredictionModel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PredictionModel {
    /// Linear regression
    LinearRegression,
    /// Time series model
    TimeSeriesModel { model_type: String },
    /// Neural network
    NeuralNetwork { layers: Vec<u32> },
    /// Ensemble model
    EnsembleModel { models: Vec<PredictionModel> },
    /// Custom model
    CustomModel { name: String, parameters: HashMap<String, f32> },
}

/// Resource Optimization Task Input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceOptimizationTaskInput {
    /// Current resource allocation
    pub current_allocation: HashMap<String, ResourceAllocation>,
    /// Resource demands
    pub resource_demands: Vec<ResourceDemand>,
    /// Performance metrics
    pub performance_metrics: HashMap<String, f32>,
    /// Optimization constraints
    pub optimization_constraints: OptimizationConstraints,
    /// Optimization objectives
    pub optimization_objectives: Vec<OptimizationObjective>,
}

/// Resource Allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// Agent ID
    pub agent_id: String,
    /// Resource type
    pub resource_type: ResourceType,
    /// Allocated amount
    pub allocated_amount: f32,
    /// Utilization percentage
    pub utilization_percentage: f32,
    /// Allocation timestamp
    pub allocation_timestamp: chrono::DateTime<chrono::Utc>,
    /// Expiration timestamp
    pub expiration_timestamp: Option<chrono::DateTime<chrono::Utc>>,
}

/// Resource Demand
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceDemand {
    /// Agent ID
    pub agent_id: String,
    /// Resource type
    pub resource_type: ResourceType,
    /// Required amount
    pub required_amount: f32,
    /// Priority level
    pub priority_level: u8,
    /// Duration seconds
    pub duration_seconds: u64,
    /// Flexible allocation
    pub flexible_allocation: bool,
}

/// Optimization Constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConstraints {
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
    /// Time constraints
    pub time_constraints: TimeConstraints,
    /// Budget constraints
    pub budget_constraints: BudgetConstraints,
    /// Policy constraints
    pub policy_constraints: Vec<PolicyConstraint>,
}

/// Time Constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeConstraints {
    /// Optimization window
    pub optimization_window_seconds: u64,
    /// Maximum execution time
    pub max_execution_time_seconds: u64,
    /// Deadline
    pub deadline: Option<chrono::DateTime<chrono::Utc>>,
    /// Frequency constraints
    pub frequency_constraints: FrequencyConstraints,
}

/// Frequency Constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyConstraints {
    /// Minimum interval between optimizations
    pub min_interval_seconds: u64,
    /// Maximum optimizations per hour
    pub max_optimizations_per_hour: u32,
    /// Peak hour restrictions
    pub peak_hour_restrictions: bool,
}

/// Policy Constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyConstraint {
    /// Constraint ID
    pub constraint_id: String,
    /// Constraint description
    pub description: String,
    /// Constraint type
    pub constraint_type: PolicyConstraintType,
    /// Enforcement level
    pub enforcement_level: EnforcementLevel,
}

/// PolicyConstraintType
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyConstraintType {
    /// Security policy
    SecurityPolicy,
    /// Compliance policy
    CompliancePolicy,
    /// Business policy
    BusinessPolicy,
    /// Technical policy
    TechnicalPolicy,
    /// Resource policy
    ResourcePolicy,
}

/// EnforcementLevel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnforcementLevel {
    /// Advisory
    Advisory,
    /// Warning
    Warning,
    /// Error
    Error,
    /// Critical
    Critical,
}

/// Resource Optimization Task Output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceOptimizationTaskOutput {
    /// Optimized allocation
    pub optimized_allocation: HashMap<String, ResourceAllocation>,
    /// Optimization results
    pub optimization_results: OptimizationResults,
    /// Performance improvements
    pub performance_improvements: PerformanceImprovements,
    /// Cost analysis
    pub cost_analysis: CostAnalysis,
    /// Recommendations
    pub recommendations: Vec<OptimizationRecommendation>,
}

/// Optimization Results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResults {
    /// Success indicator
    pub success: bool,
    /// Optimization score
    pub optimization_score: f32,
    /// Resource utilization improvement
    pub resource_utilization_improvement: f32,
    /// Performance improvement
    pub performance_improvement: f32,
    /// Cost savings
    pub cost_savings: f32,
    /// Execution time
    pub execution_time_ms: u64,
}

/// Performance Improvements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImprovements {
    /// Latency improvement
    pub latency_improvement: f32,
    /// Throughput improvement
    pub throughput_improvement: f32,
    /// Resource efficiency improvement
    pub resource_efficiency_improvement: f32,
    /// Quality improvement
    pub quality_improvement: f32,
    /// User experience improvement
    pub user_experience_improvement: f32,
}

/// Cost Analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAnalysis {
    /// Current cost per hour
    pub current_cost_per_hour: f32,
    /// Optimized cost per hour
    pub optimized_cost_per_hour: f32,
    /// Cost savings percentage
    pub cost_savings_percentage: f32,
    /// Cost breakdown
    pub cost_breakdown: HashMap<String, f32>,
    /// Roi calculation
    pub roi_calculation: RoiCalculation,
}

/// RoiCalculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoiCalculation {
    /// Investment cost
    pub investment_cost: f32,
    /// Expected savings per month
    pub expected_savings_per_month: f32,
    /// Payback period months
    pub payback_period_months: f32,
    /// Annual ROI percentage
    pub annual_roi_percentage: f32,
}

/// Optimization Recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Recommendation ID
    pub recommendation_id: String,
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Description
    pub description: String,
    /// Expected impact
    pub expected_impact: ExpectedImpact,
    /// Implementation complexity
    pub implementation_complexity: ImplementationComplexity,
    /// Priority level
    pub priority_level: u8,
}

/// RecommendationType
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Resource scaling
    ResourceScaling,
    /// Load balancing
    LoadBalancing,
    /// Configuration tuning
    ConfigurationTuning,
    /// Architecture change
    ArchitectureChange,
    /// Cost optimization
    CostOptimization,
    /// Performance optimization
    PerformanceOptimization,
}

/// ExpectedImpact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedImpact {
    /// Performance impact
    pub performance_impact: f32,
    /// Cost impact
    pub cost_impact: f32,
    /// Resource impact
    pub resource_impact: f32,
    /// Quality impact
    pub quality_impact: f32,
    /// Time to implement
    pub time_to_implement_days: u32,
}

impl Default for ResourceOptimizerConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            optimization_strategy: OptimizationStrategy::BalancedOptimization {
                weights: OptimizationWeights {
                    performance_weight: 0.4,
                    cost_weight: 0.3,
                    efficiency_weight: 0.2,
                    reliability_weight: 0.05,
                    scalability_weight: 0.05,
                },
            },
            resource_constraints: ResourceConstraints {
                cpu_constraints: CpuConstraints {
                    max_cpu_usage: 80.0,
                    cpu_cores: 16,
                    cpu_architecture: "x86_64".to_string(),
                    specialized_hardware: vec![],
                },
                memory_constraints: MemoryConstraints {
                    max_memory_gb: 64.0,
                    memory_type: MemoryType::DDR4,
                    memory_bandwidth_gbps: 25.6,
                    swap_space_gb: 32.0,
                },
                network_constraints: NetworkConstraints {
                    max_bandwidth_mbps: 1000.0,
                    latency_requirements: LatencyRequirements {
                        max_latency_ms: 100.0,
                        avg_latency_target_ms: 50.0,
                        latency_percentile: 95.0,
                        jitter_tolerance_ms: 10.0,
                    },
                    connection_limits: ConnectionLimits {
                        max_concurrent_connections: 1000,
                        connection_rate_limit: 100,
                        timeout_settings: TimeoutSettings {
                            connection_timeout_ms: 5000,
                            read_timeout_ms: 30000,
                            write_timeout_ms: 30000,
                            idle_timeout_ms: 60000,
                        },
                    },
                    supported_protocols: vec!["HTTP".to_string(), "HTTPS".to_string(), "TCP".to_string()],
                },
                storage_constraints: StorageConstraints {
                    max_storage_gb: 1000.0,
                    storage_type: StorageType::SSD,
                    iops_requirements: IopsRequirements {
                        read_iops: 10000,
                        write_iops: 5000,
                        mixed_iops: 7500,
                        iops_consistency: 0.9,
                    },
                    throughput_requirements: ThroughputRequirements {
                        read_throughput_mbps: 500.0,
                        write_throughput_mbps: 300.0,
                        mixed_throughput_mbps: 400.0,
                        sustained_duration_seconds: 3600,
                    },
                },
                budget_constraints: BudgetConstraints {
                    max_hourly_cost: 10.0,
                    max_daily_cost: 200.0,
                    max_monthly_cost: 5000.0,
                    cost_optimization_level: CostOptimizationLevel::Moderate,
                },
            },
            performance_targets: PerformanceTargets {
                response_time_targets: ResponseTimeTargets {
                    p50_response_time_ms: 100.0,
                    p95_response_time_ms: 500.0,
                    p99_response_time_ms: 1000.0,
                    max_response_time_ms: 2000.0,
                },
                throughput_targets: ThroughputTargets {
                    requests_per_second: 1000,
                    concurrent_users: 10000,
                    data_processing_rate_mbps: 100.0,
                    peak_throughput_factor: 2.0,
                },
                availability_targets: AvailabilityTargets {
                    uptime_percentage: 99.9,
                    mtbf_hours: 8760.0,
                    mttr_minutes: 15.0,
                    disaster_recovery_time_hours: 4.0,
                },
                quality_targets: QualityTargets {
                    error_rate: 0.01,
                    success_rate: 0.99,
                    customer_satisfaction: 4.5,
                    system_reliability: 0.995,
                },
            },
            optimization_algorithms: vec![
                OptimizationAlgorithm::LinearProgramming,
                OptimizationAlgorithm::GeneticAlgorithm { population_size: 100, generations: 50 },
            ],
        }
    }
}

impl Default for OptimizationCapabilities {
    fn default() -> Self {
        Self {
            resource_allocation_optimization: true,
            performance_tuning: true,
            cost_optimization: true,
            load_balancing: true,
            auto_scaling: true,
            predictive_optimization: true,
        }
    }
}

impl Default for ResourceManagement {
    fn default() -> Self {
        Self {
            resource_pools: HashMap::new(),
            allocation_strategies: vec![
                AllocationStrategy::BestFit,
                AllocationStrategy::PriorityBased,
            ],
            resource_scheduling: ResourceScheduling {
                algorithms: vec![
                    SchedulingAlgorithm::PriorityScheduling,
                    SchedulingAlgorithm::FairShareScheduling,
                ],
                queue_management: QueueManagement {
                    queue_types: vec![QueueType::Priority, QueueType::FIFO],
                    queue_priorities: HashMap::new(),
                    queue_limits: HashMap::new(),
                    queue_overflow_handling: OverflowHandling::PriorityDrop,
                },
                priority_handling: PriorityHandling {
                    priority_levels: 5,
                    priority_inheritance: true,
                    priority_inversion_handling: true,
                    dynamic_priority_adjustment: true,
                },
                preemption_policies: vec![
                    PreemptionPolicy::PriorityPreemption,
                    PreemptionPolicy::ResourcePreemption { resources: vec![ResourceType::CPU, ResourceType::Memory] },
                ],
            },
            resource_monitoring: ResourceMonitoring {
                metrics: vec![
                    ResourceMetric::CpuUtilization,
                    ResourceMetric::MemoryUsage,
                    ResourceMetric::ResponseTime,
                    ResourceMetric::Throughput,
                ],
                collection_intervals: HashMap::new(),
                alert_thresholds: HashMap::new(),
                historical_data_retention_days: 30,
            },
        }
    }
}

impl Default for PerformanceMonitoring {
    fn default() -> Self {
        Self {
            performance_metrics: vec![
                PerformanceMetric::LatencyMetrics,
                PerformanceMetric::ThroughputMetrics,
                PerformanceMetric::ResourceEfficiency,
                PerformanceMetric::CostEfficiency,
            ],
            benchmarking: Benchmarking {
                benchmark_types: vec![
                    BenchmarkType::PerformanceBenchmark,
                    BenchmarkType::LoadBenchmark,
                ],
                benchmark_frequency_hours: 24,
                baseline_establishment: true,
                regression_detection: true,
            },
            anomaly_detection: AnomalyDetection {
                algorithms: vec![
                    AnomalyDetectionAlgorithm::StatisticalOutlier,
                    AnomalyDetectionAlgorithm::MachineLearningBased { model_type: "IsolationForest".to_string() },
                ],
                sensitivity_level: 0.7,
                false_positive_tolerance: 0.1,
                alert_mechanisms: vec![
                    AlertMechanism::DashboardAlert,
                    AlertMechanism::LogAlert,
                ],
            },
            performance_prediction: PerformancePrediction {
                prediction_models: vec![
                    PredictionModel::TimeSeriesModel { model_type: "ARIMA".to_string() },
                    PredictionModel::LinearRegression,
                ],
                prediction_horizon_hours: 24,
                confidence_thresholds: HashMap::new(),
                model_accuracy_requirements: 0.8,
            },
        }
    }
}

impl Default for ResourceOptimizerAgent {
    fn default() -> Self {
        Self {
            config: ResourceOptimizerConfig::default(),
            optimization_capabilities: OptimizationCapabilities::default(),
            resource_management: ResourceManagement::default(),
            performance_monitoring: PerformanceMonitoring::default(),
            status: AgentStatus::Idle,
            metrics: AgentMetrics {
                tasks_processed: 0,
                avg_processing_time: 0.0,
                success_rate: 1.0,
                current_load: 0.0,
                last_activity: chrono::Utc::now(),
            },
        }
    }
}

#[async_trait]
impl BaseAgent for ResourceOptimizerAgent {
    type Config = ResourceOptimizerConfig;
    type Input = ResourceOptimizationTaskInput;
    type Output = ResourceOptimizationTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let start_time = std::time::Instant::now();
        
        // Validate input
        self.validate_input(&input)?;
        
        // Analyze current allocation
        let allocation_analysis = self.analyze_current_allocation(&input).await?;
        
        // Optimize resource allocation
        let optimization_result = self.optimize_resource_allocation(&input, &allocation_analysis).await?;
        
        // Calculate performance improvements
        let performance_improvements = self.calculate_performance_improvements(&input, &optimization_result).await?;
        
        // Perform cost analysis
        let cost_analysis = self.perform_cost_analysis(&input, &optimization_result).await?;
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(&input, &optimization_result, &performance_improvements).await?;
        
        // Build output
        let output = ResourceOptimizationTaskOutput {
            optimized_allocation: optimization_result.optimized_allocation,
            optimization_results: optimization_result.results,
            performance_improvements,
            cost_analysis,
            recommendations,
        };
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        Ok(output)
    }

    fn agent_id(&self) -> &str {
        &self.config.base_config.agent_id
    }

    fn get_status(&self) -> AgentStatus {
        self.status.clone()
    }

    fn get_capabilities(&self) -> Vec<AgentCapability> {
        vec![
            AgentCapability {
                name: "resource_optimization".to_string(),
                description: "Optimizes resource allocation and utilization".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["resource_allocation".to_string(), "resource_demands".to_string()],
                output_types: vec!["optimized_allocation".to_string(), "performance_improvements".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.90,
                    avg_latency: 1000.0,
                    resource_usage: 0.8,
                    reliability: 0.95,
                },
            },
        ]
    }

    fn get_metrics(&self) -> AgentMetrics {
        self.metrics.clone()
    }

    async fn initialize(&mut self, config: Self::Config) -> AgentResult<()> {
        self.config = config;
        self.status = AgentStatus::Idle;
        Ok(())
    }

    async fn shutdown(&mut self) -> AgentResult<()> {
        self.status = AgentStatus::Disabled;
        Ok(())
    }
}

impl ResourceOptimizerAgent {
    /// Create a new Resource Optimizer Agent
    pub fn new(config: ResourceOptimizerConfig) -> Self {
        Self {
            config,
            optimization_capabilities: OptimizationCapabilities::default(),
            resource_management: ResourceManagement::default(),
            performance_monitoring: PerformanceMonitoring::default(),
            status: AgentStatus::Idle,
            metrics: AgentMetrics {
                tasks_processed: 0,
                avg_processing_time: 0.0,
                success_rate: 1.0,
                current_load: 0.0,
                last_activity: chrono::Utc::now(),
            },
        }
    }

    /// Validate resource optimization task input
    fn validate_input(&self, input: &ResourceOptimizationTaskInput) -> AgentResult<()> {
        if input.current_allocation.is_empty() {
            return Err(crate::shared::agent_types::AgentError::InvalidInput(
                "Current allocation cannot be empty".to_string()
            ));
        }
        
        if input.resource_demands.is_empty() {
            return Err(crate::shared::agent_types::AgentError::InvalidInput(
                "Resource demands cannot be empty".to_string()
            ));
        }
        
        Ok(())
    }

    /// Analyze current resource allocation
    async fn analyze_current_allocation(&self, input: &ResourceOptimizationTaskInput) -> AgentResult<AllocationAnalysis> {
        let mut utilization_rates = HashMap::new();
        let mut allocation_efficiency = HashMap::new();
        
        for (agent_id, allocation) in &input.current_allocation {
            let utilization_rate = allocation.utilization_percentage / 100.0;
            utilization_rates.insert(agent_id.clone(), utilization_rate);
            
            // Calculate allocation efficiency based on utilization
            let efficiency = if utilization_rate > 0.8 {
                0.9 // High utilization is good
            } else if utilization_rate > 0.5 {
                0.7 // Moderate utilization
            } else {
                0.4 // Low utilization is inefficient
            };
            
            allocation_efficiency.insert(agent_id.clone(), efficiency);
        }
        
        let overall_utilization = utilization_rates.values().sum::<f32>() / utilization_rates.len() as f32;
        let overall_efficiency = allocation_efficiency.values().sum::<f32>() / allocation_efficiency.len() as f32;
        
        Ok(AllocationAnalysis {
            utilization_rates,
            allocation_efficiency,
            overall_utilization,
            overall_efficiency,
        })
    }

    /// Optimize resource allocation
    async fn optimize_resource_allocation(&self, input: &ResourceOptimizationTaskInput,
                                       allocation_analysis: &AllocationAnalysis) -> AgentResult<OptimizationResult> {
        let mut optimized_allocation = HashMap::new();
        let mut total_improvement = 0.0;
        
        // Sort demands by priority
        let mut sorted_demands = input.resource_demands.clone();
        sorted_demands.sort_by(|a, b| a.priority_level.cmp(&b.priority_level));
        
        // Allocate resources based on strategy
        match &self.config.optimization_strategy {
            OptimizationStrategy::BalancedOptimization { weights } => {
                for demand in &sorted_demands {
                    let current_allocation = input.current_allocation.get(&demand.agent_id);
                    let optimized_amount = self.calculate_balanced_allocation(demand, current_allocation, weights);
                    
                    optimized_allocation.insert(demand.agent_id.clone(), ResourceAllocation {
                        agent_id: demand.agent_id.clone(),
                        resource_type: demand.resource_type.clone(),
                        allocated_amount: optimized_amount,
                        utilization_percentage: 85.0, // Target utilization
                        allocation_timestamp: chrono::Utc::now(),
                        expiration_timestamp: Some(chrono::Utc::now() + chrono::Duration::seconds(demand.duration_seconds as i64)),
                    });
                }
            },
            OptimizationStrategy::CostMinimization => {
                // Implement cost minimization logic
                for demand in &sorted_demands {
                    let optimized_amount = self.calculate_cost_minimized_allocation(demand);
                    
                    optimized_allocation.insert(demand.agent_id.clone(), ResourceAllocation {
                        agent_id: demand.agent_id.clone(),
                        resource_type: demand.resource_type.clone(),
                        allocated_amount: optimized_amount,
                        utilization_percentage: 90.0, // Higher utilization for cost efficiency
                        allocation_timestamp: chrono::Utc::now(),
                        expiration_timestamp: Some(chrono::Utc::now() + chrono::Duration::seconds(demand.duration_seconds as i64)),
                    });
                }
            },
            OptimizationStrategy::PerformanceMaximization => {
                // Implement performance maximization logic
                for demand in &sorted_demands {
                    let optimized_amount = self.calculate_performance_maximized_allocation(demand);
                    
                    optimized_allocation.insert(demand.agent_id.clone(), ResourceAllocation {
                        agent_id: demand.agent_id.clone(),
                        resource_type: demand.resource_type.clone(),
                        allocated_amount: optimized_amount,
                        utilization_percentage: 70.0, // Lower utilization for better performance
                        allocation_timestamp: chrono::Utc::now(),
                        expiration_timestamp: Some(chrono::Utc::now() + chrono::Duration::seconds(demand.duration_seconds as i64)),
                    });
                }
            },
            _ => {
                // Default to current allocation
                optimized_allocation = input.current_allocation.clone();
            }
        }
        
        // Calculate optimization score
        let optimization_score = self.calculate_optimization_score(&optimized_allocation, allocation_analysis);
        
        Ok(OptimizationResult {
            optimized_allocation,
            results: OptimizationResults {
                success: true,
                optimization_score,
                resource_utilization_improvement: (optimization_score - allocation_analysis.overall_efficiency) * 100.0,
                performance_improvement: optimization_score * 15.0, // Simplified calculation
                cost_savings: optimization_score * 10.0, // Simplified calculation
                execution_time_ms: 500, // Simplified
            },
        })
    }

    /// Calculate performance improvements
    async fn calculate_performance_improvements(&self, _input: &ResourceOptimizationTaskInput,
                                              optimization_result: &OptimizationResult) -> AgentResult<PerformanceImprovements> {
        let improvement_factor = optimization_result.results.optimization_score;
        
        Ok(PerformanceImprovements {
            latency_improvement: improvement_factor * 20.0, // 20% latency improvement
            throughput_improvement: improvement_factor * 25.0, // 25% throughput improvement
            resource_efficiency_improvement: optimization_result.results.resource_utilization_improvement,
            quality_improvement: improvement_factor * 10.0, // 10% quality improvement
            user_experience_improvement: improvement_factor * 15.0, // 15% UX improvement
        })
    }

    /// Perform cost analysis
    async fn perform_cost_analysis(&self, input: &ResourceOptimizationTaskInput,
                                 optimization_result: &OptimizationResult) -> AgentResult<CostAnalysis> {
        // Calculate current cost (simplified)
        let current_cost_per_hour = input.current_allocation.values()
            .map(|alloc| alloc.allocated_amount * 0.01) // $0.01 per unit
            .sum();
        
        // Calculate optimized cost
        let optimized_cost_per_hour = optimization_result.optimized_allocation.values()
            .map(|alloc| alloc.allocated_amount * 0.01)
            .sum();
        
        let cost_savings = current_cost_per_hour - optimized_cost_per_hour;
        let cost_savings_percentage = if current_cost_per_hour > 0.0 {
            (cost_savings / current_cost_per_hour) * 100.0
        } else {
            0.0
        };
        
        // Create cost breakdown
        let mut cost_breakdown = HashMap::new();
        cost_breakdown.insert("CPU".to_string(), optimized_cost_per_hour * 0.4);
        cost_breakdown.insert("Memory".to_string(), optimized_cost_per_hour * 0.3);
        cost_breakdown.insert("Storage".to_string(), optimized_cost_per_hour * 0.2);
        cost_breakdown.insert("Network".to_string(), optimized_cost_per_hour * 0.1);
        
        // Calculate ROI
        let roi_calculation = RoiCalculation {
            investment_cost: 1000.0, // Implementation cost
            expected_savings_per_month: cost_savings * 24.0 * 30.0,
            payback_period_months: 1000.0 / (cost_savings * 24.0 * 30.0),
            annual_roi_percentage: (cost_savings * 24.0 * 365.0 / 1000.0) * 100.0,
        };
        
        Ok(CostAnalysis {
            current_cost_per_hour,
            optimized_cost_per_hour,
            cost_savings_percentage,
            cost_breakdown,
            roi_calculation,
        })
    }

    /// Generate optimization recommendations
    async fn generate_recommendations(&self, input: &ResourceOptimizationTaskInput,
                                    optimization_result: &OptimizationResult,
                                    performance_improvements: &PerformanceImprovements) -> AgentResult<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();
        
        // Resource scaling recommendation
        if optimization_result.results.resource_utilization_improvement > 10.0 {
            recommendations.push(OptimizationRecommendation {
                recommendation_id: "rec_001".to_string(),
                recommendation_type: RecommendationType::ResourceScaling,
                description: "Scale resources based on demand patterns".to_string(),
                expected_impact: ExpectedImpact {
                    performance_impact: performance_improvements.throughput_improvement,
                    cost_impact: -5.0, // 5% cost reduction
                    resource_impact: 15.0, // 15% resource efficiency
                    quality_impact: 5.0, // 5% quality improvement
                    time_to_implement_days: 7,
                },
                implementation_complexity: ImplementationComplexity::Medium,
                priority_level: 1,
            });
        }
        
        // Load balancing recommendation
        if performance_improvements.latency_improvement > 15.0 {
            recommendations.push(OptimizationRecommendation {
                recommendation_id: "rec_002".to_string(),
                recommendation_type: RecommendationType::LoadBalancing,
                description: "Implement intelligent load balancing".to_string(),
                expected_impact: ExpectedImpact {
                    performance_impact: performance_improvements.latency_improvement,
                    cost_impact: 2.0, // 2% cost increase
                    resource_impact: 10.0, // 10% resource efficiency
                    quality_impact: 8.0, // 8% quality improvement
                    time_to_implement_days: 14,
                },
                implementation_complexity: ImplementationComplexity::High,
                priority_level: 2,
            });
        }
        
        // Cost optimization recommendation
        if optimization_result.results.cost_savings > 50.0 {
            recommendations.push(OptimizationRecommendation {
                recommendation_id: "rec_003".to_string(),
                recommendation_type: RecommendationType::CostOptimization,
                description: "Optimize resource scheduling for cost efficiency".to_string(),
                expected_impact: ExpectedImpact {
                    performance_impact: 5.0, // 5% performance impact
                    cost_impact: -15.0, // 15% cost reduction
                    resource_impact: 20.0, // 20% resource efficiency
                    quality_impact: 2.0, // 2% quality impact
                    time_to_implement_days: 3,
                },
                implementation_complexity: ImplementationComplexity::Low,
                priority_level: 3,
            });
        }
        
        Ok(recommendations)
    }

    /// Calculate balanced allocation
    fn calculate_balanced_allocation(&self, demand: &ResourceDemand, 
                                  current_allocation: Option<&ResourceAllocation>,
                                  weights: &OptimizationWeights) -> f32 {
        let base_amount = demand.required_amount;
        let current_amount = current_allocation.map(|alloc| alloc.allocated_amount).unwrap_or(0.0);
        
        // Calculate weighted score
        let performance_factor = weights.performance_weight;
        let cost_factor = weights.cost_weight;
        let efficiency_factor = weights.efficiency_weight;
        
        // Adjust based on priority
        let priority_multiplier = match demand.priority_level {
            1 => 1.2, // High priority gets 20% more
            2 => 1.0, // Normal priority
            3 => 0.8, // Low priority gets 20% less
            _ => 1.0,
        };
        
        let adjusted_amount = base_amount * priority_multiplier * (performance_factor + cost_factor + efficiency_factor);
        
        // Ensure we don't exceed current allocation by too much
        if current_amount > 0.0 {
            adjusted_amount.min(current_amount * 1.5).max(current_amount * 0.5)
        } else {
            adjusted_amount
        }
    }

    /// Calculate cost minimized allocation
    fn calculate_cost_minimized_allocation(&self, demand: &ResourceDemand) -> f32 {
        // For cost minimization, allocate the minimum required
        demand.required_amount * 0.9 // 10% buffer
    }

    /// Calculate performance maximized allocation
    fn calculate_performance_maximized_allocation(&self, demand: &ResourceDemand) -> f32 {
        // For performance maximization, allocate more resources
        demand.required_amount * 1.3 // 30% extra for performance
    }

    /// Calculate optimization score
    fn calculate_optimization_score(&self, _optimized_allocation: &HashMap<String, ResourceAllocation>,
                                   allocation_analysis: &AllocationAnalysis) -> f32 {
        // Simplified optimization score calculation
        let utilization_score = allocation_analysis.overall_utilization;
        let efficiency_score = allocation_analysis.overall_efficiency;
        
        (utilization_score + efficiency_score) / 2.0
    }
}

// Helper structs for internal processing
#[derive(Debug, Clone)]
struct AllocationAnalysis {
    utilization_rates: HashMap<String, f32>,
    allocation_efficiency: HashMap<String, f32>,
    overall_utilization: f32,
    overall_efficiency: f32,
}

#[derive(Debug, Clone)]
struct OptimizationResult {
    optimized_allocation: HashMap<String, ResourceAllocation>,
    results: OptimizationResults,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_optimizer_agent_creation() {
        let agent = ResourceOptimizerAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_resource_optimization_task_processing() {
        let agent = ResourceOptimizerAgent::default();
        let input = ResourceOptimizationTaskInput {
            current_allocation: HashMap::from([
                ("agent1".to_string(), ResourceAllocation {
                    agent_id: "agent1".to_string(),
                    resource_type: ResourceType::CPU,
                    allocated_amount: 4.0,
                    utilization_percentage: 60.0,
                    allocation_timestamp: chrono::Utc::now(),
                    expiration_timestamp: None,
                }),
            ]),
            resource_demands: vec![
                ResourceDemand {
                    agent_id: "agent1".to_string(),
                    resource_type: ResourceType::CPU,
                    required_amount: 6.0,
                    priority_level: 1,
                    duration_seconds: 3600,
                    flexible_allocation: true,
                },
            ],
            performance_metrics: HashMap::new(),
            optimization_constraints: OptimizationConstraints {
                resource_constraints: ResourceConstraints::default(),
                time_constraints: TimeConstraints {
                    optimization_window_seconds: 300,
                    max_execution_time_seconds: 60,
                    deadline: None,
                    frequency_constraints: FrequencyConstraints {
                        min_interval_seconds: 300,
                        max_optimizations_per_hour: 12,
                        peak_hour_restrictions: false,
                    },
                },
                budget_constraints: BudgetConstraints::default(),
                policy_constraints: vec![],
            },
            optimization_objectives: vec![],
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(!output.optimized_allocation.is_empty());
        assert!(output.optimization_results.success);
        assert!(output.optimization_results.optimization_score > 0.0);
    }

    #[test]
    fn test_balanced_allocation_calculation() {
        let agent = ResourceOptimizerAgent::default();
        let weights = OptimizationWeights {
            performance_weight: 0.4,
            cost_weight: 0.3,
            efficiency_weight: 0.2,
            reliability_weight: 0.05,
            scalability_weight: 0.05,
        };
        
        let demand = ResourceDemand {
            agent_id: "agent1".to_string(),
            resource_type: ResourceType::CPU,
            required_amount: 8.0,
            priority_level: 1,
            duration_seconds: 3600,
            flexible_allocation: true,
        };
        
        let allocation = agent.calculate_balanced_allocation(&demand, None, &weights);
        assert!(allocation > 0.0);
        assert!(allocation >= demand.required_amount);
    }

    #[test]
    fn test_optimization_strategies() {
        let config = ResourceOptimizerConfig {
            optimization_strategy: OptimizationStrategy::CostMinimization,
            ..Default::default()
        };
        let agent = ResourceOptimizerAgent::new(config);
        
        assert!(matches!(agent.config.optimization_strategy, OptimizationStrategy::CostMinimization));
    }
}
