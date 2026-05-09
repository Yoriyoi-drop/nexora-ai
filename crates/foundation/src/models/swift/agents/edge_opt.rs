//! Edge Opt Agent
//! 
//! Runtime optimization for dynamic edge conditions

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// Edge Opt Agent - Runtime optimization for edge conditions
#[derive(Debug, Clone)]
pub struct EdgeOptAgent {
    pub config: EdgeOptConfig,
    pub resource_monitor: ResourceMonitor,
    pub optimization_engine: OptimizationEngine,
    pub adaptation_strategy: AdaptationStrategy,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeOptConfig {
    pub base_config: BaseAgentConfig,
    pub optimization_mode: OptimizationMode,
    pub adaptation_frequency_hz: f32,
    pub resource_constraints: ResourceConstraints,
    pub performance_targets: PerformanceTargets,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationMode {
    /// Balance between performance and resource usage
    Balanced,
    /// Maximum performance regardless of resource usage
    Performance,
    /// Maximum resource efficiency
    Efficiency,
    /// Adaptive based on conditions
    Adaptive,
    /// Battery optimization priority
    Battery,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    pub max_memory_mb: u32,
    pub max_cpu_percent: f32,
    pub max_power_mw: u32,
    pub max_temperature_celsius: f32,
    pub thermal_throttling_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    pub target_latency_ms: u32,
    pub target_throughput_ops_per_sec: f64,
    pub target_accuracy: f32,
    pub target_energy_efficiency: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMonitor {
    pub hardware_state: HardwareState,
    pub monitoring_interval_ms: u32,
    pub alert_thresholds: AlertThresholds,
    pub historical_data: Vec<HardwareSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareState {
    pub cpu_usage_percent: f32,
    pub memory_usage_mb: f32,
    pub battery_level_percent: Option<f32>,
    pub thermal_state_celsius: f32,
    pub network_connectivity: NetworkState,
    pub power_consumption_mw: f32,
    pub cpu_frequency_ghz: f32,
    pub gpu_utilization: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkState {
    Connected { bandwidth_mbps: f32, latency_ms: u32 },
    Disconnected,
    Poor { signal_strength: f32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub cpu_warning: f32,
    pub cpu_critical: f32,
    pub memory_warning_mb: f32,
    pub memory_critical_mb: f32,
    pub thermal_warning_celsius: f32,
    pub thermal_critical_celsius: f32,
    pub battery_warning_percent: f32,
    pub battery_critical_percent: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSnapshot {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub state: HardwareState,
    pub performance_metrics: PerformanceSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub latency_ms: f32,
    pub throughput_ops_per_sec: f64,
    pub accuracy: f32,
    pub energy_efficiency: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationEngine {
    pub optimization_algorithms: Vec<String>,
    pub decision_matrix: DecisionMatrix,
    pub optimization_history: Vec<OptimizationDecision>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionMatrix {
    pub weight_cpu: f32,
    pub weight_memory: f32,
    pub weight_battery: f32,
    pub weight_thermal: f32,
    pub weight_performance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationDecision {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub hardware_state: HardwareState,
    pub optimization_action: OptimizationAction,
    pub expected_impact: ImpactPrediction,
    pub actual_impact: Option<ImpactMeasurement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationAction {
    /// Adjust CPU frequency scaling
    AdjustCpuFrequency { target_ghz: f32 },
    /// Enable/disable CPU cores
    ManageCpuCores { active_cores: u32 },
    /// Adjust memory allocation
    OptimizeMemory { target_mb: u32, strategy: MemoryStrategy },
    /// Enable thermal throttling
    EnableThermalThrottling { max_temp_celsius: f32 },
    /// Switch to battery saver mode
    EnableBatterySaver { aggressiveness: BatterySaverLevel },
    /// Adjust model complexity
    AdaptModelComplexity { complexity_level: ModelComplexity },
    /// Optimize inference pipeline
    OptimizePipeline { optimization_level: PipelineOptimization },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryStrategy {
    Aggressive,
    Conservative,
    Balanced,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatterySaverLevel {
    Low,
    Medium,
    High,
    Maximum,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelComplexity {
    Minimal,
    Low,
    Medium,
    High,
    Maximum,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineOptimization {
    Minimal,
    Standard,
    Aggressive,
    Maximum,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactPrediction {
    pub latency_change_percent: f32,
    pub throughput_change_percent: f32,
    pub accuracy_change_percent: f32,
    pub power_change_percent: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactMeasurement {
    pub latency_change_percent: f32,
    pub throughput_change_percent: f32,
    pub accuracy_change_percent: f32,
    pub power_change_percent: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationStrategy {
    pub strategy_type: AdaptationType,
    pub adaptation_rate: AdaptationRate,
    pub stability_margin: f32,
    pub prediction_horizon_seconds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationType {
    /// Reactive - respond to current conditions
    Reactive,
    /// Proactive - predict and prepare for future conditions
    Proactive,
    /// Hybrid - combination of reactive and proactive
    Hybrid,
    /// Learning - improve over time based on history
    Learning,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationRate {
    Slow { interval_seconds: u32 },
    Medium { interval_seconds: u32 },
    Fast { interval_seconds: u32 },
    Dynamic { min_seconds: u32, max_seconds: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeOptTaskInput {
    pub current_hardware_state: HardwareState,
    pub performance_requirements: PerformanceTargets,
    pub optimization_constraints: ResourceConstraints,
    pub adaptation_request: Option<AdaptationRequest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationRequest {
    pub urgency: UrgencyLevel,
    pub optimization_goals: Vec<OptimizationGoal>,
    pub time_constraint_seconds: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UrgencyLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationGoal {
    MinimizeLatency,
    MaximizeThroughput,
    PreserveBattery,
    ReduceThermal,
    MaintainAccuracy,
    BalanceResources,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeOptTaskOutput {
    pub optimization_plan: OptimizationPlan,
    pub resource_assessment: ResourceAssessment,
    pub performance_forecast: PerformanceForecast,
    pub adaptation_recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationPlan {
    pub primary_action: OptimizationAction,
    pub secondary_actions: Vec<OptimizationAction>,
    pub expected_impact: ImpactPrediction,
    pub implementation_time_ms: u32,
    pub rollback_plan: Option<OptimizationAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAssessment {
    pub current_utilization: ResourceUtilization,
    pub bottleneck_analysis: BottleneckAnalysis,
    pub resource_trends: ResourceTrends,
    pub capacity_remaining: CapacityRemaining,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub cpu_utilization_percent: f32,
    pub memory_utilization_percent: f32,
    pub power_utilization_percent: f32,
    pub thermal_utilization_percent: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    pub primary_bottleneck: Option<BottleneckType>,
    pub secondary_bottlenecks: Vec<BottleneckType>,
    pub bottleneck_severity: f32,
    pub mitigation_suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    CPU,
    Memory,
    Power,
    Thermal,
    Network,
    Storage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceTrends {
    pub cpu_trend: TrendDirection,
    pub memory_trend: TrendDirection,
    pub power_trend: TrendDirection,
    pub thermal_trend: TrendDirection,
    pub trend_confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityRemaining {
    pub cpu_capacity_percent: f32,
    pub memory_capacity_mb: f32,
    pub power_capacity_mw: f32,
    pub thermal_headroom_celsius: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceForecast {
    pub predicted_latency_ms: f32,
    pub predicted_throughput_ops_per_sec: f64,
    pub predicted_accuracy: f32,
    pub predicted_energy_efficiency: f32,
    pub forecast_confidence: f32,
    pub time_horizon_seconds: u32,
}

impl Default for EdgeOptConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            optimization_mode: OptimizationMode::Adaptive,
            adaptation_frequency_hz: 1.0,
            resource_constraints: ResourceConstraints {
                max_memory_mb: 512,
                max_cpu_percent: 80.0,
                max_power_mw: 2000,
                max_temperature_celsius: 85.0,
                thermal_throttling_enabled: true,
            },
            performance_targets: PerformanceTargets {
                target_latency_ms: 5,
                target_throughput_ops_per_sec: 1000.0,
                target_accuracy: 0.9,
                target_energy_efficiency: 0.8,
            },
        }
    }
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self {
            hardware_state: HardwareState::default(),
            monitoring_interval_ms: 1000,
            alert_thresholds: AlertThresholds::default(),
            historical_data: Vec::new(),
        }
    }
}

impl Default for HardwareState {
    fn default() -> Self {
        Self {
            cpu_usage_percent: 20.0,
            memory_usage_mb: 128.0,
            battery_level_percent: Some(80.0),
            thermal_state_celsius: 45.0,
            network_connectivity: NetworkState::Connected { bandwidth_mbps: 100.0, latency_ms: 50 },
            power_consumption_mw: 500.0,
            cpu_frequency_ghz: 2.0,
            gpu_utilization: None,
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_warning: 70.0,
            cpu_critical: 90.0,
            memory_warning_mb: 400.0,
            memory_critical_mb: 480.0,
            thermal_warning_celsius: 75.0,
            thermal_critical_celsius: 85.0,
            battery_warning_percent: 20.0,
            battery_critical_percent: 10.0,
        }
    }
}

impl Default for OptimizationEngine {
    fn default() -> Self {
        Self {
            optimization_algorithms: vec![
                "gradient_descent".to_string(),
                "genetic_algorithm".to_string(),
                "reinforcement_learning".to_string(),
                "rule_based".to_string(),
            ],
            decision_matrix: DecisionMatrix::default(),
            optimization_history: Vec::new(),
        }
    }
}

impl Default for DecisionMatrix {
    fn default() -> Self {
        Self {
            weight_cpu: 0.3,
            weight_memory: 0.25,
            weight_battery: 0.2,
            weight_thermal: 0.15,
            weight_performance: 0.1,
        }
    }
}

impl Default for AdaptationStrategy {
    fn default() -> Self {
        Self {
            strategy_type: AdaptationType::Hybrid,
            adaptation_rate: AdaptationRate::Dynamic { min_seconds: 1, max_seconds: 10 },
            stability_margin: 0.1,
            prediction_horizon_seconds: 30,
        }
    }
}

impl Default for EdgeOptAgent {
    fn default() -> Self {
        Self {
            config: EdgeOptConfig::default(),
            resource_monitor: ResourceMonitor::default(),
            optimization_engine: OptimizationEngine::default(),
            adaptation_strategy: AdaptationStrategy::default(),
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
impl BaseAgent for EdgeOptAgent {
    type Config = EdgeOptConfig;
    type Input = EdgeOptTaskInput;
    type Output = EdgeOptTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        // Analyze current resource state
        let resource_assessment = self.assess_resources(&input).await?;
        
        // Generate optimization plan
        let optimization_plan = self.generate_optimization_plan(&input, &resource_assessment).await?;
        
        // Forecast performance impact
        let performance_forecast = self.forecast_performance(&optimization_plan, &input).await?;
        
        // Generate adaptation recommendations
        let adaptation_recommendations = self.generate_adaptation_recommendations(&resource_assessment, &optimization_plan).await?;

        Ok(EdgeOptTaskOutput {
            optimization_plan,
            resource_assessment,
            performance_forecast,
            adaptation_recommendations,
        })
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
                name: "edge_opt".to_string(),
                description: "Runtime optimization for dynamic edge conditions".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["hardware_state".to_string(), "performance_requirements".to_string()],
                output_types: vec!["optimization_plan".to_string(), "resource_assessment".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.94,
                    avg_latency: 2.0,
                    resource_usage: 0.5,
                    reliability: 0.96,
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

impl EdgeOptAgent {
    pub fn new(config: EdgeOptConfig) -> Self {
        Self {
            config,
            resource_monitor: ResourceMonitor::default(),
            optimization_engine: OptimizationEngine::default(),
            adaptation_strategy: AdaptationStrategy::default(),
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

    async fn assess_resources(&self, input: &EdgeOptTaskInput) -> AgentResult<ResourceAssessment> {
        let current_utilization = ResourceUtilization {
            cpu_utilization_percent: input.current_hardware_state.cpu_usage_percent,
            memory_utilization_percent: (input.current_hardware_state.memory_usage_mb / input.optimization_constraints.max_memory_mb as f32) * 100.0,
            power_utilization_percent: (input.current_hardware_state.power_consumption_mw / input.optimization_constraints.max_power_mw as f32) * 100.0,
            thermal_utilization_percent: (input.current_hardware_state.thermal_state_celsius / input.optimization_constraints.max_temperature_celsius) * 100.0,
        };

        let primary_bottleneck = self.identify_primary_bottleneck(&current_utilization);
        let secondary_bottlenecks = self.identify_secondary_bottlenecks(&current_utilization);

        let bottleneck_analysis = BottleneckAnalysis {
            primary_bottleneck,
            secondary_bottlenecks,
            bottleneck_severity: self.calculate_bottleneck_severity(&current_utilization),
            mitigation_suggestions: self.generate_mitigation_suggestions(&primary_bottleneck, &secondary_bottlenecks),
        };

        let resource_trends = self.analyze_resource_trends(&input.current_hardware_state);
        let capacity_remaining = self.calculate_capacity_remaining(&input.current_hardware_state, &input.optimization_constraints);

        Ok(ResourceAssessment {
            current_utilization,
            bottleneck_analysis,
            resource_trends,
            capacity_remaining,
        })
    }

    fn identify_primary_bottleneck(&self, utilization: &ResourceUtilization) -> Option<BottleneckType> {
        let thresholds = vec![
            (utilization.cpu_utilization_percent, BottleneckType::CPU),
            (utilization.memory_utilization_percent, BottleneckType::Memory),
            (utilization.power_utilization_percent, BottleneckType::Power),
            (utilization.thermal_utilization_percent, BottleneckType::Thermal),
        ];

        thresholds.into_iter()
            .filter(|(percent, _)| *percent > 80.0)
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, bottleneck_type)| bottleneck_type)
    }

    fn identify_secondary_bottlenecks(&self, utilization: &ResourceUtilization) -> Vec<BottleneckType> {
        let thresholds = vec![
            (utilization.cpu_utilization_percent, BottleneckType::CPU),
            (utilization.memory_utilization_percent, BottleneckType::Memory),
            (utilization.power_utilization_percent, BottleneckType::Power),
            (utilization.thermal_utilization_percent, BottleneckType::Thermal),
        ];

        thresholds.into_iter()
            .filter(|(percent, _)| *percent > 60.0 && *percent <= 80.0)
            .map(|(_, bottleneck_type)| bottleneck_type)
            .collect()
    }

    fn calculate_bottleneck_severity(&self, utilization: &ResourceUtilization) -> f32 {
        let max_utilization = utilization.cpu_utilization_percent
            .max(utilization.memory_utilization_percent)
            .max(utilization.power_utilization_percent)
            .max(utilization.thermal_utilization_percent);
        
        (max_utilization - 80.0).max(0.0) / 20.0 // Normalize to 0-1 range
    }

    fn generate_mitigation_suggestions(&self, primary: &Option<BottleneckType>, secondary: &[BottleneckType]) -> Vec<String> {
        let mut suggestions = Vec::new();

        if let Some(bottleneck) = primary {
            suggestions.push(match bottleneck {
                BottleneckType::CPU => "Reduce CPU frequency or enable parallel processing".to_string(),
                BottleneckType::Memory => "Implement memory pooling or reduce model size".to_string(),
                BottleneckType::Power => "Enable battery saver mode or reduce processing intensity".to_string(),
                BottleneckType::Thermal => "Enable thermal throttling or reduce workload".to_string(),
                _ => "Monitor and optimize resource usage".to_string(),
            });
        }

        for bottleneck in secondary {
            suggestions.push(match bottleneck {
                BottleneckType::CPU => "Consider CPU optimization strategies".to_string(),
                BottleneckType::Memory => "Monitor memory usage patterns".to_string(),
                BottleneckType::Power => "Optimize for power efficiency".to_string(),
                BottleneckType::Thermal => "Monitor thermal trends".to_string(),
                _ => "General resource monitoring".to_string(),
            });
        }

        suggestions
    }

    fn analyze_resource_trends(&self, _state: &HardwareState) -> ResourceTrends {
        // Simplified trend analysis - in real implementation, would use historical data
        ResourceTrends {
            cpu_trend: TrendDirection::Stable,
            memory_trend: TrendDirection::Stable,
            power_trend: TrendDirection::Stable,
            thermal_trend: TrendDirection::Stable,
            trend_confidence: 0.5,
        }
    }

    fn calculate_capacity_remaining(&self, state: &HardwareState, constraints: &ResourceConstraints) -> CapacityRemaining {
        CapacityRemaining {
            cpu_capacity_percent: (constraints.max_cpu_percent - state.cpu_usage_percent).max(0.0),
            memory_capacity_mb: (constraints.max_memory_mb as f32 - state.memory_usage_mb).max(0.0),
            power_capacity_mw: (constraints.max_power_mw as f32 - state.power_consumption_mw).max(0.0),
            thermal_headroom_celsius: (constraints.max_temperature_celsius - state.thermal_state_celsius).max(0.0),
        }
    }

    async fn generate_optimization_plan(&self, input: &EdgeOptTaskInput, assessment: &ResourceAssessment) -> AgentResult<OptimizationPlan> {
        let primary_action = self.determine_primary_optimization_action(input, assessment);
        let secondary_actions = self.determine_secondary_optimization_actions(input, assessment);
        let expected_impact = self.predict_optimization_impact(&primary_action, assessment);
        let implementation_time_ms = self.estimate_implementation_time(&primary_action);
        let rollback_plan = self.generate_rollback_plan(&primary_action);

        Ok(OptimizationPlan {
            primary_action,
            secondary_actions,
            expected_impact,
            implementation_time_ms,
            rollback_plan,
        })
    }

    fn determine_primary_optimization_action(&self, input: &EdgeOptTaskInput, assessment: &ResourceAssessment) -> OptimizationAction {
        if let Some(bottleneck) = &assessment.bottleneck_analysis.primary_bottleneck {
            match bottleneck {
                BottleneckType::CPU => OptimizationAction::AdjustCpuFrequency { target_ghz: 1.5 },
                BottleneckType::Memory => OptimizationAction::OptimizeMemory { 
                    target_mb: 256, 
                    strategy: MemoryStrategy::Balanced 
                },
                BottleneckType::Power => OptimizationAction::EnableBatterySaver { 
                    aggressiveness: BatterySaverLevel::Medium 
                },
                BottleneckType::Thermal => OptimizationAction::EnableThermalThrottling { 
                    max_temp_celsius: 75.0 
                },
                _ => OptimizationAction::AdaptModelComplexity { 
                    complexity_level: ModelComplexity::Medium 
                },
            }
        } else {
            OptimizationAction::AdaptModelComplexity { 
                complexity_level: ModelComplexity::High 
            }
        }
    }

    fn determine_secondary_optimization_actions(&self, _input: &EdgeOptTaskInput, assessment: &ResourceAssessment) -> Vec<OptimizationAction> {
        let mut actions = Vec::new();

        for bottleneck in &assessment.bottleneck_analysis.secondary_bottlenecks {
            actions.push(match bottleneck {
                BottleneckType::CPU => OptimizationAction::ManageCpuCores { active_cores: 2 },
                BottleneckType::Memory => OptimizationAction::OptimizeMemory { 
                    target_mb: 200, 
                    strategy: MemoryStrategy::Conservative 
                },
                BottleneckType::Power => OptimizationAction::EnableBatterySaver { 
                    aggressiveness: BatterySaverLevel::Low 
                },
                BottleneckType::Thermal => OptimizationAction::EnableThermalThrottling { 
                    max_temp_celsius: 80.0 
                },
                _ => continue,
            });
        }

        actions
    }

    fn predict_optimization_impact(&self, action: &OptimizationAction, _assessment: &ResourceAssessment) -> ImpactPrediction {
        match action {
            OptimizationAction::AdjustCpuFrequency { .. } => ImpactPrediction {
                latency_change_percent: -15.0,
                throughput_change_percent: 20.0,
                accuracy_change_percent: -2.0,
                power_change_percent: -10.0,
                confidence: 0.8,
            },
            OptimizationAction::OptimizeMemory { .. } => ImpactPrediction {
                latency_change_percent: -5.0,
                throughput_change_percent: 10.0,
                accuracy_change_percent: 0.0,
                power_change_percent: -5.0,
                confidence: 0.7,
            },
            OptimizationAction::EnableBatterySaver { .. } => ImpactPrediction {
                latency_change_percent: 10.0,
                throughput_change_percent: -20.0,
                accuracy_change_percent: -5.0,
                power_change_percent: -30.0,
                confidence: 0.9,
            },
            _ => ImpactPrediction {
                latency_change_percent: 0.0,
                throughput_change_percent: 0.0,
                accuracy_change_percent: 0.0,
                power_change_percent: 0.0,
                confidence: 0.5,
            },
        }
    }

    fn estimate_implementation_time(&self, action: &OptimizationAction) -> u32 {
        match action {
            OptimizationAction::AdjustCpuFrequency { .. } => 100,
            OptimizationAction::OptimizeMemory { .. } => 500,
            OptimizationAction::EnableBatterySaver { .. } => 200,
            OptimizationAction::EnableThermalThrottling { .. } => 50,
            _ => 1000,
        }
    }

    fn generate_rollback_plan(&self, action: &OptimizationAction) -> Option<OptimizationAction> {
        match action {
            OptimizationAction::AdjustCpuFrequency { .. } => Some(OptimizationAction::AdjustCpuFrequency { target_ghz: 2.0 }),
            OptimizationAction::OptimizeMemory { .. } => Some(OptimizationAction::OptimizeMemory { 
                target_mb: 512, 
                strategy: MemoryStrategy::Balanced 
            }),
            OptimizationAction::EnableBatterySaver { .. } => Some(OptimizationAction::EnableBatterySaver { 
                aggressiveness: BatterySaverLevel::Low 
            }),
            _ => None,
        }
    }

    async fn forecast_performance(&self, plan: &OptimizationPlan, input: &EdgeOptTaskInput) -> AgentResult<PerformanceForecast> {
        let base_latency = input.performance_requirements.target_latency_ms as f32;
        let base_throughput = input.performance_requirements.target_throughput_ops_per_sec;
        let base_accuracy = input.performance_requirements.target_accuracy;
        let base_efficiency = input.performance_requirements.target_energy_efficiency;

        let impact = &plan.expected_impact;

        Ok(PerformanceForecast {
            predicted_latency_ms: base_latency * (1.0 + impact.latency_change_percent / 100.0),
            predicted_throughput_ops_per_sec: base_throughput * (1.0 + impact.throughput_change_percent / 100.0),
            predicted_accuracy: base_accuracy * (1.0 + impact.accuracy_change_percent / 100.0),
            predicted_energy_efficiency: base_efficiency * (1.0 + impact.power_change_percent / 100.0),
            forecast_confidence: impact.confidence,
            time_horizon_seconds: 60,
        })
    }

    async fn generate_adaptation_recommendations(&self, assessment: &ResourceAssessment, plan: &OptimizationPlan) -> AgentResult<Vec<String>> {
        let mut recommendations = Vec::new();

        // Add recommendations based on bottlenecks
        if let Some(bottleneck) = &assessment.bottleneck_analysis.primary_bottleneck {
            recommendations.push(match bottleneck {
                BottleneckType::CPU => "Consider upgrading CPU or implementing better parallel processing".to_string(),
                BottleneckType::Memory => "Memory pressure detected - consider memory optimization techniques".to_string(),
                BottleneckType::Power => "High power consumption - battery optimization recommended".to_string(),
                BottleneckType::Thermal => "Thermal throttling may be needed - monitor temperature closely".to_string(),
                _ => "Monitor system resources continuously".to_string(),
            });
        }

        // Add recommendations based on capacity
        if assessment.capacity_remaining.cpu_capacity_percent < 20.0 {
            recommendations.push("Low CPU capacity remaining - consider load balancing".to_string());
        }

        if assessment.capacity_remaining.memory_capacity_mb < 100.0 {
            recommendations.push("Low memory capacity - implement memory cleanup".to_string());
        }

        // Add optimization-specific recommendations
        recommendations.push(format!("Primary optimization: {:?}", plan.primary_action));
        
        for (i, action) in plan.secondary_actions.iter().enumerate() {
            recommendations.push(format!("Secondary optimization {}: {:?}", i + 1, action));
        }

        Ok(recommendations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_opt_agent_creation() {
        let agent = EdgeOptAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_resource_assessment() {
        let agent = EdgeOptAgent::default();
        let input = EdgeOptTaskInput {
            current_hardware_state: HardwareState {
                cpu_usage_percent: 85.0,
                memory_usage_mb: 400.0,
                battery_level_percent: Some(50.0),
                thermal_state_celsius: 80.0,
                network_connectivity: NetworkState::Connected { bandwidth_mbps: 100.0, latency_ms: 50 },
                power_consumption_mw: 1500.0,
                cpu_frequency_ghz: 2.0,
                gpu_utilization: None,
            },
            performance_requirements: PerformanceTargets::default(),
            optimization_constraints: ResourceConstraints::default(),
            adaptation_request: None,
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(matches!(output.resource_assessment.bottleneck_analysis.primary_bottleneck, Some(BottleneckType::CPU)));
        assert!(!output.adaptation_recommendations.is_empty());
    }

    #[test]
    fn test_bottleneck_identification() {
        let agent = EdgeOptAgent::default();
        let utilization = ResourceUtilization {
            cpu_utilization_percent: 90.0,
            memory_utilization_percent: 70.0,
            power_utilization_percent: 60.0,
            thermal_utilization_percent: 50.0,
        };

        let primary = agent.identify_primary_bottleneck(&utilization);
        assert!(matches!(primary, Some(BottleneckType::CPU)));
    }
}
