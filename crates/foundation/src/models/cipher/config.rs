//! NXR-CIPHER Configuration
//! 
//! Model-specific configuration for NXR-CIPHER

use serde::{Deserialize, Serialize};
use crate::shared::{model_config::NxrModelConfig, deeplearning_integration::DeepLearningConfig};

/// NXR-CIPHER Specific Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CipherConfig {
    /// Base configuration
    pub base: NxrModelConfig,
    /// Security analysis configuration
    pub security_analysis: SecurityAnalysisConfig,
    /// Vulnerability scanning configuration
    pub vulnerability_scanning: VulnerabilityScanningConfig,
    /// Penetration testing configuration
    pub penetration_testing: PenetrationTestingConfig,
    /// Threat intelligence configuration
    pub threat_intelligence: ThreatIntelligenceConfig,
    /// Agent configuration
    pub agents: AgentConfig,
    /// Deep learning configuration
    pub deep_learning: DeepLearningConfig,
}

/// Security Analysis Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAnalysisConfig {
    /// Analysis depth
    pub analysis_depth: AnalysisDepth,
    /// Analysis scope
    pub analysis_scope: AnalysisScope,
    /// Analysis methods
    pub analysis_methods: Vec<AnalysisMethod>,
    /// Reporting detail level
    pub reporting_detail: ReportingDetail,
}

/// Analysis Depth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisDepth {
    /// Surface level analysis
    Surface,
    /// Deep analysis
    Deep,
    /// Comprehensive analysis
    Comprehensive,
    /// Forensic analysis
    Forensic,
}

/// Analysis Scope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisScope {
    /// Network scope
    Network,
    /// Application scope
    Application,
    /// System scope
    System,
    /// Full enterprise scope
    Enterprise,
}

/// Analysis Method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisMethod {
    /// Static analysis
    Static,
    /// Dynamic analysis
    Dynamic,
    /// Hybrid analysis
    Hybrid,
    /// Behavioral analysis
    Behavioral,
}

/// Reporting Detail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportingDetail {
    /// Minimal detail
    Minimal,
    /// Standard detail
    Standard,
    /// Detailed report
    Detailed,
    /// Comprehensive report
    Comprehensive,
}

/// Vulnerability Scanning Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilityScanningConfig {
    /// Scan intensity
    pub scan_intensity: ScanIntensity,
    /// Scan frequency
    pub scan_frequency: ScanFrequency,
    /// Vulnerability database
    pub vulnerability_database: VulnerabilityDatabase,
    /// False positive handling
    pub false_positive_handling: FalsePositiveHandling,
}

/// Scan Intensity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScanIntensity {
    /// Light scan
    Light,
    /// Moderate scan
    Moderate,
    /// Intensive scan
    Intensive,
    /// Aggressive scan
    Aggressive,
}

/// Scan Frequency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScanFrequency {
    /// One-time scan
    OneTime,
    /// Daily scan
    Daily,
    /// Weekly scan
    Weekly,
    /// Continuous scan
    Continuous,
}

/// Vulnerability Database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VulnerabilityDatabase {
    /// Built-in database
    BuiltIn,
    /// External database
    External { url: String, api_key: Option<String> },
    /// Hybrid database
    Hybrid { built_in: bool, external: Option<String> },
}

/// False Positive Handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FalsePositiveHandling {
    /// Ignore false positives
    Ignore,
    /// Flag false positives
    Flag,
    /// Manual review
    ManualReview,
    /// Automatic filtering
    AutomaticFiltering,
}

/// Penetration Testing Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenetrationTestingConfig {
    /// Testing methodology
    pub methodology: TestingMethodology,
    /// Test coverage
    pub test_coverage: TestCoverage,
    /// Exploitation limits
    pub exploitation_limits: ExploitationLimits,
    /// Reporting format
    pub reporting_format: ReportingFormat,
}

/// Testing Methodology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestingMethodology {
    /// Black box testing
    BlackBox,
    /// White box testing
    WhiteBox,
    /// Gray box testing
    GrayBox,
    /// Hybrid testing
    Hybrid,
}

/// Test Coverage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestCoverage {
    /// Basic coverage
    Basic,
    /// Standard coverage
    Standard,
    /// Comprehensive coverage
    Comprehensive,
    /// Full coverage
    Full,
}

/// Exploitation Limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExploitationLimits {
    /// Allow exploitation
    pub allow_exploitation: bool,
    /// Exploitation depth
    pub exploitation_depth: ExploitationDepth,
    /// Data access limits
    pub data_access_limits: DataAccessLimits,
}

/// Exploitation Depth
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExploitationDepth {
    /// No exploitation
    None,
    /// Read-only exploitation
    ReadOnly,
    /// Limited exploitation
    Limited,
    /// Full exploitation
    Full,
}

/// Data Access Limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataAccessLimits {
    /// No data access
    NoAccess,
    /// Public data only
    PublicOnly,
    /// Non-sensitive data
    NonSensitive,
    /// Full data access
    FullAccess,
}

/// Reporting Format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportingFormat {
    /// JSON format
    JSON,
    /// PDF format
    PDF,
    /// HTML format
    HTML,
    /// Custom format
    Custom { template: String },
}

/// Threat Intelligence Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatIntelligenceConfig {
    /// Intelligence sources
    pub intelligence_sources: Vec<IntelligenceSource>,
    /// Update frequency
    pub update_frequency: UpdateFrequency,
    /// Threat scoring
    pub threat_scoring: ThreatScoring,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
}

/// Intelligence Source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntelligenceSource {
    /// Internal sources
    Internal,
    /// External feeds
    ExternalFeeds { url: String },
    /// Community sources
    Community,
    /// Commercial sources
    Commercial { provider: String, api_key: Option<String> },
}

/// Update Frequency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateFrequency {
    /// Real-time updates
    RealTime,
    /// Hourly updates
    Hourly,
    /// Daily updates
    Daily,
    /// Weekly updates
    Weekly,
}

/// Threat Scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatScoring {
    /// Scoring methodology
    pub scoring_methodology: ScoringMethodology,
    /// Risk factors
    pub risk_factors: Vec<RiskFactor>,
    /// Score ranges
    pub score_ranges: ScoreRanges,
}

/// Scoring Methodology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScoringMethodology {
    /// CVSS scoring
    CVSS,
    /// Custom scoring
    Custom,
    /// Hybrid scoring
    Hybrid,
}

/// Risk Factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    /// Factor name
    pub name: String,
    /// Factor weight
    pub weight: f32,
    /// Factor description
    pub description: String,
}

/// Score Ranges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreRanges {
    /// Critical range
    pub critical: (f32, f32),
    /// High range
    pub high: (f32, f32),
    /// Medium range
    pub medium: (f32, f32),
    /// Low range
    pub low: (f32, f32),
}

/// Alert Thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Critical threshold
    pub critical_threshold: f32,
    /// High threshold
    pub high_threshold: f32,
    /// Medium threshold
    pub medium_threshold: f32,
}

/// Agent Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// PENTEST-BOT configuration
    pub pentest_bot: PentestBotConfig,
    /// VULN-SCAN configuration
    pub vuln_scan: VulnScanConfig,
    /// FIREWALL-AI configuration
    pub firewall_ai: FirewallAiConfig,
    /// THREAT-HUNT configuration
    pub threat_hunt: ThreatHuntConfig,
}

/// PENTEST-BOT Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PentestBotConfig {
    /// Testing methodology
    pub testing_methodology: TestingMethodology,
    /// Automation level
    pub automation_level: AutomationLevel,
    /// Report generation
    pub report_generation: bool,
    /// Execution timeout
    pub execution_timeout_seconds: u64,
}

/// Automation Level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AutomationLevel {
    /// Manual execution
    Manual,
    /// Semi-automated
    SemiAutomated,
    /// Fully automated
    FullyAutomated,
}

/// VULN-SCAN Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnScanConfig {
    /// Scan intensity
    pub scan_intensity: ScanIntensity,
    /// Scan targets
    pub scan_targets: Vec<ScanTarget>,
    /// False positive tolerance
    pub false_positive_tolerance: f32,
    /// Scan parallelization
    pub scan_parallelization: bool,
}

/// Scan Target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanTarget {
    /// Target type
    pub target_type: TargetType,
    /// Target address
    pub address: String,
    /// Target priority
    pub priority: u8,
}

/// Target Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TargetType {
    /// Network target
    Network,
    /// Application target
    Application,
    /// Database target
    Database,
    /// API target
    API,
}

/// FIREWALL-AI Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallAiConfig {
    /// Rule generation
    pub rule_generation: bool,
    /// Rule optimization
    pub rule_optimization: bool,
    /// Anomaly detection
    pub anomaly_detection: bool,
    /// Learning mode
    pub learning_mode: LearningMode,
}

/// Learning Mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningMode {
    /// Supervised learning
    Supervised,
    /// Unsupervised learning
    Unsupervised,
    /// Reinforcement learning
    Reinforcement,
    /// Hybrid learning
    Hybrid,
}

/// THREAT-HUNT Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatHuntConfig {
    /// Hunt methodology
    pub hunt_methodology: HuntMethodology,
    /// Data sources
    pub data_sources: Vec<DataSource>,
    /// Investigation depth
    pub investigation_depth: InvestigationDepth,
    /// Automation level
    pub automation_level: AutomationLevel,
}

/// Hunt Methodology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HuntMethodology {
    /// Hypothesis-based hunting
    HypothesisBased,
    /// Indicator-based hunting
    IndicatorBased,
    /// Behavioral hunting
    Behavioral,
    /// Hybrid hunting
    Hybrid,
}

/// Data Source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSource {
    /// Log files
    Logs,
    /// Network traffic
    NetworkTraffic,
    /// System events
    SystemEvents,
    /// Application events
    ApplicationEvents,
}

/// Investigation Depth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InvestigationDepth {
    /// Surface investigation
    Surface,
    /// Deep investigation
    Deep,
    /// Forensic investigation
    Forensic,
}

impl Default for CipherConfig {
    fn default() -> Self {
        Self {
            base: NxrModelConfig::for_model(crate::shared::model_identity::NxrModelId::Cipher),
            security_analysis: SecurityAnalysisConfig::default(),
            vulnerability_scanning: VulnerabilityScanningConfig::default(),
            penetration_testing: PenetrationTestingConfig::default(),
            threat_intelligence: ThreatIntelligenceConfig::default(),
            agents: AgentConfig::default(),
            deep_learning: DeepLearningConfig::star_x(),
        }
    }
}

impl Default for SecurityAnalysisConfig {
    fn default() -> Self {
        Self {
            analysis_depth: AnalysisDepth::Deep,
            analysis_scope: AnalysisScope::Enterprise,
            analysis_methods: vec![
                AnalysisMethod::Static,
                AnalysisMethod::Dynamic,
                AnalysisMethod::Hybrid,
            ],
            reporting_detail: ReportingDetail::Detailed,
        }
    }
}

impl Default for VulnerabilityScanningConfig {
    fn default() -> Self {
        Self {
            scan_intensity: ScanIntensity::Intensive,
            scan_frequency: ScanFrequency::Continuous,
            vulnerability_database: VulnerabilityDatabase::Hybrid {
                built_in: true,
                external: Some("https://nvd.nist.gov".to_string()),
            },
            false_positive_handling: FalsePositiveHandling::AutomaticFiltering,
        }
    }
}

impl Default for PenetrationTestingConfig {
    fn default() -> Self {
        Self {
            methodology: TestingMethodology::Hybrid,
            test_coverage: TestCoverage::Comprehensive,
            exploitation_limits: ExploitationLimits {
                allow_exploitation: true,
                exploitation_depth: ExploitationDepth::ReadOnly,
                data_access_limits: DataAccessLimits::NonSensitive,
            },
            reporting_format: ReportingFormat::JSON,
        }
    }
}

impl Default for ThreatIntelligenceConfig {
    fn default() -> Self {
        Self {
            intelligence_sources: vec![
                IntelligenceSource::Internal,
                IntelligenceSource::Community,
            ],
            update_frequency: UpdateFrequency::Hourly,
            threat_scoring: ThreatScoring {
                scoring_methodology: ScoringMethodology::CVSS,
                risk_factors: vec![
                    RiskFactor {
                        name: "severity".to_string(),
                        weight: 0.4,
                        description: "Vulnerability severity".to_string(),
                    },
                    RiskFactor {
                        name: "exploitability".to_string(),
                        weight: 0.3,
                        description: "Ease of exploitation".to_string(),
                    },
                    RiskFactor {
                        name: "impact".to_string(),
                        weight: 0.3,
                        description: "Potential impact".to_string(),
                    },
                ],
                score_ranges: ScoreRanges {
                    critical: (9.0, 10.0),
                    high: (7.0, 8.9),
                    medium: (4.0, 6.9),
                    low: (0.0, 3.9),
                },
            },
            alert_thresholds: AlertThresholds {
                critical_threshold: 9.0,
                high_threshold: 7.0,
                medium_threshold: 4.0,
            },
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            pentest_bot: PentestBotConfig::default(),
            vuln_scan: VulnScanConfig::default(),
            firewall_ai: FirewallAiConfig::default(),
            threat_hunt: ThreatHuntConfig::default(),
        }
    }
}

impl Default for PentestBotConfig {
    fn default() -> Self {
        Self {
            testing_methodology: TestingMethodology::GrayBox,
            automation_level: AutomationLevel::SemiAutomated,
            report_generation: true,
            execution_timeout_seconds: 3600,
        }
    }
}

impl Default for VulnScanConfig {
    fn default() -> Self {
        Self {
            scan_intensity: ScanIntensity::Intensive,
            scan_targets: vec![],
            false_positive_tolerance: 0.1,
            scan_parallelization: true,
        }
    }
}

impl Default for FirewallAiConfig {
    fn default() -> Self {
        Self {
            rule_generation: true,
            rule_optimization: true,
            anomaly_detection: true,
            learning_mode: LearningMode::Hybrid,
        }
    }
}

impl Default for ThreatHuntConfig {
    fn default() -> Self {
        Self {
            hunt_methodology: HuntMethodology::Hybrid,
            data_sources: vec![
                DataSource::Logs,
                DataSource::NetworkTraffic,
                DataSource::SystemEvents,
            ],
            investigation_depth: InvestigationDepth::Deep,
            automation_level: AutomationLevel::SemiAutomated,
        }
    }
}

impl CipherConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate base configuration
        self.base.validate()?;

        // Validate security analysis configuration
        if matches!(self.security_analysis.reporting_detail, ReportingDetail::Minimal) {
            return Err("Reporting detail cannot be minimal".to_string());
        }

        // Validate penetration testing configuration
        if self.penetration_testing.exploitation_limits.exploitation_depth == ExploitationDepth::Full 
           && !self.penetration_testing.exploitation_limits.allow_exploitation {
            return Err("Full exploitation requires allow_exploitation to be true".to_string());
        }

        // Validate threat intelligence configuration
        if self.threat_intelligence.alert_thresholds.critical_threshold < self.threat_intelligence.alert_thresholds.high_threshold {
            return Err("Critical threshold must be >= high threshold".to_string());
        }

        Ok(())
    }

    /// Get configuration for specific agent
    pub fn get_agent_config(&self, agent_name: &str) -> Option<serde_json::Value> {
        match agent_name {
            "pentest_bot" => Some(serde_json::to_value(&self.agents.pentest_bot).unwrap_or_default()),
            "vuln_scan" => Some(serde_json::to_value(&self.agents.vuln_scan).unwrap_or_default()),
            "firewall_ai" => Some(serde_json::to_value(&self.agents.firewall_ai).unwrap_or_default()),
            "threat_hunt" => Some(serde_json::to_value(&self.agents.threat_hunt).unwrap_or_default()),
            _ => None,
        }
    }
}
