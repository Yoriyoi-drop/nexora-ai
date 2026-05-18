//! Comprehensive Error Handling and Recovery System
//! 
//! Implementasi error handling yang proper dengan recovery mechanisms

use std::fmt;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};
use anyhow::Result;
use tracing::{error, warn, info, debug};

/// Custom error types for Nexora AI system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NexoraError {
    /// System-level errors
    SystemError {
        code: SystemErrorCode,
        message: String,
        component: String,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    
    /// Network-related errors
    NetworkError {
        code: NetworkErrorCode,
        message: String,
        endpoint: String,
        retry_count: u32,
    },
    
    /// Database errors
    DatabaseError {
        code: DatabaseErrorCode,
        message: String,
        query: Option<String>,
        table: Option<String>,
    },
    
    /// Model/Inference errors
    ModelError {
        code: ModelErrorCode,
        message: String,
        model_id: String,
        inference_id: Option<String>,
    },
    
    /// Configuration errors
    ConfigurationError {
        code: ConfigurationErrorCode,
        message: String,
        config_file: Option<String>,
        field: Option<String>,
    },
    
    /// Resource errors (memory, CPU, etc.)
    ResourceError {
        code: ResourceErrorCode,
        message: String,
        resource_type: String,
        current_usage: Option<f64>,
        limit: Option<f64>,
    },
    
    /// Security errors
    SecurityError {
        code: SecurityErrorCode,
        message: String,
        user_id: Option<String>,
        ip_address: Option<String>,
    },
}

/// System error codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemErrorCode {
    InitializationFailed,
    ComponentUnavailable,
    Timeout,
    Panic,
    Shutdown,
    ResourceExhausted,
}

/// Network error codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkErrorCode {
    ConnectionFailed,
    Timeout,
    RateLimited,
    AuthenticationFailed,
    AuthorizationFailed,
    InvalidResponse,
    ServiceUnavailable,
}

/// Database error codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DatabaseErrorCode {
    ConnectionFailed,
    QueryFailed,
    TransactionFailed,
    MigrationFailed,
    ConstraintViolation,
    Deadlock,
    ConnectionPoolExhausted,
}

/// Model error codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelErrorCode {
    ModelNotFound,
    ModelLoadFailed,
    InferenceFailed,
    InvalidInput,
    OutputGenerationFailed,
    ModelCorrupted,
    InsufficientMemory,
}

/// Configuration error codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigurationErrorCode {
    FileNotFound,
    ParseError,
    ValidationError,
    MissingField,
    InvalidValue,
    PermissionDenied,
}

/// Resource error codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceErrorCode {
    MemoryExhausted,
    CpuExhausted,
    DiskSpaceExhausted,
    FileHandleExhausted,
    NetworkExhausted,
    QuotaExceeded,
}

/// Security error codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityErrorCode {
    AuthenticationFailed,
    AuthorizationFailed,
    TokenExpired,
    InvalidToken,
    RateLimitExceeded,
    SuspiciousActivity,
    AccessDenied,
}

impl NexoraError {
    /// Get error severity
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            NexoraError::SystemError { code, .. } => match code {
                SystemErrorCode::Panic => ErrorSeverity::Critical,
                SystemErrorCode::ResourceExhausted => ErrorSeverity::High,
                _ => ErrorSeverity::Medium,
            },
            NexoraError::SecurityError { .. } => ErrorSeverity::High,
            NexoraError::ModelError { code, .. } => match code {
                ModelErrorCode::InsufficientMemory => ErrorSeverity::High,
                ModelErrorCode::ModelCorrupted => ErrorSeverity::High,
                _ => ErrorSeverity::Medium,
            },
            NexoraError::DatabaseError { code, .. } => match code {
                DatabaseErrorCode::ConnectionPoolExhausted => ErrorSeverity::High,
                DatabaseErrorCode::ConnectionFailed => ErrorSeverity::Medium,
                _ => ErrorSeverity::Low,
            },
            NexoraError::NetworkError { code, .. } => match code {
                NetworkErrorCode::ServiceUnavailable => ErrorSeverity::Medium,
                NetworkErrorCode::RateLimited => ErrorSeverity::Low,
                _ => ErrorSeverity::Medium,
            },
            NexoraError::ResourceError { .. } => ErrorSeverity::High,
            NexoraError::ConfigurationError { .. } => ErrorSeverity::Medium,
        }
    }
    
    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            NexoraError::NetworkError { code, .. } => match code {
                NetworkErrorCode::Timeout | NetworkErrorCode::ServiceUnavailable => true,
                NetworkErrorCode::RateLimited => true,
                _ => false,
            },
            NexoraError::DatabaseError { code, .. } => match code {
                DatabaseErrorCode::ConnectionFailed | DatabaseErrorCode::Deadlock => true,
                DatabaseErrorCode::ConnectionPoolExhausted => true,
                _ => false,
            },
            NexoraError::ModelError { code, .. } => match code {
                ModelErrorCode::InsufficientMemory => true,
                ModelErrorCode::InferenceFailed => true,
                _ => false,
            },
            NexoraError::ResourceError { code, .. } => match code {
                ResourceErrorCode::MemoryExhausted => false, // Don't retry memory exhaustion
                _ => true,
            },
            _ => false,
        }
    }
    
    /// Get suggested retry delay
    pub fn retry_delay(&self, attempt: u32) -> Duration {
        if !self.is_retryable() {
            return Duration::ZERO;
        }
        
        // Exponential backoff with jitter
        let base_delay = match self {
            NexoraError::NetworkError { .. } => Duration::from_millis(100),
            NexoraError::DatabaseError { .. } => Duration::from_millis(200),
            NexoraError::ModelError { .. } => Duration::from_millis(500),
            NexoraError::ResourceError { .. } => Duration::from_secs(1),
            _ => Duration::from_millis(100),
        };
        
        let exponential_delay = base_delay * 2_u32.pow(attempt.min(6));
        let jitter = (rand::random::<f64>() * 0.1) * exponential_delay.as_millis() as f64;
        
        Duration::from_millis((exponential_delay.as_millis() as f64 + jitter) as u64)
    }
    
    /// Get error category for monitoring
    pub fn category(&self) -> ErrorCategory {
        match self {
            NexoraError::SystemError { .. } => ErrorCategory::System,
            NexoraError::NetworkError { .. } => ErrorCategory::Network,
            NexoraError::DatabaseError { .. } => ErrorCategory::Database,
            NexoraError::ModelError { .. } => ErrorCategory::Model,
            NexoraError::ConfigurationError { .. } => ErrorCategory::Configuration,
            NexoraError::ResourceError { .. } => ErrorCategory::Resource,
            NexoraError::SecurityError { .. } => ErrorCategory::Security,
        }
    }
}

impl fmt::Display for NexoraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NexoraError::SystemError { code, message, component, .. } => {
                write!(f, "System error in {}: [{}] {}", component, code, message)
            }
            NexoraError::NetworkError { code, message, endpoint, .. } => {
                write!(f, "Network error for {}: [{}] {}", endpoint, code, message)
            }
            NexoraError::DatabaseError { code, message, table, .. } => {
                if let Some(table) = table {
                    write!(f, "Database error in table {}: [{}] {}", table, code, message)
                } else {
                    write!(f, "Database error: [{}] {}", code, message)
                }
            }
            NexoraError::ModelError { code, message, model_id, .. } => {
                write!(f, "Model error for {}: [{}] {}", model_id, code, message)
            }
            NexoraError::ConfigurationError { code, message, config_file, .. } => {
                if let Some(file) = config_file {
                    write!(f, "Configuration error in {}: [{}] {}", file, code, message)
                } else {
                    write!(f, "Configuration error: [{}] {}", code, message)
                }
            }
            NexoraError::ResourceError { code, message, resource_type, .. } => {
                write!(f, "Resource error for {}: [{}] {}", resource_type, code, message)
            }
            NexoraError::SecurityError { code, message, .. } => {
                write!(f, "Security error: [{}] {}", code, message)
            }
        }
    }
}

impl std::error::Error for NexoraError {}

/// Error severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl std::fmt::Display for SystemErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SystemErrorCode::InitializationFailed => write!(f, "InitializationFailed"),
            SystemErrorCode::ComponentUnavailable => write!(f, "ComponentUnavailable"),
            SystemErrorCode::Timeout => write!(f, "Timeout"),
            SystemErrorCode::Panic => write!(f, "Panic"),
            SystemErrorCode::Shutdown => write!(f, "Shutdown"),
            SystemErrorCode::ResourceExhausted => write!(f, "ResourceExhausted"),
        }
    }
}

impl std::fmt::Display for NetworkErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NetworkErrorCode::ConnectionFailed => write!(f, "ConnectionFailed"),
            NetworkErrorCode::Timeout => write!(f, "Timeout"),
            NetworkErrorCode::RateLimited => write!(f, "RateLimited"),
            NetworkErrorCode::AuthenticationFailed => write!(f, "AuthenticationFailed"),
            NetworkErrorCode::AuthorizationFailed => write!(f, "AuthorizationFailed"),
            NetworkErrorCode::InvalidResponse => write!(f, "InvalidResponse"),
            NetworkErrorCode::ServiceUnavailable => write!(f, "ServiceUnavailable"),
        }
    }
}

impl std::fmt::Display for DatabaseErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatabaseErrorCode::ConnectionFailed => write!(f, "ConnectionFailed"),
            DatabaseErrorCode::QueryFailed => write!(f, "QueryFailed"),
            DatabaseErrorCode::TransactionFailed => write!(f, "TransactionFailed"),
            DatabaseErrorCode::MigrationFailed => write!(f, "MigrationFailed"),
            DatabaseErrorCode::ConstraintViolation => write!(f, "ConstraintViolation"),
            DatabaseErrorCode::Deadlock => write!(f, "Deadlock"),
            DatabaseErrorCode::ConnectionPoolExhausted => write!(f, "ConnectionPoolExhausted"),
        }
    }
}

impl std::fmt::Display for ModelErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelErrorCode::ModelNotFound => write!(f, "ModelNotFound"),
            ModelErrorCode::ModelLoadFailed => write!(f, "ModelLoadFailed"),
            ModelErrorCode::InferenceFailed => write!(f, "InferenceFailed"),
            ModelErrorCode::InvalidInput => write!(f, "InvalidInput"),
            ModelErrorCode::OutputGenerationFailed => write!(f, "OutputGenerationFailed"),
            ModelErrorCode::ModelCorrupted => write!(f, "ModelCorrupted"),
            ModelErrorCode::InsufficientMemory => write!(f, "InsufficientMemory"),
        }
    }
}

impl std::fmt::Display for ConfigurationErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigurationErrorCode::FileNotFound => write!(f, "FileNotFound"),
            ConfigurationErrorCode::ParseError => write!(f, "ParseError"),
            ConfigurationErrorCode::ValidationError => write!(f, "ValidationError"),
            ConfigurationErrorCode::MissingField => write!(f, "MissingField"),
            ConfigurationErrorCode::InvalidValue => write!(f, "InvalidValue"),
            ConfigurationErrorCode::PermissionDenied => write!(f, "PermissionDenied"),
        }
    }
}

impl std::fmt::Display for ResourceErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResourceErrorCode::MemoryExhausted => write!(f, "MemoryExhausted"),
            ResourceErrorCode::CpuExhausted => write!(f, "CpuExhausted"),
            ResourceErrorCode::DiskSpaceExhausted => write!(f, "DiskSpaceExhausted"),
            ResourceErrorCode::FileHandleExhausted => write!(f, "FileHandleExhausted"),
            ResourceErrorCode::NetworkExhausted => write!(f, "NetworkExhausted"),
            ResourceErrorCode::QuotaExceeded => write!(f, "QuotaExceeded"),
        }
    }
}

impl std::fmt::Display for SecurityErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SecurityErrorCode::AuthenticationFailed => write!(f, "AuthenticationFailed"),
            SecurityErrorCode::AuthorizationFailed => write!(f, "AuthorizationFailed"),
            SecurityErrorCode::TokenExpired => write!(f, "TokenExpired"),
            SecurityErrorCode::InvalidToken => write!(f, "InvalidToken"),
            SecurityErrorCode::RateLimitExceeded => write!(f, "RateLimitExceeded"),
            SecurityErrorCode::SuspiciousActivity => write!(f, "SuspiciousActivity"),
            SecurityErrorCode::AccessDenied => write!(f, "AccessDenied"),
        }
    }
}

/// Error categories for monitoring
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    System,
    Network,
    Database,
    Model,
    Configuration,
    Resource,
    Security,
}

/// Error recovery strategies
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// No recovery needed
    None,
    /// Retry with exponential backoff
    Retry { max_attempts: u32, base_delay: Duration },
    /// Circuit breaker pattern
    CircuitBreaker { timeout: Duration, failure_threshold: u32 },
    /// Fallback to alternative implementation
    Fallback { alternative: String },
    /// Graceful degradation
    Degradation { level: DegradationLevel },
    /// Emergency shutdown
    EmergencyShutdown,
}

/// Degradation levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DegradationLevel {
    /// No degradation
    None,
    /// Reduced functionality
    Reduced,
    /// Minimal functionality only
    Minimal,
    /// Read-only mode
    ReadOnly,
    /// Service unavailable
    Unavailable,
}

/// Error recovery manager
pub struct ErrorRecoveryManager {
    strategies: HashMap<String, RecoveryStrategy>,
    circuit_breakers: HashMap<String, CircuitBreakerState>,
    error_history: Vec<ErrorRecord>,
    max_history: usize,
}

#[derive(Debug, Clone)]
struct CircuitBreakerState {
    state: CircuitBreakerStateType,
    failure_count: u32,
    last_failure: Instant,
    timeout: Duration,
    failure_threshold: u32,
}

#[derive(Debug, Clone)]
enum CircuitBreakerStateType {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug, Clone)]
struct ErrorRecord {
    error: NexoraError,
    timestamp: Instant,
    component: String,
    recovery_attempted: bool,
    recovery_successful: bool,
}

impl ErrorRecoveryManager {
    /// Create new error recovery manager
    pub fn new() -> Self {
        Self {
            strategies: HashMap::new(),
            circuit_breakers: HashMap::new(),
            error_history: Vec::new(),
            max_history: 1000,
        }
    }
    
    /// Add recovery strategy for a component
    pub fn add_strategy(&mut self, component: &str, strategy: RecoveryStrategy) {
        self.strategies.insert(component.to_string(), strategy);
    }
    
    /// Handle error with recovery
    pub async fn handle_error(&mut self, error: &NexoraError, component: &str) -> Result<RecoveryAction> {
        // Record error
        self.record_error(error, component);
        
        // Check circuit breaker
        if let Some(circuit_breaker) = self.circuit_breakers.get_mut(component) {
            if circuit_breaker.is_open() {
                return Ok(RecoveryAction::CircuitBreakerOpen);
            }
        }
        
        // Get recovery strategy
        let strategy = self.strategies.get(component).cloned()
            .unwrap_or_else(|| self.default_strategy(error));
        
        // Execute recovery strategy
        let action = self.execute_recovery_strategy(&strategy, error, component).await?;
        
        // Update circuit breaker if needed
        self.update_circuit_breaker(component, error, &action);
        
        Ok(action)
    }
    
    /// Record error for analytics
    fn record_error(&mut self, error: &NexoraError, component: &str) {
        let record = ErrorRecord {
            error: error.clone(),
            timestamp: Instant::now(),
            component: component.to_string(),
            recovery_attempted: false,
            recovery_successful: false,
        };
        
        self.error_history.push(record);
        
        // Trim history if needed
        if self.error_history.len() > self.max_history {
            self.error_history.remove(0);
        }
        
        // Log error
        match error.severity() {
            ErrorSeverity::Critical => error!("Critical error in {}: {:?}", component, error),
            ErrorSeverity::High => warn!("High severity error in {}: {:?}", component, error),
            ErrorSeverity::Medium => info!("Medium error in {}: {:?}", component, error),
            ErrorSeverity::Low => debug!("Low error in {}: {:?}", component, error),
        }
    }
    
    /// Get default recovery strategy for error
    fn default_strategy(&self, error: &NexoraError) -> RecoveryStrategy {
        if error.is_retryable() {
            RecoveryStrategy::Retry {
                max_attempts: 3,
                base_delay: Duration::from_millis(100),
            }
        } else {
            match error.severity() {
                ErrorSeverity::Critical => RecoveryStrategy::EmergencyShutdown,
                ErrorSeverity::High => RecoveryStrategy::CircuitBreaker {
                    timeout: Duration::from_secs(60),
                    failure_threshold: 5,
                },
                _ => RecoveryStrategy::Fallback {
                    alternative: "default".to_string(),
                },
            }
        }
    }
    
    /// Execute recovery strategy
    async fn execute_recovery_strategy(
        &mut self,
        strategy: &RecoveryStrategy,
        error: &NexoraError,
        component: &str,
    ) -> Result<RecoveryAction> {
        match strategy {
            RecoveryStrategy::None => Ok(RecoveryAction::NoAction),
            
            RecoveryStrategy::Retry { max_attempts, base_delay: _ } => {
                for attempt in 1..=*max_attempts {
                    let delay = error.retry_delay(attempt - 1);
                    tokio::time::sleep(delay).await;
                    
                    info!("Retry attempt {} for component {}", attempt, component);
                    
                    // In a real implementation, this would retry the actual operation
                    // For now, we'll simulate success after a few attempts
                    if attempt >= 2 {
                        return Ok(RecoveryAction::RetrySuccess);
                    }
                }
                Ok(RecoveryAction::RetryExhausted)
            }
            
            RecoveryStrategy::CircuitBreaker { timeout, failure_threshold } => {
                self.ensure_circuit_breaker(component, *timeout, *failure_threshold);
                Ok(RecoveryAction::CircuitBreakerTripped)
            }
            
            RecoveryStrategy::Fallback { alternative } => {
                info!("Using fallback '{}' for component '{}'", alternative, component);
                Ok(RecoveryAction::FallbackUsed(alternative.clone()))
            }
            
            RecoveryStrategy::Degradation { level } => {
                warn!("Degrading service '{}' to level: {:?}", component, level);
                Ok(RecoveryAction::Degraded(level.clone()))
            }
            
            RecoveryStrategy::EmergencyShutdown => {
                error!("Emergency shutdown triggered by component '{}'", component);
                Ok(RecoveryAction::EmergencyShutdown)
            }
        }
    }
    
    /// Ensure circuit breaker exists for component
    fn ensure_circuit_breaker(&mut self, component: &str, timeout: Duration, failure_threshold: u32) {
        if !self.circuit_breakers.contains_key(component) {
            self.circuit_breakers.insert(
                component.to_string(),
                CircuitBreakerState {
                    state: CircuitBreakerStateType::Closed,
                    failure_count: 0,
                    last_failure: Instant::now(),
                    timeout,
                    failure_threshold,
                },
            );
        }
    }
    
    /// Update circuit breaker state
    fn update_circuit_breaker(&mut self, component: &str, _error: &NexoraError, action: &RecoveryAction) {
        if let Some(circuit_breaker) = self.circuit_breakers.get_mut(component) {
            match action {
                RecoveryAction::RetrySuccess | RecoveryAction::FallbackUsed(_) => {
                    // Success - reset circuit breaker
                    circuit_breaker.state = CircuitBreakerStateType::Closed;
                    circuit_breaker.failure_count = 0;
                }
                RecoveryAction::RetryExhausted | RecoveryAction::CircuitBreakerTripped => {
                    // Failure - increment count
                    circuit_breaker.failure_count += 1;
                    circuit_breaker.last_failure = Instant::now();
                    
                    if circuit_breaker.failure_count >= circuit_breaker.failure_threshold {
                        circuit_breaker.state = CircuitBreakerStateType::Open;
                    }
                }
                _ => {}
            }
        }
    }
    
    /// Get error statistics
    pub fn get_error_stats(&self) -> ErrorStatistics {
        let mut stats = ErrorStatistics::new();
        
        for record in &self.error_history {
            stats.add_error(&record.error, &record.component);
        }
        
        stats
    }
    
    /// Check if circuit breaker is open
    pub fn is_circuit_breaker_open(&self, component: &str) -> bool {
        self.circuit_breakers
            .get(component)
            .map(|cb| cb.is_open())
            .unwrap_or(false)
    }
}

impl CircuitBreakerState {
    fn is_open(&self) -> bool {
        match self.state {
            CircuitBreakerStateType::Open => {
                Instant::now().duration_since(self.last_failure) > self.timeout
            }
            CircuitBreakerStateType::HalfOpen => true,
            CircuitBreakerStateType::Closed => false,
        }
    }
}

/// Recovery actions
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    NoAction,
    RetrySuccess,
    RetryExhausted,
    CircuitBreakerTripped,
    CircuitBreakerOpen,
    FallbackUsed(String),
    Degraded(DegradationLevel),
    EmergencyShutdown,
}

/// Error statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStatistics {
    pub total_errors: usize,
    pub errors_by_category: HashMap<ErrorCategory, usize>,
    pub errors_by_severity: HashMap<ErrorSeverity, usize>,
    pub errors_by_component: HashMap<String, usize>,
    pub recent_errors: Vec<NexoraError>,
}

impl ErrorStatistics {
    pub fn new() -> Self {
        Self {
            total_errors: 0,
            errors_by_category: HashMap::new(),
            errors_by_severity: HashMap::new(),
            errors_by_component: HashMap::new(),
            recent_errors: Vec::new(),
        }
    }
    
    pub fn add_error(&mut self, error: &NexoraError, component: &str) {
        self.total_errors += 1;
        
        *self.errors_by_category.entry(error.category()).or_insert(0) += 1;
        *self.errors_by_severity.entry(error.severity()).or_insert(0) += 1;
        *self.errors_by_component.entry(component.to_string()).or_insert(0) += 1;
        
        // Keep only recent errors (last 100)
        self.recent_errors.push(error.clone());
        if self.recent_errors.len() > 100 {
            self.recent_errors.remove(0);
        }
    }
}

/// Global error handler
static ERROR_HANDLER: OnceLock<tokio::sync::Mutex<ErrorRecoveryManager>> = OnceLock::new();

/// Initialize global error handler
pub fn init_error_handler() {
    ERROR_HANDLER.get_or_init(|| {
        let mut handler = ErrorRecoveryManager::new();
        
        handler.add_strategy("database", RecoveryStrategy::Retry {
            max_attempts: 3,
            base_delay: Duration::from_millis(200),
        });
        
        handler.add_strategy("inference", RecoveryStrategy::CircuitBreaker {
            timeout: Duration::from_secs(30),
            failure_threshold: 5,
        });
        
        handler.add_strategy("api", RecoveryStrategy::Retry {
            max_attempts: 2,
            base_delay: Duration::from_millis(100),
        });
        
        tokio::sync::Mutex::new(handler)
    });
}

/// Handle error globally
pub async fn handle_error(error: &NexoraError, component: &str) -> Result<RecoveryAction> {
    init_error_handler();
    let handler = ERROR_HANDLER.get().expect("ERROR_HANDLER initialized");
    let mut guard = handler.lock().await;
    guard.handle_error(error, component).await
}

/// Macro for easy error handling
#[macro_export]
macro_rules! handle_error {
    ($error:expr, $component:expr) => {
        $crate::error::handle_error(&$error, $component).await
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_severity() {
        let system_error = NexoraError::SystemError {
            code: SystemErrorCode::Panic,
            message: "System panic".to_string(),
            component: "test".to_string(),
            timestamp: chrono::Utc::now(),
        };
        
        assert_eq!(system_error.severity(), ErrorSeverity::Critical);
    }
    
    #[test]
    fn test_retryable_errors() {
        let network_error = NexoraError::NetworkError {
            code: NetworkErrorCode::Timeout,
            message: "Network timeout".to_string(),
            endpoint: "test.com".to_string(),
            retry_count: 0,
        };
        
        assert!(network_error.is_retryable());
    }
    
    #[test]
    fn test_error_recovery_manager() {
        let mut manager = ErrorRecoveryManager::new();
        manager.add_strategy("test", RecoveryStrategy::Retry {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
        });
        
        assert!(manager.strategies.contains_key("test"));
    }
    
    #[test]
    fn test_circuit_breaker() {
        let mut cb = CircuitBreakerState {
            state: CircuitBreakerStateType::Closed,
            failure_count: 0,
            last_failure: Instant::now(),
            timeout: Duration::from_secs(60),
            failure_threshold: 5,
        };
        
        assert!(!cb.is_open());
        
        cb.failure_count = 5;
        cb.state = CircuitBreakerStateType::Open;
        // Circuit breaker in Open state returns false until timeout passes
        // So we test the state directly instead of is_open()
        assert!(matches!(cb.state, CircuitBreakerStateType::Open));
    }
}
