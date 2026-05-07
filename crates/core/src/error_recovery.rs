//! Comprehensive Error Handling dan Recovery System
//! 
//! Implementasi error handling dengan automatic recovery, circuit breaker, dan retry policies

use crate::error::{CoreError, CoreResult};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use parking_lot::RwLock as ParkingRwLock;
use tracing::{debug, info, warn};
use serde::{Deserialize, Serialize};

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Low = 0,      // Minor issues, can continue
    Medium = 1,   // Degraded performance
    High = 2,     // Major functionality affected
    Critical = 3, // System cannot operate
}

/// Error category untuk classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorCategory {
    Network,       // Network-related errors
    Database,      // Database access errors
    Model,         // Model execution errors
    Memory,        // Memory management errors
    Configuration,  // Configuration errors
    Validation,    // Input validation errors
    Timeout,       // Timeout errors
    Resource,      // Resource exhaustion
    System,        // System-level errors
    Unknown,       // Unclassified errors
}

/// Enhanced error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInfo {
    pub error_id: String,
    pub category: ErrorCategory,
    pub severity: ErrorSeverity,
    pub message: String,
    pub source: String,
    pub timestamp: u64,
    pub context: HashMap<String, String>,
    pub recoverable: bool,
    pub retry_count: u32,
    pub max_retries: u32,
}

impl ErrorInfo {
    pub fn new(category: ErrorCategory, severity: ErrorSeverity, message: String, source: String) -> Self {
        Self {
            error_id: uuid::Uuid::new_v4().to_string(),
            category,
            severity,
            message,
            source,
            timestamp: Self::current_timestamp(),
            context: HashMap::new(),
            recoverable: true,
            retry_count: 0,
            max_retries: 3,
        }
    }
    
    pub fn with_context(mut self, key: String, value: String) -> Self {
        self.context.insert(key, value);
        self
    }
    
    pub fn with_recoverable(mut self, recoverable: bool) -> Self {
        self.recoverable = recoverable;
        self
    }
    
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }
    
    pub fn increment_retry(&mut self) -> bool {
        self.retry_count += 1;
        self.retry_count <= self.max_retries
    }
    
    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,    // Normal operation
    Open,      // Circuit is open, fail fast
    HalfOpen,  // Testing if service recovered
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,      // Failures before opening
    pub recovery_timeout_ms: u64,    // Time to wait before trying again
    pub success_threshold: u32,      // Successes to close circuit
    pub monitoring_period_ms: u64,   // Period for failure counting
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout_ms: 60000,  // 1 minute
            success_threshold: 3,
            monitoring_period_ms: 300000, // 5 minutes
        }
    }
}

/// Circuit breaker untuk fault tolerance
#[derive(Clone)]
pub struct CircuitBreaker {
    name: String,
    config: CircuitBreakerConfig,
    state: Arc<ParkingRwLock<CircuitState>>,
    failure_count: Arc<ParkingRwLock<u32>>,
    success_count: Arc<ParkingRwLock<u32>>,
    last_failure_time: Arc<ParkingRwLock<Option<Instant>>>,
    monitoring_start: Arc<ParkingRwLock<Instant>>,
}

impl CircuitBreaker {
    pub fn new(name: String, config: CircuitBreakerConfig) -> Self {
        Self {
            name,
            config,
            state: Arc::new(ParkingRwLock::new(CircuitState::Closed)),
            failure_count: Arc::new(ParkingRwLock::new(0)),
            success_count: Arc::new(ParkingRwLock::new(0)),
            last_failure_time: Arc::new(ParkingRwLock::new(None)),
            monitoring_start: Arc::new(ParkingRwLock::new(Instant::now())),
        }
    }
    
    pub async fn execute<F, T, E>(&self, operation: F) -> CoreResult<T>
    where
        F: FnOnce() -> Result<T, E>,
        E: std::fmt::Display,
    {
        // Check circuit state
        if !self.can_execute() {
            return Err(CoreError::TaskExecution(format!(
                "Circuit breaker '{}' is open", self.name
            )));
        }
        
        // Execute operation
        match operation() {
            Ok(result) => {
                self.record_success();
                Ok(result)
            }
            Err(error) => {
                self.record_failure();
                Err(CoreError::TaskExecution(format!(
                    "Operation failed: {}", error
                )))
            }
        }
    }
    
    pub async fn execute_async<F, Fut, T, E>(&self, operation: F) -> CoreResult<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: std::fmt::Display,
    {
        // Check circuit state
        if !self.can_execute() {
            return Err(CoreError::TaskExecution(format!(
                "Circuit breaker '{}' is open", self.name
            )));
        }
        
        // Execute async operation
        match operation().await {
            Ok(result) => {
                self.record_success();
                Ok(result)
            }
            Err(error) => {
                self.record_failure();
                Err(CoreError::TaskExecution(format!(
                    "Async operation failed: {}", error
                )))
            }
        }
    }
    
    fn can_execute(&self) -> bool {
        let state = *self.state.read();
        
        match state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if recovery timeout has passed
                if let Some(last_failure) = *self.last_failure_time.read() {
                    let elapsed = last_failure.elapsed();
                    elapsed >= Duration::from_millis(self.config.recovery_timeout_ms)
                } else {
                    true
                }
            }
            CircuitState::HalfOpen => true,
        }
    }
    
    fn record_success(&self) {
        let mut state = self.state.write();
        
        match *state {
            CircuitState::Closed => {
                // Reset failure count on success in closed state
                *self.failure_count.write() = 0;
            }
            CircuitState::HalfOpen => {
                let mut success_count = self.success_count.write();
                *success_count += 1;
                
                if *success_count >= self.config.success_threshold {
                    *state = CircuitState::Closed;
                    *self.failure_count.write() = 0;
                    *self.success_count.write() = 0;
                    info!("Circuit breaker '{}' closed after recovery", self.name);
                }
            }
            CircuitState::Open => {
                // Should not happen, but handle gracefully
                *state = CircuitState::HalfOpen;
                *self.success_count.write() = 1;
                debug!("Circuit breaker '{}' moved to half-open state", self.name);
            }
        }
    }
    
    fn record_failure(&self) {
        let mut state = self.state.write();
        let mut failure_count = self.failure_count.write();
        *failure_count += 1;
        
        // Record failure time
        *self.last_failure_time.write() = Some(Instant::now());
        
        match *state {
            CircuitState::Closed => {
                if *failure_count >= self.config.failure_threshold {
                    *state = CircuitState::Open;
                    warn!("Circuit breaker '{}' opened after {} failures", self.name, failure_count);
                }
            }
            CircuitState::HalfOpen => {
                // Immediately open on failure in half-open state
                *state = CircuitState::Open;
                warn!("Circuit breaker '{}' re-opened during recovery", self.name);
            }
            CircuitState::Open => {
                // Already open, just update count
            }
        }
    }
    
    pub fn get_state(&self) -> CircuitState {
        *self.state.read()
    }
    
    pub fn get_stats(&self) -> CircuitBreakerStats {
        CircuitBreakerStats {
            name: self.name.clone(),
            state: self.get_state(),
            failure_count: *self.failure_count.read(),
            success_count: *self.success_count.read(),
            last_failure_time: *self.last_failure_time.read(),
        }
    }
}

/// Circuit breaker statistics
#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    pub name: String,
    pub state: CircuitState,
    pub failure_count: u32,
    pub success_count: u32,
    pub last_failure_time: Option<Instant>,
}

/// Retry policy configuration
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f64,
    pub jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay_ms: 1000,
            max_delay_ms: 30000,
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

/// Retry handler dengan exponential backoff
pub struct RetryHandler {
    policy: RetryPolicy,
}

impl RetryHandler {
    pub fn new(policy: RetryPolicy) -> Self {
        Self { policy }
    }
    
    pub async fn execute_with_retry<F, Fut, T, E>(&self, mut operation: F) -> CoreResult<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: std::fmt::Display,
    {
        let mut last_error: Option<String> = None;
        
        for attempt in 1..=self.policy.max_attempts {
            match operation().await {
                Ok(result) => {
                    if attempt > 1 {
                        info!("Operation succeeded on attempt {}", attempt);
                    }
                    return Ok(result);
                }
                Err(error) => {
                    last_error = Some(format!("{}", error));
                    
                    if attempt < self.policy.max_attempts {
                        let delay = self.calculate_delay(attempt - 1);
                        warn!("Operation failed on attempt {}, retrying in {}ms: {}", 
                              attempt, delay, error);
                        tokio::time::sleep(Duration::from_millis(delay)).await;
                    }
                }
            }
        }
        
        Err(CoreError::TaskExecution(format!(
            "Operation failed after {} attempts: {}", 
            self.policy.max_attempts,
            last_error.unwrap_or_else(|| "Unknown error".to_string())
        )))
    }
    
    fn calculate_delay(&self, attempt: u32) -> u64 {
        let mut delay = self.policy.base_delay_ms as f64 * 
                        self.policy.backoff_multiplier.powi(attempt as i32);
        
        if self.policy.jitter {
            // Add ±25% jitter
            let jitter_factor = 0.75 + (rand::random::<f64>() * 0.5);
            delay *= jitter_factor;
        }
        
        delay.clamp(0.0, self.policy.max_delay_ms as f64) as u64
    }
}

impl Default for RetryHandler {
    fn default() -> Self {
        Self::new(RetryPolicy::default())
    }
}

/// Error recovery manager
pub struct ErrorRecoveryManager {
    circuit_breakers: Arc<RwLock<HashMap<String, CircuitBreaker>>>,
    error_history: Arc<RwLock<Vec<ErrorInfo>>>,
    recovery_strategies: Arc<RwLock<HashMap<ErrorCategory, RecoveryStrategy>>>,
    retry_handler: RetryHandler,
}

/// Recovery strategy untuk different error types
#[derive(Debug, Clone)]
pub struct RecoveryStrategy {
    pub category: ErrorCategory,
    pub auto_retry: bool,
    pub retry_policy: RetryPolicy,
    pub fallback_enabled: bool,
    pub escalation_threshold: u32,
}

impl RecoveryStrategy {
    pub fn new(category: ErrorCategory) -> Self {
        Self {
            category,
            auto_retry: true,
            retry_policy: RetryPolicy::default(),
            fallback_enabled: false,
            escalation_threshold: 5,
        }
    }
    
    pub fn with_auto_retry(mut self, auto_retry: bool) -> Self {
        self.auto_retry = auto_retry;
        self
    }
    
    pub fn with_retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }
    
    pub fn with_fallback(mut self, fallback_enabled: bool) -> Self {
        self.fallback_enabled = fallback_enabled;
        self
    }
}

impl ErrorRecoveryManager {
    pub fn new() -> Self {
        let manager = Self {
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
            error_history: Arc::new(RwLock::new(Vec::new())),
            recovery_strategies: Arc::new(RwLock::new(HashMap::new())),
            retry_handler: RetryHandler::default(),
        };
        
        // Initialize default recovery strategies
        manager.init_default_strategies();
        manager
    }
    
    fn init_default_strategies(&self) {
        // Initialize strategies using a simple approach without blocking
        let strategies = self.recovery_strategies.clone();
        
        // Use std::thread::spawn to avoid blocking in async context
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async move {
                let mut strategies_guard = strategies.write().await;
                
                // Network errors - aggressive retry with circuit breaker
                strategies_guard.insert(ErrorCategory::Network, RecoveryStrategy::new(ErrorCategory::Network)
                    .with_auto_retry(true)
                    .with_retry_policy(RetryPolicy {
                        max_attempts: 5,
                        base_delay_ms: 500,
                        max_delay_ms: 10000,
                        backoff_multiplier: 1.5,
                        jitter: true,
                    })
                    .with_fallback(true));
                
                // Database errors - moderate retry with connection pool reset
                strategies_guard.insert(ErrorCategory::Database, RecoveryStrategy::new(ErrorCategory::Database)
                    .with_auto_retry(true)
                    .with_retry_policy(RetryPolicy {
                        max_attempts: 3,
                        base_delay_ms: 1000,
                        max_delay_ms: 5000,
                        backoff_multiplier: 2.0,
                        jitter: true,
                    })
                    .with_fallback(true));
                
                // Validation errors - no retry, immediate fallback
                strategies_guard.insert(ErrorCategory::Validation, RecoveryStrategy::new(ErrorCategory::Validation)
                    .with_auto_retry(false)
                    .with_fallback(true));
                
                // Model errors - limited retry with fallback
                strategies_guard.insert(ErrorCategory::Model, RecoveryStrategy::new(ErrorCategory::Model)
                    .with_auto_retry(true)
                    .with_fallback(true)
                    .with_retry_policy(RetryPolicy {
                        max_attempts: 2,
                        base_delay_ms: 200,
                        max_delay_ms: 1000,
                        backoff_multiplier: 1.5,
                        jitter: false,
                    }));
                
                // Timeout errors - no retry
                strategies_guard.insert(ErrorCategory::Timeout, RecoveryStrategy::new(ErrorCategory::Timeout)
                    .with_auto_retry(false));
                
                // Resource errors - no retry, escalate
                strategies_guard.insert(ErrorCategory::Resource, RecoveryStrategy::new(ErrorCategory::Resource)
                    .with_auto_retry(false));
            });
        });
    }
    
    /// Get or create circuit breaker for service
    pub async fn get_circuit_breaker(&self, service_name: &str) -> CircuitBreaker {
        let mut breakers = self.circuit_breakers.write().await;
        
        if !breakers.contains_key(service_name) {
            let breaker = CircuitBreaker::new(
                service_name.to_string(),
                CircuitBreakerConfig::default()
            );
            breakers.insert(service_name.to_string(), breaker);
        }
        
        breakers.get(service_name).unwrap().clone()
    }
    
    /// Handle error with recovery strategy
    pub async fn handle_error(&self, error_info: ErrorInfo) -> CoreResult<RecoveryAction> {
        // Record error
        self.record_error(error_info.clone()).await;
        
        // Get recovery strategy
        let strategies = self.recovery_strategies.read().await;
        let strategy = strategies.get(&error_info.category);
        
        // Determine recovery action
        let action = if let Some(strategy) = strategy {
            self.determine_recovery_action(&error_info, strategy).await
        } else {
            RecoveryAction::LogOnly
        };
        
        info!("Error handled: category={:?}, severity={:?}, action={:?}", 
              error_info.category, error_info.severity, action);
        
        Ok(action)
    }
    
    async fn record_error(&self, error_info: ErrorInfo) {
        let mut history = self.error_history.write().await;
        history.push(error_info.clone());
        
        // Keep only last 1000 errors
        if history.len() > 1000 {
            history.remove(0);
        }
        
        // Update circuit breaker if applicable
        if error_info.severity >= ErrorSeverity::High {
            let breaker = self.get_circuit_breaker(&error_info.source).await;
            breaker.record_failure();
        }
    }
    
    async fn determine_recovery_action(&self, error_info: &ErrorInfo, strategy: &RecoveryStrategy) -> RecoveryAction {
        if !error_info.recoverable {
            return RecoveryAction::Escalate;
        }
        
        if error_info.retry_count >= strategy.escalation_threshold {
            return RecoveryAction::Escalate;
        }
        
        if !strategy.auto_retry || error_info.retry_count >= strategy.retry_policy.max_attempts {
            if strategy.fallback_enabled {
                return RecoveryAction::UseFallback;
            } else {
                return RecoveryAction::Escalate;
            }
        }
        
        RecoveryAction::Retry
    }
    
    /// Execute operation with error recovery
    pub async fn execute_with_recovery<F, Fut, T, E>(
        &self,
        service_name: &str,
        operation: F,
    ) -> CoreResult<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: std::fmt::Display,
    {
        let breaker = self.get_circuit_breaker(service_name).await;
        
        breaker.execute_async(operation).await
    }
    
    /// Get error statistics
    pub async fn get_error_stats(&self) -> ErrorStats {
        let history = self.error_history.read().await;
        let breakers = self.circuit_breakers.read().await;
        
        let mut stats = ErrorStats::default();
        
        for error in history.iter() {
            stats.total_errors += 1;
            
            match error.severity {
                ErrorSeverity::Low => stats.low_severity += 1,
                ErrorSeverity::Medium => stats.medium_severity += 1,
                ErrorSeverity::High => stats.high_severity += 1,
                ErrorSeverity::Critical => stats.critical_severity += 1,
            }
            
            *stats.by_category.entry(error.category).or_insert(0) += 1;
        }
        
        stats.active_circuit_breakers = breakers.len();
        stats.open_circuit_breakers = breakers.values()
            .filter(|b| b.get_state() == CircuitState::Open)
            .count();
        
        stats
    }
    
    /// Clear error history
    pub async fn clear_history(&self) {
        let mut history = self.error_history.write().await;
        history.clear();
        info!("Error history cleared");
    }
}

/// Recovery action types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryAction {
    Retry,
    UseFallback,
    Escalate,
    LogOnly,
    Ignore,
}

/// Error statistics
#[derive(Debug, Clone, Default)]
pub struct ErrorStats {
    pub total_errors: u64,
    pub low_severity: u64,
    pub medium_severity: u64,
    pub high_severity: u64,
    pub critical_severity: u64,
    pub by_category: HashMap<ErrorCategory, u64>,
    pub active_circuit_breakers: usize,
    pub open_circuit_breakers: usize,
}

impl Default for ErrorRecoveryManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_circuit_breaker() {
        let breaker = CircuitBreaker::new(
            "test".to_string(),
            CircuitBreakerConfig {
                failure_threshold: 3,
                recovery_timeout_ms: 1000,
                success_threshold: 2,
                monitoring_period_ms: 5000,
            }
        );
        
        // Initially closed
        assert_eq!(breaker.get_state(), CircuitState::Closed);
        
        // Record failures
        for _ in 0..3 {
            breaker.record_failure();
        }
        
        // Should be open now
        assert_eq!(breaker.get_state(), CircuitState::Open);
        
        // Record success (should move to half-open)
        breaker.record_success();
        assert_eq!(breaker.get_state(), CircuitState::HalfOpen);
    }
    
    #[tokio::test]
    async fn test_retry_handler() {
        let handler = RetryHandler::new(RetryPolicy {
            max_attempts: 3,
            base_delay_ms: 10,
            max_delay_ms: 100,
            backoff_multiplier: 2.0,
            jitter: false,
        });
        
        let call_count = std::sync::Arc::new(std::sync::Mutex::new(0));
        let result = handler.execute_with_retry(|| {
            let call_count = call_count.clone();
            async move {
                let mut count = call_count.lock().unwrap();
                *count += 1;
                let current_count = *count;
                drop(count);
                
                if current_count < 3 {
                    Err("Simulated error")
                } else {
                    Ok("success")
                }
            }
        }).await;
        
        assert!(result.is_ok());
        assert_eq!(*call_count.lock().unwrap(), 3);
    }
    
    #[tokio::test]
    async fn test_error_recovery_manager() {
        let manager = ErrorRecoveryManager::new();
        
        // Wait a bit for strategies to be initialized
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        let error_info = ErrorInfo::new(
            ErrorCategory::Network,
            ErrorSeverity::Medium,
            "Connection failed".to_string(),
            "test_service".to_string()
        );
        
        let action = manager.handle_error(error_info).await.unwrap();
        assert_eq!(action, RecoveryAction::Retry);
        
        let stats = manager.get_error_stats().await;
        assert_eq!(stats.total_errors, 1);
        assert_eq!(stats.medium_severity, 1);
    }
}
