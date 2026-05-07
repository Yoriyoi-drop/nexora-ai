//! Error Recovery Mechanisms
//! 
//! Comprehensive error handling and recovery strategies for Nexora AI

use anyhow::{Result, Context};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{warn, error, info, debug};

/// Error recovery strategy
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Retry with exponential backoff
    ExponentialBackoff { max_attempts: u32, base_delay: Duration },
    /// Retry with fixed delay
    FixedDelay { attempts: u32, delay: Duration },
    /// Circuit breaker pattern
    CircuitBreaker { failure_threshold: u32, recovery_timeout: Duration },
    /// Fail fast
    FailFast,
}

/// Error recovery configuration
#[derive(Debug, Clone)]
pub struct RecoveryConfig {
    pub strategy: RecoveryStrategy,
    pub max_retry_delay: Duration,
    pub timeout: Duration,
    pub jitter: bool,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            strategy: RecoveryStrategy::ExponentialBackoff {
                max_attempts: 3,
                base_delay: Duration::from_millis(100),
            },
            max_retry_delay: Duration::from_secs(30),
            timeout: Duration::from_secs(60),
            jitter: true,
        }
    }
}

/// Error recovery manager
pub struct ErrorRecovery {
    config: RecoveryConfig,
    failure_count: std::sync::atomic::AtomicU32,
    last_failure: std::sync::Mutex<Option<Instant>>,
}

impl ErrorRecovery {
    pub fn new(config: RecoveryConfig) -> Self {
        Self {
            config,
            failure_count: std::sync::atomic::AtomicU32::new(0),
            last_failure: std::sync::Mutex::new(None),
        }
    }
    
    /// Execute operation with error recovery
    pub async fn execute_with_recovery<F, T, E>(&self, operation: F) -> Result<T>
    where
        F: Fn() -> Result<T, E>,
        E: std::fmt::Display + Send + Sync + 'static,
    {
        match &self.config.strategy {
            RecoveryStrategy::ExponentialBackoff { max_attempts, base_delay } => {
                self.exponential_backoff(operation, *max_attempts, *base_delay).await
            }
            RecoveryStrategy::FixedDelay { attempts, delay } => {
                self.fixed_delay(operation, *attempts, *delay).await
            }
            RecoveryStrategy::CircuitBreaker { failure_threshold, recovery_timeout } => {
                self.circuit_breaker(operation, *failure_threshold, *recovery_timeout).await
            }
            RecoveryStrategy::FailFast => {
                operation().map_err(|e| anyhow::anyhow!("{}", e))
            }
        }
    }
    
    /// Execute with exponential backoff
    async fn exponential_backoff<F, T, E>(&self, operation: F, max_attempts: u32, base_delay: Duration) -> Result<T>
    where
        F: Fn() -> Result<T, E>,
        E: std::fmt::Display + Send + Sync + 'static,
    {
        let mut attempt = 0;
        let mut delay = base_delay;
        
        loop {
            attempt += 1;
            
            match operation() {
                Ok(result) => {
                    if attempt > 1 {
                        info!("Operation succeeded on attempt {}", attempt);
                    }
                    self.reset_failure_count();
                    return Ok(result);
                }
                Err(e) => {
                    self.increment_failure_count();
                    
                    if attempt >= max_attempts {
                        error!(
                            attempt = attempt,
                            max_attempts = max_attempts,
                            error = %e,
                            "Operation failed after maximum retry attempts"
                        );
                        return Err(anyhow::anyhow!("Operation failed after {} attempts: {}", max_attempts, e));
                    }
                    
                    let actual_delay = if self.config.jitter {
                        self.add_jitter(delay)
                    } else {
                        delay
                    };
                    
                    warn!(
                        attempt = attempt,
                        max_attempts = max_attempts,
                        delay_ms = actual_delay.as_millis(),
                        error = %e,
                        "Operation failed, retrying after delay"
                    );
                    
                    sleep(actual_delay).await;
                    delay = std::cmp::min(delay * 2, self.config.max_retry_delay);
                }
            }
        }
    }
    
    /// Execute with fixed delay
    async fn fixed_delay<F, T, E>(&self, operation: F, attempts: u32, delay: Duration) -> Result<T>
    where
        F: Fn() -> Result<T, E>,
        E: std::fmt::Display + Send + Sync + 'static,
    {
        for attempt in 1..=attempts {
            match operation() {
                Ok(result) => {
                    if attempt > 1 {
                        info!("Operation succeeded on attempt {}", attempt);
                    }
                    self.reset_failure_count();
                    return Ok(result);
                }
                Err(e) => {
                    self.increment_failure_count();
                    
                    if attempt == attempts {
                        error!(
                            attempt = attempt,
                            attempts = attempts,
                            error = %e,
                            "Operation failed after maximum retry attempts"
                        );
                        return Err(anyhow::anyhow!("Operation failed after {} attempts: {}", attempts, e));
                    }
                    
                    warn!(
                        attempt = attempt,
                        attempts = attempts,
                        delay_ms = delay.as_millis(),
                        error = %e,
                        "Operation failed, retrying after delay"
                    );
                    
                    sleep(delay).await;
                }
            }
        }
        
        unreachable!()
    }
    
    /// Execute with circuit breaker pattern
    async fn circuit_breaker<F, T, E>(&self, operation: F, failure_threshold: u32, recovery_timeout: Duration) -> Result<T>
    where
        F: Fn() -> Result<T, E>,
        E: std::fmt::Display + Send + Sync + 'static,
    {
        let current_failures = self.failure_count.load(std::sync::atomic::Ordering::Relaxed);
        
        // Check if circuit is open
        if current_failures >= failure_threshold {
            if let Some(last_failure_time) = *self.last_failure.lock().unwrap() {
                if last_failure_time.elapsed() < recovery_timeout {
                    error!(
                        failures = current_failures,
                        threshold = failure_threshold,
                        recovery_time_ms = recovery_timeout.as_millis(),
                        "Circuit breaker is open, rejecting operation"
                    );
                    return Err(anyhow::anyhow!("Circuit breaker is open"));
                } else {
                    info!("Circuit breaker recovery timeout reached, attempting reset");
                    self.reset_failure_count();
                }
            }
        }
        
        match operation() {
            Ok(result) => {
                self.reset_failure_count();
                Ok(result)
            }
            Err(e) => {
                self.increment_failure_count();
                let new_failures = self.failure_count.load(std::sync::atomic::Ordering::Relaxed);
                
                if new_failures >= failure_threshold {
                    error!(
                        failures = new_failures,
                        threshold = failure_threshold,
                        error = %e,
                        "Failure threshold reached, opening circuit breaker"
                    );
                } else {
                    warn!(
                        failures = new_failures,
                        threshold = failure_threshold,
                        error = %e,
                        "Operation failed"
                    );
                }
                
                Err(anyhow::anyhow!("{}", e))
            }
        }
    }
    
    /// Add jitter to delay to prevent thundering herd
    fn add_jitter(&self, delay: Duration) -> Duration {
        use rand::Rng;
        let jitter_range = delay.as_millis() as f64 * 0.1; // 10% jitter
        let jitter = rand::thread_rng().gen_range(-jitter_range..=jitter_range) as i64;
        Duration::from_millis((delay.as_millis() as i64 + jitter).max(0) as u64)
    }
    
    /// Increment failure count
    fn increment_failure_count(&self) {
        self.failure_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        *self.last_failure.lock().unwrap() = Some(Instant::now());
    }
    
    /// Reset failure count
    fn reset_failure_count(&self) {
        self.failure_count.store(0, std::sync::atomic::Ordering::Relaxed);
        *self.last_failure.lock().unwrap() = None;
    }
    
    /// Get current failure count
    pub fn failure_count(&self) -> u32 {
        self.failure_count.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    /// Check if circuit is open
    pub fn is_circuit_open(&self) -> bool {
        if let RecoveryStrategy::CircuitBreaker { failure_threshold, recovery_timeout } = &self.config.strategy {
            let current_failures = self.failure_count.load(std::sync::atomic::Ordering::Relaxed);
            
            if current_failures >= *failure_threshold {
                if let Some(last_failure_time) = *self.last_failure.lock().unwrap() {
                    last_failure_time.elapsed() < *recovery_timeout
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        }
    }
}

/// Macro for easy error recovery
#[macro_export]
macro_rules! with_recovery {
    ($config:expr, $operation:expr) => {
        {
            let recovery = $crate::error_recovery::ErrorRecovery::new($config);
            recovery.execute_with_recovery($operation).await
        }
    };
}

/// Macro for retry with exponential backoff
#[macro_export]
macro_rules! retry_exponential {
    ($operation:expr, $max_attempts:expr, $base_delay:expr) => {
        {
            let config = $crate::error_recovery::RecoveryConfig {
                strategy: $crate::error_recovery::RecoveryStrategy::ExponentialBackoff {
                    max_attempts: $max_attempts,
                    base_delay: $base_delay,
                },
                ..Default::default()
            };
            $crate::with_recovery!(config, $operation)
        }
    };
}

/// Macro for retry with fixed delay
#[macro_export]
macro_rules! retry_fixed {
    ($operation:expr, $attempts:expr, $delay:expr) => {
        {
            let config = $crate::error_recovery::RecoveryConfig {
                strategy: $crate::error_recovery::RecoveryStrategy::FixedDelay {
                    attempts: $attempts,
                    delay: $delay,
                },
                ..Default::default()
            };
            $crate::with_recovery!(config, $operation)
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    
    #[tokio::test]
    async fn test_exponential_backoff_success() {
        let attempts = AtomicU32::new(0);
        
        let operation = || {
            attempts.fetch_add(1, Ordering::Relaxed);
            if attempts.load(Ordering::Relaxed) < 2 {
                Err("Simulated failure")
            } else {
                Ok("success")
            }
        };
        
        let config = RecoveryConfig {
            strategy: RecoveryStrategy::ExponentialBackoff {
                max_attempts: 3,
                base_delay: Duration::from_millis(10),
            },
            ..Default::default()
        };
        
        let recovery = ErrorRecovery::new(config);
        let result = recovery.execute_with_recovery(operation).await;
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        assert_eq!(attempts.load(Ordering::Relaxed), 2);
    }
    
    #[tokio::test]
    async fn test_circuit_breaker() {
        let config = RecoveryConfig {
            strategy: RecoveryStrategy::CircuitBreaker {
                failure_threshold: 2,
                recovery_timeout: Duration::from_millis(100),
            },
            ..Default::default()
        };
        
        let recovery = ErrorRecovery::new(config);
        
        // First failure
        let result1: Result<String> = recovery.execute_with_recovery(|| Err("failure")).await;
        assert!(result1.is_err());
        assert!(!recovery.is_circuit_open());
        
        // Second failure - should open circuit
        let result2: Result<String> = recovery.execute_with_recovery(|| Err("failure")).await;
        assert!(result2.is_err());
        assert!(recovery.is_circuit_open());
        
        // Third attempt - should be rejected
        let result3: Result<String> = recovery.execute_with_recovery(|| Ok("success")).await;
        assert!(result3.is_err());
        assert!(recovery.is_circuit_open());
    }
}
