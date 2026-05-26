use std::future::Future;
use std::time::Duration;

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 100,
            max_delay_ms: 5000,
            jitter: true,
        }
    }
}

impl RetryConfig {
    pub fn new(max_retries: u32, base_delay_ms: u64) -> Self {
        Self {
            max_retries,
            base_delay_ms,
            max_delay_ms: 30_000,
            jitter: true,
        }
    }

    /// Execute a fallible async operation with retry
    pub async fn retry<F, Fut, T, E>(&self, operation: F) -> Result<T, E>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<T, E>>,
        E: std::fmt::Display,
    {
        let mut last_err = None;
        for attempt in 0..=self.max_retries {
            match operation().await {
                Ok(val) => return Ok(val),
                Err(e) => {
                    if attempt == self.max_retries {
                        return Err(e);
                    }
                    let delay = self.calculate_delay(attempt);
                    tracing::warn!(
                        "Operation failed (attempt {}/{}): {}. Retrying in {}ms",
                        attempt + 1,
                        self.max_retries + 1,
                        e,
                        delay
                    );
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                    last_err = Some(e);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| unreachable!("retry loop always executes at least once because max_retries >= 0")))
    }

    pub fn calculate_delay(&self, attempt: u32) -> u64 {
        let multiplier = if attempt < 64 { 2u64.pow(attempt) } else { u64::MAX };
        let delay = self.base_delay_ms.saturating_mul(multiplier);
        let delay = delay.min(self.max_delay_ms);
        if self.jitter {
            delay / 2 + rand::random::<u64>() % (delay / 2 + 1)
        } else {
            delay
        }
    }
}
