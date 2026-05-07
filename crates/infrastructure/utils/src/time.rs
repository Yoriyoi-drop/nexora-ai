//! Time utilities untuk Nexora

use std::time::{SystemTime, UNIX_EPOCH, Duration};
use anyhow::Result;
use chrono::{DateTime, Utc, Local};

pub struct TimeUtils;

impl TimeUtils {
    /// Get current timestamp in milliseconds
    pub fn current_timestamp_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
    
    /// Get current timestamp in seconds
    pub fn current_timestamp_s() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
    
    /// Get current timestamp (alias for consistency)
    pub fn current_timestamp() -> u64 {
        Self::current_timestamp_ms()
    }
    
    /// Format timestamp to ISO 8601 string
    pub fn format_timestamp(timestamp: u64) -> Result<String> {
        let datetime = DateTime::<Utc>::from_timestamp(
            (timestamp / 1000) as i64,
            ((timestamp % 1000) * 1_000_000) as u32
        );
        
        match datetime {
            Some(dt) => Ok(dt.to_rfc3339()),
            None => Err(anyhow::anyhow!("Invalid timestamp")),
        }
    }
    
    /// Format timestamp to local time string
    pub fn format_timestamp_local(timestamp: u64) -> Result<String> {
        let datetime = DateTime::<Utc>::from_timestamp(
            (timestamp / 1000) as i64,
            ((timestamp % 1000) * 1_000_000) as u32
        );
        
        match datetime {
            Some(dt) => {
                let local_dt: DateTime<Local> = dt.into();
                Ok(local_dt.to_rfc3339())
            }
            None => Err(anyhow::anyhow!("Invalid timestamp")),
        }
    }
    
    /// Parse ISO 8601 timestamp string
    pub fn parse_timestamp(timestamp_str: &str) -> Result<u64> {
        let datetime = DateTime::parse_from_rfc3339(timestamp_str)?;
        Ok(datetime.timestamp_millis() as u64)
    }
    
    /// Calculate duration between two timestamps
    pub fn duration_between(start: u64, end: u64) -> Duration {
        Duration::from_millis(end.saturating_sub(start))
    }
    
    /// Format duration to human readable string
    pub fn format_duration(duration: Duration) -> String {
        let total_seconds = duration.as_secs();
        let hours = total_seconds / 3600;
        let minutes = (total_seconds % 3600) / 60;
        let seconds = total_seconds % 60;
        let milliseconds = duration.subsec_millis();
        
        if hours > 0 {
            format!("{}h {}m {}s {}ms", hours, minutes, seconds, milliseconds)
        } else if minutes > 0 {
            format!("{}m {}s {}ms", minutes, seconds, milliseconds)
        } else if seconds > 0 {
            format!("{}s {}ms", seconds, milliseconds)
        } else {
            format!("{}ms", milliseconds)
        }
    }
    
    /// Check if timestamp is within last N seconds
    pub fn is_within_last_seconds(timestamp: u64, seconds: u64) -> bool {
        let now = Self::current_timestamp_s();
        timestamp >= now.saturating_sub(seconds)
    }
    
    /// Check if timestamp is within last N minutes
    pub fn is_within_last_minutes(timestamp: u64, minutes: u64) -> bool {
        Self::is_within_last_seconds(timestamp, minutes * 60)
    }
    
    /// Check if timestamp is within last N hours
    pub fn is_within_last_hours(timestamp: u64, hours: u64) -> bool {
        Self::is_within_last_seconds(timestamp, hours * 3600)
    }
    
    /// Get timestamp for N seconds ago
    pub fn seconds_ago(seconds: u64) -> u64 {
        let now = Self::current_timestamp_s();
        now.saturating_sub(seconds)
    }
    
    /// Get timestamp for N minutes ago
    pub fn minutes_ago(minutes: u64) -> u64 {
        Self::seconds_ago(minutes * 60)
    }
    
    /// Get timestamp for N hours ago
    pub fn hours_ago(hours: u64) -> u64 {
        Self::seconds_ago(hours * 3600)
    }
    
    /// Get timestamp for N days ago
    pub fn days_ago(days: u64) -> u64 {
        Self::seconds_ago(days * 86400)
    }
    
    /// Add seconds to timestamp
    pub fn add_seconds(timestamp: u64, seconds: u64) -> u64 {
        timestamp.saturating_add(seconds * 1000)
    }
    
    /// Add minutes to timestamp
    pub fn add_minutes(timestamp: u64, minutes: u64) -> u64 {
        Self::add_seconds(timestamp, minutes * 60)
    }
    
    /// Add hours to timestamp
    pub fn add_hours(timestamp: u64, hours: u64) -> u64 {
        Self::add_seconds(timestamp, hours * 3600)
    }
    
    /// Add days to timestamp
    pub fn add_days(timestamp: u64, days: u64) -> u64 {
        Self::add_seconds(timestamp, days * 86400)
    }
    
    /// Get current date string
    pub fn current_date_string() -> String {
        let now = Self::current_timestamp();
        Self::format_timestamp(now)
            .unwrap_or_else(|_| now.to_string())
            .split('T')
            .next()
            .unwrap_or("unknown")
            .to_string()
    }
    
    /// Get current time string
    pub fn current_time_string() -> String {
        let now = Self::current_timestamp();
        Self::format_timestamp(now)
            .unwrap_or_else(|_| now.to_string())
            .split('T')
            .nth(1)
            .unwrap_or("unknown")
            .split('.')
            .next()
            .unwrap_or("unknown")
            .to_string()
    }
    
    /// Sleep for specified milliseconds
    pub async fn sleep_ms(ms: u64) {
        tokio::time::sleep(Duration::from_millis(ms)).await;
    }
    
    /// Sleep for specified seconds
    pub async fn sleep_s(seconds: u64) {
        tokio::time::sleep(Duration::from_secs(seconds)).await;
    }
    
    /// Measure execution time of a function
    pub async fn measure_time<F, T>(f: F) -> (T, Duration)
    where
        F: std::future::Future<Output = T>,
    {
        let start = SystemTime::now();
        let result = f.await;
        let duration = SystemTime::now().duration_since(start).unwrap_or_default();
        (result, duration)
    }
    
    /// Benchmark function execution
    pub async fn benchmark<F, T>(f: F, iterations: usize) -> (Vec<Duration>, Duration)
    where
        F: Fn() -> T + Clone,
    {
        let mut durations = Vec::with_capacity(iterations);
        let start = SystemTime::now();
        
        for _ in 0..iterations {
            let (_, duration) = Self::measure_time(async { f() }).await;
            durations.push(duration);
        }
        
        let total_duration = SystemTime::now().duration_since(start).unwrap_or_default();
        (durations, total_duration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_time_utils() {
        // Test current timestamp
        let now = TimeUtils::current_timestamp();
        assert!(now > 0);
        
        // Test timestamp formatting
        let formatted = TimeUtils::format_timestamp(now).unwrap();
        assert!(!formatted.is_empty());
        
        // Test duration formatting
        let duration = Duration::from_millis(1500);
        let formatted_duration = TimeUtils::format_duration(duration);
        assert!(formatted_duration.contains("1s"));
        
        // Test time arithmetic
        let past = TimeUtils::minutes_ago(5);
        assert!(past < now);
        
        let future = TimeUtils::add_minutes(past, 10);
        assert!(future > past);
        
        // Test time measurement
        let (result, duration) = TimeUtils::measure_time(async {
            TimeUtils::sleep_ms(10).await;
            "test"
        }).await;
        
        assert_eq!(result, "test");
        assert!(duration >= Duration::from_millis(10));
    }
}
