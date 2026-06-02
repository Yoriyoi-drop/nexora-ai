use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Konfigurasi TTL Oracle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleTtlConfig {
    /// Default TTL untuk segment (30 menit)
    pub default_ttl: Duration,
    /// TTL untuk segment yang jarang diakses
    pub cold_ttl: Duration,
    /// TTL untuk segment yang sering diakses
    pub hot_ttl: Duration,
    /// Hit count threshold untuk dianggap "hot"
    pub hot_threshold: u64,
    /// Interval pembersihan segment expired
    pub cleanup_interval: Duration,
    /// Maksimum segment yang bisa diload
    pub max_segments: usize,
}

impl Default for OracleTtlConfig {
    fn default() -> Self {
        Self {
            default_ttl: Duration::from_secs(1800),
            cold_ttl: Duration::from_secs(300),
            hot_ttl: Duration::from_secs(7200),
            hot_threshold: 10,
            cleanup_interval: Duration::from_secs(60),
            max_segments: 100,
        }
    }
}

/// Sebuah segment Oracle dengan metadata TTL
#[derive(Debug, Clone)]
pub struct OracleSegment {
    pub id: String,
    pub data_size_bytes: usize,
    pub hit_count: u64,
    pub created_at: Instant,
    pub last_access: Instant,
    pub ttl: Duration,
    pub is_loaded: bool,
}

impl OracleSegment {
    pub fn new(id: String, ttl: Duration, data_size_bytes: usize) -> Self {
        let now = Instant::now();
        Self {
            id,
            data_size_bytes,
            hit_count: 0,
            created_at: now,
            last_access: now,
            ttl,
            is_loaded: false,
        }
    }

    pub fn access(&mut self) {
        self.hit_count += 1;
        self.last_access = Instant::now();
    }

    pub fn is_expired(&self) -> bool {
        self.last_access.elapsed() > self.ttl
    }

    pub fn time_since_last_access(&self) -> Duration {
        self.last_access.elapsed()
    }

    pub fn total_age(&self) -> Duration {
        self.created_at.elapsed()
    }

    pub fn should_promote_to_hot(&self, threshold: u64) -> bool {
        self.hit_count >= threshold && self.ttl < Duration::from_secs(7200)
    }

    pub fn should_demote_to_cold(&self, threshold: u64) -> bool {
        self.hit_count < threshold && self.ttl > Duration::from_secs(300)
    }
}

/// TTL Manager — mengelola siklus hidup segment Oracle
pub struct OracleTtlManager {
    segments: HashMap<String, OracleSegment>,
    config: OracleTtlConfig,
    total_evictions: u64,
    total_promotions: u64,
    total_demotions: u64,
    last_cleanup: Instant,
}

impl OracleTtlManager {
    pub fn new(config: OracleTtlConfig) -> Self {
        Self {
            segments: HashMap::new(),
            config,
            total_evictions: 0,
            total_promotions: 0,
            total_demotions: 0,
            last_cleanup: Instant::now(),
        }
    }

    pub fn register_segment(&mut self, id: String, data_size_bytes: usize) {
        if self.segments.len() >= self.config.max_segments {
            self.evict_lru();
        }
        let segment = OracleSegment::new(id.clone(), self.config.default_ttl, data_size_bytes);
        self.segments.insert(id, segment);
    }

    pub fn access_segment(&mut self, id: &str) -> Option<&mut OracleSegment> {
        let segment = self.segments.get_mut(id)?;
        segment.access();
        let hit_count = segment.hit_count;

        let hot_ttl = self.config.hot_ttl;
        let cold_ttl = self.config.cold_ttl;
        let hot_threshold = self.config.hot_threshold;

        if hit_count >= hot_threshold && segment.ttl < hot_ttl {
            segment.ttl = hot_ttl;
            self.total_promotions += 1;
        } else if hit_count < hot_threshold / 2 && segment.ttl > cold_ttl {
            segment.ttl = cold_ttl;
            self.total_demotions += 1;
        }
        Some(segment)
    }

    pub fn remove_segment(&mut self, id: &str) -> Option<OracleSegment> {
        self.segments.remove(id)
    }

    pub fn get_segment(&self, id: &str) -> Option<&OracleSegment> {
        self.segments.get(id)
    }

    pub fn contains(&self, id: &str) -> bool {
        self.segments.contains_key(id)
    }

    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    pub fn total_memory_bytes(&self) -> usize {
        self.segments
            .values()
            .map(|s| s.data_size_bytes)
            .sum()
    }

    /// Evict segment yang expired
    pub fn evict_expired(&mut self) -> usize {
        let expired: Vec<String> = self
            .segments
            .iter()
            .filter(|(_, s)| s.is_expired())
            .map(|(id, _)| id.clone())
            .collect();

        let count = expired.len();
        for id in expired {
            self.segments.remove(&id);
            self.total_evictions += 1;
        }
        count
    }

    /// Evict LRU segment
    pub fn evict_lru(&mut self) -> Option<String> {
        let lru_id = self
            .segments
            .iter()
            .min_by_key(|(_, s)| s.last_access)
            .map(|(id, _)| id.clone())?;

        self.segments.remove(&lru_id);
        self.total_evictions += 1;
        Some(lru_id)
    }

    /// Periodic cleanup — panggil tiap N detik
    pub fn periodic_cleanup(&mut self) -> CleanupReport {
        let evicted = self.evict_expired();
        self.last_cleanup = Instant::now();

        CleanupReport {
            evicted_expired: evicted,
            remaining_segments: self.segments.len(),
            total_memory_bytes: self.total_memory_bytes(),
            total_evictions: self.total_evictions,
        }
    }

    pub fn stats(&self) -> TtlStats {
        let active = self
            .segments
            .values()
            .filter(|s| s.is_loaded)
            .count();
        let loaded = self
            .segments
            .values()
            .filter(|s| !s.is_loaded)
            .count();

        TtlStats {
            total_segments: self.segments.len(),
            active_segments: active,
            unloaded_segments: loaded,
            total_memory_bytes: self.total_memory_bytes(),
            total_evictions: self.total_evictions,
            total_promotions: self.total_promotions,
            total_demotions: self.total_demotions,
        }
    }

    pub fn needs_cleanup(&self) -> bool {
        self.last_cleanup.elapsed() > self.config.cleanup_interval
    }
}

impl Default for OracleTtlManager {
    fn default() -> Self {
        Self::new(OracleTtlConfig::default())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupReport {
    pub evicted_expired: usize,
    pub remaining_segments: usize,
    pub total_memory_bytes: usize,
    pub total_evictions: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtlStats {
    pub total_segments: usize,
    pub active_segments: usize,
    pub unloaded_segments: usize,
    pub total_memory_bytes: usize,
    pub total_evictions: u64,
    pub total_promotions: u64,
    pub total_demotions: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_segment_creation() {
        let segment = OracleSegment::new("test".into(), Duration::from_secs(1800), 1024);
        assert_eq!(segment.id, "test");
        assert_eq!(segment.hit_count, 0);
        assert!(!segment.is_expired());
    }

    #[test]
    fn test_access_increments_hit_count() {
        let mut segment = OracleSegment::new("test".into(), Duration::from_secs(1800), 1024);
        segment.access();
        assert_eq!(segment.hit_count, 1);
    }

    #[test]
    fn test_register_and_access() {
        let mut mgr = OracleTtlManager::default();
        mgr.register_segment("seg1".into(), 2048);
        assert!(mgr.contains("seg1"));

        let seg = mgr.access_segment("seg1");
        assert!(seg.is_some());
        assert_eq!(seg.unwrap().hit_count, 1);
    }

    #[test]
    fn test_evict_expired() {
        let mut mgr = OracleTtlManager::new(OracleTtlConfig {
            default_ttl: Duration::from_millis(1),
            ..Default::default()
        });
        mgr.register_segment("expired".into(), 100);
        std::thread::sleep(Duration::from_millis(2));
        let evicted = mgr.evict_expired();
        assert_eq!(evicted, 1);
        assert!(!mgr.contains("expired"));
    }

    #[test]
    fn test_evict_lru() {
        let mut mgr = OracleTtlManager::new(OracleTtlConfig {
            max_segments: 2,
            ..Default::default()
        });
        mgr.register_segment("a".into(), 100);
        mgr.register_segment("b".into(), 100);
        mgr.register_segment("c".into(), 100); // should evict LRU
        assert!(mgr.contains("b") || mgr.contains("c"));
    }

    #[test]
    fn test_ttl_adjustment() {
        let mut mgr = OracleTtlManager::new(OracleTtlConfig {
            hot_threshold: 3,
            cold_ttl: Duration::from_secs(300),
            hot_ttl: Duration::from_secs(7200),
            default_ttl: Duration::from_secs(1800),
            ..Default::default()
        });
        mgr.register_segment("hot_potato".into(), 100);
        for _ in 0..5 {
            mgr.access_segment("hot_potato");
        }
        let seg = mgr.get_segment("hot_potato").unwrap();
        assert_eq!(seg.ttl, Duration::from_secs(7200));
    }

    #[test]
    fn test_stats() {
        let mut mgr = OracleTtlManager::default();
        mgr.register_segment("s1".into(), 100);
        mgr.register_segment("s2".into(), 200);
        mgr.access_segment("s1");
        mgr.evict_lru();
        let stats = mgr.stats();
        assert_eq!(stats.total_evictions, 1);
    }
}
