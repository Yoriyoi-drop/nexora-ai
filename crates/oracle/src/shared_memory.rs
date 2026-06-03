use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use chrono::Utc;
use serde::{Deserialize, Serialize};

/// Entry di shared memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub value: String,
    pub hit_count: u64,
    pub created_at: i64,
    pub last_access: i64,
    pub ttl_seconds: u64,
    pub metadata: HashMap<String, String>,
}

impl MemoryEntry {
    pub fn new(value: String, ttl_seconds: u64) -> Self {
        let now = Utc::now().timestamp();
        Self {
            value,
            hit_count: 0,
            created_at: now,
            last_access: now,
            ttl_seconds,
            metadata: HashMap::new(),
        }
    }

    pub fn is_expired(&self) -> bool {
        let now = Utc::now().timestamp();
        (now - self.last_access) > self.ttl_seconds as i64
    }
}

/// Shared Memory — global store untuk cache Oracle, dibaca semua NXR instance
pub struct SharedOracleMemory {
    data: HashMap<String, MemoryEntry>,
    max_entries: usize,
    total_inserts: u64,
    total_hits: u64,
    total_misses: u64,
    total_evictions: u64,
}

impl SharedOracleMemory {
    pub fn new(max_entries: usize) -> Self {
        Self {
            data: HashMap::with_capacity(max_entries),
            max_entries,
            total_inserts: 0,
            total_hits: 0,
            total_misses: 0,
            total_evictions: 0,
        }
    }

    pub fn get(&mut self, key: &str) -> Option<&MemoryEntry> {
        let entry = match self.data.get_mut(key) {
            Some(e) => e,
            None => {
                self.total_misses += 1;
                return None;
            }
        };
        entry.hit_count += 1;
        entry.last_access = Utc::now().timestamp();
        self.total_hits += 1;
        Some(entry)
    }

    pub fn get_value(&mut self, key: &str) -> Option<String> {
        let entry = match self.data.get_mut(key) {
            Some(e) => e,
            None => {
                self.total_misses += 1;
                return None;
            }
        };
        entry.hit_count += 1;
        entry.last_access = Utc::now().timestamp();
        self.total_hits += 1;
        Some(entry.value.clone())
    }

    pub fn set(&mut self, key: String, value: String, ttl_seconds: u64) {
        if self.data.len() >= self.max_entries && !self.data.contains_key(&key) {
            self.evict_lru(1);
        }
        let entry = MemoryEntry::new(value, ttl_seconds);
        self.data.insert(key, entry);
        self.total_inserts += 1;
    }

    pub fn remove(&mut self, key: &str) -> Option<MemoryEntry> {
        self.data.remove(key)
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn contains(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn evict_expired(&mut self) -> usize {
        let expired_keys: Vec<String> = self
            .data
            .iter()
            .filter(|(_, e)| e.is_expired())
            .map(|(k, _)| k.clone())
            .collect();
        let count = expired_keys.len();
        for key in expired_keys {
            self.data.remove(&key);
            self.total_evictions += 1;
        }
        count
    }

    pub fn evict_lru(&mut self, count: usize) -> usize {
        let mut entries: Vec<(String, i64)> = self
            .data
            .iter()
            .map(|(k, e)| (k.clone(), e.last_access))
            .collect();
        entries.sort_by(|a, b| a.1.cmp(&b.1));
        let to_evict = entries.into_iter().take(count);
        let mut evicted = 0;
        for (key, _) in to_evict {
            if self.data.remove(&key).is_some() {
                self.total_evictions += 1;
                evicted += 1;
            }
        }
        evicted
    }

    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            entries: self.data.len(),
            max_entries: self.max_entries,
            total_inserts: self.total_inserts,
            total_hits: self.total_hits,
            total_misses: self.total_misses,
            total_evictions: self.total_evictions,
            hit_rate: if self.total_hits + self.total_misses > 0 {
                self.total_hits as f64 / (self.total_hits + self.total_misses) as f64
            } else {
                0.0
            },
        }
    }
}

impl Default for SharedOracleMemory {
    fn default() -> Self {
        Self::new(10_000)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub entries: usize,
    pub max_entries: usize,
    pub total_inserts: u64,
    pub total_hits: u64,
    pub total_misses: u64,
    pub total_evictions: u64,
    pub hit_rate: f64,
}

static GLOBAL_ORACLE_MEMORY: OnceLock<RwLock<SharedOracleMemory>> = OnceLock::new();

pub fn global_oracle_memory() -> &'static RwLock<SharedOracleMemory> {
    GLOBAL_ORACLE_MEMORY
        .get_or_init(|| RwLock::new(SharedOracleMemory::new(10_000)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_and_get() {
        let mut mem = SharedOracleMemory::new(100);
        mem.set("key1".into(), "value1".into(), 3600);
        assert_eq!(mem.get_value("key1"), Some("value1".into()));
    }

    #[test]
    fn test_miss_returns_none() {
        let mut mem = SharedOracleMemory::new(100);
        assert_eq!(mem.get_value("nonexistent"), None);
    }

    #[test]
    fn test_eviction_lru() {
        let mut mem = SharedOracleMemory::new(3);
        mem.set("a".into(), "1".into(), 3600);
        mem.set("b".into(), "2".into(), 3600);
        mem.set("c".into(), "3".into(), 3600);
        mem.set("d".into(), "4".into(), 3600);
        assert_eq!(mem.len(), 3);
        assert!(mem.get_value("d").is_some());
    }

    #[test]
    fn test_stats() {
        let mut mem = SharedOracleMemory::new(10);
        mem.set("k".into(), "v".into(), 3600);
        mem.get_value("k");
        mem.get_value("missing");
        let stats = mem.stats();
        assert_eq!(stats.total_hits, 1);
        assert_eq!(stats.total_misses, 1); // get_value("missing") increment miss
    }

    #[test]
    fn test_global_memory_access() {
        let mem = global_oracle_memory();
        let mut guard = mem.write().unwrap();
        guard.set("global_key".into(), "global_val".into(), 3600);
        assert_eq!(guard.get_value("global_key"), Some("global_val".into()));
    }

    #[test]
    fn test_remove() {
        let mut mem = SharedOracleMemory::new(10);
        mem.set("x".into(), "y".into(), 3600);
        assert!(mem.contains("x"));
        mem.remove("x");
        assert!(!mem.contains("x"));
    }

    #[test]
    fn test_clear() {
        let mut mem = SharedOracleMemory::new(10);
        mem.set("a".into(), "1".into(), 3600);
        mem.set("b".into(), "2".into(), 3600);
        mem.clear();
        assert!(mem.is_empty());
    }
}
