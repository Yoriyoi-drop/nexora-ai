use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::backbone::OracleBackboneConfig;
use crate::OracleBackbone;

/// Status instance Oracle di pool
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OracleInstanceStatus {
    Idle,
    Busy,
    Error,
    Evicted,
}

/// Instance Oracle dalam pool
pub struct OraclePoolInstance {
    pub id: usize,
    pub backbone: Option<OracleBackbone>,
    pub status: OracleInstanceStatus,
    pub last_used: Instant,
    pub total_uses: u64,
    pub created_at: Instant,
}

impl OraclePoolInstance {
    fn new(id: usize) -> Self {
        Self {
            id,
            backbone: None,
            status: OracleInstanceStatus::Idle,
            last_used: Instant::now(),
            total_uses: 0,
            created_at: Instant::now(),
        }
    }
}

/// Pool metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OraclePoolMetrics {
    pub pool_size: usize,
    pub max_size: usize,
    pub active: usize,
    pub idle: usize,
    pub total_acquires: u64,
    pub total_releases: u64,
    pub total_timeouts: u64,
    pub avg_acquire_ms: f64,
}

/// Oracle Pool — maksimal 2 instance, NXR lain antre
pub struct OraclePool {
    instances: Vec<Mutex<OraclePoolInstance>>,
    max_size: usize,
    vocab_size: usize,
    backbone_config: OracleBackboneConfig,
    total_acquires: AtomicU64,
    total_releases: AtomicU64,
    total_timeouts: AtomicU64,
    acquire_times_ns: AtomicU64,
    acquire_count: AtomicU64,
}

impl OraclePool {
    pub fn new(max_size: usize, vocab_size: usize) -> Self {
        Self::new_with_config(max_size, vocab_size, OracleBackboneConfig::default())
    }

    pub fn new_with_config(max_size: usize, vocab_size: usize, backbone_config: OracleBackboneConfig) -> Self {
        let instances = (0..max_size)
            .map(|id| Mutex::new(OraclePoolInstance::new(id)))
            .collect();

        Self {
            instances,
            max_size,
            vocab_size,
            backbone_config,
            total_acquires: AtomicU64::new(0),
            total_releases: AtomicU64::new(0),
            total_timeouts: AtomicU64::new(0),
            acquire_times_ns: AtomicU64::new(0),
            acquire_count: AtomicU64::new(0),
        }
    }

    /// Acquire instance Oracle dari pool. Blocking sampai ada yang available.
    /// Pool harus dibungkus `Arc` — gunakan `acquire_arc()`.
    #[deprecated(since = "0.2.0", note = "use acquire_arc() with Arc<OraclePool>")]
    pub fn acquire(&self) -> Result<PooledOracle, PoolError> {
        let start = Instant::now();
        self.total_acquires.fetch_add(1, Ordering::SeqCst);

        loop {
            for instance in &self.instances {
                let mut guard = instance.lock().map_err(|e| PoolError::LockError(e.to_string()))?;
                if guard.status == OracleInstanceStatus::Idle {
                    guard.status = OracleInstanceStatus::Busy;
                    guard.last_used = Instant::now();
                    guard.total_uses += 1;

                    if guard.backbone.is_none() {
                        guard.backbone = Some(OracleBackbone::new(self.backbone_config.clone(), self.vocab_size));
                    }

                    let elapsed = start.elapsed();
                    self.acquire_times_ns.fetch_add(elapsed.as_nanos() as u64, Ordering::SeqCst);
                    self.acquire_count.fetch_add(1, Ordering::SeqCst);

                    return Ok(PooledOracle {
                        id: guard.id,
                        pool: None,
                        released: false,
                        _private: (),
                    });
                }
            }

            if start.elapsed() > std::time::Duration::from_secs(30) {
                self.total_timeouts.fetch_add(1, Ordering::SeqCst);
                return Err(PoolError::Timeout);
            }

            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    /// Acquire dari pool yang dibungkus `Arc<OraclePool>` — aman terhadap use-after-free.
    pub fn acquire_arc(pool: &Arc<Self>) -> Result<PooledOracle, PoolError> {
        let start = Instant::now();
        pool.total_acquires.fetch_add(1, Ordering::SeqCst);

        loop {
            for instance in &pool.instances {
                let mut guard = instance.lock().map_err(|e| PoolError::LockError(e.to_string()))?;
                if guard.status == OracleInstanceStatus::Idle {
                    guard.status = OracleInstanceStatus::Busy;
                    guard.last_used = Instant::now();
                    guard.total_uses += 1;

                    if guard.backbone.is_none() {
                        guard.backbone = Some(OracleBackbone::new(pool.backbone_config.clone(), pool.vocab_size));
                    }

                    let elapsed = start.elapsed();
                    pool.acquire_times_ns.fetch_add(elapsed.as_nanos() as u64, Ordering::SeqCst);
                    pool.acquire_count.fetch_add(1, Ordering::SeqCst);

                    return Ok(PooledOracle {
                        id: guard.id,
                        pool: Some(Arc::downgrade(pool)),
                        released: false,
                        _private: (),
                    });
                }
            }

            if start.elapsed() > std::time::Duration::from_secs(30) {
                pool.total_timeouts.fetch_add(1, Ordering::SeqCst);
                return Err(PoolError::Timeout);
            }

            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    /// Release instance kembali ke pool (dipanggil otomatis oleh Drop)
    fn release(&self, id: usize) {
        if let Some(instance) = self.instances.get(id) {
            if let Ok(mut guard) = instance.lock() {
                guard.status = OracleInstanceStatus::Idle;
                self.total_releases.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    /// Dapatkan reference ke backbone
    pub fn with_backbone<F, R>(&self, id: usize, f: F) -> Result<R, PoolError>
    where
        F: FnOnce(&OracleBackbone) -> R,
    {
        if let Some(instance) = self.instances.get(id) {
            let guard = instance.lock().map_err(|e| PoolError::LockError(e.to_string()))?;
            if let Some(ref backbone) = guard.backbone {
                Ok(f(backbone))
            } else {
                Err(PoolError::NotInitialized)
            }
        } else {
            Err(PoolError::InvalidId(id))
        }
    }

    /// Metrics
    pub fn metrics(&self) -> OraclePoolMetrics {
        let mut active = 0;
        let mut idle = 0;

        for instance in &self.instances {
            if let Ok(guard) = instance.lock() {
                match guard.status {
                    OracleInstanceStatus::Busy => active += 1,
                    OracleInstanceStatus::Idle => idle += 1,
                    _ => {}
                }
            }
        }

        let avg_ms = if self.acquire_count.load(Ordering::SeqCst) > 0 {
            let total_ns = self.acquire_times_ns.load(Ordering::SeqCst);
            let count = self.acquire_count.load(Ordering::SeqCst);
            total_ns as f64 / count as f64 / 1_000_000.0
        } else {
            0.0
        };

        OraclePoolMetrics {
            pool_size: self.instances.len(),
            max_size: self.max_size,
            active,
            idle,
            total_acquires: self.total_acquires.load(Ordering::SeqCst),
            total_releases: self.total_releases.load(Ordering::SeqCst),
            total_timeouts: self.total_timeouts.load(Ordering::SeqCst),
            avg_acquire_ms: avg_ms,
        }
    }
}

impl Default for OraclePool {
    fn default() -> Self {
        Self::new_with_config(2, 50000, OracleBackboneConfig::default())
    }
}

/// Handle PooledOracle — auto-release saat Drop.
/// Holds a `Weak<OraclePool>` to prevent use-after-free if the pool is dropped
/// before all handles are released.
pub struct PooledOracle {
    id: usize,
    pool: Option<Weak<OraclePool>>,
    /// Cleared after release to detect double-drop
    released: bool,
    _private: (),
}

impl PooledOracle {
    pub fn id(&self) -> usize {
        self.id
    }
}

impl Drop for PooledOracle {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        if let Some(weak) = self.pool.take() {
            if let Some(pool) = weak.upgrade() {
                pool.release(self.id);
            }
        }
    }
}

#[derive(Debug)]
pub enum PoolError {
    Timeout,
    LockError(String),
    NotInitialized,
    InvalidId(usize),
}

impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoolError::Timeout => write!(f, "Oracle pool: timeout waiting for available instance"),
            PoolError::LockError(e) => write!(f, "Oracle pool: lock error: {}", e),
            PoolError::NotInitialized => write!(f, "Oracle pool: instance not initialized"),
            PoolError::InvalidId(id) => write!(f, "Oracle pool: invalid instance id: {}", id),
        }
    }
}

impl std::error::Error for PoolError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_create() {
        let pool = OraclePool::new(2, 1000);
        let metrics = pool.metrics();
        assert_eq!(metrics.max_size, 2);
        assert_eq!(metrics.pool_size, 2);
    }

    fn tiny_config() -> OracleBackboneConfig {
        OracleBackboneConfig {
            d_model: 16,
            n_heads: 2,
            n_experts: 1,
            top_k: 1,
            latent_dim: 8,
            context_size: 64,
            mlp_hidden: 32,
            dropout: 0.0,
        }
    }

    #[test]
    fn test_acquire_release() {
        let pool = Arc::new(OraclePool::new_with_config(2, 1000, tiny_config()));
        {
            let _oracle = OraclePool::acquire_arc(&pool).expect("acquire should succeed");
            let metrics = pool.metrics();
            assert_eq!(metrics.active, 1);
            assert_eq!(metrics.idle, 1);
        }
        let metrics = pool.metrics();
        assert_eq!(metrics.active, 0);
        assert_eq!(metrics.idle, 2);
    }

    #[test]
    fn test_with_backbone() {
        let pool = Arc::new(OraclePool::new_with_config(1, 1000, tiny_config()));
        let oracle = OraclePool::acquire_arc(&pool).expect("acquire");
        let id = oracle.id();
        let result = pool.with_backbone(id, |bb| {
            format!("d_model={}", bb.config.d_model)
        });
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "d_model=16");
    }

    #[test]
    fn test_pool_metrics_tracking() {
        let pool = Arc::new(OraclePool::new_with_config(2, 1000, tiny_config()));
        let m1 = pool.metrics();
        assert_eq!(m1.pool_size, 2);
        assert_eq!(m1.total_acquires, 0);

        let handle = OraclePool::acquire_arc(&pool).expect("acquire");
        drop(handle);

        let m2 = pool.metrics();
        assert_eq!(m2.total_acquires, 1);
        assert_eq!(m2.total_releases, 1);
    }
}
