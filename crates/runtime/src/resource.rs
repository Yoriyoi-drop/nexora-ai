//! Resource Management Module
//! 
//! Provides resource pooling and management capabilities

use crate::{Result, InferenceError};
use std::sync::Arc;
use tokio::sync::{Semaphore, OwnedSemaphorePermit};

/// Resource manager for handling system resources
pub struct ResourceManager {
    semaphore: Arc<Semaphore>,
}

impl ResourceManager {
    /// Create new resource manager
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
        }
    }
    
    /// Acquire resource — returns a permit that MUST be held for the duration of use
    pub async fn acquire(&self) -> Result<ResourceGuard> {
        let permit = self.semaphore.acquire_owned().await
            .map_err(|_| InferenceError::ResourceExhausted("Failed to acquire resource".to_string()))?;
        Ok(ResourceGuard { _permit: permit })
    }
}

/// Guard for acquired resources — permit is alive as long as this guard lives
pub struct ResourceGuard {
    _permit: OwnedSemaphorePermit,
}

impl ResourceGuard {
    pub fn new(_permit: OwnedSemaphorePermit) -> Self {
        Self { _permit }
    }
}
