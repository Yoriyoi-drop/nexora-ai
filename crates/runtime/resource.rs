//! Resource Management
//! 
//! Shared resource management utilities for AI frameworks

use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;
use tracing::{debug, warn};

/// Resource pool for managing reusable resources
pub struct ResourcePool<T> {
    name: String,
    resources: Arc<RwLock<Vec<T>>>,
    max_size: usize,
    created: usize,
    reused: usize,
}

impl<T> ResourcePool<T>
where
    T: Clone,
{
    /// Create new resource pool
    pub fn new(name: String, max_size: usize) -> Self {
        Self {
            name,
            resources: Arc::new(RwLock::new(Vec::new())),
            max_size,
            created: 0,
            reused: 0,
        }
    }
    
    /// Acquire resource from pool
    pub async fn acquire<F>(&mut self, creator: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        // Try to get from pool first
        {
            let mut resources = self.resources.write().await;
            if let Some(resource) = resources.pop() {
                self.reused += 1;
                debug!("ResourcePool {}: Reused resource (created: {}, reused: {})", 
                       self.name, self.created, self.reused);
                return Ok(resource);
            }
        }
        
        // Create new resource
        let resource = creator()?;
        self.created += 1;
        debug!("ResourcePool {}: Created new resource (created: {}, reused: {})", 
               self.name, self.created, self.reused);
        Ok(resource)
    }
    
    /// Return resource to pool
    pub async fn release(&mut self, resource: T) -> Result<()> {
        let mut resources = self.resources.write().await;
        
        if resources.len() < self.max_size {
            resources.push(resource);
            debug!("ResourcePool {}: Resource returned to pool (pool size: {})", 
                   self.name, resources.len());
        } else {
            debug!("ResourcePool {}: Pool full, discarding resource", self.name);
        }
        
        Ok(())
    }
    
    /// Get pool statistics
    pub fn get_statistics(&self) -> ResourcePoolStatistics {
        ResourcePoolStatistics {
            name: self.name.clone(),
            pool_size: self.resources.blocking_read().len(),
            max_size: self.max_size,
            created: self.created,
            reused: self.reused,
            reuse_ratio: if self.created > 0 {
                self.reused as f64 / self.created as f64
            } else {
                0.0
            },
        }
    }
    
    /// Clear pool
    pub async fn clear(&mut self) {
        let mut resources = self.resources.write().await;
        resources.clear();
        debug!("ResourcePool {}: Pool cleared", self.name);
    }
}

/// Resource pool statistics
#[derive(Debug, Clone)]
pub struct ResourcePoolStatistics {
    pub name: String,
    pub pool_size: usize,
    pub max_size: usize,
    pub created: usize,
    pub reused: usize,
    pub reuse_ratio: f64,
}

/// Memory manager for tracking memory usage
pub struct MemoryManager {
    name: String,
    allocations: Arc<RwLock<Vec<MemoryAllocation>>>,
    total_allocated: Arc<RwLock<usize>>,
    peak_usage: Arc<RwLock<usize>>,
    allocation_count: Arc<RwLock<usize>>,
}

impl MemoryManager {
    /// Create new memory manager
    pub fn new(name: String) -> Self {
        Self {
            name,
            allocations: Arc::new(RwLock::new(Vec::new())),
            total_allocated: Arc::new(RwLock::new(0)),
            peak_usage: Arc::new(RwLock::new(0)),
            allocation_count: Arc::new(RwLock::new(0)),
        }
    }
    
    /// Allocate memory
    pub async fn allocate(&self, size: usize, purpose: String) -> Result<MemoryAllocation> {
        let allocation = MemoryAllocation {
            id: format!("{}-{}", self.name, *self.allocation_count.read().await),
            size,
            purpose,
            allocated_at: std::time::SystemTime::now(),
        };
        
        // Update tracking
        {
            let mut allocations = self.allocations.write().await;
            allocations.push(allocation.clone());
        }
        
        {
            let mut total = self.total_allocated.write().await;
            *total += size;
            
            let mut peak = self.peak_usage.write().await;
            if *total > *peak {
                *peak = *total;
            }
        }
        
        {
            let mut count = self.allocation_count.write().await;
            *count += 1;
        }
        
        debug!("MemoryManager {}: Allocated {} bytes for '{}'. Total: {} bytes", 
               self.name, size, allocation.purpose, *self.total_allocated.read().await);
        
        Ok(allocation)
    }
    
    /// Deallocate memory
    pub async fn deallocate(&self, allocation_id: &str) -> Result<bool> {
        let mut allocations = self.allocations.write().await;
        
        if let Some(pos) = allocations.iter().position(|a| a.id == allocation_id) {
            let allocation = allocations.remove(pos);
            
            let mut total = self.total_allocated.write().await;
            *total = total.saturating_sub(allocation.size);
            
            debug!("MemoryManager {}: Deallocated {} bytes from '{}'. Total: {} bytes", 
                   self.name, allocation.size, allocation.purpose, *total);
            
            Ok(true)
        } else {
            warn!("MemoryManager {}: Allocation '{}' not found", self.name, allocation_id);
            Ok(false)
        }
    }
    
    /// Get memory statistics
    pub async fn get_statistics(&self) -> MemoryStatistics {
        let allocations = self.allocations.read().await;
        let total_allocated = *self.total_allocated.read().await;
        let peak_usage = *self.peak_usage.read().await;
        let allocation_count = *self.allocation_count.read().await;
        
        MemoryStatistics {
            name: self.name.clone(),
            current_allocated: total_allocated,
            peak_usage,
            allocation_count,
            active_allocations: allocations.len(),
            average_allocation_size: if allocation_count > 0 {
                total_allocated / allocation_count
            } else {
                0
            },
        }
    }
    
    /// Get allocations by purpose
    pub async fn get_allocations_by_purpose(&self, purpose: &str) -> Vec<MemoryAllocation> {
        let allocations = self.allocations.read().await;
        allocations
            .iter()
            .filter(|a| a.purpose.contains(purpose))
            .cloned()
            .collect()
    }
    
    /// Clear all allocations
    pub async fn clear(&self) {
        let mut allocations = self.allocations.write().await;
        allocations.clear();
        
        let mut total = self.total_allocated.write().await;
        *total = 0;
        
        debug!("MemoryManager {}: All allocations cleared", self.name);
    }
}

/// Memory allocation record
#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    pub id: String,
    pub size: usize,
    pub purpose: String,
    pub allocated_at: std::time::SystemTime,
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    pub name: String,
    pub current_allocated: usize,
    pub peak_usage: usize,
    pub allocation_count: usize,
    pub active_allocations: usize,
    pub average_allocation_size: usize,
}

/// CPU manager for tracking CPU usage
pub struct CpuManager {
    name: String,
    cpu_count: usize,
    last_usage: Arc<RwLock<f64>>,
}

impl CpuManager {
    /// Create new CPU manager
    pub fn new(name: String) -> Self {
        Self {
            name,
            cpu_count: num_cpus::get(),
            last_usage: Arc::new(RwLock::new(0.0)),
        }
    }
    
    /// Get CPU count
    pub fn cpu_count(&self) -> usize {
        self.cpu_count
    }
    
    /// Get current CPU usage (placeholder - would use system monitoring)
    pub async fn get_cpu_usage(&self) -> Result<f64> {
        // This would use actual system monitoring libraries
        // For now, return a simulated value
        let usage = rand::random::<f64>() * 100.0;
        *self.last_usage.write().await = usage;
        Ok(usage)
    }
    
    /// Get last recorded CPU usage
    pub async fn get_last_usage(&self) -> f64 {
        *self.last_usage.read().await
    }
}

impl Default for CpuManager {
    fn default() -> Self {
        Self::new("default".to_string())
    }
}

impl<T> Default for ResourcePool<T>
where
    T: Clone,
{
    fn default() -> Self {
        Self::new("default".to_string(), 10)
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new("default".to_string())
    }
}
