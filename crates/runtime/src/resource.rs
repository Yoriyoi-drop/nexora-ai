//! Resource Management Module
//!
//! Provides resource pooling and management capabilities,
//! including VRAM budget tracking for OOM prevention.

use crate::vram_budget::{VramBudget, VramPressure, VramReservation};
use crate::{InferenceError, Result};
use std::sync::{Arc, Mutex};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Resource manager for handling system resources
pub struct ResourceManager {
    semaphore: Arc<Semaphore>,
    vram_budget: Arc<Mutex<VramBudget>>,
    max_vram_bytes: u64,
}

impl ResourceManager {
    /// Create new resource manager
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            vram_budget: Arc::new(Mutex::new(VramBudget::new(24_000_000_000))),
            max_vram_bytes: 24_000_000_000,
        }
    }

    /// Create with explicit VRAM budget
    pub fn with_vram(max_concurrent: usize, total_vram_bytes: u64) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            vram_budget: Arc::new(Mutex::new(VramBudget::new(total_vram_bytes))),
            max_vram_bytes: total_vram_bytes,
        }
    }

    /// Acquire resource — returns a permit that MUST be held for the duration of use
    pub async fn acquire(&self) -> Result<ResourceGuard> {
        let permit = self.semaphore.clone().acquire_owned().await.map_err(|_| {
            InferenceError::ResourceExhausted("Failed to acquire resource".to_string())
        })?;
        let pressure = tokio::task::block_in_place(|| self.vram_budget.lock().unwrap_or_else(|e| e.into_inner())).pressure();
        Ok(ResourceGuard { _permit: permit, _vram_pressure: pressure })
    }

    /// Acquire with VRAM check: rejects if allocation would cause OOM.
    pub async fn acquire_with_vram(&self, needed_vram: u64) -> Result<ResourceGuardVram> {
        // Check VRAM availability first
        {
            let budget = tokio::task::block_in_place(|| self.vram_budget.lock().unwrap_or_else(|e| e.into_inner()));
            if !budget.can_allocate(needed_vram) {
                return Err(InferenceError::ResourceExhausted(format!(
                    "VRAM: need {} bytes, only {} available (pressure={:?})",
                    needed_vram,
                    budget.available(),
                    budget.pressure()
                )).into());
            }
        }
        // Then acquire semaphore
        let permit = self.semaphore.clone().acquire_owned().await.map_err(|_| {
            InferenceError::ResourceExhausted("Failed to acquire resource".to_string())
        })?;
        let reservation = {
            let mut budget = tokio::task::block_in_place(|| self.vram_budget.lock().unwrap_or_else(|e| e.into_inner()));
            budget.reserve(needed_vram, self.vram_budget.clone())
                .map_err(|e| InferenceError::ResourceExhausted(e))?
        };
        Ok(ResourceGuardVram { _permit: permit, _reservation: Some(reservation) })
    }

    /// Check current VRAM pressure
    pub fn vram_pressure(&self) -> VramPressure {
        self.vram_budget.lock().unwrap_or_else(|e| e.into_inner()).pressure()
    }

    /// Configure VRAM budget for a specific model
    pub fn auto_configure_vram(&self, model_params: usize, num_experts: usize, bits_per_weight: usize) {
        let mut budget = self.vram_budget.lock().unwrap_or_else(|e| e.into_inner());
        budget.auto_configure(model_params, num_experts, bits_per_weight);
    }

    /// Get VRAM budget reference for external updates (e.g., offloader)
    pub fn vram_budget(&self) -> Arc<Mutex<VramBudget>> {
        self.vram_budget.clone()
    }

    /// Log VRAM status
    pub fn print_vram_status(&self) {
        let budget = self.vram_budget.lock().unwrap_or_else(|e| e.into_inner());
        budget.print_status();
    }
}

/// Guard for acquired resources — permit is alive as long as this guard lives
pub struct ResourceGuard {
    _permit: OwnedSemaphorePermit,
    _vram_pressure: VramPressure,
}

impl ResourceGuard {
    pub fn new(_permit: OwnedSemaphorePermit) -> Self {
        Self { _permit, _vram_pressure: VramPressure::Ok }
    }
}

/// Guard with VRAM reservation — VRAM released on drop
#[must_use]
pub struct ResourceGuardVram {
    _permit: OwnedSemaphorePermit,
    _reservation: Option<VramReservation>,
}
