//! Active-Standby Scheduler untuk training 10 NXR paralel.
//!
//! Alih-alih menjalankan semua 10 model secara bersamaan (memboroskan RAM/VRAM),
//! hanya `num_active` model yang aktif pada satu waktu. Sisanya dalam status
//! standby — model di-unload dari memory dan di-load kembali saat gilirannya tiba.
//!
//! Strategi:
//! - 2 model aktif, 8 model standby
//! - Rotasi setiap N step
//! - Model standby menyimpan state ke checkpoint, hanya menyimpan konfigurasi
//! - Model yang akan aktif me-load checkpoint terakhir

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tracing::info;

use nexora_shared::model_identity::NxrModelId;
use crate::causal_lm_model::CausalLmModel;

/// Konfigurasi Active-Standby Scheduler
#[derive(Debug, Clone)]
pub struct ActiveStandbyConfig {
    /// Jumlah model yang aktif secara simultan
    pub num_active: usize,
    /// Jumlah total model
    pub num_total: usize,
    /// Rotasi setiap N step training
    pub rotate_every_steps: usize,
    /// Direktori untuk menyimpan checkpoint sementara saat rotasi
    pub swap_dir: PathBuf,
    /// Jika true, model standby tetap di memory (tidak di-unload)
    pub keep_in_memory: bool,
}

impl Default for ActiveStandbyConfig {
    fn default() -> Self {
        Self {
            num_active: 2,
            num_total: 10,
            rotate_every_steps: 50,
            swap_dir: PathBuf::from("/tmp/nexora_swap"),
            keep_in_memory: false,
        }
    }
}

/// Status sebuah model dalam scheduler
#[derive(Debug, Clone, PartialEq, Eq)]
enum ModelStatus {
    Active,
    Standby,
}

/// Informasi tracking per-model
#[derive(Debug, Clone)]
struct ModelState {
    id: NxrModelId,
    status: ModelStatus,
    steps_trained: usize,
    last_checkpoint: Option<PathBuf>,
}

/// Active-Standby Scheduler untuk training multi-model.
///
/// Mengelola rotasi model aktif/standby sehingga hanya `num_active` model
/// yang berada di memory pada satu waktu.
pub struct ActiveStandbyScheduler {
    config: ActiveStandbyConfig,
    models: Vec<ModelState>,
    steps_since_rotation: usize,
    last_rotation: Instant,
    total_rotations: usize,
    checkpoints: HashMap<NxrModelId, PathBuf>,
}

impl ActiveStandbyScheduler {
    /// Buat scheduler baru — default: 2 aktif, 8 standby dari total 10 model.
    pub fn new(config: ActiveStandbyConfig) -> Self {
        let model_ids = NxrModelId::all();
        let num_total = config.num_total.min(model_ids.len());

        let mut models: Vec<ModelState> = model_ids
            .iter()
            .take(num_total)
            .enumerate()
            .map(|(i, id)| ModelState {
                id: *id,
                status: if i < config.num_active { ModelStatus::Active } else { ModelStatus::Standby },
                steps_trained: 0,
                last_checkpoint: None,
            })
            .collect();

        Self {
            config,
            models,
            steps_since_rotation: 0,
            last_rotation: Instant::now(),
            total_rotations: 0,
            checkpoints: HashMap::new(),
        }
    }

    /// Daftar ID model yang aktif saat ini.
    pub fn active_model_ids(&self) -> Vec<NxrModelId> {
        self.models.iter().filter(|m| m.status == ModelStatus::Active).map(|m| m.id).collect()
    }

    /// Daftar ID model yang standby.
    pub fn standby_model_ids(&self) -> Vec<NxrModelId> {
        self.models.iter().filter(|m| m.status == ModelStatus::Standby).map(|m| m.id).collect()
    }

    /// Catat step training. Return true jika perlu rotasi.
    pub fn record_step(&mut self, model_id: NxrModelId) -> bool {
        if let Some(model) = self.models.iter_mut().find(|m| m.id == model_id) {
            model.steps_trained += 1;
        }
        self.steps_since_rotation += 1;
        if self.steps_since_rotation >= self.config.rotate_every_steps {
            self.steps_since_rotation = 0;
            return true;
        }
        false
    }

    /// Pasangan rotasi berikutnya: (deactivate, activate).
    pub fn next_rotation(&self) -> Option<(NxrModelId, NxrModelId)> {
        let first_active = self.models.iter().find(|m| m.status == ModelStatus::Active)?;
        let first_standby = self.models.iter().find(|m| m.status == ModelStatus::Standby)?;
        Some((first_active.id, first_standby.id))
    }

    /// Lakukan rotasi async: save checkpoint → unload → load.
    pub async fn rotate(
        &mut self,
        registry: &crate::shared::model_registry::NxrModelRegistry,
    ) -> Option<(NxrModelId, NxrModelId)> {
        let (deactivate_id, activate_id) = self.next_rotation()?;

        info!(
            "ActiveStandby: rotating {:?} -> standby, {:?} -> active",
            deactivate_id, activate_id
        );

        // 1. Save checkpoint untuk model yang akan dinonaktifkan
        if let Some(model) = self.models.iter().find(|m| m.id == deactivate_id) {
            if let Some(ref ckpt) = model.last_checkpoint {
                save_checkpoint_model(registry, deactivate_id, ckpt).await;
            }
        }

        // 2. Unload model yang akan dinonaktifkan
        if !self.config.keep_in_memory {
            unload_model(registry, deactivate_id).await;
        }
        if let Some(model) = self.models.iter_mut().find(|m| m.id == deactivate_id) {
            model.status = ModelStatus::Standby;
        }

        // 3. Load checkpoint untuk model yang akan diaktifkan
        if let Some(model) = self.models.iter().find(|m| m.id == activate_id) {
            if let Some(ref ckpt) = model.last_checkpoint {
                load_checkpoint_model(registry, activate_id, ckpt).await;
            }
        }
        if let Some(model) = self.models.iter_mut().find(|m| m.id == activate_id) {
            model.status = ModelStatus::Active;
        }

        self.total_rotations += 1;
        self.last_rotation = Instant::now();
        Some((deactivate_id, activate_id))
    }

    /// Set checkpoint path untuk model.
    pub fn set_checkpoint_path(&mut self, model_id: NxrModelId, path: PathBuf) {
        if let Some(model) = self.models.iter_mut().find(|m| m.id == model_id) {
            model.last_checkpoint = Some(path.clone());
        }
        self.checkpoints.insert(model_id, path);
    }

    /// Catat jumlah step.
    pub fn set_steps_trained(&mut self, model_id: NxrModelId, steps: usize) {
        if let Some(model) = self.models.iter_mut().find(|m| m.id == model_id) {
            model.steps_trained = steps;
        }
    }

    /// Statistik.
    pub fn stats(&self) -> ActiveStandbyStats {
        ActiveStandbyStats {
            num_active: self.active_model_ids().len(),
            num_standby: self.standby_model_ids().len(),
            total_rotations: self.total_rotations,
            steps_since_rotation: self.steps_since_rotation,
            active_models: self.active_model_ids(),
            standby_models: self.standby_model_ids(),
            total_models: self.models.len(),
            completed_models: self.models.iter().filter(|m| m.steps_trained > 0).count(),
        }
    }
}

async fn save_checkpoint_model(
    registry: &crate::shared::model_registry::NxrModelRegistry,
    model_id: NxrModelId,
    path: &PathBuf,
) {
    let raw = match registry.get_model_raw(&model_id).await {
        Ok(r) => r,
        Err(e) => { tracing::warn!("save_checkpoint: model {:?} not found: {}", model_id, e); return; }
    };
    let model: Arc<CausalLmModel> = match raw.downcast::<CausalLmModel>() {
        Ok(m) => m,
        Err(_) => { tracing::warn!("save_checkpoint: downcast failed for {:?}", model_id); return; }
    };
    if let Err(e) = model.save_checkpoint(path.to_str().unwrap_or("model.safetensors")).await {
        tracing::warn!("save_checkpoint: save failed for {:?}: {}", model_id, e);
    }
}

async fn load_checkpoint_model(
    registry: &crate::shared::model_registry::NxrModelRegistry,
    model_id: NxrModelId,
    path: &PathBuf,
) {
    let raw = match registry.get_model_raw(&model_id).await {
        Ok(r) => r,
        Err(e) => { tracing::warn!("load_checkpoint: model {:?} not found: {}", model_id, e); return; }
    };
    let model: Arc<CausalLmModel> = match raw.downcast::<CausalLmModel>() {
        Ok(m) => m,
        Err(_) => { tracing::warn!("load_checkpoint: downcast failed for {:?}", model_id); return; }
    };
    if let Err(e) = model.load_checkpoint(path.to_str().unwrap_or("model.safetensors")).await {
        tracing::warn!("load_checkpoint: load failed for {:?}: {}", model_id, e);
    }
}

async fn unload_model(
    registry: &crate::shared::model_registry::NxrModelRegistry,
    model_id: NxrModelId,
) {
    let raw = match registry.get_model_raw(&model_id).await {
        Ok(r) => r,
        Err(_) => return,
    };
    if let Some(causal) = raw.downcast_ref::<CausalLmModel>() {
        causal.unload_model().await;
    }
}

/// Statistik Active-Standby Scheduler
#[derive(Debug, Clone)]
pub struct ActiveStandbyStats {
    pub num_active: usize,
    pub num_standby: usize,
    pub total_rotations: usize,
    pub steps_since_rotation: usize,
    pub active_models: Vec<NxrModelId>,
    pub standby_models: Vec<NxrModelId>,
    pub completed_models: usize,
    pub total_models: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_initial_state() {
        let config = ActiveStandbyConfig { num_active: 2, num_total: 10, ..Default::default() };
        let scheduler = ActiveStandbyScheduler::new(config);
        assert_eq!(scheduler.active_model_ids().len(), 2);
        assert_eq!(scheduler.standby_model_ids().len(), 8);
        assert_eq!(scheduler.total_rotations, 0);
    }

    #[test]
    fn test_next_rotation() {
        let config = ActiveStandbyConfig { num_active: 2, num_total: 5, ..Default::default() };
        let scheduler = ActiveStandbyScheduler::new(config);
        let rotation = scheduler.next_rotation();
        assert!(rotation.is_some());
        let (deactivate, activate) = rotation.unwrap();
        assert_ne!(deactivate, activate);
    }

    #[test]
    fn test_record_step_triggers_rotation() {
        let config = ActiveStandbyConfig { num_active: 2, num_total: 4, rotate_every_steps: 3, ..Default::default() };
        let mut scheduler = ActiveStandbyScheduler::new(config);
        let first = scheduler.active_model_ids()[0];
        let mut needs_rotate = false;
        for _ in 0..3 {
            if scheduler.record_step(first) { needs_rotate = true; }
        }
        assert!(needs_rotate);
        assert_eq!(scheduler.steps_since_rotation, 0);
    }
}
