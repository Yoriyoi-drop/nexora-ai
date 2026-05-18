//! Training Strategy for HLDVA-T
//!
//! Implementasi 4-tahap curriculum learning:
//! - Stage 0: Pre-training VAE dan CLIP
//! - Stage 1: Base DiT training
//! - Stage 2: Cascaded upsampler training  
//! - Stage 3: End-to-end fine-tuning

pub mod curriculum;
pub mod optimizer;
pub mod scheduler;
pub mod checkpoint;

use self::optimizer::{AdamOptimizer, AdamConfig, Optimizer};

use crate::{
    config::{HLDVAConfig, TrainingConfig},
    types::*,
    HLDVAPipeline,
    ddpm::DDPMLoss,
};
use nexora_atqs::Tensor;

/// Main Trainer
pub struct HLDVATrainer {
    config: HLDVAConfig,
    
    // Pipeline components
    pipeline: HLDVAPipeline,
    
    // Training components
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    loss_calculator: DDPMLoss,
    
    // Training state
    state: TrainingState,
    
    // Checkpoint manager
    checkpoint_manager: CheckpointManager,
}

impl HLDVATrainer {
    /// Create new trainer
    pub fn new(config: HLDVAConfig) -> HLDVAResult<Self> {
        let pipeline = HLDVAPipeline::new(config)?;
        let optimizer = AdamW::new(&pipeline.config.training)?;
        let scheduler = CosineAnnealingLR::new(&pipeline.config.training)?;
        let loss_calculator = DDPMLoss::new(&pipeline.config.ddpm)?;
        
        let state = TrainingState::default();
        let checkpoint_manager = CheckpointManager::new("checkpoints")?;
        
        Ok(Self {
            config: pipeline.config.clone(),
            pipeline,
            optimizer,
            scheduler,
            loss_calculator,
            state,
            checkpoint_manager,
        })
    }
    
    /// Start training dengan 4-tahap curriculum learning
    pub fn train(&mut self, dataset: &dyn Dataset) -> HLDVAResult<()> {
        println!("Starting HLDVA-T training with 4-stage curriculum learning");
        
        // Stage 0: Pre-training
        self.stage_0_pretraining(dataset)?;
        
        // Stage 1: Base DiT training
        self.stage_1_base_dit(dataset)?;
        
        // Stage 2: Cascaded upsampler training
        self.stage_2_cascaded(dataset)?;
        
        // Stage 3: End-to-end fine-tuning
        self.stage_3_finetune(dataset)?;
        
        println!("Training completed!");
        Ok(())
    }
    
    /// Stage 0: Pre-training VAE dan CLIP
    fn stage_0_pretraining(&mut self, dataset: &dyn Dataset) -> HLDVAResult<()> {
        println!("Stage 0: Pre-training VAE and CLIP");
        
        self.state.current_stage = 0;
        
        for epoch in 0..10 { // 10 epochs untuk pre-training
            println!("Pre-training epoch {}/10", epoch + 1);
            
            // Train VAE
            let vae_loss = self.train_vae_epoch(dataset)?;
            
            // Train CLIP (jika perlu)
            let clip_loss = self.train_clip_epoch(dataset)?;
            
            println!("VAE Loss: {:.4}, CLIP Loss: {:.4}", vae_loss, clip_loss);
            
            self.state.current_epoch = epoch;
            self.state.total_steps += 1;
        }
        
        // Freeze VAE dan CLIP
        self.freeze_vae_clip()?;
        
        Ok(())
    }
    
    /// Stage 1: Base DiT training
    fn stage_1_base_dit(&mut self, dataset: &dyn Dataset) -> HLDVAResult<()> {
        println!("Stage 1: Base DiT training");
        
        self.state.current_stage = 1;
        
        for epoch in 0..50 { // 50 epochs untuk base DiT
            println!("Base DiT epoch {}/50", epoch + 1);
            
            let dit_loss = self.train_dit_epoch(dataset)?;
            
            println!("DiT Loss: {:.4}", dit_loss);
            
            self.state.current_epoch = epoch;
            self.state.total_steps += 1;
            
            // Evaluation setiap 10 epochs
            if epoch % 10 == 9 {
                self.evaluate_base_dit()?;
            }
        }
        
        Ok(())
    }
    
    /// Stage 2: Cascaded upsampler training
    fn stage_2_cascaded(&mut self, dataset: &dyn Dataset) -> HLDVAResult<()> {
        println!("Stage 2: Cascaded upsampler training");
        
        self.state.current_stage = 2;
        
        // Train upsampler 1 (64→256)
        self.train_upsampler_stage(dataset, 0)?;
        
        // Train upsampler 2 (256→1024)
        self.train_upsampler_stage(dataset, 1)?;
        
        Ok(())
    }
    
    /// Stage 3: End-to-end fine-tuning
    fn stage_3_finetune(&mut self, dataset: &dyn Dataset) -> HLDVAResult<()> {
        println!("Stage 3: End-to-end fine-tuning");
        
        self.state.current_stage = 3;
        
        // Unfreeze semua komponen kecuali VAE dan CLIP
        self.unfreeze_components()?;
        
        for epoch in 0..20 { // 20 epochs untuk fine-tuning
            println!("Fine-tuning epoch {}/20", epoch + 1);
            
            let finetune_loss = self.train_finetune_epoch(dataset)?;
            
            println!("Fine-tune Loss: {:.4}", finetune_loss);
            
            self.state.current_epoch = epoch;
            self.state.total_steps += 1;
            
            // Save checkpoint setiap 5 epochs
            if epoch % 5 == 4 {
                self.save_checkpoint()?;
            }
        }
        
        Ok(())
    }
    
    /// Train VAE untuk satu epoch
    fn train_vae_epoch(&mut self, dataset: &dyn Dataset) -> HLDVAResult<f32> {
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        
        for batch_idx in 0..dataset.num_batches() {
            let batch = dataset.get_vae_batch(batch_idx)?;
            
            // Forward pass through VAE
            let latent_space = LatentSpace::new(batch.images.clone(), Resolution::new(64, 64), 4);
            let reconstructed = self.pipeline.vae_decoder.decode(&latent_space)?;
            let mu = Tensor::new(vec![0.0], vec![1]);
            let kl_loss = Tensor::new(vec![0.0], vec![1]);
            
            // Calculate reconstruction loss
            let recon_loss = self.mse_loss(&batch.images, &reconstructed)?;
            
            // Total VAE loss
            let vae_loss = recon_loss + self.config.vae.kl_weight * kl_loss.data()[0];
            
            // Backward pass dan update
            self.optimizer.step(vae_loss)?;
            
            total_loss += vae_loss;
            num_batches += 1;
        }
        
        Ok(total_loss / num_batches as f32)
    }
    
    /// Train CLIP untuk satu epoch
    fn train_clip_epoch(&mut self, dataset: &dyn Dataset) -> HLDVAResult<f32> {
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        
        for batch_idx in 0..dataset.num_batches() {
            let batch = dataset.get_clip_batch(batch_idx)?;
            
            // Encode text dan image
            let text_features = self.pipeline.clip_encoder.encode(&batch.prompts[0])?;
            let image_features = self.pipeline.clip_encoder.encode_image(&batch.images)?;
            
            // Calculate contrastive loss
            let clip_loss = self.contrastive_loss(&text_features.text_features, &image_features)?;
            
            // Backward pass dan update
            self.optimizer.step(clip_loss)?;
            
            total_loss += clip_loss;
            num_batches += 1;
        }
        
        Ok(total_loss / num_batches as f32)
    }
    
    /// Train DiT untuk satu epoch
    fn train_dit_epoch(&mut self, dataset: &dyn Dataset) -> HLDVAResult<f32> {
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        
        for batch_idx in 0..dataset.num_batches() {
            let batch = dataset.get_dit_batch(batch_idx)?;
            
            // Encode gambar ke latent
            let latent_space = LatentSpace::new(batch.images.clone(), Resolution::new(64, 64), 4);
            let latent = self.pipeline.vae_decoder.decode(&latent_space)?;
            
            // Sample timestep dan noise
            let timestep = self.sample_timestep();
            let noise = self.sample_noise(latent.shape())?;
            
            // Add noise
            let noisy_latent = self.pipeline.ddpm.add_noise(&latent, &noise, timestep)?;
            
            // Encode text
            let clip_embedding = self.pipeline.clip_encoder.encode(&batch.prompts[0])?;
            
            // Predict noise dengan DiT
            let noise_pred = self.pipeline.dit_model.predict_noise(
                &noisy_latent,
                timestep,
                &clip_embedding,
                1.0,
            )?;
            
            // Calculate loss
            let dit_loss = self.mse_loss(&noise, &noise_pred.predicted_noise)?;
            
            // Backward pass dan update
            self.optimizer.step(dit_loss)?;
            
            total_loss += dit_loss;
            num_batches += 1;
        }
        
        Ok(total_loss / num_batches as f32)
    }
    
    /// Train upsampler stage
    fn train_upsampler_stage(&mut self, dataset: &dyn Dataset, stage_idx: usize) -> HLDVAResult<()> {
        println!("Training upsampler stage {}", stage_idx + 1);
        
        for epoch in 0..25 { // 25 epochs per upsampler
            println!("Upsampler {} epoch {}/25", stage_idx + 1, epoch + 1);
            
            let upsampler_loss = self.train_upsampler_epoch(dataset, stage_idx)?;
            
            println!("Upsampler {} Loss: {:.4}", stage_idx + 1, upsampler_loss);
        }
        
        Ok(())
    }
    
    /// Train upsampler untuk satu epoch
    fn train_upsampler_epoch(&mut self, dataset: &dyn Dataset, stage_idx: usize) -> HLDVAResult<f32> {
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        
        for batch_idx in 0..dataset.num_batches() {
            let batch = dataset.get_upsampler_batch(batch_idx, stage_idx)?;
            
            // Get low-res dan high-res latents
            let low_res_latent = batch.latents.clone();
            let high_res_latent = batch.latents.clone(); // Simplified - using same latent for both
            
            // Encode text
            let clip_embedding = self.pipeline.clip_encoder.encode(&batch.prompts[0])?;
            
            // Upsample dengan cascaded model
            let low_res_latent_space = LatentSpace::new(low_res_latent, Resolution::new(64, 64), 4);
            let upscaled = self.pipeline.cascaded_model.upsample(
                low_res_latent_space,
                &clip_embedding,
                None,
                &self.config.cascaded.upsamplers[stage_idx],
            )?;
            
            // Calculate loss
            let upsampler_loss = self.mse_loss(&high_res_latent, &upscaled.data)?;
            
            // Backward pass dan update
            self.optimizer.step(upsampler_loss)?;
            
            total_loss += upsampler_loss;
            num_batches += 1;
        }
        
        Ok(total_loss / num_batches as f32)
    }
    
    /// Train fine-tuning untuk satu epoch
    fn train_finetune_epoch(&mut self, dataset: &dyn Dataset) -> HLDVAResult<f32> {
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        
        for batch_idx in 0..dataset.num_batches() {
            let batch = dataset.get_finetune_batch(batch_idx)?;
            
            // Full pipeline forward pass
            let input = HLDVAInput {
                prompt: batch.prompts[0].clone(),
                target_resolution: Resolution::new(1024, 1024),
                ..Default::default()
            };
            
            let output = self.pipeline.generate(input)?;
            
            // Calculate loss (simplified - would need ground truth)
            let finetune_loss = self.calculate_finetune_loss(&output, &batch)?;
            
            // Backward pass dan update
            self.optimizer.step(finetune_loss)?;
            
            total_loss += finetune_loss;
            num_batches += 1;
        }
        
        Ok(total_loss / num_batches as f32)
    }
    
    /// Helper functions
    fn freeze_vae_clip(&mut self) -> HLDVAResult<()> {
        println!("Freezing VAE and CLIP parameters");
        // Implementation would freeze parameters
        Ok(())
    }
    
    fn unfreeze_components(&mut self) -> HLDVAResult<()> {
        println!("Unfreezing components for fine-tuning");
        // Implementation would unfreeze parameters
        Ok(())
    }
    
    fn evaluate_base_dit(&self) -> HLDVAResult<()> {
        println!("Evaluating base DiT...");
        let metrics = self.pipeline.evaluate()?;
        println!("CLIP Score: {:.4}", metrics.clip_score.unwrap_or(0.0));
        Ok(())
    }
    
    fn save_checkpoint(&self) -> HLDVAResult<()> {
        let checkpoint_path = format!("stage_{}_epoch_{}", self.state.current_stage, self.state.current_epoch);
        self.checkpoint_manager.save(&self.pipeline, &checkpoint_path)?;
        println!("Saved checkpoint: {}", checkpoint_path);
        Ok(())
    }
    
    fn sample_timestep(&self) -> Timestep {
        Timestep(rand::random::<usize>() % self.config.ddpm.num_timesteps)
    }
    
    fn sample_noise(&self, shape: &[usize]) -> HLDVAResult<Tensor> {
        let total_size = shape.iter().product();
        let mut noise = Vec::with_capacity(total_size);
        
        for _ in 0..total_size {
            noise.push(self.randn());
        }
        
        Ok(Tensor::new(noise, shape.to_vec()))
    }
    
    fn randn(&self) -> f32 {
        use std::f64::consts::PI;
        let u1: f64 = rand::random();
        let u2: f64 = rand::random();
        
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        z0 as f32
    }
    
    fn mse_loss(&self, target: &Tensor, prediction: &Tensor) -> HLDVAResult<f32> {
        let target_data = target.data();
        let pred_data = prediction.data();
        
        let mut mse = 0.0;
        let count = target_data.len().min(pred_data.len());
        
        for i in 0..count {
            let diff = target_data[i] - pred_data[i];
            mse += diff * diff;
        }
        
        Ok(mse / count as f32)
    }
    
    fn contrastive_loss(&self, text_features: &Tensor, image_features: &Tensor) -> HLDVAResult<f32> {
        // Simplified contrastive loss
        let similarity = self.cosine_similarity(text_features, image_features);
        Ok(1.0 - similarity) // Maximize similarity
    }
    
    fn cosine_similarity(&self, a: &Tensor, b: &Tensor) -> f32 {
        let a_data = a.data();
        let b_data = b.data();
        
        let mut dot_product = 0.0;
        let mut a_norm_sq = 0.0;
        let mut b_norm_sq = 0.0;
        
        for i in 0..a_data.len().min(b_data.len()) {
            dot_product += a_data[i] * b_data[i];
            a_norm_sq += a_data[i] * a_data[i];
            b_norm_sq += b_data[i] * b_data[i];
        }
        
        let a_norm = a_norm_sq.sqrt();
        let b_norm = b_norm_sq.sqrt();
        
        if a_norm > 0.0 && b_norm > 0.0 {
            dot_product / (a_norm * b_norm)
        } else {
            0.0
        }
    }
    
    fn calculate_finetune_loss(&self, output: &HLDVAOutput, batch: &TrainingBatch) -> HLDVAResult<f32> {
        // Simplified fine-tuning loss
        let clip_score = output.metrics.clip_score.unwrap_or(0.0);
        Ok(1.0 - clip_score) // Maximize CLIP score
    }
    
    /// Get training state
    pub fn state(&self) -> &TrainingState {
        &self.state
    }
}

/// Dataset trait
pub trait Dataset {
    fn num_batches(&self) -> usize;
    fn get_vae_batch(&self, batch_idx: usize) -> HLDVAResult<TrainingBatch>;
    fn get_clip_batch(&self, batch_idx: usize) -> HLDVAResult<TrainingBatch>;
    fn get_dit_batch(&self, batch_idx: usize) -> HLDVAResult<TrainingBatch>;
    fn get_upsampler_batch(&self, batch_idx: usize, stage_idx: usize) -> HLDVAResult<TrainingBatch>;
    fn get_finetune_batch(&self, batch_idx: usize) -> HLDVAResult<TrainingBatch>;
}

/// AdamW Optimizer — wraps the full AdamOptimizer with decoupled weight decay
pub struct AdamW {
    inner: AdamOptimizer,
    weight_decay: f32,
    params: Vec<nexora_atqs::Tensor>,
    grads: Vec<nexora_atqs::Tensor>,
}

impl AdamW {
    pub fn new(config: &TrainingConfig) -> HLDVAResult<Self> {
        let adam_config = AdamConfig {
            learning_rate: config.learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: config.weight_decay,
        };
        Ok(Self {
            inner: AdamOptimizer::new(adam_config),
            weight_decay: config.weight_decay,
            params: Vec::new(),
            grads: Vec::new(),
        })
    }

    pub fn register_parameters(&mut self, parameters: Vec<nexora_atqs::Tensor>) {
        let shapes: Vec<_> = parameters.iter().map(|p| p.shape().to_vec()).collect();
        self.params = parameters;
        self.grads = shapes.into_iter()
            .map(|s| nexora_atqs::Tensor::new(vec![0.0; s.iter().product()], s))
            .collect();
    }

    pub fn zero_grad(&mut self) {
        for grad in &mut self.grads {
            for val in grad.data_mut().iter_mut() {
                *val = 0.0;
            }
        }
    }

    pub fn step(&mut self, _loss: f32) -> HLDVAResult<()> {
        if self.params.is_empty() {
            return Ok(());
        }
        // Apply decoupled weight decay (AdamW-style)
        for param in &mut self.params {
            for val in param.data_mut().iter_mut() {
                *val *= 1.0 - self.inner.learning_rate() * self.weight_decay;
            }
        }
        // Adam update via inner optimizer
        Optimizer::step(&mut self.inner, &mut self.params, &self.grads)
    }
}

/// Cosine Annealing Learning Rate Scheduler
pub struct CosineAnnealingLR {
    initial_lr: f32,
    t_max: usize,
}

impl CosineAnnealingLR {
    pub fn new(config: &TrainingConfig) -> HLDVAResult<Self> {
        Ok(Self {
            initial_lr: config.learning_rate,
            t_max: config.num_epochs,
        })
    }
    
    pub fn get_lr(&self, epoch: usize) -> f32 {
        let progress = epoch as f32 / self.t_max as f32;
        self.initial_lr * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
    }
}

/// Checkpoint Manager
pub struct CheckpointManager {
    checkpoint_dir: String,
}

impl CheckpointManager {
    pub fn new(dir: &str) -> HLDVAResult<Self> {
        std::fs::create_dir_all(dir)?;
        Ok(Self {
            checkpoint_dir: dir.to_string(),
        })
    }
    
    pub fn save(&self, pipeline: &HLDVAPipeline, name: &str) -> HLDVAResult<()> {
        let path = format!("{}/{}.json", self.checkpoint_dir, name);
        pipeline.save_checkpoint(&path)?;
        Ok(())
    }
    
    pub fn load(&mut self, pipeline: &mut HLDVAPipeline, name: &str) -> HLDVAResult<()> {
        let path = format!("{}/{}.json", self.checkpoint_dir, name);
        pipeline.load_checkpoint(&path)?;
        Ok(())
    }
}
