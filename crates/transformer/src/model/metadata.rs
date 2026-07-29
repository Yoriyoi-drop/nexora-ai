use ndarray::Array2;
use tracing::info;
use crate::model::builder::CausalLM;
use crate::{TransformerResult, TransformerError};

impl CausalLM {
    pub fn parameter_count(&self) -> usize {
        let mut count = self.token_embedding.as_ref().map_or(0, |w| w.len());
        if !self.weight_tied {
            count += self.lm_head.as_ref().map_or(0, |w| w.len());
        }
        for block in &self.blocks {
            count += block.attention.wq.as_ref().map_or(0, |w| w.len());
            count += block.attention.wk.as_ref().map_or(0, |w| w.len());
            count += block.attention.wv.as_ref().map_or(0, |w| w.len());
            count += block.attention.wo.as_ref().map_or(0, |w| w.len());
            count += block.ffn.w1.as_ref().map_or(0, |w| w.len());
            count += block.ffn.w2.as_ref().map_or(0, |w| w.len());
            count += block.ffn.w3.as_ref().map_or(0, |w| w.len());
            count += block.attention_norm.weight.as_ref().map_or(0, |w| w.len());
            count += block.ffn_norm.weight.as_ref().map_or(0, |w| w.len());
        }
        count += self.norm.weight.as_ref().map_or(0, |w| w.len());
        count
    }

    /// Collect all 2D weight matrices for SEDC compression.
    /// Returns (weights, names, fusion_pairs).
    pub fn collect_weights_for_sedc(&self) -> (Vec<Array2<f32>>, Vec<String>, Vec<(usize, usize)>) {
        let mut weights: Vec<Array2<f32>> = Vec::new();
        let mut names: Vec<String> = Vec::new();

        weights.push(self.token_embedding.clone().unwrap_or(Array2::zeros((0, 0))));
        names.push("token_embedding".to_string());

        if self.weight_tied {
            weights.push(Array2::zeros((0, 0)));
            names.push("lm_head (tied)".to_string());
        } else {
            weights.push(self.lm_head.clone().unwrap_or(Array2::zeros((0, 0))));
            names.push("lm_head".to_string());
        }

        for (i, block) in self.blocks.iter().enumerate() {
            let prefix = format!("blocks.{}", i);
            weights.push(block.attention.wq.clone().unwrap_or(Array2::zeros((0, 0))));
            names.push(format!("{}.attention.wq", prefix));
            weights.push(block.attention.wk.clone().unwrap_or(Array2::zeros((0, 0))));
            names.push(format!("{}.attention.wk", prefix));
            weights.push(block.attention.wv.clone().unwrap_or(Array2::zeros((0, 0))));
            names.push(format!("{}.attention.wv", prefix));
            weights.push(block.attention.wo.clone().unwrap_or(Array2::zeros((0, 0))));
            names.push(format!("{}.attention.wo", prefix));
            weights.push(block.ffn.w1.clone().unwrap_or(Array2::zeros((0, 0))));
            names.push(format!("{}.ffn.w1", prefix));
            weights.push(block.ffn.w2.clone().unwrap_or(Array2::zeros((0, 0))));
            names.push(format!("{}.ffn.w2", prefix));
            weights.push(block.ffn.w3.clone().unwrap_or(Array2::zeros((0, 0))));
            names.push(format!("{}.ffn.w3", prefix));
        }

        // Fuse consecutive FFN layers in each block: w2 (intermediate→hidden) with next block's wq (hidden→head)
        let num_fuse = self.blocks.len().saturating_sub(1);
        let mut fuse_pairs = Vec::with_capacity(num_fuse);
        // Indices into weights vec: ffn.w2 at offset 2 + i*7, attention.wq at offset 2 + (i+1)*7
        let per_block = 7;
        let base = 2; // token_embedding + lm_head
        for i in 0..num_fuse {
            let w2_idx = base + i * per_block + 6; // ffn.w2 is 7th matrix in block (0-indexed: wq=0, wk=1, wv=2, wo=3, w1=4, w2=5, w3=6)
            let wq_idx = base + (i + 1) * per_block + 0;
            if w2_idx < weights.len() && wq_idx < weights.len() {
                fuse_pairs.push((w2_idx, wq_idx));
            }
        }

        (weights, names, fuse_pairs)
    }

    /// Run SEDC compression on all model weights with default config.
    /// Returns compression report as JSON if GPU is available.
    pub fn compress_sedc_default(&self) -> TransformerResult<Option<serde_json::Value>> {
        #[cfg(feature = "gpu")]
        {
            let config = nexora_deeplearning::autograd::gpu_sedc::SedcConfig::default();
            self.compress_sedc_json(&config)
        }
        #[cfg(not(feature = "gpu"))]
        {
            info!("SEDC requires GPU feature — disabled");
            Ok(None)
        }
    }

    /// Run SEDC compression on all model weights with custom config.
    /// Returns compression report as JSON if GPU is available.
    #[cfg(feature = "gpu")]
    pub fn compress_sedc_json(
        &self,
        config: &nexora_deeplearning::autograd::gpu_sedc::SedcConfig,
    ) -> TransformerResult<Option<serde_json::Value>> {
        use nexora_deeplearning::autograd::gpu_sedc::SedcCompressor;

        let (weights, names, fuse_pairs) = self.collect_weights_for_sedc();

        if nexora_deeplearning::autograd::gpu::GpuContext::global().is_err() {
            info!("No GPU context available — SEDC compression skipped");
            return Ok(None);
        }

        let compressor = SedcCompressor::new(config.clone());
        let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let fused = compressor
            .compress_model(&weights, &name_refs, &fuse_pairs)
            .map_err(|e| TransformerError::Implementation(format!("SEDC compress: {}", e)))?;

        let report = fused.report;
        info!(
            "SEDC compression complete: {}→{} params ({:.2}% ratio), {:.4} error",
            report.total_original,
            report.total_compressed,
            report.total_ratio * 100.0,
            report.mean_relative_error,
        );

        let layers_json: Vec<serde_json::Value> = report
            .layers
            .iter()
            .map(|l| {
                serde_json::json!({
                    "layer": l.layer,
                    "name": l.name,
                    "original_params": l.original_params,
                    "compressed_params": l.compressed_params,
                    "compression_ratio": l.compression_ratio,
                    "rank": l.rank,
                    "spectral_entropy": l.spectral_entropy,
                    "relative_error": l.relative_error,
                    "sparsity": l.sparsity,
                })
            })
            .collect();

        Ok(Some(serde_json::json!({
            "total_original": report.total_original,
            "total_compressed": report.total_compressed,
            "total_ratio": report.total_ratio,
            "mean_relative_error": report.mean_relative_error,
            "mean_sparsity": report.mean_sparsity,
            "layers": layers_json,
        })))
    }

}
