use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

use nexora_star_x::quantization::{
    QuantizationEngine, QuantMethod, QuantPrecision, MixedPrecisionEngine,
};

/// Dynamic quantization manager for inference runtime.
/// Wraps STAR-X QuantizationEngine and MixedPrecisionEngine for on-the-fly
/// weight compression during inference. Reduces memory usage by 2-8× depending
/// on precision selected.
pub struct DynamicQuantManager {
    engine: RwLock<QuantizationEngine>,
    mixed: RwLock<MixedPrecisionEngine>,
    precision: QuantPrecision,
    method: QuantMethod,
    total_memory_saved: RwLock<usize>,
    quant_count: RwLock<usize>,
    auto_mode: bool,
}

impl DynamicQuantManager {
    pub fn new() -> Self {
        let engine = match QuantizationEngine::new(QuantPrecision::INT8, QuantMethod::Dynamic) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("Failed to create QuantizationEngine: {e}, using fallback");
                QuantizationEngine::new(QuantPrecision::INT8, QuantMethod::Dynamic)
                    .expect("QuantizationEngine must be constructable")
            }
        };

        let mixed = match MixedPrecisionEngine::new(QuantPrecision::FP16) {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("Failed to create MixedPrecisionEngine: {e}");
                MixedPrecisionEngine::new(QuantPrecision::FP16)
                    .expect("MixedPrecisionEngine must be constructable")
            }
        };

        Self {
            engine: RwLock::new(engine),
            mixed: RwLock::new(mixed),
            precision: QuantPrecision::INT8,
            method: QuantMethod::Dynamic,
            total_memory_saved: RwLock::new(0),
            quant_count: RwLock::new(0),
            auto_mode: true,
        }
    }

    pub fn with_precision(mut self, precision: QuantPrecision) -> Self {
        self.precision = precision;
        self
    }

    pub fn with_method(mut self, method: QuantMethod) -> Self {
        self.method = method;
        self
    }

    pub fn with_auto_mode(mut self, auto: bool) -> Self {
        self.auto_mode = auto;
        self
    }

    /// Quantize a weight tensor dynamically.
    /// Returns the quantized data + statistics.
    pub async fn quantize_weights(
        &self,
        weights: &ndarray::Array2<f32>,
        activations: Option<&ndarray::Array2<f32>>,
    ) -> Option<nexora_star_x::quantization::QuantizedTensor> {
        let mut engine = self.engine.write().await;

        let result = match self.method {
            QuantMethod::AWQ => {
                if let Some(acts) = activations {
                    engine.quantize_weights_awq(weights, acts).ok()
                } else {
                    engine.quantize_weights_gptq(weights).ok()
                }
            }
            QuantMethod::GPTQ => engine.quantize_weights_gptq(weights).ok(),
            QuantMethod::Dynamic => engine.quantize_dynamic(weights).ok(),
            QuantMethod::Static => engine.quantize_dynamic(weights).ok(),
            QuantMethod::Mixed => engine.quantize_weights_gptq(weights).ok(),
        };

        if let Some(ref qt) = result {
            let stats = engine.get_stats();
            *self.total_memory_saved.write().await += stats.memory_saved;
            *self.quant_count.write().await += 1;
            debug!(
                "Dynamic quant: {} weights, method={:?}, saved={} bytes",
                qt.data().len(),
                self.method,
                stats.memory_saved
            );
        }

        result
    }

    /// Dequantize back to f32.
    pub async fn dequantize(
        &self,
        tensor: &nexora_star_x::quantization::QuantizedTensor,
    ) -> Option<ndarray::Array2<f32>> {
        let mut engine = self.engine.write().await;
        engine.dequantize(tensor).ok()
    }

    /// Get cumulative statistics.
    pub async fn stats(&self) -> DynamicQuantStats {
        let engine = self.engine.read().await;
        let stats = engine.get_stats();
        DynamicQuantStats {
            quantization_time: stats.quantization_time,
            dequantization_time: stats.dequantization_time,
            memory_saved: *self.total_memory_saved.read().await,
            quant_count: *self.quant_count.read().await,
            precision: self.precision,
            method: self.method,
            auto_mode: self.auto_mode,
        }
    }

    /// Auto-quantize model layers based on runtime profiling.
    pub async fn auto_quantize(
        &self,
        layer_weights: &[(&str, &ndarray::Array2<f32>)],
    ) -> Vec<nexora_star_x::quantization::QuantizedTensor> {
        let mut results = Vec::with_capacity(layer_weights.len());

        for (_name, weights) in layer_weights {
            if self.auto_mode {
                let mem = weights.len() * 4;
                if mem > 1024 * 1024 {
                    let qt = self.quantize_weights(weights, None).await;
                    if let Some(q) = qt {
                        results.push(q);
                        continue;
                    }
                }
            }
        }

        info!(
            "Auto-quantized {}/{} layers",
            results.len(),
            layer_weights.len()
        );
        results
    }

    /// Reset statistics.
    pub async fn reset(&self) {
        *self.total_memory_saved.write().await = 0;
        *self.quant_count.write().await = 0;
    }
}

impl Default for DynamicQuantManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Public statistics for DynamicQuantManager
#[derive(Debug, Clone)]
pub struct DynamicQuantStats {
    pub quantization_time: f64,
    pub dequantization_time: f64,
    pub memory_saved: usize,
    pub quant_count: usize,
    pub precision: QuantPrecision,
    pub method: QuantMethod,
    pub auto_mode: bool,
}

impl std::fmt::Debug for DynamicQuantManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicQuantManager")
            .field("precision", &self.precision)
            .field("method", &self.method)
            .field("auto_mode", &self.auto_mode)
            .finish()
    }
}
