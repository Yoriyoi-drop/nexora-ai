use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::backbone::{LinearLayer, MLPExpert, OracleBackbone, OracleBackboneConfig};

/// Skema kuantisasi
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationScheme {
    F32,
    Q8,
}

impl QuantizationScheme {
    pub fn bits_per_weight(&self) -> u8 {
        match self {
            QuantizationScheme::F32 => 32,
            QuantizationScheme::Q8 => 8,
        }
    }

    pub fn memory_ratio(&self) -> f32 {
        match self {
            QuantizationScheme::F32 => 1.0,
            QuantizationScheme::Q8 => 0.25,
        }
    }
}

/// Bobot terkuantisasi Q8 (8-bit symmetric)
#[derive(Debug, Clone)]
pub struct Q8Weight {
    /// Scale factor per row
    pub scales: Vec<f32>,
    /// Quantized values i8
    pub data: Vec<i8>,
    pub shape: (usize, usize),
}

impl Q8Weight {
    pub fn from_f32(weight: &Array2<f32>) -> Self {
        let (rows, cols) = weight.dim();
        let mut scales = Vec::with_capacity(rows);
        let mut data = Vec::with_capacity(rows * cols);

        for row in weight.rows() {
            let abs_max = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if abs_max > 0.0 { 127.0 / abs_max } else { 1.0 };
            scales.push(1.0 / scale);
            for v in row.iter() {
                data.push((v * scale).round().clamp(-128.0, 127.0) as i8);
            }
        }

        Self {
            scales,
            data,
            shape: (rows, cols),
        }
    }

    pub fn to_f32(&self) -> Array2<f32> {
        let (rows, cols) = self.shape;
        if self.scales.is_empty() || self.data.is_empty() {
            return Array2::zeros((rows, cols));
        }
        let mut result = Array2::zeros((rows, cols));
        for i in 0..rows {
            let scale = self.scales[i];
            for j in 0..cols {
                let idx = i * cols + j;
                result[[i, j]] = self.data[idx] as f32 * scale;
            }
        }
        result
    }

    pub fn memory_bytes(&self) -> usize {
        let (rows, cols) = self.shape;
        rows * cols + rows * 4
    }

    pub fn compression_ratio(&self) -> f32 {
        let f32_size = self.shape.0 * self.shape.1 * 4;
        f32_size as f32 / self.memory_bytes() as f32
    }
}

/// LinearLayer terkuantisasi
#[derive(Debug, Clone)]
pub struct QuantizedLinearLayer {
    pub weight: Q8Weight,
    pub bias: Option<Vec<f32>>,
}

impl QuantizedLinearLayer {
    pub fn from_linear(layer: &LinearLayer) -> Self {
        Self {
            weight: Q8Weight::from_f32(&layer.weight),
            bias: Some(layer.bias.clone().into_raw_vec()),
        }
    }

    pub fn dequantize(&self) -> LinearLayer {
        LinearLayer::from_arrays(
            self.weight.to_f32(),
            self.bias
                .as_ref()
                .map(|b| ndarray::Array1::from_vec(b.clone()))
                .unwrap_or_else(|| ndarray::Array1::zeros(self.weight.shape.1)),
        )
    }
}

/// Expert MLP terkuantisasi
#[derive(Debug, Clone)]
pub struct QuantizedMLPExpert {
    pub w1: Q8Weight,
    pub w2: Q8Weight,
    pub b1: Vec<f32>,
    pub b2: Vec<f32>,
}

impl QuantizedMLPExpert {
    pub fn from_expert(expert: &MLPExpert) -> Self {
        Self {
            w1: Q8Weight::from_f32(&expert.w1),
            w2: Q8Weight::from_f32(&expert.w2),
            b1: expert.b1.clone().into_raw_vec(),
            b2: expert.b2.clone().into_raw_vec(),
        }
    }

    pub fn memory_bytes(&self) -> usize {
        self.w1.memory_bytes() + self.w2.memory_bytes()
            + self.b1.len() * 4 + self.b2.len() * 4
    }
}

/// OracleBackbone dengan bobot terkuantisasi Q8
pub struct QuantizedOracleBackbone {
    pub config: OracleBackboneConfig,
    pub vocab_size: usize,
    pub quantized_embedding: Q8Weight,
    pub quantized_moe_layers: Vec<Vec<QuantizedMLPExpert>>,
    pub quantized_gate_layers: Vec<QuantizedLinearLayer>,
    pub quantized_output: QuantizedLinearLayer,
    quantization: QuantizationScheme,
}

impl QuantizedOracleBackbone {
    /// Create quantized backbone langsung dari config — zero weight init.
    /// Cuma nyimpen shape, cocok untuk memory estimation cepat.
    pub fn from_config(config: OracleBackboneConfig, vocab_size: usize, scheme: QuantizationScheme) -> Self {
        let n_layers = 12;
        let d_model = config.d_model;
        let mlp_hidden = config.mlp_hidden;
        let n_experts = config.n_experts;
        Self {
            config,
            vocab_size,
            quantized_embedding: Q8Weight {
                scales: vec![],
                data: vec![],
                shape: (vocab_size, d_model),
            },
            quantized_moe_layers: (0..n_layers)
                .map(|_| {
                    (0..n_experts)
                        .map(|_| QuantizedMLPExpert {
                            w1: Q8Weight { scales: vec![], data: vec![], shape: (d_model, mlp_hidden) },
                            w2: Q8Weight { scales: vec![], data: vec![], shape: (mlp_hidden, d_model) },
                            b1: vec![0.0; mlp_hidden],
                            b2: vec![0.0; d_model],
                        })
                        .collect()
                })
                .collect(),
            quantized_gate_layers: (0..n_layers)
                .map(|_| QuantizedLinearLayer {
                    weight: Q8Weight { scales: vec![], data: vec![], shape: (d_model, n_experts) },
                    bias: Some(vec![0.0; n_experts]),
                })
                .collect(),
            quantized_output: QuantizedLinearLayer {
                weight: Q8Weight { scales: vec![], data: vec![], shape: (d_model, vocab_size) },
                bias: Some(vec![0.0; vocab_size]),
            },
            quantization: scheme,
        }
    }

    pub fn from_backbone(backbone: &OracleBackbone, scheme: QuantizationScheme) -> Self {
        let vocab_size = backbone.embedding.embeddings.dim().0;
        let quantized_embedding = Q8Weight::from_f32(&backbone.embedding.embeddings);

        let quantized_moe_layers: Vec<Vec<QuantizedMLPExpert>> = backbone
            .moe_layers
            .iter()
            .map(|layer| {
                layer
                    .experts
                    .iter()
                    .map(QuantizedMLPExpert::from_expert)
                    .collect()
            })
            .collect();

        let quantized_gate_layers: Vec<QuantizedLinearLayer> = backbone
            .moe_layers
            .iter()
            .map(|layer| QuantizedLinearLayer::from_linear(&layer.gate))
            .collect();

        let quantized_output = QuantizedLinearLayer::from_linear(&backbone.output_projection);

        Self {
            config: backbone.config.clone(),
            vocab_size,
            quantized_embedding,
            quantized_moe_layers,
            quantized_gate_layers,
            quantized_output,
            quantization: scheme,
        }
    }

    pub fn memory_savings_report(&self) -> QuantizationReport {
        let f32_total = self.estimate_f32_size();
        let q8_total = self.estimate_q8_size();
        QuantizationReport {
            f32_bytes: f32_total,
            q8_bytes: q8_total,
            savings_bytes: f32_total - q8_total,
            savings_percent: if f32_total > 0 {
                ((f32_total - q8_total) as f64 / f32_total as f64 * 100.0) as f32
            } else {
                0.0
            },
            scheme: self.quantization,
        }
    }

    fn estimate_f32_size(&self) -> usize {
        let mut total = self.vocab_size * self.config.d_model * 4;
        for layer_experts in &self.quantized_moe_layers {
            for expert in layer_experts {
                total += expert.w1.shape.0 * expert.w1.shape.1 * 4;
                total += expert.w2.shape.0 * expert.w2.shape.1 * 4;
                total += expert.b1.len() * 4;
                total += expert.b2.len() * 4;
            }
        }
        total
    }

    fn estimate_q8_size(&self) -> usize {
        let mut total = self.quantized_embedding.memory_bytes();
        for layer_experts in &self.quantized_moe_layers {
            for expert in layer_experts {
                total += expert.memory_bytes();
            }
        }
        total
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationReport {
    pub f32_bytes: usize,
    pub q8_bytes: usize,
    pub savings_bytes: usize,
    pub savings_percent: f32,
    pub scheme: QuantizationScheme,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q8_roundtrip() {
        let weight = Array2::from_shape_fn((4, 8), |(i, j)| (i * 8 + j) as f32);
        let q8 = Q8Weight::from_f32(&weight);
        let recovered = q8.to_f32();
        for i in 0..4 {
            for j in 0..8 {
                let diff = (weight[[i, j]] - recovered[[i, j]]).abs();
                assert!(diff < 1.0, "diff too large at ({},{}): {}", i, j, diff);
            }
        }
    }

    #[test]
    fn test_q8_compression_ratio() {
        let weight = Array2::from_shape_fn((64, 128), |(i, j)| (i * 128 + j) as f32);
        let q8 = Q8Weight::from_f32(&weight);
        let ratio = q8.compression_ratio();
        assert!(ratio > 3.0, "compression ratio too low: {}", ratio);
        assert!(ratio < 5.0, "compression ratio too high: {}", ratio);
    }

    #[test]
    fn test_quantized_linear_conversion() {
        let linear = LinearLayer::new(64, 128);
        let qlinear = QuantizedLinearLayer::from_linear(&linear);
        let deq = qlinear.dequantize();
        assert_eq!(deq.weight.dim(), linear.weight.dim());
    }

    #[test]
    fn test_memory_report() {
        let config = OracleBackboneConfig::default();
        let qbb = QuantizedOracleBackbone::from_config(config, 50000, QuantizationScheme::Q8);
        let report = qbb.memory_savings_report();
        assert!(report.savings_percent > 50.0, "savings too low: {}%", report.savings_percent);
    }
}
