use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::backbone::{LinearLayer, OracleBackboneConfig};

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


}
