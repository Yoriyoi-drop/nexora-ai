use ndarray::{Array1, Array2, ArrayD};

#[derive(Debug, Clone)]
pub struct AWQConfig {
    pub group_size: usize,
    pub bits: u8,
    pub calibration_samples: usize,
    pub sym: bool,
    pub clip_alpha: f32,
}

impl Default for AWQConfig {
    fn default() -> Self {
        Self {
            group_size: 128,
            bits: 4,
            calibration_samples: 128,
            sym: false,
            clip_alpha: 1.0,
        }
    }
}

impl AWQConfig {
    pub fn w4a16() -> Self {
        Self { bits: 4, group_size: 128, ..Default::default() }
    }

    pub fn w3a16() -> Self {
        Self { bits: 3, group_size: 128, ..Default::default() }
    }
}

#[derive(Debug, Clone)]
pub struct AWQScale {
    pub scale: f32,
    pub zero_point: i32,
    pub group_idx: usize,
}

#[derive(Debug, Clone)]
pub struct AWQGroup {
    pub scales: Vec<AWQScale>,
    pub max_scale: f32,
    pub is_salient: bool,
    pub saliency_score: f32,
}

#[derive(Debug, Clone)]
pub struct AWQQuantizedTensor {
    pub qdata: Vec<u8>,
    pub scales: Vec<f32>,
    pub zero_points: Vec<i32>,
    pub group_size: usize,
    pub bits: u8,
    pub original_shape: Vec<usize>,
}

impl AWQQuantizedTensor {
    pub fn compress_ratio(&self) -> f32 {
        let original_bits = self.original_shape.iter().product::<usize>() as f32 * 32.0;
        let compressed_bits = self.qdata.len() as f32 * 8.0 +
            self.scales.len() as f32 * 32.0 +
            self.zero_points.len() as f32 * 32.0;
        original_bits / compressed_bits
    }
}

pub struct AWQEngine {
    pub config: AWQConfig,
}

impl AWQEngine {
    pub fn new(config: AWQConfig) -> Self {
        Self { config }
    }

    pub fn compute_activation_scale(&self, activations: &[Array1<f32>]) -> Array1<f32> {
        let num_cols = activations[0].len();
        let mut max_vals = Array1::zeros(num_cols);
        for col in 0..num_cols {
            let abs_max = activations.iter()
                .map(|a| a[col].abs())
                .fold(0.0f32, f32::max);
            max_vals[col] = if abs_max > 0.0 { abs_max } else { 1.0 };
        }
        max_vals
    }

    pub fn compute_saliency(&self, weight: &Array2<f32>, activation_scale: &Array1<f32>) -> Array1<f32> {
        let in_features = weight.shape()[1];
        let mut saliency = Array1::zeros(in_features);
        for j in 0..in_features {
            let col_norm: f32 = weight.column(j).iter().map(|v| v.abs()).sum();
            saliency[j] = col_norm * activation_scale[j];
        }
        saliency
    }

    pub fn find_optimal_scales(
        &self,
        weight: &Array2<f32>,
        saliency: &Array1<f32>,
    ) -> (Vec<f32>, Vec<i32>) {
        let num_groups = (weight.shape()[1] + self.config.group_size - 1) / self.config.group_size;
        let max_quant = (1 << self.config.bits) as f32 - 1.0;

        let mut scales = Vec::with_capacity(num_groups);
        let mut zero_points = Vec::with_capacity(num_groups);

        for g in 0..num_groups {
            let start = g * self.config.group_size;
            let end = (start + self.config.group_size).min(weight.shape()[1]);

            let mut group_vals: Vec<f32> = Vec::with_capacity((end - start) * weight.shape()[0]);
            for i in 0..weight.shape()[0] {
                for j in start..end {
                    group_vals.push(weight[[i, j]]);
                }
            }

            let min_val = group_vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = group_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            if (max_val - min_val).abs() < 1e-10 {
                scales.push(1.0f32);
                zero_points.push(0);
                continue;
            }

            let group_saliency: f32 = (start..end).map(|j| saliency[j]).sum();
            let is_salient = group_saliency > saliency.iter().sum::<f32>() / num_groups as f32;

            let s = if is_salient {
                (max_val - min_val) / max_quant * self.config.clip_alpha
            } else {
                (max_val - min_val) / max_quant
            };
            let z = if self.config.sym {
                0
            } else {
                (-min_val / s).round() as i32
            };

            scales.push(s.max(1e-10));
            zero_points.push(z);
        }

        (scales, zero_points)
    }

    pub fn quantize(&self, weight: &Array2<f32>, scales: &[f32], zero_points: &[i32]) -> AWQQuantizedTensor {
        let num_groups = scales.len();
        let max_quant = (1 << self.config.bits) as f32 - 1.0;
        let entries_per_byte = 8 / self.config.bits as usize;
        let total_elements = weight.shape()[0] * weight.shape()[1];
        let qdata_len = (total_elements + entries_per_byte - 1) / entries_per_byte;

        let mut qdata = vec![0u8; qdata_len];
        let mut idx = 0;

        for g in 0..num_groups {
            let start = g * self.config.group_size;
            let end = (start + self.config.group_size).min(weight.shape()[1]);
            let s = scales[g];
            let zp = zero_points[g] as f32;

            for i in 0..weight.shape()[0] {
                for j in start..end {
                    let q = ((weight[[i, j]] / s) + zp).round().clamp(0.0, max_quant) as u8;
                    let bit_offset = (idx % entries_per_byte) * self.config.bits as usize;
                    qdata[idx / entries_per_byte] |= q << bit_offset;
                    idx += 1;
                }
            }
        }

        AWQQuantizedTensor {
            qdata,
            scales: scales.to_vec(),
            zero_points: zero_points.to_vec(),
            group_size: self.config.group_size,
            bits: self.config.bits,
            original_shape: vec![weight.shape()[0], weight.shape()[1]],
        }
    }

    pub fn dequantize(&self, q: &AWQQuantizedTensor) -> Array2<f32> {
        let rows = q.original_shape[0];
        let cols = q.original_shape[1];
        let max_quant = (1 << q.bits) as f32;
        let mask = (1 << q.bits) - 1;
        let entries_per_byte = 8 / q.bits as usize;

        let mut result = Array2::zeros((rows, cols));
        let mut idx = 0;

        let num_groups = q.scales.len();

        for g in 0..num_groups {
            let start = g * q.group_size;
            let end = (start + q.group_size).min(cols);
            let s = q.scales[g];
            let zp = q.zero_points[g] as f32;

            for i in 0..rows {
                for j in start..end {
                    let byte = q.qdata[idx / entries_per_byte];
                    let bit_offset = (idx % entries_per_byte) * q.bits as usize;
                    let q_val = ((byte >> bit_offset) & mask) as f32;
                    result[[i, j]] = (q_val - zp) * s;
                    idx += 1;
                }
            }
        }

        result
    }

    pub fn pseudo_quantize(&self, weight: &Array2<f32>, saliency: &Array1<f32>) -> Array2<f32> {
        let (scales, zero_points) = self.find_optimal_scales(weight, saliency);
        let quantized = self.quantize(weight, &scales, &zero_points);
        self.dequantize(&quantized)
    }

    pub fn quantize_model<'a>(
        &self,
        weights: Vec<(&'a str, ArrayD<f32>)>,
    ) -> Vec<(&'a str, AWQQuantizedTensor)> {
        weights.into_iter()
            .filter_map(|(name, w)| {
                let w2 = w.into_dimensionality::<ndarray::Ix2>().ok()?;
                let dummy_act = Array1::from_elem(w2.shape()[1], 1.0);
                let (scales, zps) = self.find_optimal_scales(&w2, &dummy_act);
                let q = self.quantize(&w2, &scales, &zps);
                Some((name, q))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let engine = AWQEngine::new(AWQConfig::w4a16());
        let weight = Array2::from_shape_fn((16, 32), |(i, j)| {
            (i * 32 + j) as f32 / 128.0 - 1.0
        });
        let act_scale = engine.compute_activation_scale(&[Array1::from_elem(32, 0.5)]);
        let saliency = engine.compute_saliency(&weight, &act_scale);
        let (scales, zps) = engine.find_optimal_scales(&weight, &saliency);
        let q = engine.quantize(&weight, &scales, &zps);
        let deq = engine.dequantize(&q);
        let mse: f32 = weight.iter()
            .zip(deq.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / (weight.len() as f32);
        assert!(mse < 1.0, "mse = {mse} too high");
    }

    #[test]
    fn test_compress_ratio() {
        let engine = AWQEngine::new(AWQConfig::w4a16());
        let weight = Array2::zeros((64, 128));
        let (scales, zps) = engine.find_optimal_scales(&weight, &Array1::from_elem(128, 1.0));
        let q = engine.quantize(&weight, &scales, &zps);
        assert!(q.compress_ratio() > 1.0);
    }

    #[test]
    fn test_saliency_high_for_large_activations() {
        let engine = AWQEngine::new(AWQConfig::default());
        let weight = Array2::ones((16, 32));
        let activations = vec![
            Array1::from_shape_fn(32, |j| if j < 4 { 10.0 } else { 0.1 }),
        ];
        let act_scale = engine.compute_activation_scale(&activations);
        let saliency = engine.compute_saliency(&weight, &act_scale);
        for j in 0..4 {
            assert!(saliency[j] > saliency[10], "channel {j} should have high saliency");
        }
    }
}
