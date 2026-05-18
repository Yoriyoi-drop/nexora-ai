//! Weight quantization helpers — INT8 and INT4 (packed/groupwise).
//!
//! Provides:
//! - `QuantizedDtype` — enum for supported quantized data types
//! - `QuantizedTensor` — packed weight storage with scales
//! - Quantize/dequantize functions for INT8 (per-tensor) and INT4 (packed, groupwise)
//! - `quantize_linear` / `dequantize_linear` that wraps the full pipeline

use ndarray::Array2;

/// Supported quantized data types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizedDtype {
    /// 8-bit integer: 1 scale per tensor
    Int8,
    /// 4-bit packed (2 values per byte): 1 scale per tensor
    Int4Packed,
    /// 4-bit groupwise: configurable group_size, scale per group
    Int4Groupwise { group_size: usize },
}

impl QuantizedDtype {
    /// Bits per weight element.
    pub fn bits_per_element(&self) -> usize {
        match self {
            QuantizedDtype::Int8 => 8,
            QuantizedDtype::Int4Packed | QuantizedDtype::Int4Groupwise { .. } => 4,
        }
    }

    /// Memory savings ratio vs f32 (32 bits per element).
    pub fn compression_ratio(&self) -> f64 {
        32.0 / self.bits_per_element() as f64
    }
}

/// A quantized weight tensor with metadata for dequantization.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Which quantization scheme was used
    pub dtype: QuantizedDtype,
    /// Packed quantized data (1 or 2 values per byte)
    pub data: Vec<u8>,
    /// Original shape before quantization
    pub shape: (usize, usize),
    /// Scale factor(s): 1 for per-tensor, N for per-group
    pub scales: Vec<f32>,
    /// Zero point (only for INT8 symmetric)
    pub zero_point: i16,
}

impl QuantizedTensor {
    /// Number of elements in the original tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.0 * self.shape.1
    }

    /// Memory used by quantized data in bytes (data + scales).
    pub fn memory_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4
    }

    /// Memory used by the original f32 tensor in bytes.
    pub fn original_memory_bytes(&self) -> usize {
        self.num_elements() * 4
    }

    /// Compression ratio (higher = better).
    pub fn compression_ratio(&self) -> f64 {
        self.original_memory_bytes() as f64 / self.memory_bytes() as f64
    }
}

// ─── INT8 per-tensor quantization ──────────────────────────────────────────

/// Quantize f32 weights to INT8 with per-tensor scale.
/// Returns (quantized_data, scale, zero_point).
pub fn quantize_f32_to_int8(weights: &Array2<f32>) -> (Vec<u8>, f32, i16) {
    let elements: Vec<f32> = weights.iter().copied().collect();
    if elements.is_empty() {
        return (Vec::new(), 1.0, 0);
    }

    let max_val = elements
        .iter()
        .copied()
        .fold(0.0f32, |a, b| a.max(b.abs()))
        .max(1e-10);

    let scale = max_val / 127.0;
    let mut quantized = Vec::with_capacity(elements.len());
    for &v in &elements {
        let q = (v / scale).round().clamp(-128.0, 127.0) as i8;
        quantized.push(q as u8);
    }
    (quantized, scale, 0)
}

/// Dequantize INT8 data back to f32.
pub fn dequantize_int8_to_f32(
    data: &[u8],
    scale: f32,
    zero_point: i16,
    rows: usize,
    cols: usize,
) -> Array2<f32> {
    let expected = rows * cols;
    let mut out = Array2::zeros((rows, cols));
    for (i, &byte) in data.iter().enumerate().take(expected.min(data.len())) {
        let q = byte as i8;
        let row = i / cols;
        let col = i % cols;
        out[[row, col]] = (q as i16 - zero_point) as f32 * scale;
    }
    out
}

/// Dequantize INT8 and multiply: output = input @ W^T (where W is quantized).
/// W is stored in row-major order: shape (output_dim, input_dim).
pub fn matmul_int8(
    input: &Array2<f32>,
    w_data: &[u8],
    w_scale: f32,
    w_zero: i16,
    w_rows: usize,
    w_cols: usize,
) -> Array2<f32> {
    let batch = input.shape()[0];
    let mut output = Array2::zeros((batch, w_rows));
    for b in 0..batch {
        for r in 0..w_rows {
            let mut dot = 0.0;
            for c in 0..w_cols {
                let idx = r * w_cols + c;
                let w_val = if idx < w_data.len() {
                    (w_data[idx] as i8 as i16 - w_zero) as f32 * w_scale
                } else {
                    0.0
                };
                dot += input[[b, c]] * w_val;
            }
            output[[b, r]] = dot;
        }
    }
    output
}

// ─── INT4 packed per-tensor quantization ───────────────────────────────────

/// Quantize f32 to INT4 packed (2 values per byte).
/// Upper nibble = first value, lower nibble = second value.
/// Nibble stores two's complement signed 4-bit: [0..7] = [0..7], [8..15] = [-8..-1].
pub fn quantize_f32_to_int4_packed(weights: &Array2<f32>) -> (Vec<u8>, f32) {
    let elements: Vec<f32> = weights.iter().copied().collect();
    if elements.is_empty() {
        return (Vec::new(), 1.0);
    }

    let max_val = elements
        .iter()
        .copied()
        .fold(0.0f32, |a, b| a.max(b.abs()))
        .max(1e-10);

    let scale = max_val / 7.0; // INT4 range: [-8, 7]
    let n = elements.len();
    let packed_len = (n + 1) / 2;
    let mut packed = vec![0u8; packed_len];

    for i in 0..n {
        let q = (elements[i] / scale).round().clamp(-8.0, 7.0) as i8;
        let nibble = (q as u8) & 0x0F;
        if i % 2 == 0 {
            packed[i / 2] = (nibble << 4) | (packed[i / 2] & 0x0F);
        } else {
            packed[i / 2] = (packed[i / 2] & 0xF0) | nibble;
        }
    }
    (packed, scale)
}

/// Sign-extend a 4-bit value to i8.
#[inline]
fn sign_extend_4bit(nibble: u8) -> i8 {
    ((nibble as i8) << 4) >> 4
}

/// Dequantize INT4 packed data back to f32.
/// Nibble is stored in two's complement, sign-extended to i8.
pub fn dequantize_int4_packed_to_f32(
    data: &[u8],
    scale: f32,
    rows: usize,
    cols: usize,
) -> Array2<f32> {
    let expected = rows * cols;
    let mut out = Array2::zeros((rows, cols));
    for i in 0..expected.min(data.len() * 2) {
        let byte = data[i / 2];
        let nibble = if i % 2 == 0 {
            (byte >> 4) & 0x0F
        } else {
            byte & 0x0F
        };
        let val = sign_extend_4bit(nibble) as f32 * scale;
        let row = i / cols;
        let col = i % cols;
        out[[row, col]] = val;
    }
    out
}

// ─── INT4 groupwise quantization ───────────────────────────────────────────

/// Quantize f32 weights to INT4 with per-group scale.
/// Each group of `group_size` elements shares a scale factor.
pub fn quantize_f32_to_int4_groupwise(
    weights: &Array2<f32>,
    group_size: usize,
) -> (Vec<u8>, Vec<f32>) {
    let (rows, cols) = weights.dim();
    let n = rows * cols;
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let gs = group_size.max(1);
    let num_groups = (n + gs - 1) / gs;
    let mut packed = vec![0u8; (n + 1) / 2];
    let mut scales = Vec::with_capacity(num_groups);

    let flat: Vec<f32> = weights.iter().copied().collect();

    for g in 0..num_groups {
        let start = g * gs;
        let end = (start + gs).min(n);

        let mut max_abs = 0.0f32;
        for &v in flat[start..end].iter() {
            max_abs = max_abs.max(v.abs());
        }
        max_abs = max_abs.max(1e-10);

        let scale = max_abs / 7.0;

        for i in start..end {
            let q = (flat[i] / scale).round().clamp(-8.0, 7.0) as i8;
            let nibble = (q as u8) & 0x0F;
            let offset = i - start;
            let packed_idx = (start + offset) / 2;
            if offset % 2 == 0 {
                packed[packed_idx] = (nibble << 4) | (packed[packed_idx] & 0x0F);
            } else {
                packed[packed_idx] = (packed[packed_idx] & 0xF0) | nibble;
            }
        }
        scales.push(scale);
    }

    (packed, scales)
}

/// Dequantize INT4 groupwise data back to f32.
pub fn dequantize_int4_groupwise_to_f32(
    data: &[u8],
    scales: &[f32],
    group_size: usize,
    rows: usize,
    cols: usize,
) -> Array2<f32> {
    let n = rows * cols;
    let gs = group_size.max(1);
    let mut out = Array2::zeros((rows, cols));

    for i in 0..n.min(data.len() * 2) {
        let g = i / gs;
        let byte = data[i / 2];
        let nibble = if i % 2 == 0 {
            (byte >> 4) & 0x0F
        } else {
            byte & 0x0F
        };
        let scale = scales.get(g).copied().unwrap_or(1.0);
        let val = sign_extend_4bit(nibble) as f32 * scale;
        let row = i / cols;
        let col = i % cols;
        out[[row, col]] = val;
    }
    out
}

// ─── High-level helpers ────────────────────────────────────────────────────

/// Quantize an f32 weight matrix into a `QuantizedTensor`.
pub fn quantize_linear(weights: &Array2<f32>, dtype: QuantizedDtype) -> QuantizedTensor {
    let shape = weights.dim();
    match dtype {
        QuantizedDtype::Int8 => {
            let (data, scale, zp) = quantize_f32_to_int8(weights);
            QuantizedTensor {
                dtype,
                data,
                shape,
                scales: vec![scale],
                zero_point: zp,
            }
        }
        QuantizedDtype::Int4Packed => {
            let (data, scale) = quantize_f32_to_int4_packed(weights);
            QuantizedTensor {
                dtype,
                data,
                shape,
                scales: vec![scale],
                zero_point: 0,
            }
        }
        QuantizedDtype::Int4Groupwise { group_size } => {
            let (data, scales) = quantize_f32_to_int4_groupwise(weights, group_size);
            QuantizedTensor {
                dtype,
                data,
                shape,
                scales,
                zero_point: 0,
            }
        }
    }
}

/// Dequantize a `QuantizedTensor` back to f32.
pub fn dequantize_linear(tensor: &QuantizedTensor) -> Array2<f32> {
    let (rows, cols) = tensor.shape;
    match tensor.dtype {
        QuantizedDtype::Int8 => {
            let scale = tensor.scales.first().copied().unwrap_or(1.0);
            dequantize_int8_to_f32(&tensor.data, scale, tensor.zero_point, rows, cols)
        }
        QuantizedDtype::Int4Packed => {
            let scale = tensor.scales.first().copied().unwrap_or(1.0);
            dequantize_int4_packed_to_f32(&tensor.data, scale, rows, cols)
        }
        QuantizedDtype::Int4Groupwise { group_size } => {
            dequantize_int4_groupwise_to_f32(
                &tensor.data,
                &tensor.scales,
                group_size,
                rows,
                cols,
            )
        }
    }
}

/// Compute the RMSE between original and quantized-dequantized weights.
pub fn quantization_error(original: &Array2<f32>, reconstructed: &Array2<f32>) -> f64 {
    let n = original.len().max(1);
    let sum_sq: f64 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (*a - *b) as f64)
        .map(|d| d * d)
        .sum();
    (sum_sq / n as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_weights() -> Array2<f32> {
        Array2::from_shape_vec((4, 3), vec![
            0.5, -1.2, 2.3,
            -3.4, 4.5, -5.6,
            6.7, -7.8, 8.9,
            -9.0, 0.1, -0.2,
        ]).unwrap()
    }

    #[test]
    fn test_int8_roundtrip() {
        let w = test_weights();
        let qt = quantize_linear(&w, QuantizedDtype::Int8);
        let w2 = dequantize_linear(&qt);
        let err = quantization_error(&w, &w2);
        assert!(
            err < 0.06,
            "INT8 roundtrip error too high: {err}"
        );
        assert_eq!(w2.dim(), (4, 3));
        assert!(qt.compression_ratio() >= 3.0, "ratio={}", qt.compression_ratio());
    }

    #[test]
    fn test_int4_packed_roundtrip() {
        let w = test_weights();
        let qt = quantize_linear(&w, QuantizedDtype::Int4Packed);
        let w2 = dequantize_linear(&qt);
        let err = quantization_error(&w, &w2);
        assert!(
            err < 1.0,
            "INT4 packed roundtrip error too high: {err}"
        );
        assert_eq!(qt.data.len(), (w.len() + 1) / 2);
        assert!(qt.compression_ratio() >= 4.0, "ratio={}", qt.compression_ratio());
    }

    #[test]
    fn test_int4_groupwise_roundtrip() {
        let w = test_weights();
        let qt = quantize_linear(&w, QuantizedDtype::Int4Groupwise { group_size: 4 });
        let w2 = dequantize_linear(&qt);
        let err = quantization_error(&w, &w2);
        assert!(
            err < 1.0,
            "INT4 groupwise roundtrip error too high: {err}"
        );
        assert!(qt.compression_ratio() >= 2.0, "ratio={}", qt.compression_ratio());
        assert_eq!(qt.scales.len(), (w.len() + 3) / 4);
    }

    #[test]
    fn test_int8_matmul() {
        // input: (2, 3), weights: (4, 3) → output: (2, 4)
        let input = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, -1.0, 0.5, -0.5, 0.0]).unwrap();
        let w_f32 = test_weights();
        let (w_data, scale, zp) = quantize_f32_to_int8(&w_f32);

        // Reference: f32 matmul
        let expected = input.dot(&w_f32.t());

        // INT8 matmul
        let (rows, cols) = w_f32.dim();
        let result = matmul_int8(&input, &w_data, scale, zp, rows, cols);

        for i in 0..2 {
            for j in 0..4 {
                let diff = (result[[i, j]] - expected[[i, j]]).abs();
                assert!(diff < 0.1, "matmul_int8 mismatch at ({i},{j}): got {}, expected {}", result[[i, j]], expected[[i, j]]);
            }
        }
    }

    #[test]
    fn test_quantize_empty() {
        let w = Array2::<f32>::zeros((0, 0));
        let qt = quantize_linear(&w, QuantizedDtype::Int8);
        assert_eq!(qt.num_elements(), 0);
        assert!(qt.data.is_empty());
    }

    #[test]
    fn test_quantize_small_values() {
        let w = Array2::from_shape_vec((2, 2), vec![1e-6, -1e-6, 2e-6, -2e-6]).unwrap();
        let qt = quantize_linear(&w, QuantizedDtype::Int8);
        let w2 = dequantize_linear(&qt);
        let err = quantization_error(&w, &w2);
        assert!(err < 1e-5, "small values error too high: {err}");
    }

    #[test]
    fn test_int4_packed_zeros() {
        let w = Array2::<f32>::zeros((4, 4));
        let qt = quantize_linear(&w, QuantizedDtype::Int4Packed);
        let w2 = dequantize_linear(&qt);
        let err = quantization_error(&w, &w2);
        assert!(err < 1e-6, "zeros should roundtrip: {err}");
    }
}
