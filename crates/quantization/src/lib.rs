//! Weight quantization — unified format support.
//!
//! Provides:
//! - `QFormat` — unified enum for F16, BF16, Q8, Q6, Q5, Q4
//! - `QuantizedTensor` — packed weight storage with scales
//! - Quantize/dequantize for all formats (CPU, storage-only)
//! - `quantize_linear` / `dequantize_linear` high-level pipeline
//!
//! WARNING: CPU quantization is storage-only — all computation dequantizes to f32.
//! For GPU inference, `nexora-autograd` provides actual INT8/F16 matmul kernels.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Set to `true` to make it impossible to ignore: this crate stores weights in
/// quantized format but converts back to FP32 for every CPU operation. No speed gain.
/// For GPU inference, `nexora-autograd` provides actual quantized matmul kernels.
pub const QUANTIZATION_IS_STORAGE_ONLY: bool = true;

static QUANT_WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Warn the user at runtime that CPU quantization is storage-only.
/// Call this once at the start of any quantization pipeline.
/// The warning is logged only on the first call (rate-limited to 1).
pub fn warn_storage_only() {
    if !QUANT_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!(
            "nexora-quantization: CPU quantization is STORAGE-ONLY — no performance benefit. \
             Use GPU int8 matmul (nexora-autograd) for actual quantized compute. \
             See QUANTIZATION_IS_STORAGE_ONLY constant."
        );
    }
}

/// Unified quantization format — maps directly to safetensors dtype strings
/// and model config. Covers all formats requested in the 10-year plan.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum QFormat {
    /// 16-bit half-precision float (2 bytes per element)
    F16,
    /// 16-bit brain float (2 bytes per element, wider exponent range)
    BF16,
    /// 8-bit integer, per-group symmetric (default group_size=128)
    Q8 { group_size: usize },
    /// 6-bit integer, per-group symmetric (default group_size=128)
    Q6 { group_size: usize },
    /// 5-bit integer, per-group symmetric (default group_size=128)
    Q5 { group_size: usize },
    /// 4-bit integer, per-group symmetric (default group_size=128)
    Q4 { group_size: usize },
}

impl QFormat {
    pub fn bits_per_element(&self) -> usize {
        match self {
            QFormat::F16 | QFormat::BF16 => 16,
            QFormat::Q8 { .. } => 8,
            QFormat::Q6 { .. } => 6,
            QFormat::Q5 { .. } => 5,
            QFormat::Q4 { .. } => 4,
        }
    }

    /// Compression ratio vs f32 (32 bits).
    pub fn compression_ratio(&self) -> f64 {
        32.0 / self.bits_per_element() as f64
    }

    /// Human-readable format name matching safetensors dtype convention.
    pub fn dtype_name(&self) -> &'static str {
        match self {
            QFormat::F16 => "F16",
            QFormat::BF16 => "BF16",
            QFormat::Q8 { .. } => "Q8",
            QFormat::Q6 { .. } => "Q6",
            QFormat::Q5 { .. } => "Q5",
            QFormat::Q4 { .. } => "Q4",
        }
    }

    /// Default group size for integer formats.
    pub fn default_group_size() -> usize {
        128
    }

    pub fn is_float(&self) -> bool {
        matches!(self, QFormat::F16 | QFormat::BF16)
    }

    pub fn is_integer(&self) -> bool {
        !self.is_float()
    }
}

impl Default for QFormat {
    fn default() -> Self {
        QFormat::F16
    }
}

// ─── Backward-compat alias ─────────────────────────────────────────────────

/// Legacy enum — prefer `QFormat` for new code.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizedDtype {
    Int8,
    Int4Packed,
    Int4Groupwise { group_size: usize },
}

impl QuantizedDtype {
    pub fn bits_per_element(&self) -> usize {
        match self {
            QuantizedDtype::Int8 => 8,
            QuantizedDtype::Int4Packed | QuantizedDtype::Int4Groupwise { .. } => 4,
        }
    }

    pub fn compression_ratio(&self) -> f64 {
        32.0 / self.bits_per_element() as f64
    }
}

impl From<QFormat> for QuantizedDtype {
    fn from(f: QFormat) -> Self {
        match f {
            QFormat::Q8 { .. } => QuantizedDtype::Int8,
            QFormat::Q4 { group_size } => QuantizedDtype::Int4Groupwise { group_size },
            QFormat::Q6 { .. } | QFormat::Q5 { .. } => QuantizedDtype::Int4Packed,
            QFormat::F16 | QFormat::BF16 => QuantizedDtype::Int8,
        }
    }
}

// ─── QuantizedTensor ───────────────────────────────────────────────────────

/// A quantized weight tensor with metadata for dequantization.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub dtype: QuantizedDtype,
    /// Packed quantized data (1 or 2 values per byte)
    pub data: Vec<u8>,
    /// Original shape before quantization
    pub shape: (usize, usize),
    /// Scale factor(s): 1 for per-tensor, N for per-group
    pub scales: Vec<f32>,
    /// Zero point (only for INT8 symmetric)
    pub zero_point: i16,
    /// Optional format tag for Q5/Q6/BF16 extended formats
    pub format: Option<QFormat>,
}

impl QuantizedTensor {
    pub fn num_elements(&self) -> usize {
        self.shape.0 * self.shape.1
    }

    pub fn memory_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4
    }

    pub fn original_memory_bytes(&self) -> usize {
        self.num_elements() * 4
    }

    pub fn compression_ratio(&self) -> f64 {
        self.original_memory_bytes() as f64 / self.memory_bytes() as f64
    }
}

// ─── BF16 conversion ───────────────────────────────────────────────────────

/// Convert f32 to BF16 bits (truncate mantissa from 23 to 7 bits).
#[inline]
pub fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    // Round to nearest even for the truncated mantissa
    let round = ((bits >> 16) & 1) + 0x7FFF;
    ((bits + round) >> 16) as u16
}

/// Convert BF16 bits back to f32.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Quantize f32 slice to BF16 (storage only).
pub fn quantize_f32_to_bf16(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&v| f32_to_bf16(v)).collect()
}

/// Dequantize BF16 back to f32.
pub fn dequantize_bf16_to_f32(data: &[u16]) -> Vec<f32> {
    data.iter().map(|&b| bf16_to_f32(b)).collect()
}

/// Quantize f32 Array2 to BF16, return packed u16 + shape.
pub fn quantize_f32_to_bf16_array(weights: &Array2<f32>) -> (Vec<u16>, (usize, usize)) {
    let shape = weights.dim();
    (quantize_f32_to_bf16(weights.as_slice().unwrap_or(&[])), shape)
}

/// Dequantize BF16 data + shape back to Array2.
pub fn dequantize_bf16_to_array(data: &[u16], rows: usize, cols: usize) -> Array2<f32> {
    let flat = dequantize_bf16_to_f32(data);
    Array2::from_shape_vec((rows, cols), flat).unwrap_or_else(|_| Array2::zeros((rows, cols)))
}

// ─── F16 conversion (moved from transformer crate) ─────────────────────────

/// Convert f32 to F16 bits (IEEE 754 half-precision).
#[inline]
pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3FF;

    (if exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7C00 | if mant != 0 { 0x0200 } else { 0 }
    } else {
        sign | ((exp as u32) << 10) | mant
    }) as u16
}

/// Convert F16 bits back to f32.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        f32::from_bits(sign | ((127 - 15) as u32) << 23 | mant << 13)
    } else if exp == 31 {
        f32::from_bits(sign | 0x7F80_0000 | if mant != 0 { 0x0040_0000 } else { 0 })
    } else {
        f32::from_bits(sign | ((exp + 127 - 15) as u32) << 23 | mant << 13)
    }
}

/// Pack f32 slice into F16 u16 values.
pub fn pack_f32_to_f16(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&v| f32_to_f16(v)).collect()
}

/// Unpack F16 u16 values back to f32.
pub fn unpack_f16_to_f32(data: &[u16]) -> Vec<f32> {
    data.iter().map(|&b| f16_to_f32(b)).collect()
}

// ─── INT8 per-tensor quantization (existing) ───────────────────────────────

pub fn quantize_f32_to_int8(weights: &Array2<f32>) -> (Vec<u8>, f32, i16) {
    let elements: Vec<f32> = weights.iter().copied().collect();
    if elements.is_empty() {
        return (Vec::new(), 1.0, 0);
    }
    let max_val = elements.iter().copied().fold(0.0f32, |a, b| a.max(b.abs())).max(1e-10);
    let scale = max_val / 127.0;
    let mut quantized = Vec::with_capacity(elements.len());
    for &v in &elements {
        let q = (v / scale).round().clamp(-128.0, 127.0) as i8;
        quantized.push(q as u8);
    }
    (quantized, scale, 0)
}

pub fn dequantize_int8_to_f32(data: &[u8], scale: f32, zero_point: i16, rows: usize, cols: usize) -> Array2<f32> {
    let expected = rows * cols;
    let mut out = Array2::zeros((rows, cols));
    for (i, &byte) in data.iter().enumerate().take(expected.min(data.len())) {
        let q = byte as i8;
        out[[i / cols, i % cols]] = (q as i16 - zero_point) as f32 * scale;
    }
    out
}

pub fn matmul_int8(input: &Array2<f32>, w_data: &[u8], w_scale: f32, w_zero: i16, w_rows: usize, w_cols: usize) -> Array2<f32> {
    let batch = input.shape()[0];
    let mut output = Array2::zeros((batch, w_rows));
    for b in 0..batch {
        for r in 0..w_rows {
            let mut dot = 0.0;
            for c in 0..w_cols {
                let idx = r * w_cols + c;
                let w_val = if idx < w_data.len() {
                    (w_data[idx] as i8 as i16 - w_zero) as f32 * w_scale
                } else { 0.0 };
                dot += input[[b, c]] * w_val;
            }
            output[[b, r]] = dot;
        }
    }
    output
}

// ─── INT4 packed per-tensor quantization (existing) ────────────────────────

pub fn quantize_f32_to_int4_packed(weights: &Array2<f32>) -> (Vec<u8>, f32) {
    let elements: Vec<f32> = weights.iter().copied().collect();
    if elements.is_empty() {
        return (Vec::new(), 1.0);
    }
    let max_val = elements.iter().copied().fold(0.0f32, |a, b| a.max(b.abs())).max(1e-10);
    let scale = max_val / 7.0;
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

#[inline]
fn sign_extend_4bit(nibble: u8) -> i8 {
    ((nibble as i8) << 4) >> 4
}

pub fn dequantize_int4_packed_to_f32(data: &[u8], scale: f32, rows: usize, cols: usize) -> Array2<f32> {
    let expected = rows * cols;
    let mut out = Array2::zeros((rows, cols));
    for i in 0..expected.min(data.len() * 2) {
        let byte = data[i / 2];
        let nibble = if i % 2 == 0 { (byte >> 4) & 0x0F } else { byte & 0x0F };
        let val = sign_extend_4bit(nibble) as f32 * scale;
        out[[i / cols, i % cols]] = val;
    }
    out
}

// ─── INT4 groupwise quantization (existing) ────────────────────────────────

pub fn quantize_f32_to_int4_groupwise(weights: &Array2<f32>, group_size: usize) -> (Vec<u8>, Vec<f32>) {
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
        for &v in flat[start..end].iter() { max_abs = max_abs.max(v.abs()); }
        max_abs = max_abs.max(1e-10);
        let scale = max_abs / 7.0;
        for i in start..end {
            let q = (flat[i] / scale).round().clamp(-8.0, 7.0) as i8;
            let nibble = (q as u8) & 0x0F;
            let offset = i - start;
            if offset % 2 == 0 {
                packed[(start + offset) / 2] = (nibble << 4) | (packed[(start + offset) / 2] & 0x0F);
            } else {
                packed[(start + offset) / 2] = (packed[(start + offset) / 2] & 0xF0) | nibble;
            }
        }
        scales.push(scale);
    }
    (packed, scales)
}

pub fn dequantize_int4_groupwise_to_f32(data: &[u8], scales: &[f32], group_size: usize, rows: usize, cols: usize) -> Array2<f32> {
    let n = rows * cols;
    let gs = group_size.max(1);
    let mut out = Array2::zeros((rows, cols));
    for i in 0..n.min(data.len() * 2) {
        let g = i / gs;
        let byte = data[i / 2];
        let nibble = if i % 2 == 0 { (byte >> 4) & 0x0F } else { byte & 0x0F };
        let scale = scales.get(g).copied().unwrap_or(1.0);
        let val = sign_extend_4bit(nibble) as f32 * scale;
        out[[i / cols, i % cols]] = val;
    }
    out
}

// ─── Q6 (6-bit) per-group quantization ─────────────────────────────────────

/// Pack 6-bit values into byte array (4 values per 3 bytes, little-endian packing).
/// Each value occupies 6 bits: bits [5:0] of a conceptual 24-bit sequence.
fn pack_6bit(values: &[i8], max_bit: u8) -> Vec<u8> {
    let n = values.len();
    let packed_len = (n * max_bit as usize + 7) / 8;
    let mut packed = vec![0u8; packed_len];
    let mut bit_pos = 0;
    for &v in values {
        let uv = (v as u8) & ((1 << max_bit) - 1);
        for b in 0..max_bit as usize {
            if (uv >> b) & 1 != 0 {
                let byte_idx = bit_pos / 8;
                if byte_idx < packed_len {
                    packed[byte_idx] |= 1 << (bit_pos % 8);
                }
            }
            bit_pos += 1;
        }
    }
    packed
}

fn unpack_to_i8(data: &[u8], n: usize, bits: u8) -> Vec<i8> {
    let _mask = (1u8 << bits) - 1;
    let mut out = Vec::with_capacity(n);
    let mut bit_pos = 0;
    for _ in 0..n {
        let mut uv = 0u8;
        for b in 0..bits as usize {
            let byte_idx = bit_pos / 8;
            let bit = if byte_idx < data.len() {
                (data[byte_idx] >> (bit_pos % 8)) & 1
            } else { 0 };
            uv |= bit << b;
            bit_pos += 1;
        }
        // Sign-extend: if the msb of the value is set, extend with 1s
        let val = if uv & (1 << (bits - 1)) != 0 {
            (uv as i8).wrapping_sub(1 << bits)
        } else {
            uv as i8
        };
        out.push(val);
    }
    out
}

/// Quantize f32 weights to N-bit per-group (N=5 or 6).
/// Returns (packed_bytes, scales).
pub fn quantize_f32_to_nbit_groupwise(weights: &Array2<f32>, group_size: usize, bits: u8) -> (Vec<u8>, Vec<f32>) {
    let (rows, cols) = weights.dim();
    let n = rows * cols;
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let gs = group_size.max(1);
    let num_groups = (n + gs - 1) / gs;
    let max_quant = (1i32 << (bits - 1)) - 1; // symmetric range: e.g. 15 for 5-bit, 31 for 6-bit
    let mut quantized = Vec::with_capacity(n);
    let mut scales = Vec::with_capacity(num_groups);
    let flat: Vec<f32> = weights.iter().copied().collect();

    for g in 0..num_groups {
        let start = g * gs;
        let end = (start + gs).min(n);
        let mut max_abs = 0.0f32;
        for &v in flat[start..end].iter() { max_abs = max_abs.max(v.abs()); }
        max_abs = max_abs.max(1e-10);
        let scale = max_abs / max_quant as f32;
        for i in start..end {
            let q = (flat[i] / scale).round().clamp(-(max_quant as f32), max_quant as f32) as i8;
            quantized.push(q);
        }
        scales.push(scale);
    }
    let packed = pack_6bit(&quantized, bits);
    (packed, scales)
}

/// Dequantize N-bit data back to f32.
pub fn dequantize_nbit_groupwise_to_f32(data: &[u8], scales: &[f32], group_size: usize, bits: u8, rows: usize, cols: usize) -> Array2<f32> {
    let n = rows * cols;
    let gs = group_size.max(1);
    let vals = unpack_to_i8(data, n, bits);
    let mut out = Array2::zeros((rows, cols));
    for i in 0..n.min(vals.len()) {
        let g = i / gs;
        let scale = scales.get(g).copied().unwrap_or(1.0);
        out[[i / cols, i % cols]] = vals[i] as f32 * scale;
    }
    out
}

// ─── High-level helpers ────────────────────────────────────────────────────

/// Quantize an f32 weight matrix into a `QuantizedTensor` using the given format.
pub fn quantize_linear(weights: &Array2<f32>, dtype: QuantizedDtype) -> QuantizedTensor {
    warn_storage_only();
    let shape = weights.dim();
    match dtype {
        QuantizedDtype::Int8 => {
            let (data, scale, zp) = quantize_f32_to_int8(weights);
            QuantizedTensor { dtype, data, shape, scales: vec![scale], zero_point: zp, format: None }
        }
        QuantizedDtype::Int4Packed => {
            let (data, scale) = quantize_f32_to_int4_packed(weights);
            QuantizedTensor { dtype, data, shape, scales: vec![scale], zero_point: 0, format: None }
        }
        QuantizedDtype::Int4Groupwise { group_size } => {
            let (data, scales) = quantize_f32_to_int4_groupwise(weights, group_size);
            QuantizedTensor { dtype, data, shape, scales, zero_point: 0, format: None }
        }
    }
}

/// Quantize using the unified `QFormat` — converts Q5/Q6 via N-bit path,
/// F16/BF16 via float path, Q8/Q4 via existing paths.
pub fn quantize_with_format(weights: &Array2<f32>, format: QFormat) -> QuantizedTensor {
    warn_storage_only();
    let shape = weights.dim();
    match format {
        QFormat::F16 => {
            let data_u16 = pack_f32_to_f16(weights.as_slice().unwrap_or(&[]));
            let data: Vec<u8> = data_u16.iter().flat_map(|&b| b.to_le_bytes()).collect();
            QuantizedTensor { dtype: QuantizedDtype::Int8, data, shape, scales: vec![1.0], zero_point: 0, format: Some(format) }
        }
        QFormat::BF16 => {
            let data_u16 = quantize_f32_to_bf16(weights.as_slice().unwrap_or(&[]));
            let data: Vec<u8> = data_u16.iter().flat_map(|&b| b.to_le_bytes()).collect();
            QuantizedTensor { dtype: QuantizedDtype::Int8, data, shape, scales: vec![1.0], zero_point: 0, format: Some(format) }
        }
        QFormat::Q8 { group_size } => {
            let gs = if group_size == 0 { QFormat::default_group_size() } else { group_size };
            let (data, scales) = quantize_f32_to_int4_groupwise(weights, gs);
            QuantizedTensor { dtype: QuantizedDtype::Int4Groupwise { group_size: gs }, data, shape, scales, zero_point: 0, format: Some(format) }
        }
        QFormat::Q6 { group_size } => {
            let gs = if group_size == 0 { QFormat::default_group_size() } else { group_size };
            let (data, scales) = quantize_f32_to_nbit_groupwise(weights, gs, 6);
            QuantizedTensor { dtype: QuantizedDtype::Int4Groupwise { group_size: gs }, data, shape, scales, zero_point: 0, format: Some(format) }
        }
        QFormat::Q5 { group_size } => {
            let gs = if group_size == 0 { QFormat::default_group_size() } else { group_size };
            let (data, scales) = quantize_f32_to_nbit_groupwise(weights, gs, 5);
            QuantizedTensor { dtype: QuantizedDtype::Int4Groupwise { group_size: gs }, data, shape, scales, zero_point: 0, format: Some(format) }
        }
        QFormat::Q4 { group_size } => {
            let gs = if group_size == 0 { QFormat::default_group_size() } else { group_size };
            let (data, scales) = quantize_f32_to_int4_groupwise(weights, gs);
            QuantizedTensor { dtype: QuantizedDtype::Int4Groupwise { group_size: gs }, data, shape, scales, zero_point: 0, format: Some(format) }
        }
    }
}

/// Dequantize any `QuantizedTensor` back to f32, respecting `format` tag.
pub fn dequantize_with_format(tensor: &QuantizedTensor) -> Array2<f32> {
    warn_storage_only();
    let (rows, cols) = tensor.shape;
    match tensor.format {
        Some(QFormat::F16) => {
            let data_u16: Vec<u16> = tensor.data.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            dequantize_bf16_to_array(&data_u16, rows, cols)
        }
        Some(QFormat::BF16) => {
            let data_u16: Vec<u16> = tensor.data.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            dequantize_bf16_to_array(&data_u16, rows, cols)
        }
        Some(QFormat::Q6 { group_size }) | Some(QFormat::Q5 { group_size }) => {
            let bits = tensor.format.map(|f| f.bits_per_element() as u8).unwrap_or(4);
            let gs = if group_size == 0 { QFormat::default_group_size() } else { group_size };
            dequantize_nbit_groupwise_to_f32(&tensor.data, &tensor.scales, gs, bits, rows, cols)
        }
        _ => dequantize_linear(tensor),
    }
}

pub fn dequantize_linear(tensor: &QuantizedTensor) -> Array2<f32> {
    warn_storage_only();
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
            dequantize_int4_groupwise_to_f32(&tensor.data, &tensor.scales, group_size, rows, cols)
        }
    }
}

/// Compute the RMSE between original and quantized-dequantized weights.
pub fn quantization_error(original: &Array2<f32>, reconstructed: &Array2<f32>) -> f64 {
    let n = original.len().max(1);
    let sum_sq: f64 = original.iter().zip(reconstructed.iter()).map(|(a, b)| (*a - *b) as f64).map(|d| d * d).sum();
    (sum_sq / n as f64).sqrt()
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_weights() -> Array2<f32> {
        Array2::from_shape_vec((4, 3), vec![0.5, -1.2, 2.3, -3.4, 4.5, -5.6, 6.7, -7.8, 8.9, -9.0, 0.1, -0.2]).unwrap()
    }

    #[test]
    fn test_int8_roundtrip() {
        let w = test_weights();
        let qt = quantize_linear(&w, QuantizedDtype::Int8);
        let w2 = dequantize_linear(&qt);
        let err = quantization_error(&w, &w2);
        assert!(err < 0.06, "INT8 roundtrip error too high: {err}");
        assert!(qt.compression_ratio() >= 3.0, "ratio={}", qt.compression_ratio());
    }

    #[test]
    fn test_int4_packed_roundtrip() {
        let w = test_weights();
        let qt = quantize_linear(&w, QuantizedDtype::Int4Packed);
        let w2 = dequantize_linear(&qt);
        let err = quantization_error(&w, &w2);
        assert!(err < 1.0, "INT4 packed roundtrip error too high: {err}");
        assert!(qt.compression_ratio() >= 4.0, "ratio={}", qt.compression_ratio());
    }

    #[test]
    fn test_int4_groupwise_roundtrip() {
        let w = test_weights();
        let qt = quantize_linear(&w, QuantizedDtype::Int4Groupwise { group_size: 4 });
        let w2 = dequantize_linear(&qt);
        let err = quantization_error(&w, &w2);
        assert!(err < 1.0, "INT4 groupwise roundtrip error too high: {err}");
        assert!(qt.compression_ratio() >= 2.0, "ratio={}", qt.compression_ratio());
    }

    #[test]
    fn test_f16_roundtrip() {
        let w = test_weights();
        let data_u16 = pack_f32_to_f16(w.as_slice().unwrap());
        let back = unpack_f16_to_f32(&data_u16);
        let w2 = Array2::from_shape_vec(w.dim(), back).unwrap();
        let err = quantization_error(&w, &w2);
        assert!(err < 0.01, "F16 roundtrip error too high: {err}");
    }

    #[test]
    fn test_bf16_roundtrip() {
        let w = test_weights();
        let data_u16 = quantize_f32_to_bf16(w.as_slice().unwrap());
        let back = dequantize_bf16_to_f32(&data_u16);
        let w2 = Array2::from_shape_vec(w.dim(), back).unwrap();
        let err = quantization_error(&w, &w2);
        assert!(err < 0.1, "BF16 roundtrip error too high: {err}");
    }

    #[test]
    fn test_q6_roundtrip() {
        let w = test_weights();
        let qt = quantize_with_format(&w, QFormat::Q6 { group_size: 4 });
        let w2 = dequantize_with_format(&qt);
        let err = quantization_error(&w, &w2);
        assert!(err < 1.0, "Q6 roundtrip error too high: {err}");
        assert!((32.0 / 6.0 - qt.format.unwrap().compression_ratio()).abs() < 1.0);
    }

    #[test]
    fn test_q5_roundtrip() {
        let w = test_weights();
        let qt = quantize_with_format(&w, QFormat::Q5 { group_size: 4 });
        let w2 = dequantize_with_format(&qt);
        let err = quantization_error(&w, &w2);
        assert!(err < 1.0, "Q5 roundtrip error too high: {err}");
        assert!((32.0 / 5.0 - qt.format.unwrap().compression_ratio()).abs() < 1.0);
    }

    #[test]
    fn test_qformat_default() {
        assert_eq!(QFormat::default(), QFormat::F16);
        assert_eq!(QFormat::F16.bits_per_element(), 16);
        assert_eq!(QFormat::Q4 { group_size: 128 }.bits_per_element(), 4);
        assert_eq!(QFormat::F16.compression_ratio(), 2.0);
        assert_eq!(QFormat::Q4 { group_size: 128 }.compression_ratio(), 8.0);
        assert_eq!(QFormat::F16.dtype_name(), "F16");
        assert_eq!(QFormat::Q6 { group_size: 128 }.dtype_name(), "Q6");
    }

    #[test]
    fn test_qformat_is_float() {
        assert!(QFormat::F16.is_float());
        assert!(QFormat::BF16.is_float());
        assert!(!QFormat::Q8 { group_size: 128 }.is_float());
        assert!(!QFormat::Q4 { group_size: 128 }.is_float());
    }

    #[test]
    fn test_empty_weights() {
        let w = Array2::<f32>::zeros((0, 0));
        for fmt in [QFormat::F16, QFormat::BF16, QFormat::Q8 { group_size: 128 }, QFormat::Q4 { group_size: 128 }] {
            let qt = quantize_with_format(&w, fmt);
            assert!(qt.data.is_empty(), "{:?} should produce empty data", fmt);
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

    #[test]
    fn test_int8_matmul() {
        let input = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, -1.0, 0.5, -0.5, 0.0]).unwrap();
        let w_f32 = test_weights();
        let (w_data, scale, zp) = quantize_f32_to_int8(&w_f32);
        let expected = input.dot(&w_f32.t());
        let (rows, cols) = w_f32.dim();
        let result = matmul_int8(&input, &w_data, scale, zp, rows, cols);
        for i in 0..2 {
            for j in 0..4 {
                let diff = (result[[i, j]] - expected[[i, j]]).abs();
                assert!(diff < 0.1, "matmul_int8 mismatch at ({i},{j}): got {}, expected {}", result[[i, j]], expected[[i, j]]);
            }
        }
    }
}
