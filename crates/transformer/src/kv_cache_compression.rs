/// 4-bit KV Cache Compression
///
/// Compresses K/V cache entries from f32 (4 bytes per value) to
/// packed 4-bit integers (0.5 bytes per value) — 8× memory savings.
///
/// Per-head symmetric quantization: each KV head has its own scale factor,
/// computed from the running min/max of that head's values.
///
/// Layout:
///   Values are quantized to signed 4-bit range [-8, 7].
///   Two values are packed per byte: byte = (val0 << 4) | (val1 & 0x0F).
///   Scales are stored per-head as f32: `num_kv_heads` scales.

use ndarray::Array2;

/// Quantize a flat K/V slice into packed 4-bit with per-head scales.
///
/// # Arguments
/// * `data` - Flat f32 values, layout `[tokens * kv_dim]` where `kv_dim = num_kv_heads * head_dim`
/// * `num_kv_heads` - Number of KV heads
/// * `head_dim` - Dimension per head
/// * `seq_len` - Number of tokens
///
/// # Returns
/// `(packed_data, scales)` where:
/// * `packed_data` - `u8` vec with 2 values per byte (length = data.len() / 2 + 1 if odd)
/// * `scales` - Per-head f32 scale factors (length = num_kv_heads)
pub fn quantize_4bit(
    data: &[f32],
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> (Vec<u8>, Vec<f32>) {
    let kv_dim = num_kv_heads * head_dim;
    if data.is_empty() || num_kv_heads == 0 || head_dim == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut scales = Vec::with_capacity(num_kv_heads);
    let packed_len = (data.len() + 1) / 2;
    let mut packed = vec![0u8; packed_len];

    for h in 0..num_kv_heads {
        let mut max_abs = 0.0_f32;
        for t in 0..seq_len {
            for d in 0..head_dim {
                let idx = t * kv_dim + h * head_dim + d;
                if idx < data.len() {
                    let val = data[idx].abs();
                    if val > max_abs {
                        max_abs = val;
                    }
                }
            }
        }
        let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
        scales.push(scale);
    }

    for (i, &val) in data.iter().enumerate() {
        let h = (i % kv_dim) / head_dim;
        let scale = scales[h];
        let q = (val / scale).round().clamp(-8.0, 7.0) as i8;
        let byte_idx = i / 2;
        if i % 2 == 0 {
            packed[byte_idx] = ((q as u8) & 0x0F) << 4;
        } else {
            packed[byte_idx] |= (q as u8) & 0x0F;
        }
    }

    (packed, scales)
}

/// Dequantize packed 4-bit data back to f32.
///
/// # Arguments
/// * `packed` - Packed 4-bit data (2 values per byte)
/// * `scales` - Per-head f32 scale factors
/// * `num_kv_heads` - Number of KV heads
/// * `head_dim` - Dimension per head
/// * `seq_len` - Number of tokens
///
/// # Returns
/// Flat f32 vec of length `seq_len * kv_dim`
pub fn dequantize_4bit(
    packed: &[u8],
    scales: &[f32],
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    let kv_dim = num_kv_heads * head_dim;
    let total = seq_len * kv_dim;
    let mut out = Vec::with_capacity(total);

    for i in 0..total {
        let byte_idx = i / 2;
        let u4 = if i % 2 == 0 {
            packed[byte_idx] >> 4
        } else {
            packed[byte_idx] & 0x0F
        };
        let q = if u4 >= 8 { (u4 as i8) - 16 } else { u4 as i8 };
        let h = (i % kv_dim) / head_dim;
        let scale = scales.get(h).copied().unwrap_or(1.0);
        out.push((q as f32) * scale);
    }

    out
}

/// Compress a 2D K/V array (shape `[seq_len, kv_dim]`).
pub fn quantize_kv_2d(
    kv: &Array2<f32>,
    num_kv_heads: usize,
    head_dim: usize,
) -> (Vec<u8>, Vec<f32>) {
    let seq_len = kv.shape()[0];
    quantize_4bit(kv.as_slice().unwrap_or(&[]), num_kv_heads, head_dim, seq_len)
}

/// Decompress back to 2D array (shape `[seq_len, kv_dim]`).
pub fn dequantize_kv_2d(
    packed: &[u8],
    scales: &[f32],
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Array2<f32> {
    let kv_dim = num_kv_heads * head_dim;
    let flat = dequantize_4bit(packed, scales, num_kv_heads, head_dim, seq_len);
    Array2::from_shape_vec((seq_len, kv_dim), flat)
        .unwrap_or_else(|_| Array2::zeros((seq_len, kv_dim)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_roundtrip_small() {
        let data = vec![1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 7.0, -7.0];
        let (packed, scales) = quantize_4bit(&data, 2, 2, 2);
        let deq = dequantize_4bit(&packed, &scales, 2, 2, 2);
        assert_eq!(deq.len(), data.len());
        for (a, b) in data.iter().zip(deq.iter()) {
            let err = (a - b).abs();
            assert!(err < 1.5, "q error too large: {} vs {} (err={})", a, b, err);
        }
    }

    #[test]
    fn test_quantize_empty() {
        let (packed, scales) = quantize_4bit(&[], 4, 64, 0);
        assert!(packed.is_empty());
        assert!(scales.is_empty());
    }

    #[test]
    fn test_quantize_packed_size() {
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let (packed, _) = quantize_4bit(&data, 2, 2, 2);
        // 8 f32 → 4 bytes packed (2 values per byte)
        assert_eq!(packed.len(), 4);
    }

    #[test]
    fn test_dequantize_2d() {
        let kv = Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let (packed, scales) = quantize_kv_2d(&kv, 2, 2);
        let deq = dequantize_kv_2d(&packed, &scales, 2, 2, 2);
        assert_eq!(deq.shape(), &[2, 4]);
    }

    #[test]
    fn test_stress_random() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..1024).map(|_| rng.gen::<f32>() * 10.0 - 5.0).collect();
        let (packed, scales) = quantize_4bit(&data, 8, 16, 8); // 8 heads, head_dim=16, 8 tokens
        let deq = dequantize_4bit(&packed, &scales, 8, 16, 8);
        let max_err: f32 = data.iter().zip(deq.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        // Max theoretical error: scale * 0.5 (rounding). With max_abs/7 scale, max error ≈ (max/7)*0.5
        assert!(max_err < 2.0, "max error too large: {}", max_err);
    }
}

fn clamp(val: f32, min: f32, max: f32) -> f32 {
    if val < min { min } else if val > max { max } else { val }
}
