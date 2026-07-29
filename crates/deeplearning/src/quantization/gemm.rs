//! True INT2/INT4/INT8 matrix multiply — compute directly from quantized weights
//! without full dequantization to FP32.
//!
//! Previously, `QUANTIZATION_IS_STORAGE_ONLY = true` meant all Q4 weights were
//! decompressed to FP32 before compute. This module provides native INT2/INT4 matmul
//! that operates on packed data, reducing VRAM by 4×–16× for weights.
//!
//! # Compute modes
//! - `matmul_int2`: Q2 weight, FP32 activation → FP32 output (CPU, scalar, 4 values/byte)
//! - `matmul_int4`: Q4 weight, FP32 activation → FP32 output (CPU, scalar, 2 values/byte)
//! - `matmul_int8`: INT8 weight, FP32 activation → FP32 output (CPU, scalar)
//! - `dequant_tile_q4`: Per-tile on-the-fly dequant (for GPU kernels)
//! - `dequant_tile_q2`: Per-tile on-the-fly dequant for Q2
//!
//! # Packing format
//! Q4 weights are packed as 2 values per byte (low nibble = index 0, high = index 1).
//! Q2 weights are packed as 4 values per byte (2 bits each, little-endian order).
//! Scales are per-group (default group_size=128 for Q4, 256 for Q2).

/// Compute mode for quantized matmul.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum QuantComputeMode {
    /// Old behavior — dequantize all weights to FP32 first (high VRAM).
    DequantFallback,
    /// Compute directly from Q2 packed weights (16× VRAM savings vs FP32).
    Int2Direct,
    /// Compute directly from Q4 packed weights (4× VRAM savings vs FP32).
    #[default]
    Int4Direct,
    /// Compute directly from INT8 weights (2× VRAM savings vs FP32).
    Int8Direct,
}


/// Pack two Q4 values (0–15 each) into one byte.
/// `low` = first value (lower nibble), `high` = second value (upper nibble).
pub fn pack_q4(low: u8, high: u8) -> u8 {
    (low & 0x0F) | ((high & 0x0F) << 4)
}

/// Unpack one byte into two Q4 values (0–15).
pub fn unpack_q4(packed: u8) -> (u8, u8) {
    (packed & 0x0F, (packed >> 4) & 0x0F)
}

/// INT4 matmul: `C[M, N] = A[M, K] @ B_q4[K/2, N]` with per-group scales.
///
/// # Arguments
/// * `a` - Activation matrix in FP32, row-major `[M * K]`
/// * `b_packed` - Q4 weight matrix packed `[(K/2) * N]` bytes
/// * `scales` - Per-group scale factors `[num_groups * N]`
/// * `out` - Output buffer `[M * N]`, will be filled
/// * `m` - Rows of A and C
/// * `n` - Columns of B and C
/// * `k` - Inner dimension (must be even, will be rounded down)
/// * `group_size` - Elements per quantization group (default 128)
///
/// Every `group_size` elements along K share one scale per column N.
pub fn matmul_int4(
    a: &[f32],
    b_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) {
    let k_aligned = k / 2 * 2; // ensure even
    let num_groups = k_aligned.div_ceil(group_size);
    let groups = num_groups.max(1);

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            let _b_idx = j; // column-major packed access
            for g in 0..groups {
                let g_start = g * group_size;
                let g_end = (g_start + group_size).min(k_aligned);
                let scale = scales[g * n + j];
                for kk in (g_start..g_end).step_by(2) {
                    let a0 = a[i * k + kk];
                    let a1 = if kk + 1 < k { a[i * k + kk + 1] } else { 0.0 };
                    let packed = b_packed[(kk / 2) * n + j];
                    let (b0, b1) = unpack_q4(packed);
                    let b0f = (b0 as f32 - 8.0) * scale; // center around 0
                    let b1f = (b1 as f32 - 8.0) * scale;
                    sum += a0 * b0f + a1 * b1f;
                }
            }
            out[i * n + j] = sum;
        }
    }
}

/// INT8 matmul: `C[M, N] = A[M, K] @ B_int8[K, N]` with per-group scales.
///
/// # Arguments
/// * `a` - Activation in FP32 `[M * K]`
/// * `b` - INT8 weight `[K * N]`
/// * `scales` - Per-group scale factors `[num_groups * N]`
/// * `out` - Output `[M * N]`
/// * `m, n, k` - Matrix dimensions
/// * `group_size` - Elements per quantization group
pub fn matmul_int8(
    a: &[f32],
    b: &[i8],
    scales: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) {
    let num_groups = k.div_ceil(group_size);
    let groups = num_groups.max(1);

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for g in 0..groups {
                let g_start = g * group_size;
                let g_end = (g_start + group_size).min(k);
                let scale = scales[g * n + j];
                let mut g_sum = 0i32;
                for kk in g_start..g_end {
                    let a_val = a[i * k + kk];
                    let b_val = b[kk * n + j] as i32;
                    g_sum += (a_val * b_val as f32) as i32;
                }
                sum += g_sum as f32 * scale;
            }
            out[i * n + j] = sum;
        }
    }
}

/// Dequantize one tile of Q4 weights on-the-fly (for GPU kernel use).
///
/// Reads packed Q4 from `packed[0..tile_m * tile_k / 2]`, reads scales from
/// `scales[0..groups * tile_n]`, writes FP32 to `tile[0..tile_m * tile_k]`.
/// Only dequantizes the tile region, not the full matrix.
pub fn dequant_tile_q4(
    packed: &[u8],
    scales: &[f32],
    tile: &mut [f32],
    tile_m: usize,
    tile_k: usize,
    n: usize,
    group_size: usize,
) {
    let num_groups = tile_k.div_ceil(group_size);
    let groups = num_groups.max(1);

    for i in 0..tile_m {
        for j in 0..tile_k / 2 {
            // packed layout: [K_pairs, N] row-major → idx = j * n + i
            let idx = j * n + i;
            if idx >= packed.len() { break; }
            let p = packed[idx];
            let (b0, b1) = unpack_q4(p);
            let g = j / (group_size / 2);
            let g = g.min(groups - 1);
            let scale = scales[g * n + i];
            tile[i * tile_k + j * 2] = (b0 as f32 - 8.0) * scale;
            if j * 2 + 1 < tile_k {
                tile[i * tile_k + j * 2 + 1] = (b1 as f32 - 8.0) * scale;
            }
        }
    }
}

/// Quantize FP32 weights to packed Q4 format.
///
/// # Returns
/// `(packed, scales)` where:
/// - `packed`: `[K/2 * N]` bytes
/// - `scales`: `[num_groups * N]` FP32 scales
pub fn quantize_fp32_to_q4(
    weights: &[f32],
    n: usize,
    k: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<f32>) {
    let k_aligned = k;
    let num_groups = k_aligned.div_ceil(group_size);
    let groups = num_groups.max(1);
    let mut packed = vec![0u8; (k_aligned / 2) * n];
    let mut scales = vec![0.0f32; groups * n];

    for j in 0..n {
        for g in 0..groups {
            let g_start = g * group_size;
            let g_end = (g_start + group_size).min(k_aligned);
            // Compute scale = max absolute value in group / 7 (for Q4 symmetric)
            let mut max_abs = 0.0f32;
            for kk in g_start..g_end {
                let v = weights[kk * n + j].abs();
                if v > max_abs { max_abs = v; }
            }
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            scales[g * n + j] = scale;

            // Quantize
            for kk in g_start..g_end {
                let v = (weights[kk * n + j] / scale + 8.0).round().clamp(0.0, 15.0) as u8;
                let byte_idx = (kk / 2) * n + j;
                if kk % 2 == 0 {
                    packed[byte_idx] = (packed[byte_idx] & 0xF0) | v;
                } else {
                    packed[byte_idx] = (packed[byte_idx] & 0x0F) | (v << 4);
                }
            }
        }
    }

    (packed, scales)
}

/// Pack four Q2 values (0–3 each) into one byte.
/// `v0` = bits [1:0], `v1` = bits [3:2], `v2` = bits [5:4], `v3` = bits [7:6]
pub fn pack_q2(v0: u8, v1: u8, v2: u8, v3: u8) -> u8 {
    (v0 & 0x03) | ((v1 & 0x03) << 2) | ((v2 & 0x03) << 4) | ((v3 & 0x03) << 6)
}

/// Unpack one byte into four Q2 values (0–3).
pub fn unpack_q2(packed: u8) -> (u8, u8, u8, u8) {
    (packed & 0x03, (packed >> 2) & 0x03, (packed >> 4) & 0x03, (packed >> 6) & 0x03)
}

/// INT2 matmul: `C[M, N] = A[M, K] @ B_q2[K/4, N]` with per-group scales.
///
/// # Arguments
/// * `a` - Activation matrix in FP32, row-major `[M * K]`
/// * `b_packed` - Q2 weight matrix packed `[(K/4) * N]` bytes (4 values per byte)
/// * `scales` - Per-group scale factors `[num_groups * N]`
/// * `out` - Output buffer `[M * N]`, will be filled
/// * `m` - Rows of A and C
/// * `n` - Columns of B and C
/// * `k` - Inner dimension (will be rounded down to multiple of 4)
/// * `group_size` - Elements per quantization group (default 256)
///
/// Every `group_size` elements along K share one scale per column N.
pub fn matmul_int2(
    a: &[f32],
    b_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) {
    let k_aligned = k / 4 * 4;
    let num_groups = k_aligned.div_ceil(group_size);
    let groups = num_groups.max(1);

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for g in 0..groups {
                let g_start = g * group_size;
                let g_end = (g_start + group_size).min(k_aligned);
                let scale = scales[g * n + j];
                for kk in (g_start..g_end).step_by(4) {
                    let a0 = a[i * k + kk];
                    let a1 = if kk + 1 < k { a[i * k + kk + 1] } else { 0.0 };
                    let a2 = if kk + 2 < k { a[i * k + kk + 2] } else { 0.0 };
                    let a3 = if kk + 3 < k { a[i * k + kk + 3] } else { 0.0 };
                    let packed = b_packed[(kk / 4) * n + j];
                    let (b0, b1, b2, b3) = unpack_q2(packed);
                    let b0f = (b0 as f32 - 1.5) * scale;
                    let b1f = (b1 as f32 - 1.5) * scale;
                    let b2f = (b2 as f32 - 1.5) * scale;
                    let b3f = (b3 as f32 - 1.5) * scale;
                    sum += a0 * b0f + a1 * b1f + a2 * b2f + a3 * b3f;
                }
            }
            out[i * n + j] = sum;
        }
    }
}

/// Dequantize one tile of Q2 weights on-the-fly (for GPU kernel use).
///
/// Reads packed Q2 from `packed[0..tile_m * tile_k / 4]`, reads scales from
/// `scales[0..groups * tile_n]`, writes FP32 to `tile[0..tile_m * tile_k]`.
pub fn dequant_tile_q2(
    packed: &[u8],
    scales: &[f32],
    tile: &mut [f32],
    tile_m: usize,
    tile_k: usize,
    n: usize,
    group_size: usize,
) {
    let num_groups = tile_k.div_ceil(group_size);
    let groups = num_groups.max(1);

    for i in 0..tile_m {
        for j in 0..tile_k / 4 {
            let idx = j * n + i;
            if idx >= packed.len() { break; }
            let p = packed[idx];
            let (b0, b1, b2, b3) = unpack_q2(p);
            let g = j / (group_size / 4);
            let g = g.min(groups - 1);
            let scale = scales[g * n + i];
            tile[i * tile_k + j * 4] = (b0 as f32 - 1.5) * scale;
            if j * 4 + 1 < tile_k {
                tile[i * tile_k + j * 4 + 1] = (b1 as f32 - 1.5) * scale;
            }
            if j * 4 + 2 < tile_k {
                tile[i * tile_k + j * 4 + 2] = (b2 as f32 - 1.5) * scale;
            }
            if j * 4 + 3 < tile_k {
                tile[i * tile_k + j * 4 + 3] = (b3 as f32 - 1.5) * scale;
            }
        }
    }
}

/// Quantize FP32 weights to packed Q2 format.
///
/// # Returns
/// `(packed, scales)` where:
/// - `packed`: `[K/4 * N]` bytes (4 values per byte)
/// - `scales`: `[num_groups * N]` FP32 scales
///
/// Range: 0..3 → -1.5..1.5 (centered), symmetric around 0.
pub fn quantize_fp32_to_q2(
    weights: &[f32],
    n: usize,
    k: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<f32>) {
    let k_aligned = k;
    let num_groups = k_aligned.div_ceil(group_size);
    let groups = num_groups.max(1);
    let mut packed = vec![0u8; (k_aligned / 4) * n];
    let mut scales = vec![0.0f32; groups * n];

    for j in 0..n {
        for g in 0..groups {
            let g_start = g * group_size;
            let g_end = (g_start + group_size).min(k_aligned);
            let mut max_abs = 0.0f32;
            for kk in g_start..g_end {
                let v = weights[kk * n + j].abs();
                if v > max_abs { max_abs = v; }
            }
            let scale = if max_abs > 0.0 { max_abs / 1.5 } else { 1.0 };
            scales[g * n + j] = scale;

            for kk in g_start..g_end {
                let v = (weights[kk * n + j] / scale + 1.5).round().clamp(0.0, 3.0) as u8;
                let byte_idx = (kk / 4) * n + j;
                let sub_idx = kk % 4;
                packed[byte_idx] = (packed[byte_idx] & !(0x03 << (sub_idx * 2))) | ((v & 0x03) << (sub_idx * 2));
            }
        }
    }

    (packed, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let pairs = [(0, 0), (3, 7), (15, 15), (8, 4)];
        for &(a, b) in &pairs {
            let packed = pack_q4(a, b);
            let (ua, ub) = unpack_q4(packed);
            assert_eq!(ua, a, "low nibble mismatch");
            assert_eq!(ub, b, "high nibble mismatch");
        }
    }

    #[test]
    fn test_matmul_int4_small() {
        // A = [2, 4], B_q4 = 4x3 quantized
        let m = 2;
        let n = 3;
        let k = 4;
        let group_size = 4;
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // B = [4, 3] in FP32 first: all ones
        let b_fp32 = vec![1.0f32; k * n];
        let (packed, scales) = quantize_fp32_to_q4(&b_fp32, n, k, group_size);

        let mut out = vec![0.0f32; m * n];
        matmul_int4(&a, &packed, &scales, &mut out, m, n, k, group_size);

        // Row 0: 1+2+3+4 = 10 per column
        assert!((out[0] - 10.0).abs() < 0.5, "out[0]={}", out[0]);
        assert!((out[1] - 10.0).abs() < 0.5, "out[1]={}", out[1]);
        // Row 1: 5+6+7+8 = 26 per column
        assert!((out[3] - 26.0).abs() < 1.0, "out[3]={}", out[3]);
    }

    #[test]
    fn test_matmul_int8_small() {
        let m = 2;
        let n = 3;
        let k = 4;
        let group_size = 4;
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b: Vec<i8> = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let scales = vec![1.0f32; n]; // 1 group × n

        let mut out = vec![0.0f32; m * n];
        matmul_int8(&a, &b, &scales, &mut out, m, n, k, group_size);

        assert!((out[0] - 10.0).abs() < 0.1);
        assert!((out[3] - 26.0).abs() < 0.1);
    }

    #[test]
    fn test_quantize_dequant_tile_roundtrip() {
        let n = 2;
        let k = 8;
        let group_size = 4;
        let weights: Vec<f32> = (0..(k * n)).map(|i| (i as f32 - 4.0) * 0.5).collect();

        let (packed, scales) = quantize_fp32_to_q4(&weights, n, k, group_size);

        let tile_m = 2;
        let tile_k = 8;
        let mut tile = vec![0.0f32; tile_m * tile_k];
        dequant_tile_q4(&packed, &scales, &mut tile, tile_m, tile_k, n, group_size);

        // Check first column roughly matches
        for i in 0..tile_m {
            let orig = weights[i * n + 0];
            let deq = tile[i * tile_k + 0];
            assert!((orig - deq).abs() < 1.0, "orig={}, deq={}", orig, deq);
        }
    }

    #[test]
    fn test_quantize_fp32_to_q4_output_dims() {
        let weights = vec![1.0f32; 8 * 4];
        let (packed, scales) = quantize_fp32_to_q4(&weights, 4, 8, 4);
        assert_eq!(packed.len(), (8 / 2) * 4);
        assert_eq!(scales.len(), (8 / 4) * 4);
    }

    #[test]
    fn test_q2_pack_unpack_roundtrip() {
        let quads = [(0, 0, 0, 0), (3, 2, 1, 0), (1, 1, 2, 2), (3, 3, 3, 3)];
        for &(a, b, c, d) in &quads {
            let packed = pack_q2(a, b, c, d);
            let (ua, ub, uc, ud) = unpack_q2(packed);
            assert_eq!(ua, a, "v0 mismatch");
            assert_eq!(ub, b, "v1 mismatch");
            assert_eq!(uc, c, "v2 mismatch");
            assert_eq!(ud, d, "v3 mismatch");
        }
    }

    #[test]
    fn test_matmul_int2_small() {
        let m = 2;
        let n = 3;
        let k = 8;
        let group_size = 8;
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                     0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];

        let b_fp32 = vec![1.0f32; k * n];
        let (packed, scales) = quantize_fp32_to_q2(&b_fp32, n, k, group_size);

        let mut out = vec![0.0f32; m * n];
        matmul_int2(&a, &packed, &scales, &mut out, m, n, k, group_size);

        // All ones in B → sum of A rows
        for j in 0..n {
            assert!((out[j] - 36.0).abs() < 5.0, "out[{j}]={}", out[j]);
            assert!((out[n + j] - 18.0).abs() < 3.0, "out[{n}+{j}]={}", out[n + j]);
        }
    }

    #[test]
    fn test_quantize_fp32_to_q2_output_dims() {
        let weights = vec![1.0f32; 16 * 4];
        let (packed, scales) = quantize_fp32_to_q2(&weights, 4, 16, 8);
        assert_eq!(packed.len(), (16 / 4) * 4);
        assert_eq!(scales.len(), (16 / 8) * 4);
    }

    #[test]
    fn test_dequant_tile_q2_roundtrip() {
        let n = 2;
        let k = 16;
        let group_size = 8;
        let weights: Vec<f32> = (0..(k * n)).map(|i| (i as f32 - 8.0) * 0.25).collect();

        let (packed, scales) = quantize_fp32_to_q2(&weights, n, k, group_size);

        let tile_m = 2;
        let tile_k = 16;
        let mut tile = vec![0.0f32; tile_m * tile_k];
        dequant_tile_q2(&packed, &scales, &mut tile, tile_m, tile_k, n, group_size);

        for i in 0..tile_m {
            let orig = weights[i * n + 0];
            let deq = tile[i * tile_k + 0];
            assert!((orig - deq).abs() < 2.0, "orig={}, deq={}", orig, deq);
        }
    }
}
