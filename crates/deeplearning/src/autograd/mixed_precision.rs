use ndarray::ArrayD;

/// Numeric data type / precision
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
}

impl DType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 | DType::BF16 => 4,
            DType::F16 => 2,
        }
    }

    pub fn is_half(&self) -> bool {
        matches!(self, DType::F16 | DType::BF16)
    }

    /// Memory required for `n` elements
    pub fn memory(&self, n: usize) -> usize {
        n * self.size_in_bytes()
    }
}

impl Default for DType {
    fn default() -> Self {
        DType::F32
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
        }
    }
}

// ─── Loss Scaler ──────────────────────────────────────────────────────────────

/// Gradient scaler for FP16 mixed-precision training.
/// Prevents underflow in half-precision gradients by scaling the loss up
/// before backward, then unscale gradients before the optimizer step.
pub struct LossScaler {
    scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: usize,
    steps_since_growth: usize,
    max_scale: f32,
    overflow_count: usize,
    total_steps: usize,
}

impl LossScaler {
    pub fn new(initial_scale: f32) -> Self {
        Self {
            scale: initial_scale,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            steps_since_growth: 0,
            max_scale: 2_f32.powi(24),
            overflow_count: 0,
            total_steps: 0,
        }
    }

    /// Default loss scaler with recommended initial values
    pub fn default() -> Self {
        Self::new(2_f32.powi(16)) // 65536
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Scale loss value before backward pass
    pub fn scale_loss(&self, loss: f32) -> f32 {
        loss * self.scale
    }

    /// Unscale a single gradient array in-place
    pub fn unscale_grad(grad: &mut ArrayD<f32>, scale: f32) {
        grad.mapv_inplace(|g| g / scale);
    }

    /// Check gradient for overflow (inf/nan)
    pub fn has_overflow(grad: &ArrayD<f32>) -> bool {
        grad.iter().any(|&x| x.is_infinite() || x.is_nan())
    }

    /// Update scaler state based on whether gradients contained overflow
    pub fn update(&mut self, grad_overflow: bool) {
        self.total_steps += 1;
        if grad_overflow {
            self.scale *= self.backoff_factor;
            self.steps_since_growth = 0;
            self.overflow_count += 1;
        } else {
            self.steps_since_growth += 1;
            if self.steps_since_growth >= self.growth_interval {
                self.scale = (self.scale * self.growth_factor).min(self.max_scale);
                self.steps_since_growth = 0;
            }
        }
    }

    pub fn overflow_count(&self) -> usize {
        self.overflow_count
    }

    pub fn overflow_rate(&self) -> f32 {
        if self.total_steps == 0 {
            0.0
        } else {
            self.overflow_count as f32 / self.total_steps as f32
        }
    }
}

// ─── FP32 ←→ FP16 conversion helpers ─────────────────────────────────────────

/// Convert f32 slice to half::f16 byte representation
pub fn f32_to_f16_bytes(data: &[f32]) -> Vec<u8> {
    use half::f16;
    let n = data.len();
    let mut out = Vec::with_capacity(n * 2);
    for &v in data {
        out.extend_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
    }
    out
}

/// Convert half::f16 bytes to f32
pub fn f16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    use half::f16;
    data.chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            f16::from_bits(bits).to_f32()
        })
        .collect()
}

/// Convert f32 slice to bf16 byte representation
pub fn f32_to_bf16_bytes(data: &[f32]) -> Vec<u8> {
    // BF16: truncate f32 to lower 16 bits (preserves exponent, loses mantissa)
    let mut out = Vec::with_capacity(data.len() * 2);
    for &v in data {
        let bits = v.to_bits();
        let bf16_bits = (bits >> 16) as u16;
        out.extend_from_slice(&bf16_bits.to_le_bytes());
    }
    out
}

/// Convert bf16 bytes to f32
pub fn bf16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|chunk| {
            let bits = (u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16;
            f32::from_bits(bits)
        })
        .collect()
}

// ─── Half-precision array conversion ──────────────────────────────────────────

/// Convert f32 ndarray to f16 ndarray (byte-compact form)
pub fn array_f32_to_f16_bytes(arr: &ArrayD<f32>) -> Vec<u8> {
    f32_to_f16_bytes(arr.as_slice().expect("non-contiguous array"))
}

/// Convert f32 ndarray to bf16 ndarray (byte-compact form)
pub fn array_f32_to_bf16_bytes(arr: &ArrayD<f32>) -> Vec<u8> {
    f32_to_bf16_bytes(arr.as_slice().expect("non-contiguous array"))
}

/// Expand f16 bytes back to f32 ndarray
pub fn f16_bytes_to_array(data: &[u8], shape: &[usize]) -> ArrayD<f32> {
    let flat = f16_bytes_to_f32(data);
    ArrayD::from_shape_vec(shape.to_vec(), flat).expect("shape mismatch")
}

/// Expand bf16 bytes back to f32 ndarray
pub fn bf16_bytes_to_array(data: &[u8], shape: &[usize]) -> ArrayD<f32> {
    let flat = bf16_bytes_to_f32(data);
    ArrayD::from_shape_vec(shape.to_vec(), flat).expect("shape mismatch")
}

// ─── Automatic Mixed Precision (AMP) Optimizer ───────────────────────────────

use super::tensor::Tensor;

/// Wraps an existing optimizer with FP16/BF16 mixed precision support.
///
/// Keeps FP32 master weights, runs forward/backward in half-precision,
/// and unscales gradients before each optimizer step.
pub struct AmpOptimizer {
    /// The inner optimizer (Adam by default)
    pub inner: super::Adam,
    /// Gradient loss scaler
    loss_scaler: LossScaler,
    /// Target compute dtype (F16 or BF16)
    pub compute_dtype: DType,
    /// FP32 master weight copies
    master_params: Vec<ArrayD<f32>>,
}

impl AmpOptimizer {
    /// Create a new AMP optimizer wrapping the given Adam optimizer.
    pub fn new(adam: super::Adam, compute_dtype: DType) -> Self {
        let master_params = adam
            .parameters
            .iter()
            .map(|p| p.data())
            .collect();
        Self {
            inner: adam,
            loss_scaler: LossScaler::default(),
            compute_dtype,
            master_params,
        }
    }

    /// Set loss scaler parameters
    pub fn set_loss_scaler(&mut self, initial_scale: f32, growth_interval: usize) {
        self.loss_scaler = LossScaler::new(initial_scale);
        self.loss_scaler.growth_interval = growth_interval;
    }

    /// Get reference to loss scaler
    pub fn loss_scaler(&self) -> &LossScaler {
        &self.loss_scaler
    }

    /// Get mutable reference to loss scaler
    pub fn loss_scaler_mut(&mut self) -> &mut LossScaler {
        &mut self.loss_scaler
    }

    /// Zero all parameter gradients
    pub fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    /// Scale loss before backward pass
    pub fn scale_loss(&self, loss: f32) -> f32 {
        self.loss_scaler.scale_loss(loss)
    }

    /// Perform optimizer step with gradient unscaling + overflow handling.
    /// Returns true if step succeeded, false if overflow was detected.
    pub fn step(&mut self) -> bool {
        // Check gradients for overflow
        let has_overflow = self
            .inner
            .parameters
            .iter()
            .any(|p| p.grad().map_or(false, |g| LossScaler::has_overflow(&g)));

        if has_overflow {
            self.loss_scaler.update(true);
            // Zero out overflowed gradients
            self.inner.zero_grad();
            return false;
        }

        // Unscale gradients
        for p in self.inner.parameters.iter() {
            if let Some(g) = p.grad() {
                let mut grad = g;
                LossScaler::unscale_grad(&mut grad, self.loss_scaler.scale);
                p.set_grad(grad);
            }
        }

        // Sync master weights → compute weights (if compute_dtype != F32)
        if self.compute_dtype.is_half() {
            for (i, p) in self.inner.parameters.iter().enumerate() {
                p.set_data(self.master_params[i].clone());
            }
        }

        // Inner optimizer step (on FP32 data)
        self.inner.step();

        // Save updated FP32 weights as master
        for (i, p) in self.inner.parameters.iter().enumerate() {
            self.master_params[i] = p.data();
        }

        self.loss_scaler.update(false);
        true
    }

    /// Cast model parameters to half-precision for forward/backward.
    /// Only needed when compute_dtype != F32 and before each forward pass.
    pub fn cast_model_to_compute_dtype(&self) {
        if !self.compute_dtype.is_half() {
            return;
        }
        // Parameters are kept in FP32 by inner optimizer;
        // we convert to half implicitly via data storage when needed.
        // The actual conversion happens inside ops based on dtype tag.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_basics() {
        assert_eq!(DType::F32.size_in_bytes(), 4);
        assert_eq!(DType::F16.size_in_bytes(), 2);
        assert!(DType::F16.is_half());
        assert!(!DType::F32.is_half());
        assert_eq!(format!("{}", DType::F32), "f32");
        assert_eq!(format!("{}", DType::F16), "f16");
        assert_eq!(format!("{}", DType::BF16), "bf16");
    }

    #[test]
    fn test_f32_f16_roundtrip() {
        let input = vec![1.0f32, 2.5, -3.0, 0.001, 1000.0, -0.0001];
        let bytes = f32_to_f16_bytes(&input);
        let output = f16_bytes_to_f32(&bytes);
        for (a, b) in input.iter().zip(output.iter()) {
            let diff = (a - b).abs();
            assert!(diff < 0.01 || diff / a.abs() < 0.01,
                "f16 roundtrip failed for {}: got {}", a, b);
        }
    }

    #[test]
    fn test_f32_bf16_roundtrip() {
        let input = vec![1.0f32, 2.5, -3.0, 100.0, 0.5, -0.25];
        let bytes = f32_to_bf16_bytes(&input);
        let output = bf16_bytes_to_f32(&bytes);
        for (a, b) in input.iter().zip(output.iter()) {
            let diff = (a - b).abs();
            assert!(diff < 0.1 || diff / a.abs() < 0.01,
                "bf16 roundtrip failed for {}: got {}", a, b);
        }
    }

    #[test]
    fn test_loss_scaler_initial() {
        let scaler = LossScaler::default();
        assert!(scaler.scale() > 0.0);
        assert_eq!(scaler.overflow_count(), 0);
    }

    #[test]
    fn test_loss_scaler_growth() {
        let mut scaler = LossScaler::new(1024.0);
        scaler.growth_interval = 5;
        for _ in 0..5 {
            scaler.update(false);
        }
        // After growth_interval steps without overflow, scale should double
        assert!((scaler.scale() - 2048.0).abs() < 1.0);
    }

    #[test]
    fn test_loss_scaler_backoff() {
        let mut scaler = LossScaler::new(1024.0);
        scaler.update(true);
        assert!((scaler.scale() - 512.0).abs() < 1.0);
        assert_eq!(scaler.overflow_count(), 1);
    }

    #[test]
    fn test_scale_unscale_grad() {
        let mut grad = ArrayD::from_shape_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let scale = 1024.0;
        let original = grad.clone();

        // unscale should divide by scale
        LossScaler::unscale_grad(&mut grad, scale);
        for i in 0..4 {
            assert!((grad[i] - original[i] / scale).abs() < 1e-6);
        }
    }

    #[test]
    fn test_overflow_detection() {
        let mut grad = ArrayD::from_shape_vec(vec![3], vec![1.0, f32::INFINITY, 3.0]).unwrap();
        assert!(LossScaler::has_overflow(&grad));

        grad = ArrayD::from_shape_vec(vec![3], vec![1.0, f32::NAN, 3.0]).unwrap();
        assert!(LossScaler::has_overflow(&grad));

        grad = ArrayD::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        assert!(!LossScaler::has_overflow(&grad));
    }

    #[test]
    fn test_array_conversion() {
        let arr = ArrayD::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let f16_bytes = array_f32_to_f16_bytes(&arr);
        let recovered = f16_bytes_to_array(&f16_bytes, &[2, 3]);
        assert_eq!(recovered.shape(), arr.shape());
    }

    #[test]
    fn test_loss_scaler_scale_loss() {
        let scaler = LossScaler::new(128.0);
        let loss = 0.5;
        let scaled = scaler.scale_loss(loss);
        assert!((scaled - 64.0).abs() < 1e-6);
    }
}
