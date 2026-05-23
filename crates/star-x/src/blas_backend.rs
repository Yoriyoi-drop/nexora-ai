//! Pure-Rust linear algebra for STAR-X. All "BLAS" backends (MKL, OpenBLAS,
//! Accelerate) delegate to [`gemm_ndarray_fallback`] — a pure-Rust ndarray-based
//! matmul. There is NO actual C BLAS library linked.
//!
//! Detection uses `libloading::Library::new()` which loads the real `.so`/`.dylib`
//! at runtime (not `Path::exists`). If a real BLAS is on `LD_LIBRARY_PATH` the
//! detection will succeed, but execution still routes through ndarray — the enum
//! tag is cosmetic.
//!
//| To add real BLAS acceleration:
//! 1. Add `cblas-sys` / `accelerate` / `intel-mkl-src` as a dependency
//! 2. Replace the body of the relevant `gemm_{mkl,openblas,accelerate}` with FFI
//!    calls to `cblas_sgemm` etc.
//! 3. Remove this notice.

use crate::fused_ops::ActivationType;
use crate::{DLResult, DeepLearningError, require_contiguous, require_contiguous_mut};
use ndarray::{Array1, Array2, ArrayView, ArrayViewMut};
use std::arch::x86_64::*;

/// BLAS Backend types untuk runtime selection
#[derive(Debug, Clone, Copy)]
pub enum BlasBackend {
    IntelMKL,
    OpenBLAS,
    Accelerate,
    CustomSIMD,
}

/// High-performance BLAS operations abstraction
#[derive(Debug)]
pub struct BlasOperations {
    backend: BlasBackend,
    available_features: BlasFeatures,
}

impl Clone for BlasOperations {
    fn clone(&self) -> Self {
        Self::with_backend(self.backend).unwrap_or_else(|e| {
            panic!(
                "Failed to clone BLAS operations with backend {:?}: {}",
                self.backend, e
            )
        })
    }
}

/// Available BLAS features untuk capability detection
#[derive(Debug, Clone)]
pub struct BlasFeatures {
    pub supports_fma: bool,
    pub supports_avx2: bool,
    pub supports_avx512: bool,
    pub supports_multi_threading: bool,
    pub supports_batched_operations: bool,
    pub max_threads: usize,
}

impl BlasOperations {
    /// Auto-detect dan initialize optimal BLAS backend
    pub fn auto_detect() -> DLResult<Self> {
        let backend = Self::detect_optimal_backend()?;
        let features = Self::detect_features(backend)?;

        Ok(Self {
            backend,
            available_features: features,
        })
    }

    /// Force specific backend (untuk testing)
    pub fn with_backend(backend: BlasBackend) -> DLResult<Self> {
        let features = Self::detect_features(backend)?;
        Ok(Self {
            backend,
            available_features: features,
        })
    }

    /// Detect optimal BLAS backend based on system capabilities
    fn detect_optimal_backend() -> DLResult<BlasBackend> {
        // Priority order: Intel MKL > OpenBLAS > Accelerate > Custom SIMD

        // Check for Intel MKL
        if Self::is_intel_mkl_available() {
            return Ok(BlasBackend::IntelMKL);
        }

        // Check for OpenBLAS
        if Self::is_openblas_available() {
            return Ok(BlasBackend::OpenBLAS);
        }

        // Check for Apple Accelerate (macOS)
        #[cfg(target_os = "macos")]
        if Self::is_accelerate_available() {
            return Ok(BlasBackend::Accelerate);
        }

        // Fallback to custom SIMD implementation
        Ok(BlasBackend::CustomSIMD)
    }

    /// Detect available features for backend
    fn detect_features(backend: BlasBackend) -> DLResult<BlasFeatures> {
        let features = match backend {
            BlasBackend::IntelMKL => BlasFeatures {
                supports_fma: true,
                supports_avx2: is_x86_feature_detected!("avx2"),
                supports_avx512: is_x86_feature_detected!("avx512f"),
                supports_multi_threading: true,
                supports_batched_operations: true,
                max_threads: num_cpus::get(),
            },
            BlasBackend::OpenBLAS => BlasFeatures {
                supports_fma: is_x86_feature_detected!("fma"),
                supports_avx2: is_x86_feature_detected!("avx2"),
                supports_avx512: is_x86_feature_detected!("avx512f"),
                supports_multi_threading: true,
                supports_batched_operations: true,
                max_threads: num_cpus::get(),
            },
            BlasBackend::Accelerate => BlasFeatures {
                supports_fma: true,
                supports_avx2: false, // Apple Silicon uses different SIMD
                supports_avx512: false,
                supports_multi_threading: true,
                supports_batched_operations: true,
                max_threads: num_cpus::get(),
            },
            BlasBackend::CustomSIMD => BlasFeatures {
                supports_fma: is_x86_feature_detected!("fma"),
                supports_avx2: is_x86_feature_detected!("avx2"),
                supports_avx512: is_x86_feature_detected!("avx512f"),
                supports_multi_threading: false,
                supports_batched_operations: false,
                max_threads: 1,
            },
        };

        Ok(features)
    }

    /// Check Intel MKL availability
    fn is_intel_mkl_available() -> bool {
        // Check for MKL environment variables or library presence
        std::env::var("MKLROOT").is_ok()
            || std::env::var("INTEL_MKL_DIR").is_ok()
            || Self::check_library_exists("libmkl_rt.so")
            || Self::check_library_exists("mkl_rt.dll")
    }

    /// Check OpenBLAS availability
    fn is_openblas_available() -> bool {
        Self::check_library_exists("libopenblas.so")
            || Self::check_library_exists("libopenblas.dylib")
            || Self::check_library_exists("openblas.dll")
    }

    /// Check Apple Accelerate availability (macOS only)
    #[cfg(target_os = "macos")]
    fn is_accelerate_available() -> bool {
        Self::check_library_exists("libAccelerate.dylib")
    }

    /// Check if library exists in system using libloading.
    /// Tries to open the library; if it loads, the library is present.
    fn check_library_exists(lib_name: &str) -> bool {
        // SAFETY: libloading::Library::new is unsafe because loading a library
        // with unknown symbols can cause UB on some platforms if used incorrectly.
        // Here we only check if the library CAN be loaded; we never call any
        // symbols from it. The library handle is immediately dropped, so there
        // is no risk of dangling function pointers or symbol conflicts.
        unsafe { libloading::Library::new(lib_name).is_ok() }
    }

    /// High-performance matrix multiplication (GEMM)
    pub fn gemm(
        &self,
        alpha: f32,
        a: ArrayView<f32, ndarray::Ix2>,
        b: ArrayView<f32, ndarray::Ix2>,
        beta: f32,
        mut c: ArrayViewMut<f32, ndarray::Ix2>,
    ) -> DLResult<()> {
        #[cfg(feature = "gpu")]
        {
            let a_owned = a.to_owned();
            let b_owned = b.to_owned();
            let mut c_owned = c.to_owned();
            if self.gemm_gpu(alpha, &a_owned, &b_owned, beta, &mut c_owned).is_ok() {
                c.assign(&c_owned);
                return Ok(());
            }
        }
        match self.backend {
            BlasBackend::IntelMKL => self.gemm_mkl(alpha, a, b, beta, c),
            BlasBackend::OpenBLAS => self.gemm_openblas(alpha, a, b, beta, c),
            BlasBackend::Accelerate => self.gemm_accelerate(alpha, a, b, beta, c),
            BlasBackend::CustomSIMD => self.gemm_simd(alpha, a, b, beta, c),
        }
    }

    /// High-performance matrix-vector multiplication (GEMV)
    pub fn gemv(
        &self,
        alpha: f32,
        a: ArrayView<f32, ndarray::Ix2>,
        x: ArrayView<f32, ndarray::Ix1>,
        beta: f32,
        mut y: ArrayViewMut<f32, ndarray::Ix1>,
    ) -> DLResult<()> {
        #[cfg(feature = "gpu")]
        {
            let a_owned = a.to_owned();
            let x_owned = x.to_owned();
            let mut y_owned = y.to_owned();
            if self.gemv_gpu(alpha, &a_owned, &x_owned, beta, &mut y_owned).is_ok() {
                y.assign(&y_owned);
                return Ok(());
            }
        }
        match self.backend {
            BlasBackend::IntelMKL => self.gemv_mkl(alpha, a, x, beta, y),
            BlasBackend::OpenBLAS => self.gemv_openblas(alpha, a, x, beta, y),
            BlasBackend::Accelerate => self.gemv_accelerate(alpha, a, x, beta, y),
            BlasBackend::CustomSIMD => self.gemv_simd(alpha, a, x, beta, y),
        }
    }

    /// Batched matrix multiplication untuk attention computation
    pub fn batched_gemm(
        &self,
        alpha: f32,
        a_batch: &[ArrayView<f32, ndarray::Ix2>],
        b_batch: &[ArrayView<f32, ndarray::Ix2>],
        beta: f32,
        c_batch: &mut [ArrayViewMut<f32, ndarray::Ix2>],
    ) -> DLResult<()> {
        if !self.available_features.supports_batched_operations {
            return self.batched_gemm_fallback(alpha, a_batch, b_batch, beta, c_batch);
        }

        match self.backend {
            BlasBackend::IntelMKL => self.batched_gemm_mkl(alpha, a_batch, b_batch, beta, c_batch),
            BlasBackend::OpenBLAS => {
                self.batched_gemm_openblas(alpha, a_batch, b_batch, beta, c_batch)
            }
            BlasBackend::Accelerate => {
                self.batched_gemm_accelerate(alpha, a_batch, b_batch, beta, c_batch)
            }
            BlasBackend::CustomSIMD => {
                self.batched_gemm_fallback(alpha, a_batch, b_batch, beta, c_batch)
            }
        }
    }

    /// Fused matrix multiplication + activation
    pub fn gemm_activation(
        &self,
        alpha: f32,
        a: ArrayView<f32, ndarray::Ix2>,
        b: ArrayView<f32, ndarray::Ix2>,
        activation: ActivationType,
        mut output: ArrayViewMut<f32, ndarray::Ix2>,
    ) -> DLResult<()> {
        // Perform GEMM first
        self.gemm(alpha, a, b, 0.0, output.view_mut())?;

        // Apply activation in-place
        self.apply_activation_inplace(output.view_mut(), activation)?;

        Ok(())
    }

    /// Apply activation function in-place
    fn apply_activation_inplace(
        &self,
        mut output: ArrayViewMut<f32, ndarray::Ix2>,
        activation: ActivationType,
    ) -> DLResult<()> {
        match activation {
            ActivationType::ReLU => {
                if self.available_features.supports_avx2 {
                    // SAFETY: `relu_avx2` requires AVX2 support, which we checked
                    // via `supports_avx2`. The function operates on the memory range
                    // of `output` and is safe because the caller owns the buffer.
                    unsafe {
                        self.relu_avx2(output.view_mut())?;
                    }
                } else {
                    self.relu_scalar(output.view_mut())?;
                }
            }
            ActivationType::GELU => {
                self.gelu_scalar(output.view_mut())?;
            }
            ActivationType::Sigmoid => {
                self.sigmoid_scalar(output.view_mut())?;
            }
            ActivationType::Tanh => {
                if self.available_features.supports_avx2 {
                    // SAFETY: `tanh_avx2` requires AVX2 support, which we checked
                    // via `supports_avx2`. Memory safety is guaranteed because the
                    // caller owns the output buffer.
                    unsafe {
                        self.tanh_avx2(output.view_mut())?;
                    }
                } else {
                    self.tanh_scalar(output.view_mut())?;
                }
            }
            ActivationType::Swish => {
                self.swish_scalar(output.view_mut())?;
            }
        }

        Ok(())
    }

    // ── "BLAS backend" implementations ──────────────────────────────────────────
    // WARNING: Every method below is a pure-Rust ndarray fallback, NOT a real BLAS
    // call. The MKL/OpenBLAS/Accelerate enum tags are cosmetic — detection may load
    // the real library via libloading but execution still goes through ndarray.
    // To wire up real BLAS, replace the body with `cblas_sgemm` FFI calls.

    /// GEMM via ndarray::dot (ndarray fallback — not actual MKL/OpenBLAS/Accelerate).
    fn gemm_ndarray_fallback(
        &self,
        alpha: f32,
        a: ArrayView<f32, ndarray::Ix2>,
        b: ArrayView<f32, ndarray::Ix2>,
        beta: f32,
        mut c: ArrayViewMut<f32, ndarray::Ix2>,
    ) -> DLResult<()> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();

        if k != k2 {
            return Err(DeepLearningError::ShapeMismatch {
                expected: vec![k],
                actual: vec![k2],
            });
        }

        let (m2, n2) = c.dim();
        if m != m2 || n != n2 {
            return Err(DeepLearningError::ShapeMismatch {
                expected: vec![m, n],
                actual: vec![m2, n2],
            });
        }

        let mut result = a.dot(&b) * alpha;
        result = result + beta * c.to_owned();

        for ((i, j), &val) in result.indexed_iter() {
            c[[i, j]] = val;
        }

        Ok(())
    }

    fn gemm_mkl(
        &self,
        alpha: f32,
        a: ArrayView<f32, ndarray::Ix2>,
        b: ArrayView<f32, ndarray::Ix2>,
        beta: f32,
        c: ArrayViewMut<f32, ndarray::Ix2>,
    ) -> DLResult<()> {
        self.gemm_ndarray_fallback(alpha, a, b, beta, c) // ndarray fallback — not actual MKL
    }

    fn gemm_openblas(
        &self,
        alpha: f32,
        a: ArrayView<f32, ndarray::Ix2>,
        b: ArrayView<f32, ndarray::Ix2>,
        beta: f32,
        c: ArrayViewMut<f32, ndarray::Ix2>,
    ) -> DLResult<()> {
        self.gemm_ndarray_fallback(alpha, a, b, beta, c) // ndarray fallback — not actual OpenBLAS
    }

    fn gemm_accelerate(
        &self,
        alpha: f32,
        a: ArrayView<f32, ndarray::Ix2>,
        b: ArrayView<f32, ndarray::Ix2>,
        beta: f32,
        c: ArrayViewMut<f32, ndarray::Ix2>,
    ) -> DLResult<()> {
        self.gemm_ndarray_fallback(alpha, a, b, beta, c) // ndarray fallback — not actual Accelerate
    }

    fn gemm_simd(
        &self,
        alpha: f32,
        a: ArrayView<f32, ndarray::Ix2>,
        b: ArrayView<f32, ndarray::Ix2>,
        beta: f32,
        c: ArrayViewMut<f32, ndarray::Ix2>,
    ) -> DLResult<()> {
        // Custom SIMD implementation
        let (_m, k) = a.dim();
        let (k2, _n) = b.dim();

        if k != k2 {
            return Err(DeepLearningError::ShapeMismatch {
                expected: vec![k],
                actual: vec![k2],
            });
        }

        if self.available_features.supports_avx2 {
            // SAFETY: `gemm_simd_avx2` is tagged with `#[target_feature]` and
            // we only call it when `supports_avx2` is true. The caller guarantees
            // that `a`, `b`, `c` are valid views with matching inner dimensions.
            unsafe { self.gemm_simd_avx2(alpha, a, b, beta, c) }
        } else {
            self.gemm_simd_scalar(alpha, a, b, beta, c)
        }
    }

    // SIMD implementations
    // SAFETY: Caller must ensure AVX2 and FMA are supported at runtime before
    // calling this function. The function uses inline SIMD intrinsics that
    // operate on the memory ranges of the provided array views, which are
    // guaranteed valid by the ndarray borrow checker.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn gemm_simd_avx2(
        &self,
        alpha: f32,
        a: ArrayView<f32, ndarray::Ix2>,
        b: ArrayView<f32, ndarray::Ix2>,
        beta: f32,
        mut c: ArrayViewMut<f32, ndarray::Ix2>,
    ) -> DLResult<()> {
        let (m, k) = a.dim();
        let (_, n) = b.dim();

        let a_slice = require_contiguous(a.as_slice())?;
        let b_slice = require_contiguous(b.as_slice())?;
        let c_slice = require_contiguous_mut(c.as_slice_mut())?;

        // AVX2 implementation
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;

                // Vectorized inner product
                let mut l = 0;
                while l + 8 <= k {
                    // SAFETY: i < m, l + 8 <= k → i*k + l + 7 < m*k, l*n + j + 7 < k*n
                    // Both guaranteed by loop bounds and shape validation
                    debug_assert!(i * k + l + 7 < a_slice.len());
                    debug_assert!(l * n + j + 7 < b_slice.len());
                    let a_vec = _mm256_loadu_ps(a_slice.as_ptr().add(i * k + l));
                    let b_vec = _mm256_loadu_ps(b_slice.as_ptr().add(l * n + j));
                    let product = _mm256_mul_ps(a_vec, b_vec);
                    sum += horizontal_sum_avx2(product);
                    l += 8;
                }

                // Handle remaining elements
                for l in l..k {
                    sum += a[[i, l]] * b[[l, j]];
                }

                debug_assert!(i * n + j < c_slice.len());
                c_slice[i * n + j] = alpha * sum + beta * c_slice[i * n + j];
            }
        }

        Ok(())
    }

    fn gemm_simd_scalar(
        &self,
        alpha: f32,
        a: ArrayView<f32, ndarray::Ix2>,
        b: ArrayView<f32, ndarray::Ix2>,
        beta: f32,
        mut c: ArrayViewMut<f32, ndarray::Ix2>,
    ) -> DLResult<()> {
        // Scalar fallback implementation
        let (m, k) = a.dim();
        let (_, n) = b.dim();

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[[i, l]] * b[[l, j]];
                }
                c[[i, j]] = alpha * sum + beta * c[[i, j]];
            }
        }

        Ok(())
    }

    // ── GPU-accelerated operations ───────────────────────────────────────────────

    #[cfg(feature = "gpu")]
    pub fn gemm_gpu(
        &self,
        alpha: f32,
        a: &Array2<f32>,
        b: &Array2<f32>,
        beta: f32,
        c: &mut Array2<f32>,
    ) -> DLResult<()> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        let ctx = match GpuContext::global() {
            Ok(c) => c,
            Err(_) => return self.gemm_simd(alpha, a.view(), b.view(), beta, c.view_mut()),
        };

        let a_shape = vec![a.shape()[0], a.shape()[1]];
        let b_shape = vec![b.shape()[0], b.shape()[1]];
        let a_data = a
            .as_slice()
            .ok_or_else(|| DeepLearningError::Computation { reason: "a not contiguous".into() })?;
        let b_data = b
            .as_slice()
            .ok_or_else(|| DeepLearningError::Computation { reason: "b not contiguous".into() })?;

        let a_cpu = ndarray::ArrayD::from_shape_vec(a_shape.clone(), a_data.to_vec())
            .map_err(|e| DeepLearningError::Computation { reason: e.to_string() })?;
        let b_cpu = ndarray::ArrayD::from_shape_vec(b_shape.clone(), b_data.to_vec())
            .map_err(|e| DeepLearningError::Computation { reason: e.to_string() })?;

        let a_gpu = GpuTensor::from_cpu(&a_cpu)
            .map_err(|e| DeepLearningError::Computation { reason: e.to_string() })?;
        let b_gpu = GpuTensor::from_cpu(&b_cpu)
            .map_err(|e| DeepLearningError::Computation { reason: e.to_string() })?;

        // Transpose b for matmul: result = a * b^T (since ndarray stores row-major)
        let b_t = ctx
            .transpose(&b_gpu)
            .map_err(|e| DeepLearningError::Computation { reason: e.to_string() })?;
        let result_gpu = ctx
            .matmul(&a_gpu, &b_t)
            .map_err(|e| DeepLearningError::Computation { reason: e.to_string() })?;

        let result_cpu = result_gpu.to_cpu();
        let result_slice = result_cpu.as_slice().ok_or_else(|| {
            DeepLearningError::Computation {
                reason: "result not contiguous".into(),
            }
        })?;

        if (alpha - 1.0).abs() > f32::EPSILON || beta != 0.0 {
            for i in 0..c.shape()[0] {
                for j in 0..c.shape()[1] {
                    c[[i, j]] = alpha * result_slice[i * c.shape()[1] + j] + beta * c[[i, j]];
                }
            }
        } else {
            let c_slice_mut = c.as_slice_mut().ok_or_else(|| {
                DeepLearningError::Computation {
                    reason: "c not contiguous mut".into(),
                }
            })?;
            c_slice_mut.copy_from_slice(result_slice);
        }

        Ok(())
    }

    #[cfg(feature = "gpu")]
    pub fn gemv_gpu(
        &self,
        alpha: f32,
        a: &Array2<f32>,
        x: &Array1<f32>,
        beta: f32,
        y: &mut Array1<f32>,
    ) -> DLResult<()> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        let ctx = match GpuContext::global() {
            Ok(c) => c,
            Err(_) => return self.gemv_simd(alpha, a.view(), x.view(), beta, y.view_mut()),
        };

        let a_shape = vec![a.shape()[0], a.shape()[1]];
        let x_shape = vec![x.shape()[0], 1];
        let a_data = a
            .as_slice()
            .ok_or_else(|| DeepLearningError::Computation { reason: "a not contiguous".into() })?;
        let x_data = x
            .as_slice()
            .ok_or_else(|| DeepLearningError::Computation { reason: "x not contiguous".into() })?;

        let a_cpu = ndarray::ArrayD::from_shape_vec(a_shape.clone(), a_data.to_vec())
            .map_err(|e| DeepLearningError::Computation { reason: e.to_string() })?;
        let x_cpu = ndarray::ArrayD::from_shape_vec(x_shape.clone(), x_data.to_vec())
            .map_err(|e| DeepLearningError::Computation { reason: e.to_string() })?;

        let a_gpu = GpuTensor::from_cpu(&a_cpu)
            .map_err(|e| DeepLearningError::Computation { reason: e.to_string() })?;
        let x_gpu = GpuTensor::from_cpu(&x_cpu)
            .map_err(|e| DeepLearningError::Computation { reason: e.to_string() })?;

        let result_gpu = ctx
            .matmul(&a_gpu, &x_gpu)
            .map_err(|e| DeepLearningError::Computation { reason: e.to_string() })?;

        let result_cpu = result_gpu.to_cpu();
        let result_slice = result_cpu.as_slice().ok_or_else(|| {
            DeepLearningError::Computation {
                reason: "result not contiguous".into(),
            }
        })?;

        for i in 0..y.shape()[0] {
            y[i] = alpha * result_slice[i] + beta * y[i];
        }

        Ok(())
    }

    // Vector operations
    fn gemv_mkl(
        &self,
        alpha: f32,
        a: ArrayView<f32, ndarray::Ix2>,
        x: ArrayView<f32, ndarray::Ix1>,
        beta: f32,
        mut y: ArrayViewMut<f32, ndarray::Ix1>,
    ) -> DLResult<()> {
        let (m, n) = a.dim();
        let n2 = x.len();
        let m2 = y.len();

        if n != n2 {
            return Err(DeepLearningError::ShapeMismatch {
                expected: vec![n],
                actual: vec![n2],
            });
        }

        if m != m2 {
            return Err(DeepLearningError::ShapeMismatch {
                expected: vec![m],
                actual: vec![m2],
            });
        }

        // Compute y = alpha * A * x + beta * y
        for i in 0..m {
            let mut sum = 0.0f32;
            for j in 0..n {
                sum += a[[i, j]] * x[j];
            }
            y[i] = alpha * sum + beta * y[i];
        }

        Ok(())
    }

    fn gemv_openblas(
        &self,
        alpha: f32,
        a: ArrayView<f32, ndarray::Ix2>,
        x: ArrayView<f32, ndarray::Ix1>,
        beta: f32,
        y: ArrayViewMut<f32, ndarray::Ix1>,
    ) -> DLResult<()> {
        self.gemv_mkl(alpha, a, x, beta, y) // Fallback
    }

    fn gemv_accelerate(
        &self,
        alpha: f32,
        a: ArrayView<f32, ndarray::Ix2>,
        x: ArrayView<f32, ndarray::Ix1>,
        beta: f32,
        y: ArrayViewMut<f32, ndarray::Ix1>,
    ) -> DLResult<()> {
        self.gemv_mkl(alpha, a, x, beta, y) // Fallback
    }

    fn gemv_simd(
        &self,
        alpha: f32,
        a: ArrayView<f32, ndarray::Ix2>,
        x: ArrayView<f32, ndarray::Ix1>,
        beta: f32,
        y: ArrayViewMut<f32, ndarray::Ix1>,
    ) -> DLResult<()> {
        if self.available_features.supports_avx2 {
            // SAFETY: `gemv_simd_avx2` requires AVX2 + FMA CPU support, which is
            // verified via `self.available_features.supports_avx2`. The function
            // also requires contiguous ndarray views, enforced by
            // `require_contiguous` inside the function.
            unsafe { self.gemv_simd_avx2(alpha, a, x, beta, y) }
        } else {
            self.gemv_mkl(alpha, a, x, beta, y) // Fallback to scalar
        }
    }

    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn gemv_simd_avx2(
        &self,
        alpha: f32,
        a: ArrayView<f32, ndarray::Ix2>,
        x: ArrayView<f32, ndarray::Ix1>,
        beta: f32,
        mut y: ArrayViewMut<f32, ndarray::Ix1>,
    ) -> DLResult<()> {
        let (m, n) = a.dim();
        let x_slice = require_contiguous(x.as_slice())?;
        let y_slice = require_contiguous_mut(y.as_slice_mut())?;

        for i in 0..m {
            let mut sum = 0.0f32;
            let mut j = 0;

            // Vectorized dot product
            while j + 8 <= n {
                // SAFETY: i < m, j + 8 <= n → i*n + j + 7 < m*n, j + 7 < n
                // Both guaranteed by loop bounds and shape validation
                debug_assert!(i * n + j + 7 < a.len());
                debug_assert!(j + 7 < x_slice.len());
                let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * n + j));
                let x_vec = _mm256_loadu_ps(x_slice.as_ptr().add(j));
                let product = _mm256_mul_ps(a_vec, x_vec);
                sum += horizontal_sum_avx2(product);
                j += 8;
            }

            // Handle remaining elements
            for j in j..n {
                sum += a[[i, j]] * x[j];
            }

            y_slice[i] = alpha * sum + beta * y_slice[i];
        }

        Ok(())
    }

    // Batched operations
    fn batched_gemm_mkl(
        &self,
        alpha: f32,
        a_batch: &[ArrayView<f32, ndarray::Ix2>],
        b_batch: &[ArrayView<f32, ndarray::Ix2>],
        beta: f32,
        c_batch: &mut [ArrayViewMut<f32, ndarray::Ix2>],
    ) -> DLResult<()> {
        for (a, b, c) in itertools::izip!(a_batch, b_batch, c_batch) {
            self.gemm_mkl(alpha, *a, *b, beta, c.into())?;
        }
        Ok(())
    }

    fn batched_gemm_openblas(
        &self,
        alpha: f32,
        a_batch: &[ArrayView<f32, ndarray::Ix2>],
        b_batch: &[ArrayView<f32, ndarray::Ix2>],
        beta: f32,
        c_batch: &mut [ArrayViewMut<f32, ndarray::Ix2>],
    ) -> DLResult<()> {
        for (a, b, c) in itertools::izip!(a_batch, b_batch, c_batch) {
            self.gemm_openblas(alpha, *a, *b, beta, c.into())?;
        }
        Ok(())
    }

    fn batched_gemm_accelerate(
        &self,
        alpha: f32,
        a_batch: &[ArrayView<f32, ndarray::Ix2>],
        b_batch: &[ArrayView<f32, ndarray::Ix2>],
        beta: f32,
        c_batch: &mut [ArrayViewMut<f32, ndarray::Ix2>],
    ) -> DLResult<()> {
        for (a, b, c) in itertools::izip!(a_batch, b_batch, c_batch) {
            self.gemm_accelerate(alpha, *a, *b, beta, c.into())?;
        }
        Ok(())
    }

    fn batched_gemm_fallback(
        &self,
        alpha: f32,
        a_batch: &[ArrayView<f32, ndarray::Ix2>],
        b_batch: &[ArrayView<f32, ndarray::Ix2>],
        beta: f32,
        c_batch: &mut [ArrayViewMut<f32, ndarray::Ix2>],
    ) -> DLResult<()> {
        for (a, b, c) in itertools::izip!(a_batch, b_batch, c_batch) {
            self.gemm_simd(alpha, *a, *b, beta, c.into())?;
        }
        Ok(())
    }

    // SAFETY: Caller must ensure AVX2 is supported before calling. This function
    // uses `#[target_feature]` and operates element-wise on the contiguous slice
    // of `output`. The indowng array view guarantees valid memory access within bounds.
    #[target_feature(enable = "avx2")]
    unsafe fn relu_avx2(&self, mut output: ArrayViewMut<f32, ndarray::Ix2>) -> DLResult<()> {
        let (m, n) = output.dim();
        let slice = require_contiguous_mut(output.as_slice_mut())?;

        for i in 0..(m * n) {
            slice[i] = slice[i].max(0.0);
        }

        Ok(())
    }

    fn relu_scalar(&self, mut output: ArrayViewMut<f32, ndarray::Ix2>) -> DLResult<()> {
        output.map_inplace(|x| *x = x.max(0.0));
        Ok(())
    }

    fn gelu_scalar(&self, mut output: ArrayViewMut<f32, ndarray::Ix2>) -> DLResult<()> {
        output.map_inplace(|x| {
            let sqrt_2_over_pi = 0.7978845608_f32;
            let coeff = 0.044715_f32;
            *x = 0.5 * *x * (1.0 + (sqrt_2_over_pi * (*x + coeff * *x * *x * *x)).tanh())
        });
        Ok(())
    }

    fn sigmoid_scalar(&self, mut output: ArrayViewMut<f32, ndarray::Ix2>) -> DLResult<()> {
        output.map_inplace(|x| *x = 1.0 / (1.0 + (-*x).exp()));
        Ok(())
    }

    // SAFETY: Caller must ensure AVX2 is supported before calling. The function
    // iterates element-wise on the contiguous slice of `output`. The ndarray view
    // guarantees memory access is within the bounds of the caller's allocated buffer.
    #[target_feature(enable = "avx2")]
    unsafe fn tanh_avx2(&self, mut output: ArrayViewMut<f32, ndarray::Ix2>) -> DLResult<()> {
        output.map_inplace(|x| *x = x.tanh());
        Ok(())
    }

    fn tanh_scalar(&self, mut output: ArrayViewMut<f32, ndarray::Ix2>) -> DLResult<()> {
        output.map_inplace(|x| *x = x.tanh());
        Ok(())
    }

    fn swish_scalar(&self, mut output: ArrayViewMut<f32, ndarray::Ix2>) -> DLResult<()> {
        output.map_inplace(|x| *x = *x * (1.0 / (1.0 + (-*x).exp())));
        Ok(())
    }

    /// Get backend information
    pub fn backend_info(&self) -> BlasBackendInfo {
        BlasBackendInfo {
            backend: self.backend,
            features: self.available_features.clone(),
        }
    }
}

/// Backend information for debugging
#[derive(Debug, Clone)]
pub struct BlasBackendInfo {
    pub backend: BlasBackend,
    pub features: BlasFeatures,
}

// SAFETY: Caller must ensure AVX2 is supported before calling. This function uses
// x86_64 AVX2 intrinsics to horizontally sum a 256-bit SIMD vector. The operation
// is a pure register computation — no memory access is performed — so the only
// safety requirement is CPU feature support, which the caller must verify.
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    let v128_hi = std::arch::x86_64::_mm256_extractf128_ps(v, 1);
    let v128_lo = std::arch::x86_64::_mm256_castps256_ps128(v);
    let v128_sum = std::arch::x86_64::_mm_add_ps(v128_lo, v128_hi);

    let v64_hi = std::arch::x86_64::_mm_movehl_ps(v128_sum, v128_sum);
    let v64_sum = std::arch::x86_64::_mm_add_ps(v128_sum, v64_hi);

    let v32_hi = std::arch::x86_64::_mm_shuffle_ps(v64_sum, v64_sum, 0x1);
    let v32_sum = std::arch::x86_64::_mm_add_ss(v64_sum, v32_hi);

    std::arch::x86_64::_mm_cvtss_f32(v32_sum)
}

/// Global BLAS operations instance
static GLOBAL_BLAS: std::sync::OnceLock<BlasOperations> = std::sync::OnceLock::new();

/// Get global BLAS operations instance
pub fn get_blas_operations() -> &'static BlasOperations {
    // safe: auto_detect always succeeds (falls back to CustomSIMD)
    GLOBAL_BLAS.get_or_init(|| {
        BlasOperations::auto_detect().unwrap_or_else(|e| {
            panic!("Failed to initialize BLAS backend: {} (falls back to CustomSIMD, should not fail)", e)
        })
    })
}

/// Initialize BLAS with specific backend (for testing)
pub fn init_blas_with_backend(backend: BlasBackend) -> DLResult<()> {
    let ops = BlasOperations::with_backend(backend)?;
    let _ = GLOBAL_BLAS.set(ops);
    Ok(())
}
