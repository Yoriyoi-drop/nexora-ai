//! BLAS Backend Abstraction for STAR-X Performance Optimization
//!
//! High-performance linear algebra operations dengan multiple backend support:
//! - Intel MKL (Intel Math Kernel Library)
//! - OpenBLAS (Open-source BLAS implementation)
//! - Accelerate (Apple's BLAS framework)
//! - Custom SIMD implementation (fallback)
//! - Auto-detection dan runtime selection

use crate::{DLResult, DeepLearningError};
use ndarray::{ArrayD, Array1, Array2, ArrayView, ArrayViewMut, ShapeBuilder};
use std::arch::x86_64::*;
use std::sync::Arc;
use std::ptr;

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
        // Create new instance with same backend
        Self::with_backend(self.backend).expect("Failed to clone BLAS operations")
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
        std::env::var("MKLROOT").is_ok() || 
        std::env::var("INTEL_MKL_DIR").is_ok() ||
        Self::check_library_exists("libmkl_rt.so") ||
        Self::check_library_exists("mkl_rt.dll")
    }

    /// Check OpenBLAS availability
    fn is_openblas_available() -> bool {
        Self::check_library_exists("libopenblas.so") ||
        Self::check_library_exists("libopenblas.dylib") ||
        Self::check_library_exists("openblas.dll")
    }

    /// Check Apple Accelerate availability (macOS only)
    #[cfg(target_os = "macos")]
    fn is_accelerate_available() -> bool {
        Self::check_library_exists("libAccelerate.dylib")
    }

    /// Check if library exists in system
    fn check_library_exists(lib_name: &str) -> bool {
        // Simple check - in production, use proper library detection
        std::path::Path::new(lib_name).exists()
    }

    /// High-performance matrix multiplication (GEMM)
    pub fn gemm(
        &self,
        alpha: f32,
        a: ArrayView<f32, ndarray::Ix2>,
        b: ArrayView<f32, ndarray::Ix2>,
        beta: f32,
        c: ArrayViewMut<f32, ndarray::Ix2>,
    ) -> DLResult<()> {
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
        y: ArrayViewMut<f32, ndarray::Ix1>,
    ) -> DLResult<()> {
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
            BlasBackend::OpenBLAS => self.batched_gemm_openblas(alpha, a_batch, b_batch, beta, c_batch),
            BlasBackend::Accelerate => self.batched_gemm_accelerate(alpha, a_batch, b_batch, beta, c_batch),
            BlasBackend::CustomSIMD => self.batched_gemm_fallback(alpha, a_batch, b_batch, beta, c_batch),
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
                    unsafe { self.relu_avx2(output.view_mut())?; }
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
                    unsafe { self.tanh_avx2(output.view_mut())?; }
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

    // Backend-specific implementations
    fn gemm_mkl(&self, alpha: f32, a: ArrayView<f32, ndarray::Ix2>, b: ArrayView<f32, ndarray::Ix2>, beta: f32, mut c: ArrayViewMut<f32, ndarray::Ix2>) -> DLResult<()> {
        // Intel MKL SGEMM implementation
        // In production, use actual MKL C bindings
        
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
        
        // For now, fallback to ndarray implementation
        // In production, call cblas_sgemm from MKL
        let mut result = a.dot(&b) * alpha;
        result = result + beta * c.to_owned();
        
        for ((i, j), &val) in result.indexed_iter() {
            c[[i, j]] = val;
        }
        
        Ok(())
    }

    fn gemm_openblas(&self, alpha: f32, a: ArrayView<f32, ndarray::Ix2>, b: ArrayView<f32, ndarray::Ix2>, beta: f32, c: ArrayViewMut<f32, ndarray::Ix2>) -> DLResult<()> {
        // OpenBLAS SGEMM implementation
        // Similar to MKL but with OpenBLAS C bindings
        self.gemm_mkl(alpha, a, b, beta, c) // Fallback for now
    }

    fn gemm_accelerate(&self, alpha: f32, a: ArrayView<f32, ndarray::Ix2>, b: ArrayView<f32, ndarray::Ix2>, beta: f32, c: ArrayViewMut<f32, ndarray::Ix2>) -> DLResult<()> {
        // Apple Accelerate vDSP BLAS implementation
        self.gemm_mkl(alpha, a, b, beta, c) // Fallback for now
    }

    fn gemm_simd(&self, alpha: f32, a: ArrayView<f32, ndarray::Ix2>, b: ArrayView<f32, ndarray::Ix2>, beta: f32, c: ArrayViewMut<f32, ndarray::Ix2>) -> DLResult<()> {
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
            unsafe { self.gemm_simd_avx2(alpha, a, b, beta, c) }
        } else {
            self.gemm_simd_scalar(alpha, a, b, beta, c)
        }
    }

    // SIMD implementations
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
        
        let a_slice = a.as_slice().unwrap();
        let b_slice = b.as_slice().unwrap();
        let c_slice = c.as_slice_mut().unwrap();
        
        // AVX2 implementation
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                
                // Vectorized inner product
                let mut l = 0;
                while l + 8 <= k {
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
                
                c_slice[i * n + j] = alpha * sum + beta * c_slice[i * n + j];
            }
        }
        
        Ok(())
    }

    fn gemm_simd_scalar(&self, alpha: f32, a: ArrayView<f32, ndarray::Ix2>, b: ArrayView<f32, ndarray::Ix2>, beta: f32, mut c: ArrayViewMut<f32, ndarray::Ix2>) -> DLResult<()> {
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

    // Vector operations
    fn gemv_mkl(&self, alpha: f32, a: ArrayView<f32, ndarray::Ix2>, x: ArrayView<f32, ndarray::Ix1>, beta: f32, mut y: ArrayViewMut<f32, ndarray::Ix1>) -> DLResult<()> {
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

    fn gemv_openblas(&self, alpha: f32, a: ArrayView<f32, ndarray::Ix2>, x: ArrayView<f32, ndarray::Ix1>, beta: f32, y: ArrayViewMut<f32, ndarray::Ix1>) -> DLResult<()> {
        self.gemv_mkl(alpha, a, x, beta, y) // Fallback
    }

    fn gemv_accelerate(&self, alpha: f32, a: ArrayView<f32, ndarray::Ix2>, x: ArrayView<f32, ndarray::Ix1>, beta: f32, y: ArrayViewMut<f32, ndarray::Ix1>) -> DLResult<()> {
        self.gemv_mkl(alpha, a, x, beta, y) // Fallback
    }

    fn gemv_simd(&self, alpha: f32, a: ArrayView<f32, ndarray::Ix2>, x: ArrayView<f32, ndarray::Ix1>, beta: f32, y: ArrayViewMut<f32, ndarray::Ix1>) -> DLResult<()> {
        if self.available_features.supports_avx2 {
            unsafe { self.gemv_simd_avx2(alpha, a, x, beta, y) }
        } else {
            self.gemv_mkl(alpha, a, x, beta, y) // Fallback to scalar
        }
    }

    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn gemv_simd_avx2(&self, alpha: f32, a: ArrayView<f32, ndarray::Ix2>, x: ArrayView<f32, ndarray::Ix1>, beta: f32, mut y: ArrayViewMut<f32, ndarray::Ix1>) -> DLResult<()> {
        let (m, n) = a.dim();
        let x_slice = x.as_slice().unwrap();
        let y_slice = y.as_slice_mut().unwrap();
        
        for i in 0..m {
            let mut sum = 0.0f32;
            let mut j = 0;
            
            // Vectorized dot product
            while j + 8 <= n {
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
    fn batched_gemm_mkl(&self, alpha: f32, a_batch: &[ArrayView<f32, ndarray::Ix2>], b_batch: &[ArrayView<f32, ndarray::Ix2>], beta: f32, c_batch: &mut [ArrayViewMut<f32, ndarray::Ix2>]) -> DLResult<()> {
        for (a, b, c) in itertools::izip!(a_batch, b_batch, c_batch) {
            self.gemm_mkl(alpha, *a, *b, beta, c.into())?;
        }
        Ok(())
    }

    fn batched_gemm_openblas(&self, alpha: f32, a_batch: &[ArrayView<f32, ndarray::Ix2>], b_batch: &[ArrayView<f32, ndarray::Ix2>], beta: f32, c_batch: &mut [ArrayViewMut<f32, ndarray::Ix2>]) -> DLResult<()> {
        for (a, b, c) in itertools::izip!(a_batch, b_batch, c_batch) {
            self.gemm_openblas(alpha, *a, *b, beta, c.into())?;
        }
        Ok(())
    }

    fn batched_gemm_accelerate(&self, alpha: f32, a_batch: &[ArrayView<f32, ndarray::Ix2>], b_batch: &[ArrayView<f32, ndarray::Ix2>], beta: f32, c_batch: &mut [ArrayViewMut<f32, ndarray::Ix2>]) -> DLResult<()> {
        for (a, b, c) in itertools::izip!(a_batch, b_batch, c_batch) {
            self.gemm_accelerate(alpha, *a, *b, beta, c.into())?;
        }
        Ok(())
    }

    fn batched_gemm_fallback(&self, alpha: f32, a_batch: &[ArrayView<f32, ndarray::Ix2>], b_batch: &[ArrayView<f32, ndarray::Ix2>], beta: f32, c_batch: &mut [ArrayViewMut<f32, ndarray::Ix2>]) -> DLResult<()> {
        for (a, b, c) in itertools::izip!(a_batch, b_batch, c_batch) {
            self.gemm_simd(alpha, *a, *b, beta, c.into())?;
        }
        Ok(())
    }

    // Activation functions
    #[target_feature(enable = "avx2")]
    unsafe fn relu_avx2(&self, mut output: ArrayViewMut<f32, ndarray::Ix2>) -> DLResult<()> {
        let (m, n) = output.dim();
        let slice = output.as_slice_mut().unwrap();
        
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

/// Activation types for fused operations
#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    ReLU,
    GELU,
    Sigmoid,
    Tanh,
    Swish,
}

// Helper function for AVX2 horizontal sum
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
static mut GLOBAL_BLAS: Option<BlasOperations> = None;
static BLAS_INIT: std::sync::Once = std::sync::Once::new();

/// Get global BLAS operations instance
pub fn get_blas_operations() -> &'static BlasOperations {
    unsafe {
        BLAS_INIT.call_once(|| {
            GLOBAL_BLAS = Some(BlasOperations::auto_detect().expect("Failed to initialize BLAS backend"));
        });
        GLOBAL_BLAS.as_ref().unwrap()
    }
}

/// Initialize BLAS with specific backend (for testing)
pub fn init_blas_with_backend(backend: BlasBackend) -> DLResult<()> {
    unsafe {
        let ops = BlasOperations::with_backend(backend)?;
        GLOBAL_BLAS = Some(ops);
    }
    Ok(())
}
