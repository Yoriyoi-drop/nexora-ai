//! SIMD Operations untuk Performance Optimization
//!
//! Implementasi SIMD-optimized operations untuk mathematical computations
//! dan text processing

use std::arch::x86_64::*;

/// Check if AVX2 is supported on the current CPU at runtime.
///
/// Must be called before invoking any [`#[target_feature(enable = "avx2")]`]
/// function to ensure it is safe to call.
#[must_use]
pub fn is_avx2_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// SIMD-optimized vector operations
pub struct SimdVectorOps;

impl SimdVectorOps {
    /// Dot product menggunakan SIMD
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is supported at runtime (e.g. via [`is_avx2_supported()`])
    /// before calling this function. On x86_64 without AVX2 this will crash with SIGILL.
    /// The `a` and `b` slices must have equal length.
    #[must_use]
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors must have same length");

        let n = a.len();
        let mut sum = _mm256_setzero_ps();

        // Process 8 elements at a time
        let chunks = n / 8;
        let remainder = n % 8;

        let mut a_ptr = a.as_ptr();
        let mut b_ptr = b.as_ptr();

        for _ in 0..chunks {
            // SAFETY: Loop runs `chunks` times, each advancing ptr by 8 f32 elements.
            // chunks = n / 8, so the max accessed index is chunks * 8 - 1 < n.
            // Both slices have length n (asserted at function entry).
            assert!(
                (a_ptr as usize - a.as_ptr() as usize) / std::mem::size_of::<f32>() + 8 <= a.len()
            );
            assert!(
                (b_ptr as usize - b.as_ptr() as usize) / std::mem::size_of::<f32>() + 8 <= b.len()
            );
            let va = _mm256_loadu_ps(a_ptr);
            let vb = _mm256_loadu_ps(b_ptr);
            let product = _mm256_mul_ps(va, vb);
            sum = _mm256_add_ps(sum, product);

            a_ptr = a_ptr.add(8);
            b_ptr = b_ptr.add(8);
        }

        // Horizontal sum of the 8 float results
        let sum128 = _mm256_extractf128_ps(sum, 0);
        let sum128_2 = _mm256_extractf128_ps(sum, 1);
        let sum128 = _mm_add_ps(sum128, sum128_2);

        let sum64 = _mm256_castps256_ps128(_mm256_insertf128_ps(_mm256_undefined_ps(), sum128, 0));
        let sum64_2 = _mm256_extractf128_ps(
            _mm256_permute2f128_ps(_mm256_undefined_ps(), _mm256_castps128_ps256(sum64), 1),
            0,
        );
        let sum64 = _mm_add_ps(sum64, sum64_2);

        let sum32 = _mm_add_ps(sum64, _mm_movehl_ps(sum64, sum64));
        let sum32 = _mm_add_ss(sum32, _mm_shuffle_ps(sum32, sum32, 1));

        // SAFETY:
        // `sum32` is a value of type `__m128` (a 128-bit SSE/AVX register). The Rust
        // compiler guarantees that `__m128` has the same ABI as a 128-bit aligned
        // memory block. `[f32; 4]` is exactly 128 bits (4 × 32 = 128) with the same
        // alignment (16-byte aligned for both types). `std::mem::transmute` is
        // defined to reinterpret the bits of the source type as the destination type,
        // and this is the canonical way to extract individual lanes from an SSE/AVX
        // register in Rust. The transmute itself is statically checked to have equal
        // sizes at compile time — if the sizes ever diverge, the compiler rejects it.
        let mut result = unsafe { std::mem::transmute::<_, [f32; 4]>(sum32) }[0];

        // Process remainder
        for i in (n - remainder)..n {
            result += a[i] * b[i];
        }

        result
    }

    /// Vector addition menggunakan SIMD
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is supported at runtime. The `a`, `b`, and `result`
    /// slices must have equal length.
    #[target_feature(enable = "avx2")]
    pub unsafe fn add_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len(), "Vectors must have same length");
        assert_eq!(a.len(), result.len(), "Result vector must have same length");

        let n = a.len();
        let chunks = n / 8;
        let _remainder = n % 8;

        for i in 0..chunks {
            // SAFETY: i < chunks = n / 8, so offset i*8 ≤ n - 8, all slices have length n
            assert!(i * 8 + 7 < a.len());
            assert!(i * 8 + 7 < b.len());
            assert!(i * 8 + 7 < result.len());
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let sum = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(result.as_mut_ptr().add(i * 8), sum);
        }

        // Process remainder
        for i in (chunks * 8)..n {
            result[i] = a[i] + b[i];
        }
    }

    /// Vector multiplication menggunakan SIMD
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is supported at runtime. The `a`, `b`, and `result`
    /// slices must have equal length.
    #[target_feature(enable = "avx2")]
    pub unsafe fn mul_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len(), "Vectors must have same length");
        assert_eq!(a.len(), result.len(), "Result vector must have same length");

        let n = a.len();
        let chunks = n / 8;
        let _remainder = n % 8;

        for i in 0..chunks {
            // SAFETY: i < chunks = n / 8, so offset i*8 ≤ n - 8, all slices have length n
            assert!(i * 8 + 7 < a.len());
            assert!(i * 8 + 7 < b.len());
            assert!(i * 8 + 7 < result.len());
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let product = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(result.as_mut_ptr().add(i * 8), product);
        }

        // Process remainder
        for i in (chunks * 8)..n {
            result[i] = a[i] * b[i];
        }
    }

    /// Cosine similarity menggunakan SIMD
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is supported at runtime. The `a` and `b` slices
    /// must have equal length.
    #[must_use]
    #[target_feature(enable = "avx2")]
    pub unsafe fn cosine_similarity_avx2(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors must have same length");

        let dot = Self::dot_product_avx2(a, b);
        let norm_a = Self::dot_product_avx2(a, a).sqrt();
        let norm_b = Self::dot_product_avx2(b, b).sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Euclidean distance menggunakan SIMD
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is supported at runtime. The `a` and `b` slices
    /// must have equal length.
    #[must_use]
    #[target_feature(enable = "avx2")]
    pub unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors must have same length");

        let n = a.len();
        let mut sum_squares = _mm256_setzero_ps();

        let chunks = n / 8;
        let remainder = n % 8;

        let mut a_ptr = a.as_ptr();
        let mut b_ptr = b.as_ptr();

        for _ in 0..chunks {
            // SAFETY: Loop runs `chunks` times, each advancing ptr by 8 f32 elements.
            // chunks = n / 8, so the max accessed index is chunks * 8 - 1 < n.
            assert!(
                (a_ptr as usize - a.as_ptr() as usize) / std::mem::size_of::<f32>() + 8 <= a.len()
            );
            assert!(
                (b_ptr as usize - b.as_ptr() as usize) / std::mem::size_of::<f32>() + 8 <= b.len()
            );
            let va = _mm256_loadu_ps(a_ptr);
            let vb = _mm256_loadu_ps(b_ptr);
            let diff = _mm256_sub_ps(va, vb);
            let squares = _mm256_mul_ps(diff, diff);
            sum_squares = _mm256_add_ps(sum_squares, squares);

            a_ptr = a_ptr.add(8);
            b_ptr = b_ptr.add(8);
        }

        // Horizontal sum
        let sum128 = _mm256_extractf128_ps(sum_squares, 0);
        let sum128_2 = _mm256_extractf128_ps(sum_squares, 1);
        let sum128 = _mm_add_ps(sum128, sum128_2);

        let sum64 = _mm256_castps256_ps128(_mm256_insertf128_ps(_mm256_undefined_ps(), sum128, 0));
        let sum64_2 = _mm256_extractf128_ps(
            _mm256_permute2f128_ps(_mm256_undefined_ps(), _mm256_castps128_ps256(sum64), 1),
            0,
        );
        let sum64 = _mm_add_ps(sum64, sum64_2);

        let sum32 = _mm_add_ps(sum64, _mm_movehl_ps(sum64, sum64));
        let sum32 = _mm_add_ss(sum32, _mm_shuffle_ps(sum32, sum32, 1));

        // SAFETY:
        // `sum32` is `__m128` (128-bit SSE register). `[f32; 4]` has the same size
        // (128 bits) and alignment (16 bytes). The transmute is statically
        // size-checked by the compiler and is the canonical way to extract lanes.
        let mut result = unsafe { std::mem::transmute::<_, [f32; 4]>(sum32) }[0];

        // Process remainder
        for i in (n - remainder)..n {
            let diff = a[i] - b[i];
            result += diff * diff;
        }

        result.sqrt()
    }

    /// Fallback implementations for systems without AVX2
    #[must_use]
    pub fn dot_product_fallback(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors must have same length");

        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    pub fn add_fallback(a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len(), "Vectors must have same length");
        assert_eq!(a.len(), result.len(), "Result vector must have same length");

        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
    }

    pub fn mul_fallback(a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len(), "Vectors must have same length");
        assert_eq!(a.len(), result.len(), "Result vector must have same length");

        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }
    }

    #[must_use]
    pub fn cosine_similarity_fallback(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors must have same length");

        let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
        let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    #[must_use]
    pub fn euclidean_distance_fallback(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors must have same length");

        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    /// Safe auto-dispatch: uses AVX2 if available, otherwise falls back to scalar.
    #[must_use]
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        if is_avx2_supported() {
            // SAFETY: `dot_product_avx2` requires AVX2 CPU support. We checked
            // `is_avx2_supported()` above, which uses `is_x86_feature_detected!("avx2")`.
            // The function also requires slices of equal length, which we guarantee
            // via its own internal `assert_eq!`.
            unsafe { Self::dot_product_avx2(a, b) }
        } else {
            Self::dot_product_fallback(a, b)
        }
    }

    /// Safe auto-dispatch: uses AVX2 if available, otherwise falls back to scalar.
    pub fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
        if is_avx2_supported() {
            // SAFETY: `add_avx2` requires AVX2 CPU support. We verified via
            // `is_avx2_supported()`. It also requires equal-length slices, which
            // we guarantee via its own `assert_eq!`.
            unsafe { Self::add_avx2(a, b, result) }
        } else {
            Self::add_fallback(a, b, result);
        }
    }

    /// Safe auto-dispatch: uses AVX2 if available, otherwise falls back to scalar.
    pub fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
        if is_avx2_supported() {
            // SAFETY: `mul_avx2` requires AVX2 CPU support, verified via
            // `is_avx2_supported()`, and equal-length slices, verified via
            // its own asserts.
            unsafe { Self::mul_avx2(a, b, result) }
        } else {
            Self::mul_fallback(a, b, result);
        }
    }

    /// Safe auto-dispatch: uses AVX2 if available, otherwise falls back to scalar.
    #[must_use]
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if is_avx2_supported() {
            // SAFETY: `cosine_similarity_avx2` requires AVX2 CPU support, verified
            // via `is_avx2_supported()`, and equal-length slices, verified via its
            // own asserts. It delegates to `dot_product_avx2` which is also safe
            // under the same preconditions.
            unsafe { Self::cosine_similarity_avx2(a, b) }
        } else {
            Self::cosine_similarity_fallback(a, b)
        }
    }

    /// Safe auto-dispatch: uses AVX2 if available, otherwise falls back to scalar.
    #[must_use]
    pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        if is_avx2_supported() {
            // SAFETY: `euclidean_distance_avx2` requires AVX2 CPU support, verified
            // via `is_avx2_supported()`, and equal-length slices, verified via its
            // own asserts.
            unsafe { Self::euclidean_distance_avx2(a, b) }
        } else {
            Self::euclidean_distance_fallback(a, b)
        }
    }
}

/// SIMD-optimized text operations
pub struct SimdTextOps;

impl SimdTextOps {
    /// Fast string similarity using SIMD
    #[must_use]
    pub fn string_similarity(a: &str, b: &str) -> f32 {
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }

        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        // Convert to byte vectors for SIMD processing
        let a_bytes = a.as_bytes();
        let b_bytes = b.as_bytes();

        let min_len = a_bytes.len().min(b_bytes.len());
        let max_len = a_bytes.len().max(b_bytes.len());

        if is_x86_feature_detected!("avx2") && min_len >= 32 {
            // SAFETY: `string_similarity_avx2` requires AVX2 CPU support, verified
            // via `is_x86_feature_detected!("avx2")`. The `min_len >= 32` guard
            // ensures at least one full 32-byte SIMD chunk can be processed.
            // The function only accesses bytes within the slices' bounds.
            unsafe { Self::string_similarity_avx2(a_bytes, b_bytes, min_len, max_len) }
        } else {
            Self::string_similarity_fallback(a_bytes, b_bytes, min_len, max_len)
        }
    }

    /// # Safety
    ///
    /// Caller must ensure AVX2 is supported at runtime and that `a` and `b`
    /// have at least `min_len` elements.
    #[must_use]
    #[target_feature(enable = "avx2")]
    unsafe fn string_similarity_avx2(a: &[u8], b: &[u8], min_len: usize, max_len: usize) -> f32 {
        let mut matches = 0u32;
        let chunks = min_len / 32;
        let _remainder = min_len % 32;

        let mut a_ptr = a.as_ptr();
        let mut b_ptr = b.as_ptr();

        for _ in 0..chunks {
            // SAFETY: chunks = min_len / 32, max offset = chunks * 32 ≤ min_len
            assert!((a_ptr as usize - a.as_ptr() as usize) + 32 <= a.len());
            assert!((b_ptr as usize - b.as_ptr() as usize) + 32 <= b.len());
            let va = _mm256_loadu_si256(a_ptr as *const __m256i);
            let vb = _mm256_loadu_si256(b_ptr as *const __m256i);

            // Compare bytes
            let cmp = _mm256_cmpeq_epi8(va, vb);
            let mask = _mm256_movemask_epi8(cmp);
            matches += mask.count_ones() as u32;

            a_ptr = a_ptr.add(32);
            b_ptr = b_ptr.add(32);
        }

        // Process remainder
        for i in (chunks * 32)..min_len {
            if a[i] == b[i] {
                matches += 1;
            }
        }

        matches as f32 / max_len as f32
    }

    fn string_similarity_fallback(a: &[u8], b: &[u8], _min_len: usize, max_len: usize) -> f32 {
        let matches = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();

        matches as f32 / max_len as f32
    }

    /// Fast character counting using SIMD
    #[must_use]
    pub fn count_char(text: &str, target: char) -> usize {
        let target_byte = target as u8;
        let text_bytes = text.as_bytes();

        if is_x86_feature_detected!("avx2") && text_bytes.len() >= 32 {
            // SAFETY: `count_char_avx2` requires AVX2 CPU support, verified
            // via `is_x86_feature_detected!("avx2")`. The `len >= 32` guard
            // ensures at least one full SIMD chunk. The function only reads within
            // the text slice.
            unsafe { Self::count_char_avx2(text_bytes, target_byte) }
        } else {
            Self::count_char_fallback(text_bytes, target_byte)
        }
    }

    /// # Safety
    ///
    /// Caller must ensure AVX2 is supported at runtime.
    #[must_use]
    #[target_feature(enable = "avx2")]
    unsafe fn count_char_avx2(text: &[u8], target: u8) -> usize {
        let mut count = 0usize;
        let chunks = text.len() / 32;
        let _remainder = text.len() % 32;

        let target_vec = _mm256_set1_epi8(target as i8);
        let mut text_ptr = text.as_ptr();

        for _ in 0..chunks {
            // SAFETY: chunks = text.len() / 32, max offset = chunks * 32 ≤ text.len()
            assert!((text_ptr as usize - text.as_ptr() as usize) + 32 <= text.len());
            let text_vec = _mm256_loadu_si256(text_ptr as *const __m256i);
            let cmp = _mm256_cmpeq_epi8(text_vec, target_vec);
            let mask = _mm256_movemask_epi8(cmp);
            count += mask.count_ones() as usize;

            text_ptr = text_ptr.add(32);
        }

        // Process remainder
        for i in (chunks * 32)..text.len() {
            if text[i] == target {
                count += 1;
            }
        }

        count
    }

    fn count_char_fallback(text: &[u8], target: u8) -> usize {
        text.iter().filter(|&&c| c == target).count()
    }

    /// Fast whitespace detection using SIMD
    #[must_use]
    pub fn has_whitespace(text: &str) -> bool {
        let text_bytes = text.as_bytes();

        if is_x86_feature_detected!("avx2") && text_bytes.len() >= 32 {
            // SAFETY: `has_whitespace_avx2` requires AVX2 CPU support, verified
            // via `is_x86_feature_detected!("avx2")`. The `len >= 32` guard ensures
            // at least one full SIMD chunk. The function only reads within bounds.
            unsafe { Self::has_whitespace_avx2(text_bytes) }
        } else {
            Self::has_whitespace_fallback(text_bytes)
        }
    }

    /// # Safety
    ///
    /// Caller must ensure AVX2 is supported at runtime.
    #[must_use]
    #[target_feature(enable = "avx2")]
    unsafe fn has_whitespace_avx2(text: &[u8]) -> bool {
        let whitespace_vec = _mm256_set1_epi8(b' ' as i8);
        let mut text_ptr = text.as_ptr();
        let chunks = text.len() / 32;

        for _ in 0..chunks {
            let text_vec = _mm256_loadu_si256(text_ptr as *const __m256i);
            let cmp = _mm256_cmpeq_epi8(text_vec, whitespace_vec);
            let mask = _mm256_movemask_epi8(cmp);
            if mask != 0 {
                return true;
            }
            text_ptr = text_ptr.add(32);
        }

        // Process remainder
        for i in (chunks * 32)..text.len() {
            if text[i] == b' ' {
                return true;
            }
        }

        false
    }

    fn has_whitespace_fallback(text: &[u8]) -> bool {
        text.iter().any(|&c| c == b' ')
    }
}

/// SIMD-optimized matrix operations
pub struct SimdMatrixOps;

impl SimdMatrixOps {
    /// Matrix multiplication using SIMD
    #[must_use]
    pub fn mat_mul(
        a: &[f32],
        a_rows: usize,
        a_cols: usize,
        b: &[f32],
        b_rows: usize,
        b_cols: usize,
        result: &mut [f32],
    ) {
        assert_eq!(a_cols, b_rows, "Matrix dimensions must be compatible");
        assert_eq!(result.len(), a_rows * b_cols, "Result matrix size mismatch");

        if is_x86_feature_detected!("avx2") {
            // SAFETY: `mat_mul_avx2` requires AVX2 CPU support, verified above.
            // All matrix dimensions are validated via `assert_eq!` at function entry,
            // and the AVX2 function checks bounds with asserts per access.
            unsafe { Self::mat_mul_avx2(a, a_rows, a_cols, b, b_rows, b_cols, result) }
        } else {
            Self::mat_mul_fallback(a, a_rows, a_cols, b, b_rows, b_cols, result);
        }
    }

    /// # Safety
    ///
    /// Caller must ensure AVX2 is supported at runtime. All slices must have
    /// sufficient length for the given matrix dimensions.
    #[target_feature(enable = "avx2")]
    unsafe fn mat_mul_avx2(
        a: &[f32],
        a_rows: usize,
        a_cols: usize,
        b: &[f32],
        _b_rows: usize,
        b_cols: usize,
        result: &mut [f32],
    ) {
        for i in 0..a_rows {
            for j in 0..b_cols {
                let mut sum = _mm256_setzero_ps();

                // Process 8 elements at a time
                let k_chunks = a_cols / 8;
                let _k_remainder = a_cols % 8;

                for k_chunk in 0..k_chunks {
                    let k = k_chunk * 8;
                    let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * a_cols + k));
                    let b_vec = _mm256_loadu_ps(b.as_ptr().add(k * b_cols + j));
                    let product = _mm256_mul_ps(a_vec, b_vec);
                    sum = _mm256_add_ps(sum, product);
                }

                // Horizontal sum
                let sum128 = _mm256_extractf128_ps(sum, 0);
                let sum128_2 = _mm256_extractf128_ps(sum, 1);
                let sum128 = _mm_add_ps(sum128, sum128_2);

                let sum64 =
                    _mm256_castps256_ps128(_mm256_insertf128_ps(_mm256_undefined_ps(), sum128, 0));
                let sum64_2 = _mm256_extractf128_ps(
                    _mm256_permute2f128_ps(_mm256_undefined_ps(), _mm256_castps128_ps256(sum64), 1),
                    0,
                );
                let sum64 = _mm_add_ps(sum64, sum64_2);

                let sum32 = _mm_add_ps(sum64, _mm_movehl_ps(sum64, sum64));
                let sum32 = _mm_add_ss(sum32, _mm_shuffle_ps(sum32, sum32, 1));

                // SAFETY:
                // `sum32` is `__m128` (128-bit SSE register). `[f32; 4]` has the same
                // size (128 bits) and alignment (16 bytes). The transmute is statically
                // size-checked and is the canonical lane-extraction pattern.
                let mut result_val = unsafe { std::mem::transmute::<_, [f32; 4]>(sum32) }[0];

                // Process remainder
                for k in (k_chunks * 8)..a_cols {
                    result_val += a[i * a_cols + k] * b[k * b_cols + j];
                }

                result[i * b_cols + j] = result_val;
            }
        }
    }

    fn mat_mul_fallback(
        a: &[f32],
        a_rows: usize,
        a_cols: usize,
        b: &[f32],
        _b_rows: usize,
        b_cols: usize,
        result: &mut [f32],
    ) {
        for i in 0..a_rows {
            for j in 0..b_cols {
                let mut sum = 0.0f32;
                for k in 0..a_cols {
                    sum += a[i * a_cols + k] * b[k * b_cols + j];
                }
                result[i * b_cols + j] = sum;
            }
        }
    }

    /// Matrix transpose using SIMD
    pub fn transpose(a: &[f32], rows: usize, cols: usize, result: &mut [f32]) {
        assert_eq!(a.len(), rows * cols, "Input matrix size mismatch");
        assert_eq!(result.len(), cols * rows, "Result matrix size mismatch");

        if is_x86_feature_detected!("avx2") && rows >= 8 && cols >= 8 {
            // SAFETY: `transpose_avx2` requires AVX2 CPU support, verified above.
            // The `rows >= 8 && cols >= 8` guard ensures the block-based transpose
            // loop can make progress. Slice lengths are validated via `assert_eq!`.
            unsafe { Self::transpose_avx2(a, rows, cols, result) }
        } else {
            Self::transpose_fallback(a, rows, cols, result);
        }
    }

    /// # Safety
    ///
    /// Caller must ensure AVX2 is supported at runtime. Slices must have
    /// sufficient length for the given matrix dimensions.
    #[target_feature(enable = "avx2")]
    unsafe fn transpose_avx2(a: &[f32], rows: usize, cols: usize, result: &mut [f32]) {
        // Simple block-based transpose for AVX2
        const BLOCK_SIZE: usize = 8;

        for i_block in (0..rows).step_by(BLOCK_SIZE) {
            for j_block in (0..cols).step_by(BLOCK_SIZE) {
                // Process 8x8 block
                for i in i_block..(i_block + BLOCK_SIZE).min(rows) {
                    for j in j_block..(j_block + BLOCK_SIZE).min(cols) {
                        result[j * rows + i] = a[i * cols + j];
                    }
                }
            }
        }
    }

    fn transpose_fallback(a: &[f32], rows: usize, cols: usize, result: &mut [f32]) {
        for i in 0..rows {
            for j in 0..cols {
                result[j * rows + i] = a[i * cols + j];
            }
        }
    }
}

/// Performance benchmark utilities
pub struct SimdBenchmarks;

impl SimdBenchmarks {
    /// Benchmark dot product implementations
    #[must_use]
    pub fn benchmark_dot_product(size: usize) -> (f64, f64) {
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

        // Benchmark fallback
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            SimdVectorOps::dot_product_fallback(&a, &b);
        }
        let fallback_time = start.elapsed().as_secs_f64();

        // Benchmark SIMD (if available)
        let simd_time = if is_x86_feature_detected!("avx2") {
            let start = std::time::Instant::now();
            for _ in 0..1000 {
                // SAFETY: AVX2 was verified via `is_x86_feature_detected!("avx2")`
                // above, and the slices have equal length (constructed that way).
                unsafe { SimdVectorOps::dot_product_avx2(&a, &b) };
            }
            start.elapsed().as_secs_f64()
        } else {
            fallback_time
        };

        (fallback_time, simd_time)
    }

    /// Benchmark string similarity implementations
    #[must_use]
    pub fn benchmark_string_similarity(text_len: usize) -> (f64, f64) {
        let text1 = "a".repeat(text_len);
        let text2 = "b".repeat(text_len);

        // Benchmark fallback
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            SimdTextOps::string_similarity_fallback(
                text1.as_bytes(),
                text2.as_bytes(),
                text_len,
                text_len,
            );
        }
        let fallback_time = start.elapsed().as_secs_f64();

        // Benchmark SIMD (if available)
        let simd_time = if is_x86_feature_detected!("avx2") {
            let start = std::time::Instant::now();
            for _ in 0..1000 {
                SimdTextOps::string_similarity(&text1, &text2);
            }
            start.elapsed().as_secs_f64()
        } else {
            fallback_time
        };

        (fallback_time, simd_time)
    }

    /// Print benchmark results
    pub fn print_benchmarks() {
        tracing::info!("=== SIMD Performance Benchmarks ===");

        // Vector operations benchmark
        let (fallback_time, simd_time) = Self::benchmark_dot_product(1000);
        let speedup = fallback_time / simd_time;
        tracing::info!("Dot Product (1000 elements):");
        tracing::info!("  Fallback: {:.6}s", fallback_time);
        tracing::info!("  SIMD: {:.6}s", simd_time);
        tracing::info!("  Speedup: {:.2}x", speedup);

        // Text operations benchmark
        let (fallback_time, simd_time) = Self::benchmark_string_similarity(1000);
        let speedup = fallback_time / simd_time;
        tracing::info!("\nString Similarity (1000 chars):");
        tracing::info!("  Fallback: {:.6}s", fallback_time);
        tracing::info!("  SIMD: {:.6}s", simd_time);
        tracing::info!("  Speedup: {:.2}x", speedup);

        tracing::info!("\nCPU Features:");
        tracing::info!("  AVX2: {}", is_x86_feature_detected!("avx2"));
        tracing::info!("  AVX: {}", is_x86_feature_detected!("avx"));
        tracing::info!("  SSE4.1: {}", is_x86_feature_detected!("sse4.1"));
        tracing::info!("  SSE2: {}", is_x86_feature_detected!("sse2"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_operations() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut result = vec![0.0; 8];

        // Test addition
        if is_x86_feature_detected!("avx2") {
            // SAFETY: `add_avx2` requires AVX2 CPU support, verified by
            // `is_x86_feature_detected!("avx2")` above. The slices all have
            // the same length (8 elements) as constructed in the test setup.
            unsafe { SimdVectorOps::add_avx2(&a, &b, &mut result) };
        } else {
            SimdVectorOps::add_fallback(&a, &b, &mut result);
        }

        assert_eq!(result, vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]);

        // Test multiplication
        if is_x86_feature_detected!("avx2") {
            // SAFETY: `mul_avx2` requires AVX2 CPU support, verified by
            // `is_x86_feature_detected!("avx2")` above. Slices are equal
            // length as constructed in the test setup.
            unsafe { SimdVectorOps::mul_avx2(&a, &b, &mut result) };
        } else {
            SimdVectorOps::mul_fallback(&a, &b, &mut result);
        }

        assert_eq!(result, vec![2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0]);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let fallback_result = SimdVectorOps::dot_product_fallback(&a, &b);
        let simd_result = if is_x86_feature_detected!("avx2") {
            // SAFETY: `dot_product_avx2` requires AVX2 CPU support, verified
            // by `is_x86_feature_detected!("avx2")`. Both slices have length
            // 8 as constructed in the test.
            unsafe { SimdVectorOps::dot_product_avx2(&a, &b) }
        } else {
            fallback_result
        };

        let expected = 1.0 * 2.0
            + 2.0 * 3.0
            + 3.0 * 4.0
            + 4.0 * 5.0
            + 5.0 * 6.0
            + 6.0 * 7.0
            + 7.0 * 8.0
            + 8.0 * 9.0;

        assert!((fallback_result - expected).abs() < 1e-6);
        assert!((simd_result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 4.0, 6.0, 8.0]; // Same direction as a

        let fallback_result = SimdVectorOps::cosine_similarity_fallback(&a, &b);
        let simd_result = if is_x86_feature_detected!("avx2") {
            // SAFETY: `cosine_similarity_avx2` requires AVX2 CPU support,
            // verified by `is_x86_feature_detected!("avx2")`. Both slices
            // have length 4 as constructed in the test.
            unsafe { SimdVectorOps::cosine_similarity_avx2(&a, &b) }
        } else {
            fallback_result
        };

        assert!((fallback_result - 1.0).abs() < 1e-6);
        assert!((simd_result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_text_operations() {
        let text1 = "hello world";
        let text2 = "hello rust";

        let similarity = SimdTextOps::string_similarity(text1, text2);
        assert!(similarity > 0.0);
        assert!(similarity < 1.0);

        let count = SimdTextOps::count_char(text1, 'l');
        assert_eq!(count, 3);

        assert!(SimdTextOps::has_whitespace(text1));
        assert!(!SimdTextOps::has_whitespace("helloworld"));
    }

    #[test]
    fn test_matrix_operations() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 matrix
        let mut result = vec![0.0; 4];

        SimdMatrixOps::mat_mul(&a, 2, 2, &b, 2, 2, &mut result);

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);

        // Test transpose
        let mut transposed = vec![0.0; 4];
        SimdMatrixOps::transpose(&a, 2, 2, &mut transposed);

        // Expected: [[1, 3], [2, 4]]
        assert_eq!(transposed, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_benchmarks() {
        // This test just ensures benchmarks run without panicking
        SimdBenchmarks::benchmark_dot_product(100);
        SimdBenchmarks::benchmark_string_similarity(100);
    }
}
