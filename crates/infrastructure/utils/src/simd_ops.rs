//! SIMD Operations untuk Performance Optimization
//! 
//! Implementasi SIMD-optimized operations untuk mathematical computations
//! dan text processing

use std::arch::x86_64::*;
use std::mem;
use tracing::debug;

/// SIMD-optimized vector operations
pub struct SimdVectorOps;

impl SimdVectorOps {
    /// Dot product menggunakan SIMD
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
        let sum64_2 = _mm256_extractf128_ps(_mm256_permute2f128_ps(_mm256_undefined_ps(), _mm256_castps128_ps256(sum64), 1), 0);
        let sum64 = _mm_add_ps(sum64, sum64_2);
        
        let sum32 = _mm_add_ps(sum64, _mm_movehl_ps(sum64, sum64));
        let sum32 = _mm_add_ss(sum32, _mm_shuffle_ps(sum32, sum32, 1));
        
        let mut result = unsafe { std::mem::transmute::<_, [f32; 4]>(sum32) }[0];
        
        // Process remainder
        for i in (n - remainder)..n {
            result += a[i] * b[i];
        }
        
        result
    }
    
    /// Vector addition menggunakan SIMD
    #[target_feature(enable = "avx2")]
    pub unsafe fn add_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len(), "Vectors must have same length");
        assert_eq!(a.len(), result.len(), "Result vector must have same length");
        
        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;
        
        for i in 0..chunks {
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
    #[target_feature(enable = "avx2")]
    pub unsafe fn mul_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len(), "Vectors must have same length");
        assert_eq!(a.len(), result.len(), "Result vector must have same length");
        
        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;
        
        for i in 0..chunks {
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
        let sum64_2 = _mm256_extractf128_ps(_mm256_permute2f128_ps(_mm256_undefined_ps(), _mm256_castps128_ps256(sum64), 1), 0);
        let sum64 = _mm_add_ps(sum64, sum64_2);
        
        let sum32 = _mm_add_ps(sum64, _mm_movehl_ps(sum64, sum64));
        let sum32 = _mm_add_ss(sum32, _mm_shuffle_ps(sum32, sum32, 1));
        
        let mut result = unsafe { std::mem::transmute::<_, [f32; 4]>(sum32) }[0];
        
        // Process remainder
        for i in (n - remainder)..n {
            let diff = a[i] - b[i];
            result += diff * diff;
        }
        
        result.sqrt()
    }
    
    /// Fallback implementations for systems without AVX2
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
    
    pub fn euclidean_distance_fallback(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors must have same length");
        
        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }
}

/// SIMD-optimized text operations
pub struct SimdTextOps;

impl SimdTextOps {
    /// Fast string similarity using SIMD
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
            unsafe { Self::string_similarity_avx2(a_bytes, b_bytes, min_len, max_len) }
        } else {
            Self::string_similarity_fallback(a_bytes, b_bytes, min_len, max_len)
        }
    }
    
    #[target_feature(enable = "avx2")]
    unsafe fn string_similarity_avx2(a: &[u8], b: &[u8], min_len: usize, max_len: usize) -> f32 {
        let mut matches = 0u32;
        let chunks = min_len / 32;
        let remainder = min_len % 32;
        
        let mut a_ptr = a.as_ptr();
        let mut b_ptr = b.as_ptr();
        
        for _ in 0..chunks {
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
    
    fn string_similarity_fallback(a: &[u8], b: &[u8], min_len: usize, max_len: usize) -> f32 {
        let matches = a.iter().zip(b.iter())
            .filter(|(x, y)| x == y)
            .count();
        
        matches as f32 / max_len as f32
    }
    
    /// Fast character counting using SIMD
    pub fn count_char(text: &str, target: char) -> usize {
        let target_byte = target as u8;
        let text_bytes = text.as_bytes();
        
        if is_x86_feature_detected!("avx2") && text_bytes.len() >= 32 {
            unsafe { Self::count_char_avx2(text_bytes, target_byte) }
        } else {
            Self::count_char_fallback(text_bytes, target_byte)
        }
    }
    
    #[target_feature(enable = "avx2")]
    unsafe fn count_char_avx2(text: &[u8], target: u8) -> usize {
        let mut count = 0usize;
        let chunks = text.len() / 32;
        let remainder = text.len() % 32;
        
        let target_vec = _mm256_set1_epi8(target as i8);
        let mut text_ptr = text.as_ptr();
        
        for _ in 0..chunks {
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
    pub fn has_whitespace(text: &str) -> bool {
        let text_bytes = text.as_bytes();
        
        if is_x86_feature_detected!("avx2") && text_bytes.len() >= 32 {
            unsafe { Self::has_whitespace_avx2(text_bytes) }
        } else {
            Self::has_whitespace_fallback(text_bytes)
        }
    }
    
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
    pub fn mat_mul(a: &[f32], a_rows: usize, a_cols: usize,
                   b: &[f32], b_rows: usize, b_cols: usize,
                   result: &mut [f32]) {
        assert_eq!(a_cols, b_rows, "Matrix dimensions must be compatible");
        assert_eq!(result.len(), a_rows * b_cols, "Result matrix size mismatch");
        
        if is_x86_feature_detected!("avx2") {
            unsafe { Self::mat_mul_avx2(a, a_rows, a_cols, b, b_rows, b_cols, result) }
        } else {
            Self::mat_mul_fallback(a, a_rows, a_cols, b, b_rows, b_cols, result);
        }
    }
    
    #[target_feature(enable = "avx2")]
    unsafe fn mat_mul_avx2(a: &[f32], a_rows: usize, a_cols: usize,
                         b: &[f32], b_rows: usize, b_cols: usize,
                         result: &mut [f32]) {
        for i in 0..a_rows {
            for j in 0..b_cols {
                let mut sum = _mm256_setzero_ps();
                
                // Process 8 elements at a time
                let k_chunks = a_cols / 8;
                let k_remainder = a_cols % 8;
                
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
                
                let sum64 = _mm256_castps256_ps128(_mm256_insertf128_ps(_mm256_undefined_ps(), sum128, 0));
                let sum64_2 = _mm256_extractf128_ps(_mm256_permute2f128_ps(_mm256_undefined_ps(), _mm256_castps128_ps256(sum64), 1), 0);
                let sum64 = _mm_add_ps(sum64, sum64_2);
                
                let sum32 = _mm_add_ps(sum64, _mm_movehl_ps(sum64, sum64));
                let sum32 = _mm_add_ss(sum32, _mm_shuffle_ps(sum32, sum32, 1));
                
                let mut result_val = unsafe { std::mem::transmute::<_, [f32; 4]>(sum32) }[0];
                
                // Process remainder
                for k in (k_chunks * 8)..a_cols {
                    result_val += a[i * a_cols + k] * b[k * b_cols + j];
                }
                
                result[i * b_cols + j] = result_val;
            }
        }
    }
    
    fn mat_mul_fallback(a: &[f32], a_rows: usize, a_cols: usize,
                        b: &[f32], b_rows: usize, b_cols: usize,
                        result: &mut [f32]) {
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
            unsafe { Self::transpose_avx2(a, rows, cols, result) }
        } else {
            Self::transpose_fallback(a, rows, cols, result);
        }
    }
    
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
                unsafe { SimdVectorOps::dot_product_avx2(&a, &b) };
            }
            start.elapsed().as_secs_f64()
        } else {
            fallback_time
        };
        
        (fallback_time, simd_time)
    }
    
    /// Benchmark string similarity implementations
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
                text_len
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
        println!("=== SIMD Performance Benchmarks ===");
        
        // Vector operations benchmark
        let (fallback_time, simd_time) = Self::benchmark_dot_product(1000);
        let speedup = fallback_time / simd_time;
        println!("Dot Product (1000 elements):");
        println!("  Fallback: {:.6}s", fallback_time);
        println!("  SIMD: {:.6}s", simd_time);
        println!("  Speedup: {:.2}x", speedup);
        
        // Text operations benchmark
        let (fallback_time, simd_time) = Self::benchmark_string_similarity(1000);
        let speedup = fallback_time / simd_time;
        println!("\nString Similarity (1000 chars):");
        println!("  Fallback: {:.6}s", fallback_time);
        println!("  SIMD: {:.6}s", simd_time);
        println!("  Speedup: {:.2}x", speedup);
        
        println!("\nCPU Features:");
        println!("  AVX2: {}", is_x86_feature_detected!("avx2"));
        println!("  AVX: {}", is_x86_feature_detected!("avx"));
        println!("  SSE4.1: {}", is_x86_feature_detected!("sse4.1"));
        println!("  SSE2: {}", is_x86_feature_detected!("sse2"));
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
            unsafe { SimdVectorOps::add_avx2(&a, &b, &mut result) };
        } else {
            SimdVectorOps::add_fallback(&a, &b, &mut result);
        }
        
        assert_eq!(result, vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]);
        
        // Test multiplication
        if is_x86_feature_detected!("avx2") {
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
            unsafe { SimdVectorOps::dot_product_avx2(&a, &b) }
        } else {
            fallback_result
        };
        
        let expected = 1.0*2.0 + 2.0*3.0 + 3.0*4.0 + 4.0*5.0 + 5.0*6.0 + 6.0*7.0 + 7.0*8.0 + 8.0*9.0;
        
        assert!((fallback_result - expected).abs() < 1e-6);
        assert!((simd_result - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 4.0, 6.0, 8.0]; // Same direction as a
        
        let fallback_result = SimdVectorOps::cosine_similarity_fallback(&a, &b);
        let simd_result = if is_x86_feature_detected!("avx2") {
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
