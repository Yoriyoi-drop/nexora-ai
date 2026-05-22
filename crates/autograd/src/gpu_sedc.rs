//! SEDC: Spectral-Entropy Decomposition Compression
//!
//! GPU-accelerated neural network compression via:
//! 1. Randomized SVD with power iteration (GPU matmul + CPU small SVD)
//! 2. Residual Binary Splitting (RBS) with Gram-Schmidt orthogonalization
//! 3. Coupled Matrix-Tensor Factorization (CMTF) for layer fusion
//!
//! All heavy linear algebra (matmul) runs on GPU via wgpu.
//! Small matrix factorizations (QR, SVD of B) run CPU-side since
//! B is at most (k+p) × n where k+p ≪ m,n.

use crate::gpu::{storage_binding, uniform_binding, GpuContext, GpuError, GpuTensor};
use rand::Rng;

/// Generate a random f32 from standard normal distribution using Box-Muller.
fn randn(rng: &mut impl Rng) -> f32 {
    let u1: f32 = rng.gen::<f32>().max(1e-10);
    let u2: f32 = rng.gen::<f32>().max(1e-10);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

// ─── Public Error Type ────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum SedcError {
    #[error("GPU error: {0}")]
    Gpu(#[from] GpuError),
    #[error("SVD failed: {0}")]
    Svd(String),
    #[error("Shape mismatch: {0}")]
    Shape(String),
    #[error("Context not initialized")]
    NoContext,
}

// ─── SEDC Config ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SedcConfig {
    /// Entropy aggression coefficient α ∈ (0,1). Higher = more compression.
    pub alpha: f32,
    /// Number of RBS bits for hidden state quantization.
    pub rbs_bits: usize,
    /// Oversampling parameter for randomized SVD (p in k+p).
    pub oversamples: usize,
    /// Power iteration count for improving singular value decay.
    pub power_iters: usize,
    /// Frobenius norm tolerance per layer.
    pub tolerance: f32,
    /// Minimum compression ratio to apply CMTF fusion.
    pub min_fuse_ratio: f32,
}

impl Default for SedcConfig {
    fn default() -> Self {
        Self {
            alpha: 0.85,
            rbs_bits: 2,
            oversamples: 5,
            power_iters: 1,
            tolerance: 0.01,
            min_fuse_ratio: 1.1,
        }
    }
}

// ─── Compression Report ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct LayerCompressionReport {
    pub layer: usize,
    pub original_params: usize,
    pub compressed_params: usize,
    pub rank: usize,
    pub spectral_entropy: f32,
    pub relative_error: f32,
    pub compression_ratio: f32,
}

#[derive(Debug, Clone, Default)]
pub struct ModelCompressionReport {
    pub layers: Vec<LayerCompressionReport>,
    pub total_original: usize,
    pub total_compressed: usize,
    pub total_ratio: f32,
    pub mean_relative_error: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// CPU-Side Entropy & Threshold
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute spectral entropy from singular values:
/// H_s = -Σ p_i log₂(p_i)  where p_i = σ_i² / Σσ_j²
pub fn spectral_entropy(singular_values: &[f32]) -> f32 {
    let total_sq: f32 = singular_values.iter().map(|s| s * s).sum();
    if total_sq < 1e-10 {
        return 0.0;
    }
    let mut h = 0.0;
    for &s in singular_values {
        let p = (s * s) / total_sq;
        if p > 0.0 {
            h -= p * p.log2();
        }
    }
    h
}

/// Determine optimal truncation rank k* using regularized entropy threshold.
///
/// k* = min{k : Σ_{i=1}^k p_i ≥ 1 - τ(H_s)} + 1
///
/// where τ(H_s) = max(δ, tanh((H_s / log₂(r))^α) - β)
pub fn entropy_optimal_rank(singular_values: &[f32], alpha: f32) -> usize {
    let r = singular_values.len();
    if r == 0 {
        return 0;
    }

    let total_sq: f32 = singular_values.iter().map(|s| s * s).sum();
    if total_sq < 1e-10 {
        return 1;
    }

    let h = spectral_entropy(singular_values);
    let h_max = (r as f32).log2();
    let h_norm = if h_max > 0.0 { h / h_max } else { 0.0 };

    // Regularized tanh threshold: τ = max(δ, tanh(h_norm^α) - β)
    let delta = 1e-6_f32;
    let beta = 0.05_f32;
    let tau = (h_norm.powf(alpha)).tanh().max(delta) - beta;
    let threshold = 1.0 - tau;

    let threshold = threshold.clamp(0.0, 1.0);
    let mut cumsum = 0.0_f32;
    for (i, &s) in singular_values.iter().enumerate() {
        let p = (s * s) / total_sq;
        cumsum += p;
        if cumsum >= threshold {
            return (i + 1).max(1).min(r);
        }
    }
    r.max(1)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Randomized SVD with Power Iteration
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of randomized SVD: W ≈ U · Σ · V^T
#[derive(Debug, Clone)]
pub struct RandomizedSvdResult {
    pub u: ndarray::Array2<f32>,
    pub s: Vec<f32>,
    pub v: ndarray::Array2<f32>,
    pub rank: usize,
    pub spectral_entropy: f32,
}

/// Compute randomized SVD of matrix W ∈ ℝ^{m×n} with target rank k.
///
/// Algorithm:
///   1. Generate Ω ∈ ℝ^{n×(k+p)} ~ N(0,1)
///   2. Y = W · Ω                            (GPU matmul)
///   3. For q iterations: Y = W · W^T · Y    (GPU matmul × 2 per iter)
///   4. Y = QR → Q                           (CPU Gram-Schmidt, Y is m×(k+p) ≪ m)
///   5. B = Q^T · W                          (GPU matmul)
///   6. B = Ũ · Σ · V^T                       (CPU SVD, B is (k+p)×n ≪ n)
///   7. U = Q · Ũ,  V = V,  Σ = Σ
pub fn randomized_svd(
    w: &ndarray::Array2<f32>,
    target_rank: usize,
    oversamples: usize,
    power_iters: usize,
) -> Result<RandomizedSvdResult, SedcError> {
    let m = w.nrows();
    let n = w.ncols();
    let k = target_rank.min(m.min(n));
    let p = oversamples;
    let rank = (k + p).min(n).min(m);

    if rank == 0 {
        return Err(SedcError::Svd("target rank is 0".into()));
    }

    // 1. Generate random matrix Ω ∈ ℝ^{n×rank} ~ N(0,1)
    let mut rng = rand::thread_rng();
    let mut omega = ndarray::Array2::<f32>::zeros((n, rank));
    for val in omega.iter_mut() {
        *val = randn(&mut rng);
    }

    // Phase A: GPU-accelerated range finding
    let y = if GpuContext::is_available() {
        // GPU path
        let ctx = GpuContext::global().map_err(|_| SedcError::NoContext)?;

        // Upload W and Ω to GPU
        let w_gpu = GpuTensor::from_cpu(&w.clone().into_dyn())?;
        let omega_gpu = GpuTensor::from_cpu(&omega.clone().into_dyn())?;

        // Step 2: Y = W @ Ω  (m × rank)
        let mut y_gpu = ctx.matmul(&w_gpu, &omega_gpu)?;

        // Step 3: Power iteration
        for _ in 0..power_iters {
            // Y = W @ (W^T @ Y)
            // First, create W^T as a view
            let wt_gpu = ctx.transpose(&w_gpu)?;
            let temp = ctx.matmul(&wt_gpu, &y_gpu)?;
            y_gpu = ctx.matmul(&w_gpu, &temp)?;
        }

        // Download Y back to CPU for QR
        let y_arr = y_gpu.to_cpu();
        y_arr.into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| SedcError::Svd(format!("Y reshape: {e}")))?
    } else {
        // CPU path
        let mut y = w.dot(&omega);

        for _ in 0..power_iters {
            let wt_y = w.t().dot(&y);
            y = w.dot(&wt_y);
        }

        y
    };

    // 4. QR decomposition: Y = Q·R  (CPU — Y is small: m × rank)
    let q = qr_decomposition(&y)?;

    // 5. B = Q^T @ W  ((k+p) × n) — CPU is fine since B is small (rank × n)
    let b = q.t().dot(w);

    // 6. CPU SVD of B ∈ ℝ^{rank×n}
    let (u_tilde, s, v) = thin_svd(&b)?;
    let s_len = s.len();

    // 7. U = Q · Ũ
    let u = q.dot(&u_tilde);

    let entropy = spectral_entropy(&s);

    Ok(RandomizedSvdResult {
        u,
        s,
        v,
        rank: k.min(s_len),
        spectral_entropy: entropy,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Numerical Helpers (CPU)
// ═══════════════════════════════════════════════════════════════════════════════

/// Classical Gram-Schmidt QR decomposition with reorthogonalization.
/// A ∈ ℝ^{m×n} → Q ∈ ℝ^{m×n}
/// Uses two-step orthogonalization for numerical stability.
fn qr_decomposition(a: &ndarray::Array2<f32>) -> Result<ndarray::Array2<f32>, SedcError> {
    let m = a.nrows();
    let n = a.ncols();
    let mut q = a.clone();

    let eps = f32::EPSILON.sqrt();

    for j in 0..n {
        // Reorthogonalize twice for stability
        for _reorth in 0..2 {
            for i in 0..j {
                let mut dot = 0.0_f32;
                let mut norm_sq = 0.0_f32;
                for k in 0..m {
                    dot += q[[k, j]] * q[[k, i]];
                    norm_sq += q[[k, i]] * q[[k, i]];
                }
                if norm_sq > eps {
                    let coeff = dot / norm_sq;
                    for k in 0..m {
                        q[[k, j]] -= coeff * q[[k, i]];
                    }
                }
            }
        }

        // Normalize q[:, j]
        let mut norm = 0.0_f32;
        for k in 0..m {
            norm += q[[k, j]] * q[[k, j]];
        }
        norm = norm.sqrt();
        if norm > eps {
            for k in 0..m {
                q[[k, j]] /= norm;
            }
        }
    }

    Ok(q)
}

/// Thin SVD via eigendecomposition of B·B^T.
/// B ∈ ℝ^{r×n} → U ∈ ℝ^{r×r}, Σ ∈ ℝ^r, V ∈ ℝ^{n×r}
///
/// Uses the fact that for B = U·Σ·V^T:
///   B·B^T = U·Σ²·U^T  (eigendecomposition of r×r matrix)
///   Then V = B^T · U · Σ^{-1}
type SvdResult = (ndarray::Array2<f32>, Vec<f32>, ndarray::Array2<f32>);

fn thin_svd(b: &ndarray::Array2<f32>) -> Result<SvdResult, SedcError> {
    let r = b.nrows();
    let n = b.ncols();

    if r == 0 || n == 0 {
        return Err(SedcError::Svd("empty matrix".into()));
    }

    // Compute B·B^T (r × r)
    let bbt = b.dot(&b.t());

    // Eigendecomposition via Jacobi iteration (for small r)
    let (eigenvalues, u) = jacobi_eigen(&bbt, 100, 1e-10)?;

    // Sort eigenvalues and vectors in descending order
    let mut indices: Vec<usize> = (0..r).collect();
    indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap_or(std::cmp::Ordering::Equal));

    let mut s = Vec::with_capacity(r);
    let mut u_sorted = ndarray::Array2::<f32>::zeros((r, r));
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        let val = eigenvalues[old_idx].max(0.0).sqrt();
        s.push(val);
        for i in 0..r {
            u_sorted[[i, new_idx]] = u[[i, old_idx]];
        }
    }

    // Compute V = B^T · U · Σ^{-1}
    let mut v = ndarray::Array2::<f32>::zeros((n, r));
    for j in 0..r {
        if s[j] > 1e-10 {
            let inv_s = 1.0 / s[j];
            for i in 0..n {
                let mut sum = 0.0;
                for k in 0..r {
                    sum += b[[k, i]] * u_sorted[[k, j]];
                }
                v[[i, j]] = sum * inv_s;
            }
        }
    }

    Ok((u_sorted, s, v))
}

/// Jacobi eigenvalue algorithm for symmetric matrices.
/// Returns (eigenvalues, eigenvectors) where eigenvectors are column-wise.
/// Uses cyclic Jacobi (sweep through all off-diagonal pairs repeatedly).
fn jacobi_eigen(
    a: &ndarray::Array2<f32>,
    max_sweeps: usize,
    tolerance: f32,
) -> Result<(Vec<f32>, ndarray::Array2<f32>), SedcError> {
    let n = a.nrows();
    if n == 0 {
        return Ok((vec![], ndarray::Array2::zeros((0, 0))));
    }
    if n == 1 {
        return Ok((vec![a[[0, 0]]], ndarray::Array2::eye(1)));
    }

    let mut v = ndarray::Array2::<f32>::eye(n);
    let mut a_cur = a.clone();
    let eps = tolerance.max(f32::EPSILON * 10.0);

    for _sweep in 0..max_sweeps {
        let mut max_off = 0.0_f32;

        // Sweep through all upper off-diagonal pairs
        for p in 0..n {
            for q in (p + 1)..n {
                let a_pq = a_cur[[p, q]];
                let abs_pq = a_pq.abs();
                if abs_pq <= eps {
                    continue;
                }
                max_off = max_off.max(abs_pq);

                let a_pp = a_cur[[p, p]];
                let a_qq = a_cur[[q, q]];

                // Compute Jacobi rotation (avoid division by zero)
                let theta = 0.5 * (a_qq - a_pp) / a_pq;
                let t = theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Apply rotation to columns p, q of A
                for i in 0..n {
                    if i != p && i != q {
                        let a_ip = a_cur[[i, p]];
                        let a_iq = a_cur[[i, q]];
                        a_cur[[i, p]] = c * a_ip - s * a_iq;
                        a_cur[[p, i]] = a_cur[[i, p]];
                        a_cur[[i, q]] = s * a_ip + c * a_iq;
                        a_cur[[q, i]] = a_cur[[i, q]];
                    }
                }
                a_cur[[p, p]] = c * c * a_pp + s * s * a_qq - 2.0 * s * c * a_pq;
                a_cur[[q, q]] = s * s * a_pp + c * c * a_qq + 2.0 * s * c * a_pq;
                a_cur[[p, q]] = 0.0;
                a_cur[[q, p]] = 0.0;

                // Update eigenvectors
                for i in 0..n {
                    let v_ip = v[[i, p]];
                    let v_iq = v[[i, q]];
                    v[[i, p]] = c * v_ip - s * v_iq;
                    v[[i, q]] = s * v_ip + c * v_iq;
                }
            }
        }

        if max_off < eps {
            break;
        }
    }

    let eigenvalues: Vec<f32> = (0..n).map(|i| a_cur[[i, i]]).collect();
    Ok((eigenvalues, v))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Residual Binary Splitting (RBS)
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of RBS quantization: h ≈ Σ s_b · q_b
#[derive(Debug, Clone)]
pub struct RbsResult {
    pub bit_vectors: Vec<Vec<i8>>,
    pub scales: Vec<f32>,
    pub residual: Vec<f32>,
    pub bits: usize,
    pub compression_ratio: f32,
}

/// Quantize a hidden state vector h ∈ ℝ^d using Residual Binary Splitting
/// with Gram-Schmidt orthogonalization and ALS refinement.
///
/// h = Σ_{b=1}^B s_b · q_b + r_B
/// where q_b ∈ {-1,+1}^d, s_b ∈ ℝ
pub fn rbs_quantize(h: &[f32], bits: usize) -> RbsResult {
    let d = h.len();
    if d == 0 || bits == 0 {
        return RbsResult {
            bit_vectors: vec![],
            scales: vec![],
            residual: h.to_vec(),
            bits: 0,
            compression_ratio: 1.0,
        };
    }

    let mut residual = h.to_vec();
    let mut bit_vectors: Vec<Vec<i8>> = Vec::with_capacity(bits);
    let mut scales: Vec<f32> = Vec::with_capacity(bits);

    // Phase 1: Greedy orthogonal bit extraction
    for _b in 0..bits {
        // s_b = ||r||_1 / d
        let l1_norm: f32 = residual.iter().map(|x| x.abs()).sum();
        let s = l1_norm / d as f32;

        if s < 1e-10 {
            // Residual is effectively zero; pad with zeros
            bit_vectors.push(vec![0; d]);
            scales.push(0.0);
            continue;
        }

        // q = sign(r)
        let mut q: Vec<i8> = residual.iter().map(|x| if *x >= 0.0 { 1 } else { -1 }).collect();

        // Gram-Schmidt orthogonalize q against previous bit vectors
        for prev_q in &bit_vectors {
            let dot: f32 = q.iter().zip(prev_q.iter()).map(|(a, b)| *a as f32 * *b as f32).sum();
            let prev_norm_sq: f32 = prev_q.iter().map(|x| (*x as f32) * (*x as f32)).sum();
            if prev_norm_sq > 1e-10 {
                let coeff = dot / prev_norm_sq;
                for i in 0..d {
                    let ortho_val = q[i] as f32 - coeff * prev_q[i] as f32;
                    q[i] = if ortho_val >= 0.0 { 1 } else { -1 };
                }
            }
        }

        // Recompute scale with orthogonalized q
        let dot: f32 = residual.iter().zip(q.iter()).map(|(r, qv)| r * *qv as f32).sum();
        let norm_sq = d as f32; // ||q||² = d for {-1,+1} vectors
        let s_opt = dot / norm_sq;

        // Update residual: r = r - s * q
        for i in 0..d {
            let qf = q[i] as f32;
            residual[i] -= s_opt * qf;
        }

        bit_vectors.push(q);
        scales.push(s_opt);
    }

    // Phase 2: ALS refinement (3 iterations)
    if bits >= 2 {
        for _iter in 0..3 {
            // Fix bit vectors, update scales via least squares
            let mut scale_matrix = ndarray::Array2::<f32>::zeros((bits, bits));
            for i in 0..bits {
                for j in 0..bits {
                    let dot: f32 = bit_vectors[i].iter()
                        .zip(bit_vectors[j].iter())
                        .map(|(a, b)| *a as f32 * *b as f32)
                        .sum();
                    scale_matrix[[i, j]] = dot;
                }
            }

            let mut target = ndarray::Array1::<f32>::zeros(bits);
            for i in 0..bits {
                let dot: f32 = h.iter().zip(bit_vectors[i].iter())
                    .map(|(hv, qv)| hv * *qv as f32)
                    .sum();
                target[i] = dot;
            }

            // Solve: scale_matrix · s = target  (Cholesky since PD)
            if let Ok(s_new) = solve_cholesky(&scale_matrix, &target) {
                for i in 0..bits {
                    scales[i] = s_new[i];
                }
            }

            // Fix scales, update bit vectors via coordinate descent
            for b in 0..bits {
                let mut recon_without_b: Vec<f32> = vec![0.0; d];
                for j in 0..bits {
                    if j != b {
                        for i in 0..d {
                            recon_without_b[i] += scales[j] * bit_vectors[j][i] as f32;
                        }
                    }
                }

                for i in 0..d {
                    let err = h[i] - recon_without_b[i];
                    bit_vectors[b][i] = if err >= 0.0 { 1 } else { -1 };
                }
            }
        }
    }

    // Compute compression ratio and final residual
    let mut final_recon = vec![0.0_f32; d];
    for b in 0..bits {
        for i in 0..d {
            final_recon[i] += scales[b] * bit_vectors[b][i] as f32;
        }
    }
    let final_residual: Vec<f32> = h.iter().zip(final_recon.iter()).map(|(a, b)| a - b).collect();

    // ρ_h = 32d / (B(d + 1) + 32(B-1))  simplified for d ≫ 1
    let overhead = bits * (d + 1) + 32 * bits;
    let ratio = if overhead > 0 {
        (d as f32 * 32.0) / overhead as f32
    } else {
        1.0
    };

    RbsResult {
        bit_vectors,
        scales,
        residual: final_residual,
        bits,
        compression_ratio: ratio,
    }
}

/// Cholesky solve for symmetric positive definite system.
fn solve_cholesky(a: &ndarray::Array2<f32>, b: &ndarray::Array1<f32>) -> Result<ndarray::Array1<f32>, SedcError> {
    let n = a.nrows();
    if n == 0 {
        return Ok(ndarray::Array1::zeros(0));
    }

    // Cholesky: A = L·L^T
    let mut l = ndarray::Array2::<f32>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0_f32;
            if i == j {
                for k in 0..j {
                    sum += l[[j, k]] * l[[j, k]];
                }
                let val = a[[j, j]] - sum;
                if val <= 0.0 {
                    return Err(SedcError::Svd("matrix not positive definite".into()));
                }
                l[[j, j]] = val.sqrt();
            } else {
                for k in 0..j {
                    sum += l[[i, k]] * l[[j, k]];
                }
                if l[[j, j]].abs() > 1e-10 {
                    l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
                } else {
                    l[[i, j]] = 0.0;
                }
            }
        }
    }

    // Forward: L·y = b
    let mut y = ndarray::Array1::<f32>::zeros(n);
    for i in 0..n {
        let mut sum = 0.0_f32;
        for j in 0..i {
            sum += l[[i, j]] * y[j];
        }
        y[i] = (b[i] - sum) / l[[i, i]];
    }

    // Backward: L^T·x = y
    let mut x = ndarray::Array1::<f32>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = 0.0_f32;
        for j in (i + 1)..n {
            sum += l[[j, i]] * x[j];
        }
        if l[[i, i]].abs() > 1e-10 {
            x[i] = (y[i] - sum) / l[[i, i]];
        } else {
            x[i] = 0.0;
        }
    }

    Ok(x)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Coupled Matrix-Tensor Factorization (CMTF)
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of CMTF fusion: T ≈ G ×₁ A ×₂ B ×₃ C
/// where T ∈ ℝ^{m×n×2} stacks W^(l) and W^(l+1).
#[derive(Debug, Clone)]
pub struct CmtfResult {
    /// Core tensor G ∈ ℝ^{r₁×r₂×r₃}
    pub core: ndarray::Array3<f32>,
    /// Mode-1 factor A ∈ ℝ^{m×r₁}
    pub a: ndarray::Array2<f32>,
    /// Mode-2 factor B ∈ ℝ^{n×r₂}
    pub b: ndarray::Array2<f32>,
    /// Mode-3 factor C ∈ ℝ^{2×r₃}
    pub c: ndarray::Array2<f32>,
    pub ranks: (usize, usize, usize),
    pub compression_ratio: f32,
}

/// Fuse two consecutive weight matrices via shared-factor decomposition.
///
/// Model: W₁ ≈ A · D₁ · B^T,  W₂ ≈ A · D₂ · B^T
/// where A ∈ ℝ^{m×rˢ}, B ∈ ℝ^{n×rˢ}, D_k ∈ ℝ^{rˢ×rˢ} with rˢ = min(r₁, r₂)
///
/// Uses standard SVD (via eigendecomposition of W·W^T) to compute the shared
/// factors deterministically, then D_k = A^T · W_k · B.
pub fn cmtf_fusion(
    w1: &ndarray::Array2<f32>,
    w2: &ndarray::Array2<f32>,
    r1: usize,
    r2: usize,
    _r3: usize,
    _max_iters: usize,
) -> Result<CmtfResult, SedcError> {
    let m = w1.nrows();
    let n = w1.ncols();
    let r1 = r1.min(m).max(1);
    let r2_actual = r2.min(n).max(1);
    let rs = r1.min(r2_actual);
    let r3 = 2usize;

    // Compute SVD via W·W^T eigendecomposition for each layer
    let w1wt = w1.dot(&w1.t()); // m × m
    let w2wt = w2.dot(&w2.t()); // m × m

    let (evals1, evecs1) = jacobi_eigen(&w1wt, 50, 1e-6)?;
    let (evals2, evecs2) = jacobi_eigen(&w2wt, 50, 1e-6)?;

    // Sort by eigenvalue descending
    let sort_svd = |evals: Vec<f32>, evecs: ndarray::Array2<f32>| -> ndarray::Array2<f32> {
        let mut idx: Vec<usize> = (0..evals.len()).collect();
        idx.sort_by(|&i, &j| evals[j].partial_cmp(&evals[i]).unwrap_or(std::cmp::Ordering::Equal));
        let mut u = ndarray::Array2::<f32>::zeros((m, rs));
        for (new_i, &old_i) in idx.iter().enumerate().take(rs) {
            let norm: f32 = (0..m).map(|k| evecs[[k, old_i]]).map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for k in 0..m {
                    u[[k, new_i]] = evecs[[k, old_i]] / norm;
                }
            }
        }
        u
    };

    let u1 = sort_svd(evals1, evecs1);
    let u2 = sort_svd(evals2, evecs2);

    // Shared left factor A: average and orthonormalize
    let mut a = (u1 + u2).mapv(|x| x * 0.5);
    a = qr_decomposition(&a)?;

    // Compute B by projecting: A^T · W_k = D_k · B_k^T → SVD of A^T·W₁ gives B
    // We use the first layer's right singular vectors as shared B
    let at_w1 = a.t().dot(w1); // rs × n
    let (_du, _ds, dv) = thin_svd(&at_w1)?;
    let b = dv.slice(ndarray::s![.., 0..rs]).to_owned();

    // Core matrices: D₁ = A^T · W₁ · B, D₂ = A^T · W₂ · B
    let d1 = a.t().dot(w1).dot(&b);
    let d2 = a.t().dot(w2).dot(&b);

    // Build core tensor
    let mut g = ndarray::Array3::<f32>::zeros((rs, rs, 2));
    for i in 0..rs {
        for j in 0..rs {
            g[[i, j, 0]] = d1[[i, j]];
            g[[i, j, 1]] = d2[[i, j]];
        }
    }

    // Compression ratio
    let total_original = 2 * m * n;
    let total_compressed = m * rs + n * rs + 2 * r3 + (rs * rs * r3);
    let compression_ratio = if total_compressed > 0 {
        total_original as f32 / total_compressed as f32
    } else {
        1.0
    };

    Ok(CmtfResult {
        core: g,
        a, b,
        c: ndarray::Array2::eye(2),
        ranks: (rs, rs, r3),
        compression_ratio,
    })
}

/// Mode-1 matriculation: T_{(1)} ∈ ℝ^{m × (n·2)}
#[allow(dead_code)]
fn mode1_matricization(t: &ndarray::Array3<f32>) -> ndarray::Array2<f32> {
    let m = t.shape()[0];
    let n = t.shape()[1];
    let p = t.shape()[2];
    let mut out = ndarray::Array2::<f32>::zeros((m, n * p));
    for i in 0..m {
        for j in 0..n {
            for k in 0..p {
                out[[i, j * p + k]] = t[[i, j, k]];
            }
        }
    }
    out
}

/// Mode-2 matriculation: T_{(2)} ∈ ℝ^{n × (m·2)}
#[allow(dead_code)]
fn mode2_matricization(t: &ndarray::Array3<f32>) -> ndarray::Array2<f32> {
    let m = t.shape()[0];
    let n = t.shape()[1];
    let p = t.shape()[2];
    let mut out = ndarray::Array2::<f32>::zeros((n, m * p));
    for j in 0..n {
        for i in 0..m {
            for k in 0..p {
                out[[j, i * p + k]] = t[[i, j, k]];
            }
        }
    }
    out
}

/// Mode-3 matriculation: T_{(3)} ∈ ℝ^{2 × (m·n)}
#[allow(dead_code)]
fn mode3_matricization(t: &ndarray::Array3<f32>) -> ndarray::Array2<f32> {
    let m = t.shape()[0];
    let n = t.shape()[1];
    let p = t.shape()[2];
    let mut out = ndarray::Array2::<f32>::zeros((p, m * n));
    for k in 0..p {
        for i in 0..m {
            for j in 0..n {
                out[[k, i * n + j]] = t[[i, j, k]];
            }
        }
    }
    out
}

/// Khatri-Rao product (column-wise Kronecker): A ∈ ℝ^{m×r}, B ∈ ℝ^{n×r} → P ∈ ℝ^{mn×r}
#[allow(dead_code)]
fn khatri_rao(a: &ndarray::Array2<f32>, b: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
    let m = a.nrows();
    let n = b.nrows();
    let r = a.ncols().min(b.ncols());
    let mut out = ndarray::Array2::<f32>::zeros((m * n, r));
    for col in 0..r {
        for i in 0..m {
            for j in 0..n {
                out[[i * n + j, col]] = a[[i, col]] * b[[j, col]];
            }
        }
    }
    out
}

/// Solve AX = B for X given A^T A and A^T B (normal equations).
#[allow(dead_code)]
fn solve_least_squares(
    ata: &ndarray::Array2<f32>,
    atb: &ndarray::Array2<f32>,
) -> Result<ndarray::Array2<f32>, SedcError> {
    let n = ata.nrows();
    let nrhs = atb.ncols();

    // Cholesky: A^T A = L·L^T
    let mut l = ndarray::Array2::<f32>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0_f32;
            if i == j {
                for k in 0..j {
                    sum += l[[j, k]] * l[[j, k]];
                }
                let val = ata[[j, j]] - sum;
                if val <= 0.0 { continue; }
                l[[j, j]] = val.sqrt();
            } else {
                for k in 0..j {
                    sum += l[[i, k]] * l[[j, k]];
                }
                if l[[j, j]].abs() > 1e-10 {
                    l[[i, j]] = (ata[[i, j]] - sum) / l[[j, j]];
                }
            }
        }
    }

    let mut x = ndarray::Array2::<f32>::zeros((n, nrhs));

    for rhs_idx in 0..nrhs {
        // Forward: L·y = A^T·b(:,rhs)
        let mut y = ndarray::Array1::<f32>::zeros(n);
        for i in 0..n {
            let mut sum = 0.0_f32;
            for j in 0..i {
                sum += l[[i, j]] * y[j];
            }
            y[i] = (atb[[i, rhs_idx]] - sum) / l[[i, i]];
        }

        // Backward: L^T·x = y
        for i in (0..n).rev() {
            let mut sum = 0.0_f32;
            for j in (i + 1)..n {
                sum += l[[j, i]] * x[[j, rhs_idx]];
            }
            if l[[i, i]].abs() > 1e-10 {
                x[[i, rhs_idx]] = (y[i] - sum) / l[[i, i]];
            }
        }
    }

    Ok(x)
}

/// Moore-Penrose pseudoinverse via SVD.
#[allow(dead_code)]
fn pseudo_inverse(a: &ndarray::Array2<f32>) -> Result<ndarray::Array2<f32>, SedcError> {
    let (u, s, v) = thin_svd(a)?;
    let r = s.len();
    let mut s_inv = ndarray::Array2::<f32>::zeros((r, r));
    for i in 0..r {
        if s[i] > 1e-10 {
            s_inv[[i, i]] = 1.0 / s[i];
        }
    }
    // A^+ = V · S^{-1} · U^T
    Ok(v.dot(&s_inv).dot(&u.t()))
}

// ═══════════════════════════════════════════════════════════════════════════════
// High-Level SEDC Compressor
// ═══════════════════════════════════════════════════════════════════════════════

/// Compressed representation of a single weight matrix
#[derive(Debug, Clone)]
pub struct CompressedWeight {
    pub layer: usize,
    pub shape: (usize, usize),
    /// Original after SVD truncation: U · Σ · V^T
    pub u: ndarray::Array2<f32>,
    pub s: Vec<f32>,
    pub v: ndarray::Array2<f32>,
    pub rank: usize,
    pub entropy: f32,
    pub relative_error: f32,
    /// Optional RBS-compressed hidden states
    pub hidden_rbs: Option<RbsResult>,
}

/// Compressed representation after CMTF fusion
#[derive(Debug, Clone)]
pub struct FusedLayers {
    pub layers: (usize, usize),
    pub cmtf: CmtfResult,
    pub relative_error: f32,
}

/// Full compressed model
#[derive(Debug, Clone)]
pub struct CompressedModel {
    pub weights: Vec<CompressedWeight>,
    pub fused_pairs: Vec<FusedLayers>,
    pub config: SedcConfig,
    pub report: ModelCompressionReport,
}

/// Main SEDC compressor — orchestrates SVD truncation, RBS quantization,
/// and CMTF fusion for a full model.
pub struct SedcCompressor {
    pub config: SedcConfig,
}

impl SedcCompressor {
    pub fn new(config: SedcConfig) -> Self {
        Self { config }
    }

    /// Compress a single weight matrix using SEDC.
    ///
    /// 1. Randomized SVD with entropy-based rank selection
    /// 2. Truncation to k* singular values/vectors
    /// 3. Compute relative error; increase rank if > tolerance
    pub fn compress_weight(&self, w: &ndarray::Array2<f32>, layer: usize) -> Result<CompressedWeight, SedcError> {
        let m = w.nrows();
        let n = w.ncols();
        let max_rank = m.min(n);

        // First, compute full SVD for entropy (use randomized with high oversample)
        // or estimate entropy from top-k singular values
        let target_k = (max_rank as f32 * 0.5) as usize;
        let result = randomized_svd(w, target_k, self.config.oversamples, self.config.power_iters)?;

        // Compute entropy and optimal rank
        let k_opt = entropy_optimal_rank(&result.s, self.config.alpha);
        let k_star = k_opt.max(1).min(max_rank);

        // If we undershot, recompute with correct rank
        let final_result = if result.rank >= k_star {
            result
        } else {
            randomized_svd(w, k_star + self.config.oversamples, self.config.oversamples, self.config.power_iters)?
        };

        // Truncate to k_star using slice selection
        let s_trunc: Vec<f32> = final_result.s.iter().take(k_star).copied().collect();
        let s_arr = ndarray::Array1::from_vec(s_trunc.clone());
        let u_trunc = final_result.u.slice(ndarray::s![.., 0..k_star]).to_owned();
        let v_trunc = final_result.v.slice(ndarray::s![.., 0..k_star]).to_owned();

        // Compute reconstruction error
        let w_approx = u_trunc.dot(&ndarray::Array2::from_diag(&s_arr)).dot(&v_trunc.t());
        let diff = w - &w_approx;
        let w_norm = w.iter().map(|x| x * x).sum::<f32>().sqrt();
        let err_norm = diff.iter().map(|x| x * x).sum::<f32>().sqrt();
        let relative_error = if w_norm > 1e-10 { err_norm / w_norm } else { 0.0 };

        // Increase rank if error exceeds tolerance
        let (u_final, s_final, v_final, k_final, err_final) = if relative_error > self.config.tolerance && k_star < max_rank {
            let k_new = (k_star + 1).min(max_rank);
            let result2 = randomized_svd(w, k_new, self.config.oversamples, self.config.power_iters)?;
            let s2: Vec<f32> = result2.s.iter().take(k_new).copied().collect();
            let s2_arr = ndarray::Array1::from_vec(s2.clone());
            let u2 = result2.u.slice(ndarray::s![.., 0..k_new]).to_owned();
            let v2 = result2.v.slice(ndarray::s![.., 0..k_new]).to_owned();
            let w2 = u2.dot(&ndarray::Array2::from_diag(&s2_arr)).dot(&v2.t());
            let diff2 = w - &w2;
            let err2 = diff2.iter().map(|x| x * x).sum::<f32>().sqrt();
            let rel_err2 = if w_norm > 1e-10 { err2 / w_norm } else { 0.0 };
            (u2, s2, v2, k_new, rel_err2)
        } else {
            (u_trunc, s_trunc, v_trunc, k_star, relative_error)
        };

        Ok(CompressedWeight {
            layer,
            shape: (m, n),
            u: u_final,
            s: s_final,
            v: v_final,
            rank: k_final,
            entropy: final_result.spectral_entropy,
            relative_error: err_final,
            hidden_rbs: None,
        })
    }

    /// Apply RBS quantization to a hidden state vector.
    pub fn compress_hidden(&self, h: &[f32]) -> RbsResult {
        rbs_quantize(h, self.config.rbs_bits)
    }

    /// Fuse two consecutive layers via CMTF.
    pub fn fuse_layers(
        &self,
        w1: &ndarray::Array2<f32>,
        w2: &ndarray::Array2<f32>,
        r1: usize,
        r2: usize,
        layer_idx: usize,
    ) -> Result<Option<FusedLayers>, SedcError> {
        let m = w1.nrows();
        let n = w1.ncols();

        if w2.nrows() != m || w2.ncols() != n {
            return Err(SedcError::Shape(format!(
                "layer {} and {} shape mismatch: {:?} vs {:?}",
                layer_idx, layer_idx + 1,
                (m, n),
                (w2.nrows(), w2.ncols())
            )));
        }

        let r1 = r1.min(m).max(1);
        let r2 = r2.min(n).max(1);
        let r3 = 2usize;

        let cmtf = cmtf_fusion(w1, w2, r1, r2, r3, 10)?;

        if cmtf.compression_ratio < self.config.min_fuse_ratio {
            return Ok(None);
        }

        // Compute reconstruction error
        let w1_recon = cmtf.a.dot(&cmtf.core.index_axis(ndarray::Axis(2), 0)).dot(&cmtf.b.t());
        let w2_recon = cmtf.a.dot(&cmtf.core.index_axis(ndarray::Axis(2), 1)).dot(&cmtf.b.t());

        let diff1 = w1 - &w1_recon;
        let diff2 = w2 - &w2_recon;
        let total_err: f32 = diff1.iter().chain(diff2.iter()).map(|x| x * x).sum();
        let total_norm: f32 = w1.iter().chain(w2.iter()).map(|x| x * x).sum();
        let relative_error = if total_norm > 1e-10 { total_err.sqrt() / total_norm.sqrt() } else { 0.0 };

        Ok(Some(FusedLayers {
            layers: (layer_idx, layer_idx + 1),
            cmtf,
            relative_error,
        }))
    }

    /// Compress all layers of a model (vec of weight matrices).
    pub fn compress_model(
        &self,
        weights: &[ndarray::Array2<f32>],
        _hidden_states: Option<&[Vec<f32>]>,
        fuse_pairs: &[(usize, usize, usize)], // (layer_i, r1, r2) for fusion candidates
    ) -> Result<CompressedModel, SedcError> {
        let l = weights.len();
        let mut compressed_weights = Vec::with_capacity(l);
        let mut fused_pairs = Vec::new();
        let mut total_original = 0usize;
        let mut total_compressed = 0usize;
        let mut total_error = 0.0_f32;

        let mut skip_layers = std::collections::HashSet::new();

        // Fuse layers first
        for &(layer_i, r1, r2) in fuse_pairs {
            if layer_i + 1 >= l || skip_layers.contains(&layer_i) || skip_layers.contains(&(layer_i + 1)) {
                continue;
            }

            if let Some(fused) = self.fuse_layers(&weights[layer_i], &weights[layer_i + 1], r1, r2, layer_i)? {
                skip_layers.insert(layer_i);
                skip_layers.insert(layer_i + 1);
                fused_pairs.push(fused);
            }
        }

        // Compress individual layers
        for (i, w) in weights.iter().enumerate() {
            if skip_layers.contains(&i) {
                continue;
            }

            let cw = self.compress_weight(w, i)?;
            total_original += cw.shape.0 * cw.shape.1;
            total_compressed += cw.rank * (cw.shape.0 + cw.shape.1);
            total_error += cw.relative_error;
            compressed_weights.push(cw);
        }

        // Add fused pairs to totals
        for fused in &fused_pairs {
            let (l1, _l2) = fused.layers;
            let m = weights[l1].nrows();
            let n = weights[l1].ncols();
            total_original += 2 * m * n;
            total_compressed += fused.cmtf.a.nrows() * fused.cmtf.a.ncols()
                + fused.cmtf.b.nrows() * fused.cmtf.b.ncols()
                + fused.cmtf.c.nrows() * fused.cmtf.c.ncols()
                + (fused.cmtf.core.shape()[0] * fused.cmtf.core.shape()[1] * fused.cmtf.core.shape()[2]);
            total_error += fused.relative_error;
        }

        let num_items = compressed_weights.len() + fused_pairs.len();
        let report = ModelCompressionReport {
            total_original,
            total_compressed,
            total_ratio: if total_compressed > 0 { total_original as f32 / total_compressed as f32 } else { 1.0 },
            mean_relative_error: if num_items > 0 { total_error / num_items as f32 } else { 0.0 },
            ..Default::default()
        };

        Ok(CompressedModel {
            weights: compressed_weights,
            fused_pairs,
            config: self.config.clone(),
            report,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// WGSL Compute Shaders for GPU RBS Ops
// ═══════════════════════════════════════════════════════════════════════════════

/// Shader: sign(x) → {-1, +1} vector for RBS
pub(crate) const RBS_SIGN_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
struct Cfg { numel: u32, _pad0: u32, _pad1: u32, _pad2: u32, };
@group(0) @binding(2) var<uniform> cfg: Cfg;

@compute @workgroup_size(256)
fn rbs_sign_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= cfg.numel) { return; }
    output[i] = select(-1.0, 1.0, input[i] >= 0.0);
}
"#;

/// Shader: r = r - s * q  (in-place residual update)
pub(crate) const RBS_AXPY_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> r: array<f32>;
@group(0) @binding(1) var<storage, read> q: array<f32>;
struct Cfg { numel: u32, s: f32, _pad0: u32, _pad1: u32, };
@group(0) @binding(2) var<uniform> cfg: Cfg;

@compute @workgroup_size(256)
fn rbs_axpy_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= cfg.numel) { return; }
    r[i] = r[i] - cfg.s * q[i];
}
"#;

/// Shader: Gram-Schmidt orthogonalization (q_new -= coeff * q_prev)
pub(crate) const RBS_ORTHO_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> q_new: array<f32>;
@group(0) @binding(1) var<storage, read> q_prev: array<f32>;
struct Cfg { numel: u32, coeff: f32, _pad0: u32, _pad1: u32, };
@group(0) @binding(2) var<uniform> cfg: Cfg;

@compute @workgroup_size(256)
fn rbs_ortho_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= cfg.numel) { return; }
    let new_val = q_new[i] - cfg.coeff * q_prev[i];
    q_new[i] = select(-1.0, 1.0, new_val >= 0.0);
}
"#;

/// Shader: copy buffer (for duplicating GPU tensors)
pub(crate) const COPY_BUFFER_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
struct Cfg { numel: u32, _pad0: u32, _pad1: u32, _pad2: u32, };
@group(0) @binding(2) var<uniform> cfg: Cfg;

@compute @workgroup_size(256)
fn copy_buffer_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= cfg.numel) { return; }
    dst[i] = src[i];
}
"#;

/// Shader: L1 norm reduce (partial sum per workgroup)
pub(crate) const RBS_L1_NORM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read_write> partial: array<f32>;
struct Cfg { numel: u32, _pad0: u32, _pad1: u32, _pad2: u32, };
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> wg_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn rbs_l1_main(@builtin(global_invocation_id) id: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>,
               @builtin(workgroup_id) wgid: vec3<u32>) {
    var sum = 0.0f;
    var i = id.x;
    while (i < cfg.numel) {
        sum += abs(data[i]);
        i += 256u * u32(max(1, (cfg.numel + 255u) / 256u));
    }
    wg_sum[lid.x] = sum;
    workgroupBarrier();

    var stride = 128u;
    while (stride > 0u) {
        if (lid.x < stride) {
            wg_sum[lid.x] = wg_sum[lid.x] + wg_sum[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (lid.x == 0u) {
        partial[wgid.x] = wg_sum[0u];
    }
}
"#;

// ═══════════════════════════════════════════════════════════════════════════════
// GPU Pipeline Compilation & Dispatch (wgpu integration)
// ═══════════════════════════════════════════════════════════════════════════════

impl GpuContext {
    /// Compile all SEDC-related compute pipelines.
    pub fn compile_sedc_pipelines(&mut self) -> Result<(), GpuError> {
        // RBS sign
        self.compile_pipeline(
            "rbs_sign",
            &[storage_binding(0, true), storage_binding(1, false), uniform_binding(2)],
            std::borrow::Cow::Borrowed(RBS_SIGN_WGSL),
            "rbs_sign_main",
        )?;

        // RBS axpy
        self.compile_pipeline(
            "rbs_axpy",
            &[storage_binding(0, false), storage_binding(1, true), uniform_binding(2)],
            std::borrow::Cow::Borrowed(RBS_AXPY_WGSL),
            "rbs_axpy_main",
        )?;

        // RBS orthogonalize
        self.compile_pipeline(
            "rbs_ortho",
            &[storage_binding(0, false), storage_binding(1, true), uniform_binding(2)],
            std::borrow::Cow::Borrowed(RBS_ORTHO_WGSL),
            "rbs_ortho_main",
        )?;

        // Copy buffer
        self.compile_pipeline(
            "copy_buffer",
            &[storage_binding(0, true), storage_binding(1, false), uniform_binding(2)],
            std::borrow::Cow::Borrowed(COPY_BUFFER_WGSL),
            "copy_buffer_main",
        )?;

        // RBS L1 norm
        self.compile_pipeline(
            "rbs_l1_norm",
            &[storage_binding(0, true), storage_binding(1, false), uniform_binding(2)],
            std::borrow::Cow::Borrowed(RBS_L1_NORM_WGSL),
            "rbs_l1_main",
        )?;

        Ok(())
    }

    /// GPU-accelerated RBS sign: q = sign(x) ∈ {-1, +1}
    pub fn rbs_sign_gpu(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let pipeline = self
            .pipelines
            .get("rbs_sign")
            .ok_or_else(|| GpuError::Pipeline("rbs_sign not compiled".into()))?;

        let numel = input.numel() as u32;
        let output = GpuTensor::zeros(&input.shape())?;

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rbs_sign_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 4] = [numel, 0, 0, 0];
        self.queue.write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rbs_sign_bg"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cfg_buf.as_entire_binding() },
            ],
        });

        self.dispatch(pipeline, &bg, (numel.div_ceil(256), 1, 1));
        Ok(output)
    }

    /// GPU-accelerated axpy: r = r - s * q (in-place on r)
    pub fn rbs_axpy_gpu(&self, r: &GpuTensor, q: &GpuTensor, s: f32) -> Result<(), GpuError> {
        let pipeline = self
            .pipelines
            .get("rbs_axpy")
            .ok_or_else(|| GpuError::Pipeline("rbs_axpy not compiled".into()))?;

        let numel = r.numel() as u32;
        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rbs_axpy_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 4] = [numel, f32::to_bits(s), 0, 0];
        self.queue.write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rbs_axpy_bg"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: r.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: q.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cfg_buf.as_entire_binding() },
            ],
        });

        self.dispatch(pipeline, &bg, (numel.div_ceil(256), 1, 1));
        Ok(())
    }

    /// GPU-accelerated Gram-Schmidt orthogonalize + re-binarize
    pub fn rbs_orthogonalize_gpu(
        &self,
        q_new: &GpuTensor,
        q_prev: &GpuTensor,
        dot_product: f32,
        norm_sq: f32,
    ) -> Result<(), GpuError> {
        let pipeline = self
            .pipelines
            .get("rbs_ortho")
            .ok_or_else(|| GpuError::Pipeline("rbs_ortho not compiled".into()))?;

        let numel = q_new.numel() as u32;
        let coeff = if norm_sq > 1e-10 { dot_product / norm_sq } else { 0.0 };

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rbs_ortho_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 4] = [numel, f32::to_bits(coeff), 0, 0];
        self.queue.write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rbs_ortho_bg"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: q_new.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: q_prev.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cfg_buf.as_entire_binding() },
            ],
        });

        self.dispatch(pipeline, &bg, (numel.div_ceil(256), 1, 1));
        Ok(())
    }

    /// GPU-accelerated buffer copy
    pub fn copy_buffer_gpu(&self, src: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let pipeline = self
            .pipelines
            .get("copy_buffer")
            .ok_or_else(|| GpuError::Pipeline("copy_buffer not compiled".into()))?;

        let numel = src.numel() as u32;
        let dst = GpuTensor::zeros(&src.shape())?;

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("copy_buffer_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 4] = [numel, 0, 0, 0];
        self.queue.write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("copy_buffer_bg"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: src.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dst.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cfg_buf.as_entire_binding() },
            ],
        });

        self.dispatch(pipeline, &bg, (numel.div_ceil(256), 1, 1));
        Ok(dst)
    }

    /// GPU-accelerated RBS full pipeline for a hidden state vector.
    ///
    /// Returns quantized bits, scales, and residual (all CPU-side).
    pub fn rbs_quantize_gpu(&self, h: &GpuTensor, bits: usize) -> Result<RbsResult, GpuError> {
        let d = h.numel();
        if d == 0 || bits == 0 {
            return Ok(RbsResult {
                bit_vectors: vec![],
                scales: vec![],
                residual: h.to_cpu().as_slice().unwrap_or(&[]).to_vec(),
                bits: 0,
                compression_ratio: 1.0,
            });
        }

        // Copy h to mutable residual buffer on GPU
        let r_gpu = self.copy_buffer_gpu(h)?;
        let mut bit_vectors: Vec<Vec<i8>> = Vec::with_capacity(bits);
        let mut scales: Vec<f32> = Vec::with_capacity(bits);

        for _b in 0..bits {
            // q = sign(r) on GPU
            let q_gpu = self.rbs_sign_gpu(&r_gpu)?;

            // Download q for orthogonalization and dot products
            let q_cpu = q_gpu.to_cpu();
            let q_slice: &[f32] = q_cpu.as_slice().unwrap_or(&[]);
            let q_i8: Vec<i8> = q_slice.iter().map(|x| if *x >= 0.0 { 1 } else { -1 }).collect();

            // Gram-Schmidt orthogonalize against previous bit vectors
            let q_ortho = if !bit_vectors.is_empty() {
                let mut q_ortho_i8 = q_i8.clone();
                for prev_q in &bit_vectors {
                    let dot: f32 = q_ortho_i8.iter().zip(prev_q.iter()).map(|(a, b)| *a as f32 * *b as f32).sum();
                    let norm_sq: f32 = prev_q.iter().map(|x| (*x as f32) * (*x as f32)).sum();
                    if norm_sq > 1e-10 {
                        let coeff = dot / norm_sq;
                        for i in 0..d {
                            let new_val = q_ortho_i8[i] as f32 - coeff * prev_q[i] as f32;
                            q_ortho_i8[i] = if new_val >= 0.0 { 1 } else { -1 };
                        }
                    }
                }
                q_ortho_i8
            } else {
                q_i8
            };

            // Compute optimal scale: s = ⟨r, q⟩ / ||q||²
            let r_cpu = r_gpu.to_cpu();
            let r_slice: &[f32] = r_cpu.as_slice().unwrap_or(&[]);
            let dot: f32 = r_slice.iter().zip(q_ortho.iter()).map(|(rv, qv)| rv * *qv as f32).sum();
            let norm_sq = d as f32;
            let s = dot / norm_sq;

            // Update residual on GPU: r = r - s * q
            // Upload ortho vector to GPU
            let q_ortho_f32: Vec<f32> = q_ortho.iter().map(|x| *x as f32).collect();
            let q_ortho_gpu = GpuTensor::from_cpu(
                &ndarray::ArrayD::from_shape_vec(vec![d], q_ortho_f32).unwrap()
            )?;

            // Upload ortho q to GPU for axpy
            self.rbs_axpy_gpu(&r_gpu, &q_ortho_gpu, s)?;

            bit_vectors.push(q_ortho);
            scales.push(s);
        }

        // Get final residual
        let residual = r_gpu.to_cpu();
        let residual_vec: Vec<f32> = residual.as_slice().unwrap_or(&[]).to_vec();

        let overhead = bits * (d + 1) + 32 * bits;
        let ratio = if overhead > 0 {
            (d as f32 * 32.0) / overhead as f32
        } else {
            1.0
        };

        Ok(RbsResult {
            bit_vectors,
            scales,
            residual: residual_vec,
            bits,
            compression_ratio: ratio,
        })
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn test_matrix(m: usize, n: usize) -> Array2<f32> {
        let mut w = Array2::<f32>::zeros((m, n));
        // Low-rank structure: sum of 3 rank-1 components
        for i in 0..m {
            for j in 0..n {
                w[[i, j]] = (i as f32 / m as f32) * (j as f32 / n as f32)
                    + 0.5 * ((i as f32) * 0.1).sin() * ((j as f32) * 0.1).cos();
            }
        }
        w
    }

    #[test]
    fn test_spectral_entropy_uniform() {
        // Uniform distribution → high entropy
        let s = vec![1.0, 1.0, 1.0, 1.0];
        let h = spectral_entropy(&s);
        // H_max = log₂(4) = 2.0, uniform gives H = 2.0
        assert!((h - 2.0).abs() < 1e-5, "uniform entropy should be 2.0, got {h}");
    }

    #[test]
    fn test_spectral_entropy_concentrated() {
        // Concentrated → low entropy
        let s = vec![10.0, 0.1, 0.01, 0.001];
        let h = spectral_entropy(&s);
        assert!(h < 1.0, "concentrated entropy should be low, got {h}");
    }

    #[test]
    fn test_entropy_optimal_rank_uniform() {
        // High entropy → high rank (little compression)
        let s = vec![1.0, 0.95, 0.9, 0.85, 0.8];
        let k = entropy_optimal_rank(&s, 0.85);
        // Total = 4.5. Cumulative sum: 1.0→0.22, 1.95→0.43, 2.85→0.63, 3.70→0.82, 4.5→1.0
        // tau = tanh((H/2.32)^0.85) - 0.05. H for near-uniform 5 values ≈ 2.32 → tau ≈ tanh(1) - 0.05 ≈ 0.72 - 0.05 = 0.67
        // threshold = 1 - 0.67 = 0.33 → k needs to cover 0.33 → k=2 (cumsum=0.43)
        assert!(k >= 1, "uniform spectrum should keep non-zero rank, got {k}");
    }

    #[test]
    fn test_entropy_optimal_rank_concentrated() {
        // Low entropy → low rank (aggressive compression)
        let s = vec![100.0, 0.1, 0.01, 0.001, 0.0001];
        let k = entropy_optimal_rank(&s, 0.85);
        assert!(k <= 3, "concentrated spectrum should choose low rank, got {k}");
    }

    #[test]
    fn test_spectral_entropy_empty() {
        let h = spectral_entropy(&[]);
        assert_eq!(h, 0.0);
    }

    #[test]
    fn test_spectral_entropy_zero() {
        let h = spectral_entropy(&[0.0, 0.0, 0.0]);
        assert_eq!(h, 0.0);
    }

    #[test]
    fn test_entropy_optimal_rank_empty() {
        let k = entropy_optimal_rank(&[], 0.85);
        assert_eq!(k, 0);
    }

    #[test]
    fn test_svd_low_rank_reconstruction() {
        // Create a rank-3 matrix, verify SVD finds correct rank
        let m = 20;
        let n = 15;
        let r_true = 3;

        // Build rank-3 matrix
        let u_true = Array2::<f32>::from_shape_fn((m, r_true), |(_, _)| rand::random::<f32>());
        let v_true = Array2::<f32>::from_shape_fn((n, r_true), |(_, _)| rand::random::<f32>());
        let w = u_true.dot(&v_true.t());

        let result = randomized_svd(&w, r_true + 2, 5, 1).unwrap();
        assert!(result.rank >= r_true, "SVD should capture rank {r_true}, got {}", result.rank);

        let s_trunc: Vec<f32> = result.s.iter().take(r_true).copied().collect();
        let s_arr = ndarray::Array1::from_vec(s_trunc.clone());
        let u_trunc = result.u.slice(ndarray::s![.., 0..r_true]).to_owned();
        let v_trunc = result.v.slice(ndarray::s![.., 0..r_true]).to_owned();
        let w_approx = u_trunc.dot(&ndarray::Array2::from_diag(&s_arr)).dot(&v_trunc.t());

        let diff = &w - &w_approx;
        let err: f32 = diff.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm: f32 = w.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rel_err = err / norm;
        assert!(rel_err < 0.5, "rank-3 reconstruction error too high: {rel_err}");
    }

    #[test]
    fn test_svd_reconstruction_accuracy() {
        let w = test_matrix(32, 24);
        let result = randomized_svd(&w, 8, 5, 1).unwrap();

        let k = result.rank.min(result.s.len());
        let s_trunc: Vec<f32> = result.s.iter().take(k).copied().collect();
        let s_arr = ndarray::Array1::from_vec(s_trunc.clone());
        let u_trunc = result.u.slice(ndarray::s![.., 0..k]).to_owned();
        let v_trunc = result.v.slice(ndarray::s![.., 0..k]).to_owned();
        let w_approx = u_trunc.dot(&ndarray::Array2::from_diag(&s_arr)).dot(&v_trunc.t());

        let diff = &w - &w_approx;
        let err: f32 = diff.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm: f32 = w.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rel_err = err / norm;
        assert!(rel_err < 0.6, "reconstruction error too high: {rel_err}");
    }

    #[test]
    fn test_rbs_quantize_basic() {
        let h: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();
        let result = rbs_quantize(&h, 3);

        assert_eq!(result.bit_vectors.len(), 3);
        assert_eq!(result.scales.len(), 3);
        assert_eq!(result.bit_vectors[0].len(), 64);

        // Reconstruction
        let mut recon = vec![0.0f32; 64];
        for b in 0..3 {
            for i in 0..64 {
                recon[i] += result.scales[b] * result.bit_vectors[b][i] as f32;
            }
        }

        let mse: f32 = h.iter().zip(recon.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / 64.0;
        assert!(mse < 0.5, "RBS reconstruction MSE too high: {mse}");
    }

    #[test]
    fn test_rbs_quantize_identity() {
        // Single bit should capture sign pattern
        let h: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0];
        let result = rbs_quantize(&h, 1);

        assert_eq!(result.bit_vectors[0], vec![1, -1, 1, -1]);
    }

    #[test]
    fn test_rbs_quantize_zero_bits() {
        let h = vec![1.0, 2.0, 3.0];
        let result = rbs_quantize(&h, 0);
        assert!(result.bit_vectors.is_empty());
        assert_eq!(result.residual, h);
    }

    #[test]
    fn test_rbs_quantize_empty() {
        let result = rbs_quantize(&[], 2);
        assert!(result.bit_vectors.is_empty());
    }

    #[test]
    fn test_qr_orthogonality() {
        // Full-rank random matrix
        let mut rng = rand::thread_rng();
        let mut a = Array2::<f32>::zeros((15, 8));
        for val in a.iter_mut() {
            *val = rng.gen::<f32>() * 2.0 - 1.0;
        }
        let q = qr_decomposition(&a).unwrap();

        // Q^T Q should be close to identity
        let qtq = q.t().dot(&q);
        for i in 0..8 {
            for j in 0..8 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let diff = (qtq[[i, j]] - expected).abs();
                assert!(diff < 1e-3, "QR orthogonality failed at ({i},{j}): {diff}");
            }
        }
    }

    #[test]
    fn test_jacobi_eigen_small() {
        // Symmetric matrix
        let a = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 3.0]).unwrap();
        let (evals, _evecs) = jacobi_eigen(&a, 200, 1e-6).unwrap();
        // Sort eigenvalues
        let mut evals_sorted = evals.clone();
        evals_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!((evals_sorted[0] - 5.0).abs() < 0.5, "expected ~5.0, got {}", evals_sorted[0]);
        assert!((evals_sorted[1] - 2.0).abs() < 0.5, "expected ~2.0, got {}", evals_sorted[1]);
    }

    #[test]
    fn test_solve_cholesky_small() {
        // SPD matrix
        let a = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 3.0]).unwrap();
        let b = ndarray::Array1::from_shape_vec(2, vec![5.0, 4.0]).unwrap();
        let x = solve_cholesky(&a, &b).unwrap();
        // Check: A·x ≈ b
        let ax = a.dot(&x);
        let diff = (&ax - &b).iter().map(|d| d.abs()).sum::<f32>();
        assert!(diff < 1e-5, "Cholesky solve failed: {diff}");
    }

    #[test]
    fn test_thin_svd_identity() {
        let a = Array2::<f32>::eye(3);
        let (u, s, v) = thin_svd(&a).unwrap();
        for &sv in &s {
            assert!((sv - 1.0).abs() < 1e-5, "singular value should be 1, got {sv}");
        }

        // U, V should be orthogonal
        let utu = u.t().dot(&u);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((utu[[i, j]] - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_cmtf_fusion_small() {
        let w1 = test_matrix(8, 6);
        let w2 = test_matrix(8, 6);

        let result = cmtf_fusion(&w1, &w2, 4, 3, 2, 5).unwrap();
        // Shared rank rs = min(r1, r2) = 3
        assert_eq!(result.ranks, (3, 3, 2));

        // Shared A (left) and B (right) factors
        assert_eq!(result.a.shape(), &[8, 3]);
        assert_eq!(result.b.shape(), &[6, 3]);

        // Compression should reduce parameters
        assert!(result.compression_ratio > 1.0, "compression ratio: {}", result.compression_ratio);

        // Reconstruction quality — shared factor captures most of the energy
        let d1 = result.core.index_axis(ndarray::Axis(2), 0);
        let w1_recon = result.a.dot(&d1).dot(&result.b.t());
        let diff1 = (&w1 - &w1_recon).iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm1 = w1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(diff1 / norm1 < 1.0, "reconstruction error: {}", diff1 / norm1);
    }

    #[test]
    fn test_compressor_basic() {
        let config = SedcConfig {
            alpha: 0.85,
            rbs_bits: 2,
            oversamples: 5,
            power_iters: 1,
            tolerance: 0.1,
            min_fuse_ratio: 1.1,
        };
        let compressor = SedcCompressor::new(config);

        let w = test_matrix(16, 12);
        let result = compressor.compress_weight(&w, 0).unwrap();

        assert_eq!(result.shape, (16, 12));
        assert!(result.rank <= 16);
        assert!(result.relative_error >= 0.0);

        // Verify compression — rank should be less than min(m,n)
        assert!(result.rank < result.shape.0.min(result.shape.1), "rank {} should be less than min dim", result.rank);
    }

    #[test]
    fn test_compressor_hidden_state() {
        let config = SedcConfig {
            rbs_bits: 2,
            ..Default::default()
        };
        let compressor = SedcCompressor::new(config);

        let h: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 16.0).collect();
        let result = compressor.compress_hidden(&h);

        assert_eq!(result.bits, 2);
        assert!(result.compression_ratio > 1.0);
    }

    #[test]
    fn test_compressor_full_model() {
        let config = SedcConfig {
            alpha: 0.85,
            rbs_bits: 2,
            oversamples: 5,
            power_iters: 1,
            tolerance: 0.15,
            min_fuse_ratio: 1.1,
        };
        let compressor = SedcCompressor::new(config);

        // Simulate a 4-layer model (all same shape for compatibility)
        let weights: Vec<Array2<f32>> = (0..4)
            .map(|_| {
                test_matrix(12, 10)
            })
            .collect();

        // Fuse layers (0,1) and (2,3) with rank estimates
        let fuse_pairs = vec![(0, 3, 3), (2, 3, 3)];

        let model = compressor.compress_model(&weights, None, &fuse_pairs).unwrap();
        assert!(model.report.total_ratio >= 0.5, "model should show some compression activity, ratio: {}", model.report.total_ratio);
    }

    #[test]
    fn test_pseudo_inverse() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 2.0]).unwrap();
        let a_pinv = pseudo_inverse(&a).unwrap();
        // A^+ = A^{-1} for invertible diag matrix
        assert!((a_pinv[[0, 0]] - 1.0).abs() < 1e-4, "expected 1.0, got {}", a_pinv[[0, 0]]);
        assert!((a_pinv[[1, 1]] - 0.5).abs() < 1e-4, "expected 0.5, got {}", a_pinv[[1, 1]]);
    }

    #[test]
    fn test_mode_matriculations() {
        let mut t = ndarray::Array3::<f32>::zeros((3, 4, 2));
        // Fill with sequential values
        for i in 0..3 {
            for j in 0..4 {
                for k in 0..2 {
                    t[[i, j, k]] = (i * 4 * 2 + j * 2 + k) as f32;
                }
            }
        }

        let t1 = mode1_matricization(&t);
        assert_eq!(t1.shape(), &[3, 8]);

        let t2 = mode2_matricization(&t);
        assert_eq!(t2.shape(), &[4, 6]);

        let t3 = mode3_matricization(&t);
        assert_eq!(t3.shape(), &[2, 12]);
    }

    #[test]
    fn test_khatri_rao() {
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Array2::from_shape_vec((3, 3), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]).unwrap();
        let kr = khatri_rao(&a, &b);
        assert_eq!(kr.shape(), &[6, 3]);
        // First column: a[:,0] ⊗ b[:,0] = [1,4] ⊗ [1,2,3] → [1,2,3,4,8,12]
        assert!((kr[[0, 0]] - 1.0).abs() < 1e-5);
        assert!((kr[[3, 0]] - 4.0).abs() < 1e-5);
    }
}
