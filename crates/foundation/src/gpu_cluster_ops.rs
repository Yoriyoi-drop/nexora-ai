use ndarray::ArrayD;

/// Compute full N×N pairwise Euclidean distance matrix using GPU matmul.
///
/// Formula: dist²[i,j] = ||data[i]||² + ||data[j]||² - 2·data[i]·data[j]
pub fn gpu_pairwise_distances(data: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
    let ctx = nexora_deeplearning::gpu::GpuContext::global()
        .map_err(|e| e.to_string())?;

    let n = data.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let d = data[0].len();

    // Flatten data to contiguous ArrayD<f32> [N, D]
    let flat: Vec<f32> = data.iter().flat_map(|v| v.iter()).cloned().collect();
    let arr = ArrayD::from_shape_vec(vec![n, d], flat)
        .map_err(|e| e.to_string())?;

    // Upload to GPU
    let gpu = nexora_deeplearning::gpu::GpuTensor::from_cpu(&arr)
        .map_err(|e| e.to_string())?;

    // data @ data^T = [N, D] @ [D, N] = [N, N] (dot products)
    let gpu_t = ctx.transpose(&gpu).map_err(|e| e.to_string())?;
    let dots = ctx.matmul(&gpu, &gpu_t).map_err(|e| e.to_string())?;

    // Compute squared norms on CPU (O(N·D) — negligible compared to O(N²·D) matmul)
    let norms: Vec<f32> = data
        .iter()
        .map(|v| v.iter().map(|x| x * x).sum::<f32>())
        .collect();

    // Download dot product matrix
    let dots_cpu = dots.to_cpu();
    let dots_slice = dots_cpu.as_slice().ok_or("dots not contiguous")?;

    // Build distance matrix: dist²[i,j] = norms[i] + norms[j] - 2·dots[i,j]
    let mut result = vec![vec![0.0_f32; n]; n];
    for i in 0..n {
        for j in 0..n {
            let d2 = norms[i] + norms[j] - 2.0 * dots_slice[i * n + j];
            result[i][j] = d2.max(0.0).sqrt();
        }
    }

    Ok(result)
}
