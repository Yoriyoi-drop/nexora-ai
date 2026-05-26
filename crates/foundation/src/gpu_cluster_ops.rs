use ndarray::ArrayD;

/// Compute full N×N pairwise Euclidean distance matrix using GPU matmul.
///
/// Formula: dist²[i,j] = ||data[i]||² + ||data[j]||² - 2·data[i]·data[j]
pub fn gpu_pairwise_distances(data: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
    let n = data.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let d = data[0].len();

    #[cfg(feature = "gpu")]
    {
        let ctx = nexora_deeplearning::gpu::GpuContext::global().map_err(|e| {
            tracing::warn!("GPU context unavailable: {}", e);
            e.to_string()
        })?;

        let flat: Vec<f32> = data.iter().flat_map(|v| v.iter()).cloned().collect();
        let arr = ArrayD::from_shape_vec(vec![n, d], flat).map_err(|e| e.to_string())?;

        let gpu = nexora_deeplearning::gpu::GpuTensor::from_cpu(&arr).map_err(|e| {
            tracing::warn!("GPU upload failed: {}", e);
            e.to_string()
        })?;

        let gpu_t = ctx.transpose(&gpu).map_err(|e| e.to_string())?;
        let dots = ctx.matmul(&gpu, &gpu_t).map_err(|e| e.to_string())?;

        let dots_cpu = dots.to_cpu().map_err(|e| e.to_string())?;
        let dots_slice = dots_cpu.as_slice().ok_or("dots not contiguous")?;

        let mut result = vec![vec![0.0_f32; n]; n];
        for i in 0..n {
            let norm_i: f32 = data[i].iter().map(|x| x * x).sum();
            for j in 0..n {
                let norm_j: f32 = data[j].iter().map(|x| x * x).sum();
                let d2 = norm_i + norm_j - 2.0 * dots_slice[i * n + j];
                result[i][j] = d2.max(0.0).sqrt();
            }
        }
        Ok(result)
    }

    #[cfg(not(feature = "gpu"))]
    {
        tracing::warn!("gpu feature not enabled, falling back to CPU distance computation");
        let mut result = vec![vec![0.0_f32; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = data[i]
                    .iter()
                    .zip(data[j].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                result[i][j] = d;
                result[j][i] = d;
            }
        }
        Ok(result)
    }
}
