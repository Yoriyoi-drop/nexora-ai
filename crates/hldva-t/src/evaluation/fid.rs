//! FID (Fréchet Inception Distance) metric implementation

use crate::types::*;
use nexora_atqs::Tensor;

/// FID Metric implementation
pub struct FIDMetric {
    _config: FIDConfig,
}

impl FIDMetric {
    /// Create new FID metric
    pub fn new() -> HLDVAResult<Self> {
        Ok(Self {
            _config: FIDConfig::default(),
        })
    }

    /// Calculate FID score between two sets of images
    pub fn calculate_fid(
        &self,
        real_images: &[Tensor],
        generated_images: &[Tensor],
    ) -> HLDVAResult<f32> {
        if real_images.is_empty() || generated_images.is_empty() {
            return Err(HLDVAError::Evaluation(
                "Need at least one real and one generated image".to_string(),
            ));
        }

        // Extract features from both sets
        let real_features = self.extract_inception_features(real_images)?;
        let generated_features = self.extract_inception_features(generated_images)?;

        // Calculate statistics
        let real_stats = self.calculate_statistics(&real_features)?;
        let generated_stats = self.calculate_statistics(&generated_features)?;

        // Calculate FID
        let fid = self.compute_frechet_distance(&real_stats, &generated_stats)?;

        Ok(fid)
    }

    /// Extract multi-scale statistical features as a substitute for Inception-v3.
    ///
    /// Uses: per-channel moments (mean, std, skewness, kurtosis) at 3 spatial scales,
    /// channel cross-correlations, gradient edge histogram, and intensity distribution.
    /// Total feature dimension: ≈ 180 features.
    ///
    /// For canonical FID scores, replace with a pre-trained Inception-v3 model
    /// (e.g. via tch-rs or burn) extracting 2048-dim penultimate layer activations.
    fn extract_inception_features(&self, images: &[Tensor]) -> HLDVAResult<Vec<Vec<f32>>> {
        let mut all_features = Vec::new();

        for image in images {
            let image_data = image.data();
            let features = self.pseudo_feature_extraction(image_data)?;
            all_features.push(features);
        }

        if all_features.is_empty() {
            return Err(HLDVAError::Evaluation(
                "No features could be extracted from images (FID requires non-empty input)".to_string(),
            ));
        }

        Ok(all_features)
    }

    /// Multi-scale statistical feature extraction.
    ///
    /// Computes features at 3 spatial scales (full, half, quarter resolution via
    /// block averaging), each yielding per-channel moments + cross-channel correlations,
    /// plus a gradient edge histogram and intensity distribution.
    fn pseudo_feature_extraction(&self, image_data: &[f32]) -> HLDVAResult<Vec<f32>> {
        let n = image_data.len();
        if n == 0 {
            return Err(HLDVAError::Evaluation("Empty image data".to_string()));
        }

        let channels = 3usize;
        let pixels_per_channel = n / channels;

        let mut features = Vec::with_capacity(180);

        // Multi-scale moments: full, half, quarter resolution
        for &scale in &[1usize, 2usize, 4usize] {
            let block = scale * scale;
            let reduced_len = pixels_per_channel / block;
            if reduced_len == 0 {
                continue;
            }

            for c in 0..channels {
                let start = c * pixels_per_channel;
                let end = if c == channels - 1 { n } else { (c + 1) * pixels_per_channel };
                let channel_data = &image_data[start..end];

                // Block-average to get reduced resolution
                let reduced: Vec<f32> = (0..reduced_len)
                    .map(|i| {
                        let base = i * block;
                        let sum: f32 = (0..block).map(|j| channel_data.get(base + j).copied().unwrap_or(0.0)).sum();
                        sum / block as f32
                    })
                    .collect();

                let len = reduced.len() as f32;
                let mean: f32 = reduced.iter().sum::<f32>() / len;
                let variance: f32 = reduced.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / len;
                let std = variance.sqrt();
                features.push(mean);
                features.push(std);

                let skewness = if variance > 0.0 {
                    reduced.iter().map(|&x| (x - mean).powi(3)).sum::<f32>() / (len * variance * std)
                } else {
                    0.0
                };
                features.push(skewness);

                let kurtosis = if variance > 0.0 {
                    reduced.iter().map(|&x| (x - mean).powi(4)).sum::<f32>() / (len * variance * variance) - 3.0
                } else {
                    0.0
                };
                features.push(kurtosis);

                // 5th percentile and 95th percentile for distribution range
                let mut sorted = reduced.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                features.push(sorted[(reduced.len() as f32 * 0.05) as usize].max(0.0));
                features.push(sorted[(reduced.len() as f32 * 0.95) as usize].min(1.0));
            }
        }

        // Channel cross-correlations (at full resolution)
        for c1 in 0..channels {
            let start1 = c1 * pixels_per_channel;
            let end1 = if c1 == channels - 1 { n } else { (c1 + 1) * pixels_per_channel };
            let ch1 = &image_data[start1..end1];
            let len1 = ch1.len();
            let mean1: f32 = ch1.iter().sum::<f32>() / len1 as f32;

            for c2 in (c1 + 1)..channels {
                let start2 = c2 * pixels_per_channel;
                let end2 = if c2 == channels - 1 { n } else { (c2 + 1) * pixels_per_channel };
                let ch2 = &image_data[start2..end2];
                let len2 = ch2.len();
                let mean2: f32 = ch2.iter().sum::<f32>() / len2 as f32;

                let min_len = len1.min(len2);
                let cov: f32 = (0..min_len).map(|i| (ch1[i] - mean1) * (ch2[i] - mean2)).sum::<f32>() / min_len as f32;
                let var1: f32 = ch1.iter().map(|&x| (x - mean1).powi(2)).sum::<f32>() / len1 as f32;
                let var2: f32 = ch2.iter().map(|&x| (x - mean2).powi(2)).sum::<f32>() / len2 as f32;
                let corr = if var1 > 0.0 && var2 > 0.0 { cov / (var1.sqrt() * var2.sqrt()) } else { 0.0 };
                features.push(corr);
            }
        }

        // Gradient edge histogram
        let width = (pixels_per_channel as f32).sqrt() as usize;
        if width >= 4 {
            let gh_bins = 8usize;
            let mut gradient_hist = vec![0.0f32; gh_bins];
            let luminance: Vec<f32> = (0..pixels_per_channel)
                .map(|i| {
                    let r = image_data[i].max(0.0);
                    let g = image_data.get(i + pixels_per_channel).copied().unwrap_or(0.0).max(0.0);
                    let b = image_data.get(i + 2 * pixels_per_channel).copied().unwrap_or(0.0).max(0.0);
                    0.299 * r + 0.587 * g + 0.114 * b
                })
                .collect();

            let mut edge_count = 0.0;
            for y in 1..(width - 1) {
                for x in 1..(width - 1) {
                    let gx = luminance[(y + 1) * width + x] - luminance[(y - 1) * width + x];
                    let gy = luminance[y * width + (x + 1)] - luminance[y * width + (x - 1)];
                    let mag = (gx * gx + gy * gy).sqrt();
                    if mag > 0.02 {
                        let angle = gy.atan2(gx);
                        let bin = (((angle / std::f32::consts::PI) + 1.0) / 2.0 * gh_bins as f32) as usize % gh_bins;
                        gradient_hist[bin] += mag;
                        edge_count += 1.0;
                    }
                }
            }
            if edge_count > 0.0 {
                for h in gradient_hist.iter_mut() {
                    *h /= edge_count;
                }
            }
            features.extend(gradient_hist);
        }

        // Intensity histogram (16 bins across all channels)
        let histogram_bins = 16;
        let mut histogram = vec![0.0; histogram_bins];
        let sample_count = n.min(4096);
        for &pixel in image_data.iter().take(sample_count) {
            let bin = ((pixel * histogram_bins as f32) as usize).min(histogram_bins - 1);
            histogram[bin] += 1.0;
        }
        let hist_total: f32 = histogram.iter().sum();
        if hist_total > 0.0 {
            for h in histogram.iter_mut() {
                *h /= hist_total;
            }
        }
        features.extend(histogram);

        Ok(features)
    }

    /// Calculate mean and covariance statistics
    fn calculate_statistics(&self, features: &[Vec<f32>]) -> HLDVAResult<FeatureStatistics> {
        if features.is_empty() {
            return Err(HLDVAError::Evaluation("No features provided".to_string()));
        }

        let feature_dim = features[0].len();
        let mut mean = vec![0.0; feature_dim];

        // Calculate mean
        for feature_vec in features {
            for (i, &val) in feature_vec.iter().enumerate() {
                mean[i] += val;
            }
        }

        for val in &mut mean {
            *val /= features.len() as f32;
        }

        // Calculate covariance (simplified - using diagonal covariance)
        let mut covariance = vec![0.0; feature_dim];
        for feature_vec in features {
            for (i, &val) in feature_vec.iter().enumerate() {
                let diff = val - mean[i];
                covariance[i] += diff * diff;
            }
        }

        for val in &mut covariance {
            *val /= (features.len() - 1) as f32;
        }

        Ok(FeatureStatistics { mean, covariance })
    }

    /// Compute Fréchet distance between two distributions
    fn compute_frechet_distance(
        &self,
        stats1: &FeatureStatistics,
        stats2: &FeatureStatistics,
    ) -> HLDVAResult<f32> {
        if stats1.mean.len() != stats2.mean.len() {
            return Err(HLDVAError::Evaluation(
                "Feature dimensions must match".to_string(),
            ));
        }

        let mut distance_sq = 0.0;

        // Calculate ||μ₁ - μ₂||²
        for (m1, m2) in stats1.mean.iter().zip(stats2.mean.iter()) {
            let diff = m1 - m2;
            distance_sq += diff * diff;
        }

        // Add trace terms (simplified for diagonal covariance)
        for (c1, c2) in stats1.covariance.iter().zip(stats2.covariance.iter()) {
            distance_sq += c1 + c2 - 2.0 * (c1 * c2).sqrt();
        }

        Ok(distance_sq.sqrt())
    }
}

/// Feature statistics
#[derive(Debug, Clone)]
pub struct FeatureStatistics {
    pub mean: Vec<f32>,
    pub covariance: Vec<f32>,
}

/// FID configuration
#[derive(Debug, Clone)]
pub struct FIDConfig {
    pub feature_dim: usize,
    pub batch_size: usize,
}

impl Default for FIDConfig {
    fn default() -> Self {
        Self {
            feature_dim: 2048,
            batch_size: 50,
        }
    }
}

/// FID Metric trait implementation
impl super::Metric for FIDMetric {
    fn calculate(&self, inputs: &[&Tensor]) -> HLDVAResult<f32> {
        if inputs.len() < 2 {
            return Err(HLDVAError::Evaluation(
                "Need at least 2 inputs for FID".to_string(),
            ));
        }

        // Split inputs into real and generated
        let mid = inputs.len() / 2;
        let real_images: Vec<Tensor> = inputs[..mid].iter().map(|&t| t.clone()).collect();
        let generated_images: Vec<Tensor> = inputs[mid..].iter().map(|&t| t.clone()).collect();

        self.calculate_fid(&real_images, &generated_images)
    }

    fn name(&self) -> &str {
        "fid"
    }
}
