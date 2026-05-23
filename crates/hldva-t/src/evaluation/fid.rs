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

    /// STUB: uses pixel-level statistics as placeholder features.
    /// This is NOT real Inception-v3 feature extraction — FID scores from this
    /// function will NOT match paper-canonical values.
    ///
    /// To implement properly:
    /// 1. Add a pre-trained Inception-v3 model (e.g. via tch-rs or burn)
    /// 2. Load weights from pytorch pretrained
    /// 3. Remove the pooling/classification layers
    /// 4. Extract 2048-dim features from the penultimate layer
    ///
    /// Current implementation uses per-channel mean + standard deviation
    /// plus spatial intensity histogram as pseudo-features. This provides
    /// a rough distribution comparison but is NOT production-quality FID.
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

    /// Pseudo feature extraction using image statistics as a better placeholder.
    /// Uses: per-channel mean, std, skewness, and intensity histogram bins.
    /// Total dimension: 4 (moments) * 3 (RGB) + 16 (histogram) = 28 features.
    /// This is NOT Inception-v3 — replaces the old sin/cos noise features.
    fn pseudo_feature_extraction(&self, image_data: &[f32]) -> HLDVAResult<Vec<f32>> {
        let n = image_data.len();
        if n == 0 {
            return Err(HLDVAError::Evaluation("Empty image data".to_string()));
        }

        // Assume 3-channel (RGB) image, compute per-channel statistics
        let channels = 3usize;
        let pixels_per_channel = n / channels;

        let mut features = Vec::with_capacity(28);

        for c in 0..channels {
            let start = c * pixels_per_channel;
            let end = if c == channels - 1 { n } else { (c + 1) * pixels_per_channel };
            let channel_data = &image_data[start..end];
            let len = channel_data.len() as f32;

            if len == 0.0 {
                features.push(0.0); // mean
                features.push(0.0); // std
                features.push(0.0); // skew
                features.push(0.0); // kurtosis
                continue;
            }

            // Mean
            let mean: f32 = channel_data.iter().sum::<f32>() / len;
            features.push(mean);

            // Variance and std
            let variance: f32 = channel_data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / len;
            let std = variance.sqrt();
            features.push(std);

            // Skewness (third moment)
            let skewness = if variance > 0.0 {
                channel_data.iter().map(|&x| (x - mean).powi(3)).sum::<f32>() / (len * variance * std)
            } else {
                0.0
            };
            features.push(skewness);

            // Kurtosis (fourth moment)
            let kurtosis = if variance > 0.0 {
                channel_data.iter().map(|&x| (x - mean).powi(4)).sum::<f32>() / (len * variance * variance) - 3.0
            } else {
                0.0
            };
            features.push(kurtosis);
        }

        // Intensity histogram (16 bins across all channels)
        let histogram_bins = 16;
        let mut histogram = vec![0.0; histogram_bins];
        for &pixel in image_data.iter().take(256) {
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
