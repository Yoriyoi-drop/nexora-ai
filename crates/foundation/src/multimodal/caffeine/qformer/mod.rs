//! Tri-Query Former implementation for CAFFEINE
//! 
//! Implements Stage 2: Hierarchical Tri-Query Former (dari BLIP-2)
//! - Semantic Query Tokens
//! - Spatial Query Tokens  
//! - Temporal Query Tokens

pub mod tri_query_former;
pub mod attention;
pub mod cross_modal;

pub use tri_query_former::*;
pub use attention::*;
pub use cross_modal::*;

use crate::multimodal::caffeine::types::*;
use crate::multimodal::caffeine::error::Result;
use ndarray::ArrayD;
use tracing::warn;

/// Tri-Query Former main interface
pub struct TriQueryFormer {
    semantic_queries: QuerySet,
    spatial_queries: QuerySet,
    temporal_queries: QuerySet,
    attention_mechanism: CrossModalAttention,
    config: crate::multimodal::caffeine::config::QFormerConfig,
}

impl TriQueryFormer {
    /// Create new Tri-Query Former with validation
    pub fn new(config: crate::multimodal::caffeine::config::QFormerConfig) -> Result<Self> {
        // Validate configuration
        if config.hidden_dim == 0 {
            return Err(crate::multimodal::caffeine::error::CaffeineError::qformer(
                "Hidden dimension must be greater than 0"
            ));
        }
        
        if config.num_attention_heads == 0 {
            return Err(crate::multimodal::caffeine::error::CaffeineError::qformer(
                "Number of attention heads must be greater than 0"
            ));
        }
        
        if config.hidden_dim % config.num_attention_heads != 0 {
            return Err(crate::multimodal::caffeine::error::CaffeineError::qformer(
                "Hidden dimension must be divisible by number of attention heads"
            ));
        }
        
        let semantic_queries = QuerySet::new(
            config.num_semantic_queries,
            config.hidden_dim,
            "semantic".to_string(),
        )?;
        
        let spatial_queries = QuerySet::new(
            config.num_spatial_queries,
            config.hidden_dim,
            "spatial".to_string(),
        )?;
        
        let temporal_queries = QuerySet::new(
            config.num_temporal_queries,
            config.hidden_dim,
            "temporal".to_string(),
        )?;
        
        let attention_mechanism = CrossModalAttention::new(
            config.hidden_dim,
            config.num_attention_heads,
            config.dropout_rate,
        )?;
        
        Ok(Self {
            semantic_queries,
            spatial_queries,
            temporal_queries,
            attention_mechanism,
            config,
        })
    }
    
    /// Transform encoded features through tri-query processing with enhanced validation
    pub fn transform(&mut self, features: &EncodedFeatures) -> Result<QueryFeatures> {
        // Comprehensive input validation
        self.validate_encoded_features(features)?;
        
        // Extract features from different modalities with validation
        let image_features = features.image_features.as_ref();
        let audio_features = features.audio_features.as_ref();
        let video_features = features.video_features.as_ref();
        let text_features = features.text_features.as_ref();
        
        // Validate that at least one modality is present
        let modality_count = [
            image_features.is_some(),
            audio_features.is_some(),
            video_features.is_some(),
            text_features.is_some(),
        ].iter().filter(|&&x| x).count();
        
        if modality_count == 0 {
            return Err(crate::multimodal::caffeine::error::CaffeineError::qformer(
                "At least one modality must be present"
            ));
        }
        
        // Process semantic queries with error handling
        let semantic_features = self.process_semantic_queries_safe(
            image_features, audio_features, video_features, text_features
        )?;
        
        // Process spatial queries (only for modalities that support spatial features)
        let spatial_features = self.process_spatial_queries_safe(
            image_features, video_features
        )?;
        
        // Process temporal queries (only for modalities that support temporal features)
        let temporal_features = self.process_temporal_queries_safe(
            audio_features, video_features
        )?;
        
        // Compute attention weights with validation
        let attention_weights = self.compute_attention_weights_safe(
            &semantic_features, &spatial_features, &temporal_features
        )?;
        
        // Validate output features
        self.validate_output_features(&semantic_features, &spatial_features, &temporal_features)?;
        
        Ok(QueryFeatures {
            semantic_features,
            spatial_features,
            temporal_features,
            attention_weights: Some(attention_weights),
        })
    }
    
    /// Comprehensive validation of encoded features
    fn validate_encoded_features(&self, features: &EncodedFeatures) -> Result<()> {
        // Validate each modality if present
        if let Some(ref img_feat) = features.image_features {
            self.validate_tensor_dimensions(img_feat, "image")?;
        }
        
        if let Some(ref audio_feat) = features.audio_features {
            self.validate_tensor_dimensions(audio_feat, "audio")?;
        }
        
        if let Some(ref video_feat) = features.video_features {
            self.validate_tensor_dimensions(video_feat, "video")?;
        }
        
        if let Some(ref text_feat) = features.text_features {
            self.validate_tensor_dimensions(text_feat, "text")?;
        }
        
        // Validate cross-modality consistency
        self.validate_cross_modality_consistency(features)?;
        
        Ok(())
    }
    
    /// Validate tensor dimensions and values
    fn validate_tensor_dimensions(&self, tensor: &ArrayD<f32>, modality: &str) -> Result<()> {
        let shape = tensor.shape();
        
        // Check minimum dimensions
        if shape.len() < 2 {
            return Err(crate::multimodal::caffeine::error::CaffeineError::tensor_operation(
                &format!("{} tensor must be at least 2D", modality)
            ));
        }
        
        // Check for valid dimensions
        if shape[0] == 0 || shape[1] == 0 {
            return Err(crate::multimodal::caffeine::error::CaffeineError::tensor_operation(
                &format!("{} tensor has invalid dimensions: {:?}", modality, shape)
            ));
        }
        
        // Check for finite values
        for (idx, &val) in tensor.iter().enumerate() {
            if !val.is_finite() {
                return Err(crate::multimodal::caffeine::error::CaffeineError::tensor_operation(
                    &format!("{} tensor contains non-finite value at index {}", modality, idx)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Validate cross-modality consistency
    fn validate_cross_modality_consistency(&self, features: &EncodedFeatures) -> Result<()> {
        let mut batch_sizes = Vec::new();
        let mut embed_dims = Vec::new();
        
        if let Some(ref img_feat) = features.image_features {
            let shape = img_feat.shape();
            batch_sizes.push(shape[0]);
            embed_dims.push(shape[shape.len() - 1]);
        }
        
        if let Some(ref audio_feat) = features.audio_features {
            let shape = audio_feat.shape();
            batch_sizes.push(shape[0]);
            embed_dims.push(shape[shape.len() - 1]);
        }
        
        if let Some(ref video_feat) = features.video_features {
            let shape = video_feat.shape();
            batch_sizes.push(shape[0]);
            embed_dims.push(shape[shape.len() - 1]);
        }
        
        if let Some(ref text_feat) = features.text_features {
            let shape = text_feat.shape();
            batch_sizes.push(shape[0]);
            embed_dims.push(shape[shape.len() - 1]);
        }
        
        // Check batch size consistency
        if !batch_sizes.is_empty() {
            let first_batch_size = batch_sizes[0];
            for (i, &batch_size) in batch_sizes.iter().enumerate() {
                if batch_size != first_batch_size {
                    return Err(crate::multimodal::caffeine::error::CaffeineError::tensor_operation(
                        &format!("Inconsistent batch sizes: modality {} has batch size {}, expected {}", 
                                i, batch_size, first_batch_size)
                    ));
                }
            }
        }
        
        // Check embedding dimension compatibility
        if !embed_dims.is_empty() {
            let max_embed_dim = *embed_dims.iter().max().expect("embed_dims is non-empty");
            let min_embed_dim = *embed_dims.iter().min().expect("embed_dims is non-empty");
            
            if max_embed_dim > min_embed_dim * 4 {
                return Err(crate::multimodal::caffeine::error::CaffeineError::tensor_operation(
                    &format!("Embedding dimensions too different: min={}, max={}", min_embed_dim, max_embed_dim)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Safe semantic query processing with error handling
    fn process_semantic_queries_safe(
        &mut self,
        image_features: Option<&ArrayD<f32>>,
        audio_features: Option<&ArrayD<f32>>,
        video_features: Option<&ArrayD<f32>>,
        text_features: Option<&ArrayD<f32>>,
    ) -> Result<ArrayD<f32>> {
        let mut semantic_input = Vec::new();
        
        // Collect semantic features from all modalities with validation
        if let Some(img_feat) = image_features {
            let pooled = self.global_pool_safe(img_feat)?;
            semantic_input.push(pooled);
        }
        
        if let Some(audio_feat) = audio_features {
            let pooled = self.global_pool_safe(audio_feat)?;
            semantic_input.push(pooled);
        }
        
        if let Some(video_feat) = video_features {
            let pooled = self.global_pool_safe(video_feat)?;
            semantic_input.push(pooled);
        }
        
        if let Some(text_feat) = text_features {
            let pooled = self.global_pool_safe(text_feat)?;
            semantic_input.push(pooled);
        }
        
        if semantic_input.is_empty() {
            return Err(crate::multimodal::caffeine::error::CaffeineError::qformer(
                "No semantic features available for processing"
            ));
        }
        
        // Apply semantic queries with error handling
        self.semantic_queries.forward(&semantic_input)
            .map_err(|e| crate::multimodal::caffeine::error::CaffeineError::qformer(
                &format!("Semantic query processing failed: {}", e)
            ))
    }
    
    /// Safe global pooling with enhanced validation
    fn global_pool_safe(&self, features: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let shape = features.shape();
        
        // Enhanced input validation
        if shape.len() < 3 {
            return Err(crate::multimodal::caffeine::error::CaffeineError::tensor_operation(
                "Features must be at least 3D (batch_size, seq_len, embed_dim)"
            ));
        }
        
        let batch_size = shape[0];
        let seq_len = shape[1];
        let embed_dim = shape[2];
        
        if seq_len == 0 {
            return Err(crate::multimodal::caffeine::error::CaffeineError::tensor_operation(
                "Sequence length cannot be 0"
            ));
        }
        
        if embed_dim == 0 {
            return Err(crate::multimodal::caffeine::error::CaffeineError::tensor_operation(
                "Embedding dimension cannot be 0"
            ));
        }
        
        // Check for reasonable sizes
        if batch_size > 1024 || seq_len > 4096 || embed_dim > 8192 {
            warn!("Large tensor dimensions detected: batch={}, seq={}, embed={}", 
                  batch_size, seq_len, embed_dim);
        }
        
        // Optimized pooling with overflow protection
        let mut pooled = vec![0.0f32; batch_size * embed_dim];
        
        for b in 0..batch_size {
            for d in 0..embed_dim {
                let mut sum = 0.0f32;
                let mut count = 0u32;
                
                for s in 0..seq_len {
                    if let Some(&val) = features.get([b, s, d]) {
                        if val.is_finite() {
                            sum += val;
                            count += 1;
                        }
                    }
                }
                
                let idx = b * embed_dim + d;
                pooled[idx] = if count > 0 { sum / count as f32 } else { 0.0 };
            }
        }
        
        let pooled_shape = vec![batch_size, embed_dim];
        ArrayD::from_shape_vec(pooled_shape, pooled)
            .map_err(|e| crate::multimodal::caffeine::error::CaffeineError::tensor_operation(
                &format!("Failed to create pooled tensor: {}", e)
            ))
    }
    
    /// Safe spatial query processing
    fn process_spatial_queries_safe(
        &mut self,
        image_features: Option<&ArrayD<f32>>,
        video_features: Option<&ArrayD<f32>>,
    ) -> Result<ArrayD<f32>> {
        let mut spatial_input = Vec::new();
        
        if let Some(img_feat) = image_features {
            let spatial = self.extract_spatial_features_safe(img_feat)?;
            spatial_input.push(spatial);
        }
        
        if let Some(video_feat) = video_features {
            let spatial = self.extract_spatial_features_safe(video_feat)?;
            spatial_input.push(spatial);
        }
        
        if spatial_input.is_empty() {
            // Return empty tensor with correct dimensions
            let empty_shape = vec![1, 1, 1, self.config.hidden_dim];
            return Ok(ArrayD::from_elem(empty_shape, 0.0f32));
        }
        
        self.spatial_queries.forward(&spatial_input)
            .map_err(|e| crate::multimodal::caffeine::error::CaffeineError::qformer(
                &format!("Spatial query processing failed: {}", e)
            ))
    }
    
    /// Safe spatial feature extraction
    fn extract_spatial_features_safe(&self, features: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let shape = features.shape();
        
        // Enhanced validation
        if shape.len() < 3 {
            return Err(crate::multimodal::caffeine::error::CaffeineError::tensor_operation(
                "Features must be at least 3D (batch_size, seq_len, embed_dim)"
            ));
        }
        
        let batch_size = shape[0];
        let seq_len = shape[1];
        let embed_dim = shape[2];
        
        if seq_len == 0 {
            return Err(crate::multimodal::caffeine::error::CaffeineError::tensor_operation(
                "Sequence length cannot be 0 for spatial processing"
            ));
        }
        
        // Calculate optimal spatial dimensions with error handling
        let spatial_size = self.calculate_optimal_spatial_size(seq_len)?;
        
        // Validate spatial size
        if spatial_size.0 * spatial_size.1 != seq_len {
            warn!("Spatial size {}x{} doesn't match sequence length {}", 
                  spatial_size.0, spatial_size.1, seq_len);
        }
        
        let mut spatial_features = vec![0.0f32; batch_size * spatial_size.0 * spatial_size.1 * embed_dim];
        
        for b in 0..batch_size {
            for i in 0..spatial_size.0 {
                for j in 0..spatial_size.1 {
                    for d in 0..embed_dim {
                        let seq_idx = i * spatial_size.1 + j;
                        if seq_idx < seq_len {
                            if let Some(&val) = features.get([b, seq_idx, d]) {
                                if val.is_finite() {
                                    let output_idx = b * spatial_size.0 * spatial_size.1 * embed_dim + 
                                                   i * spatial_size.1 * embed_dim + j * embed_dim + d;
                                    spatial_features[output_idx] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        let spatial_shape = vec![batch_size, spatial_size.0, spatial_size.1, embed_dim];
        ArrayD::from_shape_vec(spatial_shape, spatial_features)
            .map_err(|e| crate::multimodal::caffeine::error::CaffeineError::tensor_operation(
                &format!("Failed to create spatial tensor: {}", e)
            ))
    }
    
    /// Safe temporal query processing
    fn process_temporal_queries_safe(
        &mut self,
        audio_features: Option<&ArrayD<f32>>,
        video_features: Option<&ArrayD<f32>>,
    ) -> Result<ArrayD<f32>> {
        let mut temporal_input = Vec::new();
        
        if let Some(audio_feat) = audio_features {
            let temporal = self.extract_temporal_features_safe(audio_feat)?;
            temporal_input.push(temporal);
        }
        
        if let Some(video_feat) = video_features {
            let temporal = self.extract_temporal_features_safe(video_feat)?;
            temporal_input.push(temporal);
        }
        
        if temporal_input.is_empty() {
            // Return empty tensor with correct dimensions
            let empty_shape = vec![1, 1, self.config.hidden_dim];
            return Ok(ArrayD::from_elem(empty_shape, 0.0f32));
        }
        
        self.temporal_queries.forward(&temporal_input)
            .map_err(|e| crate::multimodal::caffeine::error::CaffeineError::qformer(
                &format!("Temporal query processing failed: {}", e)
            ))
    }
    
    /// Safe temporal feature extraction
    fn extract_temporal_features_safe(&self, features: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let shape = features.shape();
        
        if shape.len() < 3 {
            return Err(crate::multimodal::caffeine::error::CaffeineError::tensor_operation(
                "Features must be at least 3D (batch_size, seq_len, embed_dim)"
            ));
        }
        
        let batch_size = shape[0];
        let seq_len = shape[1];
        let embed_dim = shape[2];
        
        if seq_len == 0 {
            return Err(crate::multimodal::caffeine::error::CaffeineError::tensor_operation(
                "Sequence length cannot be 0 for temporal processing"
            ));
        }
        
        let mut temporal_features = vec![0.0f32; batch_size * seq_len * embed_dim];
        
        for b in 0..batch_size {
            for t in 0..seq_len {
                for d in 0..embed_dim {
                    if let Some(&val) = features.get([b, t, d]) {
                        if val.is_finite() {
                            // Add temporal position encoding with bounds checking
                            let temporal_encoding = (t as f32 * 0.01).sin();
                            let output_idx = b * seq_len * embed_dim + t * embed_dim + d;
                            temporal_features[output_idx] = val + temporal_encoding;
                        }
                    }
                }
            }
        }
        
        let temporal_shape = vec![batch_size, seq_len, embed_dim];
        ArrayD::from_shape_vec(temporal_shape, temporal_features)
            .map_err(|e| crate::multimodal::caffeine::error::CaffeineError::tensor_operation(
                &format!("Failed to create temporal tensor: {}", e)
            ))
    }
    
    /// Safe attention weight computation
    fn compute_attention_weights_safe(
        &mut self,
        semantic: &ArrayD<f32>,
        spatial: &ArrayD<f32>,
        temporal: &ArrayD<f32>,
    ) -> Result<ArrayD<f32>> {
        // Validate input tensors
        let semantic_shape = semantic.shape();
        let spatial_shape = spatial.shape();
        let temporal_shape = temporal.shape();
        
        // Flatten all query features with error handling
        let semantic_flat = semantic.view().into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| crate::multimodal::caffeine::error::CaffeineError::qformer(
                &format!("Failed to convert semantic to 2D: {}", e)
            ))?;
        let spatial_flat = spatial.view().into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| crate::multimodal::caffeine::error::CaffeineError::qformer(
                &format!("Failed to convert spatial to 2D: {}", e)
            ))?;
        let temporal_flat = temporal.view().into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| crate::multimodal::caffeine::error::CaffeineError::qformer(
                &format!("Failed to convert temporal to 2D: {}", e)
            ))?;
        
        // Concatenate for cross-attention with bounds checking
        let mut combined = Vec::new();
        
        if let Some(semantic_slice) = semantic_flat.as_slice() {
            combined.extend_from_slice(semantic_slice);
        }
        
        if let Some(spatial_slice) = spatial_flat.as_slice() {
            combined.extend_from_slice(spatial_slice);
        }
        
        if let Some(temporal_slice) = temporal_flat.as_slice() {
            combined.extend_from_slice(temporal_slice);
        }
        
        let total_queries = self.config.num_semantic_queries + 
                          self.config.num_spatial_queries + 
                          self.config.num_temporal_queries;
        
        // Validate combined size
        if combined.len() != total_queries * self.config.hidden_dim {
            return Err(crate::multimodal::caffeine::error::CaffeineError::qformer(
                &format!("Combined features size {} doesn't match expected {}x{}", 
                        combined.len(), total_queries, self.config.hidden_dim)
            ));
        }
        
        let attention_weights = self.attention_mechanism.compute_cross_attention(
            &combined,
            total_queries,
            self.config.hidden_dim,
        )?;
        
        // Validate attention weights
        if attention_weights.len() != total_queries * total_queries {
            return Err(crate::multimodal::caffeine::error::CaffeineError::qformer(
                &format!("Attention weights size {} doesn't match expected {}x{}", 
                        attention_weights.len(), total_queries, total_queries)
            ));
        }
        
        let shape = vec![total_queries, total_queries];
        ArrayD::from_shape_vec(shape, attention_weights)
            .map_err(|e| crate::multimodal::caffeine::error::CaffeineError::qformer(
                &format!("Failed to create attention weights tensor: {}", e)
            ))
    }
    
    /// Validate output features
    fn validate_output_features(
        &self,
        semantic: &ArrayD<f32>,
        spatial: &ArrayD<f32>,
        temporal: &ArrayD<f32>,
    ) -> Result<()> {
        // Check for finite values
        for (name, tensor) in [
            ("semantic", semantic),
            ("spatial", spatial),
            ("temporal", temporal),
        ] {
            for (idx, &val) in tensor.iter().enumerate() {
                if !val.is_finite() {
                    return Err(crate::multimodal::caffeine::error::CaffeineError::tensor_operation(
                        &format!("{} output contains non-finite value at index {}", name, idx)
                    ));
                }
            }
        }
        
        Ok(())
    }
    
    /// Process semantic queries
    fn process_semantic_queries(
        &mut self,
        image_features: Option<&ArrayD<f32>>,
        audio_features: Option<&ArrayD<f32>>,
        video_features: Option<&ArrayD<f32>>,
        text_features: Option<&ArrayD<f32>>,
    ) -> Result<ArrayD<f32>> {
        let mut semantic_input = Vec::new();
        
        // Collect semantic features from all modalities
        if let Some(img_feat) = image_features {
            semantic_input.push(self.global_pool(img_feat)?);
        }
        
        if let Some(audio_feat) = audio_features {
            semantic_input.push(self.global_pool(audio_feat)?);
        }
        
        if let Some(video_feat) = video_features {
            semantic_input.push(self.global_pool(video_feat)?);
        }
        
        if let Some(text_feat) = text_features {
            semantic_input.push(self.global_pool(text_feat)?);
        }
        
        // Apply semantic queries
        let semantic_output = self.semantic_queries.forward(&semantic_input)?;
        
        Ok(semantic_output)
    }
    
    /// Process spatial queries
    fn process_spatial_queries(
        &mut self,
        image_features: Option<&ArrayD<f32>>,
        video_features: Option<&ArrayD<f32>>,
    ) -> Result<ArrayD<f32>> {
        let mut spatial_input = Vec::new();
        
        // Collect spatial features
        if let Some(img_feat) = image_features {
            spatial_input.push(self.extract_spatial_features(img_feat)?);
        }
        
        if let Some(video_feat) = video_features {
            spatial_input.push(self.extract_spatial_features(video_feat)?);
        }
        
        // Apply spatial queries
        let spatial_output = self.spatial_queries.forward(&spatial_input)?;
        
        Ok(spatial_output)
    }
    
    /// Process temporal queries
    fn process_temporal_queries(
        &mut self,
        audio_features: Option<&ArrayD<f32>>,
        video_features: Option<&ArrayD<f32>>,
    ) -> Result<ArrayD<f32>> {
        let mut temporal_input = Vec::new();
        
        // Collect temporal features
        if let Some(audio_feat) = audio_features {
            temporal_input.push(self.extract_temporal_features(audio_feat)?);
        }
        
        if let Some(video_feat) = video_features {
            temporal_input.push(self.extract_temporal_features(video_feat)?);
        }
        
        // Apply temporal queries
        let temporal_output = self.temporal_queries.forward(&temporal_input)?;
        
        Ok(temporal_output)
    }
    
    /// Global pooling for semantic features with optimized computation
    fn global_pool(&self, features: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let shape = features.shape();
        
        // Validate input dimensions
        if shape.len() < 3 {
            return Err(crate::multimodal::caffeine::error::CaffeineError::tensor_operation(
                "Features must be at least 3D (batch_size, seq_len, embed_dim)"
            ));
        }
        
        let batch_size = shape[0];
        let seq_len = shape[1];
        let embed_dim = shape[2];
        
        if seq_len == 0 {
            return Err(crate::multimodal::caffeine::error::CaffeineError::tensor_operation(
                "Sequence length cannot be 0"
            ));
        }
        
        // Optimized pooling using pre-allocated memory
        let mut pooled = vec![0.0f32; batch_size * embed_dim];
        
        for b in 0..batch_size {
            for d in 0..embed_dim {
                let mut sum = 0.0f32;
                let mut count = 0u32;
                
                // Vectorized sum computation
                for s in 0..seq_len {
                    if let Some(&val) = features.get([b, s, d]) {
                        sum += val;
                        count += 1;
                    }
                }
                
                let idx = b * embed_dim + d;
                pooled[idx] = if count > 0 { sum / count as f32 } else { 0.0 };
            }
        }
        
        let pooled_shape = vec![batch_size, embed_dim];
        Ok(ArrayD::from_shape_vec(pooled_shape, pooled)?)
    }
    
    /// Extract spatial features with flexible layout handling
    fn extract_spatial_features(&self, features: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let shape = features.shape();
        
        // Validate input dimensions
        if shape.len() < 3 {
            return Err(crate::multimodal::caffeine::error::CaffeineError::tensor_operation(
                "Features must be at least 3D (batch_size, seq_len, embed_dim)"
            ));
        }
        
        let batch_size = shape[0];
        let seq_len = shape[1];
        let embed_dim = shape[2];
        
        if seq_len == 0 {
            return Err(crate::multimodal::caffeine::error::CaffeineError::tensor_operation(
                "Sequence length cannot be 0"
            ));
        }
        
        // Calculate optimal spatial dimensions (not assuming square)
        let spatial_size = self.calculate_optimal_spatial_size(seq_len)?;
        let mut spatial_features = vec![0.0f32; batch_size * spatial_size.0 * spatial_size.1 * embed_dim];
        
        for b in 0..batch_size {
            for i in 0..spatial_size.0 {
                for j in 0..spatial_size.1 {
                    for d in 0..embed_dim {
                        let seq_idx = i * spatial_size.1 + j;
                        if seq_idx < seq_len {
                            if let Some(&val) = features.get([b, seq_idx, d]) {
                                let output_idx = b * spatial_size.0 * spatial_size.1 * embed_dim + 
                                               i * spatial_size.1 * embed_dim + j * embed_dim + d;
                                spatial_features[output_idx] = val;
                            }
                        }
                    }
                }
            }
        }
        
        let spatial_shape = vec![batch_size, spatial_size.0, spatial_size.1, embed_dim];
        Ok(ArrayD::from_shape_vec(spatial_shape, spatial_features)?)
    }
    
    /// Calculate optimal spatial dimensions from sequence length
    fn calculate_optimal_spatial_size(&self, seq_len: usize) -> Result<(usize, usize)> {
        if seq_len == 0 {
            return Err(crate::multimodal::caffeine::error::CaffeineError::tensor_operation(
                "Cannot calculate spatial size for zero sequence length"
            ));
        }
        
        // Find factors that are closest to square
        let mut best_size = (1, seq_len);
        let mut min_diff = seq_len;
        
        for i in 1..=(seq_len as f64).sqrt() as usize {
            if seq_len % i == 0 {
                let j = seq_len / i;
                let diff = (i as isize - j as isize).abs() as usize;
                if diff < min_diff {
                    min_diff = diff;
                    best_size = (i, j);
                }
            }
        }
        
        Ok(best_size)
    }
    
    /// Extract temporal features
    fn extract_temporal_features(&self, features: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let shape = features.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let embed_dim = shape[2];
        
        // For temporal features, we'll use the sequence dimension as time
        let mut temporal_features = vec![0.0f32; batch_size * seq_len * embed_dim];
        
        for b in 0..batch_size {
            for t in 0..seq_len {
                for d in 0..embed_dim {
                    if let Some(&val) = features.get([b, t, d]) {
                        // Add temporal position encoding
                        let temporal_encoding = (t as f32 * 0.01).sin();
                        let output_idx = b * seq_len * embed_dim + t * embed_dim + d;
                        temporal_features[output_idx] = val + temporal_encoding;
                    }
                }
            }
        }
        
        let temporal_shape = vec![batch_size, seq_len, embed_dim];
        Ok(ArrayD::from_shape_vec(temporal_shape, temporal_features)?)
    }
    
    /// Compute attention weights between query sets
    fn compute_attention_weights(
        &mut self,
        semantic: &ArrayD<f32>,
        spatial: &ArrayD<f32>,
        temporal: &ArrayD<f32>,
    ) -> Result<ArrayD<f32>> {
        // Flatten all query features
        let semantic_flat = semantic.view().into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| crate::multimodal::caffeine::error::CaffeineError::tensor_operation(&format!("Failed to convert semantic to 2D: {}", e)))?;
        let spatial_flat = spatial.view().into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| crate::multimodal::caffeine::error::CaffeineError::tensor_operation(&format!("Failed to convert spatial to 2D: {}", e)))?;
        let temporal_flat = temporal.view().into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| crate::multimodal::caffeine::error::CaffeineError::tensor_operation(&format!("Failed to convert temporal to 2D: {}", e)))?;
        
        // Concatenate for cross-attention
        let mut combined = Vec::new();
        combined.extend_from_slice(semantic_flat.as_slice().unwrap_or(&[]));
        combined.extend_from_slice(spatial_flat.as_slice().unwrap_or(&[]));
        combined.extend_from_slice(temporal_flat.as_slice().unwrap_or(&[]));
        
        let total_queries = self.config.num_semantic_queries + 
                          self.config.num_spatial_queries + 
                          self.config.num_temporal_queries;
        
        let attention_weights = self.attention_mechanism.compute_cross_attention(
            &combined,
            total_queries,
            self.config.hidden_dim,
        )?;
        
        let shape = vec![total_queries, total_queries];
        Ok(ArrayD::from_shape_vec(shape, attention_weights)?)
    }
    
    /// Get query statistics
    pub fn get_query_stats(&self) -> QueryStats {
        QueryStats {
            num_semantic_queries: self.config.num_semantic_queries,
            num_spatial_queries: self.config.num_spatial_queries,
            num_temporal_queries: self.config.num_temporal_queries,
            hidden_dim: self.config.hidden_dim,
            num_attention_heads: self.config.num_attention_heads,
        }
    }
}

/// Query statistics
#[derive(Debug, Clone)]
pub struct QueryStats {
    pub num_semantic_queries: usize,
    pub num_spatial_queries: usize,
    pub num_temporal_queries: usize,
    pub hidden_dim: usize,
    pub num_attention_heads: usize,
}
