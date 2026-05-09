//! Spatial grounding module for CAFFEINE
//! 
//! Implements bounding box detection and segmentation

use crate::caffeine::types::*;
use crate::caffeine::config::BBoxFormat as ConfigBBoxFormat;
use crate::caffeine::error::Result;
use ndarray::ArrayD;

/// Spatial grounding module
pub struct SpatialGroundingModule {
    config: crate::caffeine::config::ActionConfig,
    bbox_detector: BoundingBoxDetector,
    segmentor: SegmentationModule,
}

impl SpatialGroundingModule {
    /// Create new spatial grounding module
    pub fn new(config: crate::caffeine::config::ActionConfig) -> Result<Self> {
        let bbox_detector = BoundingBoxDetector::new(config.clone())?;
        let segmentor = SegmentationModule::new(config.clone())?;
        
        Ok(Self {
            config,
            bbox_detector,
            segmentor,
        })
    }
    
    /// Generate spatial grounding from tokens
    pub fn generate(&mut self, tokens: &[UnifiedToken], inputs: &MultiModalInputs) -> Result<Vec<SpatialGrounding>> {
        let mut grounding_results = Vec::new();
        
        // Process each image in the input
        if let Some(ref image_input) = inputs.image {
            let grounding = self.process_image(tokens, image_input)?;
            grounding_results.push(grounding);
        }
        
        // Process video frames
        if let Some(ref video_input) = inputs.video {
            for (frame_idx, frame) in video_input.frames.iter().enumerate() {
                let grounding = self.process_frame(tokens, frame, frame_idx)?;
                grounding_results.push(grounding);
            }
        }
        
        Ok(grounding_results)
    }
    
    /// Process single image for spatial grounding
    fn process_image(&mut self, tokens: &[UnifiedToken], image_input: &ImageInput) -> Result<SpatialGrounding> {
        // Detect bounding boxes
        let bounding_boxes = self.bbox_detector.detect(tokens, image_input)?;
        
        // Generate segmentation masks
        let segmentation_masks = self.segmentor.segment(tokens, image_input)?;
        
        // Generate confidence scores
        let confidence_scores = self.generate_confidence_scores(&bounding_boxes, tokens)?;
        
        // Generate class labels
        let class_labels = self.generate_class_labels(&bounding_boxes, tokens)?;
        
        Ok(SpatialGrounding {
            bounding_boxes,
            segmentation_masks: Some(segmentation_masks),
            confidence_scores,
            class_labels,
        })
    }
    
    /// Process video frame for spatial grounding
    fn process_frame(&mut self, tokens: &[UnifiedToken], frame: &ImageInput, frame_idx: usize) -> Result<SpatialGrounding> {
        // Similar to image processing but with temporal context
        let mut frame_tokens = tokens.to_vec();
        
        // Add temporal information to tokens
        for token in &mut frame_tokens {
            token.timestamp = Some(frame_idx as f32 / 30.0); // Assuming 30 FPS
        }
        
        self.process_image(&frame_tokens, frame)
    }
    
    /// Generate confidence scores
    fn generate_confidence_scores(&self, bounding_boxes: &[BoundingBox], tokens: &[UnifiedToken]) -> Result<Vec<f32>> {
        let mut scores = Vec::new();
        
        for bbox in bounding_boxes {
            // Base confidence from bbox
            let mut confidence = bbox.confidence;
            
            // Adjust based on token relevance
            let token_relevance = self.calculate_token_relevance(bbox, tokens)?;
            confidence *= (1.0 + token_relevance) / 2.0;
            
            scores.push(confidence.clamp(0.0, 1.0));
        }
        
        Ok(scores)
    }
    
    /// Calculate token relevance for bounding box
    fn calculate_token_relevance(&self, bbox: &BoundingBox, tokens: &[UnifiedToken]) -> Result<f32> {
        let mut relevance = 0.0f32;
        let mut count = 0.0f32;
        
        for token in tokens {
            if token.modality == ModalityType::Image {
                // Simple relevance calculation based on token position
                if let Some((x, y, w, h)) = token.spatial_coords {
                    let center_x = x + w / 2.0;
                    let center_y = y + h / 2.0;
                    
                    // Check if token is near bbox center
                    let bbox_center_x = (bbox.coords.0 + bbox.coords.2) / 2.0;
                    let bbox_center_y = (bbox.coords.1 + bbox.coords.3) / 2.0;
                    
                    let distance = ((center_x - bbox_center_x).powi(2) + (center_y - bbox_center_y).powi(2)).sqrt();
                    let max_distance = (bbox.coords.2 - bbox.coords.0).max(bbox.coords.3 - bbox.coords.1);
                    
                    if max_distance > 0.0 {
                        relevance += 1.0 - (distance / max_distance).min(1.0);
                    }
                    count += 1.0;
                }
            }
        }
        
        Ok(if count > 0.0 { relevance / count } else { 0.0 })
    }
    
    /// Generate class labels
    fn generate_class_labels(&self, bounding_boxes: &[BoundingBox], tokens: &[UnifiedToken]) -> Result<Vec<String>> {
        let mut labels = Vec::new();
        
        for bbox in bounding_boxes {
            if let Some(ref label) = bbox.label {
                labels.push(label.clone());
            } else {
                // Generate label from tokens
                let label = self.generate_label_from_tokens(bbox, tokens)?;
                labels.push(label);
            }
        }
        
        Ok(labels)
    }
    
    /// Generate label from tokens
    fn generate_label_from_tokens(&self, bbox: &BoundingBox, tokens: &[UnifiedToken]) -> Result<String> {
        // Find most relevant tokens for this bbox
        let mut relevant_tokens = Vec::new();
        
        for token in tokens {
            if token.modality == ModalityType::Text {
                // Simple relevance based on token embedding similarity
                relevant_tokens.push(token);
            }
        }
        
        // Select most relevant token as label
        if let Some(most_relevant) = relevant_tokens.first() {
            // Convert token ID to label (simplified)
            self.token_id_to_label(most_relevant.token_id)
        } else {
            Ok("object".to_string())
        }
    }
    
    /// Convert token ID to label
    fn token_id_to_label(&self, token_id: usize) -> Result<String> {
        let common_labels = vec![
            "person", "car", "tree", "house", "sky", "cloud", "sun", "moon", "star",
            "mountain", "river", "ocean", "beach", "forest", "road", "building",
            "animal", "dog", "cat", "bird", "flower", "grass", "rock", "sand",
            "phone", "computer", "book", "table", "chair", "door", "window",
        ];
        
        if token_id < common_labels.len() {
            Ok(common_labels[token_id].to_string())
        } else {
            Ok(format!("object_{}", token_id))
        }
    }
}

/// Bounding box detector
pub struct BoundingBoxDetector {
    config: crate::caffeine::config::ActionConfig,
    confidence_threshold: f32,
    nms_threshold: f32,
}

impl BoundingBoxDetector {
    /// Create new bounding box detector
    pub fn new(config: crate::caffeine::config::ActionConfig) -> Result<Self> {
        Ok(Self {
            config,
            confidence_threshold: 0.5,
            nms_threshold: 0.5,
        })
    }
    
    /// Detect bounding boxes
    pub fn detect(&mut self, tokens: &[UnifiedToken], image_input: &ImageInput) -> Result<Vec<BoundingBox>> {
        // Generate candidate boxes from tokens
        let mut candidates = self.generate_candidates(tokens, image_input)?;
        
        // Filter by confidence
        candidates.retain(|bbox| bbox.confidence >= self.confidence_threshold);
        
        // Apply non-maximum suppression
        let filtered_boxes = self.apply_nms(candidates)?;
        
        Ok(filtered_boxes)
    }
    
    /// Generate candidate bounding boxes
    fn generate_candidates(&self, tokens: &[UnifiedToken], image_input: &ImageInput) -> Result<Vec<BoundingBox>> {
        let mut candidates = Vec::new();
        
        // Generate boxes from image tokens
        for token in tokens {
            if token.modality == ModalityType::Image {
                if let Some((x, y, w, h)) = token.spatial_coords {
                    // Normalize coordinates to image size
                    let norm_x = x / image_input.width as f32;
                    let norm_y = y / image_input.height as f32;
                    let norm_w = w / image_input.width as f32;
                    let norm_h = h / image_input.height as f32;
                    
                    // Convert to target format
                    let coords = match self.config.bbox_format {
                        ConfigBBoxFormat::XYWH => (norm_x, norm_y, norm_w, norm_h),
                        ConfigBBoxFormat::XYXY => (
                            norm_x,
                            norm_y,
                            norm_x + norm_w,
                            norm_y + norm_h,
                        ),
                        ConfigBBoxFormat::CXCYWH => (
                            norm_x + norm_w / 2.0,
                            norm_y + norm_h / 2.0,
                            norm_w,
                            norm_h,
                        ),
                    };
                    
                    let bbox = BoundingBox {
                        coords,
                        format: match self.config.bbox_format {
                            ConfigBBoxFormat::XYWH => BBoxFormat::XYWH,
                            ConfigBBoxFormat::XYXY => BBoxFormat::XYXY,
                            ConfigBBoxFormat::CXCYWH => BBoxFormat::CXCYWH,
                        },
                        label: None,
                        confidence: 0.8, // Default confidence
                    };
                    
                    candidates.push(bbox);
                }
            }
        }
        
        // Add some default candidates if no tokens with spatial info
        if candidates.is_empty() {
            candidates.push(BoundingBox {
                coords: (0.1, 0.1, 0.8, 0.8),
                format: match self.config.bbox_format {
                    ConfigBBoxFormat::XYWH => BBoxFormat::XYWH,
                    ConfigBBoxFormat::XYXY => BBoxFormat::XYXY,
                    ConfigBBoxFormat::CXCYWH => BBoxFormat::CXCYWH,
                },
                label: Some("default".to_string()),
                confidence: 0.6,
            });
        }
        
        Ok(candidates)
    }
    
    /// Apply non-maximum suppression
    fn apply_nms(&self, mut boxes: Vec<BoundingBox>) -> Result<Vec<BoundingBox>> {
        if boxes.is_empty() {
            return Ok(boxes);
        }
        
        // Sort by confidence
        boxes.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut selected = Vec::new();
        let mut suppressed = std::collections::HashSet::new();
        
        for (i, box_i) in boxes.iter().enumerate() {
            if suppressed.contains(&i) {
                continue;
            }
            
            selected.push(box_i.clone());
            
            // Suppress overlapping boxes
            for (j, box_j) in boxes.iter().enumerate() {
                if i == j || suppressed.contains(&j) {
                    continue;
                }
                
                let iou = self.calculate_iou(box_i, box_j);
                if iou > self.nms_threshold {
                    suppressed.insert(j);
                }
            }
        }
        
        Ok(selected)
    }
    
    /// Calculate Intersection over Union (IoU)
    fn calculate_iou(&self, box1: &BoundingBox, box2: &BoundingBox) -> f32 {
        // Convert to XYXY format for calculation
        let (x1_1, y1_1, x2_1, y2_1) = self.to_xyxy(&box1.coords, &box1.format);
        let (x1_2, y1_2, x2_2, y2_2) = self.to_xyxy(&box2.coords, &box2.format);
        
        // Calculate intersection
        let x1_i = x1_1.max(x1_2);
        let y1_i = y1_1.max(y1_2);
        let x2_i = x2_1.min(x2_2);
        let y2_i = y2_1.min(y2_2);
        
        if x2_i <= x1_i || y2_i <= y1_i {
            return 0.0;
        }
        
        let intersection = (x2_i - x1_i) * (y2_i - y1_i);
        
        // Calculate union
        let area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
        let area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
        let union = area1 + area2 - intersection;
        
        if union == 0.0 {
            return 0.0;
        }
        
        intersection / union
    }
    
    /// Convert bbox coordinates to XYXY format
    fn to_xyxy(&self, coords: &(f32, f32, f32, f32), format: &BBoxFormat) -> (f32, f32, f32, f32) {
        match format {
            BBoxFormat::XYWH => (coords.0, coords.1, coords.0 + coords.2, coords.1 + coords.3),
            BBoxFormat::XYXY => *coords,
            BBoxFormat::CXCYWH => (
                coords.0 - coords.2 / 2.0,
                coords.1 - coords.3 / 2.0,
                coords.0 + coords.2 / 2.0,
                coords.1 + coords.3 / 2.0,
            ),
        }
    }
}

/// Segmentation module
pub struct SegmentationModule {
    config: crate::caffeine::config::ActionConfig,
    num_classes: usize,
}

impl SegmentationModule {
    /// Create new segmentation module
    pub fn new(config: crate::caffeine::config::ActionConfig) -> Result<Self> {
        Ok(Self {
            config,
            num_classes: 20, // Common number of classes
        })
    }
    
    /// Generate segmentation masks
    pub fn segment(&mut self, tokens: &[UnifiedToken], image_input: &ImageInput) -> Result<Vec<SegmentationMask>> {
        let mut masks = Vec::new();
        
        // Generate masks from tokens
        for token in tokens {
            if token.modality == ModalityType::Image {
                let mask = self.generate_mask_from_token(token, image_input)?;
                masks.push(mask);
            }
        }
        
        // If no masks generated, create a default one
        if masks.is_empty() {
            let default_mask = self.generate_default_mask(image_input)?;
            masks.push(default_mask);
        }
        
        Ok(masks)
    }
    
    /// Generate mask from token
    fn generate_mask_from_token(&self, token: &UnifiedToken, image_input: &ImageInput) -> Result<SegmentationMask> {
        let mut mask_data = vec![0u8; image_input.width * image_input.height];
        
        // Generate mask based on token spatial coordinates
        if let Some((x, y, w, h)) = token.spatial_coords {
            let start_x = (x * image_input.width as f32) as usize;
            let start_y = (y * image_input.height as f32) as usize;
            let end_x = ((x + w) * image_input.width as f32) as usize;
            let end_y = ((y + h) * image_input.height as f32) as usize;
            
            for py in start_y..std::cmp::min(end_y, image_input.height) {
                for px in start_x..std::cmp::min(end_x, image_input.width) {
                    let mask_idx = py * image_input.width + px;
                    if mask_idx < mask_data.len() {
                        mask_data[mask_idx] = 255;
                    }
                }
            }
        }
        
        Ok(SegmentationMask {
            mask: mask_data,
            width: image_input.width,
            height: image_input.height,
            label: Some(self.token_id_to_label(token.token_id)?),
            confidence: 0.8,
        })
    }
    
    /// Generate default mask
    fn generate_default_mask(&self, image_input: &ImageInput) -> Result<SegmentationMask> {
        let mask_data = vec![128u8; image_input.width * image_input.height]; // Gray mask
        
        Ok(SegmentationMask {
            mask: mask_data,
            width: image_input.width,
            height: image_input.height,
            label: Some("background".to_string()),
            confidence: 0.5,
        })
    }
    
    /// Convert token ID to label
    fn token_id_to_label(&self, token_id: usize) -> Result<String> {
        let class_names = vec![
            "background", "person", "car", "tree", "house", "sky", "cloud", "sun",
            "mountain", "river", "ocean", "beach", "forest", "road", "building",
            "animal", "dog", "cat", "bird", "flower",
        ];
        
        if token_id < class_names.len() {
            Ok(class_names[token_id].to_string())
        } else {
            Ok(format!("class_{}", token_id % self.num_classes))
        }
    }
}
