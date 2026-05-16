use crate::DLResult;
use crate::gnac::smart_tensor::ShapePropEntry;
use crate::gnac::canvas::GraphNode;
use crate::NodeType;
use std::collections::HashMap;

/// Shape Propagation Engine
/// Menghitung kompatibilitas dimensi tensor antar node secara otomatis
pub struct ShapePropagator {
    cache: HashMap<String, ShapePropEntry>,
}

impl ShapePropagator {
    pub fn new() -> Self {
        ShapePropagator {
            cache: HashMap::new(),
        }
    }

    /// Propagasi shape dari input ke output untuk node tertentu
    pub fn propagate(&mut self, node: &GraphNode, input_shapes: &[Vec<usize>]) -> DLResult<Vec<Vec<usize>>> {
        let cache_key = format!("{:?}:{:?}", node.node_type, input_shapes);
        if let Some(cached) = self.cache.get(&cache_key) {
            if cached.compatible {
                return Ok(vec![cached.output_shape.clone()]);
            }
            return Err(crate::DeepLearningError::ShapeMismatch {
                expected: cached.input_shape.clone(),
                actual: input_shapes[0].clone(),
            });
        }

        let result = self.compute_output_shape(&node.node_type, input_shapes);
        let entry = match &result {
            Ok(shapes) => ShapePropEntry {
                input_shape: input_shapes[0].clone(),
                output_shape: shapes[0].clone(),
                compatible: true,
                error_message: None,
            },
            Err(e) => ShapePropEntry {
                input_shape: input_shapes[0].clone(),
                output_shape: vec![],
                compatible: false,
                error_message: Some(e.to_string()),
            },
        };
        self.cache.insert(cache_key, entry);
        result
    }

    fn compute_output_shape(&self, node_type: &NodeType, inputs: &[Vec<usize>]) -> DLResult<Vec<Vec<usize>>> {
        match node_type {
            NodeType::Linear => {
                let input = &inputs[0];
                if input.len() != 2 {
                    return Err(crate::DeepLearningError::InvalidDimension { dim: input.len() });
                }
                Ok(vec![vec![input[0], 3072]])
            }
            NodeType::Conv2D => {
                let input = &inputs[0];
                if input.len() != 4 {
                    return Err(crate::DeepLearningError::InvalidDimension { dim: input.len() });
                }
                let (n, c, h, w) = (input[0], input[1], input[2], input[3]);
                // stride=1, padding=same, 64 filters
                Ok(vec![vec![n, 64, h, w]])
            }
            NodeType::SelfAttention | NodeType::MultiHeadAttention => {
                let input = &inputs[0];
                if input.len() != 3 {
                    return Err(crate::DeepLearningError::InvalidDimension { dim: input.len() });
                }
                Ok(vec![vec![input[0], input[1], input[2]], vec![input[0], 12, input[1], input[1]]])
            }
            NodeType::LayerNorm | NodeType::RMSNorm => {
                Ok(vec![inputs[0].clone()])
            }
            NodeType::ReLU | NodeType::GELU | NodeType::Sigmoid | NodeType::Tanh | NodeType::Dropout => {
                Ok(vec![inputs[0].clone()])
            }
            NodeType::MaxPool | NodeType::AvgPool => {
                let input = &inputs[0];
                if input.len() == 4 {
                    Ok(vec![vec![input[0], input[1], input[2] / 2, input[3] / 2]])
                } else {
                    Err(crate::DeepLearningError::InvalidDimension { dim: input.len() })
                }
            }
            NodeType::Concat => {
                if inputs.len() < 2 {
                    return Err(crate::DeepLearningError::Configuration {
                        reason: "Concat requires at least 2 inputs".to_string(),
                    });
                }
                let mut result = inputs[0].clone();
                let concat_dim = result.len() - 1;
                for input in inputs.iter().skip(1) {
                    result[concat_dim] += input[concat_dim];
                }
                Ok(vec![result])
            }
            NodeType::Reshape => {
                // Default: flatten to 2D
                let input = &inputs[0];
                let numel: usize = input.iter().product();
                Ok(vec![vec![numel]])
            }
            NodeType::Input => Ok(vec![inputs[0].clone()]),
            _ => Ok(vec![inputs[0].clone()]),
        }
    }

    /// Validasi kompatibilitas antara dua tensor
    pub fn validate_connection(&self, source_shape: &[usize], target_shape: &[usize]) -> DLResult<()> {
        if source_shape == target_shape {
            return Ok(());
        }
        // Allow broadcasting: source lebih pendek
        if source_shape.len() < target_shape.len() {
            let padded: Vec<usize> = std::iter::repeat(1)
                .take(target_shape.len() - source_shape.len())
                .chain(source_shape.iter().cloned())
                .collect();
            if padded.iter().zip(target_shape.iter()).all(|(a, b)| *a == *b || *a == 1 || *b == 1) {
                return Ok(());
            }
        }
        Err(crate::DeepLearningError::ShapeMismatch {
            expected: target_shape.to_vec(),
            actual: source_shape.to_vec(),
        })
    }

    /// Rekomendasi transformasi untuk memperbaiki ketidakcocokan
    pub fn suggest_fix(&self, source: &[usize], target: &[usize]) -> Vec<String> {
        let mut suggestions = Vec::new();

        if source.len() != target.len() {
            suggestions.push(format!(
                "Add Reshape Node: change shape {:?} to {:?}",
                source, target
            ));
        }

        // Product check
        let src_product: usize = source.iter().product();
        let tgt_product: usize = target.iter().product();
        if src_product == tgt_product && source.len() != target.len() {
            suggestions.push(format!(
                "Use Reshape: same element count, different dimensions ({:?} -> {:?})",
                source, target
            ));
        }

        if suggestions.is_empty() {
            suggestions.push(format!(
                "Add Projection Layer (Linear): {:?} -> {:?}",
                source, target
            ));
            suggestions.push(format!(
                "Add 1x1 Convolution to adjust channels: {:?} -> {:?}",
                source, target
            ));
        }

        suggestions
    }
}
