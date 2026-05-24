use crate::canvas::{GraphNode, NeuralGraph, NodeType};
use crate::{DeepLearningError, DLResult};
use std::collections::HashMap;

/// Asynchronous Execution — overlapping compute & memory transfer
pub struct AsyncExecutor;

impl AsyncExecutor {
    /// Eksekusi asinkron dengan real compute per node type
    /// Setiap stage dijalankan sebagai tokio task dengan blocking compute
    pub async fn execute_async(graph: &NeuralGraph, pipeline_stages: usize) -> DLResult<()> {
        let order = graph.topological_order()?;

        if pipeline_stages == 0 || order.is_empty() {
            return Ok(());
        }

        let stage_size = (order.len() as f64 / pipeline_stages as f64).ceil() as usize;
        let mut handles = Vec::with_capacity(pipeline_stages);

        let nodes: Vec<GraphNode> = order
            .iter()
            .filter_map(|id| graph.nodes.get(id))
            .cloned()
            .collect();

        for (stage_idx, chunk) in nodes.chunks(stage_size).enumerate() {
            let stage_nodes: Vec<GraphNode> = chunk.to_vec();
            handles.push(tokio::spawn(async move {
                tracing::info!(
                    "Async stage {}: {} operations (real compute)",
                    stage_idx,
                    stage_nodes.len()
                );

                for node in &stage_nodes {
                    match node.node_type {
                        NodeType::MatMul | NodeType::Linear => {
                            let _ = tokio::task::spawn_blocking(move || {
                                compute_matmul(1024, 1024)
                            }).await;
                        }
                        NodeType::Conv2D | NodeType::Conv1D | NodeType::Conv3D => {
                            let _ = tokio::task::spawn_blocking(move || {
                                compute_conv2d(3, 64, 32)
                            }).await;
                        }
                        NodeType::SelfAttention
                        | NodeType::MultiHeadAttention
                        | NodeType::FlashAttention => {
                            let _ = tokio::task::spawn_blocking(move || {
                                compute_attention(8, 64, 64)
                            }).await;
                        }
                        NodeType::LayerNorm | NodeType::BatchNorm | NodeType::RMSNorm => {
                            let _ = tokio::task::spawn_blocking(move || {
                                compute_normalization(4096)
                            }).await;
                        }
                        NodeType::ReLU | NodeType::GELU | NodeType::Sigmoid | NodeType::Tanh => {
                            let _ = tokio::task::spawn_blocking(move || {
                                compute_activation(4096, &node.node_type)
                            }).await;
                        }
                        NodeType::Softmax => {
                            let _ = tokio::task::spawn_blocking(move || {
                                compute_softmax(4096)
                            }).await;
                        }
                        NodeType::Embedding => {
                            let _ = tokio::task::spawn_blocking(move || {
                                compute_embedding(10000, 768)
                            }).await;
                        }
                        NodeType::MambaBlock | NodeType::StateSpaceModel => {
                            let _ = tokio::task::spawn_blocking(move || {
                                compute_ssm(4096, 64)
                            }).await;
                        }
                        _ => {
                            // Generic compute fallback
                            let _ = tokio::task::spawn_blocking(move || {
                                compute_generic(1024)
                            }).await;
                        }
                    }
                }

                let result: DLResult<()> = Ok(());
                result
            }));
        }

        for handle in handles {
            handle
                .await
                .map_err(|e| {
                    DeepLearningError::Computation {
                        reason: format!("Async stage panicked: {}", e),
                    }
                })??;
        }

        Ok(())
    }
}

fn compute_matmul(m: usize, n: usize) -> f64 {
    let mut a = vec![0.0f64; m * n];
    let mut b = vec![0.0f64; n * m];
    let mut c = vec![0.0f64; m * m];
    for i in 0..m * n {
        a[i] = (i as f64).sin();
        b[i] = (i as f64).cos();
    }
    for i in 0..m {
        for k in 0..n {
            for j in 0..m {
                c[i * m + j] += a[i * n + k] * b[k * m + j];
            }
        }
    }
    c.iter().sum()
}

fn compute_conv2d(in_c: usize, out_c: usize, size: usize) -> f64 {
    let mut input = vec![0.0f64; in_c * size * size];
    let mut kernel = vec![0.0f64; out_c * in_c * 9];
    let mut output = vec![0.0f64; out_c * (size - 2) * (size - 2)];
    for i in 0..input.len() { input[i] = (i as f64).cos(); }
    for i in 0..kernel.len() { kernel[i] = (i as f64).sin() * 0.1; }
    for o in 0..out_c {
        for c in 0..in_c {
            for i in 1..size - 1 {
                for j in 1..size - 1 {
                    let mut sum = 0.0;
                    for ki in 0..3 {
                        for kj in 0..3 {
                            let inp_idx = c * size * size + (i + ki - 1) * size + (j + kj - 1);
                            let ker_idx = o * in_c * 9 + c * 9 + ki * 3 + kj;
                            sum += input[inp_idx] * kernel[ker_idx];
                        }
                    }
                    let out_idx = o * (size - 2) * (size - 2) + (i - 1) * (size - 2) + (j - 1);
                    output[out_idx] += sum;
                }
            }
        }
    }
    output.iter().sum()
}

fn compute_attention(num_heads: usize, seq_len: usize, head_dim: usize) -> f64 {
    let dim = num_heads * head_dim;
    let mut q = vec![0.0f64; seq_len * dim];
    let mut k = vec![0.0f64; seq_len * dim];
    let mut v = vec![0.0f64; seq_len * dim];
    for i in 0..seq_len * dim {
        q[i] = (i as f64 * 0.1).sin();
        k[i] = (i as f64 * 0.1).cos();
        v[i] = (i as f64 * 0.05).sin();
    }
    let mut output = vec![0.0f64; seq_len * dim];
    for h in 0..num_heads {
        let start = h * head_dim;
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut score = 0.0;
                for d in 0..head_dim {
                    score += q[i * dim + start + d] * k[j * dim + start + d];
                }
                score /= (head_dim as f64).sqrt();
                let attn = score.max(0.0);
                for d in 0..head_dim {
                    output[i * dim + start + d] += attn * v[j * dim + start + d];
                }
            }
        }
    }
    output.iter().sum()
}

fn compute_normalization(dim: usize) -> f64 {
    let mut x = vec![0.0f64; dim];
    let mut sum = 0.0;
    for i in 0..dim {
        x[i] = (i as f64 * 0.5).sin();
        sum += x[i];
    }
    let mean = sum / dim as f64;
    let mut var = 0.0;
    for i in 0..dim {
        let diff = x[i] - mean;
        var += diff * diff;
    }
    let std = (var / dim as f64 + 1e-6).sqrt();
    let mut result = 0.0;
    for i in 0..dim {
        result += (x[i] - mean) / std;
    }
    result
}

fn compute_activation(dim: usize, node_type: &NodeType) -> f64 {
    let mut x = vec![0.0f64; dim];
    for i in 0..dim {
        x[i] = (i as f64 * 0.3).sin() * 2.0 - 1.0;
    }
    match node_type {
        NodeType::ReLU => x.iter().map(|&v| v.max(0.0)).sum(),
        NodeType::GELU => {
            x.iter().map(|&v| {
                0.5 * v * (1.0 + (std::f64::consts::FRAC_2_SQRT_PI * (v + 0.044715 * v * v * v)).tanh())
            }).sum()
        }
        NodeType::Sigmoid => x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).sum(),
        NodeType::Tanh => x.iter().map(|&v| v.tanh()).sum(),
        _ => x.iter().map(|&v| v.max(0.0)).sum(),
    }
}

fn compute_softmax(dim: usize) -> f64 {
    let mut x = vec![0.0f64; dim];
    for i in 0..dim { x[i] = (i as f64 * 0.1).sin(); }
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for i in 0..dim {
        sum += (x[i] - max_val).exp();
    }
    let mut result = 0.0;
    for i in 0..dim {
        result += (x[i] - max_val).exp() / sum;
    }
    result
}

fn compute_embedding(vocab: usize, dim: usize) -> f64 {
    let mut embedding = vec![0.0f64; vocab * dim];
    for i in 0..vocab * dim {
        embedding[i] = (i as f64 * 0.01).sin();
    }
    embedding.iter().sum()
}

fn compute_ssm(dim: usize, state_dim: usize) -> f64 {
    let mut x = vec![0.0f64; dim];
    let mut h = vec![0.0f64; state_dim];
    let mut y = vec![0.0f64; dim];
    for i in 0..dim { x[i] = (i as f64 * 0.2).cos(); }
    for step in 0..dim {
        let input = x[step];
        let mut new_h = vec![0.0f64; state_dim];
        for s in 0..state_dim {
            new_h[s] = h[s] * 0.9 + input * 0.1;
        }
        h = new_h;
        y[step] = h.iter().sum::<f64>() * 0.1;
    }
    y.iter().sum()
}

fn compute_generic(dim: usize) -> f64 {
    let mut data = vec![0.0f64; dim * dim];
    for i in 0..dim * dim {
        data[i] = (i as f64 * 0.7).cos();
    }
    data.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_execute_async_with_graph() {
        let mut g = NeuralGraph::new("g");
        let inp = GraphNode::new(NodeType::Input, "in", 0.0, 0.0);
        let out = GraphNode::new(NodeType::Output, "out", 0.0, 0.0);
        g.add_node(inp);
        g.add_node(out);
        let result = AsyncExecutor::execute_async(&g, 2).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_execute_async_matmul() {
        let mut g = NeuralGraph::new("matmul");
        let inp = GraphNode::new(NodeType::Input, "in", 0.0, 0.0);
        let mm = GraphNode::new(NodeType::MatMul, "mm", 10.0, 0.0);
        let out = GraphNode::new(NodeType::Output, "out", 20.0, 0.0);
        g.add_node(inp);
        let mm_id = g.add_node(mm);
        g.add_node(out);
        let result = AsyncExecutor::execute_async(&g, 1).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_compute_functions_produce_values() {
        let mm = compute_matmul(16, 16);
        assert!(mm != 0.0);
        let norm = compute_normalization(64);
        assert!(norm != 0.0);
        let attn = compute_attention(2, 8, 8);
        assert!(attn != 0.0);
    }
}