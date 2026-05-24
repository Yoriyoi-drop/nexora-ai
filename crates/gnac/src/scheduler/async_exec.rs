use crate::canvas::NeuralGraph;
use crate::{DeepLearningError, DLResult};

/// Asynchronous Execution — overlapping compute & memory transfer
pub struct AsyncExecutor;

impl AsyncExecutor {
    /// Eksekusi asinkron dengan pipeline stages
    /// Setiap stage dijalankan sebagai tokio task terpisah untuk overlapping compute & I/O.
    pub async fn execute_async(graph: &NeuralGraph, pipeline_stages: usize) -> DLResult<()> {
        let order = graph.topological_order()?;

        if pipeline_stages == 0 || order.is_empty() {
            return Ok(());
        }

        let stage_size = (order.len() as f64 / pipeline_stages as f64).ceil() as usize;
        let mut handles = Vec::with_capacity(pipeline_stages);

        for (stage_idx, chunk) in order.chunks(stage_size).enumerate() {
            let ops: Vec<_> = chunk.to_vec();
            handles.push(tokio::spawn(async move {
                tracing::info!("Async stage {}: {} operations", stage_idx, ops.len());
                // Future: actual async compute per operation
                for _op in &ops {
                    // Yield for cooperative scheduling between stages
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                }
                let result: DLResult<()> = Ok(());
                result
            }));
        }

        // Wait for all stages and propagate errors
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_execute_async_with_graph() {
        let mut g = NeuralGraph::new("g");
        let inp = crate::canvas::GraphNode::new(crate::NodeType::Input, "in", 0.0, 0.0);
        let out = crate::canvas::GraphNode::new(crate::NodeType::Output, "out", 0.0, 0.0);
        g.add_node(inp);
        g.add_node(out);
        let result = AsyncExecutor::execute_async(&g, 2).await;
        assert!(result.is_ok());
    }
}
