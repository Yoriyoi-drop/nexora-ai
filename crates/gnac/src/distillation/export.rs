use crate::canvas::NeuralGraph;
use crate::execution::{ExecutionBackend, GraphIR};
use crate::DLResult;

/// Export hasil distillation ke format deployment
pub struct ExportManager;

impl ExportManager {
    /// Export ke format backend target
    pub fn export(graph: &NeuralGraph, backend: ExecutionBackend) -> DLResult<String> {
        let ir = GraphIR::from_graph(graph, backend);
        let serialized = serde_json::to_string_pretty(&ir.name).map_err(|e| {
            crate::DeepLearningError::Computation {
                reason: e.to_string(),
            }
        })?;
        Ok(serialized)
    }

    /// Verifikasi kompatibilitas target hardware
    pub fn verify_target(graph: &NeuralGraph, target: &str) -> DLResult<Vec<String>> {
        let mut warnings = Vec::new();
        let total_flops = graph.total_flops();
        let total_params = graph.total_params();

        match target {
            "edge_tpu" => {
                if total_params > 10_000_000 {
                    warnings.push(format!(
                        "Model too large for Edge TPU: {} params",
                        total_params
                    ));
                }
            }
            "mobile" => {
                if total_flops > 1_000_000_000 {
                    warnings.push(format!(
                        "Model too compute-intensive for mobile: {} FLOPs",
                        total_flops
                    ));
                }
            }
            "browser" => {
                if total_params > 50_000_000 {
                    warnings.push(format!(
                        "Model too large for browser: {} params",
                        total_params
                    ));
                }
            }
            _ => {}
        }

        Ok(warnings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_empty() {
        let g = NeuralGraph::new("empty");
        let result = ExportManager::export(&g, ExecutionBackend::CPU);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_target_edge_tpu_ok() {
        let g = NeuralGraph::new("small");
        let warnings = ExportManager::verify_target(&g, "edge_tpu").unwrap();
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_verify_target_mobile_ok() {
        let g = NeuralGraph::new("small");
        let warnings = ExportManager::verify_target(&g, "mobile").unwrap();
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_verify_target_browser_ok() {
        let g = NeuralGraph::new("small");
        let warnings = ExportManager::verify_target(&g, "browser").unwrap();
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_verify_target_unknown() {
        let g = NeuralGraph::new("small");
        let warnings = ExportManager::verify_target(&g, "unknown").unwrap();
        assert!(warnings.is_empty());
    }
}
