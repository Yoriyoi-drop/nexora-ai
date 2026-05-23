use crate::canvas::NeuralGraph;
use crate::distillation::DistillationConfig;
use crate::DLResult;

/// Teacher-Student Knowledge Distillation
pub struct DistillationEngine;

impl DistillationEngine {
    /// Compress teacher graph to student graph
    pub fn compress(teacher: &NeuralGraph, config: &DistillationConfig) -> DLResult<NeuralGraph> {
        let mut student = NeuralGraph::new(&format!("{}_student", teacher.name));

        // Student architecture: lebih kecil dari teacher
        let teacher_params = teacher.total_params();
        let student_params = teacher_params / (config.student_width * config.student_depth).max(1);

        tracing::info!(
            "Distillation: teacher {} params -> student ~{} params (T={}, α={})",
            teacher_params,
            student_params,
            config.temperature,
            config.alpha
        );

        // Copy input & output nodes
        for input_node in teacher.get_input_nodes() {
            student.add_node(input_node.clone());
        }
        for output_node in teacher.get_output_nodes() {
            student.add_node(output_node.clone());
        }

        Ok(student)
    }

    /// Hitung distillation loss via KL divergence
    pub fn distillation_loss(
        teacher_logits: &[f64],
        student_logits: &[f64],
        temperature: f64,
    ) -> f64 {
        let soft_teacher: Vec<f64> = teacher_logits
            .iter()
            .map(|&x| (x / temperature).exp())
            .collect();
        let soft_student: Vec<f64> = student_logits
            .iter()
            .map(|&x| (x / temperature).exp())
            .collect();

        let sum_t: f64 = soft_teacher.iter().sum();
        let sum_s: f64 = soft_student.iter().sum();

        let kl_div: f64 = soft_teacher
            .iter()
            .zip(soft_student.iter())
            .map(|(t, s)| {
                let p = t / sum_t;
                let q = s / sum_s;
                if q > 0.0 {
                    p * (p / q).ln()
                } else {
                    0.0
                }
            })
            .sum();

        kl_div * temperature.powi(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::GraphNode;
    use crate::NodeType;

    #[test]
    fn test_compress_empty() {
        let teacher = NeuralGraph::new("teacher");
        let config = DistillationConfig {
            temperature: 2.0,
            alpha: 0.5,
            student_depth: 2,
            student_width: 2,
            target_hardware: "cpu".to_string(),
        };
        let student = DistillationEngine::compress(&teacher, &config).unwrap();
        assert_eq!(student.name, "teacher_student");
    }

    #[test]
    fn test_compress_with_io() {
        let mut teacher = NeuralGraph::new("teacher");
        teacher.add_node(GraphNode::new(NodeType::Input, "in", 0.0, 0.0));
        teacher.add_node(GraphNode::new(NodeType::Output, "out", 0.0, 0.0));
        let config = DistillationConfig {
            temperature: 2.0,
            alpha: 0.5,
            student_depth: 2,
            student_width: 2,
            target_hardware: "cpu".to_string(),
        };
        let student = DistillationEngine::compress(&teacher, &config).unwrap();
        assert_eq!(student.node_count(), 2);
    }

    #[test]
    fn test_distillation_loss_identical() {
        let logits = vec![1.0, 2.0, 3.0];
        let loss = DistillationEngine::distillation_loss(&logits, &logits, 1.0);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_distillation_loss_different() {
        let teacher = vec![10.0, 20.0, 30.0];
        let student = vec![1.0, 2.0, 3.0];
        let loss = DistillationEngine::distillation_loss(&teacher, &student, 2.0);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_distillation_loss_zero_teacher() {
        let teacher = vec![0.0, 0.0, 0.0];
        let student = vec![1.0, 2.0, 3.0];
        let loss = DistillationEngine::distillation_loss(&teacher, &student, 1.0);
        assert!(loss.is_finite());
    }
}
