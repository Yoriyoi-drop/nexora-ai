use crate::DLResult;
use crate::gnac::canvas::NeuralGraph;
use crate::gnac::distillation::DistillationConfig;

/// Teacher-Student Knowledge Distillation
pub struct DistillationEngine;

impl DistillationEngine {
    /// Compress teacher graph to student graph
    pub fn compress(
        teacher: &NeuralGraph,
        config: &DistillationConfig,
    ) -> DLResult<NeuralGraph> {
        let mut student = NeuralGraph::new(&format!("{}_student", teacher.name));

        // Student architecture: lebih kecil dari teacher
        let teacher_params = teacher.total_params();
        let student_params = teacher_params / (config.student_width * config.student_depth).max(1);

        log::info!(
            "Distillation: teacher {} params -> student ~{} params (T={}, α={})",
            teacher_params, student_params, config.temperature, config.alpha
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

    /// Hitung distillation loss (simulated)
    pub fn distillation_loss(
        teacher_logits: &[f64],
        student_logits: &[f64],
        temperature: f64,
    ) -> f64 {
        let soft_teacher: Vec<f64> = teacher_logits.iter()
            .map(|&x| (x / temperature).exp())
            .collect();
        let soft_student: Vec<f64> = student_logits.iter()
            .map(|&x| (x / temperature).exp())
            .collect();

        let sum_t: f64 = soft_teacher.iter().sum();
        let sum_s: f64 = soft_student.iter().sum();

        let kl_div: f64 = soft_teacher.iter()
            .zip(soft_student.iter())
            .map(|(t, s)| {
                let p = t / sum_t;
                let q = s / sum_s;
                if q > 0.0 { p * (p / q).ln() } else { 0.0 }
            })
            .sum();

        kl_div * temperature.powi(2)
    }
}
