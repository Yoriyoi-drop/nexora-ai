use uuid::Uuid;

/// Recurrent Loop Node — loop dengan kondisi terminasi
#[derive(Debug, Clone)]
pub struct RecurrentLoopNode {
    pub id: Uuid,
    pub name: String,
    pub max_iterations: usize,
    pub current_iteration: usize,
    pub convergence_threshold: f64,
    pub body_graph_id: Option<Uuid>,
}

impl RecurrentLoopNode {
    pub fn new(name: &str, max_iterations: usize, convergence_threshold: f64) -> Self {
        RecurrentLoopNode {
            id: Uuid::new_v4(),
            name: name.to_string(),
            max_iterations,
            current_iteration: 0,
            convergence_threshold,
            body_graph_id: None,
        }
    }

    /// Cek apakah loop harus dilanjutkan
    pub fn should_continue(&self, loss: f64) -> bool {
        self.current_iteration < self.max_iterations && loss > self.convergence_threshold
    }

    /// Increment iterasi
    pub fn step(&mut self) {
        self.current_iteration += 1;
    }

    /// Reset loop state
    pub fn reset(&mut self) {
        self.current_iteration = 0;
    }
}
