use crate::gnac::canvas::NeuralGraph;
use crate::gnac::experiment::diff::GraphDiff;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Branchable architecture — fork & merge untuk eksperimen paralel
#[derive(Debug, Clone)]
pub struct ArchitectureBranch {
    pub id: Uuid,
    pub name: String,
    pub parent_id: Option<Uuid>,
    pub graph: NeuralGraph,
    pub created_at: DateTime<Utc>,
    pub status: BranchStatus,
    pub authors: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BranchStatus {
    Active,
    Merged,
    Abandoned,
}

/// Branch Manager untuk eksperimen paralel
pub struct BranchManager {
    pub branches: Vec<ArchitectureBranch>,
}

impl BranchManager {
    pub fn new() -> Self {
        BranchManager { branches: Vec::new() }
    }

    /// Fork dari graf existing
    pub fn fork(&mut self, name: &str, parent_id: Option<Uuid>, graph: NeuralGraph) -> Uuid {
        let branch = ArchitectureBranch {
            id: Uuid::new_v4(),
            name: name.to_string(),
            parent_id,
            graph,
            created_at: Utc::now(),
            status: BranchStatus::Active,
            authors: Vec::new(),
        };
        let id = branch.id;
        self.branches.push(branch);
        id
    }

    /// Merge dua branch
    pub fn merge(&mut self, source_id: &Uuid, target_id: &Uuid) -> Result<GraphDiff, String> {
        let source_idx = self.branches.iter().position(|b| b.id == *source_id)
            .ok_or("Source branch not found")?;
        let target_idx = self.branches.iter().position(|b| b.id == *target_id)
            .ok_or("Target branch not found")?;

        let source = &self.branches[source_idx];
        let target = &self.branches[target_idx];

        let diff = GraphDiff::between(&target.graph, &source.graph);

        // Apply changes dari source ke target
        self.branches[target_idx].graph = source.graph.clone();
        self.branches[source_idx].status = BranchStatus::Merged;

        Ok(diff)
    }
}
