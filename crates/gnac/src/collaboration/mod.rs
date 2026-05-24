//! Collaborative Neural Workspace
//!
//! Mendukung kolaborasi multi-user real-time: live editing,
//! branchable architecture, node commenting, experiment forking,
//! dan collaborative debugging.

pub mod branching;
pub mod live_editing;

pub use branching::*;
pub use live_editing::*;

use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Operasi kolaboratif
#[derive(Debug, Clone)]
pub struct CollaboratorAction {
    pub user_id: Uuid,
    pub action_type: CollabActionType,
    pub timestamp: DateTime<Utc>,
    pub graph_id: Uuid,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CollabActionType {
    NodeAdded,
    NodeRemoved,
    NodeModified,
    EdgeAdded,
    EdgeRemoved,
    Fork,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collab_action_type_debug() {
        let t = CollabActionType::Fork;
        assert!(!format!("{:?}", t).is_empty());
    }

    #[test]
    fn test_collaborator_action() {
        let action = CollaboratorAction {
            user_id: Uuid::new_v4(),
            action_type: CollabActionType::NodeAdded,
            timestamp: Utc::now(),
            graph_id: Uuid::new_v4(),
            description: "Added conv layer".to_string(),
        };
        assert_eq!(action.action_type, CollabActionType::NodeAdded);
        assert_eq!(action.description, "Added conv layer");
    }
}
