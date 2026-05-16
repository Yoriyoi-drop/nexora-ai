//! Collaborative Neural Workspace
//!
//! Mendukung kolaborasi multi-user real-time: live editing,
//! branchable architecture, node commenting, experiment forking,
//! dan collaborative debugging.

pub mod live_editing;
pub mod branching;

pub use live_editing::*;
pub use branching::*;

use crate::DLResult;
use crate::gnac::canvas::NeuralGraph;
use uuid::Uuid;
use chrono::{DateTime, Utc};

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
    CommentAdded,
    Fork,
    Merge,
}
