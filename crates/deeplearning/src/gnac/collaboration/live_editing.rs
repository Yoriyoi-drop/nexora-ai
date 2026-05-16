use crate::gnac::canvas::NeuralGraph;
use crate::gnac::collaboration::{CollaboratorAction, CollabActionType};
use uuid::Uuid;
use chrono::Utc;

/// Manajer live editing multi-user
pub struct LiveEditingManager {
    pub actions: Vec<CollaboratorAction>,
    pub active_users: Vec<Uuid>,
}

impl LiveEditingManager {
    pub fn new() -> Self {
        LiveEditingManager {
            actions: Vec::new(),
            active_users: Vec::new(),
        }
    }

    /// Apply operasi dari user ke graf
    pub fn apply(&mut self, graph: &mut NeuralGraph, action: CollaboratorAction) {
        match action.action_type {
            CollabActionType::NodeAdded | CollabActionType::NodeRemoved |
            CollabActionType::NodeModified | CollabActionType::EdgeAdded |
            CollabActionType::EdgeRemoved => {
                graph.version += 1;
            }
            _ => {}
        }
        self.actions.push(action);
    }

    /// Join user ke sesi
    pub fn join(&mut self, user_id: Uuid) {
        if !self.active_users.contains(&user_id) {
            self.active_users.push(user_id);
        }
    }

    /// Leave sesi
    pub fn leave(&mut self, user_id: &Uuid) {
        self.active_users.retain(|u| u != user_id);
    }

    /// Get action history
    pub fn history(&self) -> &[CollaboratorAction] {
        &self.actions
    }
}
