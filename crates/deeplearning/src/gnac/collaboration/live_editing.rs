use crate::gnac::canvas::NeuralGraph;
use crate::gnac::collaboration::{CollabActionType, CollaboratorAction};
use uuid::Uuid;

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
            CollabActionType::NodeAdded
            | CollabActionType::NodeRemoved
            | CollabActionType::NodeModified
            | CollabActionType::EdgeAdded
            | CollabActionType::EdgeRemoved => {
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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn action(user: Uuid) -> CollaboratorAction {
        CollaboratorAction {
            user_id: user,
            action_type: CollabActionType::NodeAdded,
            timestamp: Utc::now(),
            graph_id: Uuid::new_v4(),
            description: "test".to_string(),
        }
    }

    #[test]
    fn test_live_editing_new() {
        let m = LiveEditingManager::new();
        assert!(m.active_users.is_empty());
        assert!(m.actions.is_empty());
    }

    #[test]
    fn test_join_leave() {
        let mut m = LiveEditingManager::new();
        let u = Uuid::new_v4();
        m.join(u);
        assert_eq!(m.active_users.len(), 1);
        m.join(u); // duplicate
        assert_eq!(m.active_users.len(), 1);
        m.leave(&u);
        assert!(m.active_users.is_empty());
    }

    #[test]
    fn test_apply_increments_version() {
        let mut m = LiveEditingManager::new();
        let mut g = NeuralGraph::new("g");
        let old_ver = g.version;
        m.apply(&mut g, action(Uuid::new_v4()));
        assert_eq!(g.version, old_ver + 1);
    }

    #[test]
    fn test_history() {
        let mut m = LiveEditingManager::new();
        let u = Uuid::new_v4();
        m.apply(&mut NeuralGraph::new("g"), action(u));
        assert_eq!(m.history().len(), 1);
    }
}
