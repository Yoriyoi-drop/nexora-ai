//! Trie - Rust implementation
//! 
//! Trie data structure for efficient token sequence lookup

use std::collections::HashMap;
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// Trie node for token sequences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrieNode {
    pub token_id: Option<u32>,
    pub result_id: Option<u32>,
    pub is_leaf: bool,
    pub children: HashMap<u32, TrieNode>,
    pub frequency: u64,
    pub priority: f32,
}

impl TrieNode {
    /// Create a new trie node
    pub fn new() -> Self {
        Self {
            token_id: None,
            result_id: None,
            is_leaf: false,
            children: HashMap::new(),
            frequency: 0,
            priority: 0.0,
        }
    }
    
    /// Create a new trie node with token ID
    pub fn with_token(token_id: u32) -> Self {
        Self {
            token_id: Some(token_id),
            result_id: None,
            is_leaf: false,
            children: HashMap::new(),
            frequency: 0,
            priority: 0.0,
        }
    }
    
    /// Create a leaf node with result
    pub fn leaf(token_id: u32, result_id: u32, frequency: u64, priority: f32) -> Self {
        Self {
            token_id: Some(token_id),
            result_id: Some(result_id),
            is_leaf: true,
            children: HashMap::new(),
            frequency,
            priority,
        }
    }
    
    /// Add a child node
    pub fn add_child(&mut self, token_id: u32, node: TrieNode) {
        self.children.insert(token_id, node);
    }
    
    /// Get child by token ID
    pub fn get_child(&self, token_id: u32) -> Option<&TrieNode> {
        self.children.get(&token_id)
    }
    
    /// Get mutable child by token ID
    pub fn get_child_mut(&mut self, token_id: u32) -> Option<&mut TrieNode> {
        self.children.get_mut(&token_id)
    }
    
    /// Check if has child with token ID
    pub fn has_child(&self, token_id: u32) -> bool {
        self.children.contains_key(&token_id)
    }
    
    /// Get all child token IDs
    pub fn child_token_ids(&self) -> Vec<u32> {
        self.children.keys().copied().collect()
    }
    
    /// Get number of children
    pub fn child_count(&self) -> usize {
        self.children.len()
    }
    
    /// Remove child
    pub fn remove_child(&mut self, token_id: u32) -> Option<TrieNode> {
        self.children.remove(&token_id)
    }
    
    /// Clear all children
    pub fn clear_children(&mut self) {
        self.children.clear();
    }
    
    /// Update frequency
    pub fn update_frequency(&mut self, frequency: u64) {
        self.frequency = frequency;
    }
    
    /// Update priority
    pub fn update_priority(&mut self, priority: f32) {
        self.priority = priority;
    }
    
    /// Set as leaf
    pub fn set_leaf(&mut self, result_id: u32, frequency: u64, priority: f32) {
        self.is_leaf = true;
        self.result_id = Some(result_id);
        self.frequency = frequency;
        self.priority = priority;
    }
    
    /// Unset leaf
    pub fn unset_leaf(&mut self) {
        self.is_leaf = false;
        self.result_id = None;
    }
}

impl Default for TrieNode {
    fn default() -> Self {
        Self::new()
    }
}

/// Trie data structure for efficient token sequence lookup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trie {
    root: TrieNode,
    size: usize,
    max_depth: usize,
}

impl Trie {
    /// Create a new trie
    pub fn new() -> Self {
        Self {
            root: TrieNode::new(),
            size: 0,
            max_depth: 0,
        }
    }
    
    /// Insert a token sequence with result
    pub fn insert(&mut self, token_ids: &[u32], result_id: u32, frequency: u64, priority: f32) -> Result<()> {
        if token_ids.is_empty() {
            return Err(anyhow::anyhow!("Cannot insert empty token sequence"));
        }
        
        let mut current = &mut self.root;
        let mut depth = 0;
        
        // Navigate through the trie
        for &token_id in token_ids {
            depth += 1;
            
            if !current.has_child(token_id) {
                let node = TrieNode::with_token(token_id);
                current.add_child(token_id, node);
                self.size += 1;
            }
            
            current = current.get_child_mut(token_id).expect("child was just inserted");
        }
        
        // Set as leaf with result
        current.set_leaf(result_id, frequency, priority);
        self.max_depth = self.max_depth.max(depth);
        
        Ok(())
    }
    
    /// Insert merge rule (convenience method)
    pub fn insert_merge(&mut self, token_ids: &[u32], result_id: u32) -> Result<()> {
        self.insert(token_ids, result_id, 1, 1.0)
    }
    
    /// Lookup a token sequence
    pub fn lookup(&self, token_ids: &[u32]) -> Option<&TrieNode> {
        if token_ids.is_empty() {
            return None;
        }
        
        let mut current = &self.root;
        
        for &token_id in token_ids {
            if let Some(child) = current.get_child(token_id) {
                current = child;
            } else {
                return None;
            }
        }
        
        Some(current)
    }
    
    /// Lookup result ID for token sequence
    pub fn lookup_result(&self, token_ids: &[u32]) -> Option<u32> {
        self.lookup(token_ids).and_then(|node| node.result_id)
    }
    
    /// Check if token sequence exists
    pub fn contains(&self, token_ids: &[u32]) -> bool {
        self.lookup(token_ids).is_some()
    }
    
    /// Find all sequences starting with prefix
    pub fn find_prefixes(&self, prefix: &[u32]) -> Vec<Vec<u32>> {
        let mut results = Vec::new();
        
        if let Some(node) = self.lookup(prefix) {
            self.collect_sequences(node, prefix, &mut results);
        }
        
        results
    }
    
    /// Collect all sequences from a node
    fn collect_sequences(&self, node: &TrieNode, prefix: &[u32], results: &mut Vec<Vec<u32>>) {
        if node.is_leaf {
            results.push(prefix.to_vec());
        }
        
        for (&token_id, child) in &node.children {
            let mut new_prefix = prefix.to_vec();
            new_prefix.push(token_id);
            self.collect_sequences(child, &new_prefix, results);
        }
    }
    
    /// Find the longest matching sequence
    pub fn longest_match(&self, token_ids: &[u32]) -> Option<(Vec<u32>, u32)> {
        let mut current = &self.root;
        let mut best_match: Option<(Vec<u32>, u32)> = None;
        let mut current_sequence = Vec::new();
        
        for &token_id in token_ids {
            if let Some(child) = current.get_child(token_id) {
                current_sequence.push(token_id);
                
                if child.is_leaf {
                    if let Some(result_id) = child.result_id {
                        best_match = Some((current_sequence.clone(), result_id));
                    }
                }
                
                current = child;
            } else {
                break;
            }
        }
        
        best_match
    }
    
    /// Get all sequences in the trie
    pub fn all_sequences(&self) -> Vec<Vec<u32>> {
        let mut results = Vec::new();
        self.collect_sequences(&self.root, &[], &mut results);
        results
    }
    
    /// Remove a token sequence
    pub fn remove(&mut self, token_ids: &[u32]) -> Result<bool> {
        if token_ids.is_empty() {
            return Ok(false);
        }
        
        // Use a helper function to avoid borrowing issues
        let removed;
        {
            fn remove_recursive(node: &mut TrieNode, token_ids: &[u32], index: usize) -> bool {
                if index == token_ids.len() {
                    if node.is_leaf {
                        node.unset_leaf();
                        return true;
                    }
                    return false;
                }
                
                let token_id = token_ids[index];
                if let Some(child) = node.children.get_mut(&token_id) {
                    if remove_recursive(child, token_ids, index + 1) {
                        // If child is now empty and not a leaf, remove it
                        if child.children.is_empty() && !child.is_leaf {
                            node.children.remove(&token_id);
                        }
                        return true;
                    }
                }
                
                false
            }
            
            removed = remove_recursive(&mut self.root, token_ids, 0);
        }
        
        if removed {
            self.cleanup_after_removal(token_ids);
        }
        
        Ok(removed)
    }
    
    /// Clean up empty nodes after removal
    fn cleanup_after_removal(&mut self, _token_ids: &[u32]) {
        // This function can be used for additional cleanup if needed
        // For now, the recursive removal handles cleanup
    }
    
    /// Clean up empty nodes after removal (simplified version)
    fn _cleanup_empty_nodes(&mut self, _path: &mut Vec<(&mut TrieNode, Option<u32>)>, _token_ids: &[u32]) {
        // Simplified cleanup - the recursive removal already handles most cleanup
        // In a more complex implementation, you would iterate through the path
        // and remove empty nodes while preserving the trie structure
    }
    
    /// Clear the trie
    pub fn clear(&mut self) {
        self.root = TrieNode::new();
        self.size = 0;
        self.max_depth = 0;
    }
    
    /// Get trie size (number of nodes)
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Get maximum depth
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }
    
    /// Check if trie is empty
    pub fn is_empty(&self) -> bool {
        self.root.children.is_empty()
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> TrieStats {
        let mut stats = TrieStats::default();
        self.calculate_stats(&self.root, &mut stats, 0);
        stats
    }
    
    /// Calculate statistics recursively
    fn calculate_stats(&self, node: &TrieNode, stats: &mut TrieStats, depth: usize) {
        stats.node_count += 1;
        stats.max_depth = stats.max_depth.max(depth);
        
        if node.is_leaf {
            stats.leaf_count += 1;
            if let Some(_result_id) = node.result_id {
                stats.result_count += 1;
                stats.total_frequency += node.frequency;
                stats.average_priority = (stats.average_priority * (stats.result_count - 1) as f32 + node.priority) / stats.result_count as f32;
            }
        }
        
        for child in node.children.values() {
            self.calculate_stats(child, stats, depth + 1);
        }
    }
    
    /// Validate trie structure
    pub fn validate(&self) -> Result<()> {
        self.validate_node(&self.root, &mut std::collections::HashSet::new());
        Ok(())
    }
    
    /// Validate node recursively
    fn validate_node(&self, node: &TrieNode, visited: &mut std::collections::HashSet<*const TrieNode>) {
        let node_ptr = node as *const TrieNode;
        
        if visited.contains(&node_ptr) {
            panic!("Cycle detected in trie");
        }
        
        visited.insert(node_ptr);
        
        for child in node.children.values() {
            self.validate_node(child, visited);
        }
    }
}

/// Trie statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrieStats {
    pub node_count: usize,
    pub leaf_count: usize,
    pub result_count: usize,
    pub max_depth: usize,
    pub total_frequency: u64,
    pub average_priority: f32,
}

impl Default for TrieStats {
    fn default() -> Self {
        Self {
            node_count: 0,
            leaf_count: 0,
            result_count: 0,
            max_depth: 0,
            total_frequency: 0,
            average_priority: 0.0,
        }
    }
}

impl Default for Trie {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions
pub fn create_trie() -> Trie {
    Trie::new()
}

pub fn lookup_sequence(trie: &Trie, token_ids: &[u32]) -> Option<u32> {
    trie.lookup_result(token_ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_trie_insert_lookup() {
        let mut trie = Trie::new();
        let sequence = vec![1, 2, 3];
        let result_id = 42;
        
        trie.insert(&sequence, result_id, 1, 1.0).unwrap();
        
        assert!(trie.contains(&sequence));
        assert_eq!(trie.lookup_result(&sequence), Some(result_id));
    }
    
    #[test]
    fn test_trie_longest_match() {
        let mut trie = Trie::new();
        
        trie.insert(&[1, 2, 3], 100, 1, 1.0).unwrap();
        trie.insert(&[1, 2], 200, 1, 0.9).unwrap();
        trie.insert(&[1], 300, 1, 0.8).unwrap();
        
        let tokens = vec![1, 2, 3, 4];
        let (matched_sequence, result_id) = trie.longest_match(&tokens).unwrap();
        
        assert_eq!(matched_sequence, vec![1, 2, 3]);
        assert_eq!(result_id, 100);
    }
    
    #[test]
    fn test_trie_prefixes() {
        let mut trie = Trie::new();
        
        trie.insert(&[1, 2, 3], 100, 1, 1.0).unwrap();
        trie.insert(&[1, 2, 4], 200, 1, 1.0).unwrap();
        trie.insert(&[1, 5], 300, 1, 1.0).unwrap();
        
        let prefixes = trie.find_prefixes(&[1, 2]);
        
        assert_eq!(prefixes.len(), 2);
        assert!(prefixes.contains(&vec![1, 2, 3]));
        assert!(prefixes.contains(&vec![1, 2, 4]));
    }
    
    #[test]
    fn test_trie_remove() {
        let mut trie = Trie::new();
        
        trie.insert(&[1, 2, 3], 100, 1, 1.0).unwrap();
        trie.insert(&[1, 2, 4], 200, 1, 1.0).unwrap();
        
        assert!(trie.remove(&[1, 2, 3]).unwrap());
        assert!(!trie.contains(&[1, 2, 3]));
        assert!(trie.contains(&[1, 2, 4]));
        
        assert!(!trie.remove(&[1, 2, 3]).unwrap()); // Already removed
    }
    
    #[test]
    fn test_trie_stats() {
        let mut trie = Trie::new();
        
        trie.insert(&[1, 2, 3], 100, 5, 1.0).unwrap();
        trie.insert(&[1, 2, 4], 200, 3, 0.8).unwrap();
        trie.insert(&[1, 5], 300, 2, 0.6).unwrap();
        
        let stats = trie.get_stats();
        
        assert_eq!(stats.leaf_count, 3);
        assert_eq!(stats.result_count, 3);
        assert_eq!(stats.total_frequency, 10);
        assert!(stats.max_depth > 0);
    }
    
    #[test]
    fn test_trie_validation() {
        let trie = Trie::new();
        assert!(trie.validate().is_ok());
        
        let mut trie = Trie::new();
        trie.insert(&[1, 2, 3], 100, 1, 1.0).unwrap();
        assert!(trie.validate().is_ok());
    }
    
    #[test]
    fn test_trie_clear() {
        let mut trie = Trie::new();
        
        trie.insert(&[1, 2, 3], 100, 1, 1.0).unwrap();
        assert!(!trie.is_empty());
        
        trie.clear();
        assert!(trie.is_empty());
        assert_eq!(trie.size(), 0);
    }
    
    #[test]
    fn test_trie_serialization() {
        let mut trie = Trie::new();
        
        trie.insert(&[1, 2, 3], 100, 5, 1.0).unwrap();
        trie.insert(&[1, 2, 4], 200, 3, 0.8).unwrap();
        
        let json = serde_json::to_string_pretty(&trie).unwrap();
        let loaded: Trie = serde_json::from_str(&json).unwrap();
        
        assert_eq!(loaded.size(), trie.size());
        assert!(loaded.contains(&[1, 2, 3]));
        assert!(loaded.contains(&[1, 2, 4]));
    }
}
