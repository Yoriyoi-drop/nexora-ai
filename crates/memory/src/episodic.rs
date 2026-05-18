//! Episodic Memory - Memory untuk pengalaman spesifik
//! 
//! Implementasi episodic memory dengan temporal indexing dan importance scoring

use std::collections::{HashMap, BTreeMap};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use anyhow::Result;
use tracing::{debug, trace};

/// Episode dalam episodic memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEpisode {
    pub id: String,
    pub timestamp: u64,
    pub content: String,
    pub context: String,
    pub importance: f32,  // 0.0 - 1.0
    pub tags: Vec<String>,
    pub related_episodes: Vec<String>, // IDs of related episodes
    pub embedding: Option<Vec<f32>>, // Semantic embedding for similarity search
}

impl MemoryEpisode {
    pub fn new(content: String, context: String) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
            
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: now,
            content,
            context,
            importance: 0.5,
            tags: Vec::new(),
            related_episodes: Vec::new(),
            embedding: None,
        }
    }
    
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }
    
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
    
    pub fn with_id(mut self, id: String) -> Self {
        self.id = id;
        self
    }
    
    pub fn with_timestamp(mut self, timestamp: u64) -> Self {
        self.timestamp = timestamp;
        self
    }
    
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
    
    pub fn add_related_episode(&mut self, episode_id: String) {
        if !self.related_episodes.contains(&episode_id) {
            self.related_episodes.push(episode_id);
        }
    }
    
    pub fn add_tag(&mut self, tag: String) {
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }
    
    /// Calculate similarity dengan episode lain berdasarkan content
    pub fn calculate_similarity(&self, other: &MemoryEpisode) -> f32 {
        // Simple text similarity (in real implementation, use embeddings)
        let self_words: std::collections::HashSet<&str> = self.content.split_whitespace().collect();
        let other_words: std::collections::HashSet<&str> = other.content.split_whitespace().collect();
        
        if self_words.is_empty() || other_words.is_empty() {
            return 0.0;
        }
        
        let intersection = self_words.intersection(&other_words).count();
        let union = self_words.union(&other_words).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }
}

/// Episodic memory system
#[derive(Debug)]
pub struct EpisodicMemory {
    episodes: HashMap<String, MemoryEpisode>,
    temporal_index: BTreeMap<u64, Vec<String>>, // timestamp -> episode IDs
    tag_index: HashMap<String, Vec<String>>,    // tag -> episode IDs
    max_episodes: usize,
    similarity_threshold: f32,
}

impl EpisodicMemory {
    pub fn new(max_episodes: usize) -> Self {
        Self {
            episodes: HashMap::new(),
            temporal_index: BTreeMap::new(),
            tag_index: HashMap::new(),
            max_episodes,
            similarity_threshold: 0.3,
        }
    }
    
    /// Add episode ke memory
    pub async fn add_episode(&mut self, key: &str, value: &str) -> Result<String> {
        trace!("Adding episode: {} = {}", key, value);
        
        let episode = MemoryEpisode::new(value.to_string(), key.to_string());
        let episode_id = episode.id.clone();
        
        self.add_episode_internal(episode).await?;
        
        Ok(episode_id)
    }
    
    /// Add episode dengan metadata lengkap
    pub async fn add_episode_with_metadata(&mut self, episode: MemoryEpisode) -> Result<String> {
        let episode_id = episode.id.clone();
        self.add_episode_internal(episode).await?;
        Ok(episode_id)
    }
    
    async fn add_episode_internal(&mut self, episode: MemoryEpisode) -> Result<()> {
        let episode_id = episode.id.clone();
        
        // Check capacity
        if self.episodes.len() >= self.max_episodes {
            self.evict_least_important().await?;
        }
        
        // Add to main storage
        self.episodes.insert(episode_id.clone(), episode.clone());
        
        // Update temporal index
        self.temporal_index
            .entry(episode.timestamp)
            .or_insert_with(Vec::new)
            .push(episode_id.clone());
        
        // Update tag index
        for tag in &episode.tags {
            self.tag_index
                .entry(tag.clone())
                .or_insert_with(Vec::new)
                .push(episode_id.clone());
        }
        
        // Find and link similar episodes
        self.find_similar_episodes(&episode_id).await?;
        
        debug!("Added episode: {} (importance: {:.2})", episode_id, episode.importance);
        Ok(())
    }
    
    /// Get episode by ID
    pub async fn get_episode(&self, episode_id: &str) -> Result<Option<&MemoryEpisode>> {
        Ok(self.episodes.get(episode_id))
    }
    
    /// Search episodes berdasarkan query
    pub async fn search(&self, query: &str) -> Result<Option<Vec<crate::MemorySearchResult>>> {
        trace!("Searching episodes: {}", query);
        
        let mut results = Vec::with_capacity(self.episodes.len());
        let query_lower = query.to_lowercase();
        
        for episode in self.episodes.values() {
            let mut relevance = 0.0;
            
            // Content matching
            if episode.content.to_lowercase().contains(&query_lower) {
                relevance += 0.7;
            }
            
            // Context matching
            if episode.context.to_lowercase().contains(&query_lower) {
                relevance += 0.5;
            }
            
            // Tag matching
            for tag in &episode.tags {
                if tag.to_lowercase().contains(&query_lower) {
                    relevance += 0.3;
                }
            }
            
            // Boost by importance
            relevance *= 1.0 + episode.importance;
            
            if relevance > 0.1 {
                results.push(crate::MemorySearchResult {
                    layer: crate::MemoryLayer::Session, // Episodic memory maps to session layer
                    key: episode.id.clone(),
                    value: episode.content.clone(),
                    relevance_score: relevance,
                    timestamp: episode.timestamp,
                });
            }
        }
        
        // Sort by relevance
        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(Some(results))
    }
    
    /// Get episodes dalam time range
    pub async fn get_episodes_in_range(&self, start_time: u64, end_time: u64) -> Result<Vec<&MemoryEpisode>> {
        let mut episodes = Vec::new();
        
        for (_timestamp, episode_ids) in self.temporal_index.range(start_time..=end_time) {
            for episode_id in episode_ids {
                if let Some(episode) = self.episodes.get(episode_id) {
                    episodes.push(episode);
                }
            }
        }
        
        Ok(episodes)
    }
    
    /// Get episodes by tag
    pub async fn get_episodes_by_tag(&self, tag: &str) -> Result<Vec<&MemoryEpisode>> {
        let mut episodes = Vec::new();
        
        if let Some(episode_ids) = self.tag_index.get(tag) {
            for episode_id in episode_ids {
                if let Some(episode) = self.episodes.get(episode_id) {
                    episodes.push(episode);
                }
            }
        }
        
        Ok(episodes)
    }
    
    /// Get related episodes
    pub async fn get_related_episodes(&self, episode_id: &str) -> Result<Vec<&MemoryEpisode>> {
        let mut related = Vec::new();
        
        if let Some(episode) = self.episodes.get(episode_id) {
            related = Vec::with_capacity(episode.related_episodes.len());
            for related_id in &episode.related_episodes {
                if let Some(related_episode) = self.episodes.get(related_id) {
                    related.push(related_episode);
                }
            }
        }
        
        Ok(related)
    }
    
    /// Update episode importance
    pub async fn update_importance(&mut self, episode_id: &str, new_importance: f32) -> Result<bool> {
        if let Some(episode) = self.episodes.get_mut(episode_id) {
            episode.importance = new_importance.clamp(0.0, 1.0);
            debug!("Updated importance for {}: {:.2}", episode_id, episode.importance);
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    /// Delete episode
    pub async fn delete_episode(&mut self, episode_id: &str) -> Result<bool> {
        if let Some(episode) = self.episodes.remove(episode_id) {
            // Remove from temporal index
            if let Some(episode_ids) = self.temporal_index.get_mut(&episode.timestamp) {
                episode_ids.retain(|id| id != episode_id);
                if episode_ids.is_empty() {
                    self.temporal_index.remove(&episode.timestamp);
                }
            }
            
            // Remove from tag index
            for tag in episode.tags {
                if let Some(tag_episodes) = self.tag_index.get_mut(&tag) {
                    tag_episodes.retain(|id| id != episode_id);
                    if tag_episodes.is_empty() {
                        self.tag_index.remove(&tag);
                    }
                }
            }
            
            // Remove from related episodes
            for related_id in episode.related_episodes {
                if let Some(related_episode) = self.episodes.get_mut(&related_id) {
                    related_episode.related_episodes.retain(|id| id != episode_id);
                }
            }
            
            debug!("Deleted episode: {}", episode_id);
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    /// Get count of episodes
    pub async fn count(&self) -> usize {
        self.episodes.len()
    }
    
    /// Get all episodes
    pub async fn get_all_episodes(&self) -> Result<Vec<&MemoryEpisode>> {
        Ok(self.episodes.values().collect())
    }
    
    /// Search episodes by query
    pub async fn search_episodes(&self, query: &str) -> Result<Vec<crate::MemorySearchResult>> {
        if let Some(results) = self.search(query).await? {
            Ok(results)
        } else {
            Ok(Vec::new())
        }
    }
    
    /// Get episodes by time range
    pub async fn get_episodes_by_time_range(&self, start_time: u64, end_time: u64) -> Result<Vec<&MemoryEpisode>> {
        self.get_episodes_in_range(start_time, end_time).await
    }
    
    /// Update episode
    pub async fn update_episode(&mut self, episode: MemoryEpisode) -> Result<bool> {
        let episode_id = episode.id.clone();
        if self.episodes.contains_key(&episode_id) {
            self.episodes.insert(episode_id, episode);
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    /// Export episodes
    pub async fn export_episodes(&self) -> Result<Vec<&MemoryEpisode>> {
        Ok(self.episodes.values().collect())
    }
    
    /// Import episodes
    pub async fn import_episodes(&mut self, episodes: Vec<MemoryEpisode>) -> Result<usize> {
        let mut imported = 0;
        for episode in episodes {
            if let Ok(_) = self.add_episode_internal(episode).await {
                imported += 1;
            }
        }
        Ok(imported)
    }
    
    /// Get statistics
    pub async fn get_stats(&self) -> EpisodicStats {
        let total_episodes = self.episodes.len();
        let mut importance_sum = 0.0;
        let mut tag_counts = HashMap::new();
        
        for episode in self.episodes.values() {
            importance_sum += episode.importance;
            
            for tag in &episode.tags {
                *tag_counts.entry(tag.clone()).or_insert(0) += 1;
            }
        }
        
        let avg_importance = if total_episodes > 0 {
            importance_sum / total_episodes as f32
        } else {
            0.0
        };
        
        EpisodicStats {
            total_episodes,
            avg_importance,
            unique_tags: tag_counts.len(),
            most_common_tag: tag_counts.iter()
                .max_by_key(|(_, &count)| count)
                .map(|(tag, _)| tag.clone()),
        }
    }
    
    /// Find similar episodes dan link them
    async fn find_similar_episodes(&mut self, new_episode_id: &str) -> Result<()> {
        let mut similar_episodes = Vec::with_capacity(self.episodes.len());
        
        if let Some(new_episode) = self.episodes.get(new_episode_id) {
            for (other_id, other_episode) in &self.episodes {
                if other_id == new_episode_id {
                    continue;
                }
                
                let similarity = new_episode.calculate_similarity(other_episode);
                if similarity >= self.similarity_threshold {
                    similar_episodes.push((other_id.clone(), similarity));
                }
            }
        }
        
        // Add related episodes
        let similar_ids: Vec<String> = similar_episodes.iter().take(5)
            .map(|(id, _)| id.clone())
            .collect();
        
        if let Some(episode) = self.episodes.get_mut(new_episode_id) {
            for similar_id in &similar_ids {
                episode.add_related_episode(similar_id.clone());
            }
        }
        
        // Update similar episodes to reference this one
        for similar_id in similar_ids {
            if let Some(similar_episode) = self.episodes.get_mut(&similar_id) {
                similar_episode.add_related_episode(new_episode_id.to_string());
            }
        }
        Ok(())
    }
    
    /// Evict least important episodes
    async fn evict_least_important(&mut self) -> Result<()> {
        if self.episodes.len() <= self.max_episodes {
            return Ok(());
        }
        
        // Find episodes to evict
        let mut episodes_to_evict: Vec<_> = self.episodes.iter()
            .map(|(id, episode)| (id.clone(), episode.importance, episode.timestamp))
            .collect();
        
        // Sort by importance (ascending) then by timestamp (older first)
        episodes_to_evict.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.2.cmp(&b.2))
        });
        
        // Evict the least important
        let to_evict_count = self.episodes.len() - self.max_episodes + 1;
        for (episode_id, _, _) in episodes_to_evict.iter().take(to_evict_count) {
            self.delete_episode(episode_id).await?;
        }
        
        debug!("Evicted {} least important episodes", to_evict_count);
        Ok(())
    }
    
    /// Get recent episodes
    pub async fn get_recent_episodes(&self, count: usize) -> Result<Vec<&MemoryEpisode>> {
        let mut recent_episodes = Vec::with_capacity(count);
        
        for (_timestamp, episode_ids) in self.temporal_index.iter().rev() {
            for episode_id in episode_ids.iter().rev() {
                if let Some(episode) = self.episodes.get(episode_id) {
                    recent_episodes.push(episode);
                    if recent_episodes.len() >= count {
                        return Ok(recent_episodes);
                    }
                }
            }
        }
        
        Ok(recent_episodes)
    }
}

/// Episodic memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicStats {
    pub total_episodes: usize,
    pub avg_importance: f32,
    pub unique_tags: usize,
    pub most_common_tag: Option<String>,
}

impl Default for EpisodicMemory {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_episodic_memory() {
        let mut episodic = EpisodicMemory::new(100);
        
        // Add episode
        let episode_id = episodic.add_episode("test_key", "test_content").await.unwrap();
        
        // Retrieve episode
        let episode = episodic.get_episode(&episode_id).await.unwrap().unwrap();
        assert_eq!(episode.content, "test_content");
        
        // Search episodes
        let results = episodic.search("test").await.unwrap().unwrap();
        assert!(!results.is_empty());
        
        // Get stats
        let stats = episodic.get_stats().await;
        assert_eq!(stats.total_episodes, 1);
    }
    
    #[tokio::test]
    async fn test_episode_similarity() {
        let episode1 = MemoryEpisode::new("hello world foo".to_string(), "test".to_string());
        let episode2 = MemoryEpisode::new("hello world bar".to_string(), "test".to_string());
        let episode3 = MemoryEpisode::new("goodbye world baz".to_string(), "test".to_string());
        
        let sim12 = episode1.calculate_similarity(&episode2);
        let sim13 = episode1.calculate_similarity(&episode3);
        
        assert!(sim12 > sim13);
    }
    
    #[tokio::test]
    async fn test_temporal_indexing() {
        let mut episodic = EpisodicMemory::new(10);
        
        // Add episodes with delay
        let id1 = episodic.add_episode("key1", "content1").await.unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        let id2 = episodic.add_episode("key2", "content2").await.unwrap();
        
        // Get recent episodes
        let recent = episodic.get_recent_episodes(2).await.unwrap();
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].id, id2); // Most recent first
        assert_eq!(recent[1].id, id1);
    }
    
    #[tokio::test]
    async fn test_tag_indexing() {
        let mut episodic = EpisodicMemory::new(10);
        
        let episode = MemoryEpisode::new("test content".to_string(), "test".to_string())
            .with_tags(vec!["important".to_string(), "test".to_string()]);
        
        let episode_id = episodic.add_episode_with_metadata(episode).await.unwrap();
        
        // Get episodes by tag
        let tagged_episodes = episodic.get_episodes_by_tag("important").await.unwrap();
        assert_eq!(tagged_episodes.len(), 1);
        assert_eq!(tagged_episodes[0].id, episode_id);
    }
    
    #[tokio::test]
    async fn test_episodic_memory_capacity_limit() {
        let mut episodic = EpisodicMemory::new(3);
        
        for i in 0..5 {
            episodic.add_episode(&format!("Episode {}", i), &format!("Content {}", i)).await.unwrap();
        }
        
        let all_episodes = episodic.get_all_episodes().await.unwrap();
        assert_eq!(all_episodes.len(), 3);
        
        assert!(!all_episodes.iter().any(|e| e.content.contains("Episode 0")));
    }
    
    #[tokio::test]
    async fn test_episodic_memory_search_with_relevance() {
        let mut episodic = EpisodicMemory::new(10);
        
        // Add episodes with different content and importance
        let episode1 = MemoryEpisode::new("Important meeting about project".to_string(), "work".to_string())
            .with_importance(0.9);
        let episode2 = MemoryEpisode::new("Lunch with friends".to_string(), "personal".to_string())
            .with_importance(0.3);
        let episode3 = MemoryEpisode::new("Project deadline discussion".to_string(), "work".to_string())
            .with_importance(0.8);
        
        episodic.add_episode_with_metadata(episode1).await.unwrap();
        episodic.add_episode_with_metadata(episode2).await.unwrap();
        episodic.add_episode_with_metadata(episode3).await.unwrap();
        
        // Search for "project"
        let results = episodic.search_episodes("project").await.unwrap();
        
        // Should find episodes 1 and 3, with episode 1 ranked higher due to importance
        assert_eq!(results.len(), 2);
        assert!(results[0].value.contains("Important meeting"));
        assert!(results[1].value.contains("Project deadline"));
    }
    
    #[tokio::test]
    async fn test_episodic_memory_get_by_time_range() {
        let mut episodic = EpisodicMemory::new(10);
        
        let base_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Add episodes at different times
        let episode1 = MemoryEpisode::new("First episode".to_string(), "test".to_string())
            .with_timestamp(base_time);
        let episode2 = MemoryEpisode::new("Second episode".to_string(), "test".to_string())
            .with_timestamp(base_time + 3600); // 1 hour later
        let episode3 = MemoryEpisode::new("Third episode".to_string(), "test".to_string())
            .with_timestamp(base_time + 7200); // 2 hours later
        
        let _id1 = episodic.add_episode_with_metadata(episode1).await.unwrap();
        let id2 = episodic.add_episode_with_metadata(episode2).await.unwrap();
        let _id3 = episodic.add_episode_with_metadata(episode3).await.unwrap();
        
        // Get episodes in time range
        let range_episodes = episodic.get_episodes_by_time_range(
            base_time + 1800, // 30 minutes after first
            base_time + 5400  // 90 minutes after first
        ).await.unwrap();
        
        // Should only find episode 2
        assert_eq!(range_episodes.len(), 1);
        assert_eq!(range_episodes[0].id, id2);
    }
    
    #[tokio::test]
    async fn test_episodic_memory_update_episode() {
        let mut episodic = EpisodicMemory::new(10);

        let episode_id = episodic.add_episode("Original content", "test").await.unwrap();

        // Update episode
        let episode_id_clone = episode_id.clone();
        let updated_episode = MemoryEpisode::new("Updated content".to_string(), "test".to_string())
            .with_id(episode_id_clone)
            .with_importance(0.8);

        let result = episodic.update_episode(updated_episode).await.unwrap();
        assert!(result);

        // Verify the update
        let episode_id_clone = episode_id.clone();
        let retrieved = episodic.get_episode(&episode_id_clone).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "Updated content");
        assert_eq!(retrieved.unwrap().importance, 0.8);
    }
    
    #[tokio::test]
    async fn test_episodic_memory_delete_episode() {
        let mut episodic = EpisodicMemory::new(10);
        
        let episode_id = episodic.add_episode("To be deleted", "test").await.unwrap();
        
        // Verify episode exists
        let before_delete = episodic.get_episode(&episode_id).await.unwrap();
        assert!(before_delete.is_some());
        
        // Delete the episode
        let result = episodic.delete_episode(&episode_id).await.unwrap();
        assert!(result);
        
        // Verify episode is deleted
        let after_delete = episodic.get_episode(&episode_id).await.unwrap();
        assert!(after_delete.is_none());
    }
    
    #[tokio::test]
    async fn test_episodic_memory_get_related_episodes() {
        let mut episodic = EpisodicMemory::new(10);
        
        // Add related episodes
        let episode1 = MemoryEpisode::new("Project kickoff meeting".to_string(), "work".to_string())
            .with_tags(vec!["project".to_string(), "meeting".to_string()]);
        let episode2 = MemoryEpisode::new("Project planning session".to_string(), "work".to_string())
            .with_tags(vec!["project".to_string(), "planning".to_string()]);
        let episode3 = MemoryEpisode::new("Lunch break".to_string(), "personal".to_string())
            .with_tags(vec!["personal".to_string(), "break".to_string()]);
        
        let id1 = episodic.add_episode_with_metadata(episode1).await.unwrap();
        let id2 = episodic.add_episode_with_metadata(episode2).await.unwrap();
        let _id3 = episodic.add_episode_with_metadata(episode3).await.unwrap();
        
        // Manually set up relationship between episode 1 and 2
        if let Some(episode1) = episodic.episodes.get_mut(&id1) {
            episode1.related_episodes.push(id2.clone());
        }
        
        // Get episodes related to episode 1
        let related = episodic.get_related_episodes(&id1).await.unwrap();
        
        // Should find episode 2 (manually related) but not episode 3
        assert_eq!(related.len(), 1);
        assert_eq!(related[0].id, id2);
    }
    
    #[tokio::test]
    async fn test_episodic_memory_export_import() {
        let mut episodic1 = EpisodicMemory::new(10);
        let mut episodic2 = EpisodicMemory::new(10);
        
        // Add episodes to first memory
        episodic1.add_episode("Episode 1", "test").await.unwrap();
        episodic1.add_episode("Episode 2", "test").await.unwrap();
        
        // Export episodes
        let exported = episodic1.export_episodes().await.unwrap();
        assert_eq!(exported.len(), 2);
        
        // Import to second memory
        let exported_owned: Vec<MemoryEpisode> = exported.into_iter().map(|e| e.clone()).collect();
        episodic2.import_episodes(exported_owned).await.unwrap();
        
        // Verify import
        let imported_episodes = episodic2.get_all_episodes().await.unwrap();
        assert_eq!(imported_episodes.len(), 2);
    }
    
    #[tokio::test]
    async fn test_episodic_memory_statistics() {
        let mut episodic = EpisodicMemory::new(10);
        
        // Add episodes with different characteristics
        let episode1 = MemoryEpisode::new("Work meeting".to_string(), "work".to_string())
            .with_importance(0.8);
        let episode2 = MemoryEpisode::new("Personal activity".to_string(), "personal".to_string())
            .with_importance(0.4);
        let episode3 = MemoryEpisode::new("Another work task".to_string(), "work".to_string())
            .with_importance(0.9);
        
        episodic.add_episode_with_metadata(episode1).await.unwrap();
        episodic.add_episode_with_metadata(episode2).await.unwrap();
        episodic.add_episode_with_metadata(episode3).await.unwrap();
        
        let stats = episodic.get_stats().await;
        
        assert_eq!(stats.total_episodes, 3);
        assert!(stats.avg_importance > 0.0);
        assert!(stats.avg_importance < 1.0);
    }
    
    #[test]
    fn test_memory_episode_creation() {
        let episode = MemoryEpisode::new("Test content".to_string(), "test".to_string());
        
        assert_eq!(episode.content, "Test content");
        assert_eq!(episode.context, "test");
        assert!(episode.timestamp > 0);
        assert!(episode.tags.is_empty());
        assert_eq!(episode.importance, 0.5); // Default importance
        assert!(!episode.id.is_empty());
    }
    
    #[test]
    fn test_memory_episode_with_metadata() {
        let episode = MemoryEpisode::new("Test content".to_string(), "test".to_string())
            .with_importance(0.8)
            .with_tags(vec!["tag1".to_string(), "tag2".to_string()])
            .with_timestamp(1234567890);
        
        assert_eq!(episode.importance, 0.8);
        assert_eq!(episode.tags.len(), 2);
        assert!(episode.tags.contains(&"tag1".to_string()));
        assert!(episode.tags.contains(&"tag2".to_string()));
        assert_eq!(episode.timestamp, 1234567890);
    }
    
    #[test]
    fn test_episodic_statistics() {
        let stats = EpisodicStats {
            total_episodes: 100,
            avg_importance: 0.7,
            unique_tags: 5,
            most_common_tag: Some("work".to_string()),
        };
        
        assert_eq!(stats.total_episodes, 100);
        assert_eq!(stats.avg_importance, 0.7);
        assert_eq!(stats.unique_tags, 5);
        assert_eq!(stats.most_common_tag, Some("work".to_string()));
    }
}
