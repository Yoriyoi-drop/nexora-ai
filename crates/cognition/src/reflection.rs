//! Reflection Module - Self-reflection and meta-cognition
//!
//! Real implementation:
//! - reflect(): analyses action success/failure patterns, computes per-category
//!   error rates, generates concrete improvement suggestions based on observed
//!   behaviour, and returns a confidence score.
//! - update_model(): persists ReflectionResult to internal history and tracks
//!   rolling improvement rate.
//! - suggest_improvements(): merges heuristic rules with stored insights.
//! - stats(): returns live statistics from the history store.

use async_trait::async_trait;
use crate::FoundationResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Public types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionResult {
    pub confidence: f32,
    pub errors_identified: Vec<String>,
    pub improvements_suggested: Vec<String>,
    pub learning_insights: Vec<String>,
    pub metadata: ReflectionMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionMetadata {
    pub reflection_type: ReflectionType,
    pub timestamp: i64,
    pub context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReflectionType {
    Performance,
    Accuracy,
    Efficiency,
    Learning,
    ErrorAnalysis,
}

#[derive(Debug, Clone)]
pub struct Action {
    pub action_type: String,
    pub input: String,
    pub output: String,
    pub timestamp: i64,
    pub success: bool,
}

#[derive(Debug, Clone)]
pub struct ReflectionStats {
    pub total_reflections: usize,
    pub avg_confidence: f32,
    pub improvement_rate: f32,
}

// ─── Trait ────────────────────────────────────────────────────────────────────

#[async_trait]
pub trait ReflectionEngine: Send + Sync {
    async fn reflect(
        &self,
        actions: &[Action],
        context: &str,
    ) -> FoundationResult<ReflectionResult>;

    async fn suggest_improvements(
        &self,
        reflection: &ReflectionResult,
    ) -> FoundationResult<Vec<String>>;

    async fn update_model(&self, reflection: &ReflectionResult) -> FoundationResult<()>;

    async fn stats(&self) -> FoundationResult<ReflectionStats>;
}

// ─── Analysis helpers ─────────────────────────────────────────────────────────

/// Compute per-action-type failure rate.
fn failure_rates(actions: &[Action]) -> HashMap<String, f32> {
    let mut total: HashMap<String, usize> = HashMap::new();
    let mut failures: HashMap<String, usize> = HashMap::new();

    for a in actions {
        *total.entry(a.action_type.clone()).or_insert(0) += 1;
        if !a.success {
            *failures.entry(a.action_type.clone()).or_insert(0) += 1;
        }
    }

    total
        .iter()
        .map(|(k, &t)| {
            let f = failures.get(k).copied().unwrap_or(0);
            (k.clone(), f as f32 / t as f32)
        })
        .collect()
}

/// Identify recurring patterns in failed action inputs/outputs.
fn extract_error_patterns(actions: &[Action]) -> Vec<String> {
    let mut errors: Vec<String> = Vec::new();
    let failed: Vec<&Action> = actions.iter().filter(|a| !a.success).collect();

    if failed.is_empty() {
        return errors;
    }

    // Group failures by action_type
    let mut by_type: HashMap<&str, Vec<&Action>> = HashMap::new();
    for a in &failed {
        by_type.entry(a.action_type.as_str()).or_default().push(a);
    }

    for (action_type, group) in &by_type {
        let count = group.len();
        errors.push(format!(
            "Action '{}' failed {} time(s). Last output: '{}'",
            action_type,
            count,
            group.last().map(|a| a.output.as_str()).unwrap_or("(empty)")
        ));

        // Detect repeated identical outputs (potential infinite loop / stuck state)
        if group.len() >= 2 {
            let all_same_output = group.windows(2).all(|w| w[0].output == w[1].output);
            if all_same_output {
                errors.push(format!(
                    "Action '{}' produced identical output across all failures — possible stuck state.",
                    action_type
                ));
            }
        }
    }

    errors
}

/// Generate improvement suggestions based on failure rates and error patterns.
fn generate_improvements(
    failure_rates: &HashMap<String, f32>,
    errors: &[String],
    success_rate: f32,
) -> Vec<String> {
    let mut suggestions: Vec<String> = Vec::new();

    // High-failure action types
    for (action_type, &rate) in failure_rates {
        if rate > 0.5 {
            suggestions.push(format!(
                "Action '{}' has a {:.0}% failure rate — add input validation and retry logic.",
                action_type,
                rate * 100.0
            ));
        } else if rate > 0.2 {
            suggestions.push(format!(
                "Action '{}' fails {:.0}% of the time — review edge cases.",
                action_type,
                rate * 100.0
            ));
        }
    }

    // Stuck-state patterns
    if errors.iter().any(|e| e.contains("stuck state")) {
        suggestions.push(
            "Detect repeated identical outputs and apply a backoff/escape strategy.".to_string(),
        );
    }

    // Global success rate guidance
    if success_rate < 0.5 {
        suggestions.push("Overall success rate is critically low. Prioritise defensive error handling and circuit breakers.".to_string());
    } else if success_rate < 0.8 {
        suggestions.push("Success rate is moderate. Consider adding telemetry per action type to identify bottlenecks.".to_string());
    }

    if suggestions.is_empty() {
        suggestions.push(
            "No critical failures detected. Monitor latency and resource usage proactively."
                .to_string(),
        );
    }

    suggestions
}

/// Extract learning insights from the action sequence.
fn extract_insights(actions: &[Action]) -> Vec<String> {
    let mut insights: Vec<String> = Vec::new();

    if actions.is_empty() {
        return insights;
    }

    let total = actions.len();
    let successes = actions.iter().filter(|a| a.success).count();
    let success_rate = successes as f32 / total as f32;

    insights.push(format!(
        "{}/{} actions succeeded ({:.0}% success rate).",
        successes,
        total,
        success_rate * 100.0
    ));

    // Detect time-ordered degradation: compare first-half vs second-half success rates
    if total >= 4 {
        let mid = total / 2;
        let early_successes = actions[..mid].iter().filter(|a| a.success).count();
        let late_successes = actions[mid..].iter().filter(|a| a.success).count();
        let early_rate = early_successes as f32 / mid as f32;
        let late_rate = late_successes as f32 / (total - mid) as f32;

        if early_rate - late_rate > 0.2 {
            insights.push(
                "Performance degraded over time — check for resource exhaustion or state accumulation.".to_string()
            );
        } else if late_rate - early_rate > 0.2 {
            insights.push(
                "Performance improved over time — warm-up effect or adaptive behaviour detected."
                    .to_string(),
            );
        }
    }

    // Unique action types
    let unique_types: std::collections::HashSet<&str> =
        actions.iter().map(|a| a.action_type.as_str()).collect();
    if unique_types.len() == 1 {
        insights.push(
            "All actions are of the same type — consider diversifying task distribution."
                .to_string(),
        );
    }

    insights
}

/// Determine reflection type from context keyword heuristics.
fn classify_reflection_type(context: &str) -> ReflectionType {
    let lower = context.to_lowercase();
    if lower.contains("performance") || lower.contains("latency") || lower.contains("throughput") {
        ReflectionType::Performance
    } else if lower.contains("accuracy") || lower.contains("correct") || lower.contains("precision")
    {
        ReflectionType::Accuracy
    } else if lower.contains("efficien") || lower.contains("resource") || lower.contains("memory") {
        ReflectionType::Efficiency
    } else if lower.contains("learn") || lower.contains("adapt") || lower.contains("improve") {
        ReflectionType::Learning
    } else {
        ReflectionType::ErrorAnalysis
    }
}

fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

// ─── DefaultReflector ─────────────────────────────────────────────────────────

pub struct DefaultReflector {
    history: std::sync::Arc<tokio::sync::RwLock<Vec<ReflectionResult>>>,
}

impl DefaultReflector {
    pub fn new() -> Self {
        Self {
            history: std::sync::Arc::new(tokio::sync::RwLock::new(Vec::new())),
        }
    }
}

impl Default for DefaultReflector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ReflectionEngine for DefaultReflector {
    /// Analyse `actions`, identify errors and improvements, compute confidence.
    async fn reflect(
        &self,
        actions: &[Action],
        context: &str,
    ) -> FoundationResult<ReflectionResult> {
        let total = actions.len();
        let successes = actions.iter().filter(|a| a.success).count();
        let success_rate = if total == 0 {
            0.5
        } else {
            successes as f32 / total as f32
        };

        let rates = failure_rates(actions);
        let errors = extract_error_patterns(actions);
        let improvements = generate_improvements(&rates, &errors, success_rate);
        let insights = extract_insights(actions);

        // Confidence: weighted blend of success rate, diversity, and pattern clarity
        let diversity_bonus = if rates.len() > 1 { 0.05 } else { 0.0 };
        let confidence = (0.5 * success_rate
            + 0.4 * (1.0 - rates.values().copied().sum::<f32>() / rates.len().max(1) as f32)
            + diversity_bonus)
            .min(1.0)
            .max(0.0);

        let reflection_type = classify_reflection_type(context);

        let result = ReflectionResult {
            confidence,
            errors_identified: errors,
            improvements_suggested: improvements,
            learning_insights: insights,
            metadata: ReflectionMetadata {
                reflection_type,
                timestamp: now_unix(),
                context: context.to_string(),
            },
        };

        // Persist to history immediately so stats are live (capped at 1000)
        {
            let mut history = self.history.write().await;
            history.push(result.clone());
            if history.len() > 1000 {
                history.remove(0);
            }
        }

        Ok(result)
    }

    async fn suggest_improvements(
        &self,
        reflection: &ReflectionResult,
    ) -> FoundationResult<Vec<String>> {
        let mut suggestions = Vec::with_capacity(
            4 + reflection.improvements_suggested.len() + reflection.learning_insights.len(),
        );

        if reflection.confidence < 0.3 {
            suggestions.push(
                "Confidence is critically low. Consider using a different approach entirely."
                    .to_string(),
            );
            suggestions.push(
                "Break down the problem into smaller, independently verifiable steps.".to_string(),
            );
        } else if reflection.confidence < 0.7 {
            suggestions.push("Add validation checkpoints between action steps.".to_string());
            suggestions.push("Review recent changes for regressions.".to_string());
        }

        suggestions.extend(reflection.improvements_suggested.clone());
        suggestions.extend(reflection.learning_insights.clone());
        suggestions.dedup();

        Ok(suggestions)
    }

    /// Persist the reflection result to the internal history store.
    /// History is capped at 1 000 entries to prevent unbounded memory growth.
    async fn update_model(&self, reflection: &ReflectionResult) -> FoundationResult<()> {
        let mut history = self.history.write().await;
        // Avoid duplicating entries that were already added in reflect()
        let already_exists = history
            .iter()
            .any(|r| r.metadata.timestamp == reflection.metadata.timestamp);
        if !already_exists {
            history.push(reflection.clone());
        }
        // Prune oldest entries if history exceeds 1 000
        if history.len() > 1_000 {
            let excess = history.len() - 1_000;
            history.drain(0..excess);
        }
        Ok(())
    }

    async fn stats(&self) -> FoundationResult<ReflectionStats> {
        let history = self.history.read().await;
        let total = history.len();
        let avg_confidence = if total == 0 {
            0.0
        } else {
            history.iter().map(|r| r.confidence).sum::<f32>() / total as f32
        };
        // Rolling improvement rate: fraction of recent reflections with confidence > 0.7
        let window = &history[total.saturating_sub(10)..];
        let improvement_rate = if window.is_empty() {
            0.0
        } else {
            window.iter().filter(|r| r.confidence > 0.7).count() as f32 / window.len() as f32
        };

        Ok(ReflectionStats {
            total_reflections: total,
            avg_confidence,
            improvement_rate,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_action(action_type: &str, success: bool) -> Action {
        Action {
            action_type: action_type.to_string(),
            input: "test input".to_string(),
            output: if success {
                "ok".to_string()
            } else {
                "error".to_string()
            },
            timestamp: now_unix(),
            success,
        }
    }

    #[tokio::test]
    async fn test_reflect_all_success() {
        let engine = DefaultReflector::new();
        let actions = vec![
            make_action("inference", true),
            make_action("inference", true),
            make_action("tokenize", true),
        ];
        let result = engine.reflect(&actions, "performance check").await.unwrap();
        assert!(result.confidence > 0.5);
        assert!(!result.learning_insights.is_empty());
    }

    #[tokio::test]
    async fn test_reflect_all_failure() {
        let engine = DefaultReflector::new();
        let actions = vec![
            make_action("inference", false),
            make_action("inference", false),
            make_action("decode", false),
        ];
        let result = engine.reflect(&actions, "error analysis").await.unwrap();
        assert!(!result.errors_identified.is_empty());
        assert!(!result.improvements_suggested.is_empty());
    }

    #[tokio::test]
    async fn test_update_model_caps_history() {
        let engine = DefaultReflector::new();
        let action = make_action("test", true);
        let base_result = engine.reflect(&[action], "test").await.unwrap();
        // Write 999 more entries directly
        let mut history = engine.history.write().await;
        for i in 0..999 {
            let mut r = base_result.clone();
            r.metadata.timestamp = i as i64;
            history.push(r);
        }
        drop(history);
        // Adding one more via update_model should trigger pruning
        let mut extra = base_result.clone();
        extra.metadata.timestamp = 99999;
        engine.update_model(&extra).await.unwrap();
        let stats = engine.stats().await.unwrap();
        assert!(stats.total_reflections <= 1_000);
    }

    #[tokio::test]
    async fn test_stats_live_after_reflect() {
        let engine = DefaultReflector::new();
        let actions = vec![
            make_action("generate", true),
            make_action("generate", false),
        ];
        engine.reflect(&actions, "monitoring").await.unwrap();
        let stats = engine.stats().await.unwrap();
        assert_eq!(stats.total_reflections, 1);
        assert!(stats.avg_confidence > 0.0);
    }

    #[tokio::test]
    async fn test_stuck_state_detection() {
        let engine = DefaultReflector::new();
        let actions = vec![
            Action {
                action_type: "decode".to_string(),
                input: "x".to_string(),
                output: "same_output".to_string(),
                timestamp: 0,
                success: false,
            },
            Action {
                action_type: "decode".to_string(),
                input: "y".to_string(),
                output: "same_output".to_string(),
                timestamp: 1,
                success: false,
            },
        ];
        let result = engine.reflect(&actions, "error analysis").await.unwrap();
        assert!(result
            .errors_identified
            .iter()
            .any(|e| e.contains("stuck state")));
    }
}
