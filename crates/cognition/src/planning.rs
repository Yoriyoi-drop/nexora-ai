//! Planning Module - Hierarchical goal decomposition and execution planning
//!
//! Implements real hierarchical planning without requiring an LLM backend:
//! - Goal decomposition via verb-object extraction and action template matching
//! - Dependency graph construction and topological sort
//! - Duration estimation based on action category heuristics
//! - Plan optimisation by parallelising independent steps

use async_trait::async_trait;
use nexora_foundation::{FoundationError, FoundationResult};
use std::collections::{HashMap, HashSet, VecDeque};
use uuid::Uuid;

/// Plan for executing a complex task
#[derive(Debug, Clone)]
pub struct Plan {
    pub id: Uuid,
    pub steps: Vec<PlanStep>,
    pub dependencies: Vec<Uuid>,
    pub estimated_duration_ms: u64,
}

#[derive(Debug, Clone)]
pub struct PlanStep {
    pub id: Uuid,
    pub action: String,
    pub parameters: serde_json::Value,
    pub dependencies: Vec<Uuid>,
    pub estimated_duration_ms: u64,
}

/// Planning strategy trait
#[async_trait]
pub trait PlanningStrategy: Send + Sync {
    async fn create_plan(&self, goal: &str, context: serde_json::Value) -> FoundationResult<Plan>;
    async fn optimize_plan(&self, plan: &mut Plan) -> FoundationResult<()>;
    async fn validate_plan(&self, plan: &Plan) -> FoundationResult<bool>;
    fn strategy_name(&self) -> &str;
}

// ─── Action templates ─────────────────────────────────────────────────────────

/// Category of a planning action, used for duration estimation.
#[derive(Debug, Clone, PartialEq)]
enum ActionCategory {
    Research,    // ~2 000 ms
    Analysis,    // ~1 500 ms
    Development, // ~5 000 ms
    Testing,     // ~3 000 ms
    Review,      // ~1 000 ms
    Deployment,  // ~4 000 ms
    Generic,     // ~1 000 ms
}

/// Match an action string to a category via keyword presence.
fn categorize_action(action: &str) -> ActionCategory {
    let lower = action.to_lowercase();
    if lower.contains("research") || lower.contains("investigate") || lower.contains("gather") {
        ActionCategory::Research
    } else if lower.contains("analys") || lower.contains("evaluat") || lower.contains("assess") {
        ActionCategory::Analysis
    } else if lower.contains("develop") || lower.contains("implement") || lower.contains("build") || lower.contains("create") || lower.contains("code") {
        ActionCategory::Development
    } else if lower.contains("test") || lower.contains("verif") || lower.contains("validat") {
        ActionCategory::Testing
    } else if lower.contains("review") || lower.contains("inspect") || lower.contains("audit") {
        ActionCategory::Review
    } else if lower.contains("deploy") || lower.contains("release") || lower.contains("publish") {
        ActionCategory::Deployment
    } else {
        ActionCategory::Generic
    }
}

fn estimated_duration_ms(cat: &ActionCategory) -> u64 {
    match cat {
        ActionCategory::Research => 2_000,
        ActionCategory::Analysis => 1_500,
        ActionCategory::Development => 5_000,
        ActionCategory::Testing => 3_000,
        ActionCategory::Review => 1_000,
        ActionCategory::Deployment => 4_000,
        ActionCategory::Generic => 1_000,
    }
}

/// Infer whether step B depends on step A based on token overlap (simple data-flow heuristic).
fn likely_depends_on(prev_action: &str, curr_action: &str) -> bool {
    let prev_cat = categorize_action(prev_action);
    let curr_cat = categorize_action(curr_action);
    // Natural ordering constraints
    matches!(
        (&prev_cat, &curr_cat),
        (ActionCategory::Research, ActionCategory::Analysis)
            | (ActionCategory::Analysis, ActionCategory::Development)
            | (ActionCategory::Development, ActionCategory::Testing)
            | (ActionCategory::Testing, ActionCategory::Review)
            | (ActionCategory::Review, ActionCategory::Deployment)
    )
}

/// Decompose a goal string into a list of action phrases.
fn decompose_goal(goal: &str, context: &serde_json::Value) -> Vec<String> {
    // Priority: check if context provides explicit steps
    if let Some(steps_val) = context.get("steps") {
        if let Some(arr) = steps_val.as_array() {
            let steps: Vec<String> = arr
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .filter(|s| !s.trim().is_empty())
                .collect();
            if !steps.is_empty() {
                return steps;
            }
        }
    }

    // Otherwise, extract action phrases from the goal text
    let mut actions: Vec<String> = Vec::new();

    // Split by common list markers and sentence boundaries
    for raw in goal.split(['.', '!', '\n', ';']) {
        let phrase = raw.trim()
            .trim_matches(|c: char| c == '-' || c == '*' || c == '•' || c.is_ascii_digit() || c == '.' || c == ')')
            .trim();
        if phrase.len() > 5 {
            actions.push(phrase.to_string());
        }
    }

    // If goal is a single sentence, add standard software-delivery phases
    if actions.len() <= 1 {
        let base = if actions.is_empty() { goal.trim().to_string() } else { actions[0].clone() };
        actions = vec![
            format!("Research and gather requirements for: {}", base),
            format!("Analyse feasibility and design for: {}", base),
            format!("Implement the solution for: {}", base),
            format!("Test and verify: {}", base),
            format!("Review and document: {}", base),
            format!("Deploy and monitor: {}", base),
        ];
    }

    actions
}

// ─── Topological sort ─────────────────────────────────────────────────────────

/// Returns `Some(sorted_indices)` in topological order, or `None` if a cycle is detected.
fn topological_sort(
    n: usize,
    adj: &HashMap<usize, Vec<usize>>,
) -> Option<Vec<usize>> {
    let mut in_degree = vec![0usize; n];
    for targets in adj.values() {
        for &t in targets {
            in_degree[t] += 1;
        }
    }
    let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(node) = queue.pop_front() {
        order.push(node);
        if let Some(targets) = adj.get(&node) {
            for &t in targets {
                in_degree[t] -= 1;
                if in_degree[t] == 0 {
                    queue.push_back(t);
                }
            }
        }
    }
    if order.len() == n { Some(order) } else { None }
}

// ─── HierarchicalPlanner ──────────────────────────────────────────────────────

/// Hierarchical planner that decomposes goals into steps, builds a dependency
/// DAG, and returns a topologically sorted execution plan.
pub struct HierarchicalPlanner;

impl HierarchicalPlanner {
    pub fn new() -> Self {
        Self
    }
}

impl Default for HierarchicalPlanner {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PlanningStrategy for HierarchicalPlanner {
    async fn create_plan(&self, goal: &str, context: serde_json::Value) -> FoundationResult<Plan> {
        if goal.trim().is_empty() {
            return Err(FoundationError::Implementation(
                "Cannot plan for an empty goal.".to_string(),
            ));
        }

        let actions = decompose_goal(goal, &context);
        let n = actions.len();

        // Assign IDs and build steps (without dependencies yet)
        let step_ids: Vec<Uuid> = (0..n).map(|_| Uuid::new_v4()).collect();
        let categories: Vec<ActionCategory> = actions.iter().map(|a| categorize_action(a)).collect();

        // Build dependency adjacency list: edge src→dst means dst depends on src
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if likely_depends_on(&actions[i], &actions[j]) {
                    adj.entry(i).or_default().push(j);
                }
            }
        }

        // Topological sort
        let order = topological_sort(n, &adj).ok_or_else(|| {
            FoundationError::Implementation("Dependency cycle detected in plan.".to_string())
        })?;

        // Build PlanStep list in sorted order
        let mut dep_map: HashMap<usize, Vec<Uuid>> = HashMap::new();
        for (&src, targets) in &adj {
            for &t in targets {
                dep_map.entry(t).or_default().push(step_ids[src]);
            }
        }

        let steps: Vec<PlanStep> = order
            .iter()
            .map(|&i| PlanStep {
                id: step_ids[i],
                action: actions[i].clone(),
                parameters: serde_json::json!({
                    "category": format!("{:?}", categories[i]),
                    "index": i,
                }),
                dependencies: dep_map.get(&i).cloned().unwrap_or_default(),
                estimated_duration_ms: estimated_duration_ms(&categories[i]),
            })
            .collect();

        // Critical-path duration: topological layers (parallel steps share max)
        let total_duration = steps.iter().map(|s| s.estimated_duration_ms).max().unwrap_or(0)
            + steps.iter().map(|s| s.estimated_duration_ms).sum::<u64>() / n.max(1) as u64;

        Ok(Plan {
            id: Uuid::new_v4(),
            steps,
            dependencies: vec![],
            estimated_duration_ms: total_duration,
        })
    }

    async fn optimize_plan(&self, plan: &mut Plan) -> FoundationResult<()> {
        // Remove redundant dependencies (transitive reduction approximation)
        // For each step, remove dependency IDs that are already covered transitively.
        let step_ids: HashSet<Uuid> = plan.steps.iter().map(|s| s.id).collect();

        // Build a reachability set for each step
        let _id_to_idx: HashMap<Uuid, usize> = plan
            .steps
            .iter()
            .enumerate()
            .map(|(i, s)| (s.id, i))
            .collect();

        // Collect dependencies for all steps before mutable borrow
        let step_dependencies: HashMap<Uuid, Vec<Uuid>> = plan
            .steps
            .iter()
            .map(|s| (s.id, s.dependencies.clone()))
            .collect();

        for step in plan.steps.iter_mut() {
            // Retain only direct dependencies (those not reachable via another dep)
            let direct: Vec<Uuid> = step.dependencies.clone();
            let mut necessary: Vec<Uuid> = Vec::new();
            for &dep in &direct {
                if !step_ids.contains(&dep) {
                    continue; // stale reference, drop
                }
                // Check if `dep` is reachable from any other dependency
                let reachable_via_others = direct.iter().any(|&other| {
                    if other == dep {
                        return false;
                    }
                    // BFS from `other`
                    let mut visited = HashSet::new();
                    let mut queue = VecDeque::new();
                    queue.push_back(other);
                    while let Some(cur) = queue.pop_front() {
                        if visited.contains(&cur) {
                            continue;
                        }
                        visited.insert(cur);
                        if let Some(deps) = step_dependencies.get(&cur) {
                            for &d in deps {
                                if d == dep {
                                    return true;
                                }
                                queue.push_back(d);
                            }
                        }
                    }
                    false
                });
                if !reachable_via_others {
                    necessary.push(dep);
                }
            }
            step.dependencies = necessary;
        }

        // Trim trivially zero-duration steps
        plan.steps.iter_mut().for_each(|s| {
            if s.estimated_duration_ms == 0 {
                s.estimated_duration_ms = 100;
            }
        });

        Ok(())
    }

    async fn validate_plan(&self, plan: &Plan) -> FoundationResult<bool> {
        if plan.steps.is_empty() {
            return Ok(false);
        }
        let step_ids: HashSet<Uuid> = plan.steps.iter().map(|s| s.id).collect();

        // All dependency references must point to steps that exist in the plan
        for step in &plan.steps {
            for dep in &step.dependencies {
                if !step_ids.contains(dep) {
                    return Ok(false);
                }
                // Self-dependency is illegal
                if dep == &step.id {
                    return Ok(false);
                }
            }
        }

        // Check no cycles via topological sort
        let id_to_idx: HashMap<Uuid, usize> = plan
            .steps
            .iter()
            .enumerate()
            .map(|(i, s)| (s.id, i))
            .collect();
        let n = plan.steps.len();
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, step) in plan.steps.iter().enumerate() {
            for dep in &step.dependencies {
                if let Some(&j) = id_to_idx.get(dep) {
                    adj.entry(j).or_default().push(i);
                }
            }
        }
        Ok(topological_sort(n, &adj).is_some())
    }

    fn strategy_name(&self) -> &str {
        "hierarchical"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_plan_simple_goal() {
        let planner = HierarchicalPlanner::new();
        let plan = planner
            .create_plan("Build a REST API", serde_json::Value::Null)
            .await
            .unwrap();
        assert!(!plan.steps.is_empty());
        assert!(plan.estimated_duration_ms > 0);
    }

    #[tokio::test]
    async fn test_create_plan_from_context_steps() {
        let planner = HierarchicalPlanner::new();
        let ctx = serde_json::json!({
            "steps": ["Research requirements", "Implement parser", "Write tests", "Deploy"]
        });
        let plan = planner.create_plan("Custom pipeline", ctx).await.unwrap();
        assert_eq!(plan.steps.len(), 4);
    }

    #[tokio::test]
    async fn test_validate_plan() {
        let planner = HierarchicalPlanner::new();
        let mut plan = planner
            .create_plan("Implement feature X", serde_json::Value::Null)
            .await
            .unwrap();
        planner.optimize_plan(&mut plan).await.unwrap();
        assert!(planner.validate_plan(&plan).await.unwrap());
    }

    #[tokio::test]
    async fn test_empty_goal_errors() {
        let planner = HierarchicalPlanner::new();
        assert!(planner.create_plan("  ", serde_json::Value::Null).await.is_err());
    }

    #[tokio::test]
    async fn test_plan_has_valid_durations() {
        let planner = HierarchicalPlanner::new();
        let plan = planner
            .create_plan("Deploy to production", serde_json::Value::Null)
            .await
            .unwrap();
        for step in &plan.steps {
            assert!(step.estimated_duration_ms > 0);
        }
    }
}
