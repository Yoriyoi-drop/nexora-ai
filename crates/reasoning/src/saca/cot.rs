//! Chain-of-Thought Reasoning Engine
//!
//! Phase 1 of SACA: Systematic reasoning before code generation
//! Implements structured thinking process to identify edge cases, assumptions, and risks

use super::{config::*, error::*, types::*};
use nexora_core::async_executor::AsyncTaskExecutor;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Chain-of-Thought reasoning engine
pub struct CoTEngine {
    config: CoTConfig,
    _executor: Arc<AsyncTaskExecutor>,
    reasoning_cache: Arc<RwLock<std::collections::HashMap<String, CoTResult>>>,
}

impl CoTEngine {
    /// Create new CoT engine
    pub fn new(config: CoTConfig) -> SACAResult<Self> {
        let executor = Arc::new(AsyncTaskExecutor::new(
            nexora_core::async_executor::ExecutorConfig::default(),
        ));

        info!(
            "CoT Engine initialized with {} max reasoning steps",
            config.max_reasoning_steps
        );

        Ok(Self {
            config,
            _executor: executor,
            reasoning_cache: Arc::new(RwLock::new(std::collections::HashMap::new())),
        })
    }

    /// Perform Chain-of-Thought reasoning on a coding task
    pub async fn reason(&self, task: &CodingTask) -> SACAResult<CoTResult> {
        debug!("Starting CoT reasoning for task: {}", task.description);

        // Check cache first
        let cache_key = self.generate_cache_key(task);
        if let Some(cached_result) = self.reasoning_cache.read().await.get(&cache_key) {
            debug!("Using cached CoT result");
            return Ok(cached_result.clone());
        }

        // Perform reasoning
        let result = self.perform_reasoning(task).await?;

        // Cache the result
        self.reasoning_cache
            .write()
            .await
            .insert(cache_key, result.clone());

        debug!(
            "CoT reasoning completed with {} steps",
            result.reasoning_steps.len()
        );
        Ok(result)
    }

    /// Core reasoning implementation
    async fn perform_reasoning(&self, task: &CodingTask) -> SACAResult<CoTResult> {
        let mut reasoning_steps = Vec::new();
        let mut edge_cases = Vec::new();
        let mut assumptions = Vec::new();
        let mut risks = Vec::new();

        // Step 1: Task Analysis and Understanding
        let task_analysis = self.analyze_task(task).await?;
        reasoning_steps.push(ReasoningStep {
            step_number: 1,
            description: "Analyze task requirements and constraints".to_string(),
            logic: task_analysis.clone(),
            expected_outcome: "Clear understanding of what needs to be implemented".to_string(),
        });

        // Step 2: Identify Key Components and Data Structures
        let components_analysis = self.identify_components(task).await?;
        reasoning_steps.push(ReasoningStep {
            step_number: 2,
            description: "Identify key components and data structures".to_string(),
            logic: components_analysis.clone(),
            expected_outcome: "List of required components and their relationships".to_string(),
        });

        // Step 3: Algorithm Selection and Design
        let algorithm_design = self.design_algorithm(task).await?;
        reasoning_steps.push(ReasoningStep {
            step_number: 3,
            description: "Select and design appropriate algorithm".to_string(),
            logic: algorithm_design.clone(),
            expected_outcome: "Clear algorithm design with complexity analysis".to_string(),
        });

        // Step 4: Edge Case Analysis
        if self.config.include_edge_cases {
            edge_cases = self.identify_edge_cases(task).await?;
            reasoning_steps.push(ReasoningStep {
                step_number: 4,
                description: "Identify and plan for edge cases".to_string(),
                logic: format!("Edge cases identified: {}", edge_cases.join(", ")),
                expected_outcome: "Comprehensive edge case handling strategy".to_string(),
            });
        }

        // Step 5: Assumption Analysis
        if self.config.include_assumptions {
            assumptions = self.identify_assumptions(task).await?;
            reasoning_steps.push(ReasoningStep {
                step_number: 5,
                description: "Identify underlying assumptions".to_string(),
                logic: format!("Assumptions: {}", assumptions.join(", ")),
                expected_outcome: "Clear documentation of all assumptions".to_string(),
            });
        }

        // Step 6: Risk Assessment
        if self.config.include_risks {
            risks = self.assess_risks(task).await?;
            reasoning_steps.push(ReasoningStep {
                step_number: 6,
                description: "Assess implementation risks".to_string(),
                logic: format!("Risks: {}", risks.join(", ")),
                expected_outcome: "Risk mitigation strategies identified".to_string(),
            });
        }

        // Step 7: Implementation Approach
        let approach = self.define_approach(task, &reasoning_steps).await?;
        reasoning_steps.push(ReasoningStep {
            step_number: reasoning_steps.len() as u32 + 1,
            description: "Define implementation approach".to_string(),
            logic: approach.clone(),
            expected_outcome: "Clear step-by-step implementation plan".to_string(),
        });

        // Additional reasoning steps based on depth configuration
        if matches!(
            self.config.reasoning_depth,
            ReasoningDepth::Deep | ReasoningDepth::Exhaustive
        ) {
            self.add_deep_reasoning_steps(&mut reasoning_steps, task)
                .await?;
        }

        if matches!(self.config.reasoning_depth, ReasoningDepth::Exhaustive) {
            self.add_exhaustive_reasoning_steps(&mut reasoning_steps, task)
                .await?;
        }

        // Limit reasoning steps if configured
        if reasoning_steps.len() > self.config.max_reasoning_steps as usize {
            reasoning_steps.truncate(self.config.max_reasoning_steps as usize);
            warn!(
                "Reasoning steps truncated to configured maximum of {}",
                self.config.max_reasoning_steps
            );
        }

        Ok(CoTResult {
            task_analysis,
            reasoning_steps,
            edge_cases,
            assumptions,
            risks,
            approach,
        })
    }

    /// Extract token frequencies from text for TF-IDF-like scoring
    fn extract_token_frequencies(text: &str) -> Vec<(String, usize)> {
        let mut freqs: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for token in text.split_whitespace() {
            let clean: String = token.chars().filter(|c| c.is_alphanumeric()).collect();
            if clean.len() > 2 {
                *freqs.entry(clean.to_lowercase()).or_insert(0) += 1;
            }
        }
        let mut result: Vec<(String, usize)> = freqs.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result.truncate(10);
        result
    }

    /// Score how strongly a dimension applies based on keyword frequency
    fn score_dimension(text: &str, keywords: &[&str]) -> f32 {
        let lower = text.to_lowercase();
        let total: usize = keywords.iter().map(|kw| lower.matches(kw).count()).sum();
        (total as f32).sqrt() / 3.0_f32.max(1.0)
    }

    /// Analyze the task requirements using structured pattern analysis
    async fn analyze_task(&self, task: &CodingTask) -> SACAResult<String> {
        let desc_lower = task.description.to_lowercase();
        let mut dimensions = Vec::new();
        let mut dimension_scores: Vec<(String, f32)> = Vec::new();

        // Use TF-IDF-style term extraction for deeper analysis
        let top_tokens = Self::extract_token_frequencies(&task.description);
        let key_terms: Vec<String> = top_tokens.into_iter().map(|(t, _)| t).collect();

        // Score each dimension using keyword frequency rather than binary contains()
        let data_score = Self::score_dimension(
            &desc_lower,
            &[
                "array",
                "list",
                "collection",
                "data",
                "stream",
                "vector",
                "matrix",
                "record",
            ],
        );
        if data_score > 0.3 {
            dimension_scores.push(("data_processing".to_string(), data_score));
        }

        let io_score = Self::score_dimension(
            &desc_lower,
            &[
                "read",
                "write",
                "parse",
                "serialize",
                "format",
                "deserialize",
                "encode",
                "decode",
            ],
        );
        if io_score > 0.3 {
            dimension_scores.push(("io".to_string(), io_score));
        }

        let net_score = Self::score_dimension(
            &desc_lower,
            &[
                "request", "response", "http", "api", "client", "server", "endpoint", "rest",
            ],
        );
        if net_score > 0.3 {
            dimension_scores.push(("network".to_string(), net_score));
        }

        let state_score = Self::score_dimension(
            &desc_lower,
            &[
                "state", "cache", "store", "persist", "memo", "session", "context",
            ],
        );
        if state_score > 0.3 {
            dimension_scores.push(("state_management".to_string(), state_score));
        }

        let conc_score = Self::score_dimension(
            &desc_lower,
            &[
                "thread",
                "async",
                "parallel",
                "concurrent",
                "sync",
                "lock",
                "atomic",
                "race",
            ],
        );
        if conc_score > 0.3 {
            dimension_scores.push(("concurrency".to_string(), conc_score));
        }

        let comp_score = Self::score_dimension(
            &desc_lower,
            &[
                "compute",
                "calculate",
                "transform",
                "process",
                "map",
                "reduce",
                "aggregate",
                "evaluate",
            ],
        );
        if comp_score > 0.3 {
            dimension_scores.push(("computation".to_string(), comp_score));
        }

        let ml_score = Self::score_dimension(
            &desc_lower,
            &[
                "train",
                "model",
                "predict",
                "classify",
                "regression",
                "neural",
                "embedding",
                "feature",
            ],
        );
        if ml_score > 0.3 {
            dimension_scores.push(("machine_learning".to_string(), ml_score));
        }

        // Sort dimensions by score descending
        dimension_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let dimension_str = if dimension_scores.is_empty() {
            "general_purpose (no specific dimension detected)".to_string()
        } else {
            dimension_scores
                .iter()
                .map(|(d, s)| format!("{}@{:.2}", d, s))
                .collect::<Vec<_>>()
                .join(", ")
        };

        dimensions.extend(dimension_scores.into_iter().map(|(d, _)| d));

        let mut analysis = format!(
            "Task: {}\nRequirements: {}\nConstraints: {}\nDimensions: [{}]\nKey terms: {}\nContext: {}",
            task.description,
            task.requirements.join(", "),
            task.constraints.join(", "),
            dimension_str,
            key_terms.join(", "),
            task.context
                .as_ref()
                .map(|c| format!("Repository: {:?}", c.repository_path))
                .unwrap_or_else(|| "None".to_string())
        );

        // Add constraint-driven analysis with severity scoring
        for constraint in &task.constraints {
            let cl = constraint.to_lowercase();
            if cl.contains("complexity") || cl.contains("performance") {
                analysis.push_str(&format!("\n  - [HIGH] Constraint insight: '{}' requires algorithmic focus on efficiency", constraint));
            }
            if cl.contains("memory") || cl.contains("resource") {
                analysis.push_str(&format!(
                    "\n  - [HIGH] Constraint insight: '{}' requires space-efficient design",
                    constraint
                ));
            }
            if cl.contains("safe") || cl.contains("error") || cl.contains("robust") {
                analysis.push_str(&format!(
                    "\n  - [MEDIUM] Constraint insight: '{}' requires defensive programming",
                    constraint
                ));
            }
            if cl.contains("real") || cl.contains("deadline") || cl.contains("time") {
                analysis.push_str(&format!(
                    "\n  - [CRITICAL] Constraint insight: '{}' imposes real-time requirements",
                    constraint
                ));
            }
        }

        Ok(analysis)
    }

    /// Identify key components needed through structural analysis
    async fn identify_components(&self, task: &CodingTask) -> SACAResult<String> {
        let desc_lower = task.description.to_lowercase();
        let mut components: Vec<(String, Vec<String>)> = Vec::new(); // (component, dependencies)
        let all_text = {
            let mut t = task.description.clone();
            for req in &task.requirements {
                t.push_str(" ");
                t.push_str(req);
            }
            for con in &task.constraints {
                t.push_str(" ");
                t.push_str(con);
            }
            t.to_lowercase()
        };

        // Build component dependency graph
        // Sort/search operations
        if Self::score_dimension(&desc_lower, &["sort", "order", "rank", "compare", "sorted"]) > 0.3
        {
            components.push(("InputCollection".to_string(), vec![]));
            components.push((
                "ComparisonStrategy".to_string(),
                vec!["InputCollection".to_string()],
            ));
            components.push((
                "SortingAlgorithm".to_string(),
                vec![
                    "ComparisonStrategy".to_string(),
                    "InputCollection".to_string(),
                ],
            ));
            components.push((
                "OrderedOutput".to_string(),
                vec!["SortingAlgorithm".to_string()],
            ));
        }
        // Search operations
        if Self::score_dimension(
            &desc_lower,
            &["search", "find", "locate", "query", "lookup"],
        ) > 0.3
        {
            components.push(("SearchableCollection".to_string(), vec![]));
            components.push(("SearchPredicate".to_string(), vec![]));
            components.push((
                "SearchAlgorithm".to_string(),
                vec![
                    "SearchableCollection".to_string(),
                    "SearchPredicate".to_string(),
                ],
            ));
            components.push((
                "ResultHandler".to_string(),
                vec!["SearchAlgorithm".to_string()],
            ));
        }
        // Parse/tokenize operations
        if Self::score_dimension(&desc_lower, &["parse", "tokenize", "lex", "grammar"]) > 0.3 {
            components.push(("InputReader".to_string(), vec![]));
            components.push(("GrammarDefinition".to_string(), vec![]));
            components.push(("Tokenizer".to_string(), vec!["InputReader".to_string()]));
            components.push((
                "ParserStateMachine".to_string(),
                vec!["Tokenizer".to_string(), "GrammarDefinition".to_string()],
            ));
            components.push((
                "AbstractSyntaxTree".to_string(),
                vec!["ParserStateMachine".to_string()],
            ));
        }
        // Transform/convert operations
        if Self::score_dimension(
            &desc_lower,
            &["transform", "convert", "encode", "decode", "map"],
        ) > 0.3
        {
            components.push(("InputValidator".to_string(), vec![]));
            components.push((
                "TransformFunction".to_string(),
                vec!["InputValidator".to_string()],
            ));
            components.push((
                "OutputEncoder".to_string(),
                vec!["TransformFunction".to_string()],
            ));
        }
        // Aggregate operations
        if Self::score_dimension(&desc_lower, &["aggregate", "summarize", "group", "reduce"]) > 0.3
        {
            components.push(("DataPartitioner".to_string(), vec![]));
            components.push((
                "AggregationFunction".to_string(),
                vec!["DataPartitioner".to_string()],
            ));
            components.push((
                "ResultCombiner".to_string(),
                vec!["AggregationFunction".to_string()],
            ));
        }
        // Graph/tree operations
        if Self::score_dimension(&desc_lower, &["graph", "tree", "node", "edge", "vertex"]) > 0.3 {
            components.push(("GraphRepresentation".to_string(), vec![]));
            components.push((
                "TraversalStrategy".to_string(),
                vec!["GraphRepresentation".to_string()],
            ));
            components.push(("VisitedTracker".to_string(), vec![]));
            components.push((
                "PathCollector".to_string(),
                vec!["TraversalStrategy".to_string()],
            ));
        }
        // ML/Model operations
        if Self::score_dimension(
            &desc_lower,
            &["train", "model", "predict", "neural", "classify"],
        ) > 0.3
        {
            components.push(("DatasetLoader".to_string(), vec![]));
            components.push((
                "FeatureExtractor".to_string(),
                vec!["DatasetLoader".to_string()],
            ));
            components.push(("ModelDefinition".to_string(), vec![]));
            components.push((
                "TrainingLoop".to_string(),
                vec![
                    "ModelDefinition".to_string(),
                    "FeatureExtractor".to_string(),
                ],
            ));
            components.push((
                "InferenceEngine".to_string(),
                vec!["ModelDefinition".to_string()],
            ));
        }

        // If no specific categories matched, derive from requirements with dependency tracking
        if components.is_empty() {
            let deps: Vec<String> = task
                .requirements
                .iter()
                .enumerate()
                .map(|(i, _r)| {
                    format!(
                        "ReqComponent_{}(depends on: {:?})",
                        i,
                        task.requirements
                            .iter()
                            .take(i)
                            .map(|pr| format!(
                                "ReqComponent_{}",
                                task.requirements.iter().position(|x| x == pr).unwrap_or(0)
                            ))
                            .collect::<Vec<_>>()
                    )
                })
                .collect();
            for d in deps {
                components.push((d, vec![]));
            }
            if components.is_empty() {
                components.push(("CoreLogic".to_string(), vec![]));
                components.push(("InputValidation".to_string(), vec!["CoreLogic".to_string()]));
                components.push(("ErrorHandling".to_string(), vec!["CoreLogic".to_string()]));
                components.push((
                    "OutputFormatting".to_string(),
                    vec!["CoreLogic".to_string()],
                ));
            }
        }

        // Derive additional components from requirements
        if Self::score_dimension(&all_text, &["type", "struct", "enum", "data"]) > 0.3 {
            components.push(("TypeDefinitions".to_string(), vec![]));
        }
        if Self::score_dimension(&all_text, &["trait", "interface", "protocol"]) > 0.3 {
            components.push(("InterfaceDefinitions".to_string(), vec![]));
        }
        if Self::score_dimension(&all_text, &["config", "setting", "parameter"]) > 0.3 {
            components.push(("ConfigurationSystem".to_string(), vec![]));
        }
        if Self::score_dimension(&all_text, &["test", "assert", "verify"]) > 0.3 {
            components.push(("TestSuite".to_string(), vec!["CoreLogic".to_string()]));
        }

        // Build dependency graph representation
        let mut graph_repr = String::from("Component dependency graph:\n");
        for (comp, deps) in &components {
            if deps.is_empty() {
                graph_repr.push_str(&format!("  {} (root)\n", comp));
            } else {
                graph_repr.push_str(&format!("  {} -> {}\n", comp, deps.join(", ")));
            }
        }

        // Component list for reporting
        let comp_list: Vec<String> = components.into_iter().map(|(c, _)| c).collect();
        graph_repr.push_str(&format!(
            "\nKey components identified: {}",
            comp_list.join(", ")
        ));

        Ok(graph_repr)
    }

    /// Design appropriate algorithm based on structural analysis
    /// Generates concrete algorithm steps with complexity analysis
    async fn design_algorithm(&self, task: &CodingTask) -> SACAResult<String> {
        let desc_lower = task.description.to_lowercase();
        let mut design = String::new();

        // Build a structured algorithm design with steps
        let add_step = |design: &mut String, num: usize, title: &str, body: &str| {
            design.push_str(&format!("\n  Step {}: {} — {}", num, title, body));
        };

        design.push_str("Algorithm design:");
        let mut step = 1;

        // Phase 1: Input processing (always first)
        add_step(&mut design, step, "Input validation",
            "Validate input format, check preconditions, normalize data. Return early on invalid input.");
        step += 1;

        // Select algorithm category based on scored dimensions
        let sort_score =
            Self::score_dimension(&desc_lower, &["sort", "order", "rank", "compare", "sorted"]);
        let search_score = Self::score_dimension(
            &desc_lower,
            &["search", "find", "locate", "query", "lookup"],
        );
        let parse_score =
            Self::score_dimension(&desc_lower, &["parse", "tokenize", "lex", "grammar"]);
        let graph_score =
            Self::score_dimension(&desc_lower, &["graph", "tree", "node", "edge", "vertex"]);
        let conc_score = Self::score_dimension(&desc_lower, &["concurr", "parallel", "thread"]);
        let ml_score = Self::score_dimension(&desc_lower, &["train", "model", "predict", "neural"]);

        // Find the dominant category
        let mut scores: Vec<(f32, &str, Vec<&str>)> = vec![
            (
                sort_score,
                "sort",
                vec![
                    "QuickSort/MergeSort O(n log n)",
                    "Partition/pivot selection",
                    "In-place merge or auxiliary buffer",
                ],
            ),
            (
                search_score,
                "search",
                vec![
                    "Binary search O(log n) for sorted data",
                    "Hash-based O(1) lookup for unsorted",
                    "Trie for string prefix matching",
                ],
            ),
            (
                parse_score,
                "parse",
                vec![
                    "Tokenize input into lexemes",
                    "Build AST via recursive descent",
                    "Error recovery with panic mode",
                ],
            ),
            (
                graph_score,
                "graph",
                vec![
                    "Adjacency list representation",
                    "BFS/DFS traversal with visited set",
                    "Dijkstra/Bellman-Ford for paths",
                ],
            ),
            (
                conc_score,
                "concurrency",
                vec![
                    "Work-stealing thread pool",
                    "Divide work into independent tasks",
                    "Join/barrier synchronization",
                ],
            ),
            (
                ml_score,
                "ml",
                vec![
                    "Feature extraction and normalization",
                    "Forward pass through model layers",
                    "Backpropagation or inference only",
                ],
            ),
        ];
        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((score, category, algo_steps)) = scores.first() {
            if *score > 0.3 {
                for (i, step_desc) in algo_steps.iter().enumerate() {
                    add_step(
                        &mut design,
                        step,
                        &format!("Core algorithm ({} phase {})", category, i + 1),
                        step_desc,
                    );
                    step += 1;
                }
            }
        }

        // If no specific category matched strongly, use generic approach
        if sort_score <= 0.3
            && search_score <= 0.3
            && parse_score <= 0.3
            && graph_score <= 0.3
            && conc_score <= 0.3
            && ml_score <= 0.3
        {
            if task
                .constraints
                .iter()
                .any(|c| c.to_lowercase().contains("recursive") || c.contains("recursion"))
            {
                add_step(
                    &mut design,
                    step,
                    "Core logic",
                    "Recursive approach with memoization and explicit recursion limit.",
                );
            } else {
                add_step(
                    &mut design,
                    step,
                    "Core logic",
                    "Iterative approach with proper error handling and validation.",
                );
            }
            step += 1;
        }

        // Phase 3: Output formatting (always last)
        add_step(
            &mut design,
            step,
            "Output formatting",
            "Format result according to expected return type, handle error propagation.",
        );
        step += 1;

        // Add constraint-driven algorithm choices
        let mut constraints_applied = false;
        for constraint in &task.constraints {
            let cl = constraint.to_lowercase();
            if cl.contains("O(1)") && !constraints_applied {
                add_step(
                    &mut design,
                    step,
                    "Optimization",
                    "Design for constant-time operations with precomputation or hash-based lookup.",
                );
                step += 1;
                constraints_applied = true;
            } else if cl.contains("O(log n)") && !constraints_applied {
                add_step(
                    &mut design,
                    step,
                    "Optimization",
                    "Binary search or balanced tree structure for logarithmic complexity.",
                );
                step += 1;
                constraints_applied = true;
            } else if cl.contains("O(n)") && !constraints_applied {
                add_step(
                    &mut design,
                    step,
                    "Optimization",
                    "Single-pass linear scan with hash set for deduplication.",
                );
                step += 1;
                constraints_applied = true;
            }
        }

        design.push_str(&format!(
            "\n\nDominant category: {} (score: {:.2})",
            if sort_score > 0.3 {
                "sort"
            } else if search_score > 0.3 {
                "search"
            } else if parse_score > 0.3 {
                "parse"
            } else if graph_score > 0.3 {
                "graph"
            } else if conc_score > 0.3 {
                "concurrency"
            } else if ml_score > 0.3 {
                "ml"
            } else {
                "generic"
            },
            scores.first().map(|(s, _, _)| *s).unwrap_or(0.0)
        ));

        Ok(design)
    }

    /// Identify potential edge cases through property-based systematic analysis
    /// For each detected data structure or operation type, generates
    /// property-specific edge cases using structural reasoning
    async fn identify_edge_cases(&self, task: &CodingTask) -> SACAResult<Vec<String>> {
        let desc_lower = task.description.to_lowercase();
        let _all_text = {
            let mut t = task.description.clone();
            for req in &task.requirements {
                t.push_str(" ");
                t.push_str(req);
            }
            for con in &task.constraints {
                t.push_str(" ");
                t.push_str(con);
            }
            t.to_lowercase()
        };
        let mut edge_cases: Vec<String> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut add_unique = |e: String| {
            if seen.insert(e.clone()) {
                edge_cases.push(e);
            }
        };

        // Universal edge cases (property-based: identity, boundary, nullability)
        add_unique("[IDENTITY] Empty input — test that identity operation on empty returns expected base case".to_string());
        add_unique(
            "[NULLABILITY] Null/None values — ensure Option/Result is handled without unwrap"
                .to_string(),
        );
        add_unique(
            "[BOUNDARY] Maximum size inputs — test memory and performance limits".to_string(),
        );
        add_unique("[TYPE] Invalid data types — test type enforcement at boundaries".to_string());

        // Data structure specific edge cases with property reasoning
        if Self::score_dimension(
            &desc_lower,
            &["array", "list", "vec", "slice", "collection"],
        ) > 0.3
        {
            add_unique("[PROPERTY:identity] Single element collection".to_string());
            add_unique("[PROPERTY:invariant] All elements identical".to_string());
            add_unique("[PROPERTY:invariant] Elements in reverse order".to_string());
            add_unique(
                "[PROPERTY:boundary] Already sorted / already in expected order".to_string(),
            );
            add_unique("[PROPERTY:invariant] Duplicate elements".to_string());
            add_unique("[PROPERTY:boundary] Very large collection (10^5+ elements)".to_string());
        }
        if Self::score_dimension(&desc_lower, &["map", "hash", "dict", "hashmap"]) > 0.3 {
            add_unique("[PROPERTY:collision] Collision-heavy keys (all same hash)".to_string());
            add_unique("[PROPERTY:identity] Lookup of missing key".to_string());
            add_unique(
                "[PROPERTY:idempotency] Overwrite of existing key produces correct state"
                    .to_string(),
            );
            add_unique(
                "[PROPERTY:boundary] Very large number of entries (hash table resize)".to_string(),
            );
        }
        if Self::score_dimension(&desc_lower, &["tree", "bst", "heap", "trie"]) > 0.3 {
            add_unique("[PROPERTY:shape] Skewed tree (all left / all right) — worst case for unbalanced trees".to_string());
            add_unique("[PROPERTY:identity] Single node".to_string());
            add_unique("[PROPERTY:shape] Unbalanced insert sequence".to_string());
            add_unique("[PROPERTY:invariant] Duplicate key handling".to_string());
        }
        if Self::score_dimension(&desc_lower, &["graph", "node", "edge", "vertex"]) > 0.3 {
            add_unique("[PROPERTY:connectivity] Disconnected components".to_string());
            add_unique("[PROPERTY:cycle] Cycles (including self-loops)".to_string());
            add_unique("[PROPERTY:identity] Single vertex with no edges".to_string());
            add_unique("[PROPERTY:shape] Complete graph (dense connectivity)".to_string());
        }

        // Operation-specific edge cases
        if Self::score_dimension(&desc_lower, &["recursive", "recursion", "recurse"]) > 0.3 {
            add_unique("[BOUNDARY] Deep recursion hitting stack limit (~5000-10000 frames depending on frame size)".to_string());
        }
        if Self::score_dimension(&desc_lower, &["sort", "order", "rank", "compare"]) > 0.3 {
            add_unique("[PROPERTY:invariant] Already sorted input (best case)".to_string());
            add_unique(
                "[PROPERTY:invariant] Reverse sorted input (worst case for some algorithms)"
                    .to_string(),
            );
            add_unique(
                "[PROPERTY:stability] Equal elements — verify stable sort preserves original order"
                    .to_string(),
            );
        }
        if Self::score_dimension(
            &desc_lower,
            &["numeric", "number", "count", "integer", "float"],
        ) > 0.3
        {
            add_unique("[BOUNDARY] Zero value".to_string());
            add_unique("[BOUNDARY] Negative values".to_string());
            add_unique(
                "[BOUNDARY] Integer overflow / underflow (use checked arithmetic)".to_string(),
            );
            add_unique("[BOUNDARY] Floating point precision (0.1 + 0.2 != 0.3)".to_string());
            add_unique("[BOUNDARY] NaN / Infinity in floating point operations".to_string());
            add_unique(
                "[BOUNDARY] Maximum/minimum integer values (i32::MAX, i64::MIN, etc.)".to_string(),
            );
        }
        if Self::score_dimension(&desc_lower, &["string", "text", "char", "utf"]) > 0.3 {
            add_unique("[IDENTITY] Empty string".to_string());
            add_unique(
                "[ENCODING] Unicode / multi-byte characters (emoji, CJK, combining marks)"
                    .to_string(),
            );
            add_unique("[IDENTITY] Whitespace-only string".to_string());
            add_unique("[BOUNDARY] Very long string (10^6+ characters)".to_string());
            add_unique(
                "[ENCODING] Special characters and escaping (null bytes, control chars)"
                    .to_string(),
            );
        }
        if Self::score_dimension(
            &desc_lower,
            &["file", "io", "read", "write", "persist", "storage"],
        ) > 0.3
        {
            add_unique("[RESOURCE] File not found".to_string());
            add_unique("[RESOURCE] Permission denied".to_string());
            add_unique("[RESOURCE] File locked by another process".to_string());
            add_unique("[INTEGRITY] Partial read / truncated data".to_string());
            add_unique("[RESOURCE] Disk full during write operation".to_string());
        }
        if Self::score_dimension(
            &desc_lower,
            &["network", "http", "api", "socket", "tcp", "url"],
        ) > 0.3
        {
            add_unique("[TIMING] Connection timeout".to_string());
            add_unique("[PROTOCOL] Server error (5xx) — retry vs fail semantics".to_string());
            add_unique("[RESOURCE] Rate limiting / throttling".to_string());
            add_unique("[ENCODING] Malformed response body".to_string());
            add_unique("[PROTOCOL] Redirect handling (301, 302, 307)".to_string());
        }

        // Traverse requirements for additional edge-case signals
        for req in &task.requirements {
            let rl = req.to_lowercase();
            if rl.contains("concurr") || rl.contains("thread") || rl.contains("parallel") {
                add_unique("[CONCURRENCY] Race condition under concurrent access".to_string());
                add_unique("[CONCURRENCY] Deadlock with multiple locks".to_string());
                add_unique("[CONCURRENCY] Thread starvation under high contention".to_string());
            }
            if rl.contains("deterministic") || rl.contains("idempotent") {
                add_unique(
                    "[PROPERTY] Operation is idempotent — repeated calls produce same result"
                        .to_string(),
                );
            }
        }

        Ok(edge_cases)
    }

    /// Identify underlying assumptions through structural analysis
    /// Extracts implicit assumptions from code structure, constraints, and requirements
    async fn identify_assumptions(&self, task: &CodingTask) -> SACAResult<Vec<String>> {
        let desc_lower = task.description.to_lowercase();
        let _all_text = {
            let mut t = task.description.clone();
            for req in &task.requirements {
                t.push_str(" ");
                t.push_str(req);
            }
            for con in &task.constraints {
                t.push_str(" ");
                t.push_str(con);
            }
            t.to_lowercase()
        };
        let mut assumptions: Vec<String> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut add_unique = |a: String| {
            if seen.insert(a.clone()) {
                assumptions.push(a);
            }
        };

        // Universal assumptions (always present)
        add_unique("Input data is in expected format and encoding".to_string());
        add_unique("Sufficient memory and compute resources available".to_string());
        add_unique("Environment supports required language features and libraries".to_string());

        // Structurally derived assumptions using scored dimensions
        if Self::score_dimension(&desc_lower, &["sort", "order", "rank", "compare", "sorted"]) > 0.3
        {
            add_unique("Elements implement a total order (comparable trait)".to_string());
            add_unique("Comparison function is consistent and transitive (non-transitive compare = undefined behavior)".to_string());
            if !desc_lower.contains("stable") {
                add_unique(
                    "Sort stability is not required — equal elements may be reordered".to_string(),
                );
            }
        }
        if Self::score_dimension(
            &desc_lower,
            &["numeric", "count", "math", "number", "integer", "float"],
        ) > 0.3
        {
            add_unique(
                "Numeric values fit within standard integer/float ranges (no overflow)".to_string(),
            );
            add_unique("Division by zero will not occur".to_string());
            add_unique(
                "Integer operations will not overflow — use checked_add/sub/mul otherwise"
                    .to_string(),
            );
            if desc_lower.contains("float")
                || desc_lower.contains("f32")
                || desc_lower.contains("f64")
            {
                add_unique(
                    "Floating point NaN/Infinity values are handled or excluded from input"
                        .to_string(),
                );
            }
        }
        if Self::score_dimension(
            &desc_lower,
            &["search", "find", "locate", "query", "lookup"],
        ) > 0.3
        {
            add_unique(
                "Search predicate is deterministic (same key → same result for same data)"
                    .to_string(),
            );
            if Self::score_dimension(&desc_lower, &["hash"]) == 0.0 {
                add_unique(
                    "Data structure supports the required search operation efficiently".to_string(),
                );
            } else {
                add_unique(
                    "Hash function distributes keys uniformly to avoid pathological collisions"
                        .to_string(),
                );
            }
        }
        if Self::score_dimension(&desc_lower, &["string", "text", "char", "utf"]) > 0.3 {
            add_unique(
                "Input strings are valid UTF-8 encoded (not arbitrary byte sequences)".to_string(),
            );
            add_unique(
                "String comparison uses Unicode-aware semantics where appropriate".to_string(),
            );
        }
        if Self::score_dimension(
            &desc_lower,
            &["concurr", "thread", "parallel", "async", "sync"],
        ) > 0.3
        {
            add_unique("Operations are safely concurrent or properly synchronized".to_string());
            add_unique(
                "Shared mutable state is protected by synchronization primitives".to_string(),
            );
            add_unique(
                "Tasks are sufficiently independent to benefit from parallel execution".to_string(),
            );
        }
        if Self::score_dimension(&desc_lower, &["file", "persist", "disk", "storage"]) > 0.3 {
            add_unique("File system has sufficient space and appropriate permissions".to_string());
            add_unique("File paths are valid and accessible at runtime".to_string());
            add_unique("File operations are atomic or handle partial writes correctly".to_string());
        }
        if Self::score_dimension(
            &desc_lower,
            &["network", "api", "http", "socket", "rest", "rpc"],
        ) > 0.3
        {
            add_unique(
                "Network is available with acceptable latency (< timeout threshold)".to_string(),
            );
            add_unique("Remote service conforms to the expected API contract (schema, types, status codes)".to_string());
            add_unique("Network failures are transient and retryable".to_string());
        }
        if Self::score_dimension(&desc_lower, &["generic", "template", "trait", "type param"]) > 0.3
        {
            add_unique("Generic type parameters satisfy the required trait bounds".to_string());
            add_unique("Monomorphization does not cause excessive code bloat".to_string());
        }

        // Derive implicit assumptions from constraints
        for constraint in &task.constraints {
            let cl = constraint.to_lowercase();
            if cl.contains("complexity") || cl.contains("big-o") || cl.contains("O(") {
                add_unique(format!(
                    "Input size assumptions match the {} complexity guarantee",
                    constraint
                ));
            }
            if cl.contains("memory") || cl.contains("ram") {
                add_unique(format!(
                    "Working set fits within the specified memory budget: {}",
                    constraint
                ));
            }
            if cl.contains("real") || cl.contains("deadline") || cl.contains("latency") {
                add_unique(format!(
                    "System meets the real-time performance requirements: {}",
                    constraint
                ));
            }
            if cl.contains("safe") || cl.contains("no panic") || cl.contains("no unsafe") {
                add_unique(format!(
                    "Code must not panic on any valid input: {}",
                    constraint
                ));
            }
            if cl.contains("deterministic") || cl.contains("idempotent") {
                add_unique(format!(
                    "Operation must produce identical results across calls: {}",
                    constraint
                ));
            }
        }

        Ok(assumptions)
    }

    /// Assess implementation risks through systematic analysis
    /// Evaluates complexity, failure modes, and domain-specific hazards
    async fn assess_risks(&self, task: &CodingTask) -> SACAResult<Vec<String>> {
        let desc_lower = task.description.to_lowercase();
        let _all_text = {
            let mut t = task.description.clone();
            for req in &task.requirements {
                t.push_str(" ");
                t.push_str(req);
            }
            for con in &task.constraints {
                t.push_str(" ");
                t.push_str(con);
            }
            t.to_lowercase()
        };
        let mut risks: std::collections::BTreeMap<String, String> =
            std::collections::BTreeMap::new();
        let risk_insert =
            |risks: &mut std::collections::BTreeMap<String, String>, id: &str, desc: String| {
                risks.entry(id.to_string()).or_insert(desc);
            };

        // Universal risks with complexity analysis
        risk_insert(
            &mut risks,
            "perf_scale",
            "[COMPLEXITY] Performance degradation with large inputs exceeding O(n) expected scale"
                .to_string(),
        );
        risk_insert(
            &mut risks,
            "memory_oom",
            "[COMPLEXITY] Memory overflow or excessive allocation leading to OOM".to_string(),
        );
        risk_insert(
            &mut risks,
            "error_handling",
            "[FAILURE-MODE] Incorrect error handling or missing edge case paths".to_string(),
        );
        risk_insert(
            &mut risks,
            "regression",
            "[MAINTENANCE] Regression in existing functionality when adding new code".to_string(),
        );

        // Analyze constraint-derived complexity risks
        for constraint in &task.constraints {
            let cl = constraint.to_lowercase();
            if cl.contains("O(1)") {
                risk_insert(&mut risks, "const_O1", "[COMPLEXITY] O(1) constraint may be violated by hidden iteration (e.g., .len() on LinkedList)".to_string());
            } else if cl.contains("O(log n)") {
                risk_insert(&mut risks, "const_logn", "[COMPLEXITY] O(log n) constraint requires balanced structures — unbalanced input degrades to O(n)".to_string());
            } else if cl.contains("O(n)") {
                risk_insert(&mut risks, "const_On", "[COMPLEXITY] O(n) constraint violated by nested loops or hidden O(n) operations inside the loop".to_string());
            } else if cl.contains("sorted") {
                risk_insert(&mut risks, "const_sorted", "[FAILURE-MODE] Sorted input assumption — unsorted input will produce incorrect results".to_string());
            }
        }

        // Domain-specific risks with failure mode classification
        if Self::score_dimension(&desc_lower, &["sort", "order", "rank", "compare"]) > 0.3 {
            risk_insert(
                &mut risks,
                "sort_stability",
                "[CORRECTNESS] Unstable sort when stability is required by downstream consumers"
                    .to_string(),
            );
            risk_insert(&mut risks, "sort_quadratic", "[PERFORMANCE] Quadratic O(n²) performance on nearly-sorted data with naive pivot selection".to_string());
            risk_insert(
                &mut risks,
                "sort_compare_inconsistency",
                "[CORRECTNESS] Non-transitive comparison function causes undefined behavior"
                    .to_string(),
            );
        }
        if Self::score_dimension(
            &desc_lower,
            &["search", "find", "locate", "query", "lookup"],
        ) > 0.3
        {
            risk_insert(
                &mut risks,
                "search_false_negative",
                "[CORRECTNESS] False negatives due to incorrect comparison or hash collision"
                    .to_string(),
            );
            risk_insert(
                &mut risks,
                "search_oob",
                "[SAFETY] Index out of bounds in edge cases (empty collection, single element)"
                    .to_string(),
            );
            if Self::score_dimension(&desc_lower, &["hash"]) > 0.3 {
                risk_insert(
                    &mut risks,
                    "search_hash_collision",
                    "[PERFORMANCE] Hash collision attack causing O(n) degradation in HashMap"
                        .to_string(),
                );
            }
        }
        if Self::score_dimension(&desc_lower, &["parse", "tokenize", "lex", "grammar"]) > 0.3 {
            risk_insert(
                &mut risks,
                "parse_infinite_loop",
                "[CORRECTNESS] Malformed input causing infinite loop or excessive backtracking"
                    .to_string(),
            );
            risk_insert(
                &mut risks,
                "parse_unicode",
                "[CORRECTNESS] Unicode/encoding issues with multi-byte characters".to_string(),
            );
            risk_insert(
                &mut risks,
                "parse_oom",
                "[RESILIENCE] Catastrophic backtracking (ReDoS) on adversarial input".to_string(),
            );
        }
        if Self::score_dimension(&desc_lower, &["recursive", "recursion", "recurse"]) > 0.3 {
            risk_insert(&mut risks, "recursion_stack", "[SAFETY] Stack overflow for deep recursion (default stack ~8MB, ~5000-10000 frames)".to_string());
        }
        if Self::score_dimension(
            &desc_lower,
            &["concurr", "thread", "parallel", "async", "race"],
        ) > 0.3
        {
            risk_insert(
                &mut risks,
                "concurrency_race",
                "[SAFETY] Data race due to unsynchronized shared state".to_string(),
            );
            risk_insert(
                &mut risks,
                "concurrency_deadlock",
                "[SAFETY] Deadlock from incorrect lock ordering".to_string(),
            );
            risk_insert(
                &mut risks,
                "concurrency_starvation",
                "[PERFORMANCE] Thread starvation or excessive context switching".to_string(),
            );
            risk_insert(
                &mut risks,
                "concurrency_poison",
                "[RESILIENCE] Mutex poisoning crashes the entire system on one thread failure"
                    .to_string(),
            );
        }
        if Self::score_dimension(&desc_lower, &["network", "http", "api", "socket", "tcp"]) > 0.3 {
            risk_insert(
                &mut risks,
                "network_timeout",
                "[RESILIENCE] Unhandled network timeout causing indefinite hang".to_string(),
            );
            risk_insert(
                &mut risks,
                "network_leak",
                "[RESILIENCE] Leaked connections or file descriptors".to_string(),
            );
            risk_insert(
                &mut risks,
                "network_retry",
                "[CORRECTNESS] Missing retry logic causes transient failures".to_string(),
            );
        }
        if Self::score_dimension(&desc_lower, &["file", "persist", "disk", "io", "storage"]) > 0.3 {
            risk_insert(
                &mut risks,
                "io_partial_write",
                "[CORRECTNESS] Partial write / corrupted data on crash".to_string(),
            );
            risk_insert(
                &mut risks,
                "io_race",
                "[SAFETY] Race condition from concurrent file access".to_string(),
            );
        }
        if Self::score_dimension(&desc_lower, &["unsafe", "pointer", "raw", "ffi"]) > 0.3 {
            risk_insert(
                &mut risks,
                "unsafe_ub",
                "[SAFETY] Undefined behavior from incorrect unsafe code".to_string(),
            );
            risk_insert(
                &mut risks,
                "unsafe_memory",
                "[SAFETY] Memory safety violation in unsafe blocks".to_string(),
            );
        }

        // Requirements-driven risk analysis
        for req in &task.requirements {
            let rl = req.to_lowercase();
            if rl.contains("fast") || rl.contains("perform") {
                risk_insert(
                    &mut risks,
                    "over_optimization",
                    "[MAINTENANCE] Over-optimization leading to unreadable or unmaintainable code"
                        .to_string(),
                );
            }
            if rl.contains("generic") || rl.contains("template") || rl.contains("polymorphic") {
                risk_insert(&mut risks, "monomorphization", "[PERFORMANCE] Monomorphization bloat from excessive generic usage increases binary size and compile time".to_string());
            }
            if rl.contains("lock") || rl.contains("mutex") {
                risk_insert(&mut risks, "lock_contention", "[PERFORMANCE] Lock contention under high concurrency reduces throughput to serial".to_string());
            }
            if rl.contains("async") || rl.contains("future") {
                risk_insert(
                    &mut risks,
                    "async_pin",
                    "[CORRECTNESS] Incorrect pinning or self-referential structs in async code"
                        .to_string(),
                );
            }
        }

        Ok(risks.into_values().collect())
    }

    /// Define implementation approach from synthesized reasoning
    async fn define_approach(
        &self,
        task: &CodingTask,
        steps: &[ReasoningStep],
    ) -> SACAResult<String> {
        let mut approach = String::new();

        // Synthesize approach from the reasoning steps already performed
        let step_descriptions: Vec<&str> = steps.iter().map(|s| s.description.as_str()).collect();

        approach.push_str(&format!(
            "Implementation approach synthesized from {} reasoning steps:\n",
            steps.len()
        ));

        for (i, desc) in step_descriptions.iter().enumerate() {
            approach.push_str(&format!("  Phase {}: {}\n", i + 1, desc));
        }

        // Add strategy based on the task type
        let desc_lower = task.description.to_lowercase();
        if desc_lower.contains("api") || desc_lower.contains("service") {
            approach.push_str(
                "Strategy: Build interface first, then implement, write integration tests. ",
            );
            approach.push_str("Use contract testing to validate API boundaries. ");
        } else if desc_lower.contains("algorithm") || desc_lower.contains("function") {
            approach
                .push_str("Strategy: Implement pure function first, add property-based tests. ");
            approach.push_str("Benchmark with representative inputs. ");
        } else if desc_lower.contains("data")
            || desc_lower.contains("pipeline")
            || desc_lower.contains("stream")
        {
            approach.push_str(
                "Strategy: Start with data contract/format, build pipeline incrementally. ",
            );
            approach.push_str("Validate at each stage with integration tests. ");
        } else {
            approach.push_str(
                "Strategy: Validate inputs, implement core logic, handle errors, format outputs. ",
            );
            approach.push_str("Write tests in parallel with implementation (TDD if applicable). ");
        }

        // Incorporate requirements into the approach
        if !task.requirements.is_empty() {
            approach.push_str("\nRequirements-driven priorities:\n");
            for req in &task.requirements {
                approach.push_str(&format!("  - {}\n", req));
            }
        }

        Ok(approach)
    }

    /// Add deep reasoning steps tied to the actual task structure
    async fn add_deep_reasoning_steps(
        &self,
        steps: &mut Vec<ReasoningStep>,
        task: &CodingTask,
    ) -> SACAResult<()> {
        let desc_lower = task.description.to_lowercase();

        // Performance characteristics - derive from task type
        let perf_logic = if desc_lower.contains("sort") || desc_lower.contains("search") {
            "Analyze time complexity: O(n log n) for comparisons, O(n) for non-comparison sorts. \
             Space complexity: in-place vs auxiliary memory. \
             Best/worst/average case analysis."
        } else if desc_lower.contains("parse") || desc_lower.contains("lex") {
            "Linearly scan input O(n) for tokenization. \
             Lookahead may be O(n*k) for backtracking parsers. \
             Memory scales with AST depth, not input size."
        } else {
            "Analyze time and space complexity for best/worst/average input sizes. \
             Identify hot paths that dominate execution time. \
             Consider precomputation vs on-demand tradeoffs."
        };
        steps.push(ReasoningStep {
            step_number: steps.len() as u32 + 1,
            description: "Analyze performance characteristics".to_string(),
            logic: perf_logic.to_string(),
            expected_outcome: "Performance optimization strategies identified".to_string(),
        });

        // Testing strategy - derive from task structure
        let test_logic = if desc_lower.contains("sort") {
            "Unit: empty, single, sorted, reverse, duplicates, random large arrays. \
             Property: output is sorted and is permutation of input. \
             Performance: benchmark on 10^5 elements."
        } else if desc_lower.contains("parse") {
            "Unit: valid input, empty, malformed, partial, unicode. \
             Fuzz: random-generated inputs. \
             Integration: round-trip parse(format(x)) == x."
        } else {
            "Unit tests for core logic and edge cases. \
             Integration tests for component interactions. \
             Property-based tests for invariants. \
             Performance benchmarks for critical paths."
        };
        steps.push(ReasoningStep {
            step_number: steps.len() as u32 + 1,
            description: "Define comprehensive testing strategy".to_string(),
            logic: test_logic.to_string(),
            expected_outcome: "Complete test coverage plan".to_string(),
        });

        Ok(())
    }

    /// Add exhaustive reasoning steps
    async fn add_exhaustive_reasoning_steps(
        &self,
        steps: &mut Vec<ReasoningStep>,
        task: &CodingTask,
    ) -> SACAResult<()> {
        let desc_lower = task.description.to_lowercase();

        // Alternative approaches - derive from structure
        let alt_logic = if desc_lower.contains("sort") {
            "QuickSort (in-place, average O(n log n), worst O(n^2)). \
             MergeSort (stable, O(n log n), O(n) space). \
             HeapSort (in-place, O(n log n), not stable). \
             RadixSort (O(n*k) if input is integer)."
        } else if desc_lower.contains("search") {
            "Linear scan O(n) for unsorted. \
             Binary search O(log n) for sorted. \
             Hash table O(1) average. \
             Trie for string search. \
             B-tree for disk-backed search."
        } else {
            "Evaluate at least three approaches before committing. \
             Consider library-based vs custom implementation. \
             Prototype the simplest viable approach first, optimize later."
        };
        steps.push(ReasoningStep {
            step_number: steps.len() as u32 + 1,
            description: "Consider alternative implementation approaches".to_string(),
            logic: alt_logic.to_string(),
            expected_outcome: "Backup implementation strategies identified".to_string(),
        });

        // Documentation requirements - generic but useful
        steps.push(ReasoningStep {
            step_number: steps.len() as u32 + 1,
            description: "Define documentation requirements".to_string(),
            logic: "Document module-level purpose, public API with examples, \
                    error conditions, complexity guarantees, and usage notes. \
                    Include inline comments for non-obvious logic and safety invariants."
                .to_string(),
            expected_outcome: "Comprehensive documentation plan".to_string(),
        });

        // Maintenance considerations - derive from task
        let maint_logic = if desc_lower.contains("api")
            || desc_lower.contains("interface")
            || desc_lower.contains("trait")
        {
            "Design for backward compatibility via versioned interfaces. \
             Use semantic versioning for breaking changes. \
             Deprecate gradually rather than remove abruptly."
        } else {
            "Organize code into small focused modules with clear boundaries. \
             Minimize public surface area. \
             Add validation gates for configuration changes. \
             Write defensive assertions for internal invariants."
        };
        steps.push(ReasoningStep {
            step_number: steps.len() as u32 + 1,
            description: "Consider maintenance and extensibility".to_string(),
            logic: maint_logic.to_string(),
            expected_outcome: "Maintainable and extensible design".to_string(),
        });

        Ok(())
    }

    /// Generate cache key for reasoning results
    fn generate_cache_key(&self, task: &CodingTask) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        task.description.hash(&mut hasher);
        task.requirements.hash(&mut hasher);
        task.constraints.hash(&mut hasher);
        format!("cot_{:x}", hasher.finish())
    }

    /// Clear reasoning cache
    pub async fn clear_cache(&self) {
        self.reasoning_cache.write().await.clear();
        info!("CoT reasoning cache cleared");
    }

    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> (usize, usize) {
        let cache = self.reasoning_cache.read().await;
        (cache.len(), cache.capacity())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cot_reasoning() {
        let config = CoTConfig::default();
        let cot = CoTEngine::new(config).expect("Failed to create CoT engine");

        let task = CodingTask {
            description: "Create a function that sorts an array of integers".to_string(),
            requirements: vec!["Use efficient sorting algorithm".to_string()],
            constraints: vec!["O(n log n) complexity".to_string()],
            context: None,
        };

        let result = cot.reason(&task).await.expect("CoT reasoning failed");
        assert!(!result.reasoning_steps.is_empty());
        assert!(!result.edge_cases.is_empty());
        assert!(!result.assumptions.is_empty());
        assert!(!result.risks.is_empty());
    }

    #[tokio::test]
    async fn test_cot_cache() {
        let config = CoTConfig::default();
        let cot = CoTEngine::new(config).expect("Failed to create CoT engine");

        let task = CodingTask {
            description: "Test task".to_string(),
            requirements: vec![],
            constraints: vec![],
            context: None,
        };

        let result1 = cot.reason(&task).await.expect("CoT reasoning failed");
        let result2 = cot.reason(&task).await.expect("CoT reasoning failed");
        assert_eq!(result1.task_analysis, result2.task_analysis);

        let stats = cot.get_cache_stats().await;
        assert_eq!(stats.0, 1); // 1 item in cache
    }
}
