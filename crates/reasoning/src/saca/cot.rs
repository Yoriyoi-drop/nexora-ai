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

    /// Analyze the task requirements using structured pattern analysis
    async fn analyze_task(&self, task: &CodingTask) -> SACAResult<String> {
        let desc_lower = task.description.to_lowercase();
        let mut dimensions = Vec::new();

        // Data processing dimension
        if desc_lower.contains("array")
            || desc_lower.contains("list")
            || desc_lower.contains("collection")
            || desc_lower.contains("data")
            || desc_lower.contains("stream")
        {
            dimensions.push("data_processing");
        }
        // IO dimension
        if desc_lower.contains("read")
            || desc_lower.contains("write")
            || desc_lower.contains("parse")
            || desc_lower.contains("serialize")
            || desc_lower.contains("format")
        {
            dimensions.push("io");
        }
        // Network dimension
        if desc_lower.contains("request")
            || desc_lower.contains("response")
            || desc_lower.contains("http")
            || desc_lower.contains("api")
            || desc_lower.contains("client")
            || desc_lower.contains("server")
        {
            dimensions.push("network");
        }
        // State management
        if desc_lower.contains("state")
            || desc_lower.contains("cache")
            || desc_lower.contains("store")
            || desc_lower.contains("persist")
            || desc_lower.contains("memo")
        {
            dimensions.push("state_management");
        }
        // Concurrency dimension
        if desc_lower.contains("thread")
            || desc_lower.contains("async")
            || desc_lower.contains("parallel")
            || desc_lower.contains("concurrent")
            || desc_lower.contains("sync")
            || desc_lower.contains("lock")
        {
            dimensions.push("concurrency");
        }
        // Computation dimension
        if desc_lower.contains("compute")
            || desc_lower.contains("calculate")
            || desc_lower.contains("transform")
            || desc_lower.contains("process")
            || desc_lower.contains("map")
            || desc_lower.contains("reduce")
        {
            dimensions.push("computation");
        }

        let dimension_str = if dimensions.is_empty() {
            "general_purpose".to_string()
        } else {
            dimensions.join(", ")
        };

        let mut analysis = format!(
            "Task: {}\nRequirements: {}\nConstraints: {}\nDimensions: [{}]\nContext: {}",
            task.description,
            task.requirements.join(", "),
            task.constraints.join(", "),
            dimension_str,
            task.context
                .as_ref()
                .map(|c| format!("Repository: {:?}", c.repository_path))
                .unwrap_or_else(|| "None".to_string())
        );

        // Add constraint-driven analysis
        for constraint in &task.constraints {
            let cl = constraint.to_lowercase();
            if cl.contains("complexity") || cl.contains("performance") {
                analysis.push_str(&format!("\n  - Constraint insight: '{}' requires algorithmic focus on efficiency", constraint));
            }
            if cl.contains("memory") || cl.contains("resource") {
                analysis.push_str(&format!("\n  - Constraint insight: '{}' requires space-efficient design", constraint));
            }
            if cl.contains("safe") || cl.contains("error") || cl.contains("robust") {
                analysis.push_str(&format!("\n  - Constraint insight: '{}' requires defensive programming", constraint));
            }
        }

        Ok(analysis)
    }

    /// Identify key components needed through structural analysis
    async fn identify_components(&self, task: &CodingTask) -> SACAResult<String> {
        let desc_lower = task.description.to_lowercase();
        let mut components: Vec<String> = Vec::new();

        // Analyze description for structural categories
        if desc_lower.contains("sort")
            || desc_lower.contains("order")
            || desc_lower.contains("rank")
        {
            components.push("Input collection".to_string());
            components.push("Element comparison strategy".to_string());
            components.push("Sorting algorithm implementation".to_string());
            components.push("Ordered output assembly".to_string());
        }
        if desc_lower.contains("search")
            || desc_lower.contains("find")
            || desc_lower.contains("locate")
            || desc_lower.contains("query")
        {
            components.push("Data structure for search (index/collection)".to_string());
            components.push("Search predicate or key extraction".to_string());
            components.push("Search algorithm (binary, linear, hash)".to_string());
            components.push("Result handling (found/not-found)".to_string());
        }
        if desc_lower.contains("parse")
            || desc_lower.contains("tokenize")
            || desc_lower.contains("lex")
        {
            components.push("Input reader / tokenizer".to_string());
            components.push("Grammar or pattern definitions".to_string());
            components.push("Parser state machine".to_string());
            components.push("Abstract syntax tree or output structure".to_string());
        }
        if desc_lower.contains("transform")
            || desc_lower.contains("convert")
            || desc_lower.contains("encode")
            || desc_lower.contains("decode")
        {
            components.push("Input validator and normalizer".to_string());
            components.push("Transformation function".to_string());
            components.push("Output encoder".to_string());
        }
        if desc_lower.contains("aggregate")
            || desc_lower.contains("summarize")
            || desc_lower.contains("group")
        {
            components.push("Data partitioner / grouper".to_string());
            components.push("Aggregation function (sum, count, avg)".to_string());
            components.push("Result combiner".to_string());
        }

        // If no specific categories matched, derive from requirements
        if components.is_empty() {
            for req in &task.requirements {
                components.push(format!("Component from requirement: {}", req));
            }
            if components.is_empty() {
                components.push("Core logic implementation".to_string());
                components.push("Input validation".to_string());
                components.push("Error handling".to_string());
                components.push("Output formatting".to_string());
            }
        }

        // Append data-driven components from requirements
        for req in &task.requirements {
            let rl = req.to_lowercase();
            if rl.contains("type") || rl.contains("struct") {
                components.push("Type definitions / data structures".to_string());
            }
            if rl.contains("trait") || rl.contains("interface") {
                components.push("Trait or interface definitions".to_string());
            }
            if rl.contains("config") || rl.contains("setting") {
                components.push("Configuration system".to_string());
            }
        }

        Ok(format!("Key components identified: {}", components.join(", ")))
    }

    /// Design appropriate algorithm based on structural analysis
    async fn design_algorithm(&self, task: &CodingTask) -> SACAResult<String> {
        let desc_lower = task.description.to_lowercase();
        let mut design = String::from("Algorithm design: ");

        // Select algorithm category based on operation type
        if desc_lower.contains("sort") || desc_lower.contains("order") {
            design.push_str("Divide-and-conquer approach (QuickSort or MergeSort) ");
            design.push_str("with O(n log n) average complexity. ");
            design.push_str("Fall back to insertion sort for small partitions (< 16 elements). ");
            if desc_lower.contains("stable") {
                design.push_str("Prefer MergeSort for stability. ");
            }
        } else if desc_lower.contains("search") || desc_lower.contains("find") {
            if desc_lower.contains("sorted") || task.constraints.iter().any(|c| c.contains("sorted")) {
                design.push_str("Binary search O(log n) on sorted data. ");
                design.push_str("Prefer iterative implementation to avoid stack overflow. ");
            } else {
                design.push_str("Hash-based lookup O(1) average or linear scan O(n). ");
                design.push_str("Use HashMap/BTreeMap for repeated lookups. ");
            }
        } else if desc_lower.contains("parse") || desc_lower.contains("tokenize") {
            design.push_str("Recursive descent or Pratt parser. ");
            design.push_str("Tokenize first into tokens, then build AST in a second pass. ");
            design.push_str("Handle errors with recovery (panic mode or error tokens). ");
        } else if desc_lower.contains("graph") || desc_lower.contains("tree") {
            if desc_lower.contains("shortest") || desc_lower.contains("path") {
                design.push_str("Dijkstra's algorithm for weighted graphs, BFS for unweighted. ");
                design.push_str("Use priority queue for O(E log V) complexity. ");
            } else if desc_lower.contains("traverse") {
                design.push_str("BFS for level-order, DFS for depth-first exploration. ");
                design.push_str("Iterative with explicit stack to avoid recursion limits. ");
            } else {
                design.push_str("Adjacency list representation. ");
                design.push_str("Graph traversal with visited set for cycle detection. ");
            }
        } else if desc_lower.contains("concurr") || desc_lower.contains("parallel") {
            design.push_str("Work-stealing thread pool pattern. ");
            design.push_str("Split work into independent tasks, join results with barriers. ");
            design.push_str("Use channels (mpmc) for communication between workers. ");
        } else {
            // Generic approach: derive from constraints
            if task.constraints.iter().any(|c| c.to_lowercase().contains("recursive") || c.contains("recursion")) {
                design.push_str("Recursive approach with memoization where applicable. ");
                design.push_str("Set explicit recursion limit or prefer iterative conversion. ");
            } else {
                design.push_str("Iterative approach with proper error handling and validation. ");
            }
            design.push_str("Input validation first, then core operation, then output formatting. ");
        }

        // Add constraint-driven algorithm choices
        for constraint in &task.constraints {
            let cl = constraint.to_lowercase();
            if cl.contains("O(1)") {
                design.push_str("Design for constant-time operations with precomputation. ");
            } else if cl.contains("O(log n)") {
                design.push_str("Binary search or balanced tree structure. ");
            } else if cl.contains("O(n)") {
                design.push_str("Single-pass linear scan with hash set for dedup. ");
            } else if cl.contains("sort") {
                design.push_str("Sort as preprocessing step to enable faster algorithms. ");
            }
        }

        Ok(design)
    }

    /// Identify potential edge cases through systematic analysis
    async fn identify_edge_cases(&self, task: &CodingTask) -> SACAResult<Vec<String>> {
        let desc_lower = task.description.to_lowercase();
        let mut edge_cases: Vec<String> = Vec::new();

        // Universal edge cases
        edge_cases.push("Empty input".to_string());
        edge_cases.push("Null/None values".to_string());
        edge_cases.push("Maximum size inputs".to_string());
        edge_cases.push("Invalid data types".to_string());

        // Data structure specific
        if desc_lower.contains("array")
            || desc_lower.contains("list")
            || desc_lower.contains("vec")
        {
            edge_cases.push("Single element collection".to_string());
            edge_cases.push("All elements identical".to_string());
            edge_cases.push("Elements in reverse order".to_string());
            edge_cases.push("Already sorted".to_string());
            edge_cases.push("Duplicate elements".to_string());
        }
        if desc_lower.contains("map") || desc_lower.contains("hash") || desc_lower.contains("dict") {
            edge_cases.push("Collision-heavy keys".to_string());
            edge_cases.push("Lookup of missing key".to_string());
            edge_cases.push("Overwrite of existing key".to_string());
        }
        if desc_lower.contains("tree") {
            edge_cases.push("Skewed tree (all left / all right)".to_string());
            edge_cases.push("Single node".to_string());
            edge_cases.push("Unbalanced insert sequence".to_string());
        }
        if desc_lower.contains("graph") {
            edge_cases.push("Disconnected components".to_string());
            edge_cases.push("Cycles".to_string());
            edge_cases.push("Self-loops".to_string());
            edge_cases.push("Single vertex".to_string());
        }

        // Operation specific
        if desc_lower.contains("recursive") || desc_lower.contains("recursion") {
            edge_cases.push("Deep recursion hitting stack limit".to_string());
        }
        if desc_lower.contains("numeric")
            || desc_lower.contains("number")
            || desc_lower.contains("count")
        {
            edge_cases.push("Zero value".to_string());
            edge_cases.push("Negative values".to_string());
            edge_cases.push("Integer overflow / underflow".to_string());
            edge_cases.push("Floating point precision".to_string());
            edge_cases.push("NaN / Infinity".to_string());
        }
        if desc_lower.contains("string") || desc_lower.contains("text") || desc_lower.contains("char") {
            edge_cases.push("Empty string".to_string());
            edge_cases.push("Unicode / multi-byte characters".to_string());
            edge_cases.push("Whitespace-only string".to_string());
            edge_cases.push("Very long string".to_string());
            edge_cases.push("Special characters and escaping".to_string());
        }
        if desc_lower.contains("file") || desc_lower.contains("io") || desc_lower.contains("read") {
            edge_cases.push("File not found".to_string());
            edge_cases.push("Permission denied".to_string());
            edge_cases.push("File locked by another process".to_string());
            edge_cases.push("Partial read / truncated data".to_string());
        }
        if desc_lower.contains("network")
            || desc_lower.contains("http")
            || desc_lower.contains("api")
        {
            edge_cases.push("Connection timeout".to_string());
            edge_cases.push("Server error (5xx)".to_string());
            edge_cases.push("Rate limiting".to_string());
            edge_cases.push("Malformed response".to_string());
        }

        // Traverse requirements for additional edge-case signals
        for req in &task.requirements {
            let rl = req.to_lowercase();
            if rl.contains("sort") {
                edge_cases.push("Already sorted input".to_string());
            }
            if rl.contains("concurr") {
                edge_cases.push("Race condition under concurrent access".to_string());
                edge_cases.push("Deadlock with multiple locks".to_string());
            }
        }

        // Deduplicate while preserving order
        let mut seen = std::collections::HashSet::new();
        edge_cases.retain(|e| seen.insert(e.clone()));

        Ok(edge_cases)
    }

    /// Identify underlying assumptions through structural analysis
    async fn identify_assumptions(&self, task: &CodingTask) -> SACAResult<Vec<String>> {
        let desc_lower = task.description.to_lowercase();
        let mut assumptions: Vec<String> = Vec::new();

        // Universal assumptions
        assumptions.push("Input data is in expected format and encoding".to_string());
        assumptions.push("Sufficient memory and compute resources available".to_string());
        assumptions.push("Environment supports required language features and libraries".to_string());

        // Structurally derived assumptions
        if desc_lower.contains("sort") || desc_lower.contains("order") {
            assumptions.push("Elements implement a total order (comparable)".to_string());
            assumptions.push("Comparison function is consistent and transitive".to_string());
        }
        if desc_lower.contains("numeric")
            || desc_lower.contains("count")
            || desc_lower.contains("math")
        {
            assumptions.push("Numeric values fit within standard integer/float ranges".to_string());
            assumptions.push("Division by zero will not occur".to_string());
        }
        if desc_lower.contains("search") || desc_lower.contains("find") {
            assumptions.push("Search predicate is deterministic (same key → same result)".to_string());
            if !desc_lower.contains("hash") {
                assumptions.push("Data structure supports the required search operation".to_string());
            }
        }
        if desc_lower.contains("concurr")
            || desc_lower.contains("thread")
            || desc_lower.contains("parallel")
        {
            assumptions.push("Operations are safely concurrent or properly synchronized".to_string());
            assumptions.push("Shared mutable state is protected by synchronization primitives".to_string());
        }
        if desc_lower.contains("file") || desc_lower.contains("persist") {
            assumptions.push("File system has sufficient space and appropriate permissions".to_string());
            assumptions.push("File paths are valid and accessible".to_string());
        }
        if desc_lower.contains("network")
            || desc_lower.contains("api")
            || desc_lower.contains("http")
        {
            assumptions.push("Network is available with acceptable latency".to_string());
            assumptions.push("Remote service conforms to the expected API contract".to_string());
        }

        // Derive from constraints
        for constraint in &task.constraints {
            let cl = constraint.to_lowercase();
            if cl.contains("complexity") {
                assumptions.push("Input size assumptions match the complexity guarantee".to_string());
            }
            if cl.contains("memory") {
                assumptions.push("Working set fits within the specified memory budget".to_string());
            }
            if cl.contains("real") || cl.contains("time") || cl.contains("latency") {
                assumptions.push("System meets the real-time performance requirements".to_string());
            }
        }

        Ok(assumptions)
    }

    /// Assess implementation risks through systematic analysis
    async fn assess_risks(&self, task: &CodingTask) -> SACAResult<Vec<String>> {
        let desc_lower = task.description.to_lowercase();
        let mut risks: Vec<String> = Vec::new();

        // Universal risks
        risks.push("Performance degradation with large inputs exceeding expected scale".to_string());
        risks.push("Memory overflow or excessive allocation leading to OOM".to_string());
        risks.push("Incorrect error handling or missing edge case paths".to_string());
        risks.push("Regression in existing functionality when adding new code".to_string());

        // Domain-specific risks
        if desc_lower.contains("sort") || desc_lower.contains("order") {
            risks.push("Unstable sort when stability is required by downstream consumers".to_string());
            risks.push("Quadratic performance on nearly-sorted data with naive pivot selection".to_string());
        }
        if desc_lower.contains("search") || desc_lower.contains("find") {
            risks.push("False negatives due to incorrect comparison or hash collision".to_string());
            risks.push("Index out of bounds in edge cases (empty collection, single element)".to_string());
        }
        if desc_lower.contains("parse") || desc_lower.contains("tokenize") {
            risks.push("Malformed input causing infinite loop or excessive backtracking".to_string());
            risks.push("Unicode/encoding issues with multi-byte characters".to_string());
        }
        if desc_lower.contains("recursive") || desc_lower.contains("recursion") {
            risks.push("Stack overflow for deep recursion (default stack ~8MB)".to_string());
        }
        if desc_lower.contains("concurr")
            || desc_lower.contains("thread")
            || desc_lower.contains("parallel")
        {
            risks.push("Data race due to unsynchronized shared state".to_string());
            risks.push("Deadlock from incorrect lock ordering".to_string());
            risks.push("Thread starvation or excessive context switching".to_string());
        }
        if desc_lower.contains("network")
            || desc_lower.contains("http")
            || desc_lower.contains("api")
        {
            risks.push("Unhandled network timeout causing hang".to_string());
            risks.push("Leaked connections or file descriptors".to_string());
        }
        if desc_lower.contains("file") || desc_lower.contains("persist") {
            risks.push("Partial write / corrupted data on crash".to_string());
            risks.push("Race condition from concurrent file access".to_string());
        }
        if desc_lower.contains("unsafe") {
            risks.push("Undefined behavior from incorrect unsafe code".to_string());
            risks.push("Memory safety violation in unsafe blocks".to_string());
        }

        // Traverse requirements for risk signals
        for req in &task.requirements {
            let rl = req.to_lowercase();
            if rl.contains("fast") || rl.contains("perform") {
                risks.push("Over-optimization leading to unreadable or unmaintainable code".to_string());
            }
            if rl.contains("generic") || rl.contains("template") {
                risks.push("Monomorphization bloat from excessive generic usage".to_string());
            }
        }

        Ok(risks)
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
            approach.push_str("Strategy: Build interface first, then implement, write integration tests. ");
            approach.push_str("Use contract testing to validate API boundaries. ");
        } else if desc_lower.contains("algorithm") || desc_lower.contains("function") {
            approach.push_str("Strategy: Implement pure function first, add property-based tests. ");
            approach.push_str("Benchmark with representative inputs. ");
        } else if desc_lower.contains("data")
            || desc_lower.contains("pipeline")
            || desc_lower.contains("stream")
        {
            approach.push_str("Strategy: Start with data contract/format, build pipeline incrementally. ");
            approach.push_str("Validate at each stage with integration tests. ");
        } else {
            approach.push_str("Strategy: Validate inputs, implement core logic, handle errors, format outputs. ");
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
                    Include inline comments for non-obvious logic and safety invariants.".to_string(),
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
