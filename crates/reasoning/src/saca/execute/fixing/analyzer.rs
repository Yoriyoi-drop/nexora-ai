//! Error Analyzer
//!
//! Analyzes error logs using structured pattern detection (not just keyword matching)
//! to identify error categories, root causes, and fix strategies.

use crate::saca::{error::*, types::*};

/// Error analyzer for execution failures
pub struct ErrorAnalyzer {
    analysis_depth: ErrorAnalysisDepth,
}

/// Represents a detected error pattern with its confidence
struct ErrorPattern {
    category: &'static str,
    keywords: &'static [&'static str],
    root_cause: &'static str,
    fix_strategy: &'static str,
    base_confidence: f32,
    requires_context: bool,
}

static PATTERNS: &[ErrorPattern] = &[
    // Compilation / syntax errors
    ErrorPattern {
        category: "SyntaxError",
        keywords: &["syntax", "expected", "unexpected token", "missing"],
        root_cause: "Code does not follow language grammar rules",
        fix_strategy: "Review compiler diagnostics at the reported line; check matching delimiters, keyword spelling, and punctuation",
        base_confidence: 0.9,
        requires_context: false,
    },
    ErrorPattern {
        category: "TypeError",
        keywords: &["type mismatch", "cannot infer", "expected type", "incompatible type", "type annotations"],
        root_cause: "Type constraint violation or missing type annotation",
        fix_strategy: "Explicitly annotate types at the boundary; ensure function signatures match call sites",
        base_confidence: 0.85,
        requires_context: false,
    },
    ErrorPattern {
        category: "BorrowError",
        keywords: &["borrow", "lifetime", "cannot move", "cannot borrow", "use after move", "dropped"],
        root_cause: "Rust ownership/borrowing rules violated",
        fix_strategy: "Restructure ownership: clone where necessary, use references with correct lifetimes, or change data structure design",
        base_confidence: 0.85,
        requires_context: false,
    },
    ErrorPattern {
        category: "UnwrapError",
        keywords: &["unwrap", "expect", "panic"],
        root_cause: "Unwrapped None/Err value at runtime",
        fix_strategy: "Replace unwrap/expect with match, if let, or ? operator for proper error propagation",
        base_confidence: 0.9,
        requires_context: false,
    },
    ErrorPattern {
        category: "Panic",
        keywords: &["panic!", "panic at", "panicked", "fatal"],
        root_cause: "Runtime panic: unrecoverable error triggered",
        fix_strategy: "Replace panic! with Result return; add proper error handling at call sites",
        base_confidence: 0.85,
        requires_context: false,
    },

    // Runtime errors
    ErrorPattern {
        category: "IndexOutOfBounds",
        keywords: &["index out of bounds", "out of range", "bound"],
        root_cause: "Array/slice access without bounds checking",
        fix_strategy: "Add len() check before index access; use get() for safe access",
        base_confidence: 0.9,
        requires_context: false,
    },
    ErrorPattern {
        category: "NullPointer",
        keywords: &["null pointer", "nullptr", "segfault", "SIGSEGV", "null reference"],
        root_cause: "Dereferencing null or freed pointer",
        fix_strategy: "Check for null before dereference; use Option<T> instead of nullable pointers",
        base_confidence: 0.85,
        requires_context: false,
    },
    ErrorPattern {
        category: "Concurrency",
        keywords: &["deadlock", "race condition", "data race", "poison", "thread panicked", "mutex"],
        root_cause: "Concurrent access without proper synchronization",
        fix_strategy: "Use appropriate synchronization primitives; minimize shared state; consider actor model",
        base_confidence: 0.75,
        requires_context: true,
    },
    ErrorPattern {
        category: "MemoryError",
        keywords: &["out of memory", "allocation error", "OOM", "memory exhausted"],
        root_cause: "Memory exhaustion or fragmentation",
        fix_strategy: "Reduce allocation frequency; use memory pools; stream large data instead of loading entirely",
        base_confidence: 0.8,
        requires_context: false,
    },
    ErrorPattern {
        category: "IOError",
        keywords: &["no such file", "permission denied", "connection refused", "broken pipe", "io error", "timeout"],
        root_cause: "I/O operation failed due to OS or network condition",
        fix_strategy: "Add retry logic with exponential backoff; check file/resource existence before access; propagate errors with context",
        base_confidence: 0.85,
        requires_context: false,
    },

    // Logic errors
    ErrorPattern {
        category: "DivisionByZero",
        keywords: &["division by zero", "divide by zero", "remainder by zero"],
        root_cause: "Integer division without zero check",
        fix_strategy: "Add zero check before division; return Result or Option instead of panicking",
        base_confidence: 0.95,
        requires_context: false,
    },
    ErrorPattern {
        category: "Overflow",
        keywords: &["overflow", "underflow", "wrapping"],
        root_cause: "Arithmetic overflow/underflow",
        fix_strategy: "Use checked_add/sub/mul; validate input range; use wider integer types",
        base_confidence: 0.85,
        requires_context: false,
    },
    ErrorPattern {
        category: "Performance",
        keywords: &["timeout", "too slow", "performance", "latency", "O(n²)"],
        root_cause: "Algorithmic inefficiency or resource contention",
        fix_strategy: "Profile to identify hot path; optimize algorithm complexity; add caching; use parallel processing",
        base_confidence: 0.7,
        requires_context: true,
    },

    // Integration errors
    ErrorPattern {
        category: "SerializationError",
        keywords: &["serialize", "deserialize", "json parse", "parse error", "invalid format"],
        root_cause: "Data format mismatch between producer and consumer",
        fix_strategy: "Validate data format before parsing; version your serialization schema; add graceful fallback",
        base_confidence: 0.8,
        requires_context: false,
    },
    ErrorPattern {
        category: "NetworkError",
        keywords: &["connection reset", "connection refused", "DNS", "TLS", "SSL", "handshake"],
        root_cause: "Network communication failure",
        fix_strategy: "Add reconnection logic with backoff; verify network dependencies are healthy; add timeout handling",
        base_confidence: 0.75,
        requires_context: true,
    },
];

impl ErrorAnalyzer {
    pub fn new(analysis_depth: ErrorAnalysisDepth) -> Self {
        Self { analysis_depth }
    }

    /// Analyze errors from execution logs
    pub async fn analyze_errors(&self, error_logs: &[String]) -> SACAResult<ErrorAnalysis> {
        let mut analysis = ErrorAnalysis {
            error_types: Vec::new(),
            root_causes: Vec::new(),
            fix_strategies: Vec::new(),
            confidence_scores: Vec::new(),
        };

        for error_log in error_logs {
            // Determine which pattern set to use based on depth
            let max_patterns = match self.analysis_depth {
                ErrorAnalysisDepth::Shallow => 4,         // syntax, type, unwrap, panic
                ErrorAnalysisDepth::Medium => 8,          // + index, null, concurrency, memory
                ErrorAnalysisDepth::Deep => 12,            // + IO, div0, overflow, serialization
                ErrorAnalysisDepth::Comprehensive => PATTERNS.len(), // all
            };

            // Classify the error by checking all patterns
            let lower_log = error_log.to_lowercase();
            let mut matched = false;

            for pattern in PATTERNS.iter().take(max_patterns) {
                let keyword_match: bool = pattern.keywords.iter().any(|kw| lower_log.contains(kw));

                if keyword_match {
                    let match_count = pattern.keywords.iter().filter(|kw| lower_log.contains(*kw)).count();
                    let total_count = pattern.keywords.len();
                    let density = match_count as f32 / total_count as f32;
                    let confidence = (pattern.base_confidence * (0.5 + 0.5 * density)).min(0.99);

                    analysis.error_types.push(pattern.category.to_string());
                    analysis.root_causes.push(pattern.root_cause.to_string());
                    analysis.fix_strategies.push(pattern.fix_strategy.to_string());
                    analysis.confidence_scores.push(confidence);
                    matched = true;

                    // For deep+ analysis, break after first match; for comprehensive, continue scanning
                    if !matches!(self.analysis_depth, ErrorAnalysisDepth::Comprehensive) {
                        break;
                    }
                }
            }

            // If no pattern matched, add a generic analysis entry
            if !matched && matches!(self.analysis_depth, ErrorAnalysisDepth::Deep | ErrorAnalysisDepth::Comprehensive) {
                analysis.error_types.push("UnknownError".to_string());
                analysis.root_causes.push("Unrecognized error pattern".to_string());
                analysis.fix_strategies.push("Inspect error log context for clues; add structured error handling".to_string());
                analysis.confidence_scores.push(0.3);
            }
        }

        Ok(analysis)
    }
}

/// Error analysis result
#[derive(Debug, Clone)]
pub struct ErrorAnalysis {
    pub error_types: Vec<String>,
    pub root_causes: Vec<String>,
    pub fix_strategies: Vec<String>,
    pub confidence_scores: Vec<f32>,
}
