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

/// Parsed error message with extracted metadata
#[derive(Debug, Clone)]
struct ParsedError {
    full_message: String,
    error_type: String,
    line_number: usize,
    column: usize,
    file_path: String,
    extracted_context: String,
    error_category: ErrorCategory,
}

#[derive(Debug, Clone, PartialEq)]
enum ErrorCategory {
    Compilation,
    Runtime,
    Logic,
    Resource,
    Concurrency,
    Unknown,
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

    /// Parse an error message to extract structured information
    fn parse_error_message(&self, error_log: &str) -> ParsedError {
        let lower = error_log.to_lowercase();

        // Extract line number using Rust compiler error format: file.rs:LINE:COL
        let mut line_number = 0usize;
        let mut column = 0usize;
        let mut file_path = String::new();

        // Try to match Rust compiler error format: " --> file.rs:LINE:COL" or "file.rs:LINE:COL"
        for part in error_log.split_whitespace() {
            if let Some(colon_pos) = part.rfind(':') {
                if let Some(second_colon) = part[..colon_pos].rfind(':') {
                    let line_str = &part[second_colon + 1..colon_pos];
                    let col_str = &part[colon_pos + 1..];
                    if let (Ok(ln), Ok(col)) = (line_str.parse::<usize>(), col_str.parse::<usize>())
                    {
                        file_path = part[..second_colon].to_string();
                        line_number = ln;
                        column = col;
                        break;
                    }
                }
            }
        }

        // If no Rust-style format found, try "line X" or "at line X"
        if line_number == 0 {
            for word in error_log.split_whitespace() {
                if word == "line" || word == "Line" || word == "LINE" {
                    // Next word might be the number
                    continue;
                }
            }
            // Also try: "at line N"
            if let Some(line_idx) = lower.find("line ") {
                let after = &lower[line_idx + 5..];
                let num_str: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
                if let Ok(ln) = num_str.parse::<usize>() {
                    line_number = ln;
                }
            }
        }

        // Extract context: one line before and after the error description
        let extracted_context = error_log
            .lines()
            .filter(|l| {
                let tl = l.trim().to_lowercase();
                !tl.is_empty()
                    && !tl.starts_with("warning")
                    && !tl.starts_with("note:")
                    && !tl.starts_with("help:")
                    && !tl.starts_with("   ")
            })
            .take(3)
            .collect::<Vec<_>>()
            .join("; ");

        // Determine error category based on content
        let error_category = if lower.contains("error[E")
            || lower.contains("compile")
            || lower.contains("syntax")
        {
            ErrorCategory::Compilation
        } else if lower.contains("panic") || lower.contains("segfault") || lower.contains("signal")
        {
            ErrorCategory::Runtime
        } else if lower.contains("logic")
            || lower.contains("assert")
            || lower.contains("invariant")
            || lower.contains("expectation")
        {
            ErrorCategory::Logic
        } else if lower.contains("oom")
            || lower.contains("out of memory")
            || lower.contains("disk")
            || lower.contains("no such file")
        {
            ErrorCategory::Resource
        } else if lower.contains("deadlock")
            || lower.contains("race")
            || lower.contains("mutex")
            || lower.contains("poison")
        {
            ErrorCategory::Concurrency
        } else {
            ErrorCategory::Unknown
        };

        ParsedError {
            full_message: error_log.to_string(),
            error_type: String::new(),
            line_number,
            column,
            file_path,
            extracted_context,
            error_category,
        }
    }

    /// Analyze errors from execution logs with proper parsing and categorization
    pub async fn analyze_errors(&self, error_logs: &[String]) -> SACAResult<ErrorAnalysis> {
        let mut analysis = ErrorAnalysis {
            error_types: Vec::new(),
            root_causes: Vec::new(),
            fix_strategies: Vec::new(),
            confidence_scores: Vec::new(),
            line_numbers: Vec::new(),
            error_categories: Vec::new(),
        };

        for error_log in error_logs {
            // Parse the error message to extract structured info
            let parsed = self.parse_error_message(error_log);

            // Determine which pattern set to use based on depth
            let max_patterns = match self.analysis_depth {
                ErrorAnalysisDepth::Shallow => 4, // syntax, type, unwrap, panic
                ErrorAnalysisDepth::Medium => 8,  // + index, null, concurrency, memory
                ErrorAnalysisDepth::Deep => 12,   // + IO, div0, overflow, serialization
                ErrorAnalysisDepth::Comprehensive => PATTERNS.len(), // all
            };

            // Classify the error by checking all patterns with confidence scoring
            let lower_log = error_log.to_lowercase();
            let mut matched = false;

            for pattern in PATTERNS.iter().take(max_patterns) {
                let keyword_match: bool = pattern.keywords.iter().any(|kw| lower_log.contains(kw));

                if keyword_match {
                    let match_count = pattern
                        .keywords
                        .iter()
                        .filter(|kw| lower_log.contains(*kw))
                        .count();
                    let total_count = pattern.keywords.len();
                    let density = match_count as f32 / total_count as f32;
                    let confidence = (pattern.base_confidence * (0.5 + 0.5 * density)).min(0.99);

                    analysis.error_types.push(pattern.category.to_string());
                    analysis.root_causes.push(pattern.root_cause.to_string());

                    // Append context information for better fix strategies
                    let mut fix_strategy = pattern.fix_strategy.to_string();
                    if !parsed.extracted_context.is_empty() {
                        fix_strategy.push_str(&format!(
                            " | Context near error: {}",
                            parsed.extracted_context
                        ));
                    }
                    analysis.fix_strategies.push(fix_strategy);
                    analysis.confidence_scores.push(confidence);
                    analysis.line_numbers.push(parsed.line_number);
                    analysis
                        .error_categories
                        .push(format!("{:?}", parsed.error_category));
                    matched = true;

                    // For deep+ analysis, break after first match; for comprehensive, continue scanning
                    if !matches!(self.analysis_depth, ErrorAnalysisDepth::Comprehensive) {
                        break;
                    }
                }
            }

            // If no pattern matched, add a generic analysis entry with parsed info
            if !matched
                && matches!(
                    self.analysis_depth,
                    ErrorAnalysisDepth::Deep | ErrorAnalysisDepth::Comprehensive
                )
            {
                let category_str = format!("{:?}", parsed.error_category);
                analysis
                    .error_types
                    .push(format!("{}({})", category_str, "UnknownError"));
                analysis.root_causes.push(format!(
                    "Unrecognized error pattern at line {}: {}",
                    parsed.line_number, parsed.extracted_context
                ));
                analysis.fix_strategies.push(
                    "Inspect error log context for clues; add structured error handling"
                        .to_string(),
                );
                analysis.confidence_scores.push(0.3);
                analysis.line_numbers.push(parsed.line_number);
                analysis.error_categories.push(category_str);
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
    pub line_numbers: Vec<usize>,
    pub error_categories: Vec<String>,
}
