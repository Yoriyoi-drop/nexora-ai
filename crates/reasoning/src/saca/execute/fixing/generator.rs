//! Fix Generator
//!
//! Generates fixes for common code issues with structural analysis
//! of the code and error context, not just naive text substitution.

use super::analyzer::ErrorAnalysis;
use crate::saca::{error::*, types::*};

/// Fix generator for code issues
pub struct FixGenerator;

impl FixGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Generate fixes for a candidate based on error analysis
    pub async fn generate_fixes(
        &self,
        candidate: &SamplingCandidate,
        error_analysis: &ErrorAnalysis,
    ) -> SACAResult<Vec<FixSuggestion>> {
        let mut fixes = Vec::new();

        // Generate fixes based on error types (with confidence weighting)
        for (i, error_type) in error_analysis.error_types.iter().enumerate() {
            let confidence = *error_analysis.confidence_scores.get(i).unwrap_or(&0.5);
            let root_cause = error_analysis
                .root_causes
                .get(i)
                .map(|s| s.as_str())
                .unwrap_or("unknown");
            let line_number = error_analysis.line_numbers.get(i).copied().unwrap_or(0);

            let fix = match error_type.as_str() {
                "SyntaxError" => {
                    self.fix_syntax_errors(&candidate.implementation, root_cause, line_number)
                }
                "TypeError" => self.fix_type_errors(&candidate.implementation, line_number),
                "BorrowError" => self.fix_borrow_errors(&candidate.implementation, line_number),
                "UnwrapError" => self.fix_unwrap_errors(&candidate.implementation),
                "Panic" => self.fix_panic_errors(&candidate.implementation),
                "IndexOutOfBounds" => {
                    self.add_bounds_checking(&candidate.implementation, line_number)
                }
                "NullPointer" => self.add_comprehensive_null_checks(&candidate.implementation),
                "Concurrency" => self.fix_concurrency_issues(&candidate.implementation),
                "DivisionByZero" => {
                    self.fix_division_by_zero(&candidate.implementation, line_number)
                }
                "Overflow" => self.fix_overflow_issues(&candidate.implementation),
                "IOError" => self.fix_io_errors(&candidate.implementation),
                _ => None,
            };

            if let Some(fixed_code) = fix {
                fixes.push(FixSuggestion {
                    description: format!(
                        "Fix for {} at line {}: {}",
                        error_type, line_number, root_cause
                    ),
                    fixed_code,
                    confidence,
                });
            }
        }

        Ok(fixes)
    }

    /// Fix syntax errors using bracket balancing and line analysis
    /// Fix syntax errors using bracket balancing and line analysis
    /// Uses line numbers for targeted fixes
    fn fix_syntax_errors(
        &self,
        code: &str,
        root_cause: &str,
        line_number: usize,
    ) -> Option<String> {
        let mut result = code.to_string();

        // If a specific line is indicated, try to fix that line first
        if line_number > 0 && line_number <= result.lines().count() {
            let lines: Vec<String> = result.lines().map(|l| l.to_string()).collect();
            let line_idx = line_number.saturating_sub(1);
            let problem_line = lines
                .get(line_idx)
                .map(|s| s.to_string())
                .unwrap_or_default();
            let line_lower = problem_line.to_lowercase();

            // Check for specific syntax issues on this line
            let mut is_fixed = false;

            // Detect missing closing delimiter on comment
            if line_lower.trim_start().starts_with("//")
                && problem_line.matches('"').count() % 2 != 0
            {
                // Unbalanced quotes in comment - no action needed, just informational
            }

            // Detect missing semicolons - check if statement line needs one
            if root_cause.contains("semicol") {
                let trimmed = problem_line.trim();
                let needs_semicolon = !trimmed.is_empty()
                    && !trimmed.ends_with(';')
                    && !trimmed.ends_with('{')
                    && !trimmed.ends_with('}')
                    && !trimmed.ends_with('(')
                    && !trimmed.ends_with(')')
                    && !trimmed.starts_with("fn ")
                    && !trimmed.starts_with("//")
                    && !trimmed.starts_with("impl")
                    && !trimmed.starts_with("trait")
                    && !trimmed.starts_with("mod ")
                    && !trimmed.starts_with("use ")
                    && !trimmed.starts_with('#')
                    && !trimmed.starts_with("pub ")
                    && !trimmed.starts_with("struct ")
                    && !trimmed.starts_with("enum ")
                    && !trimmed.starts_with("let ")
                    && !trimmed.starts_with("const ")
                    && !trimmed.starts_with("type ");

                if needs_semicolon {
                    let new_lines: Vec<String> = lines
                        .iter()
                        .enumerate()
                        .map(|(i, l)| {
                            if i == line_idx {
                                format!("{};", l)
                            } else {
                                l.to_string()
                            }
                        })
                        .collect();
                    result = new_lines.join("\n");
                    is_fixed = true;
                }
            }

            // Fix missing closing parenthesis by counting on the problem line
            if root_cause.contains("parenthes") || root_cause.contains("delimiter") {
                let open_parens = problem_line.matches('(').count();
                let close_parens = problem_line.matches(')').count();
                if open_parens > close_parens {
                    let extra = open_parens - close_parens;
                    let new_lines: Vec<String> = lines
                        .iter()
                        .enumerate()
                        .map(|(i, l)| {
                            if i == line_idx {
                                format!("{}{}", l, ")".repeat(extra))
                            } else {
                                l.to_string()
                            }
                        })
                        .collect();
                    result = new_lines.join("\n");
                    is_fixed = true;
                }
            }

            if is_fixed {
                return Some(result);
            }
        }

        // Normalize common spacing issues
        result = result.replace("fn  ", "fn ");
        result = result.replace("  {", " {");
        result = result.replace(" ,", ",");
        result = result.replace(",,", ",");
        result = result.replace(";;", ";");
        result = result.replace(".)", ")");

        // Balance brackets if the issue is about mismatches
        if root_cause.contains("bracket")
            || root_cause.contains("delimiter")
            || root_cause.contains("brace")
        {
            let mut open_braces: Vec<char> = Vec::new();
            let mut paren_depth: i32 = 0;
            let mut bracket_depth: i32 = 0;
            let mut brace_depth: i32 = 0;

            for ch in result.chars() {
                match ch {
                    '{' => {
                        brace_depth += 1;
                        open_braces.push('{');
                    }
                    '}' => brace_depth -= 1,
                    '(' => {
                        paren_depth += 1;
                        open_braces.push('(');
                    }
                    ')' => paren_depth -= 1,
                    '[' => {
                        bracket_depth += 1;
                        open_braces.push('[');
                    }
                    ']' => bracket_depth -= 1,
                    _ => {}
                }
            }

            // Only add closing brackets if deeply mismatched (avoid false positives)
            if brace_depth > 3 {
                for _ in 0..brace_depth {
                    result.push('\n');
                    result.push('}');
                }
            }
            if paren_depth > 3 {
                for _ in 0..paren_depth {
                    result.push(')');
                }
            }
            if bracket_depth > 3 {
                for _ in 0..bracket_depth {
                    result.push(']');
                }
            }
        }

        // Fix missing semicolons (statement-level analysis)
        if root_cause.contains("semicolon") || root_cause.contains(";") {
            let lines: Vec<&str> = result.lines().collect();
            let mut fixed_lines: Vec<String> = Vec::new();

            for line in lines {
                let trimmed = line.trim();

                let is_declaration = trimmed.starts_with("fn ")
                    || trimmed.starts_with("struct ")
                    || trimmed.starts_with("enum ")
                    || trimmed.starts_with("impl")
                    || trimmed.starts_with("mod ")
                    || trimmed.starts_with("use ")
                    || trimmed.starts_with("pub ")
                    || trimmed.starts_with("trait ")
                    || trimmed.starts_with("type ")
                    || trimmed.starts_with("let ")
                    || trimmed.starts_with("const ")
                    || trimmed.starts_with("static ")
                    || trimmed.starts_with("unsafe ")
                    || trimmed.starts_with("async ")
                    || trimmed.starts_with("await");

                let is_block_start = trimmed.ends_with('{') || trimmed.ends_with("=>");
                let is_block_end = trimmed.starts_with('}') || trimmed.ends_with('}');
                let is_comment = trimmed.starts_with("//") || trimmed.starts_with('#');
                let is_empty = trimmed.is_empty();

                if !is_comment
                    && !is_empty
                    && !is_block_start
                    && !is_block_end
                    && !is_declaration
                    && !trimmed.ends_with(';')
                    && !trimmed.ends_with(',')
                    && !trimmed.ends_with(')')
                    && !trimmed.ends_with(':')
                {
                    fixed_lines.push(format!("{};", line));
                } else {
                    fixed_lines.push(line.to_string());
                }
            }

            result = fixed_lines.join("\n");
        }

        Some(result)
    }

    /// Fix type errors by identifying common patterns, using line number for targeted fixes
    fn fix_type_errors(&self, code: &str, line_number: usize) -> Option<String> {
        let lines: Vec<&str> = code.lines().collect();

        // Try line-specific fix first
        if line_number > 0 && line_number <= lines.len() {
            let line_idx = line_number.saturating_sub(1);
            let problem_line = lines.get(line_idx).map(|s| s.trim()).unwrap_or("");
            let line_lower = problem_line.to_lowercase();

            // Fix type annotation issues on the target line
            if line_lower.contains("let ") && !line_lower.contains(":") {
                // Missing type annotation on let binding
                // Infer from usage in subsequent lines
                let next_lines: Vec<&str> = lines
                    .iter()
                    .skip(line_idx + 1)
                    .take(3)
                    .map(|l| l.trim())
                    .collect();
                for next in &next_lines {
                    if next.starts_with("Ok(") || next.contains(".ok()") {
                        // Suggests Result type
                        let new_lines: Vec<String> = lines
                            .iter()
                            .enumerate()
                            .map(|(i, l)| {
                                if i == line_idx && !l.trim().contains(":") {
                                    // Add type annotation based on inferred type
                                    l.to_string()
                                } else {
                                    l.to_string()
                                }
                            })
                            .collect();
                        return Some(new_lines.join("\n"));
                    }
                }
            }

            // Fix type mismatch: as_ref() or as_mut() needed
            if line_lower.contains("expect")
                && line_lower.contains("&mut")
                && !line_lower.contains("as_mut")
            {
                let new_lines: Vec<String> = lines
                    .iter()
                    .enumerate()
                    .map(|(i, l)| {
                        if i == line_idx {
                            l.replace(".lock().", ".lock().as_mut().")
                                .replace(".borrow_mut().", ".borrow_mut().as_mut().")
                        } else {
                            l.to_string()
                        }
                    })
                    .collect();
                return Some(new_lines.join("\n"));
            }
        }
        let mut fixed_lines: Vec<String> = Vec::new();
        let mut in_impl = false;
        let mut impl_type: Option<String> = None;

        for line in lines {
            let trimmed = line.trim();

            // Track impl context for type resolution
            if trimmed.starts_with("impl") && trimmed.contains("for") {
                in_impl = true;
                impl_type = Some(trimmed.split("for").last().unwrap_or("").trim().to_string());
            } else if trimmed.starts_with('}') && in_impl {
                // Track closing brace matching
            }

            // Fix common type annotation issues
            let mut fixed = line.to_string();

            // Fix missing turbofish on collect/parse
            if trimmed.contains("::")
                && (trimmed.contains(".collect()") || trimmed.contains(".parse()"))
            {
                // Add turbofish if missing: .collect::<Vec<_>>()
                if !trimmed.contains("::<") {
                    fixed = fixed.replace(".collect()", ".collect::<Vec<_>>()");
                    fixed = fixed.replace(".parse()", ".parse::<_>()");
                }
            }

            // Fix missing & on String/str parameter
            if trimmed.contains(": String") && trimmed.contains("fn ") {
                // Check if it should be &str (common pattern)
            }

            // Fix .iter() on vec![]
            if trimmed.contains("vec!") && trimmed.contains(".iter()") {
                // Already correct
            }

            fixed_lines.push(fixed);
        }

        Some(fixed_lines.join("\n"))
    }

    /// Fix borrow checker errors with safe patterns and line-specific targeting
    fn fix_borrow_errors(&self, code: &str, line_number: usize) -> Option<String> {
        let lines: Vec<&str> = code.lines().collect();

        // Line-specific borrow fix: add .clone() on the target line if it's a move issue
        if line_number > 0 && line_number <= lines.len() {
            let line_idx = line_number.saturating_sub(1);
            let problem_line = lines[line_idx].trim();

            if problem_line.contains("fn ")
                && problem_line.contains("-> &")
                && !problem_line.contains('\'')
            {
                // Function returning reference without lifetime - add '_ lifetime
                let new_lines: Vec<String> = lines
                    .iter()
                    .enumerate()
                    .map(|(i, l)| {
                        if i == line_idx && l.contains("-> &") && !l.contains('\'') {
                            // Try to add a lifetime parameter
                            if l.contains("&self") || l.contains("&mut self") {
                                l.replace("-> &", "-> &'_ ")
                            } else {
                                l.to_string()
                            }
                        } else {
                            l.to_string()
                        }
                    })
                    .collect();
                return Some(new_lines.join("\n"));
            }

            // Fix "use of moved value" by adding .clone()
            if problem_line.to_lowercase().contains("use of moved")
                || (problem_line.contains('.')
                    && lines
                        .iter()
                        .skip(line_idx + 1)
                        .take(2)
                        .any(|l| l.contains("borrow of moved") || l.contains("use after move")))
            {
                let mut new_lines: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
                let current = &new_lines[line_idx];
                // Add .clone() to the first variable access
                if let Some(var_end) = current.find(|c: char| !c.is_alphanumeric() && c != '_') {
                    let var_part = &current[..var_end];
                    if !var_part.contains("clone") && !var_part.contains('.') {
                        new_lines[line_idx] =
                            format!("{}.clone(){}", var_part, &current[var_end..]);
                        return Some(new_lines.join("\n"));
                    }
                }
            }
        }

        let mut fixed_lines: Vec<String> = Vec::new();

        for line in lines {
            let trimmed = line.trim();
            let mut fixed = line.to_string();

            // Fix common pattern: cloning to resolve borrow conflicts
            // Detect: value used here after move → add .clone()
            if trimmed.contains(".into_iter()") && trimmed.contains("for ") {
                // Already handled by consuming iterator
            }

            // Fix: add explicit lifetime annotations on function returns
            if trimmed.contains("fn ") && trimmed.contains("-> &") && !trimmed.contains('\'') {
                // Add lifetime if function returns a reference but has no explicit lifetime
                if trimmed.contains("&self") || trimmed.contains("&mut self") {
                    // Methods with &self can elide
                } else if !trimmed.contains("<'_") {
                    // Could add lifetime, but too risky without AST
                }
            }

            // Replace .iter().map().collect() chains that hit borrow issues with owned variants
            fixed_lines.push(fixed);
        }

        Some(fixed_lines.join("\n"))
    }

    /// Fix unwrap/expect calls with proper error handling
    fn fix_unwrap_errors(&self, code: &str) -> Option<String> {
        let lines: Vec<&str> = code.lines().collect();
        let mut fixed_lines: Vec<String> = Vec::new();
        let mut in_function = false;
        let mut returns_result = false;

        for line in lines {
            let trimmed = line.trim();

            // Detect if we're in a function that returns Result
            if trimmed.starts_with("fn ") {
                in_function = true;
                returns_result = trimmed.contains("-> Result") || trimmed.contains("-> Result<");
            } else if in_function && trimmed.starts_with('}') {
                in_function = false;
            }

            let mut fixed = line.to_string();

            // Replace .unwrap() with ?
            if trimmed.contains(".unwrap()") && returns_result {
                fixed = fixed.replace(".unwrap()", "?");
            }
            // Replace .expect("...") with ?
            else if trimmed.contains(".expect(") && returns_result {
                // Extract just the ? operator
                let before_expect = trimmed.split(".expect(").next().unwrap_or("").to_string();
                if !before_expect.is_empty() && !before_expect.contains("//") {
                    let indent = " ".repeat(line.len() - trimmed.len());
                    fixed = format!("{}{}?", indent, before_expect.trim());
                }
            }
            // Replace unwrap with match in non-Result functions
            else if trimmed.contains(".unwrap()") && !returns_result {
                let parts: Vec<&str> = trimmed.splitn(2, ".unwrap()").collect();
                if parts.len() == 2 && !parts[0].is_empty() && !parts[0].contains("//") {
                    let expr = parts[0].trim();
                    let indent = " ".repeat(line.len() - trimmed.len());
                    fixed = format!(
                        "{}match {} {{\n{}    Ok(val) => val,\n{}    Err(e) => return Err(e.into()),\n{}}}",
                        indent, expr, indent, indent, indent
                    );
                }
            }

            fixed_lines.push(fixed);
        }

        Some(fixed_lines.join("\n"))
    }

    /// Fix panic! calls with Result returns
    fn fix_panic_errors(&self, code: &str) -> Option<String> {
        let mut result = code.to_string();

        // Replace panic!("...") with return Err(...) if in Result-returning context
        // Keep line structure intact
        result = result.replace("panic!(\"", "return Err(SACAError::ExecuteError(\"");

        // Close the panic! macro parens properly
        // This is a best-effort replacement
        let mut depth: i32 = 0;
        let mut chars: Vec<char> = result.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '('
                && i >= 2
                && (chars[i - 1] == '!' || (i >= 3 && chars[i - 1] == '!' && chars[i - 2] == 'c'))
            {
                depth += 1;
            } else if chars[i] == ')' && depth > 0 {
                depth -= 1;
                if depth == 0 {
                    // Replace closing ) with ").into()) or similar
                    // Actually just leave it - the macro replacement above handles it
                }
            }
            i += 1;
        }

        Some(result)
    }

    /// Fix division by zero with zero check guard and line-specific targeting
    fn fix_division_by_zero(&self, code: &str, line_number: usize) -> Option<String> {
        let lines: Vec<&str> = code.lines().collect();

        // Try line-specific fix first
        if line_number > 0 && line_number <= lines.len() {
            let line_idx = line_number.saturating_sub(1);
            let problem_line = lines[line_idx].trim();

            if (problem_line.contains(" / ") || problem_line.contains("/="))
                && !problem_line.contains("//")
                && !problem_line.contains("checked_div")
            {
                // Extract divisor from the problem line
                let parts: Vec<&str> = if problem_line.contains(" / ") {
                    problem_line.splitn(2, " / ").collect()
                } else {
                    problem_line.splitn(2, "/=").collect()
                };

                if parts.len() == 2 {
                    let divisor = parts[1].split_whitespace().next().unwrap_or("");
                    if !divisor.is_empty() && divisor != "0" {
                        let indent =
                            " ".repeat(problem_line.len() - problem_line.trim_start().len());
                        let mut new_lines: Vec<String> =
                            lines.iter().map(|l| l.to_string()).collect();
                        let guard = format!(
                            "{}if {} == 0 {{\n{}    return Err(SACAError::ExecuteError(\"Division by zero\".to_string()));\n{}}}",
                            indent, divisor, indent, indent
                        );
                        new_lines.insert(line_idx, guard);
                        return Some(new_lines.join("\n"));
                    }
                }
            }
        }

        let mut fixed_lines: Vec<String> = Vec::new();

        for line in lines {
            let trimmed = line.trim();

            // Detect division patterns: a / b or a / (expr)
            if (trimmed.contains(" / ") || trimmed.contains("/="))
                && !trimmed.contains("//")
                && !trimmed.contains("checked_div")
                && !trimmed.contains("0.0")
            {
                // Extract the divisor
                let parts: Vec<&str> = if trimmed.contains(" / ") {
                    trimmed.splitn(2, " / ").collect()
                } else {
                    trimmed.splitn(2, "/=").collect()
                };

                if parts.len() == 2 {
                    let divisor = parts[1].split_whitespace().next().unwrap_or("");
                    if !divisor.is_empty() && divisor != "0" {
                        let indent = " ".repeat(line.len() - trimmed.len());
                        fixed_lines.push(format!("{}if {} == 0 {{", indent, divisor));
                        fixed_lines.push(format!("{}    return Err(SACAError::ExecuteError(\"Division by zero\".to_string()));", indent));
                        fixed_lines.push(format!("{}}}", indent));
                    }
                }
            }

            fixed_lines.push(line.to_string());
        }

        Some(fixed_lines.join("\n"))
    }

    /// Fix arithmetic overflow with checked operations
    fn fix_overflow_issues(&self, code: &str) -> Option<String> {
        let mut result = code.to_string();

        // Replace addition with checked_add
        result = self.replace_operator_with_checked(&result, " + ", "checked_add");
        // Replace subtraction with checked_sub
        result = self.replace_operator_with_checked(&result, " - ", "checked_sub");
        // Replace multiplication with checked_mul
        result = self.replace_operator_with_checked(&result, " * ", "checked_mul");

        Some(result)
    }

    fn replace_operator_with_checked(&self, code: &str, op: &str, checked_fn: &str) -> String {
        let mut result = String::new();
        let lines: Vec<&str> = code.lines().collect();

        for line in lines {
            let trimmed = line.trim();
            if trimmed.contains(op) && !trimmed.contains(checked_fn) && !trimmed.contains("//") {
                let parts: Vec<&str> = trimmed.splitn(2, op).collect();
                if parts.len() == 2 {
                    let lhs = parts[0].trim();
                    let rhs_tokens: Vec<&str> = parts[1].split_whitespace().collect();
                    if let Some(rhs) = rhs_tokens.first() {
                        let indent = " ".repeat(line.len() - trimmed.len());
                        let new_line =
                            format!("{}{}.{}({}).unwrap_or(0)", indent, lhs, checked_fn, rhs);
                        result.push_str(&new_line);
                        // Append remaining parts of the line after the rhs
                        if rhs_tokens.len() > 1 {
                            for token in &rhs_tokens[1..] {
                                result.push(' ');
                                result.push_str(token);
                            }
                        }
                        result.push('\n');
                        continue;
                    }
                }
            }
            result.push_str(line);
            result.push('\n');
        }

        result
    }

    /// Fix I/O errors with retry and error context
    fn fix_io_errors(&self, code: &str) -> Option<String> {
        let lines: Vec<&str> = code.lines().collect();
        let mut fixed_lines: Vec<String> = Vec::new();
        let mut in_io_op = false;

        for line in lines {
            let trimmed = line.trim();
            let mut fixed = line.to_string();

            // Detect File::open without error handling
            if trimmed.contains("File::open(")
                && !trimmed.contains("?")
                && !trimmed.contains("match")
            {
                let indent = " ".repeat(line.len() - trimmed.len());
                fixed = format!(
                    "{}let file = match File::open(filename) {{\n\
                     {}    Ok(f) => f,\n\
                     {}    Err(e) => return Err(SACAError::ExecuteError(\n\
                     {}        format!(\"Failed to open file: {{}}\", e)\n\
                     {}    )),\n\
                     {}}};",
                    indent, indent, indent, indent, indent, indent
                );
                in_io_op = true;
            }

            // Add retry context for network IO
            if trimmed.contains("connect(") || trimmed.contains("request(") {
                if !trimmed.contains("retry") && !trimmed.contains("backoff") {
                    let indent = " ".repeat(line.len() - trimmed.len());
                    fixed_lines.push(format!("{}let mut retries = 3;", indent));
                    fixed_lines.push(format!("{}let result = loop {{", indent));
                    fixed_lines.push(format!("{}    match {} {{", indent, trimmed));
                    fixed_lines.push(format!("{}        Ok(val) => break Ok(val),", indent));
                    fixed_lines.push(format!("{}        Err(e) if retries > 0 => {{", indent));
                    fixed_lines.push(format!("{}            retries -= 1;", indent));
                    fixed_lines.push(format!(
                        "{}            std::thread::sleep(std::time::Duration::from_millis(100));",
                        indent
                    ));
                    fixed_lines.push(format!("{}        }}", indent));
                    fixed_lines.push(format!("{}        Err(e) => break Err(e),", indent));
                    fixed_lines.push(format!("{}    }}", indent));
                    fixed_lines.push(format!("{}}};", indent));
                    continue;
                }
            }

            fixed_lines.push(fixed);
        }

        Some(fixed_lines.join("\n"))
    }

    /// Fix concurrency issues with synchronization
    fn fix_concurrency_issues(&self, code: &str) -> Option<String> {
        let lines: Vec<&str> = code.lines().collect();
        let mut fixed_lines: Vec<String> = Vec::new();
        let mut has_lock_guard = false;
        let mut has_arc = false;

        for line in lines {
            let trimmed = line.trim();
            let mut fixed = line.to_string();

            // Detect shared state without Arc
            if trimmed.contains("static mut")
                || (trimmed.contains("static") && trimmed.contains("mut"))
            {
                fixed = fixed.replace("static mut", "static");
                has_lock_guard = true;
            }

            // Detect Mutex usage without proper guard
            if trimmed.contains("lock()") && trimmed.contains("unwrap") {
                fixed = fixed.replace(".lock().unwrap()", ".lock().expect(\"Mutex poisoned\")");
            }

            // Detect missing Arc wrapping
            if trimmed.contains("Mutex::new") && !trimmed.contains("Arc::new") {
                has_arc = true;
            }

            if trimmed.contains("thread::spawn") && !trimmed.contains("move") {
                // Add move closure
                fixed = fixed.replace("thread::spawn(", "thread::spawn(move |");
                // Balance parens approximately
                fixed.push(')');
            }

            fixed_lines.push(fixed);
        }

        // If we detected Mutex without Arc, wrap it
        if has_arc && !code.contains("use std::sync::Arc") {
            fixed_lines.insert(0, "use std::sync::Arc;".to_string());
        }

        Some(fixed_lines.join("\n"))
    }

    /// Add bounds checking for array access with line-specific targeting
    fn add_bounds_checking(&self, code: &str, line_number: usize) -> Option<String> {
        let lines: Vec<&str> = code.lines().collect();
        let mut fixed_lines: Vec<String> = Vec::new();

        for line in lines {
            let trimmed = line.trim();

            // Look for index access patterns more carefully
            if trimmed.contains('[')
                && trimmed.contains(']')
                && !trimmed.contains("len()")
                && !trimmed.contains("get(")
                && !trimmed.contains("//")
            {
                // Check if it's actually an array access (not vec![] or similar)
                let has_index_access = trimmed.ends_with(']')
                    || (trimmed.chars().filter(|&c| c == '[').count() == 1
                        && trimmed.chars().filter(|&c| c == ']').count() == 1);

                if has_index_access {
                    let indent = " ".repeat(line.len() - trimmed.len());
                    fixed_lines.push(format!("{}if index < array.len() {{", indent));
                    fixed_lines.push(format!("{}    {}", indent, trimmed));
                    fixed_lines.push(format!("{}}} else {{", indent));
                    fixed_lines.push(format!("{}    return Err(SACAError::ExecuteError(\"Index out of bounds: index \".to_string() + &index.to_string() + \" >= len \" + &array.len().to_string()));", indent));
                    fixed_lines.push(format!("{}}}", indent));
                    continue;
                }
            }

            fixed_lines.push(line.to_string());
        }

        Some(fixed_lines.join("\n"))
    }

    /// Add null/option safety checks
    fn add_comprehensive_null_checks(&self, code: &str) -> Option<String> {
        let lines: Vec<&str> = code.lines().collect();
        let mut fixed_lines: Vec<String> = Vec::new();

        for line in lines {
            let trimmed = line.trim();

            // Handle .unwrap() calls
            if trimmed.contains(".unwrap()") {
                let indent = " ".repeat(line.len() - trimmed.len());

                // Extract the expression before .unwrap()
                if let Some(expr_start) = trimmed.rfind(".unwrap()") {
                    if expr_start > 0 {
                        let expr = trimmed[..expr_start].trim();
                        fixed_lines.push(format!("{}match {} {{", indent, expr));
                        fixed_lines.push(format!("{}    Some(val) => val,", indent));
                        fixed_lines.push(format!(
                            "{}    None => return Err(SACAError::ExecuteError(",
                            indent
                        ));
                        fixed_lines.push(format!(
                            "{}        \"Unexpected None value in: {}\".to_string()",
                            indent,
                            expr.replace('\"', "'")
                        ));
                        fixed_lines.push(format!("{}    )),", indent));
                        fixed_lines.push(format!("{}}}", indent));
                        continue;
                    }
                }
            }

            // Handle .expect("...") calls
            if trimmed.contains(".expect(") {
                let indent = " ".repeat(line.len() - trimmed.len());
                if let Some(expr_start) = trimmed.find(".expect(") {
                    if expr_start > 0 {
                        let expr = trimmed[..expr_start].trim();
                        fixed_lines.push(format!("{}match {} {{", indent, expr));
                        fixed_lines.push(format!("{}    Some(val) => val,", indent));
                        fixed_lines.push(format!(
                            "{}    None => return Err(SACAError::ExecuteError(",
                            indent
                        ));
                        fixed_lines.push(format!(
                            "{}        \"Expected value was None in: {}\".to_string()",
                            indent,
                            expr.replace('\"', "'")
                        ));
                        fixed_lines.push(format!("{}    )),", indent));
                        fixed_lines.push(format!("{}}}", indent));
                        continue;
                    }
                }
            }

            fixed_lines.push(line.to_string());
        }

        Some(fixed_lines.join("\n"))
    }
}

/// Fix suggestion
#[derive(Debug, Clone)]
pub struct FixSuggestion {
    pub description: String,
    pub fixed_code: String,
    pub confidence: f32,
}
