use std::collections::HashMap;

use syn::{
    visit::Visit, Expr, ExprCall, Item,
};

use crate::linters::{CodeIssue, IssueSeverity};

/// AST-based vulnerability findings
#[derive(Debug, Default)]
pub struct AstFindings {
    pub issues: Vec<CodeIssue>,
    pub metrics: HashMap<String, f32>,
}

/// Walk the AST and collect findings
struct RustSecurityVisitor<'a> {
    findings: &'a mut Vec<CodeIssue>,
}

/// Check if an item is inside #[cfg(test)] or #[test] context
fn is_test_item(item: &Item) -> bool {
    let attrs = match item {
        Item::Const(v) => &v.attrs,
        Item::Enum(v) => &v.attrs,
        Item::ExternCrate(v) => &v.attrs,
        Item::Fn(v) => &v.attrs,
        Item::ForeignMod(v) => &v.attrs,
        Item::Impl(v) => &v.attrs,
        Item::Macro(v) => &v.attrs,
        Item::Mod(v) => &v.attrs,
        Item::Static(v) => &v.attrs,
        Item::Struct(v) => &v.attrs,
        Item::Trait(v) => &v.attrs,
        Item::TraitAlias(v) => &v.attrs,
        Item::Type(v) => &v.attrs,
        Item::Union(v) => &v.attrs,
        Item::Use(v) => &v.attrs,
        _ => return false,
    };
    attrs.iter().any(|a| {
        let path_str = a.path().segments.iter()
            .map(|s| s.ident.to_string())
            .collect::<Vec<_>>()
            .join("::");
        path_str == "cfg" || path_str == "test"
    })
}

/// Extract line ranges of #[cfg(test)] and #[test] items
fn extract_test_ranges(file: &syn::File) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    for item in &file.items {
        if is_test_item(item) {
            let start = syn::spanned::Spanned::span(item).start().line;
            let end = syn::spanned::Spanned::span(item).end().line;
            ranges.push((start, end));
        }
    }
    ranges
}

impl<'ast> Visit<'ast> for RustSecurityVisitor<'ast> {
    fn visit_expr_unsafe(&mut self, node: &'ast syn::ExprUnsafe) {
        let ln = syn::spanned::Spanned::span(&node.unsafe_token).start().line;
        let block_text = quote::quote!(#node).to_string();

        let has_ptr_deref = block_text.contains("*const") || block_text.contains("*mut")
            || block_text.contains("as *") || block_text.contains("ptr::");
        let has_ffi = block_text.contains("extern") || block_text.contains("FFI")
            || block_text.contains("libc::") || block_text.contains("winapi::");
        let has_uninit = block_text.contains("uninitialized") || block_text.contains("zeroed");
        let has_union = block_text.contains("union") || block_text.contains(".__");

        if has_ptr_deref || has_uninit || has_union {
            self.findings.push(CodeIssue {
                severity: IssueSeverity::Error,
                category: "Security".to_string(),
                message: format!(
                    "High-risk unsafe block: {}",
                    if has_ptr_deref { "raw pointer dereference" }
                    else if has_uninit { "uninitialized memory" }
                    else { "union field access" }
                ),
                line_number: Some(ln),
                column_number: None,
                rule_id: "AST-UNSAFE-HIGH".to_string(),
            });
        } else if has_ffi {
            self.findings.push(CodeIssue {
                severity: IssueSeverity::Warning,
                category: "Security".to_string(),
                message: "Unsafe FFI call detected — validate C-side correctness".to_string(),
                line_number: Some(ln),
                column_number: None,
                rule_id: "AST-UNSAFE-FFI".to_string(),
            });
        }

        syn::visit::visit_expr_unsafe(self, node);
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        let ln = syn::spanned::Spanned::span(&node.func).start().line;
        if let Expr::Path(ref path_expr) = *node.func {
            let path_str = path_to_string(&path_expr.path);

            if path_str == "Command::new" || path_str == "std::process::Command::new" {
                self.findings.push(CodeIssue {
                    severity: IssueSeverity::Critical,
                    category: "Security".to_string(),
                    message: "Process::Command::new() — potential command injection if input is unsanitized".to_string(),
                    line_number: Some(ln),
                    column_number: None,
                    rule_id: "AST-CMD-INJECTION".to_string(),
                });
            }

            if path_str == "transmute" || path_str == "std::mem::transmute" {
                self.findings.push(CodeIssue {
                    severity: IssueSeverity::Error,
                    category: "Security".to_string(),
                    message: "std::mem::transmute — type confusion risk. Use safe conversions when possible".to_string(),
                    line_number: Some(ln),
                    column_number: None,
                    rule_id: "AST-TRANSMUTE".to_string(),
                });
            }

            if path_str == "process::exit" || path_str == "std::process::exit" {
                self.findings.push(CodeIssue {
                    severity: IssueSeverity::Warning,
                    category: "Security".to_string(),
                    message: "std::process::exit() — aborts all cleanup. Prefer graceful shutdown".to_string(),
                    line_number: Some(ln),
                    column_number: None,
                    rule_id: "AST-PROCESS-EXIT".to_string(),
                });
            }

            if path_str == "abort" || path_str == "std::process::abort" {
                self.findings.push(CodeIssue {
                    severity: IssueSeverity::Warning,
                    category: "Security".to_string(),
                    message: "process::abort() — immediate termination without cleanup".to_string(),
                    line_number: Some(ln),
                    column_number: None,
                    rule_id: "AST-PROCESS-ABORT".to_string(),
                });
            }
        }

        syn::visit::visit_expr_call(self, node);
    }

    fn visit_expr_if(&mut self, node: &'ast syn::ExprIf) {
        syn::visit::visit_expr_if(self, node);
    }

    fn visit_stmt_macro(&mut self, node: &'ast syn::StmtMacro) {
        let mac_path = path_to_string(&node.mac.path);
        let lower = mac_path.to_lowercase();
        if lower == "panic" || lower == "todo" || lower == "unimplemented" || lower == "unreachable" {
            let ln = syn::spanned::Spanned::span(node).start().line;
            let severity = match lower.as_str() {
                "panic" | "unreachable" => IssueSeverity::Error,
                "todo" | "unimplemented" => IssueSeverity::Warning,
                _ => IssueSeverity::Warning,
            };
            self.findings.push(CodeIssue {
                severity,
                category: "Reliability".to_string(),
                message: format!(
                    "`{}!()` call in production code — {}",
                    mac_path,
                    match lower.as_str() {
                        "panic" => "will abort execution on error path. Use Result instead",
                        "todo" => "incomplete implementation placeholder",
                        "unimplemented" => "unimplemented code path. Fill in before release",
                        "unreachable" => "marks code as unreachable. Verify assumption holds",
                        _ => "use with caution in production",
                    },
                ),
                line_number: Some(ln),
                column_number: None,
                rule_id: format!("AST-{}-MACRO", mac_path.to_uppercase()),
            });
        }
        syn::visit::visit_stmt_macro(self, node);
    }

    fn visit_expr_macro(&mut self, node: &'ast syn::ExprMacro) {
        let mac_path = path_to_string(&node.mac.path);
        let lower = mac_path.to_lowercase();
        if lower == "panic" || lower == "todo" || lower == "unimplemented" || lower == "unreachable" {
            let ln = syn::spanned::Spanned::span(node).start().line;
            let severity = match lower.as_str() {
                "panic" | "unreachable" => IssueSeverity::Error,
                "todo" | "unimplemented" => IssueSeverity::Warning,
                _ => IssueSeverity::Warning,
            };
            self.findings.push(CodeIssue {
                severity,
                category: "Reliability".to_string(),
                message: format!(
                    "`{}!()` call in production code — {}",
                    mac_path,
                    match lower.as_str() {
                        "panic" => "will abort execution on error path. Use Result instead",
                        "todo" => "incomplete implementation placeholder",
                        "unimplemented" => "unimplemented code path. Fill in before release",
                        "unreachable" => "marks code as unreachable. Verify assumption holds",
                        _ => "use with caution in production",
                    },
                ),
                line_number: Some(ln),
                column_number: None,
                rule_id: format!("AST-{}-MACRO", mac_path.to_uppercase()),
            });
        }
        syn::visit::visit_expr_macro(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        let ln = syn::spanned::Spanned::span(&node.method).start().line;
        let method_name = node.method.to_string();
        if method_name == "unwrap" || method_name == "expect" {
            self.findings.push(CodeIssue {
                severity: IssueSeverity::Warning,
                category: "Reliability".to_string(),
                message: format!(
                    ".{}() in production code — will panic on Err/None. Use unwrap_or/error propagation instead",
                    method_name,
                ),
                line_number: Some(ln),
                column_number: None,
                rule_id: format!("AST-{}", method_name.to_uppercase()),
            });
        }

        syn::visit::visit_expr_method_call(self, node);
    }

    fn visit_local(&mut self, node: &'ast syn::Local) {
        if let Some(init) = &node.init {
            let ln = syn::spanned::Spanned::span(node).start().line;
            let expr = &*init.expr;
            let expr_text = quote::quote!(#expr).to_string();
            let it = expr_text.to_lowercase().replace("format !", "format!");
            if it.contains("select ") || it.contains("insert ") || it.contains("delete from ") || it.contains("update ") {
                if it.contains("format!") || it.contains("+ ") || it.contains("push_str") {
                    self.findings.push(CodeIssue {
                        severity: IssueSeverity::Critical,
                        category: "Security".to_string(),
                        message: "Dynamic SQL via format!/concat — SQL injection risk. Use parameterized queries".to_string(),
                        line_number: Some(ln),
                        column_number: None,
                        rule_id: "AST-SQL-FORMAT".to_string(),
                    });
                }
            }

            let pat_text = quote::quote!(#node.pat).to_string();
            let pat_lower = pat_text.to_lowercase();

            let mut hardcoded = false;
            if pat_lower.contains("password") || pat_lower.contains("api_key") || pat_lower.contains("secret") || pat_lower.contains("auth_token") || pat_lower.contains("token") || pat_lower.contains("credential") || pat_lower.contains("apikey") {
                if expr_text.contains(r#"""#) || expr_text.contains(r#"'"#) {
                    hardcoded = true;
                }
            }

            if !hardcoded {
                let expr_lower = expr_text.to_lowercase();
                if expr_lower.contains("password") || expr_lower.contains("api_key") || expr_lower.contains("secret") || expr_lower.contains("auth_token") {
                    if expr_lower.contains(r#""#) || expr_lower.contains(r#"'"#) {
                        hardcoded = true;
                    }
                }
            }

            if hardcoded {
                self.findings.push(CodeIssue {
                    severity: IssueSeverity::Error,
                    category: "Security".to_string(),
                    message: "Hardcoded secret detected — use environment variables or secret store".to_string(),
                    line_number: Some(ln),
                    column_number: None,
                    rule_id: "AST-HARDCODED-SECRET".to_string(),
                });
            }
        }
        syn::visit::visit_local(self, node);
    }

    fn visit_item(&mut self, item: &'ast Item) {
        if is_test_item(item) {
            syn::visit::visit_item(self, item);
            return;
        }

        let ln = syn::spanned::Spanned::span(item).start().line;

        if let Item::Fn(ref func) = item {
            let is_unsafe = func.sig.unsafety.is_some();

            let has_no_mangle = func.attrs.iter().any(|a| a.path().is_ident("no_mangle"));
            let has_export_name = func.attrs.iter().any(|a| a.path().is_ident("export_name"));
            if has_no_mangle || has_export_name {
                self.findings.push(CodeIssue {
                    severity: IssueSeverity::Warning,
                    category: "Security".to_string(),
                    message: format!(
                        "#[{}] on `{}` — symbol export may bypass ASLR. Ensure this is intentional",
                        if has_no_mangle { "no_mangle" } else { "export_name" },
                        func.sig.ident,
                    ),
                    line_number: Some(ln),
                    column_number: None,
                    rule_id: "AST-SYMBOL-EXPORT".to_string(),
                });
            }

            if is_unsafe {
                self.findings.push(CodeIssue {
                    severity: IssueSeverity::Warning,
                    category: "Security".to_string(),
                    message: format!(
                        "Unsafe fn `{}` — entire function body is unsafe. Prefer small unsafe blocks instead",
                        func.sig.ident,
                    ),
                    line_number: Some(ln),
                    column_number: None,
                    rule_id: "AST-UNSAFE-FN".to_string(),
                });
            }
        }

        syn::visit::visit_item(self, item);
    }
}

fn path_to_string(path: &syn::Path) -> String {
    path.segments.iter().map(|s| s.ident.to_string()).collect::<Vec<_>>().join("::")
}

/// Analyze Rust code using syn AST parser
pub fn analyze_rust_ast(code: &str) -> AstFindings {
    let mut issues = Vec::new();

    match syn::parse_file(code) {
        Ok(file) => {
            let mut visitor = RustSecurityVisitor {
                findings: &mut issues,
            };
            visitor.visit_file(&file);
            check_recursive_types(&file, &mut issues);
        }
        Err(first_err) => {
            let wrapped = format!("fn __ast_snippet() {{ {code} }}");
            if let Ok(file) = syn::parse_file(&wrapped) {
                let mut visitor = RustSecurityVisitor {
                    findings: &mut issues,
                };
                visitor.visit_file(&file);
                check_recursive_types(&file, &mut issues);
            } else {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Info,
                    category: "Security".to_string(),
                    message: format!("Code is not valid Rust — AST analysis skipped: {first_err}"),
                    line_number: None,
                    column_number: None,
                    rule_id: "AST-PARSE-ERROR".to_string(),
                });
            }
        }
    }

    let mut metrics = HashMap::new();
    metrics.insert("ast_issues".to_string(), issues.len() as f32);
    metrics.insert("ast_high_severity".to_string(),
        issues.iter().filter(|i| matches!(i.severity, IssueSeverity::Critical | IssueSeverity::Error)).count() as f32,
    );

    AstFindings { issues, metrics }
}

fn check_recursive_types(file: &syn::File, issues: &mut Vec<CodeIssue>) {
    for item in &file.items {
        if let syn::Item::Struct(ref s) = item {
            let type_name = s.ident.to_string();
            let field_types: Vec<String> = s.fields.iter()
                .map(|f| quote::quote!(#f.ty).to_string())
                .collect();

            if field_types.iter().any(|ft| ft.contains(&type_name)) {
                let line = syn::spanned::Spanned::span(item).start().line;
                if !field_types.iter().any(|ft| ft.contains("Box<")) && !field_types.iter().any(|ft| ft.contains("Arc<")) {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Warning,
                        category: "Security".to_string(),
                        message: format!(
                            "Recursive type `{}` without Box/Arc indirection — stack overflow on construction",
                            type_name,
                        ),
                        line_number: Some(line),
                        column_number: None,
                        rule_id: "AST-RECURSIVE-TYPE".to_string(),
                    });
                }
            }
        }
    }
}

/// Strip comments and string literals from code for context-aware pattern matching
pub fn strip_comments_and_strings(code: &str) -> String {
    let mut result = String::with_capacity(code.len());
    let chars: Vec<char> = code.chars().collect();
    let mut i = 0;
    let mut in_line_comment = false;
    let mut in_block_comment = false;
    let mut in_string = false;
    let mut in_char = false;

    while i < chars.len() {
        if in_line_comment {
            if chars[i] == '\n' {
                in_line_comment = false;
                result.push('\n');
            }
            i += 1;
            continue;
        }

        if in_block_comment {
            if i + 1 < chars.len() && chars[i] == '*' && chars[i + 1] == '/' {
                in_block_comment = false;
                i += 2;
                continue;
            }
            i += 1;
            continue;
        }

        if in_string {
            if chars[i] == '\\' && i + 1 < chars.len() {
                i += 2;
                continue;
            }
            if chars[i] == '"' {
                in_string = false;
                i += 1;
                result.push(' ');
                continue;
            }
            i += 1;
            continue;
        }

        if in_char {
            if chars[i] == '\\' && i + 1 < chars.len() {
                i += 2;
                continue;
            }
            if chars[i] == '\'' {
                in_char = false;
                i += 1;
                continue;
            }
            i += 1;
            continue;
        }

        if i + 1 < chars.len() && chars[i] == '/' && chars[i + 1] == '/' {
            in_line_comment = true;
            i += 2;
            continue;
        }

        if i + 1 < chars.len() && chars[i] == '/' && chars[i + 1] == '*' {
            in_block_comment = true;
            i += 2;
            continue;
        }

        if chars[i] == '"' {
            in_string = true;
            i += 1;
            result.push(' ');
            continue;
        }

        if chars[i] == '\'' {
            in_char = true;
            i += 1;
            continue;
        }

        result.push(chars[i]);
        i += 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_comments_and_strings() {
        let code = "let x = \"password = 'hunter2'\"; // this is a comment\nlet y = 1;";
        let stripped = strip_comments_and_strings(code);
        assert!(!stripped.contains("hunter2"), "string content stripped");
        assert!(!stripped.contains("comment"), "comment stripped");
        assert!(stripped.contains("let y = 1;"), "code preserved");
    }

    #[test]
    fn test_ast_unsafe_high_risk() {
        let code = r#"
fn foo() {
    unsafe {
        let p = &42 as *const i32;
        let v = *p;
    }
}
"#;
        let findings = analyze_rust_ast(code);
        assert!(findings.issues.iter().any(|i| i.rule_id == "AST-UNSAFE-HIGH"));
    }

    #[test]
    fn test_ast_command_injection() {
        let code = r#"
fn run(cmd: &str) {
    let output = std::process::Command::new(cmd).output();
}
"#;
        let findings = analyze_rust_ast(code);
        assert!(findings.issues.iter().any(|i| i.rule_id == "AST-CMD-INJECTION"));
    }

    #[test]
    fn test_ast_transmute() {
        let code = r#"
fn cast(x: u32) -> f32 {
    unsafe { std::mem::transmute(x) }
}
"#;
        let findings = analyze_rust_ast(code);
        assert!(findings.issues.iter().any(|i| i.rule_id == "AST-TRANSMUTE"));
    }

    #[test]
    fn test_ast_recursive_type_no_box() {
        let code = r#"
struct Node {
    value: i32,
    children: Vec<Node>,
}
"#;
        let findings = analyze_rust_ast(code);
        assert!(findings.issues.iter().any(|i| i.rule_id == "AST-RECURSIVE-TYPE"));
    }

    #[test]
    fn test_ast_format_sql() {
        let code = r#"
let query = format!("SELECT * FROM users WHERE id = {}", user_id);
"#;
        let findings = analyze_rust_ast(code);
        assert!(findings.issues.iter().any(|i| i.rule_id == "AST-SQL-FORMAT"));
    }

    #[test]
    fn test_ast_panic_macro() {
        let code = r#"
fn check(x: i32) {
    if x < 0 {
        panic!("negative value: {}", x);
    }
}
"#;
        let findings = analyze_rust_ast(code);
        assert!(findings.issues.iter().any(|i| i.rule_id == "AST-PANIC-MACRO"));
    }

    #[test]
    fn test_ast_todo_macro() {
        let code = r#"
fn process() {
    todo!("implement this");
}
"#;
        let findings = analyze_rust_ast(code);
        assert!(findings.issues.iter().any(|i| i.rule_id == "AST-TODO-MACRO"));
    }

    #[test]
    fn test_ast_unwrap() {
        let code = r#"
fn get_value(map: &std::collections::HashMap<String, String>, key: &str) -> String {
    map.get(key).unwrap().clone()
}
"#;
        let findings = analyze_rust_ast(code);
        assert!(findings.issues.iter().any(|i| i.rule_id == "AST-UNWRAP"));
    }

    #[test]
    fn test_ast_expect() {
        let code = r#"
fn parse(s: &str) -> i32 {
    s.parse::<i32>().expect("invalid number")
}
"#;
        let findings = analyze_rust_ast(code);
        assert!(findings.issues.iter().any(|i| i.rule_id == "AST-EXPECT"));
    }

    #[test]
    fn test_ast_process_exit() {
        let code = r#"
fn cleanup() {
    std::process::exit(0);
}
"#;
        let findings = analyze_rust_ast(code);
        assert!(findings.issues.iter().any(|i| i.rule_id == "AST-PROCESS-EXIT"));
    }

    #[test]
    fn test_ast_unsafe_fn() {
        let code = r#"
unsafe fn dangerous(p: *const u8) -> u8 {
    *p
}
"#;
        let findings = analyze_rust_ast(code);
        assert!(findings.issues.iter().any(|i| i.rule_id == "AST-UNSAFE-FN"));
    }

    #[test]
    fn test_ast_hardcoded_secret() {
        let code = r#"
let api_key = "sk-1234567890abcdef";
"#;
        let findings = analyze_rust_ast(code);
        assert!(findings.issues.iter().any(|i| i.rule_id == "AST-HARDCODED-SECRET"));
    }

    #[test]
    fn test_ast_unimplemented_macro() {
        let code = r#"
fn feature() {
    unimplemented!("not ready yet");
}
"#;
        let findings = analyze_rust_ast(code);
        assert!(findings.issues.iter().any(|i| i.rule_id == "AST-UNIMPLEMENTED-MACRO"));
    }

    #[test]
    fn test_ast_no_false_positive_in_test_cfg() {
        let code = r#"
fn production() -> i32 { 42 }

#[cfg(test)]
mod tests {
    #[test]
    fn test_example() {
        let x: Option<i32> = Some(42);
        assert_eq!(x.unwrap(), 42);
    }
}
"#;
        let findings = analyze_rust_ast(code);
        // unwrap inside #[cfg(test)] should be filtered out
        let unwrap_count = findings.issues.iter().filter(|i| i.rule_id == "AST-UNWRAP").count();
        assert_eq!(unwrap_count, 0, "unwrap in #[cfg(test)] should not be flagged");
    }
}
