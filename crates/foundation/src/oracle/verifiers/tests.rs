//! Comprehensive tests for performance verifier

use super::*;
use crate::oracle::verifiers::performance::*;

#[cfg(test)]
mod performance_verifier_tests {
    use super::*;

    #[test]
    fn test_performance_verifier_creation() {
        let verifier = PerformanceVerifier::new();
        assert_eq!(verifier.verifier_name(), "PerformanceVerifier");
        assert_eq!(verifier.verifier_type(), VerifierType::Performance);
    }

    #[test]
    fn test_empty_code_validation() {
        let verifier = PerformanceVerifier::new();
        let result = verifier.verify("", "rust");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn test_large_code_validation() {
        let verifier = PerformanceVerifier::new();
        let large_code = "x".repeat(1_000_001);
        let result = verifier.verify(&large_code, "rust");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too large"));
    }

    #[test]
    fn test_rust_clone_detection() {
        let verifier = PerformanceVerifier::new();
        let code_with_clones = r#"
            let a = vec![1, 2, 3];
            let b = a.clone();
            let c = b.clone();
            let d = c.clone();
            let e = d.clone();
            let f = e.clone();
            let g = f.clone();
        "#;
        
        let result = verifier.verify(code_with_clones, "rust");
        assert!(result.is_ok());
        
        let verification_result = result.unwrap();
        let clone_issues: Vec<_> = verification_result.issues
            .iter()
            .filter(|issue| issue.rule_id == "rust_excessive_clone")
            .collect();
        
        assert!(!clone_issues.is_empty());
        assert!(clone_issues[0].message.contains("excessive cloning"));
    }

    #[test]
    fn test_rust_intermediate_allocation() {
        let verifier = PerformanceVerifier::new();
        let code_with_alloc = r#"
            let data = vec![1, 2, 3, 4, 5];
            let processed: Vec<_> = data.iter().map(|x| x * 2).collect();
        "#;
        
        let result = verifier.verify(code_with_alloc, "rust");
        assert!(result.is_ok());
        
        let verification_result = result.unwrap();
        let alloc_issues: Vec<_> = verification_result.issues
            .iter()
            .filter(|issue| issue.rule_id == "rust_intermediate_alloc")
            .collect();
        
        assert!(!alloc_issues.is_empty());
    }

    #[test]
    fn test_nested_loop_detection() {
        let verifier = PerformanceVerifier::new();
        let code_with_nested_loops = r#"
            for i in 0..100 {
                for j in 0..100 {
                    println!("{} {}", i, j);
                }
            }
        "#;
        
        let result = verifier.verify(code_with_nested_loops, "rust");
        assert!(result.is_ok());
        
        let verification_result = result.unwrap();
        let nested_issues: Vec<_> = verification_result.issues
            .iter()
            .filter(|issue| issue.rule_id == "nested_loops")
            .collect();
        
        assert!(!nested_issues.is_empty());
    }

    #[test]
    fn test_python_range_len_detection() {
        let verifier = PerformanceVerifier::new();
        let python_code = r#"
            items = [1, 2, 3, 4, 5]
            for i in range(len(items)):
                print(items[i])
        "#;
        
        let result = verifier.verify(python_code, "python");
        assert!(result.is_ok());
        
        let verification_result = result.unwrap();
        let range_issues: Vec<_> = verification_result.issues
            .iter()
            .filter(|issue| issue.rule_id == "py_range_len")
            .collect();
        
        assert!(!range_issues.is_empty());
    }

    #[test]
    fn test_javascript_dom_query_loop() {
        let verifier = PerformanceVerifier::new();
        let js_code = r#"
            for (let i = 0; i < 100; i++) {
                const element = document.getElementById('item-' + i);
                element.style.color = 'red';
            }
        "#;
        
        let result = verifier.verify(js_code, "javascript");
        assert!(result.is_ok());
        
        let verification_result = result.unwrap();
        let dom_issues: Vec<_> = verification_result.issues
            .iter()
            .filter(|issue| issue.rule_id == "js_dom_query_loop")
            .collect();
        
        assert!(!dom_issues.is_empty());
    }

    #[test]
    fn test_performance_thresholds() {
        let thresholds = PerformanceThresholds::default();
        
        // Test time scoring
        assert_eq!(thresholds.get_time_score(5), 1.0);
        assert_eq!(thresholds.get_time_score(50), 0.8);
        assert_eq!(thresholds.get_time_score(500), 0.6);
        assert_eq!(thresholds.get_time_score(5000), 0.4);
        assert_eq!(thresholds.get_time_score(50000), 0.2);
        
        // Test memory scoring
        assert_eq!(thresholds.get_memory_score(0.5), 1.0);
        assert_eq!(thresholds.get_memory_score(5.0), 0.8);
        assert_eq!(thresholds.get_memory_score(50.0), 0.6);
        assert_eq!(thresholds.get_memory_score(500.0), 0.4);
        assert_eq!(thresholds.get_memory_score(5000.0), 0.2);
    }

    #[test]
    fn test_score_calculation() {
        let verifier = PerformanceVerifier::new();
        let clean_code = r#"
            fn main() {
                println!("Hello, world!");
            }
        "#;
        
        let result = verifier.verify(clean_code, "rust");
        assert!(result.is_ok());
        
        let verification_result = result.unwrap();
        assert!(verification_result.score >= 0.7); // Should pass with high score
        assert!(verification_result.passed);
    }

    #[test]
    fn test_issue_deduplication() {
        let verifier = PerformanceVerifier::new();
        let code_with_duplicates = r#"
            for i in 0..10 {
                for j in 0..10 {
                    println!("{} {}", i, j);
                }
            }
            for i in 0..10 {
                for j in 0..10 {
                    println!("{} {}", i, j);
                }
            }
        "#;
        
        let result = verifier.verify(code_with_duplicates, "rust");
        assert!(result.is_ok());
        
        let verification_result = result.unwrap();
        let nested_issues: Vec<_> = verification_result.issues
            .iter()
            .filter(|issue| issue.rule_id == "nested_loops")
            .collect();
        
        // Should only have one nested loop issue despite two occurrences
        assert_eq!(nested_issues.len(), 1);
    }

    #[test]
    fn test_performance_suggestions() {
        let verifier = PerformanceVerifier::new();
        let code_with_issues = r#"
            let a = vec![1, 2, 3];
            let b = a.clone();
            let c = b.clone();
            let d = c.clone();
            let e = d.clone();
            let f = e.clone();
            let g = f.clone();
            
            for i in 0..100 {
                for j in 0..100 {
                    println!("{} {}", i, j);
                }
            }
        "#;
        
        let result = verifier.verify(code_with_issues, "rust");
        assert!(result.is_ok());
        
        let verification_result = result.unwrap();
        assert!(!verification_result.suggestions.is_empty());
        
        // Should have suggestions for both nested loops and cloning
        let suggestions_text = verification_result.suggestions.join(" ");
        assert!(suggestions_text.contains("nested") || suggestions_text.contains("borrowing"));
    }

    #[test]
    fn test_edge_cases() {
        let verifier = PerformanceVerifier::new();
        
        // Test with special characters
        let special_chars = "fn main() { println!(\"🚀 Hello 🌍\"); }";
        let result = verifier.verify(special_chars, "rust");
        assert!(result.is_ok());
        
        // Test with very long line
        let long_line = "x".repeat(10000);
        let result = verifier.verify(&long_line, "rust");
        assert!(result.is_ok());
        
        // Test with comments only
        let comments_only = r#"
            // This is a comment
            /* This is a block comment */
            /// This is a doc comment
        "#;
        let result = verifier.verify(comments_only, "rust");
        assert!(result.is_ok());
    }

    #[test]
    fn test_multiple_languages() {
        let verifier = PerformanceVerifier::new();
        
        let rust_code = "let x = vec![1,2,3].clone();";
        let python_code = "for i in range(len(x)): pass";
        let js_code = "for(let i=0;i<10;i++) document.getElementById('test');";
        
        for (code, lang) in [(rust_code, "rust"), (python_code, "python"), (js_code, "javascript")] {
            let result = verifier.verify(code, lang);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_metrics_calculation() {
        let verifier = PerformanceVerifier::new();
        let code = "fn main() { println!(\"Hello\"); }";
        
        let result = verifier.verify(code, "rust");
        assert!(result.is_ok());
        
        let verification_result = result.unwrap();
        
        // Check that metrics are present
        assert!(verification_result.metrics.contains_key("performance_score"));
        assert!(verification_result.metrics.contains_key("performance_issue_count"));
        assert!(verification_result.metrics.contains_key("high_impact_count"));
        
        // Check metric values are reasonable
        let score = verification_result.metrics["performance_score"];
        assert!(score >= 0.0 && score <= 1.0);
    }
}
