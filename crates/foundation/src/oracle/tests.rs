//! Tests untuk ORACLE framework

use super::*;
use super::prelude::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> OracleConfig {
        OracleConfig::default()
    }

    fn create_test_code() -> String {
        r#"
def fibonacci(n):
    """Calculate fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    result = fibonacci(10)
    print(f"Result: {result}")
"#
        .to_string()
    }

    #[test]
    fn test_oracle_config_default() {
        let config = OracleConfig::default();
        assert_eq!(config.backbone.d_model, 4096);
        assert_eq!(config.backbone.n_heads, 32);
        assert_eq!(config.backbone.n_experts, 8);
        assert_eq!(config.rope.base, 10000.0);
        assert_eq!(config.pretraining.fim_probability, 0.5);
        assert_eq!(config.dpo.learning_rate, 1e-5);
    }

    #[test]
    fn test_oracle_backbone_creation() {
        let config = OracleBackboneConfig::default();
        let backbone = OracleBackbone::new(config, 50000);
        
        // Test that backbone was created
        assert_eq!(backbone.config.d_model, 4096);
        assert_eq!(backbone.config.n_heads, 32);
    }

    #[test]
    fn test_sparse_moe_layer() {
        let config = OracleBackboneConfig::default();
        let moe_layer = SparseMoELayer::new(config);
        
        // Create test input
        let mut test_input = ndarray::Array2::zeros((2, 128, 4096));
        test_input.fill(0.1);
        
        // Test forward pass
        let result = moe_layer.forward(&test_input.view((256, 4096)).unwrap());
        assert!(result.is_ok());
    }

    #[test]
    fn test_extended_rope() {
        let config = ExtendedRopeConfig::default();
        let rope = ExtendedRope::new(config);
        
        // Test position embeddings
        let positions = vec![0, 1, 2, 3, 4];
        let (cos_emb, sin_emb) = rope.get_position_embeddings(&positions).unwrap();
        
        assert_eq!(cos_emb.dim().0, 5);
        assert_eq!(sin_emb.dim().0, 5);
        assert_eq!(cos_emb.dim().1, 128); // head_dim = 4096 / 32
        assert_eq!(sin_emb.dim().1, 128);
    }

    #[test]
    fn test_fim_processor() {
        let config = OraclePretrainingConfig::default();
        let fim_processor = FimProcessor::new(config);
        
        // Test FIM transformation
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let fim_example = fim_processor.apply_fim(&tokens).unwrap();
        
        assert!(!fim_example.prefix.is_empty());
        assert!(!fim_example.middle.is_empty() || !fim_example.suffix.is_empty());
        assert_eq!(fim_example.original, tokens);
    }

    #[test]
    fn test_contrastive_learning() {
        let config = OraclePretrainingConfig::default();
        let contrastive_learner = ContrastiveCodeLearning::new(config);
        
        // Create test code snippets
        let snippets = vec![
            CodeSnippet::new(vec![1, 2, 3, 4], "python".to_string()),
            CodeSnippet::new(vec![5, 6, 7, 8], "python".to_string()),
        ];
        
        // Test contrastive pair creation
        let pairs = contrastive_learner.create_contrastive_pairs(&snippets).unwrap();
        assert!(!pairs.is_empty());
        
        // Test contrastive loss computation
        let loss = contrastive_learner.compute_contrastive_loss(&pairs).unwrap();
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_code_dpo_trainer() {
        let config = CodeDpoConfig::default();
        let model = CodeModel::new(50000, 8192);
        let reference_model = CodeModel::new(50000, 8192);
        let dpo_trainer = CodeDpoTrainer::new(config, model, reference_model);
        
        // Test preference pair generation
        let prompt = "Write a function to calculate factorial";
        let pairs = dpo_trainer.generate_preferences(prompt, 2).unwrap();
        assert_eq!(pairs.len(), 2);
        
        // Test training step
        let loss = dpo_trainer.training_step(&pairs).unwrap();
        assert!(loss.total_loss >= 0.0);
    }

    #[test]
    fn test_code_analyzer() {
        let analyzer = CodeAnalyzer::new();
        let code = create_test_code();
        
        // Test quality analysis
        let quality_score = analyzer.analyze_quality(&code).unwrap();
        assert!(quality_score >= 0.0 && quality_score <= 1.0);
        
        // Test metrics extraction
        let metrics = analyzer.get_quality_metrics(&code);
        assert!(metrics.contains_key("lines_of_code"));
        assert!(metrics.contains_key("cyclomatic_complexity"));
    }

    #[test]
    fn test_security_analyzer() {
        let analyzer = SecurityAnalyzer::new();
        
        // Test secure code
        let secure_code = "def safe_function(x):\n    return x * 2";
        let security_score = analyzer.analyze_security(&secure_code).unwrap();
        assert!(security_score > 0.5);
        
        // Test vulnerable code
        let vulnerable_code = "def unsafe_function():\n    exec(user_input)";
        let vulnerable_score = analyzer.analyze_security(&vulnerable_code).unwrap();
        assert!(vulnerable_score < 0.5);
    }

    #[test]
    fn test_efficiency_analyzer() {
        let analyzer = EfficiencyAnalyzer::new();
        
        // Test efficient code
        let efficient_code = "def efficient_sum(lst):\n    return sum(lst)";
        let efficiency_score = analyzer.analyze_efficiency(&efficient_code).unwrap();
        assert!(efficiency_score > 0.5);
        
        // Test inefficient code
        let inefficient_code = "def inefficient_sum(lst):\n    total = 0\n    for item in lst:\n        for i in range(len(lst)):\n            total += item\n    return total";
        let inefficient_score = analyzer.analyze_efficiency(&inefficient_code).unwrap();
        assert!(inefficient_score < 0.5);
    }

    #[test]
    fn test_code_tokenizer() {
        let tokenizer = CodeTokenizer::new();
        let code = create_test_code();
        
        // Test tokenization
        let tokens = tokenizer.tokenize(&code).unwrap();
        assert!(!tokens.is_empty());
        assert_eq!(tokens[0], tokenizer.special_tokens.bos);
        assert_eq!(tokens[tokens.len() - 1], tokenizer.special_tokens.eos);
        
        // Test decoding
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_code_parser() {
        let parser = CodeParser::new("python");
        let code = create_test_code();
        
        // Test parsing
        let ast = parser.parse(&code).unwrap();
        assert_eq!(ast.language, "python");
        assert!(!ast.functions.is_empty());
        assert!(ast.functions.len() >= 2); // fibonacci and main
    }

    #[test]
    fn test_code_formatter() {
        let formatter = CodeFormatter::new("python");
        let unformatted_code = "def test():print('hello')";
        
        // Test formatting
        let formatted = formatter.format(&unformatted_code).unwrap();
        assert!(formatted.contains('\n')); // Should have proper formatting
    }

    #[test]
    fn test_code_metrics() {
        let metrics = CodeMetrics::new("python");
        let code = create_test_code();
        
        // Test metrics calculation
        let result = metrics.calculate_metrics(&code).unwrap();
        assert_eq!(result.language, "python");
        assert!(result.total_lines > 0);
        assert!(result.code_lines > 0);
        assert!(result.complexity > 0.0);
    }

    #[test]
    fn test_code_verifier() {
        let verifier = CodeVerifier::new();
        let code = create_test_code();
        
        // Test verification
        let score = verifier.verify_code(&code).unwrap();
        assert!(score >= 0.0 && score <= 1.0);
        
        // Test detailed verification
        let results = verifier.verify_detailed(&code, "python").unwrap();
        assert!(!results.is_empty());
        assert!(results.len() == 4); // Security, Performance, Correctness, Style
    }

    #[test]
    fn test_oracle_trainer_creation() {
        let config = OracleConfig::default();
        let trainer = OracleTrainer::new(config, 50000).unwrap();
        
        // Test trainer creation
        assert_eq!(trainer.config.backbone.d_model, 4096);
        assert_eq!(trainer.config.pretraining.fim_probability, 0.5);
    }

    #[test]
    fn test_training_batch_creation() {
        let examples = vec![
            TrainingExample {
                tokens: vec![1, 2, 3, 4],
                language: "python".to_string(),
                metadata: HashMap::new(),
            },
        ];
        
        let batch = TrainingBatch {
            examples,
        };
        
        assert_eq!(batch.examples.len(), 1);
        assert_eq!(batch.examples[0].language, "python");
    }

    #[test]
    fn test_dual_loss_calculator() {
        let config = OraclePretrainingConfig::default();
        let calculator = DualLossCalculator::new(config);
        
        let examples = vec![
            TrainingExample {
                tokens: vec![1, 2, 3, 4],
                language: "python".to_string(),
                metadata: HashMap::new(),
            },
        ];
        
        let batch = TrainingBatch { examples };
        let loss = calculator.compute_dual_loss(&batch).unwrap();
        
        assert!(loss.fim_loss >= 0.0);
        assert!(loss.contrastive_loss >= 0.0);
        assert!(loss.total_loss >= 0.0);
    }

    #[test]
    fn test_pretrainer() {
        let config = OraclePretrainingConfig::default();
        let mut pretrainer = OraclePretrainer::new(config);
        
        let examples = vec![
            TrainingExample {
                tokens: vec![1, 2, 3, 4],
                language: "python".to_string(),
                metadata: HashMap::new(),
            },
        ];
        
        let batch = TrainingBatch { examples };
        let loss = pretrainer.training_step(&batch).unwrap();
        
        assert!(loss.fim_loss >= 0.0);
        assert!(loss.contrastive_loss >= 0.0);
        assert!(loss.total_loss >= 0.0);
    }

    #[test]
    fn test_position_tracker() {
        let mut tracker = CrossFilePositionTracker::new();
        
        // Test position tracking
        let pos1 = tracker.add_position(0);
        let pos2 = tracker.add_position(0);
        let pos3 = tracker.add_position(1);
        
        assert_eq!(pos1, 0);
        assert_eq!(pos2, 1);
        assert_eq!(pos3, 0);
        
        // Test same file check
        assert!(tracker.same_file(pos1, pos2));
        assert!(!tracker.same_file(pos1, pos3));
    }

    #[test]
    fn test_frequency_scheduler() {
        let mut scheduler = FrequencyScheduler::new(1.0, 2.0, 100);
        
        // Test initial scaling
        assert_eq!(scheduler.get_scaling(), 1.0);
        
        // Test after some steps
        for _ in 0..50 {
            scheduler.step();
        }
        let scaling = scheduler.get_scaling();
        assert!(scaling > 1.0 && scaling < 2.0);
        
        // Test after warmup
        for _ in 0..100 {
            scheduler.step();
        }
        assert_eq!(scheduler.get_scaling(), 2.0);
    }

    #[test]
    fn test_utility_functions() {
        // Test config validation
        let config = OracleConfig::default();
        assert!(trainer::utils::validate_config(&config).is_ok());
        
        // Test language detection
        let python_code = "def test(): pass";
        let js_code = "function test() {}";
        assert_eq!(code_utils::utils::detect_language(python_code), "python");
        assert_eq!(code_utils::utils::detect_language(js_code), "javascript");
        
        // Test syntax validation
        let valid_code = "def test():\n    return 1";
        let invalid_code = "def test():\n    return 1\n    return 2"; // Unreachable code
        let valid_result = code_utils::utils::validate_syntax(valid_code, "python");
        let invalid_result = code_utils::utils::validate_syntax(invalid_code, "python");
        
        assert!(valid_result.is_valid);
        assert!(!invalid_result.is_valid || !invalid_result.warnings.is_empty());
    }

    #[test]
    fn test_training_time_estimation() {
        let config = OracleConfig::default();
        let dataset_size = 10000;
        let vocab_size = 50000;
        
        let estimate = trainer::utils::estimate_training_time(&config, dataset_size, vocab_size);
        
        assert!(estimate.pretraining_steps > 0);
        assert!(estimate.alignment_steps > 0);
        assert!(estimate.total_steps > 0);
        assert!(estimate.estimated_hours > 0.0);
    }

    #[test]
    fn test_training_report_generation() {
        let mut result = TrainingResult::new();
        
        // Create mock results
        let mut pretraining_result = PretrainingResult::new();
        pretraining_result.final_loss = 0.5;
        pretraining_result.total_epochs = 5;
        result.pretraining_result = Some(pretraining_result);
        
        let mut alignment_result = AlignmentResult::new();
        alignment_result.final_loss = 0.3;
        alignment_result.total_epochs = 3;
        result.alignment_result = Some(alignment_result);
        
        // Generate report
        let report = trainer::utils::create_training_report(&result);
        
        assert!(report.contains("ORACLE Training Report"));
        assert!(report.contains("Pretraining Results"));
        assert!(report.contains("Alignment Results"));
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_oracle_workflow() {
        let config = OracleConfig::default();
        let mut trainer = OracleTrainer::new(config, 50000).unwrap();
        
        // Create training data
        let training_data = vec![
            TrainingExample {
                tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
                language: "python".to_string(),
                metadata: {
                    let mut map = HashMap::new();
                    map.insert("file_id".to_string(), "0");
                    map
                },
            },
            TrainingExample {
                tokens: vec![9, 10, 11, 12, 13, 14, 15, 16],
                language: "python".to_string(),
                metadata: {
                    let mut map = HashMap::new();
                    map.insert("file_id".to_string(), "1");
                    map
                },
            },
        ];
        
        // Test training (simplified - just test that it doesn't panic)
        // In real scenario, this would run for multiple epochs
        let result = trainer.train(&training_data);
        
        assert!(result.is_ok());
        let training_result = result.unwrap();
        assert!(training_result.final_state.phase == TrainingPhase::Complete);
    }

    #[test]
    fn test_code_processing_pipeline() {
        let code = r#"
import os
import sys

def process_data(data):
    """Process input data"""
    if not data:
        return None
    
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    
    return result

if __name__ == "__main__":
    input_data = [1, 2, 3, 4, 5]
    output = process_data(input_data)
    print(f"Output: {output}")
"#;
        
        // Test tokenization
        let tokenizer = CodeTokenizer::new();
        let tokens = tokenizer.tokenize(&code).unwrap();
        assert!(!tokens.is_empty());
        
        // Test parsing
        let parser = CodeParser::new("python");
        let ast = parser.parse(&code).unwrap();
        assert_eq!(ast.language, "python");
        assert!(ast.functions.len() >= 2); // process_data and main
        
        // Test analysis
        let analyzer = CodeAnalyzer::new();
        let quality = analyzer.analyze_quality(&code).unwrap();
        assert!(quality > 0.0);
        
        // Test verification
        let verifier = CodeVerifier::new();
        let score = verifier.verify_code(&code).unwrap();
        assert!(score >= 0.0 && score <= 1.0);
        
        // Test metrics
        let metrics = CodeMetrics::new("python");
        let result = metrics.calculate_metrics(&code).unwrap();
        assert!(result.total_lines > 0);
        assert!(result.complexity > 0.0);
    }

    #[test]
    fn test_multi_language_support() {
        let languages = vec!["python", "javascript", "java", "cpp"];
        
        for language in languages {
            // Test parser
            let parser = CodeParser::new(language);
            let ast = parser.parse("function test() {}", language).unwrap();
            assert_eq!(ast.language, language);
            
            // Test formatter
            let formatter = CodeFormatter::new(language);
            let formatted = formatter.format("function test() {}", language).unwrap();
            assert!(!formatted.is_empty());
            
            // Test metrics
            let metrics = CodeMetrics::new(language);
            let result = metrics.calculate_metrics("function test() {}", language).unwrap();
            assert_eq!(result.language, language);
        }
    }

    #[test]
    fn test_performance_characteristics() {
        let config = OracleBackboneConfig::default();
        let backbone = OracleBackbone::new(config, 50000);
        
        // Test parameter count
        let param_count = backbone::utils::count_parameters(&backbone);
        assert!(param_count > 0);
        
        // Test FLOPs calculation
        let flops = backbone::utils::calculate_flops(&config, 32, 1024);
        assert!(flops > 0);
        
        // Test memory usage
        let memory = backbone::utils::estimate_memory_usage(&config, 32, 1024);
        assert!(memory > 0);
    }

    #[test]
    fn test_error_handling() {
        // Test invalid configurations
        let mut invalid_config = OracleConfig::default();
        invalid_config.backbone.d_model = 0;
        
        let trainer_result = OracleTrainer::new(invalid_config, 50000);
        assert!(trainer_result.is_err());
        
        // Test empty code
        let tokenizer = CodeTokenizer::new();
        let empty_tokens = tokenizer.tokenize("").unwrap();
        assert_eq!(empty_tokens.len(), 2); // BOS and EOS only
        
        // Test invalid language
        let parser = CodeParser::new("invalid_language");
        let ast = parser.parse("def test() {}", "invalid_language").unwrap();
        assert_eq!(ast.language, "invalid_language");
    }
}
