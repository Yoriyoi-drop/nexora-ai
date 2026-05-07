//! Complete ORACLE Example
//! 
//! Contoh lengkap penggunaan ORACLE framework untuk pelatihan
//! model bahasa dengan semua komponen terintegrasi.

use nexora_model::oracle::prelude::*;
use std::fs;

fn main() -> anyhow::Result<()> {
    println!("🚀 ORACLE Framework Example");
    println!("===========================");
    
    // 1. Setup konfigurasi lengkap
    println!("📋 Setting up ORACLE configuration...");
    let config = create_oracle_config();
    println!("✅ Configuration created");
    
    // 2. Inisialisasi ORACLE trainer
    println!("\n🤖 Initializing ORACLE trainer...");
    let vocab_size = 50000;
    let mut trainer = OracleTrainer::new(config, vocab_size)?;
    println!("✅ ORACLE trainer initialized");
    
    // 3. Persiapkan training data
    println!("\n📚 Preparing training data...");
    let training_data = prepare_training_data()?;
    println!("✅ Training data prepared: {} examples", training_data.len());
    
    // 4. Analisis data awal
    println!("\n📊 Analyzing training data...");
    analyze_training_data(&training_data)?;
    
    // 5. Jalankan training ORACLE
    println!("\n🏋️ Starting ORACLE training...");
    println!("=============================");
    
    let training_result = trainer.train(&training_data)?;
    
    // 6. Tampilkan hasil training
    println!("\n🎉 ORACLE Training Completed!");
    println!("=============================");
    
    display_training_results(&training_result)?;
    
    // 7. Demo komponen individual
    println!("\n🔧 Individual Component Demos");
    println!("===============================");
    
    demo_backbone_components()?;
    demo_rope_components()?;
    demo_pretraining_components()?;
    demo_alignment_components()?;
    demo_code_utilities()?;
    demo_verifiers()?;
    
    // 8. Save checkpoint final
    println!("\n💾 Saving final checkpoint...");
    let checkpoint_path = "oracle_final_checkpoint.json";
    trainer.save_checkpoint(checkpoint_path.to_string())?;
    println!("✅ Final checkpoint saved to {}", checkpoint_path);
    
    // 9. Generate training report
    println!("\n📄 Generating training report...");
    let report = trainer::utils::create_training_report(&training_result);
    fs::write("oracle_training_report.md", report)?;
    println!("✅ Training report saved to oracle_training_report.md");
    
    println!("\n🎊 ORACLE example completed successfully!");
    Ok(())
}

/// Buat konfigurasi ORACLE lengkap
fn create_oracle_config() -> OracleConfig {
    // Backbone configuration
    let backbone = OracleBackboneConfig {
        d_model: 4096,
        n_heads: 32,
        n_experts: 8,
        top_k: 2,
        latent_dim: 512,
        context_size: 32768,
        mlp_hidden: 16384,
        dropout: 0.1,
    };
    
    // RoPE configuration
    let rope = ExtendedRopeConfig {
        d_model: 4096,
        n_heads: 32,
        base: 10000.0,
        scaling_factor: 1.0,
        max_seq_len: 32768,
        dynamic_frequency: true,
        cross_file_factor: 0.1,
    };
    
    // Pretraining configuration
    let pretraining = OraclePretrainingConfig {
        fim_probability: 0.5,
        fim_span_ratio: 0.15,
        contrastive_weight: 0.1,
        contrastive_temperature: 0.1,
        contrastive_negatives: 8,
        vocab_size: 50000,
        max_seq_len: 8192,
    };
    
    // DPO configuration
    let dpo = CodeDpoConfig {
        learning_rate: 1e-5,
        beta: 0.1,
        max_seq_len: 8192,
        code_weight: 0.4,
        security_weight: 0.3,
        efficiency_weight: 0.3,
        regularization_strength: 0.01,
    };
    
    // Training configuration
    let training = TrainingConfig {
        pretraining_epochs: 3, // Dikurangi untuk demo
        alignment_epochs: 2,   // Dikurangi untuk demo
        batch_size: 16,
        learning_rate: 1e-4,
        warmup_steps: 500,
        max_seq_len: 8192,
        grad_clip_norm: 1.0,
        checkpoint_interval: 100,
        eval_interval: 50,
    };
    
    OracleConfig {
        backbone,
        rope,
        pretraining,
        dpo,
        training,
    }
}

/// Persiapkan training data
fn prepare_training_data() -> anyhow::Result<Vec<TrainingExample>> {
    let mut examples = Vec::new();
    
    // Python examples
    let python_code = r#"
def fibonacci(n):
    """Calculate fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    """Calculate factorial"""
    if n <= 1:
        return 1
    return n * factorial(n-1)

def main():
    result1 = fibonacci(10)
    result2 = factorial(5)
    print(f"Fibonacci: {result1}")
    print(f"Factorial: {result2}")
"#;
    
    let python_example = TrainingExample {
        tokens: tokenize_code(python_code),
        language: "python".to_string(),
        metadata: {
            let mut map = HashMap::new();
            map.insert("file_id".to_string(), "0");
            map.insert("filename".to_string(), "math_functions.py");
            map
        },
    };
    examples.push(python_example);
    
    // JavaScript examples
    let js_code = r#"
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

function main() {
    const result1 = fibonacci(10);
    const result2 = factorial(5);
    console.log(`Fibonacci: ${result1}`);
    console.log(`Factorial: ${result2}`);
}
"#;
    
    let js_example = TrainingExample {
        tokens: tokenize_code(js_code),
        language: "javascript".to_string(),
        metadata: {
            let mut map = HashMap::new();
            map.insert("file_id".to_string(), "1");
            map.insert("filename".to_string(), "math_functions.js");
            map
        },
    };
    examples.push(js_example);
    
    // Java examples
    let java_code = r#"
public class MathFunctions {
    public static int fibonacci(int n) {
        if (n <= 1) return n;
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
    
    public static int factorial(int n) {
        if (n <= 1) return 1;
        return n * factorial(n - 1);
    }
    
    public static void main(String[] args) {
        int result1 = fibonacci(10);
        int result2 = factorial(5);
        System.out.println("Fibonacci: " + result1);
        System.out.println("Factorial: " + result2);
    }
}
"#;
    
    let java_example = TrainingExample {
        tokens: tokenize_code(java_code),
        language: "java".to_string(),
        metadata: {
            let mut map = HashMap::new();
            map.insert("file_id".to_string(), "2");
            map.insert("filename".to_string(), "MathFunctions.java");
            map
        },
    };
    examples.push(java_example);
    
    // C++ examples
    let cpp_code = r#"
#include <iostream>
#include <vector>

int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int main() {
    int result1 = fibonacci(10);
    int result2 = factorial(5);
    std::cout << "Fibonacci: " << result1 << std::endl;
    std::cout << "Factorial: " << result2 << std::endl;
    return 0;
}
"#;
    
    let cpp_example = TrainingExample {
        tokens: tokenize_code(cpp_code),
        language: "cpp".to_string(),
        metadata: {
            let mut map = HashMap::new();
            map.insert("file_id".to_string(), "3");
            map.insert("filename".to_string(), "math_functions.cpp");
            map
        },
    };
    examples.push(cpp_example);
    
    Ok(examples)
}

/// Tokenize code (simplified)
fn tokenize_code(code: &str) -> Vec<i32> {
    // Simple word-based tokenization
    code.split_whitespace()
        .map(|word| {
            // Convert word to integer hash
            let hash = format!("{:x}", md5::compute(word.as_bytes()));
            u64::from_str_radix(&hash[..8], 16).unwrap_or(0) as i32
        })
        .collect()
}

/// Analisis training data
fn analyze_training_data(data: &[TrainingExample]) -> anyhow::Result<()> {
    println!("Training Data Analysis:");
    println!("  Total examples: {}", data.len());
    
    let mut language_counts = HashMap::new();
    let mut total_tokens = 0;
    
    for example in data {
        *language_counts.entry(example.language.clone()).or_insert(0) += 1;
        total_tokens += example.tokens.len();
    }
    
    println!("  Languages:");
    for (language, count) in language_counts {
        println!("    {}: {} examples", language, count);
    }
    
    println!("  Average tokens per example: {:.1}", total_tokens as f32 / data.len() as f32);
    
    // Analyze code complexity
    let mut total_complexity = 0.0;
    for example in data {
        let code = detokenize_code(&example.tokens);
        let complexity = code_utils::utils::analyze_code_complexity(&code);
        total_complexity += complexity.complexity_score;
    }
    
    println!("  Average complexity: {:.2}", total_complexity / data.len() as f32);
    
    Ok(())
}

/// Detokenize code (simplified)
fn detokenize_code(tokens: &[i32]) -> String {
    tokens.iter()
        .map(|&token| format!("token_{}", token))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Tampilkan hasil training
fn display_training_results(result: &TrainingResult) -> anyhow::Result<()> {
    println!("Training Results Summary:");
    println!("======================");
    
    // Pretraining results
    if let Some(pretraining) = &result.pretraining_result {
        println!("Pretraining:");
        println!("  Final Loss: {:.6}", pretraining.final_loss);
        println!("  Total Epochs: {}", pretraining.total_epochs);
        println!("  Epoch Metrics: {}", pretraining.epoch_metrics.len());
    }
    
    // Alignment results
    if let Some(alignment) = &result.alignment_result {
        println!("Alignment:");
        println!("  Final Loss: {:.6}", alignment.final_loss);
        println!("  Total Epochs: {}", alignment.total_epochs);
        println!("  Alignment Stats: {}", alignment.alignment_stats.len());
        
        if let Some(last_stats) = alignment.alignment_stats.last() {
            println!("  Final Security Score: {:.3}", last_stats.avg_security_score);
            println!("  Final Efficiency Score: {:.3}", last_stats.avg_efficiency_score);
            println!("  Final Quality Score: {:.3}", last_stats.avg_code_quality_score);
        }
    }
    
    // Final evaluation
    if let Some(metrics) = &result.final_metrics {
        println!("Final Evaluation:");
        println!("  Pretraining Loss: {:.6}", metrics.pretraining_loss);
        println!("  Alignment Loss: {:.6}", metrics.alignment_loss);
        println!("  Verification Score: {:.3}", metrics.verification_score);
        println!("  Overall Score: {:.3}", metrics.overall_score);
    }
    
    // Training state
    println!("Training State:");
    println!("  Final Epoch: {}", result.final_state.current_epoch);
    println!("  Final Step: {}", result.final_state.current_step);
    println!("  Final Phase: {:?}", result.final_state.phase);
    println!("  Best Validation Loss: {:.6}", result.final_state.best_validation_loss);
    
    Ok(())
}

/// Demo backbone components
fn demo_backbone_components() -> anyhow::Result<()> {
    println!("\n🏗️ Backbone Components Demo");
    println!("============================");
    
    // Test Sparse MoE
    println!("Testing Sparse MoE Layer...");
    let config = OracleBackboneConfig::default();
    let moe_layer = SparseMoELayer::new(config);
    
    let mut test_input = ndarray::Array3::zeros((2, 4, 4096));
    test_input.fill(0.1);
    
    let output = moe_layer.forward(&test_input.view((8, 4096)))?;
    println!("✅ Sparse MoE output shape: {:?}", output.dim());
    
    // Test Multi-head Latent Attention
    println!("Testing Multi-head Latent Attention...");
    let mla = MultiHeadLatentAttention::new(config);
    let mask = Some(ndarray::Array2::ones((4, 4)));
    
    let attn_output = mla.forward(&test_input, mask.as_ref())?;
    println!("✅ MLA output shape: {:?}", attn_output.dim());
    
    // Test expert usage
    let usage_stats = moe_layer.get_expert_usage(&test_input.view((8, 4096)))?;
    println!("✅ Expert usage: {:?}", usage_stats);
    
    Ok(())
}

/// Demo RoPE components
fn demo_rope_components() -> anyhow::Result<()> {
    println!("\n🔄 Extended RoPE Demo");
    println!("=====================");
    
    let config = ExtendedRopeConfig::default();
    let rope = ExtendedRope::new(config);
    
    // Test position embeddings
    let positions = vec![0, 1, 2, 3, 4];
    let (cos_emb, sin_emb) = rope.get_position_embeddings(&positions)?;
    println!("✅ Position embeddings shape: {:?}", cos_emb.dim());
    
    // Test cross-file awareness
    let mut test_input = ndarray::Array3::zeros((2, 5, 4096));
    test_input.fill(0.1);
    let file_ids = vec![0, 0, 1, 1, 0];
    
    let cross_file_output = rope.apply_cross_file_awareness(&test_input, &file_ids)?;
    println!("✅ Cross-file aware output shape: {:?}", cross_file_output.dim());
    
    // Test dynamic scaling
    rope.update_scaling(1.5);
    let new_scaling = rope.get_scaling();
    println!("✅ Updated scaling factor: {:.2}", new_scaling);
    
    Ok(())
}

/// Demo pretraining components
fn demo_pretraining_components() -> anyhow::Result<()> {
    println!("\n📚 Pretraining Components Demo");
    println!("===============================");
    
    let config = OraclePretrainingConfig::default();
    
    // Test FIM processor
    println!("Testing FIM Processor...");
    let fim_processor = FimProcessor::new(config.clone());
    let test_tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
    
    let fim_example = fim_processor.apply_fim(&test_tokens)?;
    println!("✅ FIM transformation: {:?}", fim_example.fim_type);
    
    let model_input = fim_processor.to_model_input(&fim_example)?;
    println!("✅ Model input length: {}", model_input.input_ids.len());
    
    // Test contrastive learning
    println!("Testing Contrastive Learning...");
    let contrastive_learner = ContrastiveCodeLearning::new(config);
    
    let snippets = vec![
        CodeSnippet::new(vec![1, 2, 3, 4], "python".to_string()),
        CodeSnippet::new(vec![5, 6, 7, 8], "python".to_string()),
        CodeSnippet::new(vec![9, 10, 11, 12], "python".to_string()),
    ];
    
    let pairs = contrastive_learner.create_contrastive_pairs(&snippets)?;
    println!("✅ Contrastive pairs created: {}", pairs.len());
    
    let contrastive_loss = contrastive_learner.compute_contrastive_loss(&pairs)?;
    println!("✅ Contrastive loss: {:.6}", contrastive_loss);
    
    // Test dual loss calculator
    println!("Testing Dual Loss Calculator...");
    let dual_calculator = DualLossCalculator::new(config);
    
    let examples = vec![
        TrainingExample {
            tokens: test_tokens,
            language: "python".to_string(),
            metadata: HashMap::new(),
        },
    ];
    
    let batch = TrainingBatch { examples };
    let dual_loss = dual_calculator.compute_dual_loss(&batch)?;
    println!("✅ Dual loss: {:.6}", dual_loss.total_loss);
    println!("  FIM Loss: {:.6}", dual_loss.fim_loss);
    println!("  Contrastive Loss: {:.6}", dual_loss.contrastive_loss);
    
    Ok(())
}

/// Demo alignment components
fn demo_alignment_components() -> anyhow::Result<()> {
    println!("\n🎯 Alignment Components Demo");
    println!("==========================");
    
    let config = CodeDpoConfig::default();
    
    // Test code analyzer
    println!("Testing Code Analyzer...");
    let analyzer = CodeAnalyzer::new();
    
    let good_code = r#"
def calculate_sum(numbers):
    """Calculate sum of numbers efficiently"""
    return sum(numbers)
    
def main():
    numbers = [1, 2, 3, 4, 5]
    result = calculate_sum(numbers)
    print(f"Sum: {result}")
"#;
    
    let bad_code = r#"
def calculate_sum(nums):
    return 0
    total = 0
    for i in range(len(nums)):
        total = total + nums[i]
    return total

def main():
    numbers = [1, 2, 3, 4, 5]
    result = calculate_sum(numbers)
    print("Sum: " + str(result))
"#;
    
    let good_quality = analyzer.analyze_quality(good_code)?;
    let bad_quality = analyzer.analyze_quality(bad_code)?;
    
    println!("✅ Good code quality: {:.3}", good_quality);
    println!("✅ Bad code quality: {:.3}", bad_quality);
    
    // Test security analyzer
    println!("Testing Security Analyzer...");
    let security_analyzer = SecurityAnalyzer::new();
    
    let secure_code = "def safe_function(x): return x * 2";
    let vulnerable_code = "def unsafe_function(): exec(user_input)";
    
    let secure_score = security_analyzer.analyze_security(&secure_code)?;
    let vulnerable_score = security_analyzer.analyze_security(&vulnerable_code)?;
    
    println!("✅ Secure code score: {:.3}", secure_score);
    println!("✅ Vulnerable code score: {:.3}", vulnerable_score);
    
    // Test efficiency analyzer
    println!("Testing Efficiency Analyzer...");
    let efficiency_analyzer = EfficiencyAnalyzer::new();
    
    let efficient_code = "def efficient_sum(lst): return sum(lst)";
    let inefficient_code = "def inefficient_sum(lst): total = 0; for i in range(len(lst)): total += lst[i]; return total";
    
    let efficient_score = efficiency_analyzer.analyze_efficiency(&efficient_code)?;
    let inefficient_score = efficiency_analyzer.analyze_efficiency(&inefficient_code)?;
    
    println!("✅ Efficient code score: {:.3}", efficient_score);
    println!("✅ Inefficient code score: {:.3}", inefficient_score);
    
    // Test DPO trainer
    println!("Testing DPO Trainer...");
    let model = CodeModel::new(50000, 8192);
    let reference_model = CodeModel::new(50000, 8192);
    let dpo_trainer = CodeDpoTrainer::new(config, model, reference_model);
    
    let preference_pairs = alignment::utils::create_preference_pairs(
        &vec!["Write a function to calculate sum".to_string()],
        &vec![good_code.to_string()],
        &vec![bad_code.to_string()],
    );
    
    let dpo_loss = dpo_trainer.training_step(&preference_pairs)?;
    println!("✅ DPO loss: {:.6}", dpo_loss.total_loss);
    
    let alignment_stats = dpo_trainer.get_alignment_stats(&preference_pairs)?;
    println!("✅ Alignment stats:");
    println!("  Security improvement rate: {:.2}%", alignment_stats.security_improvement_rate * 100.0);
    println!("  Efficiency improvement rate: {:.2}%", alignment_stats.efficiency_improvement_rate * 100.0);
    println!("  Quality improvement rate: {:.2}%", alignment_stats.quality_improvement_rate * 100.0);
    
    Ok(())
}

/// Demo code utilities
fn demo_code_utilities() -> anyhow::Result<()> {
    println!("\n🛠️ Code Utilities Demo");
    println!("=====================");
    
    // Test tokenizer
    println!("Testing Code Tokenizer...");
    let tokenizer = CodeTokenizer::new();
    let code = "def hello_world(): print('Hello, World!')";
    
    let tokens = tokenizer.tokenize(&code)?;
    let decoded = tokenizer.decode(&tokens)?;
    
    println!("✅ Original: {}", code);
    println!("✅ Tokens: {} tokens", tokens.len());
    println!("✅ Decoded: {}", decoded);
    
    // Test parser
    println!("Testing Code Parser...");
    let parser = CodeParser::new("python");
    let ast = parser.parse(&code)?;
    
    println!("✅ Parsed language: {}", ast.language);
    println!("✅ Functions found: {}", ast.functions.len());
    println!("✅ Classes found: {}", ast.classes.len());
    
    // Test formatter
    println!("Testing Code Formatter...");
    let formatter = CodeFormatter::new("python");
    let unformatted = "def test():print('hello')";
    let formatted = formatter.format(&unformatted)?;
    
    println!("✅ Unformatted: {}", unformatted);
    println!("✅ Formatted: {}", formatted);
    
    // Test metrics
    println!("Testing Code Metrics...");
    let metrics = CodeMetrics::new("python");
    let metrics_result = metrics.calculate_metrics(&code)?;
    
    println!("✅ Metrics:");
    println!("  Total lines: {}", metrics_result.total_lines);
    println!("  Code lines: {}", metrics_result.code_lines);
    println!("  Functions: {}", metrics_result.functions);
    println!("  Complexity: {:.2}", metrics_result.complexity);
    println!("  Maintainability: {:.2}", metrics_result.maintainability_index);
    
    Ok(())
}

/// Demo verifiers
fn demo_verifiers() -> anyhow::Result<()> {
    println!("\n🔍 Verifiers Demo");
    println!("==================");
    
    // Test individual verifiers
    println!("Testing Individual Verifiers...");
    
    let secure_code = r#"
import os
import hashlib

def secure_hash(data):
    """Securely hash data"""
    if not isinstance(data, str):
        data = str(data)
    return hashlib.sha256(data.encode()).hexdigest()

def main():
    user_input = input("Enter data to hash: ")
    if user_input.strip():
        result = secure_hash(user_input)
        print(f"Secure hash: {result}")
    else:
        print("Empty input not allowed")
"#;
    
    let vulnerable_code = r#"
import os
import subprocess

def insecure_function():
    user_input = input("Enter command: ")
    os.system(user_input)  # Security vulnerability

def main():
    insecure_function()
"#;
    
    let verifier = CodeVerifier::new();
    
    // Test secure code
    println!("Testing Secure Code:");
    let secure_score = verifier.verify_code(&secure_code)?;
    println!("✅ Overall score: {:.3}", secure_score);
    
    let secure_results = verifier.verify_detailed(&secure_code, "python")?;
    for result in &secure_results {
        println!("  {}: {:.3} ({} issues)", 
            result.verifier_name, result.score, result.issues.len());
    }
    
    // Test vulnerable code
    println!("Testing Vulnerable Code:");
    let vulnerable_score = verifier.verify_code(&vulnerable_code)?;
    println!("✅ Overall score: {:.3}", vulnerable_score);
    
    let vulnerable_results = verifier.verify_detailed(&vulnerable_code, "python")?;
    for result in &vulnerable_results {
        println!("  {}: {:.3} ({} issues)", 
            result.verifier_name, result.score, result.issues.len());
        
        // Show first few issues
        for (i, issue) in result.issues.iter().take(3).enumerate() {
            println!("    Issue {}: {} - {}", i + 1, issue.severity, issue.message);
        }
    }
    
    // Test verification analysis
    println!("Testing Verification Analysis...");
    let analysis = verifiers::utils::analyze_results(&secure_results);
    
    println!("✅ Analysis Results:");
    println!("  Overall Score: {:.3}", analysis.overall_score);
    println!("  Pass Rate: {:.2}%", analysis.pass_rate * 100.0);
    println!("  Total Issues: {}", analysis.total_issues);
    
    println!("  Recommendations:");
    for (i, rec) in analysis.recommendations.iter().take(3).enumerate() {
        println!("    {}. {}", i + 1, rec);
    }
    
    Ok(())
}
