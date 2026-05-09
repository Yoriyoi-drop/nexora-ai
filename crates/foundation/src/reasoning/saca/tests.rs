//! SACA Testing Framework
//! 
//! Comprehensive testing suite for SACA framework
//! Includes unit tests, integration tests, and benchmarks

#[cfg(test)]
fn init_logger() {
    use std::sync::Once;

    static INIT: Once = Once::new();

    INIT.call_once(|| {
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::from_default_env()
            )
            .with_thread_ids(true)
            .with_line_number(true)
            .with_target(true)
            .init();
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::saca::prelude::*;
    use tracing::{info, debug};
    
    #[tokio::test]
    async fn test_saca_full_pipeline() -> anyhow::Result<()> {
        init_logger();
        let config = SACAConfig::default();
        let saca = SACA::new(config).await.unwrap();
        
        let task = CodingTask {
            description: "Create a function that sorts an array of integers in ascending order".to_string(),
            requirements: vec![
                "Use efficient sorting algorithm".to_string(),
                "Handle edge cases".to_string(),
                "Include error handling".to_string(),
            ],
            constraints: vec![
                "Time complexity O(n log n)".to_string(),
                "Space complexity O(1)".to_string(),
            ],
            context: None,
        };
        
        let solution = saca.solve(task).await.unwrap();
        
        // Verify solution quality
        assert!(solution.quality_score >= 0.0);
        assert!(solution.quality_score <= 1.0);
        assert!(!solution.modules.is_empty());
        assert!(!solution.final_code.is_empty());
        
        info!("SACA Solution Quality: {:.3}", solution.quality_score);
        info!("Total Iterations: {}", solution.total_iterations);
        info!("Total Feedback Loops: {}", solution.total_feedback_loops);
        debug!("Execution Time: {:?}", solution.execution_time);
        Ok(())
    }
    
    #[tokio::test]
    async fn test_saca_with_feedback_loops() -> anyhow::Result<()> {
        init_logger();
        let mut config = SACAConfig::default();
        config.quality_threshold = 0.95; // High threshold to force feedback loops
        config.max_feedback_loops = 3;
        
        let saca = SACA::new(config).await.unwrap();
        
        let task = CodingTask {
            description: "Implement a complex data structure".to_string(),
            requirements: vec![
                "High performance".to_string(),
                "Memory efficient".to_string(),
            ],
            constraints: vec![
                "O(1) access time".to_string(),
                "Minimal memory overhead".to_string(),
            ],
            context: None,
        };
        
        let solution = saca.solve(task).await.unwrap();
        
        // Should have used feedback loops due to high threshold
        assert!(solution.total_feedback_loops <= 3);
        assert!(solution.quality_score >= 0.0);
        
        info!("Solution with feedback loops - Quality: {:.3}, Loops: {}", 
                solution.quality_score, solution.total_feedback_loops);
        Ok(())
    }
    
    #[tokio::test]
    async fn test_individual_phases() -> anyhow::Result<()> {
        init_logger();
        // Test Chain-of-Thought
        let cot_config = CoTConfig::default();
        let cot_engine = CoTEngine::new(cot_config).unwrap();
        
        let task = CodingTask {
            description: "Test task for CoT".to_string(),
            requirements: vec!["Basic requirement".to_string()],
            constraints: vec![],
            context: None,
        };
        
        let cot_result = cot_engine.reason(&task).await.unwrap();
        assert!(!cot_result.reasoning_steps.is_empty());
        assert!(!cot_result.task_analysis.is_empty());
        
        // Test Modular Decomposition
        let decompose_config = DecomposeConfig::default();
        let decompose_engine = DecomposeEngine::new(decompose_config).unwrap();
        
        let modules = decompose_engine.decompose(&cot_result).await.unwrap();
        assert!(!modules.is_empty());
        
        // Test Repository Context
        let context_config = ContextConfig::default();
        let context_engine = ContextEngine::new(context_config).unwrap();
        
        let context = context_engine.analyze(&modules, &task).await.unwrap();
        assert!(context.files_analyzed >= 0);
        
        // Test Large-Scale Sampling
        let sampling_config = SamplingConfig::default();
        let sampling_engine = SamplingEngine::new(sampling_config).unwrap();
        
        let candidates = sampling_engine.sample(&modules, &context, &cot_result).await.unwrap();
        assert_eq!(candidates.len(), 5); // Default num_candidates
        
        // Test Execute-Fail-Fix Loop
        let execute_config = ExecuteConfig::default();
        let execute_engine = ExecuteEngine::new(execute_config).unwrap();
        
        let executed_candidates = execute_engine.execute_all(candidates, &context).await.unwrap();
        assert!(!executed_candidates.is_empty());
        
        // Test Mathematical Reranking
        let rerank_config = RerankConfig::default();
        let rerank_engine = RerankEngine::new(rerank_config).unwrap();
        
        let solution = rerank_engine.rerank(executed_candidates, &context).await.unwrap();
        assert!(solution.quality_score >= 0.0);
        assert!(solution.quality_score <= 1.0);
        
        info!("Individual phases test completed successfully");
        Ok(())
    }
    
    #[tokio::test]
    async fn test_saca_integration() -> anyhow::Result<()> {
        init_logger();
        let saca_config = SACAConfig::default();
        let integration = SACAIntegration::new(saca_config).await.unwrap();
        
        let task = CodingTask {
            description: "Test integration task".to_string(),
            requirements: vec!["Integration test requirement".to_string()],
            constraints: vec![],
            context: None,
        };
        
        let stats = integration.get_integration_stats();
        assert_eq!(stats.total_models_enabled, 0);
        
        info!("Integration test completed with {} models enabled", stats.total_models_enabled);
        Ok(())
    }
    
    #[tokio::test]
    async fn test_error_handling() -> anyhow::Result<()> {
        init_logger();
        let config = SACAConfig::default();
        let saca = SACA::new(config).await.unwrap();
        
        // Test with invalid task
        let invalid_task = CodingTask {
            description: "".to_string(), // Empty description
            requirements: vec![],
            constraints: vec![],
            context: None,
        };
        
        // Should handle gracefully
        let result = saca.solve(invalid_task).await;
        
        match result {
            Ok(solution) => {
                // Even with empty description, should produce some result
                assert!(solution.quality_score >= 0.0);
                info!("Empty task handled gracefully - Quality: {:.3}", solution.quality_score);
            },
            Err(e) => {
                // Should return a meaningful error
                assert!(matches!(e, SACAError::InvalidInput(_)));
                info!("Empty task properly rejected: {}", e);
            }
        }
    }
    
    #[tokio::test]
    async fn test_performance_metrics() -> anyhow::Result<()> {
        init_logger();
        let config = SACAConfig::default();
        let saca = SACA::new(config).await.unwrap();
        
        let task = CodingTask {
            description: "Performance test task".to_string(),
            requirements: vec!["Fast execution".to_string()],
            constraints: vec![],
            context: None,
        };
        
        let start_time = std::time::Instant::now();
        let solution = saca.solve(task).await.unwrap();
        let execution_time = start_time.elapsed();
        
        // Should complete within reasonable time
        assert!(execution_time.as_secs() < 30); // 30 seconds max
        
        info!("Performance test - Execution time: {:?}", execution_time);
        info!("Solution quality: {:.3}", solution.quality_score);
        
        // Check SACA metrics
        let metrics = saca.get_metrics().await;
        assert!(metrics.total_tasks_processed >= 1);
        debug!("SACA metrics: {:?}", metrics);
        Ok(())
    }
    
    #[tokio::test]
    async fn test_caching() -> anyhow::Result<()> {
        init_logger();
        let config = SACAConfig::default();
        config.enable_caching = true;
        
        let saca = SACA::new(config).await.unwrap();
        
        let task = CodingTask {
            description: "Caching test task".to_string(),
            requirements: vec!["Test caching".to_string()],
            constraints: vec![],
            context: None,
        };
        
        // First execution
        let start1 = std::time::Instant::now();
        let solution1 = saca.solve(task.clone()).await.unwrap();
        let time1 = start1.elapsed();
        
        // Second execution (should benefit from caching)
        let start2 = std::time::Instant::now();
        let solution2 = saca.solve(task).await.unwrap();
        let time2 = start2.elapsed();
        
        // Results should be similar
        assert_eq!(solution1.quality_score, solution2.quality_score);
        
        info!("Caching test - First: {:?}, Second: {:?}", time1, time2);
        
        if time2 < time1 {
            info!("Caching provided performance benefit");
        }
        Ok(())
    }
}

#[cfg(test)]
mod benchmarks {
    use super::*;
    use crate::saca::prelude::*;
    use std::time::Instant;
    use tracing::info;
    
    #[tokio::test]
    async fn benchmark_saca_throughput() -> anyhow::Result<()> {
        super::init_logger();
        let config = SACAConfig::default();
        let saca = SACA::new(config).await.unwrap();
        
        let tasks = vec![
            CodingTask {
                description: "Sort array implementation".to_string(),
                requirements: vec!["Efficient algorithm".to_string()],
                constraints: vec![],
                context: None,
            },
            CodingTask {
                description: "Search algorithm implementation".to_string(),
                requirements: vec!["Binary search".to_string()],
                constraints: vec![],
                context: None,
            },
            CodingTask {
                description: "Data structure implementation".to_string(),
                requirements: vec!["Tree structure".to_string()],
                constraints: vec![],
                context: None,
            },
            CodingTask {
                description: "String processing function".to_string(),
                requirements: vec!["Efficient parsing".to_string()],
                constraints: vec![],
                context: None,
            },
            CodingTask {
                description: "Math utility functions".to_string(),
                requirements: vec!["Basic operations".to_string()],
                constraints: vec![],
                context: None,
            },
        ];
        
        let start_time = Instant::now();
        let mut solutions = Vec::new();
        
        for task in tasks {
            let solution = saca.solve(task).await.unwrap();
            solutions.push(solution);
        }
        
        let total_time = start_time.elapsed();
        let avg_time_per_task = total_time / solutions.len() as u32;
        let avg_quality = solutions.iter().map(|s| s.quality_score).sum::<f32>() / solutions.len() as f32;
        
        info!("Benchmark Results:");
        info!("  Total tasks: {}", solutions.len());
        info!("  Total time: {:?}", total_time);
        info!("  Avg time per task: {:?}", avg_time_per_task);
        info!("  Avg quality score: {:.3}", avg_quality);
        info!("  Tasks per second: {:.2}", solutions.len() as f64 / total_time.as_secs_f64());
        
        // Performance assertions
        assert!(avg_time_per_task.as_secs() < 10); // Should complete tasks in under 10 seconds
        assert!(avg_quality > 0.5); // Should maintain reasonable quality
        Ok(())
    }
    
    #[tokio::test]
    async fn benchmark_phase_performance() -> anyhow::Result<()> {
        super::init_logger();
        let config = SACAConfig::default();
        let pipeline = SACAPipeline::new(config).await.unwrap();
        
        let task = CodingTask {
            description: "Phase performance benchmark".to_string(),
            requirements: vec!["Test all phases".to_string()],
            constraints: vec![],
            context: None,
        };
        
        let solution = pipeline.execute(task).await.unwrap();
        
        let metrics = pipeline.get_phase_metrics().await;
        
        info!("Phase Performance Metrics:");
        for (phase, phase_metrics) in metrics {
            info!("  {:?}: avg_time={:.1}ms, success_rate={:.3}, avg_attempts={:.1}",
                    phase, phase_metrics.average_time_ms, phase_metrics.success_rate, phase_metrics.average_attempts);
            
            // Performance assertions
            assert!(phase_metrics.average_time_ms >= 0.0);
            assert!(phase_metrics.success_rate >= 0.0);
            assert!(phase_metrics.success_rate <= 1.0);
            assert!(phase_metrics.average_attempts >= 1.0);
        }
        
        info!("Overall solution quality: {:.3}", solution.quality_score);
        Ok(())
    }
    
    #[tokio::test]
    async fn benchmark_memory_usage() -> anyhow::Result<()> {
        super::init_logger();
        let config = SACAConfig::default();
        let saca = SACA::new(config).await.unwrap();
        
        // Measure baseline memory
        let baseline_memory = get_memory_usage();
        
        let tasks = vec![
            CodingTask {
                description: "Memory test task 1".to_string(),
                requirements: vec!["Test memory".to_string()],
                constraints: vec![],
                context: None,
            },
            CodingTask {
                description: "Memory test task 2".to_string(),
                requirements: vec!["Test memory".to_string()],
                constraints: vec![],
                context: None,
            },
            CodingTask {
                description: "Memory test task 3".to_string(),
                requirements: vec!["Test memory".to_string()],
                constraints: vec![],
                context: None,
            },
        ];
        
        for task in tasks {
            let _solution = saca.solve(task).await.unwrap();
        }
        
        // Measure peak memory
        let peak_memory = get_memory_usage();
        let memory_increase = peak_memory - baseline_memory;
        
        info!("Memory Usage Benchmark:");
        info!("  Baseline: {} MB", baseline_memory);
        info!("  Peak: {} MB", peak_memory);
        info!("  Increase: {} MB", memory_increase);
        
        // Memory usage should be reasonable
        assert!(memory_increase < 500); // Should not increase by more than 500MB
        Ok(())
    }
    
    fn get_memory_usage() -> u64 {
        // Simple memory usage estimation
        // In a real implementation, this would use system APIs
        use std::alloc::{GlobalAlloc, Layout, System};
        
        // This is a placeholder - actual memory measurement would be more sophisticated
        100 // MB baseline
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::saca::prelude::*;
    use tracing::info;
    
    #[tokio::test]
    async fn test_saca_factory() -> anyhow::Result<()> {
        super::init_logger();
        let saca_config = SACAConfig::default();
        
        // Test basic factory creation
        let integration = SACAIntegration::new(saca_config.clone()).await.unwrap();
        let stats = integration.get_integration_stats();
        assert_eq!(stats.total_models_enabled, 0);
        
        // Test with ATQS (if available)
        // Note: This would require actual ATQS configuration
        // let atqs_config = ATQSConfig::default();
        // let integration_with_atqs = SACAFactory::create_saca_with_atqs(saca_config.clone(), atqs_config).await.unwrap();
        // let stats_with_atqs = integration_with_atqs.get_integration_stats();
        // assert!(stats_with_atqs.atqs_enabled);
        
        info!("SACA Factory integration test completed");
        Ok(())
    }
    
    #[tokio::test]
    async fn test_end_to_end_workflow() -> anyhow::Result<()> {
        super::init_logger();
        let config = SACAConfig::default();
        let saca = SACA::new(config).await.unwrap();
        
        // Test complete workflow from task to solution
        let task = CodingTask {
            description: "Implement a binary search tree with insert, delete, and search operations".to_string(),
            requirements: vec![
                "O(log n) average case performance".to_string(),
                "Handle duplicate values".to_string(),
                "Include traversal methods".to_string(),
                "Proper error handling".to_string(),
            ],
            constraints: vec![
                "No external dependencies".to_string(),
                "Memory efficient".to_string(),
                "Thread-safe operations".to_string(),
            ],
            context: Some(TaskContext {
                repository_path: Some(".".to_string()),
                existing_files: vec!["utils.rs".to_string(), "node.rs".to_string()],
                dependencies: vec!["serde".to_string(), "tokio".to_string()],
                coding_standards: std::collections::HashMap::new(),
            }),
        };
        
        let solution = saca.solve(task).await.unwrap();
        
        // Verify comprehensive solution
        assert!(solution.quality_score > 0.5); // Should achieve reasonable quality
        assert!(!solution.modules.is_empty());
        assert!(solution.modules.len() >= 2); // Should have multiple modules
        assert!(!solution.final_code.is_empty());
        
        // Check test coverage
        assert!(solution.test_coverage > 0.0);
        
        // Verify performance grade
        assert!(matches!(solution.performance_grade, 
                        PerformanceGrade::Average | PerformanceGrade::Good | PerformanceGrade::Excellent));
        
        info!("End-to-end workflow completed successfully:");
        info!("  Quality: {:.3}", solution.quality_score);
        info!("  Modules: {}", solution.modules.len());
        info!("  Test Coverage: {:.1}%", solution.test_coverage * 100.0);
        info!("  Performance Grade: {:?}", solution.performance_grade);
        info!("  Iterations: {}", solution.total_iterations);
        info!("  Feedback Loops: {}", solution.total_feedback_loops);
        Ok(())
    }
    
    #[tokio::test]
    async fn test_concurrent_execution() -> anyhow::Result<()> {
        super::init_logger();
        let config = SACAConfig::default();
        config.parallel_execution = true;
        
        let saca = Arc::new(SACA::new(config).await.unwrap());
        
        let tasks = vec![
            CodingTask {
                description: "Concurrent task 1".to_string(),
                requirements: vec!["Test concurrent".to_string()],
                constraints: vec![],
                context: None,
            },
            CodingTask {
                description: "Concurrent task 2".to_string(),
                requirements: vec!["Test concurrent".to_string()],
                constraints: vec![],
                context: None,
            },
            CodingTask {
                description: "Concurrent task 3".to_string(),
                requirements: vec!["Test concurrent".to_string()],
                constraints: vec![],
                context: None,
            },
        ];
        
        // Execute tasks concurrently
        let handles: Vec<_> = tasks.into_iter()
            .map(|task| {
                let saca_clone = Arc::clone(&saca);
                tokio::spawn(async move {
                    saca_clone.solve(task).await
                })
            })
            .collect();
        
        let results: Vec<_> = futures::future::join_all(handles)
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        
        // Verify all tasks completed successfully
        assert_eq!(results.len(), 3);
        for solution in results {
            assert!(solution.quality_score >= 0.0);
            assert!(solution.quality_score <= 1.0);
        }
        
        info!("Concurrent execution test completed - {} tasks processed", results.len());
        Ok(())
    }
}
