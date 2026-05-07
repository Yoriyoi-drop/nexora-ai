//! Tests untuk SPARO framework

use super::*;
use super::prelude::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_model() -> PolicyModel {
        PolicyModel::new(uuid::Uuid::new_v4(), (10, 10))
    }

    fn create_test_trace() -> ReasoningTrace {
        ReasoningTrace {
            id: uuid::Uuid::new_v4(),
            prompt: "Test prompt".to_string(),
            steps: vec![
                ReasoningStep {
                    id: uuid::Uuid::new_v4(),
                    content: "Step 1: Initial reasoning".to_string(),
                    step_number: 1,
                    timestamp: chrono::Utc::now(),
                },
                ReasoningStep {
                    id: uuid::Uuid::new_v4(),
                    content: "Step 2: Final reasoning".to_string(),
                    step_number: 2,
                    timestamp: chrono::Utc::now(),
                },
            ],
            final_answer: "Test answer".to_string(),
            created_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn test_sparo_config_default() {
        let config = SparoConfig::default();
        assert_eq!(config.alpha, 0.4);
        assert_eq!(config.beta, 0.3);
        assert_eq!(config.gamma, 0.3);
        assert_eq!(config.learning_rate, 1e-4);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.max_iterations, 1000);
        assert_eq!(config.convergence_threshold, 1e-6);
    }

    #[test]
    fn test_dpo_loss_calculation() {
        let model = create_test_model();
        let config = DpoConfig::default();
        let trainer = DpoTrainer::new(model, config);

        let pair = PreferencePair {
            id: uuid::Uuid::new_v4(),
            prompt: "Test prompt".to_string(),
            chosen: "Good response".to_string(),
            rejected: "Bad response".to_string(),
            chosen_logprob: -1.0,
            rejected_logprob: -2.0,
            reference_chosen_logprob: -1.1,
            reference_rejected_logprob: -2.1,
        };

        let loss = trainer.loss_calculator.calculate_loss(&pair).unwrap();
        assert!(loss > 0.0);
    }

    #[test]
    fn test_kto_prospect_theory() {
        let model = create_test_model();
        let config = KtoConfig::default();
        let trainer = KtoTrainer::new(model, config);

        let label = IndependentLabel {
            id: uuid::Uuid::new_v4(),
            prompt: "Test prompt".to_string(),
            response: "Test response".to_string(),
            is_good: true,
            confidence: 0.8,
            log_probability: -1.0,
            reference_log_probability: -1.1,
        };

        let loss = trainer.loss_calculator.calculate_loss(&label).unwrap();
        assert!(loss > 0.0);
    }

    #[test]
    fn test_ipo_regularization() {
        let model = create_test_model();
        let config = IpoConfig::default();
        let trainer = IpoTrainer::new(model, config);

        let constraint = IdentityConstraint {
            id: uuid::Uuid::new_v4(),
            prompt: "Test prompt".to_string(),
            original_response: "Original response".to_string(),
            current_response: "Current response".to_string(),
            original_logprob: -1.0,
            current_logprob: -1.2,
            reference_logprob: -1.1,
        };

        let loss = trainer.loss_calculator.calculate_loss(&constraint).unwrap();
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_rlvf_verification() {
        let config = RlvfConfig::default();
        let manager = RlvfManager::new(config);
        let trace = create_test_trace();

        let step_feedbacks = manager.verify_trace(&trace).unwrap();
        assert_eq!(step_feedbacks.len(), trace.steps.len());

        let judge_feedbacks = manager.feedback_to_judge_feedback(&step_feedbacks);
        assert!(!judge_feedbacks.is_empty());
    }

    #[test]
    fn test_spin_self_play() {
        let student = create_test_model();
        let teacher = create_test_model();
        let config = SpinConfig::default();
        let mut trainer = SpinTrainer::new(config, student, teacher);

        let prompts = vec!["Test prompt 1".to_string(), "Test prompt 2".to_string()];
        let tournament = trainer.run_tournament(&prompts).unwrap();

        assert!(!tournament.games.is_empty());
        assert!(tournament.student_win_rate >= 0.0 && tournament.student_win_rate <= 1.0);
    }

    #[test]
    fn test_rlaif_ai_feedback() {
        let config = RlaifConfig::default();
        let manager = RlaifManager::new(config);
        let trace = create_test_trace();

        let feedback = manager.generate_feedback(&trace).unwrap();
        assert!(!feedback.is_empty());

        let stats = manager.get_judge_stats();
        assert_eq!(stats.num_judges, 1);
    }

    #[test]
    fn test_sparo_trainer_creation() {
        let student = create_test_model();
        let teacher = create_test_model();
        let config = SparoConfig::default();

        let trainer = SparoTrainer::new(config, student, teacher).unwrap();
        assert_eq!(trainer.training_state.iteration, 0);
        assert!(!trainer.training_state.converged);
    }

    #[test]
    fn test_dataset_operations() {
        let config = DatasetConfig::default();
        let mut dataset = SparoDataset::new(config);

        let traces = vec![create_test_trace()];
        let feedback = vec![];

        dataset.add_data(traces.clone(), feedback).unwrap();
        let stats = dataset.get_stats();
        assert_eq!(stats.total_traces, 1);
    }

    #[test]
    fn test_data_processor() {
        let config = DatasetConfig::default();
        let processor = DataProcessor::new(config);
        let trace = create_test_trace();

        let processed = processor.preprocess_traces(&[trace]).unwrap();
        assert_eq!(processed.len(), 1);
    }

    #[test]
    fn test_training_batch() {
        let traces = vec![create_test_trace()];
        let feedback = vec![];
        let batch = TrainingBatch::new(traces, feedback, 0);

        assert!(!batch.is_empty());
        assert_eq!(batch.size(), 1);
        assert_eq!(batch.iteration, 0);
    }

    #[test]
    fn test_sparo_loss() {
        let loss = SparoLoss {
            total_loss: 1.0,
            dpo_loss: 0.4,
            kto_loss: 0.3,
            ipo_loss: 0.3,
        };

        assert_eq!(loss.total_loss, loss.dpo_loss + loss.kto_loss + loss.ipo_loss);
    }

    #[test]
    fn test_model_state() {
        let mut state = ModelState {
            iteration: 0,
            loss_history: vec![1.0, 0.8, 0.6],
            current_loss: 0.6,
            best_loss: 0.6,
            converged: false,
        };

        assert_eq!(state.iteration, 0);
        assert_eq!(state.loss_history.len(), 3);
        assert_eq!(state.current_loss, 0.6);
        assert_eq!(state.best_loss, 0.6);
        assert!(!state.converged);
    }

    #[test]
    fn test_training_metrics() {
        let loss = SparoLoss {
            total_loss: 1.0,
            dpo_loss: 0.4,
            kto_loss: 0.3,
            ipo_loss: 0.3,
        };

        let metrics = TrainingMetrics::new(0, loss);
        assert_eq!(metrics.iteration, 0);
        assert_eq!(metrics.loss.total_loss, 1.0);
    }

    #[test]
    fn test_feedback_types() {
        let step_id = uuid::Uuid::new_v4();
        let trace_id = uuid::Uuid::new_v4();

        let independent = FeedbackType::Independent {
            step_id,
            is_good: true,
            confidence: 0.8,
        };

        let preferred_id = uuid::Uuid::new_v4();
        let rejected_id = uuid::Uuid::new_v4();
        let pairwise = FeedbackType::Pairwise {
            preferred: preferred_id,
            rejected: rejected_id,
            confidence: 0.9,
        };

        match independent {
            FeedbackType::Independent { step_id: id, is_good, confidence } => {
                assert_eq!(id, step_id);
                assert!(is_good);
                assert_eq!(confidence, 0.8);
            }
            _ => panic!("Expected Independent feedback type"),
        }

        match pairwise {
            FeedbackType::Pairwise { preferred, rejected, confidence } => {
                assert_eq!(preferred, preferred_id);
                assert_eq!(rejected, rejected_id);
                assert_eq!(confidence, 0.9);
            }
            _ => panic!("Expected Pairwise feedback type"),
        }
    }

    #[test]
    fn test_reasoning_step() {
        let step = ReasoningStep {
            id: uuid::Uuid::new_v4(),
            content: "Test step content".to_string(),
            step_number: 1,
            timestamp: chrono::Utc::now(),
        };

        assert_eq!(step.step_number, 1);
        assert_eq!(step.content, "Test step content");
    }

    #[test]
    fn test_reasoning_trace() {
        let trace = create_test_trace();
        assert_eq!(trace.steps.len(), 2);
        assert_eq!(trace.prompt, "Test prompt");
        assert_eq!(trace.final_answer, "Test answer");
    }

    #[test]
    fn test_judge_feedback() {
        let feedback = JudgeFeedback {
            id: uuid::Uuid::new_v4(),
            trace_id: uuid::Uuid::new_v4(),
            feedback_type: FeedbackType::Independent {
                step_id: uuid::Uuid::new_v4(),
                is_good: true,
                confidence: 0.8,
            },
            reasoning: "Good reasoning".to_string(),
            created_at: chrono::Utc::now(),
        };

        match feedback.feedback_type {
            FeedbackType::Independent { is_good, confidence, .. } => {
                assert!(is_good);
                assert_eq!(confidence, 0.8);
            }
            _ => panic!("Expected Independent feedback type"),
        }
    }

    #[test]
    fn test_sparo_config_validation() {
        let mut config = SparoConfig::default();
        assert!(trainer::utils::validate_config(&config).is_ok());

        // Test invalid alpha
        config.alpha = -0.1;
        assert!(trainer::utils::validate_config(&config).is_err());

        config.alpha = 1.1;
        assert!(trainer::utils::validate_config(&config).is_err());

        // Reset and test weight sum
        config = SparoConfig::default();
        config.alpha = 0.5;
        config.beta = 0.5;
        config.gamma = 0.5; // Sum = 1.5, should fail
        assert!(trainer::utils::validate_config(&config).is_err());
    }

    #[test]
    fn test_dataset_validation() {
        let traces = vec![create_test_trace()];
        let feedback = vec![];

        assert!(data::utils::validate_dataset(&traces, &feedback).is_ok());
    }

    #[test]
    fn test_sample_data_generation() {
        let (traces, feedback) = data::utils::generate_sample_data(5);
        assert_eq!(traces.len(), 5);
        assert_eq!(feedback.len(), 10); // 2 steps per trace
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_sparo_workflow() {
        // Create models
        let student = PolicyModel::new(uuid::Uuid::new_v4(), (10, 10));
        let teacher = PolicyModel::new(uuid::Uuid::new_v4(), (10, 10));
        
        // Create trainer
        let config = SparoConfig::default();
        let mut trainer = SparoTrainer::new(config, student, teacher).unwrap();
        
        // Generate training prompts
        let prompts = trainer::utils::generate_training_prompts(10);
        
        // Run a few iterations (not full training for test speed)
        for _ in 0..3 {
            let traces = trainer.generate_traces(&prompts).unwrap();
            assert!(!traces.is_empty());
            
            let ai_feedback = trainer.generate_ai_feedback(&traces).unwrap();
            let verified_feedback = trainer.verify_steps(&traces).unwrap();
            
            let combined_feedback = trainer.combine_feedback(&ai_feedback, &verified_feedback);
            assert!(!combined_feedback.is_empty());
        }
        
        // Check training stats
        let stats = trainer.get_training_stats();
        assert_eq!(stats.iterations, 3);
    }

    #[test]
    fn test_component_integration() {
        let model = create_test_model();
        
        // Test DPO
        let dpo_trainer = DpoTrainer::new(model.clone(), DpoConfig::default());
        assert!(dpo_trainer.training_step(&[]).is_ok());
        
        // Test KTO
        let kto_trainer = KtoTrainer::new(model.clone(), KtoConfig::default());
        assert!(kto_trainer.training_step(&[]).is_ok());
        
        // Test IPO
        let ipo_trainer = IpoTrainer::new(model.clone(), IpoConfig::default());
        assert!(ipo_trainer.training_step().is_ok());
        
        // Test RLVF
        let rlvf_manager = RlvfManager::new(RlvfConfig::default());
        let trace = create_test_trace();
        assert!(rlvf_manager.verify_trace(&trace).is_ok());
        
        // Test SPIN
        let spin_trainer = SpinTrainer::new(
            SpinConfig::default(),
            model.clone(),
            model.clone(),
        );
        let prompts = vec!["Test prompt".to_string()];
        assert!(spin_trainer.run_tournament(&prompts).is_ok());
        
        // Test RLAIF
        let rlaif_manager = RlaifManager::new(RlaifConfig::default());
        assert!(rlaif_manager.generate_feedback(&trace).is_ok());
    }

    #[test]
    fn test_data_pipeline() {
        let config = DatasetConfig::default();
        let mut dataset = SparoDataset::new(config);
        
        // Add sample data
        let (traces, feedback) = data::utils::generate_sample_data(10);
        dataset.add_data(traces, feedback).unwrap();
        
        // Test batch operations
        let batch = dataset.get_training_batch(5).unwrap();
        assert_eq!(batch.size(), 5);
        
        let val_batch = dataset.get_validation_batch(3).unwrap();
        assert_eq!(val_batch.size(), 3);
        
        // Test augmentation
        dataset.augment_dataset().unwrap();
        let stats = dataset.get_stats();
        assert!(stats.total_traces > 10); // Should have augmented data
    }
}
