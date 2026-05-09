//! Tests untuk STAR-X components

use crate::star_x::*;
use ndarray::Array1;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_star_x_config_default() {
        let config = StarXConfig::default();
        
        assert_eq!(config.input_size, 512);
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.output_size, 512);
        assert_eq!(config.attention_heads, 8);
        assert_eq!(config.update_threshold, 0.1);
    }
    
    #[test]
    fn test_star_x_state_creation() -> DLResult<()> {
        let config = StarXConfig::default();
        let state = StarXState::new(&config)?;
        
        assert_eq!(state.hidden_state.shape(), vec![1024]);
        assert_eq!(state.micro_state.shape(), vec![256]);
        assert_eq!(state.meso_state.shape(), vec![512]);
        assert_eq!(state.macro_state.shape(), vec![1024]);
        
        Ok(())
    }
    
    #[test]
    fn test_temporal_gating_hierarchy() -> DLResult<()> {
        let tgh = TemporalGatingHierarchy::new(512, 1024, 256, 512, 1024, 8)?;
        
        let input = Array1::zeros(512).into_dyn();
        let hidden_state = Array1::zeros(1024).into_dyn();
        let chunk_context = Array1::zeros(512).into_dyn();
        let episodic_memory = Array1::zeros(1024).into_dyn();
        
        let (micro, meso, macro_out) = tgh.process_hierarchical(
            &input, &hidden_state, &chunk_context, &episodic_memory
        )?;
        
        assert_eq!(micro.shape(), vec![256]);
        assert_eq!(meso.shape(), vec![512]);
        assert_eq!(macro_out.shape(), vec![1024]);
        
        Ok(())
    }
    
    #[test]
    fn test_sparse_causal_attention() -> DLResult<()> {
        let sca = SparseCausalAttention::new(1024, 8, 64, 0.1)?;
        
        let query = Array1::zeros(1024).into_dyn();
        let key = Array1::zeros(1024).into_dyn();
        let value = Array1::zeros(1024).into_dyn();
        let temporal_encoding = Array1::zeros(64).into_dyn();
        
        let (output, mask) = sca.compute_sparse_attention(&query, &key, &value, &temporal_encoding)?;
        
        assert_eq!(output.shape(), vec![1024]);
        assert_eq!(mask.shape(), vec![8, 64]);
        
        // Check sparsity
        let sparsity = sca.get_sparsity_ratio();
        assert!(sparsity >= 0.0 && sparsity <= 1.0);
        
        Ok(())
    }
    
    #[test]
    fn test_harmonic_temporal_encoding() -> DLResult<()> {
        let hte = HarmonicTemporalEncoding::new(64, 16, 1000)?;
        
        let encoding = hte.compute_harmonic_encoding(100)?;
        assert_eq!(encoding.shape(), vec![64]);
        
        let relative_encoding = hte.compute_relative_encoding(100, 50)?;
        assert_eq!(relative_encoding.shape(), vec![64]);
        
        let periodic_encoding = hte.compute_periodic_encoding(100, 32)?;
        assert_eq!(periodic_encoding.shape(), vec![64]);
        
        Ok(())
    }
    
    #[test]
    fn test_selective_state_update() -> DLResult<()> {
        let ssu = SelectiveStateUpdate::new(1024, 1024, 0.1, 0.5)?;
        
        let previous_state = Array1::zeros(1024).into_dyn();
        let candidate_state = Array1::ones(1024).into_dyn();
        let relevance = Array1::from_vec(vec![0.5; 1024]).into_dyn();
        
        let updated = ssu.selective_update(&previous_state, &candidate_state, &relevance, 0.1)?;
        assert_eq!(updated.shape(), vec![1024]);
        
        // Test relevance computation
        let tgh_output = Array1::zeros(1024).into_dyn();
        let sca_output = Array1::zeros(1024).into_dyn();
        let relevance_scores = ssu.compute_relevance(&tgh_output, &sca_output)?;
        assert_eq!(relevance_scores.shape(), vec![1024]);
        
        Ok(())
    }
    
    #[test]
    fn test_adaptive_gradient_resonance() -> DLResult<()> {
        let agr = AdaptiveGradientResonance::new(0.1, 0.9, 0.99)?;
        
        let current_state = Array1::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
        let previous_state = Array1::from_vec(vec![0.5, 1.0, 1.5]).into_dyn();
        
        let resonance = agr.compute_resonance(&current_state, &previous_state)?;
        assert!(resonance >= 0.0 && resonance <= 1.0);
        
        let resonated = agr.apply_resonance(&current_state, &previous_state, resonance)?;
        assert_eq!(resonated.shape(), vec![3]);
        
        Ok(())
    }
    
    #[test]
    fn test_episodic_memory_retention() -> DLResult<()> {
        let emr = EpisodicMemoryRetention::new(1000, 512, 0.7)?;
        
        let state = Array1::from_vec(vec![1.0; 512]).into_dyn();
        let gradient = Array1::from_vec(vec![0.1; 512]).into_dyn();
        
        // Test memory write
        let written = emr.write_memory(&state, &gradient, 0.8, 0.7)?;
        assert!(written); // Should be written since relevance > threshold
        
        // Test memory read
        let query = Array1::from_vec(vec![0.5; 512]).into_dyn();
        let retrieved = emr.read_memory(&query)?;
        assert_eq!(retrieved.shape(), vec![512]);
        
        // Test memory utilization
        let utilization = emr.get_memory_utilization();
        assert!(utilization >= 0.0 && utilization <= 1.0);
        
        Ok(())
    }
    
    #[test]
    fn test_adaptive_compute_allocation() -> DLResult<()> {
        let aca = AdaptiveComputeAllocation::new(512, 1024, vec![0.3, 0.6, 0.9])?;
        
        let input = Array1::zeros(512).into_dyn();
        let hidden_state = Array1::zeros(1024).into_dyn();
        
        let compute_level = aca.determine_compute_level(&input, &hidden_state)?;
        assert!(compute_level <= 2); // Should be valid level index
        
        let (efficiency, utilization, cost) = aca.get_compute_stats();
        assert!(efficiency >= 0.0 && efficiency <= 1.0);
        assert!(utilization >= 0.0);
        assert!(cost >= 0.0);
        
        Ok(())
    }
    
    #[test]
    fn test_star_x_model_creation() -> DLResult<()> {
        let model = StarXModel::default_model()?;
        
        let input = Array1::zeros(512).into_dyn();
        let output = model.forward(&input)?;
        
        assert_eq!(output.shape(), vec![1024]);
        
        // Test metrics
        let metrics = model.get_metrics();
        assert!(metrics.compute_efficiency >= 0.0 && metrics.compute_efficiency <= 1.0);
        
        Ok(())
    }
    
    #[test]
    fn test_star_x_model_sequence() -> DLResult<()> {
        let mut model = StarXModel::default_model()?;
        
        let inputs: Vec<ArrayD<f32>> = (0..10)
            .map(|_| Array1::zeros(512).into_dyn())
            .collect();
        
        let outputs = model.forward_sequence(&inputs)?;
        assert_eq!(outputs.len(), 10);
        
        for output in &outputs {
            assert_eq!(output.shape(), vec![1024]);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_star_x_specialized_models() -> DLResult<()> {
        // Test long context model
        let long_model = StarXModel::long_context_model(4096)?;
        assert_eq!(long_model.config.max_position, 4096);
        
        // Test streaming model
        let streaming_model = StarXModel::streaming_model()?;
        assert_eq!(streaming_model.config.update_threshold, 0.05);
        
        // Test multimodal model
        let multimodal_model = StarXModel::multimodal_model()?;
        assert_eq!(multimodal_model.config.harmonic_frequencies, 32);
        
        Ok(())
    }
    
    #[test]
    fn test_star_x_state_management() -> DLResult<()> {
        let mut model = StarXModel::default_model()?;
        
        // Forward pass to modify state
        let input = Array1::zeros(512).into_dyn();
        let _ = model.forward(&input);
        
        // Check state
        let state = model.get_state();
        assert_eq!(state.temporal_position, 1);
        assert!(state.hidden_state.len() > 0);
        
        // Reset state
        model.reset_state()?;
        let reset_state = model.get_state();
        assert_eq!(reset_state.temporal_position, 0);
        
        Ok(())
    }
    
    #[test]
    fn test_star_x_memory_estimation() -> DLResult<()> {
        let model = StarXModel::default_model()?;
        let memory_usage = model.estimate_memory_usage();
        
        assert!(memory_usage.total_mb > 0);
        assert!(memory_usage.model_parameters_mb > 0);
        assert!(memory_usage.state_memory_mb > 0);
        assert!(memory_usage.episodic_memory_mb > 0);
        
        Ok(())
    }
    
    #[test]
    fn test_star_x_training_optimization() -> DLResult<()> {
        let mut model = StarXModel::default_model()?;
        
        // Optimize for training
        model.optimize_for_training()?;
        assert!(model.is_training);
        
        // Optimize for inference
        model.optimize_for_inference()?;
        assert!(!model.is_training);
        
        Ok(())
    }
    
    #[test]
    fn test_component_statistics() -> DLResult<()> {
        let model = StarXModel::default_model()?;
        let stats = model.get_component_stats();
        
        // Verify all component stats are available
        assert!(stats.tgh_stats.0 >= 0); // chunk buffer size
        assert!(stats.sca_stats.0 >= 0.0); // sparsity ratio
        assert!(stats.ssu_stats.0 >= 0.0); // update frequency
        assert!(stats.agr_stats.0 >= 0.0); // resonance factor
        assert!(stats.emr_stats.0 >= 0); // current memory size
        assert!(stats.aca_stats.0 >= 0.0); // compute efficiency
        
        Ok(())
    }
}
