//! Integration tests for ECHO-Net Ω
//!
//! Test suite untuk memverifikasi semua komponen bekerja bersama dengan benar:
//! - Individual component tests
//! - Integration tests
//! - Performance benchmarks
//! - End-to-end pipeline tests

use crate::echo_net::*;
use ndarray::{ArrayD, Array1, Array2};
use std::time::Instant;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Default band frequencies for multi-band holographic processing
const DEFAULT_BAND_FREQUENCIES: [f32; 8] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0];

/// Default number of bands for multi-band processing
const DEFAULT_NUM_BANDS: usize = 8;

/// Default kernel size for convolution operations
const DEFAULT_KERNEL_SIZE: usize = 64;

/// Default compression levels for holographic compression
const DEFAULT_COMPRESSION_LEVELS: usize = 5;

/// Default compression ratio
const DEFAULT_COMPRESSION_RATIO: f32 = 0.5;

/// Default memory size for persistent resonance memory
const DEFAULT_MEMORY_SIZE: usize = 10000;

/// Default decay alpha for memory decay
const DEFAULT_DECAY_ALPHA: f32 = 0.01;

/// Default write threshold for memory operations
const DEFAULT_WRITE_THRESHOLD: f32 = 0.1;

/// Default reasoning steps for iterative resonance reasoner
const DEFAULT_REASONING_STEPS: usize = 3;

/// Default reasoning alpha for iterative resonance reasoner
const DEFAULT_REASONING_ALPHA: f32 = 0.1;

/// Default energy weight for resonance calculations
const DEFAULT_ENERGY_WEIGHT: f32 = 1.0;

/// Default entropy weight for resonance calculations
const DEFAULT_ENTROPY_WEIGHT: f32 = 0.5;

/// Default coherence weight for resonance calculations
const DEFAULT_COHERENCE_WEIGHT: f32 = 0.3;

/// Default top-k for routing operations
const DEFAULT_TOP_K: usize = 64;

/// Default routing threshold
const DEFAULT_ROUTING_THRESHOLD: f32 = 0.01;

/// Default phase learning rate
const DEFAULT_PHASE_LR: f32 = 0.01;

/// Default energy clip value
const DEFAULT_ENERGY_CLIP: f32 = 10.0;

/// Default streaming window size
const DEFAULT_STREAMING_WINDOW: usize = 1024;

/// Default update frequency
const DEFAULT_UPDATE_FREQUENCY: usize = 100;

/// Seed for reproducible random number generation
const TEST_RNG_SEED: u64 = 42;

/// Test array dimensions for compression tests
const TEST_ARRAY_SIZE: usize = 64;

/// Default test dimension for arrays
const TEST_DIMENSION: usize = 128;

/// Default context size for ISC tests
const TEST_CONTEXT_SIZE: usize = 1024;

/// Default top-k for retrieval tests
const TEST_TOP_K: usize = 5;

/// Success rate threshold for tests
const SUCCESS_RATE_THRESHOLD: f32 = 0.8;

/// High success rate threshold for critical tests
const HIGH_SUCCESS_RATE_THRESHOLD: f32 = 0.9;

/// Memory pressure threshold
const MEMORY_PRESSURE_THRESHOLD: f32 = 0.9;

/// Memory utilization threshold
const MEMORY_UTILIZATION_THRESHOLD: f32 = 0.8;

/// Softmax sum tolerance for validation
const SOFTMAX_TOLERANCE: f32 = 1e-6;

/// Test configuration for different scenarios
#[derive(Debug, Clone)]
pub struct TestConfig {
    /// Name of the test configuration
    pub name: String,
    /// Size of the vocabulary for token generation
    pub vocab_size: usize,
    /// Dimension of the embedding space
    pub embedding_dim: usize,
    /// Length of the input sequence
    pub sequence_length: usize,
    /// Number of test iterations to run
    pub test_iterations: usize,
}

impl TestConfig {
    pub fn small() -> Self {
        Self {
            name: "Small Test".to_string(),
            vocab_size: 1000,
            embedding_dim: 128,
            sequence_length: 32,
            test_iterations: 10,
        }
    }
    
    pub fn medium() -> Self {
        Self {
            name: "Medium Test".to_string(),
            vocab_size: 10000,
            embedding_dim: 512,
            sequence_length: 128,
            test_iterations: 5,
        }
    }
    
    pub fn large() -> Self {
        Self {
            name: "Large Test".to_string(),
            vocab_size: 50000,
            embedding_dim: 1024,
            sequence_length: 512,
            test_iterations: 3,
        }
    }
}

/// Test runner for ECHO-Net Ω
pub struct EchoNetTestRunner {
    /// Test configuration
    config: TestConfig,
    /// ECHO-Net model instance
    model: EchoNetModel,
    /// Generated test data (sequences of tokens)
    test_data: Vec<Vec<usize>>,
}

impl EchoNetTestRunner {
    /// Create new test runner
    pub fn new(config: TestConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let echo_config = EchoNetConfig {
            vocab_size: config.vocab_size,
            embedding_dim: config.embedding_dim,
            output_size: config.embedding_dim,
            amplitude_dim: config.embedding_dim / 2,
            phase_dim: config.embedding_dim / 4,
            resonance_dim: config.embedding_dim / 4,
            num_bands: DEFAULT_NUM_BANDS,
            band_frequencies: DEFAULT_BAND_FREQUENCIES.to_vec(),
            kernel_size: DEFAULT_KERNEL_SIZE,
            compression_levels: DEFAULT_COMPRESSION_LEVELS,
            compression_ratio: DEFAULT_COMPRESSION_RATIO,
            memory_size: DEFAULT_MEMORY_SIZE,
            decay_alpha: DEFAULT_DECAY_ALPHA,
            write_threshold: DEFAULT_WRITE_THRESHOLD,
            reasoning_steps: DEFAULT_REASONING_STEPS,
            reasoning_alpha: DEFAULT_REASONING_ALPHA,
            energy_weight: DEFAULT_ENERGY_WEIGHT,
            entropy_weight: DEFAULT_ENTROPY_WEIGHT,
            coherence_weight: DEFAULT_COHERENCE_WEIGHT,
            top_k: DEFAULT_TOP_K,
            routing_threshold: DEFAULT_ROUTING_THRESHOLD,
            phase_lr: DEFAULT_PHASE_LR,
            energy_clip: DEFAULT_ENERGY_CLIP,
            streaming_window: DEFAULT_STREAMING_WINDOW,
            update_frequency: DEFAULT_UPDATE_FREQUENCY,
            ..Default::default()
        };
        
        let model = EchoNetModel::new(echo_config)?;
        let test_data = Self::generate_test_data(&config);
        
        Ok(Self {
            config,
            model,
            test_data,
        })
    }
    
    /// Generate synthetic test data with reproducible RNG
    fn generate_test_data(config: &TestConfig) -> Vec<Vec<usize>> {
        let mut rng = StdRng::seed_from_u64(TEST_RNG_SEED);
        let mut test_data = Vec::new();
        
        for _ in 0..config.test_iterations {
            let mut sequence = Vec::new();
            for _ in 0..config.sequence_length {
                // Generate random tokens with seeded RNG
                let token = rng.gen_range(0..config.vocab_size);
                sequence.push(token);
            }
            test_data.push(sequence);
        }
        
        test_data
    }
    
    /// Helper function to create a wave from sequence
    fn create_wave_from_sequence(&self, sequence: &[usize]) -> Result<crate::echo_net::Wave, TestResult> {
        let positions: Vec<usize> = (0..sequence.len()).collect();
        match self.model.sse.forward(sequence, &positions) {
            Ok(wave) => Ok(wave),
            Err(e) => Err(TestResult::failed(
                format!("Failed to create wave from sequence: {}", e),
                None,
            )),
        }
    }
    
    /// Helper function to validate wave properties
    fn validate_wave(wave: &crate::echo_net::Wave) -> bool {
        let valid_amplitude = wave.amplitude.iter().all(|&x| x >= 0.0);
        let valid_phase = wave.phase.iter().all(|&x| x.is_finite());
        let valid_frequency = wave.frequency.iter().all(|&x| x >= 0.0);
        valid_amplitude && valid_phase && valid_frequency
    }
    
    /// Helper function to validate output properties
    fn validate_output(output: &[f32]) -> bool {
        let valid_output = output.len() > 0 &&
                          output.iter().all(|&x| x.is_finite() && x >= 0.0 && x <= 1.0);
        let output_sum: f32 = output.iter().sum();
        let valid_softmax = (output_sum - 1.0).abs() < SOFTMAX_TOLERANCE;
        valid_output && valid_softmax
    }
    
    /// Run all tests
    pub fn run_all_tests(&mut self) -> TestResults {
        let mut results = TestResults::new(self.config.name.clone());
        
        // Individual component tests
        results.add_result("SSE Test", self.test_sse());
        results.add_result("APSS Test", self.test_apss());
        results.add_result("MBHW Test", self.test_mbhw());
        results.add_result("RHC Test", self.test_rhc());
        results.add_result("PRM Test", self.test_prm());
        results.add_result("IRR Test", self.test_irr());
        results.add_result("DERR Test", self.test_derr());
        results.add_result("TKRR Test", self.test_tkrr());
        results.add_result("ISC Test", self.test_isc());
        
        // Integration tests
        results.add_result("Forward Pass Test", self.test_forward_pass());
        results.add_result("State Management Test", self.test_state_management());
        results.add_result("Memory Management Test", self.test_memory_management());
        
        // Performance tests
        results.add_result("Performance Benchmark", self.test_performance());
        results.add_result("Memory Efficiency Test", self.test_memory_efficiency());
        
        // End-to-end tests
        results.add_result("End-to-End Pipeline", self.test_end_to_end_pipeline());
        results.add_result("Streaming Test", self.test_streaming());
        
        results
    }
    
    /// Test Semantic Spectral Embedding
    fn test_sse(&mut self) -> TestResult {
        println!("[TEST] Memulai test SSE (Semantic Spectral Embedding)");
        let start_time = Instant::now();
        println!("[TEST] Konfigurasi: vocab_size={}, embedding_dim={}", self.config.vocab_size, self.config.embedding_dim);
        
        match self.test_data.first() {
            Some(sequence) => {
                println!("[TEST] Menghasilkan embedding untuk sequence dengan {} tokens", sequence.len());
                // Test embedding generation using helper
                match self.create_wave_from_sequence(sequence) {
                    Ok(wave) => {
                        let duration = start_time.elapsed();
                        
                        // Verify wave properties using helper
                        if Self::validate_wave(&wave) {
                            TestResult::passed(
                                format!("SSE test passed in {:?}", duration),
                                Some(vec![duration.as_millis() as f32]),
                            )
                        } else {
                            TestResult::failed(
                                "SSE generated invalid wave properties".to_string(),
                                None,
                            )
                        }
                    }
                    Err(result) => result,
                }
            }
            None => TestResult::failed("No test data available".to_string(), None),
        }
    }
    
    /// Test Adaptive Phase Separation Stabilizer
    fn test_apss(&mut self) -> TestResult {
        println!("[TEST] Memulai test APSS (Adaptive Phase Separation Stabilizer)");
        let start_time = Instant::now();
        println!("[TEST] Mempersiapkan wave untuk phase separation");
        
        match self.test_data.first() {
            Some(sequence) => {
                // Create test wave using helper
                match self.create_wave_from_sequence(sequence) {
                    Ok(mut wave) => {
                        let embeddings = match self.model.get_embeddings_for_tokens(sequence) {
                            Ok(emb) => emb,
                            Err(e) => return TestResult::failed(
                                format!("Failed to get embeddings: {}", e),
                                None,
                            ),
                        };
                        
                        println!("[TEST] Melakukan phase separation dengan embeddings");
                        // Test phase separation
                        let apss_result = self.model.apss.forward(&mut wave, &embeddings);
                        
                        let duration = start_time.elapsed();
                        
                        match apss_result {
                            Ok(()) => TestResult::passed(
                                format!("APSS test passed in {:?}", duration),
                                Some(vec![duration.as_millis() as f32]),
                            ),
                            Err(e) => TestResult::failed(
                                format!("APSS failed: {}", e),
                                None,
                            ),
                        }
                    }
                    Err(result) => result,
                }
            }
            None => TestResult::failed("No test data available".to_string(), None),
        }
    }
    
    /// Test Multi-Band Holographic Writer
    fn test_mbhw(&mut self) -> TestResult {
        println!("[TEST] Memulai test MBHW (Multi-Band Holographic Writer)");
        let start_time = Instant::now();
        println!("[TEST] Mempersiapkan wave untuk holographic writing");
        
        match self.test_data.first() {
            Some(sequence) => {
                // Create test wave using helper
                match self.create_wave_from_sequence(sequence) {
                    Ok(wave) => {
                        println!("[TEST] Melakukan holographic writing ke band 0");
                        // Test holographic writing
                        let mbhw_result = self.model.mbhw.forward(&wave, 0);
                        
                        let duration = start_time.elapsed();
                        
                        match mbhw_result {
                            Ok(memories) => {
                                // Verify memory properties
                                let valid_memories = memories.iter().all(|m| {
                                    m.len() > 0 && m.iter().all(|c| c.is_finite())
                                });
                                
                                if valid_memories {
                                    TestResult::passed(
                                        format!("MBHW test passed in {:?} ({} bands)", duration, memories.len()),
                                        Some(vec![duration.as_millis() as f32, memories.len() as f32]),
                                    )
                                } else {
                                    TestResult::failed("MBHW generated invalid memories".to_string(), None)
                                }
                            }
                            Err(e) => TestResult::failed(
                                format!("MBHW failed: {}", e),
                                None,
                            ),
                        }
                    }
                    Err(result) => result,
                }
            }
            None => TestResult::failed("No test data available".to_string(), None),
        }
    }
    
    /// Test Recursive Holographic Compression
    fn test_rhc(&mut self) -> TestResult {
        println!("[TEST] Memulai test RHC (Recursive Holographic Compression)");
        let start_time = Instant::now();
        println!("[TEST] Mempersiapkan test data: 64x64 array");
        
        // Create test data
        let test_data = Array2::from_shape_fn((TEST_ARRAY_SIZE, TEST_ARRAY_SIZE), |(i, j)| {
            ((i + j) % 256) as f32 / 256.0
        });
        
        println!("[TEST] Melakukan kompresi holographic");
        match self.model.rhc.forward(&test_data, 0) {
            Ok(compressed_levels) => {
                let duration = start_time.elapsed();
                
                // Verify compression
                let valid_compression = compressed_levels.iter().all(|level| {
                    level.len() > 0 && level.iter().all(|&x| x.is_finite())
                });
                
                // Verify compression ratio
                let original_size = test_data.len();
                let compressed_size: usize = compressed_levels.iter().map(|l| l.len()).sum();
                let compression_ratio = compressed_size as f32 / original_size as f32;
                
                if valid_compression && compression_ratio < 1.0 {
                    TestResult::passed(
                        format!("RHC test passed in {:?} (compression ratio: {:.3})", duration, compression_ratio),
                        Some(vec![duration.as_millis() as f32, compression_ratio]),
                    )
                } else {
                    TestResult::failed(
                        format!("RHC compression failed (ratio: {:.3})", compression_ratio),
                        None,
                    )
                }
            }
            Err(e) => TestResult::failed(
                format!("RHC failed: {}", e),
                None,
            ),
        }
    }
    
    /// Test Persistent Resonance Memory
    fn test_prm(&mut self) -> TestResult {
        println!("[TEST] Memulai test PRM (Persistent Resonance Memory)");
        let start_time = Instant::now();
        println!("[TEST] Mempersiapkan wave untuk memory write");
        
        match self.test_data.first() {
            Some(sequence) => {
                // Create test wave using helper
                match self.create_wave_from_sequence(sequence) {
                    Ok(wave) => {
                        println!("[TEST] Melakukan memory write ke slot");
                        // Test memory write
                        let write_result = self.model.prm.forward(&wave, 0);
                        
                        let duration = start_time.elapsed();
                        
                        match write_result {
                            Ok(memory_slot) => {
                                println!("[TEST] Melakukan memory retrieval dengan top_k={}", TEST_TOP_K);
                                // Test memory retrieval
                                let retrieval_result = self.model.prm.retrieve(&wave, TEST_TOP_K);
                                
                                match retrieval_result {
                                    Ok(retrieved) => {
                                        if retrieved.len() > 0 {
                                            TestResult::passed(
                                                format!("PRM test passed in {:?} (slot: {}, retrieved: {})",
                                                       duration, memory_slot, retrieved.len()),
                                                Some(vec![duration.as_millis() as f32, retrieved.len() as f32]),
                                            )
                                        } else {
                                            TestResult::failed("PRM retrieval returned empty results".to_string(), None)
                                        }
                                    }
                                    Err(e) => TestResult::failed(
                                        format!("PRM retrieval failed: {}", e),
                                        None,
                                    ),
                                }
                            }
                            Err(e) => TestResult::failed(
                                format!("PRM write failed: {}", e),
                                None,
                            ),
                        }
                    }
                    Err(result) => result,
                }
            }
            None => TestResult::failed("No test data available".to_string(), None),
        }
    }
    
    /// Test Iterative Resonance Reasoner
    fn test_irr(&mut self) -> TestResult {
        println!("[TEST] Memulai test IRR (Iterative Resonance Reasoner)");
        let start_time = Instant::now();
        println!("[TEST] Mempersiapkan input dan holographic memory");
        
        // Create test data
        let input = ArrayD::from(vec![0.5; TEST_DIMENSION]);
        let holographic_memory = ArrayD::from(vec![0.3; TEST_DIMENSION]);
        
        println!("[TEST] Melakukan reasoning dengan IRR");
        match self.model.irr.forward(&input, &holographic_memory) {
            Ok(output) => {
                let duration = start_time.elapsed();
                
                // Verify output properties
                let valid_output = output.len() == input.len() && 
                                  output.iter().all(|&x| x.is_finite() && x >= -1.0 && x <= 1.0);
                
                // Get reasoning metrics
                let metrics = self.model.irr.get_performance_metrics();
                
                if valid_output {
                    TestResult::passed(
                        format!("IRR test passed in {:?} (steps: {}, converged: {})", 
                               duration, metrics.reasoning_steps, metrics.is_converged),
                        Some(vec![duration.as_millis() as f32, metrics.reasoning_steps as f32]),
                    )
                } else {
                    TestResult::failed("IRR generated invalid output".to_string(), None)
                }
            }
            Err(e) => TestResult::failed(
                format!("IRR failed: {}", e),
                None,
            ),
        }
    }
    
    /// Test Dual Entropic Resonance Retrieval
    fn test_derr(&mut self) -> TestResult {
        println!("[TEST] Memulai test DERR (Dual Entropic Resonance Retrieval)");
        let start_time = Instant::now();
        println!("[TEST] Mempersiapkan wave dan candidates");
        
        match self.test_data.first() {
            Some(sequence) => {
                // Create test wave using helper
                match self.create_wave_from_sequence(sequence) {
                    Ok(wave) => {
                        // Create test candidates
                        let candidates = vec![
                            ArrayD::from(vec![0.1; TEST_DIMENSION]),
                            ArrayD::from(vec![0.2; TEST_DIMENSION]),
                            ArrayD::from(vec![0.3; TEST_DIMENSION]),
                        ];
                        
                        println!("[TEST] Melakukan retrieval dengan {} candidates", candidates.len());
                        // Test retrieval
                        let retrieval_result = self.model.derr.forward(&wave, &candidates);
                        
                        let duration = start_time.elapsed();
                        
                        match retrieval_result {
                            Ok(context) => {
                                // Verify context properties
                                let valid_context = context.len() > 0 && 
                                                   context.iter().all(|&x| x.is_finite());
                                
                                if valid_context {
                                    TestResult::passed(
                                        format!("DERR test passed in {:?} (context size: {})", duration, context.len()),
                                        Some(vec![duration.as_millis() as f32, context.len() as f32]),
                                    )
                                } else {
                                    TestResult::failed("DERR generated invalid context".to_string(), None)
                                }
                            }
                            Err(e) => TestResult::failed(
                                format!("DERR failed: {}", e),
                                None,
                            ),
                        }
                    }
                    Err(result) => result,
                }
            }
            None => TestResult::failed("No test data available".to_string(), None),
        }
    }
    
    /// Test Top-K Resonance Routing
    fn test_tkrr(&mut self) -> TestResult {
        println!("[TEST] Memulai test TKRR (Top-K Resonance Routing)");
        let start_time = Instant::now();
        println!("[TEST] Mempersiapkan wave, resonance data, dan holographic memory");
        
        match self.test_data.first() {
            Some(sequence) => {
                // Create test wave using helper
                match self.create_wave_from_sequence(sequence) {
                    Ok(wave) => {
                        // Create test data
                        let resonance_data = vec![
                            ArrayD::from(vec![0.1; TEST_DIMENSION]),
                            ArrayD::from(vec![0.2; TEST_DIMENSION]),
                            ArrayD::from(vec![0.3; TEST_DIMENSION]),
                        ];
                        let holographic_memory = vec![
                            ArrayD::from(vec![0.15; TEST_DIMENSION]),
                            ArrayD::from(vec![0.25; TEST_DIMENSION]),
                            ArrayD::from(vec![0.35; TEST_DIMENSION]),
                        ];
                        
                        println!("[TEST] Melakukan routing dengan TKRR");
                        // Test routing
                        let routing_result = self.model.tkrr.forward(&wave, &resonance_data, &holographic_memory);
                        
                        let duration = start_time.elapsed();
                        
                        match routing_result {
                            Ok(context) => {
                                // Verify context properties
                                let valid_context = context.len() > 0 && 
                                                   context.iter().all(|&x| x.is_finite());
                                
                                // Get routing statistics
                                let stats = self.model.tkrr.get_statistics();
                                
                                if valid_context {
                                    TestResult::passed(
                                        format!("TKRR test passed in {:?} (context size: {}, efficiency: {:.3})", 
                                               duration, context.len(), stats.routing_efficiency),
                                        Some(vec![duration.as_millis() as f32, stats.routing_efficiency]),
                                    )
                                } else {
                                    TestResult::failed("TKRR generated invalid context".to_string(), None)
                                }
                            }
                            Err(e) => TestResult::failed(
                                format!("TKRR failed: {}", e),
                                None,
                            ),
                        }
                    }
                    Err(result) => result,
                }
            }
            None => TestResult::failed("No test data available".to_string(), None),
        }
    }
    
    /// Test Inverse Spectral Collapse
    fn test_isc(&mut self) -> TestResult {
        println!("[TEST] Memulai test ISC (Inverse Spectral Collapse)");
        let start_time = Instant::now();
        println!("[TEST] Mempersiapkan context dengan 1024 elemen");
        
        // Create test context
        let context = ArrayD::from(vec![0.5; TEST_CONTEXT_SIZE]);
        
        println!("[TEST] Melakukan inverse spectral collapse");
        match self.model.isc.forward(&context, 0) {
            Ok(output) => {
                let duration = start_time.elapsed();
                
                // Verify output properties
                let valid_output = output.len() > 0 && 
                                  output.iter().all(|&x| x.is_finite() && x >= 0.0 && x <= 1.0);
                
                // Verify softmax properties
                let output_sum: f32 = output.iter().sum();
                let valid_softmax = (output_sum - 1.0).abs() < SOFTMAX_TOLERANCE;
                
                // Get collapse statistics
                let stats = self.model.isc.get_statistics();
                
                if valid_output && valid_softmax {
                    TestResult::passed(
                        format!("ISC test passed in {:?} (output size: {}, quality: {:.3})", 
                               duration, output.len(), stats.output_quality),
                        Some(vec![duration.as_millis() as f32, stats.output_quality]),
                    )
                } else {
                    TestResult::failed("ISC generated invalid output".to_string(), None)
                }
            }
            Err(e) => TestResult::failed(
                format!("ISC failed: {}", e),
                None,
            ),
        }
    }
    
    /// Test complete forward pass
    fn test_forward_pass(&mut self) -> TestResult {
        println!("[TEST] Memulai test forward pass lengkap");
        let start_time = Instant::now();
        
        match self.test_data.first() {
            Some(sequence) => {
                println!("[TEST] Memproses sequence dengan {} tokens", sequence.len());
                println!("[TEST] Melakukan forward pass melalui model");
                match self.model.forward(sequence) {
                    Ok(output) => {
                        let duration = start_time.elapsed();
                        
                        // Verify output properties using helper
                        if Self::validate_output(&output) {
                            TestResult::passed(
                                format!("Forward pass test passed in {:?} (output size: {})", duration, output.len()),
                                Some(vec![duration.as_millis() as f32, output.len() as f32]),
                            )
                        } else {
                            TestResult::failed("Forward pass generated invalid output".to_string(), None)
                        }
                    }
                    Err(e) => TestResult::failed(
                        format!("Forward pass failed: {}", e),
                        None,
                    ),
                }
            }
            None => TestResult::failed("No test data available".to_string(), None),
        }
    }
    
    /// Test state management
    fn test_state_management(&mut self) -> TestResult {
        println!("[TEST] Memulai test state management");
        let start_time = Instant::now();
        println!("[TEST] Menyimpan state awal");
        
        // Save initial state
        let initial_state = match self.model.save_state() {
            Ok(state) => state,
            Err(e) => return TestResult::failed(
                format!("Failed to save initial state: {}", e),
                None,
            ),
        };
        
        // Process some data
        if let Some(sequence) = self.test_data.first() {
            if let Err(e) = self.model.forward(sequence) {
                return TestResult::failed(
                    format!("Failed to process data during state test: {}", e),
                    None,
                );
            }
        }
        
        println!("[TEST] Mereset state model");
        // Reset state
        let reset_result = self.model.reset();
        
        // Check state after reset
        let reset_state = match self.model.save_state() {
            Ok(state) => state,
            Err(e) => return TestResult::failed(
                format!("Failed to save reset state: {}", e),
                None,
            ),
        };
        
        let duration = start_time.elapsed();
        
        match reset_result {
            Ok(()) => {
                // Verify state reset
                let state_reset = reset_state.temporal_position == initial_state.temporal_position;
                
                if state_reset {
                    TestResult::passed(
                        format!("State management test passed in {:?}", duration),
                        Some(vec![duration.as_millis() as f32]),
                    )
                } else {
                    TestResult::failed("State reset failed".to_string(), None)
                }
            }
            Err(e) => TestResult::failed(
                format!("State reset failed: {}", e),
                None,
            ),
        }
    }
    
    /// Test memory management
    fn test_memory_management(&mut self) -> TestResult {
        println!("[TEST] Memulai test memory management");
        let start_time = Instant::now();
        println!("[TEST] Memproses {} sequences untuk menguji memory pressure", self.test_data.len());
        
        // Process multiple sequences to test memory pressure
        let mut successful_processes = 0;
        
        for sequence in &self.test_data {
            match self.model.forward(sequence) {
                Ok(_) => successful_processes += 1,
                Err(_) => break,
            }
        }
        
        let duration = start_time.elapsed();
        let success_rate = successful_processes as f32 / self.test_data.len() as f32;
        
        // Check memory pressure
        let memory_pressure = self.model.memory_pressure;
        println!("[TEST] Memory pressure: {:.2}", memory_pressure);
        
        if success_rate >= SUCCESS_RATE_THRESHOLD && memory_pressure < MEMORY_PRESSURE_THRESHOLD {
            TestResult::passed(
                format!("Memory management test passed in {:?} (success rate: {:.2}, pressure: {:.2})", 
                       duration, success_rate, memory_pressure),
                Some(vec![duration.as_millis() as f32, success_rate, memory_pressure]),
            )
        } else {
            TestResult::failed(
                format!("Memory management test failed (success rate: {:.2}, pressure: {:.2})", 
                       success_rate, memory_pressure),
                None,
            )
        }
    }
    
    /// Test performance benchmark
    fn test_performance(&mut self) -> TestResult {
        println!("[TEST] Memulai test performance benchmark");
        let start_time = Instant::now();
        println!("[TEST] Menjalankan {} sequences untuk benchmark", self.test_data.len());
        
        let mut total_time = 0.0;
        let mut successful_runs = 0;
        
        for sequence in &self.test_data {
            let run_start = Instant::now();
            
            match self.model.forward(sequence) {
                Ok(_) => {
                    let run_duration = run_start.elapsed().as_secs_f32();
                    total_time += run_duration;
                    successful_runs += 1;
                }
                Err(_) => break,
            }
        }
        
        println!("[TEST] Successful runs: {}/{}", successful_runs, self.test_data.len());
        let total_test_time = start_time.elapsed();
        let avg_time_per_run = if successful_runs > 0 {
            total_time / successful_runs as f32
        } else {
            0.0
        };
        
        let tokens_per_second = if avg_time_per_run > 0.0 {
            self.config.sequence_length as f32 / avg_time_per_run
        } else {
            0.0
        };
        
        // Performance targets (adjust based on config size)
        let target_time_per_token = match self.config.embedding_dim {
            128 => 0.001,   // 1ms per token for small models
            512 => 0.005,   // 5ms per token for medium models
            1024 => 0.010,  // 10ms per token for large models
            _ => 0.020,
        };
        
        let performance_acceptable = avg_time_per_run <= target_time_per_token * self.config.sequence_length as f32;
        
        if performance_acceptable {
            TestResult::passed(
                format!("Performance test passed in {:?} (avg: {:.3}s, {:.1} tokens/s)", 
                       total_test_time, avg_time_per_run, tokens_per_second),
                Some(vec![avg_time_per_run, tokens_per_second]),
            )
        } else {
            TestResult::failed(
                format!("Performance test failed (avg: {:.3}s, target: {:.3}s)", 
                       avg_time_per_run, target_time_per_token * self.config.sequence_length as f32),
                None,
            )
        }
    }
    
    /// Test memory efficiency
    fn test_memory_efficiency(&mut self) -> TestResult {
        println!("[TEST] Memulai test memory efficiency");
        let start_time = Instant::now();
        println!("[TEST] Mendapatkan statistik memory awal");
        
        // Get initial memory statistics
        let initial_stats = self.model.get_component_statistics();
        
        // Process data to build up memory
        for sequence in &self.test_data {
            if let Err(e) = self.model.forward(sequence) {
                return TestResult::failed(
                    format!("Failed to process data during memory efficiency test: {}", e),
                    None,
                );
            }
        }
        
        println!("[TEST] Mendapatkan statistik memory akhir");
        // Get final memory statistics
        let final_stats = self.model.get_component_statistics();
        
        let duration = start_time.elapsed();
        
        // Check memory efficiency
        let memory_efficient = final_stats.prm_stats.memory_utilization <= MEMORY_UTILIZATION_THRESHOLD &&
                              final_stats.rhc_stats.memory_utilization.iter().all(|&x| x <= MEMORY_UTILIZATION_THRESHOLD);
        
        if memory_efficient {
            TestResult::passed(
                format!("Memory efficiency test passed in {:?} (PRM: {:.2}, RHC: {:.2})", 
                       duration, final_stats.prm_stats.memory_utilization,
                       final_stats.rhc_stats.memory_utilization.iter().sum::<f32>() / final_stats.rhc_stats.memory_utilization.len() as f32),
                Some(vec![final_stats.prm_stats.memory_utilization]),
            )
        } else {
            TestResult::failed(
                format!("Memory efficiency test failed (PRM: {:.2}, RHC: {:.2})", 
                       final_stats.prm_stats.memory_utilization,
                       final_stats.rhc_stats.memory_utilization.iter().sum::<f32>() / final_stats.rhc_stats.memory_utilization.len() as f32),
                None,
            )
        }
    }
    
    /// Test end-to-end pipeline
    fn test_end_to_end_pipeline(&mut self) -> TestResult {
        println!("[TEST] Memulai test end-to-end pipeline");
        let start_time = Instant::now();
        println!("[TEST] Memproses {} sequences melalui pipeline lengkap", self.test_data.len());
        
        // Test complete pipeline with multiple sequences
        let mut successful_sequences = 0;
        let mut total_output_tokens = 0;
        
        for sequence in &self.test_data {
            match self.model.forward(sequence) {
                Ok(output) => {
                    successful_sequences += 1;
                    total_output_tokens += output.len();
                }
                Err(_) => break,
            }
        }
        
        println!("[TEST] Successful sequences: {}/{}", successful_sequences, self.test_data.len());
        let duration = start_time.elapsed();
        let success_rate = successful_sequences as f32 / self.test_data.len() as f32;
        
        if success_rate >= HIGH_SUCCESS_RATE_THRESHOLD {
            TestResult::passed(
                format!("End-to-end pipeline test passed in {:?} (success rate: {:.2}, total tokens: {})", 
                       duration, success_rate, total_output_tokens),
                Some(vec![duration.as_millis() as f32, success_rate, total_output_tokens as f32]),
            )
        } else {
            TestResult::failed(
                format!("End-to-end pipeline test failed (success rate: {:.2})", success_rate),
                None,
            )
        }
    }
    
    /// Test streaming capabilities
    fn test_streaming(&mut self) -> TestResult {
        println!("[TEST] Memulai test streaming capabilities");
        let start_time = Instant::now();
        
        // Test streaming with overlapping windows
        let window_size = self.config.sequence_length / 2;
        println!("[TEST] Window size: {}", window_size);
        let mut successful_windows = 0;
        
        for i in (0..self.test_data.len()).step_by(window_size) {
            if i + window_size <= self.test_data.len() {
                let window: Vec<usize> = self.test_data[i..i + window_size]
                    .iter()
                    .flatten()
                    .copied()
                    .collect();
                
                match self.model.forward(&window) {
                    Ok(_) => successful_windows += 1,
                    Err(_) => break,
                }
            }
        }
        
        println!("[TEST] Successful windows: {}", successful_windows);
        let duration = start_time.elapsed();
        let streaming_success_rate = successful_windows as f32 / ((self.test_data.len() / window_size) as f32);
        
        if streaming_success_rate >= SUCCESS_RATE_THRESHOLD {
            TestResult::passed(
                format!("Streaming test passed in {:?} (success rate: {:.2})", 
                       duration, streaming_success_rate),
                Some(vec![duration.as_millis() as f32, streaming_success_rate]),
            )
        } else {
            TestResult::failed(
                format!("Streaming test failed (success rate: {:.2})", streaming_success_rate),
                None,
            )
        }
    }
}

/// Test result
#[derive(Debug, Clone)]
pub struct TestResult {
    pub passed: bool,
    pub message: String,
    pub metrics: Option<Vec<f32>>,
}

impl TestResult {
    pub fn passed(message: String, metrics: Option<Vec<f32>>) -> Self {
        Self {
            passed: true,
            message,
            metrics,
        }
    }
    
    pub fn failed(message: String, metrics: Option<Vec<f32>>) -> Self {
        Self {
            passed: false,
            message,
            metrics,
        }
    }
}

/// Test results collection
#[derive(Debug, Clone)]
pub struct TestResults {
    pub test_name: String,
    pub results: Vec<(String, TestResult)>,
    pub total_passed: usize,
    pub total_failed: usize,
}

impl TestResults {
    pub fn new(test_name: String) -> Self {
        Self {
            test_name,
            results: Vec::new(),
            total_passed: 0,
            total_failed: 0,
        }
    }
    
    pub fn add_result(&mut self, test_name: &str, result: TestResult) {
        if result.passed {
            self.total_passed += 1;
        } else {
            self.total_failed += 1;
        }
        
        self.results.push((test_name.to_string(), result));
    }
    
    pub fn print_summary(&self) {
        println!("\n=== {} Test Results ===", self.test_name);
        println!("Total Tests: {}", self.results.len());
        println!("Passed: {}", self.total_passed);
        println!("Failed: {}", self.total_failed);
        println!("Success Rate: {:.1}%", 
                (self.total_passed as f32 / self.results.len() as f32) * 100.0);
        println!();
        
        for (test_name, result) in &self.results {
            let status = if result.passed { "✓ PASS" } else { "✗ FAIL" };
            println!("{}: {}", status, test_name);
            if !result.passed {
                println!("  Error: {}", result.message);
            }
        }
        println!();
    }
    
    pub fn all_passed(&self) -> bool {
        self.total_failed == 0
    }
}

/// Run all test suites
pub fn run_all_test_suites() -> Vec<TestResults> {
    let mut all_results = Vec::new();
    
    // Test different configurations
    let configs = vec![
        TestConfig::small(),
        TestConfig::medium(),
        TestConfig::large(),
    ];
    
    for config in configs {
        println!("Running {} test suite...", config.name);
        
        match EchoNetTestRunner::new(config) {
            Ok(mut runner) => {
                let results = runner.run_all_tests();
                results.print_summary();
                all_results.push(results);
            }
            Err(e) => {
                println!("Failed to create test runner for {}: {}", config.name, e);
            }
        }
    }
    
    all_results
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_small_config() {
        println!("[TEST] Memulai test_small_config");
        let config = TestConfig::small();
        println!("[TEST] vocab_size: {}, embedding_dim: {}, sequence_length: {}", config.vocab_size, config.embedding_dim, config.sequence_length);
        assert_eq!(config.vocab_size, 1000);
        assert_eq!(config.embedding_dim, 128);
        assert_eq!(config.sequence_length, 32);
        assert_eq!(config.test_iterations, 10);
        assert_eq!(config.name, "Small Test");
    }
    
    #[test]
    fn test_medium_config() {
        println!("[TEST] Memulai test_medium_config");
        let config = TestConfig::medium();
        println!("[TEST] vocab_size: {}, embedding_dim: {}, sequence_length: {}", config.vocab_size, config.embedding_dim, config.sequence_length);
        assert_eq!(config.vocab_size, 10000);
        assert_eq!(config.embedding_dim, 512);
        assert_eq!(config.sequence_length, 128);
        assert_eq!(config.test_iterations, 5);
        assert_eq!(config.name, "Medium Test");
    }
    
    #[test]
    fn test_large_config() {
        println!("[TEST] Memulai test_large_config");
        let config = TestConfig::large();
        println!("[TEST] vocab_size: {}, embedding_dim: {}, sequence_length: {}", config.vocab_size, config.embedding_dim, config.sequence_length);
        assert_eq!(config.vocab_size, 50000);
        assert_eq!(config.embedding_dim, 1024);
        assert_eq!(config.sequence_length, 512);
        assert_eq!(config.test_iterations, 3);
        assert_eq!(config.name, "Large Test");
    }
    
    #[test]
    fn test_test_runner_creation() {
        println!("[TEST] Memulai test_test_runner_creation");
        let config = TestConfig::small();
        println!("[TEST] Membuat test runner dengan config small");
        let runner = EchoNetTestRunner::new(config);
        assert!(runner.is_ok());
        
        let runner = runner.unwrap();
        assert_eq!(runner.config.vocab_size, 1000);
        assert_eq!(runner.test_data.len(), 10);
        assert!(runner.test_data.iter().all(|seq| seq.len() == 32));
    }
    
    #[test]
    fn test_test_results() {
        println!("[TEST] Memulai test_test_results");
        let mut results = TestResults::new("Test Suite".to_string());
        println!("[TEST] Menambahkan test results");
        
        results.add_result("Test 1", TestResult::passed("Test 1 passed".to_string(), None));
        results.add_result("Test 2", TestResult::failed("Test 2 failed".to_string(), None));
        
        assert_eq!(results.total_passed, 1);
        assert_eq!(results.total_failed, 1);
        assert_eq!(results.results.len(), 2);
        assert!(!results.all_passed());
    }
    
    #[test]
    fn test_test_result_with_metrics() {
        println!("[TEST] Memulai test_test_result_with_metrics");
        let result = TestResult::passed("Test passed".to_string(), Some(vec![1.0, 2.0, 3.0]));
        assert!(result.passed);
        assert!(result.metrics.is_some());
        
        let metrics = result.metrics.unwrap();
        assert_eq!(metrics.len(), 3);
        assert_eq!(metrics[0], 1.0);
        assert_eq!(metrics[1], 2.0);
        assert_eq!(metrics[2], 3.0);
    }
    
    #[test]
    fn test_test_results_all_passed() {
        println!("[TEST] Memulai test_test_results_all_passed");
        let mut results = TestResults::new("All Passed Suite".to_string());
        
        results.add_result("Test 1", TestResult::passed("Test 1 passed".to_string(), None));
        results.add_result("Test 2", TestResult::passed("Test 2 passed".to_string(), None));
        
        assert_eq!(results.total_passed, 2);
        assert_eq!(results.total_failed, 0);
        assert!(results.all_passed());
    }
    
    #[test]
    fn test_config_clone() {
        println!("[TEST] Memulai test_config_clone");
        let config = TestConfig::small();
        let cloned_config = config.clone();
        
        assert_eq!(config.name, cloned_config.name);
        assert_eq!(config.vocab_size, cloned_config.vocab_size);
        assert_eq!(config.embedding_dim, cloned_config.embedding_dim);
        assert_eq!(config.sequence_length, cloned_config.sequence_length);
        assert_eq!(config.test_iterations, cloned_config.test_iterations);
    }
    
    #[test]
    fn test_constants_values() {
        println!("[TEST] Memulai test_constants_values");
        assert_eq!(TEST_RNG_SEED, 42);
        assert_eq!(TEST_ARRAY_SIZE, 64);
        assert_eq!(TEST_DIMENSION, 128);
        assert_eq!(TEST_CONTEXT_SIZE, 1024);
        assert_eq!(TEST_TOP_K, 5);
        assert_eq!(SUCCESS_RATE_THRESHOLD, 0.8);
        assert_eq!(HIGH_SUCCESS_RATE_THRESHOLD, 0.9);
        assert_eq!(MEMORY_PRESSURE_THRESHOLD, 0.9);
        assert_eq!(MEMORY_UTILIZATION_THRESHOLD, 0.8);
        assert_eq!(SOFTMAX_TOLERANCE, 1e-6);
    }
    
    #[test]
    fn test_test_result_failed() {
        println!("[TEST] Memulai test_test_result_failed");
        let result = TestResult::failed("Test failed".to_string(), Some(vec![0.5, 1.5]));
        assert!(!result.passed);
        assert!(result.metrics.is_some());
        assert_eq!(result.message, "Test failed");
        
        let metrics = result.metrics.unwrap();
        assert_eq!(metrics.len(), 2);
    }
}
