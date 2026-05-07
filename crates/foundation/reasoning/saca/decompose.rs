//! Modular Decomposition Engine
//! 
//! Phase 2 of SACA: Break down complex problems into independent modules
//! Implements CodeChain methodology with clear I/O contracts

use crate::saca::{types::*, config::*, error::*};
use nexora_core::async_executor::AsyncTaskExecutor;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Modular Decomposition engine
pub struct DecomposeEngine {
    config: DecomposeConfig,
    executor: Arc<AsyncTaskExecutor>,
    decomposition_cache: Arc<RwLock<std::collections::HashMap<String, Vec<Module>>>>,
}

impl DecomposeEngine {
    /// Create new Decompose engine
    pub fn new(config: DecomposeConfig) -> SACAResult<Self> {
        let executor = Arc::new(AsyncTaskExecutor::new(nexora_core::async_executor::ExecutorConfig::default()));
        
        info!("Decompose Engine initialized with max {} modules", config.max_modules);
        
        Ok(Self {
            config,
            executor,
            decomposition_cache: Arc::new(RwLock::new(std::collections::HashMap::new())),
        })
    }
    
    /// Decompose CoT result into independent modules
    pub async fn decompose(&self, cot_result: &CoTResult) -> SACAResult<Vec<Module>> {
        debug!("Starting modular decomposition for task analysis");
        
        // Check cache first
        let cache_key = self.generate_cache_key(cot_result);
        if let Some(cached_modules) = self.decomposition_cache.read().await.get(&cache_key) {
            debug!("Using cached decomposition result");
            return Ok(cached_modules.clone());
        }
        
        // Perform decomposition
        let modules = self.perform_decomposition(cot_result).await?;
        
        // Validate decomposition
        self.validate_decomposition(&modules).await?;
        
        // Cache the result
        self.decomposition_cache.write().await.insert(cache_key, modules.clone());
        
        info!("Decomposition completed: {} modules created", modules.len());
        Ok(modules)
    }
    
    /// Core decomposition implementation
    async fn perform_decomposition(&self, cot_result: &CoTResult) -> SACAResult<Vec<Module>> {
        let mut modules = Vec::new();
        
        // Analyze task complexity and determine decomposition strategy
        let strategy = self.determine_decomposition_strategy(cot_result).await?;
        debug!("Using decomposition strategy: {:?}", strategy);
        
        match strategy {
            DecompositionStrategy::Functional => {
                modules.extend(self.functional_decomposition(cot_result).await?);
            },
            DecompositionStrategy::Layered => {
                modules.extend(self.layered_decomposition(cot_result).await?);
            },
            DecompositionStrategy::DataDriven => {
                modules.extend(self.data_driven_decomposition(cot_result).await?);
            },
            DecompositionStrategy::Pipeline => {
                modules.extend(self.pipeline_decomposition(cot_result).await?);
            },
        }
        
        // Apply size constraints
        modules = self.apply_size_constraints(modules).await?;
        
        // Generate I/O specifications if enabled
        if self.config.interface_specification {
            modules = self.generate_io_specifications(modules).await?;
        }
        
        // Estimate complexity if enabled
        if self.config.complexity_estimation {
            modules = self.estimate_complexity(modules).await?;
        }
        
        // Analyze dependencies if enabled
        if self.config.dependency_analysis {
            modules = self.analyze_dependencies(modules).await?;
        }
        
        Ok(modules)
    }
    
    /// Determine best decomposition strategy based on task analysis
    async fn determine_decomposition_strategy(&self, cot_result: &CoTResult) -> SACAResult<DecompositionStrategy> {
        let task_desc = cot_result.task_analysis.to_lowercase();
        
        if task_desc.contains("pipeline") || task_desc.contains("flow") || task_desc.contains("process") {
            Ok(DecompositionStrategy::Pipeline)
        } else if task_desc.contains("data") || task_desc.contains("database") || task_desc.contains("storage") {
            Ok(DecompositionStrategy::DataDriven)
        } else if task_desc.contains("layer") || task_desc.contains("tier") || task_desc.contains("architecture") {
            Ok(DecompositionStrategy::Layered)
        } else {
            Ok(DecompositionStrategy::Functional)
        }
    }
    
    /// Functional decomposition approach
    async fn functional_decomposition(&self, cot_result: &CoTResult) -> SACAResult<Vec<Module>> {
        let mut modules = Vec::new();
        
        // Core logic module
        modules.push(Module {
            id: Uuid::new_v4().to_string(),
            name: "CoreLogic".to_string(),
            description: "Main business logic and algorithm implementation".to_string(),
            inputs: vec![
                ModuleIO {
                    name: "input_data".to_string(),
                    data_type: "Vec<T>".to_string(),
                    description: "Primary input data".to_string(),
                    optional: false,
                },
                ModuleIO {
                    name: "parameters".to_string(),
                    data_type: "Config".to_string(),
                    description: "Configuration parameters".to_string(),
                    optional: true,
                },
            ],
            outputs: vec![
                ModuleIO {
                    name: "result".to_string(),
                    data_type: "Result<T>".to_string(),
                    description: "Processed result".to_string(),
                    optional: false,
                },
            ],
            dependencies: vec![],
            complexity: ModuleComplexity::High,
            estimated_lines: 150,
        });
        
        // Input validation module
        modules.push(Module {
            id: Uuid::new_v4().to_string(),
            name: "InputValidator".to_string(),
            description: "Validates input data and parameters".to_string(),
            inputs: vec![
                ModuleIO {
                    name: "raw_input".to_string(),
                    data_type: "RawInput".to_string(),
                    description: "Unvalidated input".to_string(),
                    optional: false,
                },
            ],
            outputs: vec![
                ModuleIO {
                    name: "validated_input".to_string(),
                    data_type: "ValidatedInput".to_string(),
                    description: "Validated input data".to_string(),
                    optional: false,
                },
                ModuleIO {
                    name: "validation_errors".to_string(),
                    data_type: "Vec<Error>".to_string(),
                    description: "List of validation errors".to_string(),
                    optional: true,
                },
            ],
            dependencies: vec![],
            complexity: ModuleComplexity::Low,
            estimated_lines: 50,
        });
        
        // Error handling module
        modules.push(Module {
            id: Uuid::new_v4().to_string(),
            name: "ErrorHandler".to_string(),
            description: "Centralized error handling and recovery".to_string(),
            inputs: vec![
                ModuleIO {
                    name: "error".to_string(),
                    data_type: "Error".to_string(),
                    description: "Error to handle".to_string(),
                    optional: false,
                },
                ModuleIO {
                    name: "context".to_string(),
                    data_type: "Context".to_string(),
                    description: "Error context".to_string(),
                    optional: true,
                },
            ],
            outputs: vec![
                ModuleIO {
                    name: "handled_result".to_string(),
                    data_type: "Result<T>".to_string(),
                    description: "Error handling result".to_string(),
                    optional: false,
                },
            ],
            dependencies: vec![],
            complexity: ModuleComplexity::Medium,
            estimated_lines: 80,
        });
        
        // Output formatter module
        modules.push(Module {
            id: Uuid::new_v4().to_string(),
            name: "OutputFormatter".to_string(),
            description: "Formats and prepares final output".to_string(),
            inputs: vec![
                ModuleIO {
                    name: "internal_result".to_string(),
                    data_type: "InternalResult".to_string(),
                    description: "Internal processing result".to_string(),
                    optional: false,
                },
                ModuleIO {
                    name: "format_options".to_string(),
                    data_type: "FormatOptions".to_string(),
                    description: "Output formatting options".to_string(),
                    optional: true,
                },
            ],
            outputs: vec![
                ModuleIO {
                    name: "final_output".to_string(),
                    data_type: "Output".to_string(),
                    description: "Formatted output".to_string(),
                    optional: false,
                },
            ],
            dependencies: vec![],
            complexity: ModuleComplexity::Low,
            estimated_lines: 40,
        });
        
        Ok(modules)
    }
    
    /// Layered decomposition approach
    async fn layered_decomposition(&self, cot_result: &CoTResult) -> SACAResult<Vec<Module>> {
        let mut modules = Vec::new();
        
        // Presentation layer
        modules.push(Module {
            id: Uuid::new_v4().to_string(),
            name: "PresentationLayer".to_string(),
            description: "User interface and API endpoints".to_string(),
            inputs: vec![
                ModuleIO {
                    name: "user_request".to_string(),
                    data_type: "Request".to_string(),
                    description: "User request data".to_string(),
                    optional: false,
                },
            ],
            outputs: vec![
                ModuleIO {
                    name: "response".to_string(),
                    data_type: "Response".to_string(),
                    description: "Response to user".to_string(),
                    optional: false,
                },
            ],
            dependencies: vec!["BusinessLayer".to_string()],
            complexity: ModuleComplexity::Medium,
            estimated_lines: 100,
        });
        
        // Business layer
        modules.push(Module {
            id: Uuid::new_v4().to_string(),
            name: "BusinessLayer".to_string(),
            description: "Business logic and rules".to_string(),
            inputs: vec![
                ModuleIO {
                    name: "processed_request".to_string(),
                    data_type: "ProcessedRequest".to_string(),
                    description: "Request from presentation layer".to_string(),
                    optional: false,
                },
            ],
            outputs: vec![
                ModuleIO {
                    name: "business_result".to_string(),
                    data_type: "BusinessResult".to_string(),
                    description: "Business logic result".to_string(),
                    optional: false,
                },
            ],
            dependencies: vec!["DataLayer".to_string()],
            complexity: ModuleComplexity::High,
            estimated_lines: 200,
        });
        
        // Data layer
        modules.push(Module {
            id: Uuid::new_v4().to_string(),
            name: "DataLayer".to_string(),
            description: "Data access and persistence".to_string(),
            inputs: vec![
                ModuleIO {
                    name: "data_request".to_string(),
                    data_type: "DataRequest".to_string(),
                    description: "Data access request".to_string(),
                    optional: false,
                },
            ],
            outputs: vec![
                ModuleIO {
                    name: "data_result".to_string(),
                    data_type: "DataResult".to_string(),
                    description: "Data access result".to_string(),
                    optional: false,
                },
            ],
            dependencies: vec![],
            complexity: ModuleComplexity::Medium,
            estimated_lines: 120,
        });
        
        Ok(modules)
    }
    
    /// Data-driven decomposition approach
    async fn data_driven_decomposition(&self, cot_result: &CoTResult) -> SACAResult<Vec<Module>> {
        let mut modules = Vec::new();
        
        // Data model
        modules.push(Module {
            id: Uuid::new_v4().to_string(),
            name: "DataModel".to_string(),
            description: "Data structures and models".to_string(),
            inputs: vec![],
            outputs: vec![
                ModuleIO {
                    name: "model_definitions".to_string(),
                    data_type: "ModelDefs".to_string(),
                    description: "Data model definitions".to_string(),
                    optional: false,
                },
            ],
            dependencies: vec![],
            complexity: ModuleComplexity::Low,
            estimated_lines: 80,
        });
        
        // Data processor
        modules.push(Module {
            id: Uuid::new_v4().to_string(),
            name: "DataProcessor".to_string(),
            description: "Data transformation and processing".to_string(),
            inputs: vec![
                ModuleIO {
                    name: "raw_data".to_string(),
                    data_type: "RawData".to_string(),
                    description: "Raw input data".to_string(),
                    optional: false,
                },
            ],
            outputs: vec![
                ModuleIO {
                    name: "processed_data".to_string(),
                    data_type: "ProcessedData".to_string(),
                    description: "Processed data".to_string(),
                    optional: false,
                },
            ],
            dependencies: vec!["DataModel".to_string()],
            complexity: ModuleComplexity::High,
            estimated_lines: 180,
        });
        
        // Data validator
        modules.push(Module {
            id: Uuid::new_v4().to_string(),
            name: "DataValidator".to_string(),
            description: "Data validation and quality checks".to_string(),
            inputs: vec![
                ModuleIO {
                    name: "data_to_validate".to_string(),
                    data_type: "Data".to_string(),
                    description: "Data to validate".to_string(),
                    optional: false,
                },
            ],
            outputs: vec![
                ModuleIO {
                    name: "validation_report".to_string(),
                    data_type: "ValidationReport".to_string(),
                    description: "Validation results".to_string(),
                    optional: false,
                },
            ],
            dependencies: vec!["DataModel".to_string()],
            complexity: ModuleComplexity::Medium,
            estimated_lines: 90,
        });
        
        Ok(modules)
    }
    
    /// Pipeline decomposition approach
    async fn pipeline_decomposition(&self, cot_result: &CoTResult) -> SACAResult<Vec<Module>> {
        let mut modules = Vec::new();
        
        // Pipeline coordinator
        modules.push(Module {
            id: Uuid::new_v4().to_string(),
            name: "PipelineCoordinator".to_string(),
            description: "Orchestrates the entire pipeline".to_string(),
            inputs: vec![
                ModuleIO {
                    name: "pipeline_input".to_string(),
                    data_type: "PipelineInput".to_string(),
                    description: "Input to pipeline".to_string(),
                    optional: false,
                },
            ],
            outputs: vec![
                ModuleIO {
                    name: "pipeline_output".to_string(),
                    data_type: "PipelineOutput".to_string(),
                    description: "Final pipeline output".to_string(),
                    optional: false,
                },
            ],
            dependencies: vec!["Stage1".to_string(), "Stage2".to_string(), "Stage3".to_string()],
            complexity: ModuleComplexity::Medium,
            estimated_lines: 110,
        });
        
        // Pipeline stages
        for i in 1..=3 {
            modules.push(Module {
                id: Uuid::new_v4().to_string(),
                name: format!("Stage{}", i),
                description: format!("Pipeline stage {}", i),
                inputs: vec![
                    ModuleIO {
                        name: format!("stage{}_input", i),
                        data_type: format!("Stage{}Input", i),
                        description: format!("Input for stage {}", i),
                        optional: false,
                    },
                ],
                outputs: vec![
                    ModuleIO {
                        name: format!("stage{}_output", i),
                        data_type: format!("Stage{}Output", i),
                        description: format!("Output from stage {}", i),
                        optional: false,
                    },
                ],
                dependencies: vec![],
                complexity: ModuleComplexity::Medium,
                estimated_lines: 70,
            });
        }
        
        Ok(modules)
    }
    
    /// Apply size constraints to modules
    async fn apply_size_constraints(&self, mut modules: Vec<Module>) -> SACAResult<Vec<Module>> {
        // Filter modules that are too small
        modules.retain(|m| m.estimated_lines >= self.config.min_module_size);
        
        // Split modules that are too large
        let mut result = Vec::new();
        for module in modules {
            if module.estimated_lines <= self.config.max_module_size {
                result.push(module);
            } else {
                // Split large module
                let split_modules = self.split_large_module(module).await?;
                result.extend(split_modules);
            }
        }
        
        // Limit total number of modules
        if result.len() > self.config.max_modules as usize {
            result.truncate(self.config.max_modules as usize);
            warn!("Module count truncated to configured maximum of {}", self.config.max_modules);
        }
        
        Ok(result)
    }
    
    /// Split a large module into smaller ones
    async fn split_large_module(&self, module: Module) -> SACAResult<Vec<Module>> {
        let mut split_modules = Vec::new();
        let num_splits = (module.estimated_lines as f32 / self.config.max_module_size as f32).ceil() as u32;
        
        for i in 0..num_splits {
            split_modules.push(Module {
                id: Uuid::new_v4().to_string(),
                name: format!("{}_part{}", module.name, i + 1),
                description: format!("Part {} of {}", i + 1, module.description),
                inputs: module.inputs.clone(),
                outputs: module.outputs.clone(),
                dependencies: module.dependencies.clone(),
                complexity: module.complexity.clone(),
                estimated_lines: module.estimated_lines / num_splits,
            });
        }
        
        Ok(split_modules)
    }
    
    /// Generate I/O specifications for modules
    async fn generate_io_specifications(&self, mut modules: Vec<Module>) -> SACAResult<Vec<Module>> {
        for module in &mut modules {
            // Ensure each module has proper I/O specifications
            if module.inputs.is_empty() {
                module.inputs.push(ModuleIO {
                    name: "input".to_string(),
                    data_type: "Input".to_string(),
                    description: "Default input".to_string(),
                    optional: false,
                });
            }
            
            if module.outputs.is_empty() {
                module.outputs.push(ModuleIO {
                    name: "output".to_string(),
                    data_type: "Output".to_string(),
                    description: "Default output".to_string(),
                    optional: false,
                });
            }
        }
        
        Ok(modules)
    }
    
    /// Estimate complexity for modules
    async fn estimate_complexity(&self, mut modules: Vec<Module>) -> SACAResult<Vec<Module>> {
        for module in &mut modules {
            module.complexity = self.calculate_module_complexity(module).await?;
        }
        
        Ok(modules)
    }
    
    /// Calculate complexity for a single module
    async fn calculate_module_complexity(&self, module: &Module) -> SACAResult<ModuleComplexity> {
        let complexity_score = module.estimated_lines as f32 / 50.0; // 50 lines = medium complexity
        let dependency_factor = module.dependencies.len() as f32 * 0.2;
        let io_factor = (module.inputs.len() + module.outputs.len()) as f32 * 0.1;
        
        let total_score = complexity_score + dependency_factor + io_factor;
        
        Ok(if total_score < 1.0 {
            ModuleComplexity::Low
        } else if total_score < 3.0 {
            ModuleComplexity::Medium
        } else if total_score < 5.0 {
            ModuleComplexity::High
        } else {
            ModuleComplexity::Critical
        })
    }
    
    /// Analyze dependencies between modules
    async fn analyze_dependencies(&self, mut modules: Vec<Module>) -> SACAResult<Vec<Module>> {
        // Create a map of module names to IDs
        let module_map: std::collections::HashMap<String, String> = modules
            .iter()
            .map(|m| (m.name.clone(), m.id.clone()))
            .collect();
        
        // Update dependency references to use IDs
        for module in &mut modules {
            module.dependencies = module
                .dependencies
                .iter()
                .filter_map(|dep_name| module_map.get(dep_name).cloned())
                .collect();
        }
        
        Ok(modules)
    }
    
    /// Validate decomposition quality
    async fn validate_decomposition(&self, modules: &[Module]) -> SACAResult<()> {
        if modules.is_empty() {
            return Err(SACAError::DecomposeError("No modules generated".to_string()));
        }
        
        // Check for duplicate names
        let mut names = std::collections::HashSet::new();
        for module in modules {
            if !names.insert(&module.name) {
                return Err(SACAError::DecomposeError(
                    format!("Duplicate module name: {}", module.name)
                ));
            }
        }
        
        // Check dependency validity
        let module_ids: std::collections::HashSet<String> = modules
            .iter()
            .map(|m| &m.id)
            .cloned()
            .collect();
        
        for module in modules {
            for dep in &module.dependencies {
                if !module_ids.contains(dep) {
                    return Err(SACAError::DecomposeError(
                        format!("Invalid dependency in module {}: {}", module.name, dep)
                    ));
                }
            }
        }
        
        Ok(())
    }
    
    /// Generate cache key for decomposition results
    fn generate_cache_key(&self, cot_result: &CoTResult) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        cot_result.task_analysis.hash(&mut hasher);
        cot_result.approach.hash(&mut hasher);
        format!("decompose_{:x}", hasher.finish())
    }
    
    /// Clear decomposition cache
    pub async fn clear_cache(&self) {
        self.decomposition_cache.write().await.clear();
        info!("Decomposition cache cleared");
    }
}

/// Decomposition strategies
#[derive(Debug, Clone)]
enum DecompositionStrategy {
    Functional,   // Function-based decomposition
    Layered,     // Layered architecture
    DataDriven,  // Data-centric decomposition
    Pipeline,    // Pipeline-based decomposition
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_modular_decomposition() {
        let config = DecomposeConfig::default();
        let engine = DecomposeEngine::new(config).unwrap();
        
        let cot_result = CoTResult {
            task_analysis: "Create a sorting function".to_string(),
            reasoning_steps: vec![],
            edge_cases: vec![],
            assumptions: vec![],
            risks: vec![],
            approach: "Use quicksort algorithm".to_string(),
        };
        
        let modules = engine.decompose(&cot_result).await.unwrap();
        assert!(!modules.is_empty());
        assert!(modules.len() <= 20); // max_modules default
    }
    
    #[tokio::test]
    async fn test_size_constraints() {
        let mut config = DecomposeConfig::default();
        config.min_module_size = 10;
        config.max_module_size = 100;
        
        let engine = DecomposeEngine::new(config).unwrap();
        
        let cot_result = CoTResult {
            task_analysis: "Complex task".to_string(),
            reasoning_steps: vec![],
            edge_cases: vec![],
            assumptions: vec![],
            risks: vec![],
            approach: "Complex approach".to_string(),
        };
        
        let modules = engine.decompose(&cot_result).await.unwrap();
        for module in &modules {
            assert!(module.estimated_lines >= 10);
            assert!(module.estimated_lines <= 100);
        }
    }
}
