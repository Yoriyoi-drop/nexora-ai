//! Modular Decomposition Engine
//!
//! Phase 2 of SACA: Break down complex problems into independent modules
//! Implements CodeChain methodology with clear I/O contracts

use super::{config::*, error::*, types::*};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Modular Decomposition engine
pub struct DecomposeEngine {
    config: DecomposeConfig,
    decomposition_cache: Arc<RwLock<std::collections::HashMap<String, Vec<Module>>>>,
}

impl DecomposeEngine {
    /// Create new Decompose engine
    pub fn new(config: DecomposeConfig) -> SACAResult<Self> {
        info!(
            "Decompose Engine initialized with max {} modules",
            config.max_modules
        );

        Ok(Self {
            config,
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
        self.decomposition_cache
            .write()
            .await
            .insert(cache_key, modules.clone());

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
            }
            DecompositionStrategy::Layered => {
                modules.extend(self.layered_decomposition(cot_result).await?);
            }
            DecompositionStrategy::DataDriven => {
                modules.extend(self.data_driven_decomposition(cot_result).await?);
            }
            DecompositionStrategy::Pipeline => {
                modules.extend(self.pipeline_decomposition(cot_result).await?);
            }
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
    async fn determine_decomposition_strategy(
        &self,
        cot_result: &CoTResult,
    ) -> SACAResult<DecompositionStrategy> {
        let task_desc = cot_result.task_analysis.to_lowercase();

        if task_desc.contains("pipeline")
            || task_desc.contains("flow")
            || task_desc.contains("process")
        {
            Ok(DecompositionStrategy::Pipeline)
        } else if task_desc.contains("data")
            || task_desc.contains("database")
            || task_desc.contains("storage")
        {
            Ok(DecompositionStrategy::DataDriven)
        } else if task_desc.contains("layer")
            || task_desc.contains("tier")
            || task_desc.contains("architecture")
        {
            Ok(DecompositionStrategy::Layered)
        } else {
            Ok(DecompositionStrategy::Functional)
        }
    }

    /// Functional decomposition approach
    async fn functional_decomposition(&self, cot_result: &CoTResult) -> SACAResult<Vec<Module>> {
        let mut modules = Vec::new();
        let step_count = cot_result.reasoning_steps.len().max(1);
        let base_lines = 150u32.saturating_add((step_count as u32) * 20);

        // Core logic module — named after the approach
        modules.push(Module {
            id: Uuid::new_v4().to_string(),
            name: format!("Core_{}", cot_result.approach.replace(' ', "_")),
            description: format!(
                "Core logic using approach '{}': {}",
                cot_result.approach, cot_result.task_analysis
            ),
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
            outputs: vec![ModuleIO {
                name: "result".to_string(),
                data_type: "Result<T>".to_string(),
                description: "Processed result".to_string(),
                optional: false,
            }],
            dependencies: vec![],
            complexity: ModuleComplexity::High,
            estimated_lines: base_lines,
        });

        // Input validation module — incorporate edge cases from CoT
        let edge_case_descriptions: Vec<String> =
            cot_result.edge_cases.iter().take(3).cloned().collect();
        let validation_desc = if edge_case_descriptions.is_empty() {
            "Validates input data and parameters".to_string()
        } else {
            format!(
                "Validates input data and parameters. Handles edge cases: {}",
                edge_case_descriptions.join("; ")
            )
        };

        modules.push(Module {
            id: Uuid::new_v4().to_string(),
            name: "InputValidator".to_string(),
            description: validation_desc,
            inputs: vec![ModuleIO {
                name: "raw_input".to_string(),
                data_type: "RawInput".to_string(),
                description: "Unvalidated input".to_string(),
                optional: false,
            }],
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
            estimated_lines: 50u32.saturating_add((cot_result.edge_cases.len() as u32) * 10),
        });

        // Assumptions checking module — derived from CoT assumptions
        modules.push(Module {
            id: Uuid::new_v4().to_string(),
            name: "AssumptionsValidator".to_string(),
            description: format!(
                "Validates {} assumptions identified during reasoning",
                cot_result.assumptions.len()
            ),
            inputs: vec![ModuleIO {
                name: "assumptions".to_string(),
                data_type: "Vec<String>".to_string(),
                description: "Assumptions to validate".to_string(),
                optional: false,
            }],
            outputs: vec![ModuleIO {
                name: "validated_assumptions".to_string(),
                data_type: "Vec<ValidatedAssumption>".to_string(),
                description: "Validation results per assumption".to_string(),
                optional: false,
            }],
            dependencies: vec![],
            complexity: ModuleComplexity::Low,
            estimated_lines: 30,
        });

        // Risk mitigation module — derived from CoT risks
        modules.push(Module {
            id: Uuid::new_v4().to_string(),
            name: "RiskMitigator".to_string(),
            description: format!(
                "Mitigates {} identified risks: {}",
                cot_result.risks.len(),
                cot_result.risks.join(", ")
            ),
            inputs: vec![ModuleIO {
                name: "risk_signal".to_string(),
                data_type: "RiskSignal".to_string(),
                description: "Risk indicators".to_string(),
                optional: false,
            }],
            outputs: vec![ModuleIO {
                name: "mitigation_result".to_string(),
                data_type: "MitigationResult".to_string(),
                description: "Risk mitigation result".to_string(),
                optional: false,
            }],
            dependencies: vec![],
            complexity: ModuleComplexity::Medium,
            estimated_lines: 60,
        });

        // Error handling module — sized by risk count
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
            outputs: vec![ModuleIO {
                name: "handled_result".to_string(),
                data_type: "Result<T>".to_string(),
                description: "Error handling result".to_string(),
                optional: false,
            }],
            dependencies: vec![],
            complexity: ModuleComplexity::Medium,
            estimated_lines: 80u32.saturating_add((cot_result.risks.len() as u32) * 15),
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
            outputs: vec![ModuleIO {
                name: "final_output".to_string(),
                data_type: "Output".to_string(),
                description: "Formatted output".to_string(),
                optional: false,
            }],
            dependencies: vec![],
            complexity: ModuleComplexity::Low,
            estimated_lines: 40,
        });

        Ok(modules)
    }

    /// Layered decomposition approach
    async fn layered_decomposition(&self, cot_result: &CoTResult) -> SACAResult<Vec<Module>> {
        let mut modules = Vec::new();
        let reasoning_depth = cot_result.reasoning_steps.len().max(1) as u32;

        // Presentation layer — named after the task
        modules.push(Module {
            id: Uuid::new_v4().to_string(),
            name: format!("{}_Interface", cot_result.approach.replace(' ', "_")),
            description: format!(
                "User interface and API endpoints for: {}",
                cot_result.task_analysis
            ),
            inputs: vec![ModuleIO {
                name: "user_request".to_string(),
                data_type: "Request".to_string(),
                description: "User request data".to_string(),
                optional: false,
            }],
            outputs: vec![ModuleIO {
                name: "response".to_string(),
                data_type: "Response".to_string(),
                description: "Response to user".to_string(),
                optional: false,
            }],
            dependencies: vec!["BusinessLayer".to_string()],
            complexity: ModuleComplexity::Medium,
            estimated_lines: 100u32.saturating_add(reasoning_depth * 10),
        });

        // Business layer — integrates reasoning steps
        let step_summary: Vec<String> = cot_result
            .reasoning_steps
            .iter()
            .map(|s| s.description.clone())
            .collect();
        let business_desc = if step_summary.is_empty() {
            "Business logic and rules".to_string()
        } else {
            format!(
                "Business logic derived from reasoning: {}",
                step_summary.join(" -> ")
            )
        };

        modules.push(Module {
            id: Uuid::new_v4().to_string(),
            name: "BusinessLayer".to_string(),
            description: business_desc,
            inputs: vec![ModuleIO {
                name: "processed_request".to_string(),
                data_type: "ProcessedRequest".to_string(),
                description: "Request from presentation layer".to_string(),
                optional: false,
            }],
            outputs: vec![ModuleIO {
                name: "business_result".to_string(),
                data_type: "BusinessResult".to_string(),
                description: "Business logic result".to_string(),
                optional: false,
            }],
            dependencies: vec!["DataLayer".to_string()],
            complexity: ModuleComplexity::High,
            estimated_lines: 200u32.saturating_add(reasoning_depth * 25),
        });

        // Data layer — accounts for edge cases and assumptions
        modules.push(Module {
            id: Uuid::new_v4().to_string(),
            name: "DataLayer".to_string(),
            description: format!(
                "Data access and persistence. Handles {} edge cases and validates {} assumptions",
                cot_result.edge_cases.len(),
                cot_result.assumptions.len()
            ),
            inputs: vec![ModuleIO {
                name: "data_request".to_string(),
                data_type: "DataRequest".to_string(),
                description: "Data access request".to_string(),
                optional: false,
            }],
            outputs: vec![ModuleIO {
                name: "data_result".to_string(),
                data_type: "DataResult".to_string(),
                description: "Data access result".to_string(),
                optional: false,
            }],
            dependencies: vec![],
            complexity: ModuleComplexity::Medium,
            estimated_lines: 120u32.saturating_add((cot_result.edge_cases.len() as u32) * 15),
        });

        // Risk guard layer — derived from identified risks
        if !cot_result.risks.is_empty() {
            modules.push(Module {
                id: Uuid::new_v4().to_string(),
                name: "RiskGuardLayer".to_string(),
                description: format!(
                    "Monitors and guards against risks: {}",
                    cot_result.risks.join(", ")
                ),
                inputs: vec![ModuleIO {
                    name: "system_state".to_string(),
                    data_type: "SystemState".to_string(),
                    description: "Current system state for risk evaluation".to_string(),
                    optional: false,
                }],
                outputs: vec![ModuleIO {
                    name: "risk_assessment".to_string(),
                    data_type: "RiskAssessment".to_string(),
                    description: "Risk evaluation result".to_string(),
                    optional: false,
                }],
                dependencies: vec!["BusinessLayer".to_string()],
                complexity: ModuleComplexity::Medium,
                estimated_lines: 80,
            });
        }

        Ok(modules)
    }

    /// Data-driven decomposition approach
    async fn data_driven_decomposition(&self, cot_result: &CoTResult) -> SACAResult<Vec<Module>> {
        let mut modules = Vec::new();

        // Data model — named after the task approach
        modules.push(Module {
            id: Uuid::new_v4().to_string(),
            name: format!("{}_DataModel", cot_result.approach.replace(' ', "_")),
            description: format!(
                "Data structures and models for: {}. Based on approach: {}",
                cot_result.task_analysis, cot_result.approach
            ),
            inputs: vec![],
            outputs: vec![ModuleIO {
                name: "model_definitions".to_string(),
                data_type: "ModelDefs".to_string(),
                description: "Data model definitions".to_string(),
                optional: false,
            }],
            dependencies: vec![],
            complexity: ModuleComplexity::Low,
            estimated_lines: 80u32.saturating_add((cot_result.assumptions.len() as u32) * 10),
        });

        // Data processor — driven by reasoning steps
        let processor_desc = if cot_result.reasoning_steps.is_empty() {
            "Data transformation and processing".to_string()
        } else {
            format!(
                "Data transformation following reasoning: {}",
                cot_result
                    .reasoning_steps
                    .iter()
                    .map(|s| s.logic.clone())
                    .collect::<Vec<_>>()
                    .join(" -> ")
            )
        };

        modules.push(Module {
            id: Uuid::new_v4().to_string(),
            name: "DataProcessor".to_string(),
            description: processor_desc,
            inputs: vec![ModuleIO {
                name: "raw_data".to_string(),
                data_type: "RawData".to_string(),
                description: "Raw input data".to_string(),
                optional: false,
            }],
            outputs: vec![ModuleIO {
                name: "processed_data".to_string(),
                data_type: "ProcessedData".to_string(),
                description: "Processed data".to_string(),
                optional: false,
            }],
            dependencies: vec![format!(
                "{}_DataModel",
                cot_result.approach.replace(' ', "_")
            )],
            complexity: ModuleComplexity::High,
            estimated_lines: 180u32.saturating_add((cot_result.reasoning_steps.len() as u32) * 20),
        });

        // Data validator — incorporates edge cases
        modules.push(Module {
            id: Uuid::new_v4().to_string(),
            name: "DataValidator".to_string(),
            description: format!(
                "Data validation and quality checks. Validates {} edge cases and {} assumptions",
                cot_result.edge_cases.len(),
                cot_result.assumptions.len()
            ),
            inputs: vec![ModuleIO {
                name: "data_to_validate".to_string(),
                data_type: "Data".to_string(),
                description: "Data to validate".to_string(),
                optional: false,
            }],
            outputs: vec![ModuleIO {
                name: "validation_report".to_string(),
                data_type: "ValidationReport".to_string(),
                description: "Validation results".to_string(),
                optional: false,
            }],
            dependencies: vec![format!(
                "{}_DataModel",
                cot_result.approach.replace(' ', "_")
            )],
            complexity: ModuleComplexity::Medium,
            estimated_lines: 90u32.saturating_add((cot_result.edge_cases.len() as u32) * 10),
        });

        // Edge case handler — one handler per detected edge case
        for edge_case in &cot_result.edge_cases {
            modules.push(Module {
                id: Uuid::new_v4().to_string(),
                name: format!("EdgeCaseHandler_{}", edge_case.replace(' ', "_")),
                description: format!("Handles edge case: {}", edge_case),
                inputs: vec![ModuleIO {
                    name: "data".to_string(),
                    data_type: "Data".to_string(),
                    description: "Data to check for edge case".to_string(),
                    optional: false,
                }],
                outputs: vec![ModuleIO {
                    name: "handled_data".to_string(),
                    data_type: "Data".to_string(),
                    description: "Data after edge case handling".to_string(),
                    optional: false,
                }],
                dependencies: vec!["DataProcessor".to_string()],
                complexity: ModuleComplexity::Low,
                estimated_lines: 30,
            });
        }

        Ok(modules)
    }

    /// Pipeline decomposition approach
    async fn pipeline_decomposition(&self, cot_result: &CoTResult) -> SACAResult<Vec<Module>> {
        let mut modules = Vec::new();
        let num_reasoning_steps = cot_result.reasoning_steps.len().max(1);
        // Number of stages matches the reasoning steps, capped at max config
        let num_stages = num_reasoning_steps
            .min(self.config.max_modules as usize)
            .min(6);

        // Pipeline coordinator — named after the task
        let pipeline_name = format!("{}_Pipeline", cot_result.approach.replace(' ', "_"));
        modules.push(Module {
            id: Uuid::new_v4().to_string(),
            name: pipeline_name.clone(),
            description: format!(
                "Orchestrates {}-stage pipeline for: {}",
                num_stages, cot_result.task_analysis
            ),
            inputs: vec![ModuleIO {
                name: "pipeline_input".to_string(),
                data_type: "PipelineInput".to_string(),
                description: "Input to pipeline".to_string(),
                optional: false,
            }],
            outputs: vec![ModuleIO {
                name: "pipeline_output".to_string(),
                data_type: "PipelineOutput".to_string(),
                description: "Final pipeline output".to_string(),
                optional: false,
            }],
            dependencies: (1..=num_stages)
                .map(|s| format!("{}Stage{}", pipeline_name, s))
                .collect(),
            complexity: ModuleComplexity::Medium,
            estimated_lines: 110u32.saturating_add(num_stages as u32 * 15),
        });

        // Pipeline stages — each driven by a reasoning step
        for i in 0..num_stages {
            let step = cot_result
                .reasoning_steps
                .get(i)
                .map(|s| s.description.clone())
                .unwrap_or_else(|| format!("Stage {}", i + 1));
            let stage_name = format!("{}Stage{}", pipeline_name, i + 1);

            modules.push(Module {
                id: Uuid::new_v4().to_string(),
                name: stage_name.clone(),
                description: format!("Pipeline stage {}: {}", i + 1, step),
                inputs: vec![ModuleIO {
                    name: format!("{}_input", stage_name),
                    data_type: format!("{}Input", stage_name),
                    description: format!("Input for stage {}", i + 1),
                    optional: false,
                }],
                outputs: vec![ModuleIO {
                    name: format!("{}_output", stage_name),
                    data_type: format!("{}Output", stage_name),
                    description: format!("Output from stage {}", i + 1),
                    optional: false,
                }],
                dependencies: if i > 0 {
                    vec![format!("{}Stage{}", pipeline_name, i)]
                } else {
                    vec![]
                },
                complexity: ModuleComplexity::Medium,
                estimated_lines: 70u32.saturating_add(num_stages as u32 * 5),
            });
        }

        // Risk handling stage — added if risks are identified
        if !cot_result.risks.is_empty() {
            modules.push(Module {
                id: Uuid::new_v4().to_string(),
                name: "RiskHandlingStage".to_string(),
                description: format!("Handles pipeline risks: {}", cot_result.risks.join(", ")),
                inputs: vec![ModuleIO {
                    name: "error_signal".to_string(),
                    data_type: "ErrorSignal".to_string(),
                    description: "Error signal from any pipeline stage".to_string(),
                    optional: false,
                }],
                outputs: vec![ModuleIO {
                    name: "recovery_action".to_string(),
                    data_type: "RecoveryAction".to_string(),
                    description: "Recovery action to execute".to_string(),
                    optional: false,
                }],
                dependencies: (1..=num_stages)
                    .map(|s| format!("{}Stage{}", pipeline_name, s))
                    .collect(),
                complexity: ModuleComplexity::Medium,
                estimated_lines: 80,
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
            warn!(
                "Module count truncated to configured maximum of {}",
                self.config.max_modules
            );
        }

        Ok(result)
    }

    /// Split a large module into smaller ones
    async fn split_large_module(&self, module: Module) -> SACAResult<Vec<Module>> {
        let mut split_modules = Vec::new();
        let num_splits = ((module.estimated_lines as f32 / self.config.max_module_size as f32)
            .ceil() as u32)
            .max(1);

        for i in 0..num_splits {
            split_modules.push(Module {
                id: Uuid::new_v4().to_string(),
                name: module.name.clone() + "_part" + &(i + 1).to_string(),
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
    async fn generate_io_specifications(
        &self,
        mut modules: Vec<Module>,
    ) -> SACAResult<Vec<Module>> {
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
            return Err(SACAError::DecomposeError(
                "No modules generated".to_string(),
            ));
        }

        // Check for duplicate names
        let mut names = std::collections::HashSet::new();
        for module in modules {
            if !names.insert(&module.name) {
                return Err(SACAError::DecomposeError(format!(
                    "Duplicate module name: {}",
                    module.name
                )));
            }
        }

        // Check dependency validity
        let module_ids: std::collections::HashSet<String> =
            modules.iter().map(|m| &m.id).cloned().collect();

        for module in modules {
            for dep in &module.dependencies {
                if !module_ids.contains(dep) {
                    return Err(SACAError::DecomposeError(format!(
                        "Invalid dependency in module {}: {}",
                        module.name, dep
                    )));
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
    Functional, // Function-based decomposition
    Layered,    // Layered architecture
    DataDriven, // Data-centric decomposition
    Pipeline,   // Pipeline-based decomposition
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_modular_decomposition() -> Result<(), anyhow::Error> {
        let config = DecomposeConfig::default();
        let engine = DecomposeEngine::new(config)
            .map_err(|e| anyhow::anyhow!("Failed to create decompose engine: {}", e))?;

        let cot_result = CoTResult {
            task_analysis: "Create a sorting function".to_string(),
            reasoning_steps: vec![],
            edge_cases: vec![],
            assumptions: vec![],
            risks: vec![],
            approach: "Use quicksort algorithm".to_string(),
        };

        let modules = engine
            .decompose(&cot_result)
            .await
            .map_err(|e| anyhow::anyhow!("Decomposition failed: {}", e))?;
        assert!(!modules.is_empty());
        assert!(modules.len() <= 20); // max_modules default
        Ok(())
    }

    #[tokio::test]
    async fn test_size_constraints() -> Result<(), anyhow::Error> {
        let mut config = DecomposeConfig::default();
        config.min_module_size = 10;
        config.max_module_size = 100;

        let engine = DecomposeEngine::new(config)
            .map_err(|e| anyhow::anyhow!("Failed to create decompose engine: {}", e))?;

        let cot_result = CoTResult {
            task_analysis: "Complex task".to_string(),
            reasoning_steps: vec![],
            edge_cases: vec![],
            assumptions: vec![],
            risks: vec![],
            approach: "Complex approach".to_string(),
        };

        let modules = engine
            .decompose(&cot_result)
            .await
            .map_err(|e| anyhow::anyhow!("Decomposition failed: {}", e))?;
        for module in &modules {
            assert!(module.estimated_lines >= 10);
            assert!(module.estimated_lines <= 100);
        }

        Ok(())
    }
}
