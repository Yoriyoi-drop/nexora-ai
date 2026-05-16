# Foundation API Freeze

> **Status**: FROZEN — Do not add, remove, rename, or re-type any public item below
> without a v2 migration plan. Changes require CEO/CTO sign-off.

## 1. Error Types

| Item | Kind | Consumers |
|---|---|---|
| `FoundationResult<T>` | Type alias | inference, intelligence, cognition, app |
| `FoundationError` | Enum | all |
| `FoundationError::Implementation(String)` | Variant | internal |
| `FoundationError::Configuration(String)` | Variant | internal |
| `FoundationError::Resource(String)` | Variant | internal |
| `FoundationError::Timeout` | Variant | inference, app |
| `FoundationError::NotImplemented` | Variant | internal |

## 2. Core Traits (traits/)

### Tensor Traits

| Trait | Methods | Notes |
|---|---|---|
| `TensorOps` | `add`, `sub`, `mul`, `div`, `ndim`, `size`, `reshape` | |
| `TensorMath` | `sum`, `mean`, `std`, `min`, `max`, `argmin`, `argmax` | |
| `TensorTransform` | `transpose`, `map`, `enumerate_map`, `slice`, `concat` | |
| `TensorCompression` | `compress`, `decompress`, `is_compressed`, `compression_ratio` | |
| `TensorValidation` | `validate_shape`, `validate_data`, `has_nan`, `has_inf`, `is_finite` | |
| `TensorIO` | `save_to_file`, `load_from_file`, `to_bytes`, `from_bytes` | |
| `TensorCompare` | `approx_eq`, `eq`, `elementwise_eq`, `elementwise_approx_eq` | |
| `TensorAggregate` | `sum_axis`, `mean_axis`, `min_axis`, `max_axis`, `argmin_axis`, `argmax_axis` | |

### Model Traits

| Trait | Methods | Notes |
|---|---|---|
| `ModelInference` | `infer`, `batch_infer`, `input_shape`, `output_shape` | |
| `ModelTraining` | `train`, `train_epoch`, `validate`, `training_loss`, `validation_loss` | |
| `ModelOptimization` | `optimize`, `quantize`, `prune`, `model_size`, `parameter_count` | |
| `ModelSerializable` | `save`, `load`, `export`, `import` | |
| `ModelConfigurable<C>` | `set_config`, `get_config`, `validate_config` | generic over config |
| `ModelMetrics` | `get_metrics`, `reset_metrics`, `log_metrics` | |
| `ModelLayer` | `forward`, `parameters`, `gradients`, `update_parameters`, `name`, `input_shape`, `output_shape` | |
| `ActivationFunction` | `activate`, `derivative`, `activate_tensor`, `derivative_tensor`, `name` | |
| `LossFunction` | `compute`, `gradient`, `name` | |
| `Optimizer` | `step`, `reset`, `learning_rate`, `set_learning_rate`, `name` | |
| `ModelEvaluation` | `evaluate`, `accuracy`, `precision`, `recall`, `f1_score` | |
| `EvaluationMetrics` | struct: `accuracy`, `precision`, `recall`, `f1_score`, `loss`, `inference_time_ms` | |

### Core Component Traits

| Trait | Methods | Notes |
|---|---|---|
| `Initializable` | `initialize`, `is_initialized` | |
| `Resettable` | `reset` | |
| `Serializable` | `serialize`, `deserialize` | generic Output w/ serde |
| `Configurable<C>` | `configure`, `config` | generic over config |
| `Validatable` | `validate` | |
| `CloneableWithConfig<C>` | `clone_with_config` | generic over config |
| `Measurable` | `metrics`, `reset_metrics` | |

## 3. NXR Model Series (shared/ + models/)

### Core Base Types (frozen)

| Item | Kind | Notes |
|---|---|---|
| `NxrModel` trait | Trait | 12 methods — central model abstraction |
| `BaseNxrModel<C,M,S>` | Struct | Generic shared implementation |
| `NxrModelError` | Enum | 9 error variants |
| `NxrModelResult<T>` | Type alias | |
| `NxrInput` | Struct | `id`, `timestamp`, `data`, `parameters`, `metadata` |
| `NxrOutput` | Struct | `id`, `input_id`, `timestamp`, `data`, `metadata`, `performance` |
| `NxrStreamChunk` | Struct | `id`, `input_id`, `timestamp`, `data`, `is_final` |
| `InputData` | Enum | `Text`, `Tokens`, `Image`, `Audio`, `Multimodal`, `Structured` |
| `OutputData` | Enum | `Text`, `Tokens`, `Image`, `Audio`, `Multimodal`, `Structured` |
| `TokenOutput` | Struct | `token_id`, `text`, `log_prob`, `position` |
| `GenerationMetadata` | Struct | `finish_reason`, `total_tokens`, `generation_time_ms`, `model_version`, `seed` |
| `FinishReason` | Enum | 7 variants |
| `PerformanceMetrics` | Struct | 6 fields |
| `StreamChunkData` | Enum | 4 variants |
| `ValidationResult` | Struct | 4 fields |
| `ModelStatistics` | Struct | 8 fields |
| `ResourceUsage` | Struct | 9 fields |
| `ModelMeta` | Struct | identity metadata (from `model_identity`) |
| `NxrModelId` | Enum | 10 model variants (Omnis, Vortex, ..., Genesis) |
| `CapabilityVector` | Struct | (from `capability_spec`) |

### Transformer Model (frozen)

| Item | Kind | Notes |
|---|---|---|
| `TransformerConfig` | Struct | 10 fields |
| `CausalLM` | Struct | 7 fields, 6 methods |
| `KVCacheEntry` | Struct | re-exported from `gqa` |
| `RMSNorm` | Struct | re-exported from `rms_norm` |
| `RoPE` | Struct | re-exported from `rope` |
| `TrainableCausalLM` | Struct | re-exported from `trainable` |

### NXR Model Implementations (frozen - names only, internals can evolve)

| Item | Notes |
|---|---|
| `NxrOmnisModel` | Full implementation |
| `NxrVortexModel` | Full implementation |
| `NxrAetherModel` | Full implementation |
| `NxrSpectraModel` | Full implementation |
| `NxrNexumModel` | Full implementation |
| `NxrAxiomModel` | Full implementation |
| `NxrCipherModel` | Full implementation |
| `NxrSwiftModel` | Full implementation |
| `NxrKronosModel` | Full implementation |
| `NxrGenesisModel` | Full implementation |
| `get_model_implementation` | Free function returning `Box<dyn Any>` |

### Evaluation Types (frozen)

| Item | Kind |
|---|---|
| `EvaluationResult` | Struct |
| `EvaluationReport` | Struct |

## 4. SACA Reasoning (reasoning/)

| Item | Kind | Methods |
|---|---|---|
| `SACA` | Struct | `new`, `solve`, `get_current_session`, `get_metrics`, `reset_metrics`, `config` |
| `SACAConfig` | Struct | 7 sub-configs + 5 global fields |
| `SacaEngine` | Struct | `new`, `reason` |
| `SacaAetherIntegration` | Struct | `new`, `enhanced_reasoning` |
| `SacaAetherConfig` | Struct | 3 fields |
| `EnhancedReasoningResult` | Struct | `new`, `combine_results`, `summary` |
| `ReasoningResult` | Struct | `conclusion` |
| `SACAError` | Enum | error types |
| `CodingTask` | Struct | 4 fields |
| `SACASolution` | Struct | 9 fields |
| `SACAFeedback` | Struct | 6 fields |
| `SACAMetrics` | Struct | 7 fields |
| `SACASession` | Struct | 5 fields |
| `SACAPhase` | Enum | 6 variants |
| `CoTResult` | Struct | 6 fields |
| `Module` | Struct | 8 fields |
| `RepositoryContext` | Struct | 8 fields |
| `SamplingCandidate` | Struct | 8 fields |
| `SACAExecutionResult` | Struct | 10 fields |
| `RerankingCriteria` | Struct | 6 weight fields |
| All config types | Structs | `CoTConfig`, `DecomposeConfig`, `ContextConfig`, `SamplingConfig`, `ExecuteConfig`, `RerankConfig`, `FeedbackConfig`, `FeedbackConfig` |

## 5. Other Public Modules

| Module | Key Items | Notes |
|---|---|---|
| `atqs` | Tensor, ATQS-Compress, ATQS-Train core types | |
| `has_moe_ffn` | `Attention`, `AttentionConfig`, MoE-FFN | Core attention mechanism |
| `oracle` | Oracle7, Omnis agent system | |
| `alignment` | Alignment system | |
| `hldva_t` | HLDA-VT re-exported | |
| `vogp` | VOGP+ re-exported | |
| `erp` | ERP re-exported | |
| `compression` | Compression framework | |
| `multimodal` | Multimodal types | |
| `training` | Training infrastructure | |
| `clustering_orchestrator` | Model clustering | |
| `validation` | Validation utilities | |
| `safetensors` | SafeTensor loader | |

## 6. External Crate Dependencies (frozen)

The foundation crate depends on these external crates. Do not add/remove without v2 migration:

| Crate | Usage |
|---|---|
| `ndarray` | Tensor types, Transformer inference |
| `serde` / `serde_json` | Serialization |
| `tokio` | Async runtime |
| `tracing` | Instrumentation |
| `uuid` | IDs |
| `chrono` | Timestamps |
| `rand` | RNG |
| `thiserror` | Error derives |
| `async-trait` | Async trait methods |

---

## Freeze Rules

1. **No new public items** added to `lib.rs` re-exports, `traits/`, `shared/`, `reasoning/`, or `models/` without a migration plan.
2. **No renaming** of any frozen item.
3. **No signature changes** to frozen trait methods or struct fields.
4. **No new external dependencies** added to `Cargo.toml`.
5. **Internal module implementations** can change freely as long as the public API surface (names, signatures, semantics) is preserved.
6. **Bug fixes** to frozen implementations are allowed and encouraged.
7. **Adding new private modules** is allowed. Adding new **public** modules requires sign-off.

> Frozen by: CEO/CTO review — May 2026
