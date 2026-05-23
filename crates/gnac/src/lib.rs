pub mod canvas;
pub mod collaboration;
pub mod distillation;
pub mod elastic;
pub mod execution;
pub mod experiment;
pub mod intervention;
pub mod lensing;
pub mod logic;
pub mod rce;
pub mod sandbox;
pub mod scheduler;
pub mod smart_tensor;
pub mod swarm;

pub use canvas::*;
pub use collaboration::*;
pub use distillation::*;
pub use elastic::*;
pub use execution::*;
pub use experiment::*;
pub use intervention::*;
pub use lensing::*;
pub use logic::*;
pub use rce::*;
pub use sandbox::*;
pub use scheduler::*;
pub use smart_tensor::*;
pub use swarm::*;

pub type DLResult<T> = std::result::Result<T, DeepLearningError>;

#[derive(Debug, thiserror::Error)]
pub enum DeepLearningError {
    #[error("Tensor shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("Invalid input dimension: {dim}")]
    InvalidDimension { dim: usize },
    #[error("Memory allocation failed: {reason}")]
    MemoryAllocation { reason: String },
    #[error("Computation error: {reason}")]
    Computation { reason: String },
    #[error("Configuration error: {reason}")]
    Configuration { reason: String },
}

impl From<ndarray::ShapeError> for DeepLearningError {
    fn from(_err: ndarray::ShapeError) -> Self {
        DeepLearningError::ShapeMismatch {
            expected: vec![],
            actual: vec![],
        }
    }
}

/// Representasi tensor multidimensional
#[derive(Debug, Clone)]
pub struct TensorDesc {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub strides: Vec<usize>,
    pub numel: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DType {
    F32,
    F64,
    F16,
    BF16,
    I32,
    I64,
    U8,
    Bool,
}

impl TensorDesc {
    pub fn new(shape: Vec<usize>, dtype: DType) -> Self {
        let strides = shape
            .iter()
            .rev()
            .scan(1, |acc, &dim| {
                let s = *acc;
                *acc *= dim;
                Some(s)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        let numel = shape.iter().product();
        TensorDesc {
            shape,
            dtype,
            strides,
            numel,
        }
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn is_compatible_with(&self, other: &TensorDesc) -> bool {
        self.shape == other.shape && self.dtype == other.dtype
    }
}

/// Tipe node dalam graf GNAC
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NodeType {
    Conv1D,
    Conv2D,
    Conv3D,
    SelfAttention,
    CrossAttention,
    Linear,
    Embedding,
    LayerNorm,
    BatchNorm,
    InstanceNorm,
    ReLU,
    GELU,
    Sigmoid,
    Tanh,
    Softmax,
    MaxPool,
    AvgPool,
    GlobalAvgPool,
    Dropout,
    Reshape,
    Transpose,
    Concat,
    Split,
    Add,
    Mul,
    MatMul,
    MultiHeadAttention,
    FeedForward,
    RotaryEmbedding,
    RMSNorm,
    SwiGLU,
    SparseAttention,
    SlidingWindowAttention,
    FlashAttention,
    StateSpaceModel,
    MambaBlock,
    Input,
    Output,
    Condition,
    RecurrentLoop,
    AdaptiveScheduler,
    RLFeedback,
    ContextMemory,
    Distillation,
    SkipConnection,
    MetaNode,
}

/// Status kesehatan node selama training
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Warning { reason: String },
    Critical { reason: String },
    Dead,
}

/// Konfigurasi global GNAC
#[derive(Debug, Clone)]
pub struct GnacConfig {
    pub max_nodes: usize,
    pub enable_lensing: bool,
    pub enable_swarm: bool,
    pub enable_intervention: bool,
    pub enable_collaboration: bool,
    pub enable_sandbox: bool,
    pub tensor_pool_size_mb: usize,
    pub default_dtype: DType,
}

impl Default for GnacConfig {
    fn default() -> Self {
        GnacConfig {
            max_nodes: 10_000,
            enable_lensing: true,
            enable_swarm: true,
            enable_intervention: true,
            enable_collaboration: false,
            enable_sandbox: false,
            tensor_pool_size_mb: 4096,
            default_dtype: DType::F32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_desc_new() {
        let desc = TensorDesc::new(vec![2, 3, 4], DType::F32);
        assert_eq!(desc.shape, vec![2, 3, 4]);
        assert_eq!(desc.numel, 24);
        assert_eq!(desc.dtype, DType::F32);
        assert_eq!(desc.strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_tensor_desc_ndim() {
        let desc = TensorDesc::new(vec![2, 3], DType::F64);
        assert_eq!(desc.ndim(), 2);
    }

    #[test]
    fn test_tensor_desc_scalar_ndim() {
        let desc = TensorDesc::new(vec![], DType::I32);
        assert_eq!(desc.ndim(), 0);
    }

    #[test]
    fn test_tensor_desc_compatible() {
        let a = TensorDesc::new(vec![2, 3], DType::F32);
        let b = TensorDesc::new(vec![2, 3], DType::F32);
        assert!(a.is_compatible_with(&b));
    }

    #[test]
    fn test_tensor_desc_incompatible_shape() {
        let a = TensorDesc::new(vec![2, 3], DType::F32);
        let b = TensorDesc::new(vec![3, 2], DType::F32);
        assert!(!a.is_compatible_with(&b));
    }

    #[test]
    fn test_tensor_desc_incompatible_dtype() {
        let a = TensorDesc::new(vec![2, 3], DType::F32);
        let b = TensorDesc::new(vec![2, 3], DType::I32);
        assert!(!a.is_compatible_with(&b));
    }

    #[test]
    fn test_gnac_config_default() {
        let cfg = GnacConfig::default();
        assert_eq!(cfg.max_nodes, 10_000);
        assert!(cfg.enable_lensing);
        assert!(!cfg.enable_collaboration);
    }

    #[test]
    fn test_deep_learning_error_ndarray_conversion() {
        let err = ndarray::ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape);
        let dl_err: DeepLearningError = err.into();
        assert!(matches!(dl_err, DeepLearningError::ShapeMismatch { .. }));
    }

    #[test]
    fn test_dtype_debug_and_clone() {
        let a = DType::F32;
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn test_node_type_variants() {
        let variants = vec![
            NodeType::Conv2D,
            NodeType::SelfAttention,
            NodeType::Linear,
            NodeType::ReLU,
            NodeType::Dropout,
            NodeType::Input,
            NodeType::Output,
            NodeType::MambaBlock,
        ];
        assert_eq!(variants.len(), 8);
    }

    #[test]
    fn test_health_status_debug() {
        let h = HealthStatus::Healthy;
        assert_eq!(format!("{:?}", h), "Healthy");
    }
}
