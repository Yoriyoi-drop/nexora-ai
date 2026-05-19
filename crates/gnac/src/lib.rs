pub mod canvas;
pub mod smart_tensor;
pub mod lensing;
pub mod rce;
pub mod swarm;
pub mod execution;
pub mod scheduler;
pub mod logic;
pub mod intervention;
pub mod elastic;
pub mod distillation;
pub mod experiment;
pub mod collaboration;
pub mod sandbox;

pub use canvas::*;
pub use smart_tensor::*;
pub use lensing::*;
pub use rce::*;
pub use swarm::*;
pub use execution::*;
pub use scheduler::*;
pub use logic::*;
pub use intervention::*;
pub use elastic::*;
pub use distillation::*;
pub use experiment::*;
pub use collaboration::*;
pub use sandbox::*;

pub type DLResult<T> = std::result::Result<T, DeepLearningError>;

#[derive(Debug, thiserror::Error)]
pub enum DeepLearningError {
    #[error("Tensor shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },
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
        DeepLearningError::ShapeMismatch { expected: vec![], actual: vec![] }
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
        TensorDesc { shape, dtype, strides, numel }
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
