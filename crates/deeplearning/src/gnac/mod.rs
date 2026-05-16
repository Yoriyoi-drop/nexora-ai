//! GraphFlow Neural Architecture Composer (GNAC)
//!
//! Sistem visual programming berbasis graf untuk mendesain, melatih,
//! menganalisis, mengoptimalkan, dan menyebarkan model deep learning.
//!
//! # Paradigma
//! GNAC memperlakukan operasi neural sebagai node aktif, tensor sebagai
//! entitas visual hidup, dan training dynamics sebagai bagian eksplisit
//! dari antarmuka. Berbeda dari drag-and-drop konvensional, seluruh
//! komponen dirancang khusus untuk operasi tensor & deep learning.
//!
//! # Arsitektur
//! - **canvas** — Tensor Canvas & Node-Based Architecture
//! - **smart_tensor** — SmartTensor: Visual Tensor Intelligence
//! - **lensing** — Neural Graph Lensing
//! - **rce** — Resource Cognition Engine
//! - **swarm** — Swarm Agent: Constrained Neural Architecture Search
//! - **execution** — Hybrid Deferred Execution Engine
//! - **scheduler** — Tensor Scheduling Layer
//! - **logic** — Logic Flow & Training Dynamics
//! - **intervention** — Guided Intervention System
//! - **elastic** — Elastic Inference Graph
//! - **distillation** — Deployment via Distillation Node
//! - **experiment** — Deterministic Experiment System
//! - **collaboration** — Collaborative Neural Workspace
//! - **sandbox** — Secure Tensor Sandbox

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

use crate::DLResult;

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
    // Operasi tensor dasar
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

    // Arsitektur modern
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

    // Logic & kontrol
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
