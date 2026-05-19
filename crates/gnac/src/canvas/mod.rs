//! Tensor Canvas & Node-Based Architecture
//!
//! Setiap node merepresentasikan operasi tensor aktual: convolution, attention,
//! normalization, activation, pooling, routing, memory state, atau logic controller.
//! Port divisualisasikan berdasarkan struktur tensor (1D, 2D, 3D, sequence, multimodal).
//! Mendukung skip connection, dense routing, recurrent loop, cross-modal fusion,
//! conditional execution, dan hierarchical branching.

pub mod node;
pub mod port;
pub mod edge;
pub mod graph;
pub mod metanode;


pub use node::*;
pub use port::*;
pub use edge::*;
pub use graph::*;
pub use metanode::*;

/// Posisi node pada kanvas 2D
#[derive(Debug, Clone, Copy)]
pub struct CanvasPosition {
    pub x: f64,
    pub y: f64,
}

impl CanvasPosition {
    pub fn new(x: f64, y: f64) -> Self {
        CanvasPosition { x, y }
    }

    pub fn distance_to(&self, other: &CanvasPosition) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

/// Level zoom semantic untuk hierarchical graph compression
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoomLevel {
    System,
    Module,
    Tensor,
}
