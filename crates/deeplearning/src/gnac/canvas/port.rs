use crate::gnac::TensorDesc;
use crate::gnac::canvas::CanvasPosition;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PortDirection {
    Input,
    Output,
}

/// Visualisasi port berdasarkan struktur tensor
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PortVisualType {
    Vector1D,
    Matrix2D,
    Tensor3D,
    SequenceTensor,
    MultimodalEmbedding,
    Scalar,
}

impl PortVisualType {
    pub fn from_shape(shape: &[usize]) -> Self {
        match shape.len() {
            0 => PortVisualType::Scalar,
            1 => PortVisualType::Vector1D,
            2 => PortVisualType::Matrix2D,
            3 => PortVisualType::Tensor3D,
            _ => {
                if shape.len() >= 4 && shape[0] > 1 {
                    PortVisualType::SequenceTensor
                } else {
                    PortVisualType::Tensor3D
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct PortDescriptor {
    pub id: Uuid,
    pub name: String,
    pub direction: PortDirection,
    pub tensor: TensorDesc,
    pub visual_type: PortVisualType,
    pub position: Option<CanvasPosition>,
}

impl PortDescriptor {
    pub fn new(name: &str, direction: PortDirection, tensor: TensorDesc) -> Self {
        let visual_type = PortVisualType::from_shape(&tensor.shape);
        PortDescriptor {
            id: Uuid::new_v4(),
            name: name.to_string(),
            direction,
            tensor,
            visual_type,
            position: None,
        }
    }

    pub fn is_compatible_with(&self, other: &PortDescriptor) -> bool {
        self.direction != other.direction && self.tensor.is_compatible_with(&other.tensor)
    }
}
