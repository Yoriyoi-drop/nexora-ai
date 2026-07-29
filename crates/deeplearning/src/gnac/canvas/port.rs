use crate::gnac::canvas::CanvasPosition;
use crate::gnac::TensorDesc;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gnac::DType;

    fn input_port() -> PortDescriptor {
        PortDescriptor::new(
            "in",
            PortDirection::Input,
            TensorDesc::new(vec![1, 64], DType::F32),
        )
    }

    fn output_port() -> PortDescriptor {
        PortDescriptor::new(
            "out",
            PortDirection::Output,
            TensorDesc::new(vec![1, 64], DType::F32),
        )
    }

    #[test]
    fn test_port_descriptor_new() {
        let p = input_port();
        assert_eq!(p.name, "in");
        assert_eq!(p.direction, PortDirection::Input);
        assert_eq!(p.visual_type, PortVisualType::Matrix2D);
    }

    #[test]
    fn test_port_descriptor_scalar() {
        let p = PortDescriptor::new(
            "s",
            PortDirection::Input,
            TensorDesc::new(vec![], DType::F32),
        );
        assert_eq!(p.visual_type, PortVisualType::Scalar);
    }

    #[test]
    fn test_port_descriptor_1d() {
        let p = PortDescriptor::new(
            "v",
            PortDirection::Input,
            TensorDesc::new(vec![10], DType::F32),
        );
        assert_eq!(p.visual_type, PortVisualType::Vector1D);
    }

    #[test]
    fn test_port_descriptor_3d() {
        let p = PortDescriptor::new(
            "t",
            PortDirection::Input,
            TensorDesc::new(vec![1, 64, 64], DType::F32),
        );
        assert_eq!(p.visual_type, PortVisualType::Tensor3D);
    }

    #[test]
    fn test_port_descriptor_sequence() {
        let p = PortDescriptor::new(
            "seq",
            PortDirection::Input,
            TensorDesc::new(vec![4, 3, 224, 224], DType::F32),
        );
        assert_eq!(p.visual_type, PortVisualType::SequenceTensor);
    }

    #[test]
    fn test_port_compatible() {
        let inp = input_port();
        let out = output_port();
        assert!(inp.is_compatible_with(&out));
    }

    #[test]
    fn test_port_incompatible_same_direction() {
        let a = input_port();
        let b = input_port();
        assert!(!a.is_compatible_with(&b));
    }

    #[test]
    fn test_port_incompatible_shape() {
        let inp = PortDescriptor::new(
            "in",
            PortDirection::Input,
            TensorDesc::new(vec![1, 64], DType::F32),
        );
        let out = PortDescriptor::new(
            "out",
            PortDirection::Output,
            TensorDesc::new(vec![1, 128], DType::F32),
        );
        assert!(!inp.is_compatible_with(&out));
    }
}
