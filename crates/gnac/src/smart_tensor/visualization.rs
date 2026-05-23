use crate::canvas::GradientStatus;
use crate::smart_tensor::SmartTensorMetadata;

/// Parameter visualisasi untuk SmartTensor
#[derive(Debug, Clone)]
pub struct TensorVisualization {
    /// Ketebalan kabel (pixels) — proporsional terhadap bandwidth
    pub cable_thickness: f32,
    /// Warna kabel berdasarkan stabilitas gradien
    pub cable_color: CableColor,
    /// Pola animasi berdasarkan throughput
    pub animation_pattern: AnimationPattern,
    /// Opacity
    pub opacity: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CableColor {
    Green,  // gradien stabil
    Yellow, // gradien mulai tidak stabil
    Red,    // exploding/vanishing gradient
    Blue,   // tensor frozen
    Gray,   // tensor tidak aktif
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnimationPattern {
    Solid,   // throughput rendah
    Dashed,  // throughput sedang
    Dotted,  // throughput tinggi
    Pulsing, // gradien tidak stabil
}

impl TensorVisualization {
    pub fn from_metadata(meta: &SmartTensorMetadata) -> Self {
        let cable_thickness = (meta.bandwidth_estimate / 1000.0).clamp(1.0, 20.0) as f32;

        let cable_color = match meta.gradient {
            GradientStatus::Stable => {
                if meta.is_frozen {
                    CableColor::Blue
                } else {
                    CableColor::Green
                }
            }
            GradientStatus::Exploding(_) => CableColor::Red,
            GradientStatus::Vanishing(_) => CableColor::Red,
            GradientStatus::Saturated => CableColor::Yellow,
        };

        let animation_pattern = match meta.gradient {
            GradientStatus::Exploding(_) | GradientStatus::Vanishing(_) => {
                AnimationPattern::Pulsing
            }
            GradientStatus::Saturated => AnimationPattern::Dashed,
            _ => {
                if meta.bandwidth_estimate > 5000.0 {
                    AnimationPattern::Dotted
                } else if meta.bandwidth_estimate > 1000.0 {
                    AnimationPattern::Dashed
                } else {
                    AnimationPattern::Solid
                }
            }
        };

        let opacity = if meta.is_frozen { 0.5 } else { 1.0 };

        TensorVisualization {
            cable_thickness,
            cable_color,
            animation_pattern,
            opacity,
        }
    }
}

/// Mengonversi CableColor ke kode warna RGB hex
pub fn color_to_hex(color: CableColor) -> &'static str {
    match color {
        CableColor::Green => "#22C55E",
        CableColor::Yellow => "#EAB308",
        CableColor::Red => "#EF4444",
        CableColor::Blue => "#3B82F6",
        CableColor::Gray => "#6B7280",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smart_tensor::SmartTensorMetadata;
    use crate::TensorDesc;

    fn meta() -> SmartTensorMetadata {
        SmartTensorMetadata::new(TensorDesc::new(vec![1, 64], crate::DType::F32))
    }

    #[test]
    fn test_visualization_from_metadata_stable() {
        let m = meta();
        let vis = TensorVisualization::from_metadata(&m);
        assert_eq!(vis.cable_color, CableColor::Green);
        assert!((vis.opacity - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_visualization_exploding() {
        let mut m = meta();
        m.gradient = GradientStatus::Exploding(50.0);
        let vis = TensorVisualization::from_metadata(&m);
        assert_eq!(vis.cable_color, CableColor::Red);
        assert_eq!(vis.animation_pattern, AnimationPattern::Pulsing);
    }

    #[test]
    fn test_visualization_vanishing() {
        let mut m = meta();
        m.gradient = GradientStatus::Vanishing(1e-10);
        let vis = TensorVisualization::from_metadata(&m);
        assert_eq!(vis.cable_color, CableColor::Red);
        assert_eq!(vis.animation_pattern, AnimationPattern::Pulsing);
    }

    #[test]
    fn test_visualization_saturated() {
        let mut m = meta();
        m.gradient = GradientStatus::Saturated;
        let vis = TensorVisualization::from_metadata(&m);
        assert_eq!(vis.cable_color, CableColor::Yellow);
        assert_eq!(vis.animation_pattern, AnimationPattern::Dashed);
    }

    #[test]
    fn test_visualization_frozen() {
        let mut m = meta();
        m.is_frozen = true;
        let vis = TensorVisualization::from_metadata(&m);
        assert_eq!(vis.cable_color, CableColor::Blue);
        assert!((vis.opacity - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_color_to_hex() {
        assert_eq!(color_to_hex(CableColor::Green), "#22C55E");
        assert_eq!(color_to_hex(CableColor::Red), "#EF4444");
        assert_eq!(color_to_hex(CableColor::Blue), "#3B82F6");
        assert_eq!(color_to_hex(CableColor::Gray), "#6B7280");
        assert_eq!(color_to_hex(CableColor::Yellow), "#EAB308");
    }
}
