use crate::gnac::smart_tensor::SmartTensorMetadata;
use crate::gnac::canvas::GradientStatus;

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
    Green,   // gradien stabil
    Yellow,  // gradien mulai tidak stabil
    Red,     // exploding/vanishing gradient
    Blue,    // tensor frozen
    Gray,    // tensor tidak aktif
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnimationPattern {
    Solid,      // throughput rendah
    Dashed,     // throughput sedang
    Dotted,     // throughput tinggi
    Pulsing,    // gradien tidak stabil
}

impl TensorVisualization {
    pub fn from_metadata(meta: &SmartTensorMetadata) -> Self {
        let cable_thickness = (meta.bandwidth_estimate / 1000.0).clamp(1.0, 20.0) as f32;

        let cable_color = match meta.gradient {
            GradientStatus::Stable => {
                if meta.is_frozen { CableColor::Blue } else { CableColor::Green }
            }
            GradientStatus::Exploding(_) => CableColor::Red,
            GradientStatus::Vanishing(_) => CableColor::Red,
            GradientStatus::Saturated => CableColor::Yellow,
        };

        let animation_pattern = match meta.gradient {
            GradientStatus::Exploding(_) | GradientStatus::Vanishing(_) => AnimationPattern::Pulsing,
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
