//! Deep Learning Library for Nexora AI — hub crate
//!
//! Re-exports from sub-crates:
//! - `autograd`  — tape-based autograd engine (nexora-autograd)
//! - `star_x`    — STAR-X tensor ops (nexora-star-x)
//! - `gnac`      — GNAC graph composer (nexora-gnac)
//! - `echo_net`  — ECHO-Net Ω (nexora-echo-net)

use thiserror::Error;

/// Unified error type supporting conversions from all sub-crate error types.
#[derive(Error, Debug)]
pub enum DeepLearningError {
    #[error("{0}")]
    Autograd(#[from] nexora_autograd::DeepLearningError),
    #[error("{0}")]
    StarX(#[from] nexora_star_x::DeepLearningError),
    #[error("{0}")]
    Gnac(#[from] nexora_gnac::DeepLearningError),
    #[error("{0}")]
    EchoNet(#[from] nexora_echo_net::DeepLearningError),
    #[error("Configuration error: {reason}")]
    Configuration { reason: String },
}

impl From<ndarray::ShapeError> for DeepLearningError {
    fn from(err: ndarray::ShapeError) -> Self {
        DeepLearningError::Autograd(nexora_autograd::DeepLearningError::from(err))
    }
}

pub type DLResult<T> = Result<T, DeepLearningError>;

pub mod autograd {
    pub use nexora_autograd::*;
}
pub mod star_x {
    pub use nexora_star_x::*;
}
pub mod gnac {
    pub use nexora_gnac::*;
}
pub mod echo_net {
    pub use nexora_echo_net::*;
}

#[cfg(feature = "gpu")]
pub use nexora_autograd::gpu;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deep_learning_error_display() {
        let err = DeepLearningError::Configuration {
            reason: "test".to_string(),
        };
        assert_eq!(format!("{}", err), "Configuration error: test");
    }

    #[test]
    fn test_deep_learning_error_from_autograd() {
        let _err: DeepLearningError = nexora_autograd::DeepLearningError::ShapeMismatch {
            expected: vec![2, 3],
            actual: vec![3, 2],
        }
        .into();
    }

    #[test]
    fn test_deep_learning_error_from_star_x() {
        let _err: DeepLearningError = nexora_star_x::DeepLearningError::ShapeMismatch {
            expected: vec![2, 3],
            actual: vec![3, 2],
        }
        .into();
    }

    #[test]
    fn test_deep_learning_error_from_gnac() {
        let _err: DeepLearningError = nexora_gnac::DeepLearningError::Computation {
            reason: "test".to_string(),
        }
        .into();
    }

    #[test]
    fn test_deep_learning_error_from_echo_net() {
        let _err: DeepLearningError = nexora_echo_net::DeepLearningError::ShapeMismatch {
            expected: vec![2, 3],
            actual: vec![3, 2],
        }
        .into();
    }

    #[test]
    fn test_dl_result_type() {
        let ok: DLResult<i32> = Ok(42);
        assert_eq!(ok.unwrap(), 42);

        let err: DLResult<i32> = Err(DeepLearningError::Configuration {
            reason: "fail".to_string(),
        });
        assert!(err.is_err());
    }

    #[test]
    fn test_autograd_module_re_export() {
        let _ = autograd::Tensor::new(ndarray::arr1(&[1.0, 2.0, 3.0]).into_dyn());
    }

    #[test]
    fn test_star_x_module_re_export() {
        let _ = star_x::StarXConfig::default();
    }

    #[test]
    fn test_gnac_module_re_export() {
        let _ = gnac::GnacConfig::default();
    }

    #[test]
    fn test_echo_net_module_re_export() {
        use echo_net::EchoNetConfig;
        let _config = EchoNetConfig::default();
    }

    #[test]
    fn test_error_is_std_error() {
        fn assert_std_error<E: std::error::Error>() {}
        assert_std_error::<DeepLearningError>();
    }
}
