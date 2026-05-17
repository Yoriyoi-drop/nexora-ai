use std::fmt;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Compression {
    None,
    Zstd,
    Lz4,
}

impl Compression {
    pub fn from_ext(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "zst" | "zstd" => Compression::Zstd,
            "lz4" => Compression::Lz4,
            _ => Compression::None,
        }
    }

    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        match self {
            Compression::None => Ok(data.to_vec()),
            Compression::Zstd => Self::decompress_zstd(data),
            Compression::Lz4 => Self::decompress_lz4(data),
        }
    }

    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        match self {
            Compression::None => Ok(data.to_vec()),
            Compression::Zstd => Self::compress_zstd(data),
            Compression::Lz4 => Self::compress_lz4(data),
        }
    }

    #[cfg(feature = "compression-zstd")]
    fn decompress_zstd(data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        zstd::decode_all(data).map_err(|e| CompressionError::Zstd(e.to_string()))
    }

    #[cfg(not(feature = "compression-zstd"))]
    fn decompress_zstd(_data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        Err(CompressionError::Unsupported("zstd".into()))
    }

    #[cfg(feature = "compression-zstd")]
    fn compress_zstd(data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        zstd::encode_all(data, 3).map_err(|e| CompressionError::Zstd(e.to_string()))
    }

    #[cfg(not(feature = "compression-zstd"))]
    fn compress_zstd(_data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        Err(CompressionError::Unsupported("zstd".into()))
    }

    #[cfg(feature = "compression-lz4")]
    fn decompress_lz4(data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        lz4_flex::decompress_size_prepended(data).map_err(|e| CompressionError::Lz4(e.to_string()))
    }

    #[cfg(not(feature = "compression-lz4"))]
    fn decompress_lz4(_data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        Err(CompressionError::Unsupported("lz4".into()))
    }

    #[cfg(feature = "compression-lz4")]
    fn compress_lz4(data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        Ok(lz4_flex::compress_prepend_size(data))
    }

    #[cfg(not(feature = "compression-lz4"))]
    fn compress_lz4(_data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        Err(CompressionError::Unsupported("lz4".into()))
    }

}

impl fmt::Display for Compression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Compression::None => write!(f, "none"),
            Compression::Zstd => write!(f, "zstd"),
            Compression::Lz4 => write!(f, "lz4"),
        }
    }
}

#[derive(Debug)]
pub enum CompressionError {
    Zstd(String),
    Lz4(String),
    Unsupported(String),
    Io(String),
}

impl std::fmt::Display for CompressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompressionError::Zstd(msg) => write!(f, "Zstd error: {}", msg),
            CompressionError::Lz4(msg) => write!(f, "Lz4 error: {}", msg),
            CompressionError::Unsupported(codec) => write!(f, "Unsupported compression: {}", codec),
            CompressionError::Io(msg) => write!(f, "IO error: {}", msg),
        }
    }
}

impl std::error::Error for CompressionError {}
