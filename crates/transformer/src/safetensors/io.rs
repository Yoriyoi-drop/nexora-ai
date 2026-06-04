use std::collections::HashMap;
use std::path::Path;

use ndarray::ArrayD;

use crate::TransformerResult;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorEntry {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SafetensorsHeader {
    /// Top-level metadata (safetensors convention: `__metadata__` key).
    /// Stores quantization format, training step, model config, etc.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub __metadata__: Option<HashMap<String, String>>,
    #[serde(flatten)]
    pub tensors: HashMap<String, TensorEntry>,
}

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp == 0 {
        sign | (mant >> 13) as u16
    } else if exp == 255 {
        sign | 0x7c00 | ((mant >> 13) as u16)
    } else {
        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            sign | 0x7c00 | ((mant >> 13) as u16)
        } else if new_exp <= 0 {
            sign | ((mant | 0x800000) >> (13 - new_exp + 1)) as u16
        } else {
            sign | ((new_exp as u16) << 10) | (mant >> 13) as u16
        }
    }
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign)
        } else {
            let lz = (mant as u16).leading_zeros() - 6;
            let k = 9 - lz;
            let low_mask = (1u32 << k) - 1;
            let frac = mant & low_mask;
            let f32_mant = frac << (23 - k);
            let f32_exp = 112 - lz;
            f32::from_bits(sign | (f32_exp << 23) | f32_mant)
        }
    } else if exp == 31 {
        f32::from_bits(sign | (255u32 << 23) | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
    }
}

fn f32_slice_to_f16_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for &v in data {
        out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
    }
    out
}

pub fn f16_bytes_to_f32_slice(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

pub enum SaveDtype {
    F32,
    F16,
}

/// Save tensors as safetensors, with configurable dtype.
pub fn save_safetensors<S: AsRef<str>>(
    path: impl AsRef<Path>,
    tensors: &[(S, ArrayD<f32>)],
) -> TransformerResult<()> {
    save_safetensors_with_dtype(path, tensors, SaveDtype::F32)
}

/// Save tensors as F16 safetensors (2× smaller files).
pub fn save_safetensors_f16<S: AsRef<str>>(
    path: impl AsRef<Path>,
    tensors: &[(S, ArrayD<f32>)],
) -> TransformerResult<()> {
    save_safetensors_with_dtype(path, tensors, SaveDtype::F16)
}

fn save_safetensors_with_dtype<S: AsRef<str>>(
    path: impl AsRef<Path>,
    tensors: &[(S, ArrayD<f32>)],
    dtype: SaveDtype,
) -> TransformerResult<()> {
    save_safetensors_with_meta(path, tensors, dtype, None)
}

pub fn save_safetensors_with_meta<S: AsRef<str>>(
    path: impl AsRef<Path>,
    tensors: &[(S, ArrayD<f32>)],
    dtype: SaveDtype,
    metadata: Option<HashMap<String, String>>,
) -> TransformerResult<()> {
    let mut header_map = HashMap::new();
    let mut data_bytes: Vec<u8> = Vec::new();
    let mut offset: usize = 0;

    let (dtype_str, _elem_size) = match dtype {
        SaveDtype::F32 => ("F32", 4),
        SaveDtype::F16 => ("F16", 2),
    };

    for (name, arr) in tensors {
        let name = name.as_ref();
        let flat: Vec<u8> = match dtype {
            SaveDtype::F32 => arr.iter().flat_map(|v| v.to_le_bytes()).collect(),
            SaveDtype::F16 => f32_slice_to_f16_bytes(arr.as_slice().unwrap_or(&[])),
        };
        let len = flat.len();

        header_map.insert(
            name.to_string(),
            TensorEntry {
                dtype: dtype_str.to_string(),
                shape: arr.shape().to_vec(),
                data_offsets: [offset, offset + len],
            },
        );

        data_bytes.extend_from_slice(&flat);
        offset += len;
    }

    let header_obj = SafetensorsHeader {
        __metadata__: metadata,
        tensors: header_map,
    };
    let header_json = serde_json::to_string(&header_obj)
        .map_err(|e| crate::TransformerError::Implementation(format!("JSON serialize: {}", e)))?;
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut out = Vec::with_capacity(8 + header_bytes.len() + data_bytes.len());
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(header_bytes);
    out.extend_from_slice(&data_bytes);

    std::fs::write(path.as_ref(), &out)?;

    Ok(())
}

/// Load safetensors into f32 tensors plus metadata.
/// Supports both F32 and F16 dtypes. Metadata is optional (None for legacy files).
pub fn load_safetensors_with_meta(
    path: impl AsRef<Path>,
) -> TransformerResult<(
    HashMap<String, ArrayD<f32>>,
    Option<HashMap<String, String>>,
)> {
    let raw = std::fs::read(path.as_ref())?;

    if raw.len() < 8 {
        return Err(crate::TransformerError::Implementation(
            "File too small".into(),
        ));
    }

    let header_len = u64::from_le_bytes([
        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
    ]) as usize;

    let header_end = 8 + header_len;
    if header_end > raw.len() {
        return Err(crate::TransformerError::Implementation(
            "Header exceeds file size".into(),
        ));
    }

    let header_json = std::str::from_utf8(&raw[8..header_end])
        .map_err(|e| crate::TransformerError::Implementation(format!("UTF-8 error: {}", e)))?;

    let header: SafetensorsHeader = serde_json::from_str(header_json)
        .map_err(|e| crate::TransformerError::Implementation(format!("JSON parse: {}", e)))?;

    let mut result = HashMap::new();
    for (name, entry) in header.tensors {
        let start = 8 + header_len + entry.data_offsets[0];
        let end = 8 + header_len + entry.data_offsets[1];
        if end > raw.len() {
            return Err(crate::TransformerError::Implementation(format!(
                "Data offset out of range for tensor {}",
                name
            )));
        }
        let bytes = &raw[start..end];
        let total: usize = entry.shape.iter().product();

        let floats = match entry.dtype.as_str() {
            "F32" => {
                let mut v = Vec::with_capacity(total);
                for chunk in bytes.chunks_exact(4) {
                    v.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                v
            }
            "F16" => f16_bytes_to_f32_slice(bytes),
            other => {
                return Err(crate::TransformerError::Implementation(format!(
                    "Unsupported dtype '{}' for tensor '{}' (supported: F32, F16)",
                    other, name
                )));
            }
        };

        let arr = ArrayD::from_shape_vec(entry.shape, floats)
            .map_err(|e| crate::TransformerError::Implementation(format!("Shape error: {}", e)))?;
        result.insert(name, arr);
    }

    Ok((result, header.__metadata__))
}

/// Load safetensors into f32 tensors (metadata is discarded).
pub fn load_safetensors(
    path: impl AsRef<Path>,
) -> TransformerResult<HashMap<String, ArrayD<f32>>> {
    load_safetensors_with_meta(path).map(|(tensors, _meta)| tensors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_f16_roundtrip() {
        let val: f32 = 3.1415;
        let bits = f32_to_f16_bits(val);
        let back = f16_bits_to_f32(bits);
        let diff = (val - back).abs();
        assert!(diff < 0.01, "f16 roundtrip error: {diff} (expected < 0.01)");
    }

    #[test]
    fn test_f16_save_load_roundtrip() {
        let path = "/tmp/test_f16_safetensors.safetensors";
        let _ = std::fs::remove_file(path);

        let t1: ArrayD<f32> = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let t2: ArrayD<f32> = array![5.0, 6.0, 7.0].into_dyn();
        save_safetensors_f16(path, &[("weight", t1.clone()), ("bias", t2.clone())]).unwrap();

        let loaded = load_safetensors(path).unwrap();
        assert_eq!(loaded.len(), 2);

        let w = loaded.get("weight").unwrap();
        assert_eq!(w.shape(), &[2, 2]);
        assert!((w[[0, 0]] - 1.0).abs() < 0.01);
        assert!((w[[1, 1]] - 4.0).abs() < 0.01);

        let b = loaded.get("bias").unwrap();
        assert_eq!(b.shape(), &[3]);
        assert!((b[[2]] - 7.0).abs() < 0.01);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_f16_file_is_half_size() {
        let path = "/tmp/test_f16_size.safetensors";
        let _ = std::fs::remove_file(path);

        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let arr = ArrayD::from_shape_vec(vec![10, 10], data).unwrap();

        save_safetensors(path, &[("big", arr.clone())]).unwrap();
        let f32_size = std::fs::metadata(path).unwrap().len();
        let _ = std::fs::remove_file(path);

        save_safetensors_f16(path, &[("big", arr.clone())]).unwrap();
        let f16_size = std::fs::metadata(path).unwrap().len();
        let _ = std::fs::remove_file(path);

        // F16 file should be roughly half the size (header is similar, data is 2× smaller)
        // Data portion: 100 * 4 = 400 bytes for F32, 100 * 2 = 200 bytes for F16
        assert!(
            f16_size < f32_size,
            "F16 file ({f16_size}) should be smaller than F32 ({f32_size})"
        );
    }

    #[test]
    fn test_save_load_roundtrip() {
        let path = "/tmp/test_safetensors_roundtrip.safetensors";
        let _ = std::fs::remove_file(path);

        let t1: ArrayD<f32> = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let t2: ArrayD<f32> = array![5.0, 6.0, 7.0].into_dyn();
        save_safetensors(path, &[("weight", t1.clone()), ("bias", t2.clone())]).unwrap();

        let loaded = load_safetensors(path).unwrap();
        assert_eq!(loaded.len(), 2);

        let w = loaded.get("weight").unwrap();
        assert_eq!(w.shape(), &[2, 2]);
        assert!((w[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((w[[1, 1]] - 4.0).abs() < 1e-6);

        let b = loaded.get("bias").unwrap();
        assert_eq!(b.shape(), &[3]);
        assert!((b[[2]] - 7.0).abs() < 1e-6);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_save_load_empty() {
        let path = "/tmp/test_safetensors_empty.safetensors";
        let _ = std::fs::remove_file(path);

        let tensors: Vec<(&str, ArrayD<f32>)> = vec![];
        save_safetensors(path, &tensors).unwrap();
        let loaded = load_safetensors(path).unwrap();
        assert!(loaded.is_empty());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_load_bad_file() {
        let path = "/tmp/test_safetensors_bad.safetensors";
        let _ = std::fs::remove_file(path);
        std::fs::write(path, &[0u8; 4]).unwrap();
        let result = load_safetensors(path);
        assert!(result.is_err());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_load_nonexistent() {
        let result = load_safetensors("/tmp/does_not_exist_12345.safetensors");
        assert!(result.is_err());
    }

    #[test]
    fn test_save_large_tensor_values() {
        let path = "/tmp/test_safetensors_large.safetensors";
        let _ = std::fs::remove_file(path);

        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let arr = ArrayD::from_shape_vec(vec![10, 10], data).unwrap();
        save_safetensors(path, &[("big", arr.clone())]).unwrap();

        let loaded = load_safetensors(path).unwrap();
        let l = loaded.get("big").unwrap();
        assert_eq!(l.shape(), &[10, 10]);
        assert!((l[[9, 9]] - 99.0).abs() < 1e-6);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_save_load_multiple_tensors() {
        let path = "/tmp/test_safetensors_multi.safetensors";
        let _ = std::fs::remove_file(path);

        let mut tensors: Vec<(&str, ArrayD<f32>)> = Vec::new();
        for i in 0..5 {
            let arr = ArrayD::from_shape_vec(vec![3], vec![i as f32 * 10.0; 3]).unwrap();
            tensors.push((Box::leak(format!("t{}", i).into_boxed_str()), arr));
        }
        save_safetensors(path, &tensors).unwrap();

        let loaded = load_safetensors(path).unwrap();
        assert_eq!(loaded.len(), 5);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_metadata_roundtrip() {
        let path = "/tmp/test_safetensors_meta.safetensors";
        let _ = std::fs::remove_file(path);

        let t1: ArrayD<f32> = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let mut meta = std::collections::HashMap::new();
        meta.insert("quantization".to_string(), "F16".to_string());
        meta.insert("step".to_string(), "42".to_string());

        save_safetensors_with_meta(
            path,
            &[("weight", t1.clone())],
            SaveDtype::F32,
            Some(meta.clone()),
        )
        .unwrap();

        let (loaded, loaded_meta) = load_safetensors_with_meta(path).unwrap();

        // Tensors intact
        assert_eq!(loaded.len(), 1);
        let w = loaded.get("weight").unwrap();
        assert_eq!(w.shape(), &[2, 2]);
        assert!((w[[0, 0]] - 1.0).abs() < 1e-6);

        // Metadata preserved
        let loaded_meta = loaded_meta.expect("metadata should be present");
        assert_eq!(loaded_meta.get("quantization").unwrap(), "F16");
        assert_eq!(loaded_meta.get("step").unwrap(), "42");

        // Backward compat: load_safetensors ignores metadata
        let tensors_only = load_safetensors(path).unwrap();
        assert_eq!(tensors_only.len(), 1);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_metadata_old_file_no_meta() {
        // Files saved without metadata should still load fine
        let path = "/tmp/test_safetensors_no_meta.safetensors";
        let _ = std::fs::remove_file(path);

        let t1: ArrayD<f32> = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        save_safetensors(path, &[("weight", t1.clone())]).unwrap();

        let (loaded, loaded_meta) = load_safetensors_with_meta(path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert!(loaded_meta.is_none(), "old file should have no metadata");

        let _ = std::fs::remove_file(path);
    }
}
