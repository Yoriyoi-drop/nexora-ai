use std::collections::HashMap;
use std::path::Path;

use ndarray::ArrayD;

use crate::FoundationResult;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorEntry {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SafetensorsHeader {
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

fn f16_bytes_to_f32_slice(bytes: &[u8]) -> Vec<f32> {
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
pub fn save_safetensors_with_dtype(
    path: impl AsRef<Path>,
    tensors: &[(&str, ArrayD<f32>)],
    dtype: SaveDtype,
) -> FoundationResult<()> {
    let mut header_map = HashMap::new();
    let mut data_bytes: Vec<u8> = Vec::new();
    let mut offset: usize = 0;

    let (dtype_str, elem_size) = match dtype {
        SaveDtype::F32 => ("F32", 4),
        SaveDtype::F16 => ("F16", 2),
    };

    for (name, arr) in tensors {
        let flat: Vec<u8> = match dtype {
            SaveDtype::F32 => arr.iter().flat_map(|v| v.to_le_bytes()).collect(),
            SaveDtype::F16 => f32_slice_to_f16_bytes(arr.as_slice().unwrap_or(&[]) as &[f32]),
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
        tensors: header_map,
    };
    let header_json = serde_json::to_string(&header_obj).map_err(|e| {
        crate::FoundationError::Configuration(format!("JSON serialize header: {}", e))
    })?;
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut out = Vec::with_capacity(8 + header_bytes.len() + data_bytes.len());
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(header_bytes);
    out.extend_from_slice(&data_bytes);

    std::fs::write(path.as_ref(), &out).map_err(|e| {
        crate::FoundationError::Resource(format!("Write file {}: {}", path.as_ref().display(), e))
    })?;

    Ok(())
}

/// Save tensors as F32 safetensors (backward compatible).
pub fn save_safetensors(
    path: impl AsRef<Path>,
    tensors: &[(&str, ArrayD<f32>)],
) -> FoundationResult<()> {
    save_safetensors_with_dtype(path, tensors, SaveDtype::F32)
}

/// Save tensors as F16 safetensors (2× smaller files).
pub fn save_safetensors_f16(
    path: impl AsRef<Path>,
    tensors: &[(&str, ArrayD<f32>)],
) -> FoundationResult<()> {
    save_safetensors_with_dtype(path, tensors, SaveDtype::F16)
}

/// Load safetensors into f32 tensors. Supports both F32 and F16 dtypes.
pub fn load_safetensors(
    path: impl AsRef<Path>,
) -> FoundationResult<HashMap<String, ArrayD<f32>>> {
    let raw = std::fs::read(path.as_ref()).map_err(|e| {
        crate::FoundationError::Resource(format!("Read file {}: {}", path.as_ref().display(), e))
    })?;

    if raw.len() < 8 {
        return Err(crate::FoundationError::Configuration(
            "File too small: safetensors requires at least 8 bytes".into(),
        ));
    }

    let header_len = usize::try_from(u64::from_le_bytes([
        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
    ]))
    .unwrap_or(usize::MAX);

    let header_end = 8 + header_len;
    if header_end > raw.len() {
        return Err(crate::FoundationError::Configuration(format!(
            "Header length {} exceeds file size {}",
            header_len,
            raw.len()
        )));
    }

    let header_json = std::str::from_utf8(&raw[8..header_end]).map_err(|e| {
        crate::FoundationError::Configuration(format!("Invalid UTF-8 in header: {}", e))
    })?;

    let header: SafetensorsHeader = serde_json::from_str(header_json).map_err(|e| {
        crate::FoundationError::Configuration(format!("Invalid JSON header: {}", e))
    })?;

    let mut result = HashMap::new();
    for (name, entry) in &header.tensors {
        let start = 8 + header_len + entry.data_offsets[0];
        let end = 8 + header_len + entry.data_offsets[1];
        if end > raw.len() {
            return Err(crate::FoundationError::Configuration(format!(
                "Data offset [{}, {}] out of range for tensor '{}'",
                entry.data_offsets[0], entry.data_offsets[1], name
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
            "F16" => {
                f16_bytes_to_f32_slice(bytes)
            }
            other => {
                return Err(crate::FoundationError::Configuration(format!(
                    "Unsupported dtype '{}' for tensor '{}' (supported: F32, F16)",
                    other, name
                )));
            }
        };

        let arr = ArrayD::from_shape_vec(entry.shape.clone(), floats).map_err(|e| {
            crate::FoundationError::Configuration(format!(
                "Shape mismatch for tensor '{}': {}",
                name, e
            ))
        })?;
        result.insert(name.clone(), arr);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr0;
    use std::io::Write;

    #[test]
    fn test_tensor_entry_serde() {
        let entry = TensorEntry {
            dtype: "F32".to_string(),
            shape: vec![2, 3],
            data_offsets: [0, 24],
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: TensorEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back.dtype, "F32");
        assert_eq!(back.shape, vec![2, 3]);
    }

    #[test]
    fn test_safetensors_header_serde() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "weight".to_string(),
            TensorEntry {
                dtype: "F32".to_string(),
                shape: vec![3, 3],
                data_offsets: [0, 36],
            },
        );
        let header = SafetensorsHeader { tensors };
        let json = serde_json::to_string(&header).unwrap();
        let back: SafetensorsHeader = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tensors.len(), 1);
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let dir = std::env::temp_dir().join("safetensors_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.safetensors");

        let tensor =
            ArrayD::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        save_safetensors(&path, &[("weights", tensor)]).unwrap();

        let loaded = load_safetensors(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        let loaded_t = loaded.get("weights").unwrap();
        assert_eq!(loaded_t.shape(), &[2, 3]);
        assert!((loaded_t[ndarray::IxDyn(&[0, 0])] - 1.0).abs() < 1e-6);
        assert!((loaded_t[ndarray::IxDyn(&[1, 2])] - 6.0).abs() < 1e-6);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_and_load_multiple_tensors() {
        let dir = std::env::temp_dir().join("safetensors_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("multi.safetensors");

        let w = ArrayD::from_shape_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let b = ArrayD::from_shape_vec(vec![2], vec![0.1, 0.2]).unwrap();
        save_safetensors(&path, &[("weight", w), ("bias", b)]).unwrap();

        let loaded = load_safetensors(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains_key("weight"));
        assert!(loaded.contains_key("bias"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_invalid_file() {
        let result = load_safetensors("/tmp/nonexistent_safetensors_file_12345");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_too_small_file() {
        let dir = std::env::temp_dir().join("safetensors_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("small.safetensors");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"abcd").unwrap();
        drop(f);

        let result = load_safetensors(&path);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_empty_tensor_list() {
        let dir = std::env::temp_dir().join("safetensors_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("empty.safetensors");
        let result = save_safetensors(&path, &[]);
        assert!(result.is_ok());
        let loaded = load_safetensors(&path).unwrap();
        assert!(loaded.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_tensor_entry_debug() {
        let entry = TensorEntry {
            dtype: "F32".to_string(),
            shape: vec![1],
            data_offsets: [0, 4],
        };
        let _debug = format!("{:?}", entry);
    }

    #[test]
    fn test_safetensors_header_debug() {
        let header = SafetensorsHeader {
            tensors: HashMap::new(),
        };
        let _debug = format!("{:?}", header);
    }

    #[test]
    fn test_f16_roundtrip() {
        let val: f32 = 3.1415;
        let bits = f32_to_f16_bits(val);
        let back = f16_bits_to_f32(bits);
        let diff = (val - back).abs();
        assert!(diff < 0.01, "f16 roundtrip error: {diff} (expected < 0.01)");
    }

    #[test]
    fn test_f16_save_and_load() {
        let dir = std::env::temp_dir().join("safetensors_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("f16_test.safetensors");

        let tensor =
            ArrayD::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        save_safetensors_f16(&path, &[("weights", tensor)]).unwrap();

        let loaded = load_safetensors(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        let loaded_t = loaded.get("weights").unwrap();
        assert_eq!(loaded_t.shape(), &[2, 3]);
        assert!((loaded_t[ndarray::IxDyn(&[0, 0])] - 1.0).abs() < 0.01);
        assert!((loaded_t[ndarray::IxDyn(&[1, 2])] - 6.0).abs() < 0.01);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_f16_file_is_half_size() {
        let dir = std::env::temp_dir().join("safetensors_test");
        let _ = std::fs::create_dir_all(&dir);
        let f32_path = dir.join("f32_size.safetensors");
        let f16_path = dir.join("f16_size.safetensors");

        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let arr = ArrayD::from_shape_vec(vec![10, 10], data).unwrap();

        save_safetensors(&f32_path, &[("big", arr.clone())]).unwrap();
        save_safetensors_f16(&f16_path, &[("big", arr)]).unwrap();

        let f32_size = std::fs::metadata(&f32_path).unwrap().len();
        let f16_size = std::fs::metadata(&f16_path).unwrap().len();

        assert!(
            f16_size < f32_size,
            "F16 file ({f16_size}) should be smaller than F32 ({f32_size})"
        );

        let _ = std::fs::remove_file(&f32_path);
        let _ = std::fs::remove_file(&f16_path);
    }
}
