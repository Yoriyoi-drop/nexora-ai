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
    #[serde(flatten)]
    pub tensors: HashMap<String, TensorEntry>,
}

pub fn save_safetensors<S: AsRef<str>>(
    path: impl AsRef<Path>,
    tensors: &[(S, ArrayD<f32>)],
) -> TransformerResult<()> {
    let mut header_map = HashMap::new();
    let mut data_bytes: Vec<u8> = Vec::new();
    let mut offset: usize = 0;

    for (name, arr) in tensors {
        let name = name.as_ref();
        let flat: Vec<u8> = arr.iter().flat_map(|v| v.to_le_bytes()).collect();
        let len = flat.len();

        header_map.insert(
            name.to_string(),
            TensorEntry {
                dtype: "F32".to_string(),
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

pub fn load_safetensors(path: impl AsRef<Path>) -> TransformerResult<HashMap<String, ArrayD<f32>>> {
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
    for (name, entry) in &header.tensors {
        if entry.dtype != "F32" {
            return Err(crate::TransformerError::Implementation(format!(
                "Unsupported dtype: {} for tensor {}",
                entry.dtype, name
            )));
        }
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
        let mut floats = Vec::with_capacity(total);
        for chunk in bytes.chunks_exact(4) {
            floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        let arr = ArrayD::from_shape_vec(entry.shape.clone(), floats)
            .map_err(|e| crate::TransformerError::Implementation(format!("Shape error: {}", e)))?;
        result.insert(name.clone(), arr);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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
}
