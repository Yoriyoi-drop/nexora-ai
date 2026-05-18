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

pub fn save_safetensors(path: impl AsRef<Path>, tensors: &[(&str, ArrayD<f32>)]) -> TransformerResult<()> {
    let mut header_map = HashMap::new();
    let mut data_bytes: Vec<u8> = Vec::new();
    let mut offset: usize = 0;

    for (name, arr) in tensors {
        let flat: Vec<u8> = arr.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let len = flat.len();

        header_map.insert(name.to_string(), TensorEntry {
            dtype: "F32".to_string(),
            shape: arr.shape().to_vec(),
            data_offsets: [offset, offset + len],
        });

        data_bytes.extend_from_slice(&flat);
        offset += len;
    }

    let header_obj = SafetensorsHeader { tensors: header_map };
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
        return Err(crate::TransformerError::Implementation("File too small".into()));
    }

    let header_len = u64::from_le_bytes([
        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
    ]) as usize;

    let header_end = 8 + header_len;
    if header_end > raw.len() {
        return Err(crate::TransformerError::Implementation("Header exceeds file size".into()));
    }

    let header_json = std::str::from_utf8(&raw[8..header_end])
        .map_err(|e| crate::TransformerError::Implementation(format!("UTF-8 error: {}", e)))?;

    let header: SafetensorsHeader = serde_json::from_str(header_json)
        .map_err(|e| crate::TransformerError::Implementation(format!("JSON parse: {}", e)))?;

    let mut result = HashMap::new();
    for (name, entry) in &header.tensors {
        if entry.dtype != "F32" {
            return Err(crate::TransformerError::Implementation(
                format!("Unsupported dtype: {} for tensor {}", entry.dtype, name)
            ));
        }
        let start = 8 + header_len + entry.data_offsets[0];
        let end = 8 + header_len + entry.data_offsets[1];
        if end > raw.len() {
            return Err(crate::TransformerError::Implementation(
                format!("Data offset out of range for tensor {}", name)
            ));
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
