use ndarray::Array2;
use rand::Rng;

/// LoRA weights for a single weight matrix in a transformer layer.
#[derive(Debug, Clone)]
pub struct LoraWeights {
    /// LoRA A matrix: [in_dim, rank]
    pub lora_a: Array2<f32>,
    /// LoRA B matrix: [rank, out_dim]
    pub lora_b: Array2<f32>,
    /// Scaling factor: alpha / rank
    pub scaling: f32,
}

impl LoraWeights {
    pub fn new(in_dim: usize, out_dim: usize, rank: usize, alpha: f32) -> Self {
        let mut rng = rand::thread_rng();
        // A: random normal (Kaiming init for A)
        let scale_a = (1.0 / in_dim as f32).sqrt();
        let lora_a = Array2::from_shape_fn((in_dim, rank), |_| {
            rng.gen::<f32>() * 2.0 * scale_a - scale_a
        });
        // B: zeros
        let lora_b = Array2::zeros((rank, out_dim));
        let scaling = alpha / rank as f32;
        Self { lora_a, lora_b, scaling }
    }

    /// Apply LoRA: output += (x @ A) @ B * scaling
    /// x: [batch, in_dim]
    /// Returns: [batch, out_dim]
    pub fn apply(&self, x: &Array2<f32>) -> Array2<f32> {
        let hidden = x.dot(&self.lora_a);
        let out = hidden.dot(&self.lora_b);
        out * self.scaling
    }
}

/// LoRA adapters for a single transformer block layer.
/// Each weight matrix in the layer can have its own LoRA adapter.
#[derive(Debug, Clone)]
pub struct LayerLoRA {
    pub layer_idx: usize,
    /// Attention Q projection LoRA
    pub q: Option<LoraWeights>,
    /// Attention K projection LoRA
    pub k: Option<LoraWeights>,
    /// Attention V projection LoRA
    pub v: Option<LoraWeights>,
    /// Attention O projection LoRA
    pub o: Option<LoraWeights>,
    /// FFN W1 (gate) LoRA
    pub w1: Option<LoraWeights>,
    /// FFN W2 (down) LoRA
    pub w2: Option<LoraWeights>,
    /// FFN W3 (up) LoRA
    pub w3: Option<LoraWeights>,
}

impl LayerLoRA {
    pub fn new(layer_idx: usize) -> Self {
        Self {
            layer_idx,
            q: None,
            k: None,
            v: None,
            o: None,
            w1: None,
            w2: None,
            w3: None,
        }
    }

    /// Total trainable parameters in this layer's adapters.
    pub fn trainable_params(&self) -> usize {
        let sum = |w: &Option<LoraWeights>| -> usize {
            w.as_ref().map_or(0, |l| l.lora_a.len() + l.lora_b.len())
        };
        sum(&self.q) + sum(&self.k) + sum(&self.v) + sum(&self.o)
            + sum(&self.w1) + sum(&self.w2) + sum(&self.w3)
    }
}

/// Save LoRA adapters to safetensors format.
/// Each adapter weight is stored as `lora.{layer_idx}.{weight_name}.{a|b}`.
pub fn save_lora_adapters(
    adapters: &[LayerLoRA],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut tensors: Vec<String> = Vec::new();
    let mut arrays: Vec<ndarray::ArrayD<f32>> = Vec::new();

    for layer in adapters {
        let save = |name: &str, w: &Option<LoraWeights>, keys: &mut Vec<String>, vals: &mut Vec<ndarray::ArrayD<f32>>| {
            if let Some(ref lw) = w {
                keys.push(format!("lora.{}.{}.a", layer.layer_idx, name));
                vals.push(lw.lora_a.clone().into_dyn());
                keys.push(format!("lora.{}.{}.b", layer.layer_idx, name));
                vals.push(lw.lora_b.clone().into_dyn());
            }
        };
        save("q", &layer.q, &mut tensors, &mut arrays);
        save("k", &layer.k, &mut tensors, &mut arrays);
        save("v", &layer.v, &mut tensors, &mut arrays);
        save("o", &layer.o, &mut tensors, &mut arrays);
        save("w1", &layer.w1, &mut tensors, &mut arrays);
        save("w2", &layer.w2, &mut tensors, &mut arrays);
        save("w3", &layer.w3, &mut tensors, &mut arrays);
    }

    let refs: Vec<(&str, ndarray::ArrayD<f32>)> = tensors.iter()
        .zip(arrays.into_iter())
        .map(|(k, v)| (k.as_str(), v))
        .collect();

    let mut meta = std::collections::HashMap::new();
    meta.insert("format".to_string(), "lora_peft_v1".to_string());

    crate::safetensors::save_safetensors_with_meta(
        path,
        &refs,
        crate::safetensors::SaveDtype::F32,
        Some(meta),
    )?;
    Ok(())
}

/// Load LoRA adapters from safetensors.
pub fn load_lora_adapters(
    path: &str,
    num_layers: usize,
) -> Result<Vec<LayerLoRA>, Box<dyn std::error::Error>> {
    let loaded = crate::safetensors::load_safetensors(path)?;

    let mut adapters: Vec<LayerLoRA> = (0..num_layers)
        .map(|i| LayerLoRA::new(i))
        .collect();

    for (key, tensor) in loaded.iter() {
        // Parse key format: "lora.{layer_idx}.{weight_name}.{a|b}"
        let parts: Vec<&str> = key.split('.').collect();
        if parts.len() != 4 || parts[0] != "lora" {
            continue;
        }
        let layer_idx: usize = match parts[1].parse() {
            Ok(i) => i,
            Err(_) => continue,
        };
        if layer_idx >= num_layers {
            continue;
        }
        let weight_name = parts[2];
        let ab = parts[3];

        let arr_2d = tensor.clone().into_dimensionality::<ndarray::Ix2>()?;

        let set_ab = |target: &mut Option<LoraWeights>, ab: &str, arr: Array2<f32>| {
            let lw = target.get_or_insert_with(|| {
                let (in_dim, out_dim) = if ab == "a" {
                    (arr.shape()[0], arr.shape()[1])
                } else {
                    (arr.shape()[1], arr.shape()[0])
                };
                LoraWeights {
                    lora_a: Array2::zeros((in_dim, 0)),
                    lora_b: Array2::zeros((0, out_dim)),
                    scaling: 1.0,
                }
            });
            if ab == "a" {
                lw.lora_a = arr;
            } else {
                lw.lora_b = arr;
            }
        };

        match weight_name {
            "q" => set_ab(&mut adapters[layer_idx].q, ab, arr_2d),
            "k" => set_ab(&mut adapters[layer_idx].k, ab, arr_2d),
            "v" => set_ab(&mut adapters[layer_idx].v, ab, arr_2d),
            "o" => set_ab(&mut adapters[layer_idx].o, ab, arr_2d),
            "w1" => set_ab(&mut adapters[layer_idx].w1, ab, arr_2d),
            "w2" => set_ab(&mut adapters[layer_idx].w2, ab, arr_2d),
            "w3" => set_ab(&mut adapters[layer_idx].w3, ab, arr_2d),
            _ => {}
        }
    }

    Ok(adapters)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_weights_create() {
        let lw = LoraWeights::new(64, 128, 8, 16.0);
        assert_eq!(lw.lora_a.dim(), (64, 8));
        assert_eq!(lw.lora_b.dim(), (8, 128));
        assert!((lw.scaling - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_lora_apply_shape() {
        let lw = LoraWeights::new(64, 128, 8, 16.0);
        let x = Array2::ones((4, 64));
        let out = lw.apply(&x);
        assert_eq!(out.dim(), (4, 128));
    }

    #[test]
    fn test_layer_lora_trainable_params() {
        let mut layer = LayerLoRA::new(0);
        layer.q = Some(LoraWeights::new(64, 64, 8, 16.0));
        layer.v = Some(LoraWeights::new(64, 64, 8, 16.0));
        // q: 64*8 + 8*64 = 1024, v: same = 1024, total = 2048
        assert_eq!(layer.trainable_params(), 2048);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let path = "/tmp/test_lora_peft.safetensors";
        let _ = std::fs::remove_file(path);

        let mut adapters: Vec<LayerLoRA> = (0..2).map(|i| LayerLoRA::new(i)).collect();
        adapters[0].q = Some(LoraWeights::new(64, 64, 4, 8.0));
        adapters[1].v = Some(LoraWeights::new(64, 64, 4, 8.0));

        save_lora_adapters(&adapters, path).unwrap();
        let loaded = load_lora_adapters(path, 2).unwrap();

        assert_eq!(loaded.len(), 2);
        assert!(loaded[0].q.is_some());
        assert!(loaded[0].v.is_none());
        assert!(loaded[1].v.is_some());

        let _ = std::fs::remove_file(path);
    }
}
