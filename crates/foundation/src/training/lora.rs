use ndarray::{Array1, Array2};
use nexora_deeplearning::autograd::{Tensor, TensorOps};

#[derive(Debug, Clone)]
pub struct LoRAConfig {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub target_modules: Vec<String>,
    pub lr_scale: f32,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.1,
            target_modules: vec![
                "wq".into(), "wk".into(), "wv".into(), "wo".into(),
            ],
            lr_scale: 1.0,
        }
    }
}

impl LoRAConfig {
    pub fn with_rank(mut self, r: usize) -> Self {
        self.rank = r;
        self
    }

    pub fn with_alpha(mut self, a: f32) -> Self {
        self.alpha = a;
        self
    }

    pub fn with_target(mut self, modules: Vec<&str>) -> Self {
        self.target_modules = modules.into_iter().map(String::from).collect();
        self
    }
}

#[derive(Debug, Clone)]
pub struct LoRALayer {
    pub rank: usize,
    pub alpha: f32,
    pub scaling: f32,
    pub lora_a: Tensor,
    pub lora_b: Tensor,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl LoRALayer {
    pub fn new(rank: usize, alpha: f32, input_dim: usize, output_dim: usize) -> Self {
        let scale = (input_dim as f32).sqrt().recip();
        let lora_a = Tensor::from_slice(
            &(0..rank * input_dim)
                .map(|_| rand::random::<f32>() * 2.0 * scale - scale)
                .collect::<Vec<_>>(),
            &[rank, input_dim],
        );
        lora_a.set_requires_grad(true);

        let lora_b = Tensor::from_slice(
            &(0..output_dim * rank)
                .map(|_| 0.0f32)
                .collect::<Vec<_>>(),
            &[output_dim, rank],
        );
        lora_b.set_requires_grad(true);

        Self {
            rank,
            alpha,
            scaling: alpha / rank as f32,
            lora_a,
            lora_b,
            input_dim,
            output_dim,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        if self.dropout_enabled(0.0) {
            let hidden = x.matmul(&self.lora_a.transpose());
            let dropped = hidden.dropout(0.1, false);
            let out = dropped.matmul(&self.lora_b.transpose());
            out.mul(&Tensor::from_slice(&[self.scaling], &[1]))
        } else {
            let hidden = x.matmul(&self.lora_a.transpose());
            let out = hidden.matmul(&self.lora_b.transpose());
            out.mul(&Tensor::from_slice(&[self.scaling], &[1]))
        }
    }

    fn dropout_enabled(&self, _p: f32) -> bool {
        false
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.lora_a.clone(), self.lora_b.clone()]
    }

    pub fn merge_weights(&self, original: &mut Array2<f32>) {
        let a_data = self.lora_a.data();
        let b_data = self.lora_b.data();
        let a_2d = a_data.into_dimensionality::<ndarray::Ix2>()
            .expect("lora_a must be 2D");
        let b_2d = b_data.into_dimensionality::<ndarray::Ix2>()
            .expect("lora_b must be 2D");
        let delta = b_2d.dot(&a_2d).mapv(|v| v * self.scaling);
        for i in 0..original.shape()[0] {
            for j in 0..original.shape()[1] {
                original[[i, j]] += delta[[i, j]];
            }
        }
    }
}

pub struct LoRAModel {
    pub layers: Vec<(String, LoRALayer)>,
    pub config: LoRAConfig,
}

impl LoRAModel {
    pub fn new(config: LoRAConfig) -> Self {
        Self {
            layers: Vec::new(),
            config,
        }
    }

    pub fn add_layer(&mut self, name: String, input_dim: usize, output_dim: usize) {
        let layer = LoRALayer::new(
            self.config.rank,
            self.config.alpha,
            input_dim,
            output_dim,
        );
        self.layers.push((name, layer));
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        self.layers.iter()
            .flat_map(|(_, l)| l.parameters())
            .collect()
    }

    pub fn parameter_count(&self) -> usize {
        self.layers.iter()
            .map(|(_, l)| l.rank * (l.input_dim + l.output_dim))
            .sum()
    }

    pub fn merge_all(&self, model: &mut crate::models::transformer::CausalLM) {
        for (name, layer) in &self.layers {
            let parts: Vec<&str> = name.split('.').collect();
            if parts.len() < 2 { continue; }
            let layer_idx: usize = match parts[0].parse() {
                Ok(i) => i,
                Err(_) => continue,
            };
            if layer_idx >= model.blocks.len() { continue; }
            let target = parts[1];
            match target {
                "wq" => layer.merge_weights(&mut model.blocks[layer_idx].attention.wq),
                "wk" => layer.merge_weights(&mut model.blocks[layer_idx].attention.wk),
                "wv" => layer.merge_weights(&mut model.blocks[layer_idx].attention.wv),
                "wo" => layer.merge_weights(&mut model.blocks[layer_idx].attention.wo),
                "w1" => layer.merge_weights(&mut model.blocks[layer_idx].ffn.w1),
                "w2" => layer.merge_weights(&mut model.blocks[layer_idx].ffn.w2),
                "w3" => layer.merge_weights(&mut model.blocks[layer_idx].ffn.w3),
                _ => {}
            }
        }
    }

    pub fn unmerge_all(&self, model: &mut crate::models::transformer::CausalLM) {
        for (name, layer) in &self.layers {
            let parts: Vec<&str> = name.split('.').collect();
            if parts.len() < 2 { continue; }
            let layer_idx: usize = match parts[0].parse() {
                Ok(i) => i,
                Err(_) => continue,
            };
            if layer_idx >= model.blocks.len() { continue; }
            let target = parts[1];
            let a_data = layer.lora_a.data();
            let b_data = layer.lora_b.data();
            let a_2d = a_data.into_dimensionality::<ndarray::Ix2>().unwrap();
            let b_2d = b_data.into_dimensionality::<ndarray::Ix2>().unwrap();
            let delta = b_2d.dot(&a_2d).mapv(|v| v * layer.scaling);
            match target {
                "wq" => {
                    for i in 0..model.blocks[layer_idx].attention.wq.shape()[0] {
                        for j in 0..model.blocks[layer_idx].attention.wq.shape()[1] {
                            model.blocks[layer_idx].attention.wq[[i, j]] -= delta[[i, j]];
                        }
                    }
                }
                "wk" => {
                    for i in 0..model.blocks[layer_idx].attention.wk.shape()[0] {
                        for j in 0..model.blocks[layer_idx].attention.wk.shape()[1] {
                            model.blocks[layer_idx].attention.wk[[i, j]] -= delta[[i, j]];
                        }
                    }
                }
                "wv" => {
                    for i in 0..model.blocks[layer_idx].attention.wv.shape()[0] {
                        for j in 0..model.blocks[layer_idx].attention.wv.shape()[1] {
                            model.blocks[layer_idx].attention.wv[[i, j]] -= delta[[i, j]];
                        }
                    }
                }
                "wo" => {
                    for i in 0..model.blocks[layer_idx].attention.wo.shape()[0] {
                        for j in 0..model.blocks[layer_idx].attention.wo.shape()[1] {
                            model.blocks[layer_idx].attention.wo[[i, j]] -= delta[[i, j]];
                        }
                    }
                }
                _ => {}
            }
        }
    }

    pub fn save_adapter(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut tensors: Vec<(String, ndarray::ArrayD<f32>)> = Vec::new();
        for (name, layer) in &self.layers {
            tensors.push((format!("{}.lora_a", name), layer.lora_a.data()));
            tensors.push((format!("{}.lora_b", name), layer.lora_b.data()));
        }
        let refs: Vec<(&str, ndarray::ArrayD<f32>)> = tensors.iter()
            .map(|(n, a)| (n.as_str(), a.clone()))
            .collect();
        crate::safetensors::save_safetensors(path, &refs)?;
        Ok(())
    }

    pub fn load_adapter(&mut self, path: &str, target_shapes: &[Vec<usize>]) -> Result<(), Box<dyn std::error::Error>> {
        let loaded = crate::safetensors::load_safetensors(path)?;
        for (mut layer_pair) in self.layers.chunks_mut(2) {
            if layer_pair.len() < 2 { continue; }
            let name = &layer_pair[0].0;
            let layer = &mut layer_pair[0].1;
            let a_key = format!("{}.lora_a", name);
            let b_key = format!("{}.lora_b", name);
            if let Some(arr) = loaded.get(&a_key) {
                let shape = arr.shape().to_vec();
                if shape.len() == 2 {
                    layer.lora_a = Tensor::new(arr.clone());
                    layer.lora_a.set_requires_grad(true);
                }
            }
            if let Some(arr) = loaded.get(&b_key) {
                let shape = arr.shape().to_vec();
                if shape.len() == 2 {
                    layer.lora_b = Tensor::new(arr.clone());
                    layer.lora_b.set_requires_grad(true);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_layer_creation() {
        let layer = LoRALayer::new(8, 16.0, 64, 64);
        assert_eq!(layer.rank, 8);
        assert_eq!(layer.lora_a.shape(), &[8, 64]);
        assert_eq!(layer.lora_b.shape(), &[64, 8]);
    }

    #[test]
    fn test_lora_parameters() {
        let layer = LoRALayer::new(8, 16.0, 64, 64);
        let params = layer.parameters();
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_lora_model_create() {
        let config = LoRAConfig::default();
        let mut model = LoRAModel::new(config);
        model.add_layer("0.wq".into(), 64, 64);
        model.add_layer("0.wk".into(), 64, 64);
        assert_eq!(model.layers.len(), 2);
    }

    #[test]
    fn test_lora_parameter_count() {
        let config = LoRAConfig { rank: 8, ..Default::default() };
        let mut model = LoRAModel::new(config);
        model.add_layer("0.wq".into(), 64, 64);
        let count = model.parameter_count();
        assert_eq!(count, 8 * (64 + 64));
    }

    #[test]
    fn test_merge_unmerge_roundtrip() {
        use crate::models::transformer::{CausalLM, TransformerConfig};
        let cfg = TransformerConfig {
            vocab_size: 100, hidden_size: 32, num_heads: 4,
            num_kv_heads: 2, num_layers: 1, max_seq_len: 64,
            intermediate_size: 64, ..Default::default()
        };
        let mut model = CausalLM::new(cfg);
        let original_wq = model.blocks[0].attention.wq.clone();

        let mut lora_model = LoRAModel::new(LoRAConfig::default());
        lora_model.add_layer("0.wq".into(), 32, 32);
        lora_model.merge_all(&mut model);
        lora_model.unmerge_all(&mut model);

        let max_diff = model.blocks[0].attention.wq.iter()
            .zip(original_wq.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-5, "max_diff = {max_diff}");
    }
}
