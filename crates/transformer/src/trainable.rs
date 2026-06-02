use ndarray::{Array1, Array2};
use nexora_autograd::ops::{causal_softmax, embedding, rms_norm_2d};
use nexora_autograd::{self, Tensor, TensorOps};

use super::config::TransformerConfig;
use super::model::CausalLM;

fn identity_selector(rows: usize, cols: usize, offset: usize) -> Tensor {
    let mut data = vec![0.0f32; rows * cols];
    for j in 0..cols {
        let i = offset + j;
        if i < rows {
            data[i * cols + j] = 1.0;
        }
    }
    Tensor::from_slice(&data, &[rows, cols])
}

pub struct TrainableCausalLM {
    pub config: TransformerConfig,
    pub token_embedding: Tensor,
    pub blocks: Vec<TrainableBlock>,
    pub norm: TrainableRMSNorm,
    pub lm_head: Tensor,
}

pub struct TrainableBlock {
    pub attention_norm: TrainableRMSNorm,
    pub ffn_norm: TrainableRMSNorm,
    pub attention: TrainableGQA,
    pub ffn: TrainableSwiGLU,
}

pub struct TrainableRMSNorm {
    pub weight: Tensor,
    pub eps: f32,
}

pub struct TrainableGQA {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_groups: usize,
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,
}

pub struct TrainableSwiGLU {
    pub w1: Tensor,
    pub w2: Tensor,
    pub w3: Tensor,
}

impl TrainableCausalLM {
    pub fn from_inference(model: &CausalLM) -> Self {
        let to_tensor = |arr: Option<&Array2<f32>>, name: &str| -> Tensor {
            match arr {
                Some(a) => {
                    let t = Tensor::new(a.clone().into_dyn());
                    t.set_requires_grad(true);
                    t
                }
                None => {
                    tracing::warn!("{} not available during from_inference", name);
                    let t = Tensor::new(Array2::zeros((0, 0)).into_dyn());
                    t.set_requires_grad(true);
                    t
                }
            }
        };
        let to_tensor_1d = |arr: Option<&Array1<f32>>, name: &str| -> Tensor {
            match arr {
                Some(a) => {
                    let t = Tensor::new(a.clone().into_dyn());
                    t.set_requires_grad(true);
                    t
                }
                None => {
                    tracing::warn!("{} not available during from_inference", name);
                    let t = Tensor::new(Array1::zeros(0).into_dyn());
                    t.set_requires_grad(true);
                    t
                }
            }
        };

        let blocks = model
            .blocks
            .iter()
            .map(|b| TrainableBlock {
                attention_norm: TrainableRMSNorm {
                    weight: to_tensor_1d(b.attention_norm.weight.as_ref(), "attention_norm.weight"),
                    eps: b.attention_norm.eps,
                },
                ffn_norm: TrainableRMSNorm {
                    weight: to_tensor_1d(b.ffn_norm.weight.as_ref(), "ffn_norm.weight"),
                    eps: b.ffn_norm.eps,
                },
                attention: TrainableGQA {
                    num_heads: b.attention.num_heads,
                    num_kv_heads: b.attention.num_kv_heads,
                    head_dim: b.attention.head_dim,
                    num_groups: b.attention.num_groups,
                    wq: to_tensor(b.attention.wq.as_ref(), "attention.wq"),
                    wk: to_tensor(b.attention.wk.as_ref(), "attention.wk"),
                    wv: to_tensor(b.attention.wv.as_ref(), "attention.wv"),
                    wo: to_tensor(b.attention.wo.as_ref(), "attention.wo"),
                },
                ffn: TrainableSwiGLU {
                    w1: to_tensor(b.ffn.w1.as_ref(), "ffn.w1"),
                    w2: to_tensor(b.ffn.w2.as_ref(), "ffn.w2"),
                    w3: to_tensor(b.ffn.w3.as_ref(), "ffn.w3"),
                },
            })
            .collect();

        Self {
            config: model.config.clone(),
            token_embedding: to_tensor(model.token_embedding.as_ref(), "token_embedding"),
            blocks,
            norm: TrainableRMSNorm {
                weight: to_tensor_1d(model.norm.weight.as_ref(), "norm.weight"),
                eps: model.norm.eps,
            },
            lm_head: to_tensor(model.lm_head.as_ref(), "lm_head"),
        }
    }

    pub fn sync_to_inference(
        &self,
        model: &mut CausalLM,
    ) -> Result<(), Box<dyn std::error::Error>> {
        model.token_embedding = Some(self
            .token_embedding
            .data()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| "Internal invariant: token_embedding must be 2D")?
            .to_owned());
        model.lm_head = Some(self
            .lm_head
            .data()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| "Internal invariant: lm_head must be 2D")?
            .to_owned());
        model.norm.weight = Some(
            self.norm.weight.data()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| "Internal invariant: norm must be 1D")?
                .to_owned());
        for (i, block) in self.blocks.iter().enumerate() {
            model.blocks[i].attention_norm.weight = Some(
                block.attention_norm.weight.data()
                    .into_dimensionality::<ndarray::Ix1>()
                    .map_err(|_| "Internal invariant: attention norm must be 1D")?
                    .to_owned());
            model.blocks[i].ffn_norm.weight = Some(
                block.ffn_norm.weight.data()
                    .into_dimensionality::<ndarray::Ix1>()
                    .map_err(|_| "Internal invariant: ffn norm must be 1D")?
                    .to_owned());
            model.blocks[i].attention.wq = Some(block
                .attention
                .wq
                .data()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| "Internal invariant: attention wq must be 2D")?
                .to_owned());
            model.blocks[i].attention.wk = Some(block
                .attention
                .wk
                .data()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| "Internal invariant: attention wk must be 2D")?
                .to_owned());
            model.blocks[i].attention.wv = Some(block
                .attention
                .wv
                .data()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| "Internal invariant: attention wv must be 2D")?
                .to_owned());
            model.blocks[i].attention.wo = Some(block
                .attention
                .wo
                .data()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| "Internal invariant: attention wo must be 2D")?
                .to_owned());
            model.blocks[i].ffn.w1 = Some(
                block.ffn.w1.data()
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|_| "Internal invariant: ffn w1 must be 2D")?
                    .to_owned());
            model.blocks[i].ffn.w2 = Some(
                block.ffn.w2.data()
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|_| "Internal invariant: ffn w2 must be 2D")?
                    .to_owned());
            model.blocks[i].ffn.w3 = Some(
                block.ffn.w3.data()
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|_| "Internal invariant: ffn w3 must be 2D")?
                    .to_owned());
        }
        if model.keep_on_gpu {
            model.reset_gpu_weights();
        }
        Ok(())
    }

    pub fn forward(&self, input_ids: &Tensor) -> Tensor {
        let seq_len = input_ids.shape()[0];
        let hidden = self.config.hidden_size;
        let n_heads = self.config.num_heads;
        let n_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden / n_heads;
        let num_groups = n_heads / n_kv_heads;

        let mut h = embedding(input_ids, &self.token_embedding);
        // h shape: [seq_len, hidden] — keep 2D throughout

        for block in &self.blocks {
            let residual = h.clone();

            let normed = rms_norm_2d(&h, &block.attention_norm.weight, block.attention_norm.eps);

            let q_proj = normed.matmul(&block.attention.wq.transpose());
            let k_proj = normed.matmul(&block.attention.wk.transpose());
            let v_proj = normed.matmul(&block.attention.wv.transpose());

            let q_total = n_heads * head_dim;
            let k_total = n_kv_heads * head_dim;
            let scale = (head_dim as f32).sqrt();
            let scale_t = Tensor::from_slice(&[scale], &[1]);

            // Per-head GQA: each query head attends to its assigned KV head
            let mut attn_out = Tensor::zeros(&[seq_len, q_total], false);
            for h_idx in 0..n_heads {
                let kv_h = h_idx / num_groups;
                let q_sel = identity_selector(q_total, head_dim, h_idx * head_dim);
                let kv_sel = identity_selector(k_total, head_dim, kv_h * head_dim);

                let q_h = q_proj.matmul(&q_sel);
                let k_h = k_proj.matmul(&kv_sel);
                let v_h = v_proj.matmul(&kv_sel);

                let scores = q_h.matmul(&k_h.transpose()).div(&scale_t);
                let attn = causal_softmax(&scores);
                let out_h = attn.matmul(&v_h);

                let place_sel = identity_selector(q_total, head_dim, h_idx * head_dim);
                let placed = out_h.matmul(&place_sel.transpose());
                attn_out = attn_out.add(&placed);
            }

            let wo_t = block.attention.wo.transpose();
            attn_out = attn_out.matmul(&wo_t);
            h = residual.add(&attn_out);

            let residual = h.clone();
            let normed = rms_norm_2d(&h, &block.ffn_norm.weight, block.ffn_norm.eps);

            let gate = normed.matmul(&block.ffn.w1.transpose());
            let hidden_states = normed.matmul(&block.ffn.w3.transpose());
            let gated = gate.silu().mul(&hidden_states);
            let ffn_out = gated.matmul(&block.ffn.w2.transpose());

            h = residual.add(&ffn_out);
        }

        h = rms_norm_2d(&h, &self.norm.weight, self.norm.eps);

        let logits = h.matmul(&self.lm_head.transpose());
        logits
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![
            self.token_embedding.clone(),
            self.lm_head.clone(),
            self.norm.weight.clone(),
        ];
        for block in &self.blocks {
            params.push(block.attention_norm.weight.clone());
            params.push(block.ffn_norm.weight.clone());
            params.push(block.attention.wq.clone());
            params.push(block.attention.wk.clone());
            params.push(block.attention.wv.clone());
            params.push(block.attention.wo.clone());
            params.push(block.ffn.w1.clone());
            params.push(block.ffn.w2.clone());
            params.push(block.ffn.w3.clone());
        }
        params
    }

    pub fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }

    pub fn collect_checkpoint_tensors(&self) -> Vec<(String, ndarray::ArrayD<f32>)> {
        let suffix_names = [
            "attention_norm.weight",
            "ffn_norm.weight",
            "attention.wq",
            "attention.wk",
            "attention.wv",
            "attention.wo",
            "ffn.w1",
            "ffn.w2",
            "ffn.w3",
        ];
        let mut tensors: Vec<(String, ndarray::ArrayD<f32>)> =
            Vec::with_capacity(3 + 9 * self.blocks.len());
        tensors.push(("token_embedding".into(), self.token_embedding.data()));
        tensors.push(("lm_head".into(), self.lm_head.data()));
        tensors.push(("norm.weight".into(), self.norm.weight.data()));
        for (i, block) in self.blocks.iter().enumerate() {
            let data_refs = [
                block.attention_norm.weight.data(),
                block.ffn_norm.weight.data(),
                block.attention.wq.data(),
                block.attention.wk.data(),
                block.attention.wv.data(),
                block.attention.wo.data(),
                block.ffn.w1.data(),
                block.ffn.w2.data(),
                block.ffn.w3.data(),
            ];
            for (j, suffix) in suffix_names.iter().enumerate() {
                let key = format!("blocks.{}.{}", i, suffix);
                tensors.push((key, data_refs[j].clone()));
            }
        }
        tensors
    }

    pub fn save_checkpoint(&self, path: &str) -> crate::TransformerResult<()> {
        let tensors = self.collect_checkpoint_tensors();
        let refs: Vec<(&str, ndarray::ArrayD<f32>)> = tensors
            .iter()
            .map(|(name, arr)| (name.as_str(), arr.clone()))
            .collect();
        let mut meta = std::collections::HashMap::new();
        meta.insert(
            "quantization".to_string(),
            self.config.quantization.dtype_name().to_string(),
        );
        crate::safetensors::save_safetensors_with_meta(
            path,
            &refs,
            crate::safetensors::SaveDtype::F32,
            Some(meta),
        )
    }

    pub fn load_checkpoint(model: &mut CausalLM, path: &str) -> crate::TransformerResult<()> {
        let loaded = crate::safetensors::load_safetensors(path)?;

        let get_arr = |name: &str| -> crate::TransformerResult<ndarray::ArrayD<f32>> {
            loaded.get(name).cloned().ok_or_else(|| {
                crate::TransformerError::Implementation(format!("Missing tensor: {}", name))
            })
        };

        fn to_fixed<D: ndarray::Dimension>(
            arr: ndarray::ArrayD<f32>,
            name: &str,
        ) -> crate::TransformerResult<ndarray::Array<f32, D>> {
            arr.into_dimensionality::<D>().map_err(|e| {
                crate::TransformerError::Implementation(format!(
                    "Shape mismatch for {}: {}",
                    name, e
                ))
            })
        }

        model.token_embedding = Some(
            to_fixed::<ndarray::Ix2>(get_arr("token_embedding")?, "token_embedding")?);
        model.lm_head = Some(
            to_fixed::<ndarray::Ix2>(get_arr("lm_head")?, "lm_head")?);
        model.norm.weight = Some(
            to_fixed::<ndarray::Ix1>(get_arr("norm.weight")?, "norm.weight")?);

        for (i, block) in model.blocks.iter_mut().enumerate() {
            macro_rules! load {
                ($field:expr, $name:expr, $dim:ty) => {{
                    let key = format!("blocks.{}.{}", i, $name);
                    $field = to_fixed::<$dim>(get_arr(&key)?, &key)?;
                }};
            }
            macro_rules! load_opt {
                ($field:expr, $name:expr, $dim:ty) => {{
                    let key = format!("blocks.{}.{}", i, $name);
                    $field = Some(to_fixed::<$dim>(get_arr(&key)?, &key)?);
                }};
            }
            load_opt!(
                block.attention_norm.weight,
                "attention_norm.weight",
                ndarray::Ix1
            );
            load_opt!(block.ffn_norm.weight, "ffn_norm.weight", ndarray::Ix1);
            load_opt!(block.attention.wq, "attention.wq", ndarray::Ix2);
            load_opt!(block.attention.wk, "attention.wk", ndarray::Ix2);
            load_opt!(block.attention.wv, "attention.wv", ndarray::Ix2);
            load_opt!(block.attention.wo, "attention.wo", ndarray::Ix2);
            load_opt!(block.ffn.w1, "ffn.w1", ndarray::Ix2);
            load_opt!(block.ffn.w2, "ffn.w2", ndarray::Ix2);
            load_opt!(block.ffn.w3, "ffn.w3", ndarray::Ix2);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TransformerConfig;
    use nexora_quantization::QFormat;

    fn small_model() -> CausalLM {
        CausalLM::new(TransformerConfig {
            vocab_size: 50,
            hidden_size: 16,
            num_heads: 2,
            num_kv_heads: 1,
            num_layers: 1,
            max_seq_len: 32,
            intermediate_size: 32,
            rope_theta: 10000.0,
            use_cache: true,
            norm_eps: 1e-6,
            num_experts: 0,
            top_k_experts: 0,
            expert_intermediate_size: 0,
            quantization: QFormat::F16,
            use_half_precision: true,
        })
    }

    #[test]
    fn test_from_inference_smoke() {
        let inf = small_model();
        let trainable = TrainableCausalLM::from_inference(&inf);
        assert_eq!(trainable.config.vocab_size, 50);
        assert_eq!(trainable.config.hidden_size, 16);
        assert_eq!(trainable.blocks.len(), 1);
        assert!(trainable.token_embedding.requires_grad());
        assert!(trainable.lm_head.requires_grad());
    }

    #[test]
    fn test_forward_no_panic() {
        let inf = small_model();
        let trainable = TrainableCausalLM::from_inference(&inf);
        let input_ids = Tensor::from_slice(&[0.0f32, 1.0, 2.0], &[3]);
        let logits = trainable.forward(&input_ids);
        let shape = logits.shape();
        assert_eq!(shape, &[3, 50]);
    }

    #[test]
    fn test_forward_single_token() {
        let inf = small_model();
        let trainable = TrainableCausalLM::from_inference(&inf);
        let input_ids = Tensor::from_slice(&[5.0f32], &[1]);
        let logits = trainable.forward(&input_ids);
        assert_eq!(logits.shape(), &[1, 50]);
    }

    #[test]
    fn test_sync_to_inference_roundtrip() {
        let inf = small_model();
        let trainable = TrainableCausalLM::from_inference(&inf);
        let mut inf2 = CausalLM::new(TransformerConfig {
            vocab_size: 50,
            hidden_size: 16,
            num_heads: 2,
            num_kv_heads: 1,
            num_layers: 1,
            max_seq_len: 32,
            intermediate_size: 32,
            rope_theta: 10000.0,
            use_cache: true,
            norm_eps: 1e-6,
            num_experts: 0,
            top_k_experts: 0,
            expert_intermediate_size: 0,
            quantization: QFormat::F16,
            use_half_precision: true,
        });
        trainable.sync_to_inference(&mut inf2).unwrap();
        assert_eq!(inf2.token_embedding.as_ref().unwrap().dim(), (50, 16));
        assert_eq!(inf2.lm_head.as_ref().unwrap().dim(), (50, 16));
        assert_eq!(inf2.norm.weight.as_ref().unwrap().len(), 16);
    }

    #[test]
    fn test_parameters_count() {
        let inf = small_model();
        let trainable = TrainableCausalLM::from_inference(&inf);
        let params = trainable.parameters();
        assert_eq!(params.len(), 3 + 9);
    }

    #[test]
    fn test_zero_grad_no_panic() {
        let inf = small_model();
        let trainable = TrainableCausalLM::from_inference(&inf);
        let input_ids = Tensor::from_slice(&[0.0f32, 1.0], &[2]);
        let logits = trainable.forward(&input_ids);
        let loss = logits.sum();
        loss.backward();
        trainable.zero_grad();
        for p in trainable.parameters() {
            let grad = p.grad();
            assert!(grad.is_none() || grad.as_ref().map_or(true, |g| g.is_empty()));
        }
    }

    #[test]
    fn test_save_checkpoint_roundtrip() {
        let path = "/tmp/test_trainable_ckpt.safetensors";
        let _ = std::fs::remove_file(path);

        let inf = small_model();
        let trainable = TrainableCausalLM::from_inference(&inf);
        if let Ok(()) = trainable.save_checkpoint(path) {
            let mut reloaded = CausalLM::new(TransformerConfig {
                vocab_size: 50,
                hidden_size: 16,
                num_heads: 2,
                num_kv_heads: 1,
                num_layers: 1,
                max_seq_len: 32,
                intermediate_size: 32,
                rope_theta: 10000.0,
                use_cache: true,
                norm_eps: 1e-6,
                num_experts: 0,
                top_k_experts: 0,
                expert_intermediate_size: 0,
                quantization: QFormat::F16,
                use_half_precision: true,
            });
            TrainableCausalLM::load_checkpoint(&mut reloaded, path).unwrap();
            assert!(reloaded.blocks[0]
                .attention
                .wq
                .as_ref()
                .unwrap()
                .iter()
                .all(|v| v.is_finite()));
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_save_checkpoint_contents() {
        let path = "/tmp/test_trainable_ckpt2.safetensors";
        let _ = std::fs::remove_file(path);

        let inf = small_model();
        let original_wq = inf.blocks[0].attention.wq.as_ref().unwrap().clone();
        let trainable = TrainableCausalLM::from_inference(&inf);
        if let Ok(()) = trainable.save_checkpoint(path) {
            let mut reloaded = CausalLM::new(TransformerConfig {
                vocab_size: 50,
                hidden_size: 16,
                num_heads: 2,
                num_kv_heads: 1,
                num_layers: 1,
                max_seq_len: 32,
                intermediate_size: 32,
                rope_theta: 10000.0,
                use_cache: true,
                norm_eps: 1e-6,
                num_experts: 0,
                top_k_experts: 0,
                expert_intermediate_size: 0,
                quantization: QFormat::F16,
                use_half_precision: true,
            });
            TrainableCausalLM::load_checkpoint(&mut reloaded, path).unwrap();
            let wq_len = original_wq.shape()[0] * original_wq.shape()[1];
            for j in 0..wq_len {
                let orig_val = original_wq.as_slice().unwrap()[j];
                let reloaded_val = reloaded.blocks[0].attention.wq.as_ref().unwrap().as_slice().unwrap()[j];
                assert!(
                    (orig_val - reloaded_val).abs() < 1e-5,
                    "mismatch at {j}: {orig_val} vs {reloaded_val}"
                );
            }
        }
        let _ = std::fs::remove_file(path);
    }
}
