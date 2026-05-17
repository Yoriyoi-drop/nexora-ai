use ndarray::{Array1, Array2, ArrayD};
use nexora_deeplearning::autograd::{self, Tensor, TensorOps, Adam};
use nexora_deeplearning::autograd::ops::{embedding, rms_norm_2d, causal_softmax};

use super::model::CausalLM;
use super::config::TransformerConfig;
use super::gqa::KVCacheEntry;

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
        let to_tensor = |arr: &Array2<f32>| -> Tensor {
            let t = Tensor::new(arr.clone().into_dyn());
            t.set_requires_grad(true);
            t
        };
        let to_tensor_1d = |arr: &Array1<f32>| -> Tensor {
            let t = Tensor::new(arr.clone().into_dyn());
            t.set_requires_grad(true);
            t
        };

        let blocks = model.blocks.iter().map(|b| TrainableBlock {
            attention_norm: TrainableRMSNorm {
                weight: to_tensor_1d(&b.attention_norm.weight),
                eps: b.attention_norm.eps,
            },
            ffn_norm: TrainableRMSNorm {
                weight: to_tensor_1d(&b.ffn_norm.weight),
                eps: b.ffn_norm.eps,
            },
            attention: TrainableGQA {
                num_heads: b.attention.num_heads,
                num_kv_heads: b.attention.num_kv_heads,
                head_dim: b.attention.head_dim,
                num_groups: b.attention.num_groups,
                wq: to_tensor(&b.attention.wq),
                wk: to_tensor(&b.attention.wk),
                wv: to_tensor(&b.attention.wv),
                wo: to_tensor(&b.attention.wo),
            },
            ffn: TrainableSwiGLU {
                w1: to_tensor(&b.ffn.w1),
                w2: to_tensor(&b.ffn.w2),
                w3: to_tensor(&b.ffn.w3),
            },
        }).collect();

        Self {
            config: model.config.clone(),
            token_embedding: to_tensor(&model.token_embedding),
            blocks,
            norm: TrainableRMSNorm {
                weight: to_tensor_1d(&model.norm.weight),
                eps: model.norm.eps,
            },
            lm_head: to_tensor(&model.lm_head),
        }
    }

    pub fn sync_to_inference(&self, model: &mut CausalLM) {
        model.token_embedding = self.token_embedding.data()
            .into_dimensionality::<ndarray::Ix2>().expect("Internal invariant: token_embedding must be 2D").to_owned();
        model.lm_head = self.lm_head.data()
            .into_dimensionality::<ndarray::Ix2>().expect("Internal invariant: lm_head must be 2D").to_owned();
        model.norm.weight = self.norm.weight.data()
            .into_dimensionality::<ndarray::Ix1>().expect("Internal invariant: norm must be 1D").to_owned();
        for (i, block) in self.blocks.iter().enumerate() {
            model.blocks[i].attention_norm.weight = block.attention_norm.weight.data()
                .into_dimensionality::<ndarray::Ix1>().expect("Internal invariant: attention norm must be 1D").to_owned();
            model.blocks[i].ffn_norm.weight = block.ffn_norm.weight.data()
                .into_dimensionality::<ndarray::Ix1>().expect("Internal invariant: ffn norm must be 1D").to_owned();
            model.blocks[i].attention.wq = block.attention.wq.data()
                .into_dimensionality::<ndarray::Ix2>().expect("Internal invariant: attention wq must be 2D").to_owned();
            model.blocks[i].attention.wk = block.attention.wk.data()
                .into_dimensionality::<ndarray::Ix2>().expect("Internal invariant: attention wk must be 2D").to_owned();
            model.blocks[i].attention.wv = block.attention.wv.data()
                .into_dimensionality::<ndarray::Ix2>().expect("Internal invariant: attention wv must be 2D").to_owned();
            model.blocks[i].attention.wo = block.attention.wo.data()
                .into_dimensionality::<ndarray::Ix2>().expect("Internal invariant: attention wo must be 2D").to_owned();
            model.blocks[i].ffn.w1 = block.ffn.w1.data()
                .into_dimensionality::<ndarray::Ix2>().expect("Internal invariant: ffn w1 must be 2D").to_owned();
            model.blocks[i].ffn.w2 = block.ffn.w2.data()
                .into_dimensionality::<ndarray::Ix2>().expect("Internal invariant: ffn w2 must be 2D").to_owned();
            model.blocks[i].ffn.w3 = block.ffn.w3.data()
                .into_dimensionality::<ndarray::Ix2>().expect("Internal invariant: ffn w3 must be 2D").to_owned();
        }
    }

    pub fn forward(&self, input_ids: &Tensor) -> Tensor {
        let seq_len = input_ids.shape()[0];
        let hidden = self.config.hidden_size;
        let n_heads = self.config.num_heads;
        let n_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden / n_heads;
        let num_groups = n_heads / n_kv_heads;

        let mut h = embedding(input_ids, &self.token_embedding);
        h = h.reshape(&[1, seq_len, hidden]);

        for block in &self.blocks {
            let residual = h.clone();

            let normed = rms_norm_2d(
                &h.reshape(&[seq_len, hidden]),
                &block.attention_norm.weight,
                block.attention_norm.eps,
            ).reshape(&[1, seq_len, hidden]);

            let q_proj = normed.matmul(&block.attention.wq.transpose());
            let k_proj = normed.matmul(&block.attention.wk.transpose());
            let v_proj = normed.matmul(&block.attention.wv.transpose());

            let q_total = n_heads * head_dim;
            let k_total = n_kv_heads * head_dim;
            let group_dim = num_groups * head_dim;

            let mut attn_parts: Vec<Tensor> = Vec::with_capacity(n_kv_heads);
            for g in 0..n_kv_heads {
                let q_sel = identity_selector(q_total, group_dim, g * group_dim);
                let kv_sel = identity_selector(k_total, head_dim, g * head_dim);

                let q_g = q_proj.matmul(&q_sel);
                let k_g = k_proj.matmul(&kv_sel);
                let v_g = v_proj.matmul(&kv_sel);

                let scale = (head_dim as f32).sqrt();
                let scores = q_g.matmul(&k_g.transpose())
                    .div(&Tensor::from_slice(&[scale], &[1]));

                let attn = causal_softmax(&scores);
                let out_g = attn.matmul(&v_g);
                attn_parts.push(out_g);
            }

            let mut attn_out = Tensor::zeros(&[seq_len, q_total], false);
            for g in 0..n_kv_heads {
                let place_sel = identity_selector(q_total, group_dim, g * group_dim);
                let placed = attn_parts[g].matmul(&place_sel.transpose());
                attn_out = attn_out.add(&placed);
            }

            let wo_t = block.attention.wo.transpose();
            attn_out = attn_out.matmul(&wo_t);
            h = residual.add(&attn_out);

            let residual = h.clone();
            let normed = rms_norm_2d(
                &h.reshape(&[seq_len, hidden]),
                &block.ffn_norm.weight,
                block.ffn_norm.eps,
            ).reshape(&[1, seq_len, hidden]);

            let gate = normed.matmul(&block.ffn.w1.transpose());
            let hidden_states = normed.matmul(&block.ffn.w3.transpose());
            let gated = gate.silu().mul(&hidden_states);
            let ffn_out = gated.matmul(&block.ffn.w2.transpose());

            h = residual.add(&ffn_out);
        }

        h = rms_norm_2d(
            &h.reshape(&[seq_len, hidden]),
            &self.norm.weight,
            self.norm.eps,
        );

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

    pub fn save_checkpoint(&self, path: &str) -> crate::FoundationResult<()> {
        let suffix_names = ["attention_norm.weight", "ffn_norm.weight", "attention.wq", "attention.wk", "attention.wv", "attention.wo", "ffn.w1", "ffn.w2", "ffn.w3"];
        let mut tensors: Vec<(String, ndarray::ArrayD<f32>)> = Vec::with_capacity(3 + 9 * self.blocks.len());
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
        let refs: Vec<(&str, ndarray::ArrayD<f32>)> = tensors.iter()
            .map(|(name, arr)| (name.as_str(), arr.clone()))
            .collect();
        crate::safetensors::save_safetensors(path, &refs)
    }

    pub fn load_checkpoint(model: &mut CausalLM, path: &str) -> crate::FoundationResult<()> {
        let loaded = crate::safetensors::load_safetensors(path)?;

        let get_arr = |name: &str| -> crate::FoundationResult<ndarray::ArrayD<f32>> {
            loaded.get(name).cloned().ok_or_else(|| {
                crate::FoundationError::Implementation(format!("Missing tensor: {}", name))
            })
        };

        fn to_fixed<D: ndarray::Dimension>(arr: ndarray::ArrayD<f32>, name: &str) -> crate::FoundationResult<ndarray::Array<f32, D>> {
            arr.into_dimensionality::<D>()
                .map_err(|e| crate::FoundationError::Implementation(format!("Shape mismatch for {}: {}", name, e)))
        }

        model.token_embedding = to_fixed::<ndarray::Ix2>(get_arr("token_embedding")?, "token_embedding")?;
        model.lm_head = to_fixed::<ndarray::Ix2>(get_arr("lm_head")?, "lm_head")?;
        model.norm.weight = to_fixed::<ndarray::Ix1>(get_arr("norm.weight")?, "norm.weight")?;

        for (i, block) in model.blocks.iter_mut().enumerate() {
            macro_rules! load {
                ($field:expr, $name:expr, $dim:ty) => {{
                    let key = format!("blocks.{}.{}", i, $name);
                    $field = to_fixed::<$dim>(get_arr(&key)?, &key)?;
                }};
            }
            load!(block.attention_norm.weight, "attention_norm.weight", ndarray::Ix1);
            load!(block.ffn_norm.weight, "ffn_norm.weight", ndarray::Ix1);
            load!(block.attention.wq, "attention.wq", ndarray::Ix2);
            load!(block.attention.wk, "attention.wk", ndarray::Ix2);
            load!(block.attention.wv, "attention.wv", ndarray::Ix2);
            load!(block.attention.wo, "attention.wo", ndarray::Ix2);
            load!(block.ffn.w1, "ffn.w1", ndarray::Ix2);
            load!(block.ffn.w2, "ffn.w2", ndarray::Ix2);
            load!(block.ffn.w3, "ffn.w3", ndarray::Ix2);
        }

        Ok(())
    }
}
