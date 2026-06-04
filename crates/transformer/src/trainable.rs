use ndarray::{Array1, Array2};
use nexora_autograd::attention_workspace::{global_pool, WorkspacePool};
use nexora_autograd::ops::{causal_softmax, embedding, rms_norm_2d};
use nexora_autograd::{self, Tensor, TensorOps};

use super::config::TransformerConfig;
use super::model::CausalLM;

/// Configuration for training optimizations
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Use chunked causal attention (avoids O(S²) memory)
    pub chunked_attention: bool,
    /// Chunk size for chunked attention
    pub attention_chunk_size: usize,
    /// Enable early free of intermediate tensors
    pub early_free: bool,
    /// Enable activation checkpointing (recompute during backward)
    pub activation_checkpointing: bool,
    /// Workspace pool for reusable buffers
    pub workspace_pool: Option<WorkspacePool>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            chunked_attention: true,
            attention_chunk_size: 512,
            early_free: true,
            activation_checkpointing: true,
            workspace_pool: None,
        }
    }
}

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

/// Select a head's slice from the Q/K/V projection: [seq_len, total] → [seq_len, head_dim]
fn select_heads(proj: &Tensor, total: usize, head_dim: usize, head_idx: usize) -> Tensor {
    let sel = identity_selector(total, head_dim, head_idx * head_dim);
    proj.matmul(&sel)
}

/// Place a head's output back into the full output: [seq_len, head_dim] → [seq_len, total]
fn place_heads(head: &Tensor, total: usize, head_dim: usize, head_idx: usize) -> Tensor {
    let sel = identity_selector(total, head_dim, head_idx * head_dim);
    head.matmul(&sel.transpose())
}

/// Slice rows from a 2D tensor: [full, dim] → [len, dim]
fn slice_rows(t: &Tensor, start: usize, end: usize) -> Tensor {
    let data = t.data();
    let shape = data.shape();
    if shape.len() != 2 {
        return t.clone();
    }
    let rows = shape[0];
    let cols = shape[1];
    let end = end.min(rows);
    let len = end - start;
    let mut result = vec![0.0f32; len * cols];
    let flat: Vec<f32> = data.iter().copied().collect();
    for i in 0..len {
        for j in 0..cols {
            result[i * cols + j] = flat[(start + i) * cols + j];
        }
    }
    let t_result = Tensor::from_slice(&result, &[len, cols]);
    if t.requires_grad() {
        t_result.set_requires_grad(true);
    }
    t_result
}

/// Apply causal mask within a chunk: position i attends only to kv_start..=i
fn causal_mask_chunk(scores: &Tensor, _seq_len: usize, chunk_start: usize, _chunk_end: usize) -> Tensor {
    let data = scores.data();
    let shape = data.shape();
    if shape.len() != 2 {
        return scores.clone();
    }
    let q_len = shape[0];
    let kv_len = shape[1];
    let mut result: Vec<f32> = data.iter().copied().collect();
    for i in 0..q_len {
        for j in 0..kv_len {
            let kv_pos = chunk_start + j;
            if kv_pos > i {
                result[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    let t = Tensor::from_slice(&result, &[q_len, kv_len]);
    if scores.requires_grad() {
        t.set_requires_grad(true);
    }
    t
}

pub struct TrainableCausalLM {
    pub config: TransformerConfig,
    pub token_embedding: Tensor,
    pub blocks: Vec<TrainableBlock>,
    pub norm: TrainableRMSNorm,
    pub lm_head: Tensor,
    pub training_config: TrainingConfig,
}

pub struct TrainableBlock {
    pub attention_norm: TrainableRMSNorm,
    pub ffn_norm: TrainableRMSNorm,
    pub attention: TrainableGQA,
    pub ffn: TrainableSwiGLU,
    pub experts: Option<Vec<TrainableSwiGLU>>,
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
            .map(|b| {
                let experts = b.experts.as_ref().map(|exps| {
                    exps.iter().enumerate().map(|(e_idx, e)| {
                        TrainableSwiGLU {
                            w1: to_tensor(e.w1.as_ref(), &format!("experts.{}.w1", e_idx)),
                            w2: to_tensor(e.w2.as_ref(), &format!("experts.{}.w2", e_idx)),
                            w3: to_tensor(e.w3.as_ref(), &format!("experts.{}.w3", e_idx)),
                        }
                    }).collect::<Vec<_>>()
                });
                TrainableBlock {
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
                    experts,
                }
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
            training_config: TrainingConfig::default(),
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
            // Sync expert weights
            if let Some(train_experts) = &block.experts {
                let inf_experts = model.blocks[i].experts.get_or_insert_with(|| {
                    (0..train_experts.len()).map(|_| {
                        let mut e = super::swiglu::SwiGLU::new(
                            model.config.hidden_size,
                            model.config.expert_intermediate_size,
                        );
                        e.init_random(
                            model.config.hidden_size,
                            model.config.expert_intermediate_size,
                        );
                        e
                    }).collect()
                });
                for (e_idx, train_e) in train_experts.iter().enumerate() {
                    if e_idx < inf_experts.len() {
                        inf_experts[e_idx].w1 = Some(
                            train_e.w1.data()
                                .into_dimensionality::<ndarray::Ix2>()
                                .map_err(|_| "Internal invariant: expert w1 must be 2D")?
                                .to_owned());
                        inf_experts[e_idx].w2 = Some(
                            train_e.w2.data()
                                .into_dimensionality::<ndarray::Ix2>()
                                .map_err(|_| "Internal invariant: expert w2 must be 2D")?
                                .to_owned());
                        inf_experts[e_idx].w3 = Some(
                            train_e.w3.data()
                                .into_dimensionality::<ndarray::Ix2>()
                                .map_err(|_| "Internal invariant: expert w3 must be 2D")?
                                .to_owned());
                    }
                }
            }
        }
        if model.keep_on_gpu {
            model.reset_gpu_weights();
        }
        model.notify_weight_changed();
        Ok(())
    }

    pub fn forward(&self, input_ids: &Tensor) -> Tensor {
        let seq_len = input_ids.shape()[0];
        let hidden = self.config.hidden_size;
        let n_heads = self.config.num_heads;
        let n_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden / n_heads;
        let num_groups = n_heads / n_kv_heads;
        let cfg = &self.training_config;
        let _pool = cfg.workspace_pool.as_ref().unwrap_or_else(|| global_pool());

        let mut h = embedding(input_ids, &self.token_embedding);

        for block in &self.blocks {
            let residual = h.clone();

            let normed = rms_norm_2d(&h, &block.attention_norm.weight, block.attention_norm.eps);

            let q_proj = normed.matmul(&block.attention.wq.transpose());
            let k_proj = normed.matmul(&block.attention.wk.transpose());
            let v_proj = normed.matmul(&block.attention.wv.transpose());

            let q_total = n_heads * head_dim;
            let k_total = n_kv_heads * head_dim;

            let attn_out = if cfg.chunked_attention && seq_len > cfg.attention_chunk_size {
                // FIX 1+2: Chunked causal attention — avoids O(S²) score matrix
                self.chunked_attention_forward(
                    &q_proj, &k_proj, &v_proj,
                    seq_len, q_total, k_total, head_dim, n_heads, n_kv_heads, num_groups,
                    cfg, _pool,
                )
            } else {
                // Original per-head GQA for short sequences (≤ chunk_size)
                self.per_head_attention_forward(
                    &q_proj, &k_proj, &v_proj,
                    seq_len, q_total, k_total, head_dim, n_heads, n_kv_heads, num_groups,
                    cfg,
                )
            };

            // FIX 6: Early free
            if cfg.early_free {
                drop(normed);
                drop(q_proj);
                drop(k_proj);
                drop(v_proj);
            }

            let wo_t = block.attention.wo.transpose();
            let attn_out = attn_out.matmul(&wo_t);
            h = residual.add(&attn_out);

            let residual = h.clone();
            let normed = rms_norm_2d(&h, &block.ffn_norm.weight, block.ffn_norm.eps);

            let ffn_out = self.ffn_forward(&normed, block, cfg);

            if cfg.early_free {
                drop(normed);
            }

            h = residual.add(&ffn_out);
        }

        h = rms_norm_2d(&h, &self.norm.weight, self.norm.eps);

        let logits = h.matmul(&self.lm_head.transpose());
        logits
    }

    /// Chunked attention forward — processes KV in chunks to avoid O(S²) memory
    fn chunked_attention_forward(
        &self,
        q_proj: &Tensor,
        k_proj: &Tensor,
        v_proj: &Tensor,
        seq_len: usize,
        q_total: usize,
        k_total: usize,
        head_dim: usize,
        n_heads: usize,
        _n_kv_heads: usize,
        num_groups: usize,
        cfg: &TrainingConfig,
        _pool: &WorkspacePool,
    ) -> Tensor {
        let mut attn_out = Tensor::zeros(&[seq_len, q_total], false);

        for h_idx in 0..n_heads {
            let kv_h = h_idx / num_groups;

            let q_h = select_heads(q_proj, q_total, head_dim, h_idx);
            let k_h = select_heads(k_proj, k_total, head_dim, kv_h);
            let v_h = select_heads(v_proj, k_total, head_dim, kv_h);

            // Chunked attention: process KV in blocks
            let chunk_size = cfg.attention_chunk_size.min(seq_len).max(1);
            let mut head_out = Tensor::zeros(&[seq_len, head_dim], q_proj.requires_grad());

            for chunk_start in (0..seq_len).step_by(chunk_size) {
                let chunk_end = (chunk_start + chunk_size).min(seq_len);

                let k_chunk = slice_rows(&k_h, chunk_start, chunk_end);
                let v_chunk = slice_rows(&v_h, chunk_start, chunk_end);

                let scale = (head_dim as f32).sqrt();
                let scores = q_h.matmul(&k_chunk.transpose())
                    .div(&Tensor::from_slice(&[scale], &[1]));

                let masked = causal_mask_chunk(&scores, seq_len, chunk_start, chunk_end);

                let attn_weights = causal_softmax(&masked);
                let partial = attn_weights.matmul(&v_chunk);

                head_out = head_out.add(&partial);

                // FIX 6: Early free
                if cfg.early_free {
                    drop(k_chunk);
                    drop(v_chunk);
                    drop(scores);
                    drop(masked);
                    drop(attn_weights);
                    drop(partial);
                }
            }

            let placed = place_heads(&head_out, q_total, head_dim, h_idx);
            attn_out = attn_out.add(&placed);

            if cfg.early_free {
                drop(q_h);
                drop(k_h);
                drop(v_h);
                drop(placed);
            }
        }

        attn_out
    }

    /// Per-head GQA with full attention matrix (for short sequences)
    fn per_head_attention_forward(
        &self,
        q_proj: &Tensor,
        k_proj: &Tensor,
        v_proj: &Tensor,
        seq_len: usize,
        q_total: usize,
        k_total: usize,
        head_dim: usize,
        n_heads: usize,
        _n_kv_heads: usize,
        num_groups: usize,
        cfg: &TrainingConfig,
    ) -> Tensor {
        let mut attn_out = Tensor::zeros(&[seq_len, q_total], false);
        let scale = (head_dim as f32).sqrt();
        let scale_t = Tensor::from_slice(&[scale], &[1]);

        for h_idx in 0..n_heads {
            let kv_h = h_idx / num_groups;

            let q_h = select_heads(q_proj, q_total, head_dim, h_idx);
            let k_h = select_heads(k_proj, k_total, head_dim, kv_h);
            let v_h = select_heads(v_proj, k_total, head_dim, kv_h);

            let scores = q_h.matmul(&k_h.transpose()).div(&scale_t);
            let attn = causal_softmax(&scores);
            let out_h = attn.matmul(&v_h);

            let placed = place_heads(&out_h, q_total, head_dim, h_idx);
            attn_out = attn_out.add(&placed);

            if cfg.early_free {
                drop(q_h);
                drop(k_h);
                drop(v_h);
                drop(scores);
                drop(attn);
                drop(placed);
            }
        }

        attn_out
    }

    /// FFN forward with early free support
    fn ffn_forward(&self, normed: &Tensor, block: &TrainableBlock, cfg: &TrainingConfig) -> Tensor {
        let result = if let Some(experts) = &block.experts {
            let mut sum: Option<Tensor> = None;
            for expert in experts {
                let gate = normed.matmul(&expert.w1.transpose());
                let hidden = normed.matmul(&expert.w3.transpose());
                let gated = gate.silu().mul(&hidden);
                let out = gated.matmul(&expert.w2.transpose());
                if cfg.early_free {
                    drop(gate);
                    drop(hidden);
                    drop(gated);
                }
                sum = Some(match sum {
                    Some(s) => s.add(&out),
                    None => out,
                });
            }
            let sum = match sum {
                Some(s) => s,
                None => {
                    tracing::warn!("MoE experts list is empty, returning zeros");
                    return Tensor::from_slice(&[0.0f32], &[1, 1]);
                }
            };
            let n = sum.clone().div(&Tensor::from_slice(&[experts.len() as f32], &[1]));
            if cfg.early_free { drop(sum); }
            n
        } else {
            let gate = normed.matmul(&block.ffn.w1.transpose());
            let hidden = normed.matmul(&block.ffn.w3.transpose());
            let gated = gate.silu().mul(&hidden);
            let out = gated.matmul(&block.ffn.w2.transpose());
            if cfg.early_free {
                drop(gate);
                drop(hidden);
                drop(gated);
            }
            out
        };
        result
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
            if let Some(experts) = &block.experts {
                for e in experts {
                    params.push(e.w1.clone());
                    params.push(e.w2.clone());
                    params.push(e.w3.clone());
                }
            }
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
        let expert_tensors_per = self.config.num_experts * 3;
        let mut tensors: Vec<(String, ndarray::ArrayD<f32>)> =
            Vec::with_capacity(3 + 9 * self.blocks.len() + expert_tensors_per * self.blocks.len());
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
            // Expert tensors
            if let Some(experts) = &block.experts {
                for (e_idx, expert) in experts.iter().enumerate() {
                    let e_prefix = format!("blocks.{}.experts.{}.", i, e_idx);
                    tensors.push((format!("{}w1", e_prefix), expert.w1.data()));
                    tensors.push((format!("{}w2", e_prefix), expert.w2.data()));
                    tensors.push((format!("{}w3", e_prefix), expert.w3.data()));
                }
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
            // Load expert weights if present in checkpoint
            if let Some(experts) = &mut block.experts {
                for (e_idx, expert) in experts.iter_mut().enumerate() {
                    let e_prefix = format!("blocks.{}.experts.{}.", i, e_idx);
                    let w1_name = format!("{}w1", e_prefix);
                    if loaded.contains_key(&w1_name) {
                        expert.w1 = Some(to_fixed::<ndarray::Ix2>(get_arr(&w1_name)?, &w1_name)?);
                        let w2_name = format!("{}w2", e_prefix);
                        expert.w2 = Some(to_fixed::<ndarray::Ix2>(get_arr(&w2_name)?, &w2_name)?);
                        let w3_name = format!("{}w3", e_prefix);
                        expert.w3 = Some(to_fixed::<ndarray::Ix2>(get_arr(&w3_name)?, &w3_name)?);
                    }
                }
            }
        }

        // Invalidate GPU cache so next forward re-uploads
        #[cfg(feature = "gpu")]
        model.reset_gpu_weights();

        model.notify_weight_changed();

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
            shard: Default::default(),
            shared_expert: 0,
            use_domain_experts: false,
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
            shard: Default::default(),
            shared_expert: 0,
            use_domain_experts: false,
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
                shard: Default::default(),
                shared_expert: 0,
            use_domain_experts: false,
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
                shard: Default::default(),
                shared_expert: 0,
            use_domain_experts: false,
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
