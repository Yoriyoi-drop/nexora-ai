# Metode Perceptron yang Sudah Ada di Project

> Project ini **tidak** menggunakan istilah "perceptron" secara eksplisit. Namun semua komponen inti algoritma perceptron sudah tersedia.

---

## 1. Optimizer (Learning Rule)

### SGDOptimizer (Momentum-based)
**File:** `crates/foundation/src/hldva_t/training/optimizer.rs:115-188`

```rust
pub struct SGDOptimizer {
    config: SGDConfig,
    learning_rate: f32,
    momentum: f32,
    velocity: Vec<Vec<f32>>,
}
// velocity = momentum * velocity + gradient
// params -= lr * velocity
```

### SGDMomentumOptimizer (ATQS)
**File:** `crates/foundation/src/atqs/calibration/calibration_optimizer.rs:726-768`

### AdamOptimizer
**File:** `crates/foundation/src/hldva_t/training/optimizer.rs:14-113`
**File:** `crates/foundation/src/atqs/calibration/calibration_optimizer.rs:661-723`

### Optimizer Lain
| Optimizer | File |
|-----------|------|
| AdaGradOptimizer | `crates/foundation/src/atqs/calibration/calibration_optimizer.rs:771-842` |
| RMSPropOptimizer | `crates/foundation/src/atqs/calibration/calibration_optimizer.rs:845-921` |
| LAMBOptimizer | `crates/foundation/src/atqs/calibration/calibration_optimizer.rs:924-950` |

### Trait Optimizer (Generic)
**File:** `crates/foundation/src/traits/model_traits.rs:353-371`

```rust
pub trait Optimizer {
    fn step(&mut self, parameters: &mut [Tensor], gradients: &[Tensor]) -> Result<(), Self::Error>;
    fn learning_rate(&self) -> f32;
    fn set_learning_rate(&mut self, lr: f32);
}
```

---

## 2. Weight Update (Gradient Descent)

| File | Method | Formula |
|------|--------|---------|
| `crates/foundation/src/erp/reconstruction.rs:305-311` | `update_weights()` | `w -= grad * lr` + weight decay |
| `crates/deeplearning/src/echo_net/sse.rs:307-314` | `update_weights()` | `w -= grad * lr` (amplitude/phase/freq) |
| `crates/foundation/src/traits/model_traits.rs:308` | `ModelLayer::update_parameters()` | Generic `learning_rate` param |

---

## 3. Forward + Backward Traits

**File:** `crates/deeplearning/src/lib.rs:61-94`

```rust
pub trait Forward {
    type Input;
    type Output;
    fn forward(&self, input: &Self::Input) -> DLResult<Self::Output>;
}

pub trait Backward {
    type Gradient;
    fn backward(&self, grad: &Self::Gradient) -> DLResult<Self::Gradient>;
}

pub trait Trainable {
    fn parameters(&self) -> Vec<&[f32]>;
    fn parameters_mut(&mut self) -> Vec<&mut [f32]>;
    fn gradients(&self) -> Vec<&[f32]>;
    fn gradients_mut(&mut self) -> Vec<&mut [f32]>;
}
```

**Trait ModelLayer:**
**File:** `crates/foundation/src/traits/model_traits.rs:292-318`

```rust
pub trait ModelLayer {
    fn forward(&self, input: Self::Input) -> Result<Self::Output, Self::Error>;
    fn parameters(&self) -> Vec<Tensor>;
    fn gradients(&self) -> Vec<Tensor>;
    fn update_parameters(&mut self, learning_rate: f32) -> Result<(), Self::Error>;
}
```

---

## 4. FeedForward / MLP Layer

### DenseLayer (HAS-MoE-FFN)
**File:** `crates/foundation/src/has_moe_ffn/layers.rs:20-126`

- Fully-connected: `output = input * weights^T + bias`
- Aktivasi: ReLU, GELU, Sigmoid, Tanh, Swish
- Xavier/Glorot init

### FeedForward (2-layer MLP)
**File:** `crates/foundation/src/hldva_t/dit/attention.rs:354-386`

```rust
pub struct FeedForward {
    pub fc1: Linear,
    pub fc2: Linear,
    pub activation: GELU,
}
```

### LinearLayer (ORACLE)
**File:** `crates/foundation/src/oracle/backbone.rs`

- `weight: Array2<f32>`, `bias: Array1<f32>`
- Forward: `x.dot(&weight) + &bias`

---

## 5. Loss Functions

| Fungsi | File | Baris |
|--------|------|-------|
| Cross-entropy (classification) | `crates/foundation/src/vogp/mod.rs` | 134-149 |
| MSE Loss | `crates/foundation/src/atqs/calibration/calibration_optimizer.rs` | 636-647 |
| MSE Loss | `crates/foundation/src/hldva_t/training/mod.rs` | 409-422 |
| Trait LossFunction | `crates/foundation/src/traits/model_traits.rs` | 338-351 |

---

## 6. Training Loop

| File | Baris | Deskripsi |
|------|-------|-----------|
| `crates/foundation/src/vogp/training.rs` | 132-537 | Training step + epoch loop + validation |
| `crates/foundation/src/hldva_t/training/mod.rs` | 64-274 | 4-stage training (VAE→DiT→cascaded→finetune) |
| `crates/foundation/src/oracle/trainer.rs` | 192-279 | 3-phase training (pretrain→alignment→eval) |
| `crates/foundation/src/erp/training.rs` | 40-246 | ERP training pipeline |
| `apps/nexora-ai/src/cli/training.rs` | 15-109 | CLI training command |

---

## Summary

```
Perceptron Building Block       → Status di Project
────────────────────────────────────────────────────
Forward pass                    ✅ DenseLayer, FeedForward, LinearLayer
Weight init (Xavier)            ✅ DenseLayer::init_weights()
Bias                            ✅ Option<Vec<f32>> di DenseLayer
Activation function             ✅ ReLU/GELU/Sigmoid/Tanh/Swish
Loss function                   ✅ MSE + Cross-entropy
Gradient Descent / SGD          ✅ SGDOptimizer + SGDMomentumOptimizer
Weight update rule              ✅ update_weights() di ERP, SSE, ModelLayer
Training loop                   ✅ VOGP, HLDVA, ORACLE, ERP
Backward pass                   ✅ Backward trait + implementations
Binary classification           ✅ Cross-entropy (support 2+ classes)
```

Tidak ada yang bernama "perceptron", tapi semua primitifnya sudah siap pakai.
