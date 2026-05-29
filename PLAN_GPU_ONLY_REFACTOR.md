# GPU-Only Tensor Refactor (Approach B)

Target: Hapus semua CPU `Array2<f32>` / `Vec<Vec<f32>>` weight storage dari model inference. GPU weights jadi satu-satunya source of truth.

## Strategi

Setiap weight struct diganti: CPU `Array2<f32>` / `Vec<Vec<f32>>` dihapus. GPU weights (`OnceLock<GpuWeights>`) jadi primer. Fungsi `ensure_weights_gpu()` diubah agar baca dari safetensors file langsung → GPU, skip CPU `Array2`.

Untuk checkpoint save / training sync: tambah method `readback_weights()` yang readback GPU→CPU→safetensors.

CPU forward path tetap ada tapi readback GPU→CPU dulu di awal (fallback).

---

## ~~Step 1: `RMSNorm` (`crates/transformer/src/rms_norm.rs`)~~ ✅ DONE

### Perubahan:
- `weight: Array1<f32>` → `weight: Option<Array1<f32>>`
- Method baru: `drop_cpu_weight()`, `preupload_from_slice()`, `readback_weight()`
- `preupload_gpu()` handle `Option`, fallback untuk non-contiguous
- `forward()` / `forward_1d()` → `weight.as_ref().unwrap()` dengan panic message jelas
- Test baru: `test_rms_norm_drop_cpu_weight`
- Semua caller di `model.rs`, `block.rs`, `trainable.rs` diupdate handle `Option`

### Method changes:
| Method | Perubahan |
|--------|-----------|
| `new()` | Hapus `weight: Array1::from_...`, simpan `eps` doang. `gpu_weights` tetap kosong. |
| `from_weights(weight: Array1, eps)` | HAPUS — pake `from_gpu_weights()` instead |
| `forward()` CPU | HAPUS — GPU-only |
| `forward_1d()` | HAPUS — GPU-only |
| `forward_gpu()` | TETAP — GPU forward via gpu_weights |
| `preupload_gpu()` | Ganti: accept `weight_data: &[f32]` dari luar, upload langsung ke GPU |
| `readback_weight()` | BARU — readback GPU → return `Array1<f32>` |

### Affected callers:
- `TransformerBlock::new()` — berubah signature
- `CausalLM::new()` — berubah
- `from_checkpoint()` — load weight langsung ke GPU
- Tests — skip atau pake GPU

---

## Step 2: `SwiGLU` (`crates/transformer/src/swiglu.rs`)

### Struct changes:
```rust
// BEFORE:
pub struct SwiGLU {
    pub w1: Array2<f32>,          // CPU — HAPUS
    pub w2: Array2<f32>,          // CPU — HAPUS
    pub w3: Array2<f32>,          // CPU — HAPUS
    pub w1_f16: Option<Vec<u16>>, // CPU — HAPUS
    pub w2_f16: Option<Vec<u16>>, // CPU — HAPUS
    pub w3_f16: Option<Vec<u16>>, // CPU — HAPUS
    pub gpu_weights: OnceLock<SwigluGpuWeights>,
    pub use_half_precision: bool,
}

// AFTER:
pub struct SwiGLU {
    pub gpu_weights: OnceLock<SwigluGpuWeights>,
    pub use_half_precision: bool,
}
```

### `SwigluGpuWeights` — tetap, hanya gpu:
```rust
pub(crate) struct SwigluGpuWeights {
    pub w1_t: GpuTensor,
    pub w2_t: GpuTensor,
    pub w3_t: GpuTensor,
    pub w1_f16: Option<GpuTensor>,
    pub w2_f16: Option<GpuTensor>,
    pub w3_f16: Option<GpuTensor>,
}
```

### Method changes:
| Method | Perubahan |
|--------|-----------|
| `new()` | Hapus weight init. Terima `hidden_size, intermediate_size, use_half` doang. |
| `pack_f16_weights()` | HAPUS |
| `maybe_f16_matmul()` | HAPUS |
| `forward()` CPU | HAPUS |
| `ensure_weights_gpu()` | Ganti: accept `w1_data, w2_data, w3_data: &[f32]`, langsung upload |
| `preupload_gpu()` | accept weight slices |
| `forward_gpu()` | TETAP |
| `readback_weights()` | BARU — readback GPU → return 3 `Array2<f32>` |

### Affected callers:
- `TransformerBlock::new()` — berubah
- `CausalLM::new()` — berubah
- Tests — skip/ubah

---

## Step 3: `GQA` (`crates/transformer/src/gqa.rs`)

### Struct changes:
```rust
// BEFORE:
pub struct GQA {
    pub wq: Array2<f32>,           // CPU — HAPUS
    pub wk: Array2<f32>,           // CPU — HAPUS
    pub wv: Array2<f32>,           // CPU — HAPUS
    pub wo: Array2<f32>,           // CPU — HAPUS
    pub wq_f16: Option<Vec<u16>>,  // CPU — HAPUS
    pub wk_f16: Option<Vec<u16>>,  // CPU — HAPUS
    pub wv_f16: Option<Vec<u16>>,  // CPU — HAPUS
    pub wo_f16: Option<Vec<u16>>,  // CPU — HAPUS
    pub gpu_weights: OnceLock<GqaGpuWeights>,
    pub gpu_scratch: RwLock<Option<GpuKVCacheEntry>>,
    pub use_half_precision: bool,
    // ARCH: num_heads, num_kv_heads, head_dim, num_groups, head_dim_rs tetap
}

// AFTER:
pub struct GQA {
    pub gpu_weights: OnceLock<GqaGpuWeights>,
    pub gpu_scratch: RwLock<Option<GpuKVCacheEntry>>,
    pub use_half_precision: bool,
    // ARCH tetap: num_heads, num_kv_heads, head_dim, num_groups, head_dim_rs
}
```

### Method changes:
| Method | Perubahan |
|--------|-----------|
| `new()` | Hapus weight init. Terima weight slices dari luar. |
| `maybe_f16_matmul()` | HAPUS |
| `pack_f16_weights()` | HAPUS |
| `forward()` CPU | HAPUS |
| `forward_with_kv()` CPU | HAPUS |
| `forward_with_paged()` CPU | HAPUS |
| `ensure_weights_gpu()` | Ganti: accept `wq_data, wk_data, wv_data, wo_data: &[f32]` |
| `preupload_gpu()` | accept weight slices |
| All `forward_gpu_*()` | TETAP |
| Other CPU attention methods | HAPUS |
| `readback_weights()` | BARU |

### Affected callers:
- `TransformerBlock` — hapus CPU forward methods
- Tests — skip

---

## Step 4: `TransformerBlock` (`crates/transformer/src/block.rs`)

### Struct — tetap (tidak pegang weight langsung)

### Method changes:
| Method | Perubahan |
|--------|-----------|
| `new()` | Accept weight slices, pass ke sub-modules |
| `forward()` CPU | HAPUS |
| `forward_no_cache()` CPU | HAPUS |
| `forward_paged()` CPU | HAPUS |
| `forward_gpu_*()` | TETAP — semua GPU path |
| `set_use_half_precision()` | HAPUS internal pack, tetap set flag doang |
| `preupload_gpu()` | Accept weight refs, pass ke sub-modules |
| `readback_weights()` | BARU — collect dari sub-modules |

---

## Step 5: `CausalLM` (`crates/transformer/src/model.rs`)

### Struct changes:
```rust
// BEFORE:
pub struct CausalLM {
    pub token_embedding: Array2<f32>,  // CPU — HAPUS
    pub blocks: Vec<TransformerBlock>, // CPU weights inside — HAPUS
    pub norm: RMSNorm,                 // CPU weight — HAPUS
    pub lm_head: Array2<f32>,          // CPU — HAPUS
    pub gpu_weights: OnceLock<GpuWeights>,
    // ... config, rope, injectors, flags tetap
}

// AFTER:
pub struct CausalLM {
    pub blocks: Vec<TransformerBlock>,
    pub norm: RMSNorm,
    pub gpu_weights: OnceLock<GpuWeights>,
    // ... config, rope, injectors, flags tetap
}
```

### `GpuWeights` struct (`model.rs:195`):
```rust
pub(crate) struct GpuWeights {
    pub token_embedding: GpuTensor,   // TETAP
    pub lm_head_t: GpuTensor,         // TETAP
    pub lm_head_i8: Option<GpuTensor>,// TETAP
    pub lm_head_f16: Option<GpuTensor>,// TETAP
    pub norm_weight: GpuTensor,       // TETAP
    pub block_weights: Vec<BlockGpuWeights>, // TETAP
}
```

### Method changes:
| Method | Perubahan |
|--------|-----------|
| `new()` | Hanya bikin config, rope, injectors. GPU weights dari `from_checkpoint()`. |
| `from_checkpoint()` | **REWRITE** — load safetensors, langsung upload ke GPU, hapus CPU Array2 |
| `preupload_weights_gpu()` | Ganti: baca dari internal state (gpu_weights udah ada) atau dari file |
| `set_use_half_precision()` | Set flag, propagate ke blocks |
| `set_quantize_weights()` | TETAP |
| `save_checkpoint()` | BARU — readback GPU→CPU→safetensors |
| `forward()` CPU | HAPUS — panggil forward_gpu terus readback |
| All `forward_gpu*()` | TETAP |
| `readback_weights()` | BARU — collect semua weight dari GPU |
| `drop_cpu_weights()` | HAPUS — udah ga perlu |

### Affected callers:
- `TrainableCausalLM::from_inference()` — readback GPU→CPU tiap init training
- `TrainableCausalLM::sync_to_inference()` — tetap (write CPU weight yg direadback)
- `inference/src/engine.rs:CausalLM::from_checkpoint()` — berubah signature
- `foundation/src/causal_lm_model.rs:load_checkpoint()` — berubah
- `training/src/lib.rs:prepare()` — berubah (readback untuk training init)

---

## Step 6: `Router` (`crates/has-moe-ffn/src/routing.rs`)

### Struct changes:
```rust
// BEFORE:
pub struct Router {
    pub router_weights: Vec<Vec<f32>>,  // CPU — HAPUS
    pub router_weights_gpu: OnceLock<Option<GpuTensor>>,
    pub router_weights_cuda: OnceLock<Option<CudaTensor>>,
    // ...config, stats tetap
}

// AFTER:
pub struct Router {
    pub router_weights_gpu: OnceLock<Option<GpuTensor>>,
    pub router_weights_cuda: OnceLock<Option<CudaTensor>>,
    // ...config, stats tetap
}
```

### Method changes:
| Method | Perubahan |
|--------|-----------|
| `new()` | Accept `weights: &[f32]`, langsung upload ke GPU, skip CPU Vec |
| `ensure_weights_gpu()` | HAPUS — ganti init langsung di `new()` |
| `forward()` CPU naive | HAPUS |
| `forward_gpu()` / `forward_cuda()` | TETAP |
| `readback_weights()` | BARU |

---

## Step 7: `Expert` (`crates/has-moe-ffn/src/experts.rs`)

### Struct changes:
```rust
// BEFORE:
pub struct Expert {
    pub fc1_weights: Vec<Vec<f32>>,  // CPU — HAPUS
    pub fc1_bias: Vec<f32>,          // CPU — HAPUS
    pub fc2_weights: Vec<Vec<f32>>,  // CPU — HAPUS
    pub fc2_bias: Vec<f32>,          // CPU — HAPUS
    pub fc1_gpu: OnceLock<Option<GpuTensor>>,
    pub fc1_bias_gpu: OnceLock<Option<GpuTensor>>,
    pub fc2_gpu: OnceLock<Option<GpuTensor>>,
    pub fc2_bias_gpu: OnceLock<Option<GpuTensor>>,
    pub fc1_cuda: OnceLock<Option<CudaTensor>>,
    pub fc1_bias_cuda: OnceLock<Option<CudaTensor>>,
    pub fc2_cuda: OnceLock<Option<CudaTensor>>,
    pub fc2_bias_cuda: OnceLock<Option<CudaTensor>>,
}

// AFTER:
pub struct Expert {
    pub fc1_gpu: OnceLock<Option<GpuTensor>>,
    pub fc1_bias_gpu: OnceLock<Option<GpuTensor>>,
    pub fc2_gpu: OnceLock<Option<GpuTensor>>,
    pub fc2_bias_gpu: OnceLock<Option<GpuTensor>>,
    pub fc1_cuda: OnceLock<Option<CudaTensor>>,
    pub fc1_bias_cuda: OnceLock<Option<CudaTensor>>,
    pub fc2_cuda: OnceLock<Option<CudaTensor>>,
    pub fc2_bias_cuda: OnceLock<Option<CudaTensor>>,
    // ...config tetap
}
```

### Method changes:
| Method | Perubahan |
|--------|-----------|
| `new()` | Accept weight slices, langsung upload ke GPU/CUDA |
| `ensure_weights_gpu()` | HAPUS |
| `forward()` CPU | HAPUS |
| `forward_batched()` | Fallback readback dari GPU |
| `forward_batched_gpu/cuda()` | TETAP |
| `readback_weights()` | BARU |

---

## Step 8: `TrainableCausalLM` / Training (`crates/transformer/src/trainable.rs`)

`TrainableCausalLM` tetap pakai `Tensor` (bisa Storage::Gpu). Cuma `from_inference()` dan `sync_to_inference()` perlu adaptasi:

### `from_inference()`:
```rust
// SEBELUM: baca dari model.token_embedding (Array2<f32>)
// SESUDAH:  panggil model.readback_weights(), dapat Array2<f32>, trus Tensor::new()
pub fn from_inference(model: &CausalLM) -> Self {
    let cpu_weights = model.readback_weights()?;  // GPU→CPU
    // ...sama seperti sekarang, tapi baca dari cpu_weights struct
}
```

### `sync_to_inference()`:
```rust
// SEBELUM: nulis ke model.token_embedding (Array2<f32>)
// SESUDAH:  nulis tetap, tapi model.token_embedding ga ada.
//           Alternatif: upload langsung ke GPU model via model.preupload_weights_gpu()
//           Atau: write ke temporary buffer, panggil model.sync_from_trainer(tensors)
// REKOMENDASI: ganti method sync_to_inference jadi upload langsung ke GPU weights model
```

### `save_checkpoint()` — training:
```rust
// TETAP — TrainableCausalLM masih punya Tensor (CPU/GPU). Tensor::data() readback otomatis.
```

---

## Step 9: Checkpoint Save (inference)

Baru — `CausalLM::save_checkpoint()`:
```rust
pub fn save_checkpoint(&self, path: &str) -> Result<()> {
    let weights = self.readback_weights()?;  // GPU→CPU→Array2
    // ...safetensors write dari cpu_weights
}
```

### `from_checkpoint()` — baru:
```rust
pub fn from_checkpoint(config, path) -> Self {
    let loaded = load_safetensors(path)?;
    // Baca semua Array2 dari safetensors
    // Upload langsung ke GPU (panggil preupload_with_data())
    // Hapus CPU Array2
    // Simpan config, rope, etc
}
```

---

## Step 10: CPU Forward Fallback

Beberapa code path masih panggil CPU forward (`forward()` bukan `forward_gpu()`):
- Tests
- CPU-only deployment (`--no-default-features`)
- Fallback saat GPU init gagal

Solusi: bikin method `ensure_cpu_weights()`:
```rust
#[cfg(feature = "gpu")]
pub fn ensure_cpu_weights(&self) -> Result<(), Error> {
    // Kalau CPU weights belum ada, readback dari GPU
    // Simpan di internal OnceLock<Vec<Array2<f32>>>
}
```

TAPI: untuk production GPU-only, path ini ga perlu dipanggil. CPU forward cukup readback dari GPU saat fallback.

---

## Files Changed Summary

| File | Perubahan |
|------|-----------|
| `crates/transformer/src/rms_norm.rs` | Hapus `weight: Array1`, ganti `preupload_gpu()` |
| `crates/transformer/src/swiglu.rs` | Hapus semua CPU fields + methods |
| `crates/transformer/src/gqa.rs` | Hapus semua CPU fields + methods (~1000 baris dihapus) |
| `crates/transformer/src/block.rs` | Hapus CPU forward methods |
| `crates/transformer/src/model.rs` | Hapus CPU fields, `from_checkpoint` rewrite, `save_checkpoint` baru |
| `crates/transformer/src/trainable.rs` | `from_inference()` / `sync_to_inference()` adaptasi |
| `crates/has-moe-ffn/src/routing.rs` | Hapus `router_weights: Vec<Vec<f32>>` |
| `crates/has-moe-ffn/src/experts.rs` | Hapus semua CPU `Vec<Vec<f32>>` |
| `crates/inference/src/engine.rs` | Minor: `from_checkpoint()` signature |
| `crates/foundation/src/causal_lm_model.rs` | Adaptasi load_checkpoint |
| `crates/training/src/lib.rs` | Adaptasi prepare() — readback dulu |

---

## Execution Order

1. `RMSNorm` → simplest, no deps
2. `SwiGLU` → depends on RMSNorm (no), tapi test aja
3. `GQA` → big one, ~1000 line CPU code dihapus
4. `TransformerBlock` → hapus CPU forward passthrough
5. `CausalLM` → hapus CPU fields, rewrite checkpoint
6. `Router` → hapus CPU Vec
7. `Expert` → hapus CPU Vec
8. `TrainableCausalLM` → adaptasi from_inference/sync
9. Callers (inference, foundation, training) → adaptasi
10. Tests → fix
