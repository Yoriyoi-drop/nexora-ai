//! F16 storage support for STar-X tensors
//!
//! Provides zero-copy operations on f16 storage where possible,
//! and automatic conversion where f32 is required.

use crate::{F16Storage, StoragePrecision};
use ndarray::ArrayD;

/// A tensor that can be stored in either f32 or f16 precision
pub struct MixedPrecisionTensor {
    pub f32_data: Option<ArrayD<f32>>,
    pub f16_data: Option<F16Storage>,
    pub shape: Vec<usize>,
    pub precision: StoragePrecision,
}

impl MixedPrecisionTensor {
    pub fn new_f32(data: ArrayD<f32>) -> Self {
        let shape = data.shape().to_vec();
        Self {
            f32_data: Some(data),
            f16_data: None,
            shape,
            precision: StoragePrecision::F32,
        }
    }

    pub fn new_f16(storage: F16Storage, shape: Vec<usize>) -> Self {
        Self {
            f32_data: None,
            f16_data: Some(storage),
            shape,
            precision: StoragePrecision::F16,
        }
    }

    pub fn to_f32(&self) -> ArrayD<f32> {
        match (&self.f32_data, &self.f16_data) {
            (Some(f32), _) => f32.clone(),
            (None, Some(f16)) => crate::f16_storage_to_f32_tensor(f16, self.shape.clone()),
            (None, None) => ArrayD::from_shape_vec(self.shape.clone(), vec![])
                .unwrap_or_else(|_| ArrayD::zeros(self.shape.clone())),
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        match self.precision {
            StoragePrecision::F32 => self.numel() * 4,
            StoragePrecision::F16 => self.numel() * 2,
        }
    }

    /// Convert from f32 to f16 in-place (drops f32 data)
    pub fn convert_to_f16(&mut self) {
        if let Some(f32) = self.f32_data.take() {
            let storage = crate::f32_tensor_to_f16_storage(&f32);
            self.f16_data = Some(storage);
            self.precision = StoragePrecision::F16;
        }
    }

    /// Convert from f16 to f32 in-place (drops f16 data)
    pub fn convert_to_f32(&mut self) {
        if let Some(f16) = self.f16_data.take() {
            let f32 = crate::f16_storage_to_f32_tensor(&f16, self.shape.clone());
            self.f32_data = Some(f32);
            self.precision = StoragePrecision::F32;
        }
    }
}

/// Memory-efficient storage for a collection of tensors
pub struct F16TensorStore {
    tensors: Vec<MixedPrecisionTensor>,
    default_precision: StoragePrecision,
}

impl F16TensorStore {
    pub fn new(default_precision: StoragePrecision) -> Self {
        Self {
            tensors: Vec::new(),
            default_precision,
        }
    }

    pub fn add(&mut self, tensor: MixedPrecisionTensor) {
        self.tensors.push(tensor);
    }

    pub fn get(&self, index: usize) -> Option<&MixedPrecisionTensor> {
        self.tensors.get(index)
    }

    pub fn total_memory_bytes(&self) -> usize {
        self.tensors.iter().map(|t| t.memory_bytes()).sum()
    }

    pub fn convert_all_to_f16(&mut self) {
        for t in &mut self.tensors {
            t.convert_to_f16();
        }
    }

    pub fn convert_all_to_f32(&mut self) {
        for t in &mut self.tensors {
            t.convert_to_f32();
        }
    }
}
