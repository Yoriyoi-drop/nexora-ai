use ndarray::ArrayD;
use std::cell::RefCell;

use super::Tensor;
#[cfg(feature = "gpu")]
use crate::gpu::{GpuContext, GpuTensor};

thread_local! {
    pub(crate) static TAPE: RefCell<AutogradTape> = const { RefCell::new(AutogradTape {
        nodes: Vec::new(),
    }) };
}

#[cfg(feature = "gpu")]
pub(crate) type GpuBackwardFn =
    Box<dyn FnOnce(&[GpuTensor], &GpuTensor, &GpuContext) -> Result<Vec<GpuTensor>, String> + Send>;

pub(crate) struct GradFn {
    pub inputs: Vec<Tensor>,
    pub saved: Vec<ArrayD<f32>>,
    pub backward: Option<Box<dyn FnOnce(&ArrayD<f32>, &[ArrayD<f32>]) -> Vec<ArrayD<f32>>>>,
    #[cfg(feature = "gpu")]
    pub saved_gpu: Vec<GpuTensor>,
    #[cfg(feature = "gpu")]
    pub gpu_backward: Option<GpuBackwardFn>,
}

pub(crate) struct AutogradTape {
    nodes: Vec<GradFn>,
}

impl AutogradTape {
    pub fn push(&mut self, node: GradFn) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    pub fn take_backward(
        &mut self,
        idx: usize,
    ) -> Option<Box<dyn FnOnce(&ArrayD<f32>, &[ArrayD<f32>]) -> Vec<ArrayD<f32>>>> {
        if idx < self.nodes.len() {
            self.nodes[idx].backward.take()
        } else {
            None
        }
    }

    pub fn inputs(&self, idx: usize) -> Vec<Tensor> {
        if idx < self.nodes.len() {
            self.nodes[idx].inputs.clone()
        } else {
            vec![]
        }
    }

    pub fn saved(&self, idx: usize) -> Vec<ArrayD<f32>> {
        if idx < self.nodes.len() {
            self.nodes[idx].saved.clone()
        } else {
            vec![]
        }
    }
}

pub fn register_grad_fn(
    inputs: Vec<Tensor>,
    saved: Vec<ArrayD<f32>>,
    backward: Box<dyn FnOnce(&ArrayD<f32>, &[ArrayD<f32>]) -> Vec<ArrayD<f32>>>,
) -> usize {
    TAPE.with(|tape| {
        tape.borrow_mut().push(GradFn {
            inputs,
            saved,
            backward: Some(backward),
            #[cfg(feature = "gpu")]
            saved_gpu: Vec::new(),
            #[cfg(feature = "gpu")]
            gpu_backward: None,
        })
    })
}

/// Register a GPU-native backward function.
/// `saved_gpu` contains GPU tensors needed for backward.
/// `gpu_backward` is called with (saved_gpu, upstream_grad_gpu, ctx) -> Vec<GpuTensor>
#[cfg(feature = "gpu")]
pub fn register_gpu_grad_fn(
    inputs: Vec<Tensor>,
    saved: Vec<ArrayD<f32>>,
    saved_gpu: Vec<GpuTensor>,
    backward: Box<dyn FnOnce(&ArrayD<f32>, &[ArrayD<f32>]) -> Vec<ArrayD<f32>>>,
    gpu_backward: Option<GpuBackwardFn>,
) -> usize {
    TAPE.with(|tape| {
        tape.borrow_mut().push(GradFn {
            inputs,
            saved,
            backward: Some(backward),
            saved_gpu,
            gpu_backward,
        })
    })
}

/// Check if a grad node has GPU backward
#[cfg(feature = "gpu")]
pub(crate) fn has_gpu_backward(idx: usize) -> bool {
    TAPE.with(|tape| {
        let t = tape.borrow();
        if idx < t.nodes.len() {
            t.nodes[idx].gpu_backward.is_some()
        } else {
            false
        }
    })
}

/// Take GPU backward function
#[cfg(feature = "gpu")]
pub(crate) fn take_gpu_backward(idx: usize) -> Option<GpuBackwardFn> {
    TAPE.with(|tape| {
        if idx < tape.borrow().nodes.len() {
            tape.borrow_mut().nodes[idx].gpu_backward.take()
        } else {
            None
        }
    })
}

/// Get saved GPU tensors
#[cfg(feature = "gpu")]
pub(crate) fn saved_gpu(idx: usize) -> Vec<GpuTensor> {
    TAPE.with(|tape| {
        if idx < tape.borrow().nodes.len() {
            tape.borrow().nodes[idx].saved_gpu.clone()
        } else {
            vec![]
        }
    })
}

/// Clear all entries from the autograd tape.
///
/// Must be called after each training step to prevent unbounded memory growth.
/// References: PyTorch clears the tape-equivalent after every `backward()`.
pub fn clear_tape() {
    TAPE.with(|tape| tape.borrow_mut().nodes.clear());
}

pub(crate) fn with_tape_mut<F, R>(f: F) -> R
where
    F: FnOnce(&mut AutogradTape) -> R,
{
    TAPE.with(|tape| f(&mut tape.borrow_mut()))
}

pub(crate) fn with_tape<F, R>(f: F) -> R
where
    F: FnOnce(&AutogradTape) -> R,
{
    TAPE.with(|tape| f(&tape.borrow()))
}
