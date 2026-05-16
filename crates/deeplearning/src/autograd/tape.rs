use ndarray::ArrayD;
use std::cell::RefCell;

use super::Tensor;

thread_local! {
    pub(crate) static TAPE: RefCell<AutogradTape> = RefCell::new(AutogradTape {
        nodes: Vec::new(),
    });
}

pub(crate) struct GradFn {
    pub inputs: Vec<Tensor>,
    pub saved: Vec<ArrayD<f32>>,
    pub backward: Option<Box<dyn FnOnce(&ArrayD<f32>, &[ArrayD<f32>]) -> Vec<ArrayD<f32>>>>,
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
        })
    })
}

pub fn with_tape_mut<F, R>(f: F) -> R
where
    F: FnOnce(&mut AutogradTape) -> R,
{
    TAPE.with(|tape| f(&mut tape.borrow_mut()))
}

pub fn with_tape<F, R>(f: F) -> R
where
    F: FnOnce(&AutogradTape) -> R,
{
    TAPE.with(|tape| f(&*tape.borrow()))
}
