use std::collections::{HashMap, HashSet, VecDeque};

use ndarray::ArrayD;

use super::tape;
use super::Tensor;

pub(crate) fn backward_engine(output: &Tensor) {
    let mut visited = HashSet::new();
    let mut topo = Vec::new();
    let mut queue = VecDeque::new();
    let mut grad_map: HashMap<usize, Tensor> = HashMap::new();

    grad_map.insert(output.id(), output.clone());
    queue.push_back(output.id());

    while let Some(cur_id) = queue.pop_front() {
        if !visited.insert(cur_id) {
            continue;
        }

        let tensor = grad_map.get(&cur_id).cloned();
        if let Some(ref t) = tensor {
            if let Some(fn_idx) = t.get_grad_fn_idx() {
                let inputs = tape::with_tape(|tap| tap.inputs(fn_idx));
                topo.push(cur_id);

                for inp in &inputs {
                    let inp_id = inp.id();
                    if !grad_map.contains_key(&inp_id) {
                        grad_map.insert(inp_id, inp.clone());
                    }
                    queue.push_back(inp_id);
                }
            }
        }
    }



    let mut grads: HashMap<usize, ArrayD<f32>> = HashMap::new();
    if let Some(g) = output.grad() {
        grads.insert(output.id(), g);
    }

    for &node_id in &topo {
        let grad_out = match grads.get(&node_id) {
            Some(g) => g.clone(),
            None => continue,
        };

        let tensor = grad_map.get(&node_id).cloned();
        if let Some(ref t) = tensor {
            if let Some(fn_idx) = t.get_grad_fn_idx() {
                let inputs = tape::with_tape(|tap| tap.inputs(fn_idx));
                let saved = tape::with_tape(|tap| tap.saved(fn_idx));
                let backward_fn = tape::with_tape_mut(|tap| tap.take_backward(fn_idx));

                if let Some(backward) = backward_fn {
                    let grad_inputs = backward(&grad_out, &saved);

                    for (i, inp) in inputs.iter().enumerate() {
                        if i < grad_inputs.len() && inp.requires_grad() {
                            let g = grad_inputs[i].clone();
                            grads.entry(inp.id())
                                .and_modify(|existing| {
                                    if existing.shape() == g.shape() {
                                        *existing += &g;
                                    } else {
                                        *existing = g.clone();
                                    }
                                })
                                .or_insert(g);
                        }
                    }
                }
            }
        }
    }

    for (tid, g) in grads {
        if let Some(t) = grad_map.get(&tid) {
            t.accumulate_grad(&g);
        }
    }
}
