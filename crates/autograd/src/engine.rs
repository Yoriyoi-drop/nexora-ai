use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use super::device::Storage;
use super::tape;
use super::Tensor;
#[cfg(feature = "device-gpu")]
use crate::gpu::GpuContext;

pub fn backward_engine(output: &Tensor) {
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
                    grad_map.entry(inp_id).or_insert_with(|| inp.clone());
                    queue.push_back(inp_id);
                }
            }
        }
    }

    let mut grads: HashMap<usize, Storage> = HashMap::new();
    if let Some(g) = output.grad() {
        grads.insert(output.id(), Storage::Cpu(Arc::new(g)));
    }

    #[cfg(feature = "device-gpu")]
    let gpu_batch_active: bool =
        GpuContext::is_available() && topo.iter().any(|id| tape::has_gpu_backward(*id));

    #[cfg(feature = "device-gpu")]
    if gpu_batch_active {
        if let Ok(ctx) = GpuContext::global() {
            ctx.begin_batch_mode();
        }
    }

    for &node_id in &topo {
        let mut grad_out_storage = match grads.get(&node_id) {
            Some(g) => g.clone(),
            None => continue,
        };

        let tensor = grad_map.get(&node_id).cloned();
        if let Some(ref t) = tensor {
            if let Some(fn_idx) = t.get_grad_fn_idx() {
                let inputs = tape::with_tape(|tap| tap.inputs(fn_idx));
                let saved = tape::with_tape(|tap| tap.saved(fn_idx));

                #[cfg(feature = "device-gpu")]
                let used_gpu: bool = {
                    if tape::has_gpu_backward(fn_idx) {
                        if let Ok(ctx) = GpuContext::global() {
                            let saved_gpu = tape::saved_gpu(fn_idx);
                            let gpu_backward = tape::take_gpu_backward(fn_idx);
                            if let Some(backward_gpu) = gpu_backward {
                                let grad_gpu_result = match &grad_out_storage {
                                    Storage::Gpu(g, _) => Ok(g.clone()),
                                    Storage::Cpu(arr) => {
                                        crate::gpu::GpuTensor::from_cpu(arr.as_ref())
                                    }
                                    Storage::Cuda(_, _) => {
                                        Err(crate::gpu::GpuError::Unsupported("Cuda grad in Gpu backward path".into()))
                                    }
                                };
                                let used_gpu_inner = match grad_gpu_result {
                                    Ok(grad_gpu) => {
                                        match backward_gpu(&saved_gpu, &grad_gpu, ctx) {
                                            Ok(grad_inputs_gpu) => {
                                                for (i, inp) in inputs.iter().enumerate() {
                                                    if i < grad_inputs_gpu.len()
                                                        && inp.requires_grad()
                                                    {
                                                        let gpu_grad = grad_inputs_gpu[i].clone();
                                                        if let Some(existing) =
                                                            grads.get_mut(&inp.id())
                                                        {
                                                            if let Storage::Gpu(ref mut e, _) =
                                                                existing
                                                            {
                                                                if e.shape() == gpu_grad.shape() {
                                                                    if let Err(err) = ctx
                                                                        .add_inplace(e, &gpu_grad)
                                                                    {
                                                                        tracing::warn!("GPU add_inplace failed in backward: {err}");
                                                                    }
                                                                } else {
                                                                    match (
                                                                        e.to_cpu(),
                                                                        gpu_grad.to_cpu(),
                                                                    ) {
                                                                        (
                                                                            Ok(ref mut e_cpu),
                                                                            Ok(ref g_cpu),
                                                                        ) => {
                                                                            *e_cpu += g_cpu;
                                                                            *existing =
                                                                                Storage::Cpu(
                                                                                    Arc::new(
                                                                                        e_cpu
                                                                                            .clone(
                                                                                            ),
                                                                                    ),
                                                                                );
                                                                        }
                                                                        (Err(err), _)
                                                                        | (_, Err(err)) => {
                                                                            tracing::warn!("GPU backward shape mismatch readback failed: {err}");
                                                                        }
                                                                    }
                                                                }
                                                            } else {
                                                                let mut e_cpu = existing.to_cpu();
                                                                match gpu_grad.to_cpu() {
                                                                    Ok(ref g_cpu) => {
                                                                        e_cpu += g_cpu;
                                                                        *existing = Storage::Cpu(
                                                                            Arc::new(e_cpu),
                                                                        );
                                                                    }
                                                                    Err(err) => {
                                                                        tracing::warn!("GPU backward mixed storage readback failed: {err}");
                                                                    }
                                                                }
                                                            }
                                                        } else {
                                                            let gpu_grad_shape = gpu_grad.shape();
                                                            grads.insert(
                                                                 inp.id(),
                                                                 Storage::Gpu(gpu_grad, gpu_grad_shape),
                                                            );
                                                        }
                                                    }
                                                }
                                                true
                                            }
                                            Err(e) => {
                                                tracing::warn!(
                                                    "GPU backward failed, falling back to CPU: {e}"
                                                );
                                                if let Storage::Gpu(g, _) = &grad_out_storage {
                                                    if let Ok(cpu) = g.to_cpu() {
                                                        let cpu_storage =
                                                            Storage::Cpu(Arc::new(cpu));
                                                        grad_out_storage = cpu_storage.clone();
                                                        grads.insert(node_id, cpu_storage);
                                                    }
                                                }
                                                false
                                            }
                                        }
                                    }
                                    Err(_) => false,
                                };
                                used_gpu_inner
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                };

                #[cfg(not(feature = "device-gpu"))]
                let used_gpu = false;

                if !used_gpu {
                    let backward_fn = tape::with_tape_mut(|tap| tap.take_backward(fn_idx));
                    if let Some(backward) = backward_fn {
                        #[cfg(any(feature = "device-gpu", feature = "device-cuda"))]
                        let grad_cpu = match &grad_out_storage {
                            Storage::Cpu(arr) => arr.as_ref().clone(),
                            #[cfg(feature = "device-gpu")]
                            Storage::Gpu(g, _) => {
                                g.to_cpu().unwrap_or_else(|e| {
                                    tracing::warn!("Backward CPU fallback: GpuTensor readback failed: {e}");
                                    ndarray::ArrayD::zeros(grad_out_storage.shape())
                                })
                            }
                            #[cfg(feature = "device-cuda")]
                            Storage::Cuda(_, _) => {
                                tracing::warn!("Backward CPU fallback: CUDA readback not implemented");
                                ndarray::ArrayD::zeros(grad_out_storage.shape())
                            }
                        };
                        #[cfg(not(any(feature = "device-gpu", feature = "device-cuda")))]
                        let grad_cpu = grad_out_storage.to_cpu();
                        let grad_inputs = backward(&grad_cpu, &saved);
                        for (i, inp) in inputs.iter().enumerate() {
                            if i < grad_inputs.len() && inp.requires_grad() {
                                let g = grad_inputs[i].clone();
                                if let Some(existing) = grads.get_mut(&inp.id()) {
                                    match existing {
                                        Storage::Cpu(ref mut e) if e.shape() == g.shape() => {
                                            *Arc::make_mut(e) += &g;
                                        }
                                        Storage::Cpu(ref mut e) => *e = Arc::new(g),
                                        #[cfg(feature = "device-gpu")]
                                        Storage::Gpu(ref mut gpu_grad, _) => {
                                            match crate::gpu::GpuContext::global() {
                                                Ok(ctx) => {
                                                    let g_gpu =
                                                        match crate::gpu::GpuTensor::from_cpu(&g) {
                                                            Ok(t) => t,
                                                            Err(e) => {
                                                                tracing::warn!("Backward GPU: from_cpu failed: {e}");
                                                                let mut e_cpu = gpu_grad
                                                                    .to_cpu()
                                                                    .unwrap_or_else(|_| {
                                                                        ndarray::ArrayD::zeros(
                                                                            gpu_grad.shape(),
                                                                        )
                                                                    });
                                                                e_cpu += &g;
                                                                *existing =
                                                                    Storage::Cpu(Arc::new(e_cpu));
                                                                continue;
                                                            }
                                                        };
                                                    match ctx.add_inplace(gpu_grad, &g_gpu) {
                                                        Ok(()) => {}
                                                        Err(e) => tracing::warn!(
                                                            "Backward GPU add_inplace failed: {e}"
                                                        ),
                                                    }
                                                }
                                                Err(e) => {
                                                    tracing::warn!("Backward GPU context lost: {e}")
                                                }
                                            }
                                        }
                                        Storage::Cuda(_, _) => {
                                            let mut e_cpu = existing.to_cpu();
                                            e_cpu += &g;
                                            *existing = Storage::Cpu(Arc::new(e_cpu));
                                        }
                                    }
                                } else {
                                    #[cfg(any(feature = "device-gpu", feature = "device-cuda"))]
                                    {
                                        let g_shape = g.shape().to_vec();
                                        let storage = match crate::gpu::GpuContext::global() {
                                            Ok(_ctx) => match crate::gpu::GpuTensor::from_cpu(&g) {
                                                Ok(g_gpu) => Storage::Gpu(g_gpu, g_shape),
                                                Err(_) => Storage::Cpu(Arc::new(g)),
                                            },
                                            Err(_) => Storage::Cpu(Arc::new(g)),
                                        };
                                        grads.insert(inp.id(), storage);
                                    }
                                    #[cfg(not(any(feature = "device-gpu", feature = "device-cuda")))]
                                    {
                                        grads.insert(inp.id(), Storage::Cpu(Arc::new(g)));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(feature = "device-gpu")]
    if gpu_batch_active {
        if let Ok(ctx) = GpuContext::global() {
            ctx.end_batch_mode();
        }
    }

    for (tid, g) in grads {
        if let Some(t) = grad_map.get(&tid) {
            #[cfg(feature = "device-gpu")]
            t.accumulate_grad_storage(&g);
            #[cfg(not(feature = "device-gpu"))]
            t.accumulate_grad(&g.to_cpu());
        }
    }
}
