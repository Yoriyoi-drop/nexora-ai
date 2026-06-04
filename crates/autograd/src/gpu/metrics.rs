use std::time::Duration;


use super::gpu_tensor::readback_with_timeout;
use super::gpu_types::*;
#[cfg(feature = "cuda")]
use super::cuda::CudaTensor;
/// Maximum workgroups per dimension for wgpu/WebGPU (65535).
/// Any dispatch exceeding this must be chunked across multiple dispatches
/// using buffer-slice bindings with per-chunk offsets.
const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

/// Default workgroup size used by most Nexora WGSL shaders.
const DEFAULT_WORKGROUP_SIZE: u32 = 256;

impl GpuContext {
    /// Batch dispatch N times with single timestamp query around all dispatches
    /// Returns total GPU execution time for all N dispatches
    pub fn dispatch_batch_profiled(
        &self,
        pipeline: &CompiledPipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: (u32, u32, u32),
        n: usize,
    ) -> std::time::Duration {
        if let Some(query_set) = &self.profiling_query_set {
            let (start_idx, end_idx) = self.next_query_index();
            let query_count = 2;
            let buf_size = (query_count as u64 * wgpu::QUERY_SIZE as u64).div_ceil(256) * 256;
            let query_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu_timestamp_query_buffer_batch"),
                size: buf_size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                cpass.write_timestamp(query_set, start_idx); // Start timestamp
                for _ in 0..n {
                    cpass.set_pipeline(&pipeline.pipeline);
                    cpass.set_bind_group(0, bind_group, &[]);
                    cpass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
                }
                cpass.write_timestamp(query_set, end_idx); // End timestamp
            }
            encoder.resolve_query_set(query_set, start_idx..(end_idx + 1), &query_buffer, 0);

            // Readback timestamp query result (separate from kernel execution)
            let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu_timestamp_query_readback_batch"),
                size: buf_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // Combine copy into main encoder to avoid extra submit
            encoder.copy_buffer_to_buffer(&query_buffer, 0, &readback_buffer, 0, buf_size);
            self.queue.submit(Some(encoder.finish()));

            let slice = readback_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

            // Safe readback with error fallback: fall through to CPU timing on failure
            if readback_with_timeout(&self.device, &rx).is_ok() {
                if let Some((start_ts, end_ts)) =
                    self.read_timestamps_from_buffer(&slice, &readback_buffer)
                {
                    let period_ns = self.queue.get_timestamp_period() as f64;
                    let elapsed_ns: f64 = if end_ts >= start_ts {
                        (end_ts - start_ts) * period_ns
                    } else {
                        0.0
                    };
                    return std::time::Duration::from_nanos(elapsed_ns.round() as u64);
                }
            }
            readback_buffer.unmap();
        }

        let start = std::time::Instant::now();
        for _ in 0..n {
            self.dispatch(pipeline, bind_group, workgroups);
        }
        start.elapsed()
    }

    /// Dispatch multiple different operations in a single command encoder
    /// Each operation is a tuple of (pipeline, bind_group, workgroups)
    /// Returns total GPU execution time for all operations
    pub fn dispatch_multi_ops(
        &self,
        ops: &[(&CompiledPipeline, &wgpu::BindGroup, (u32, u32, u32))],
    ) -> std::time::Duration {
        if let Some(query_set) = &self.profiling_query_set {
            let (start_idx, end_idx) = self.next_query_index();
            let query_count = 2;
            let buf_size = (query_count as u64 * wgpu::QUERY_SIZE as u64).div_ceil(256) * 256;
            let query_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu_timestamp_query_buffer_multi"),
                size: buf_size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                cpass.write_timestamp(query_set, start_idx); // Start timestamp
                for (pipeline, bind_group, workgroups) in ops {
                    cpass.set_pipeline(&pipeline.pipeline);
                    cpass.set_bind_group(0, *bind_group, &[]);
                    cpass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
                }
                cpass.write_timestamp(query_set, end_idx); // End timestamp
            }
            encoder.resolve_query_set(query_set, start_idx..(end_idx + 1), &query_buffer, 0);
            self.queue.submit(Some(encoder.finish()));

            // Readback timestamp query result (separate from kernel execution)
            let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu_timestamp_query_readback_multi"),
                size: buf_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let mut copy_encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            copy_encoder.copy_buffer_to_buffer(&query_buffer, 0, &readback_buffer, 0, buf_size);
            self.queue.submit(Some(copy_encoder.finish()));

            let slice = readback_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            if self
                .device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(5)),
                })
                .is_err()
            {
                tracing::warn!("GPU poll timed out after 5s in dispatch_multi_ops, falling through to polling loop");
            }

            loop {
                match rx.try_recv() {
                    Ok(Ok(())) => break,
                    Ok(Err(e)) => {
                        tracing::warn!("Query buffer mapping failed: {e}");
                        readback_buffer.unmap();
                        return std::time::Duration::ZERO;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        self.device.poll(wgpu::PollType::Wait {
                            submission_index: None,
                            timeout: Some(Duration::from_micros(1000)),
                        });
                        continue;
                    }
                    Err(e) => {
                        tracing::warn!("Query channel disconnected: {e}");
                        readback_buffer.unmap();
                        return std::time::Duration::ZERO;
                    }
                }
            }

            let (start_ts, end_ts) = {
                let mapped = slice.get_mapped_range();
                let bytes: &[u8] = &mapped;
                let first = u64::from_le_bytes(bytes[0..8].try_into().unwrap_or_else(|_| {
                    tracing::warn!("Query timestamp readback underflow, returning 0");
                    [0u8; 8]
                }));
                let second = u64::from_le_bytes(bytes[8..16].try_into().unwrap_or_else(|_| {
                    tracing::warn!("Query timestamp readback underflow (second), returning 0");
                    [0u8; 8]
                }));
                (first as f64, second as f64)
            };
            readback_buffer.unmap();
            let period_ns = self.queue.get_timestamp_period() as f64;
            let elapsed_ns = if end_ts >= start_ts {
                (end_ts - start_ts) * period_ns
            } else {
                0.0
            };
            return std::time::Duration::from_nanos(elapsed_ns.round() as u64);
        }

        let start = std::time::Instant::now();
        for (pipeline, bind_group, workgroups) in ops {
            self.dispatch(pipeline, bind_group, *workgroups);
        }
        start.elapsed()
    }

}
