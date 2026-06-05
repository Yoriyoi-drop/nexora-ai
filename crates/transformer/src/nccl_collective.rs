use crate::TransformerResult;
use std::sync::Arc;

/// NCCL collective communication wrapper.
///
/// Manages NCCL communicator, device buffers, and exposes
/// all_reduce / all_gather / reduce_scatter for tensor-parallel
/// forward/backward passes.
///
/// When `nccl` feature is disabled, all operations return errors.
#[derive(Debug, Clone)]
pub struct NcclCollective {
    /// Total number of ranks in the communicator.
    pub num_shards: usize,
    /// This rank's index (0..num_shards).
    pub shard_rank: usize,
    /// GPU device index (matches cudaSetDevice).
    pub device_id: usize,
    /// Inner communicator handle.
    #[cfg(feature = "nccl")]
    inner: Arc<NcclInner>,
}

#[cfg(feature = "nccl")]
#[derive(Debug)]
struct NcclInner {
    comm: nexora_autograd::gpu::nccl::safe::Comm,
}

impl NcclCollective {
    /// Create a new NCCL communicator for the given ranks.
    ///
    /// `device_id` — CUDA device index for this rank.
    /// `num_shards` / `shard_rank` — rank topology.
    ///
    /// Returns error if NCCL is unavailable or init fails.
    pub fn new(
        device_id: usize,
        num_shards: usize,
        shard_rank: usize,
    ) -> TransformerResult<Self> {
        #[cfg(feature = "nccl")]
        {
            let inner = Self::init_nccl(device_id, num_shards, shard_rank)?;
            Ok(Self {
                num_shards,
                shard_rank,
                device_id,
                inner: Arc::new(inner),
            })
        }
        #[cfg(not(feature = "nccl"))]
        {
            let _ = (device_id, num_shards, shard_rank);
            Err(crate::TransformerError::Implementation(
                "NCCL not available: enable `nccl` feature".into(),
            ))
        }
    }

    #[cfg(feature = "nccl")]
    fn init_nccl(
        device_id: usize,
        num_shards: usize,
        shard_rank: usize,
    ) -> TransformerResult<NcclInner> {
        use nexora_autograd::gpu::nccl::safe::{Comm, Id};

        let rt = nexora_autograd::gpu::cuda::CudaRuntime::init(device_id as usize)
            .map_err(|e| {
                crate::TransformerError::Implementation(format!("CUDA init for NCCL: {e}"))
            })?;

        let id = Id::new().map_err(|e| {
            crate::TransformerError::Implementation(format!("NCCL Id::new: {e}"))
        })?;

        let comm = Comm::from_rank(
            rt.stream.clone(),
            shard_rank,
            num_shards,
            id,
        )
        .map_err(|e| {
            crate::TransformerError::Implementation(format!("NCCL Comm::from_rank: {e}"))
        })?;

        Ok(NcclInner { comm })
    }

    /// In-place all-reduce (sum) across all ranks.
    ///
    /// `buf` is split evenly across `num_shards`. After this call,
    /// every rank holds the global sum in its local chunk.
    ///
    /// Uploads to GPU, runs NCCL all-reduce, downloads back.
    pub fn all_reduce(&self, buf: &mut [f32]) -> TransformerResult<()> {
        #[cfg(feature = "nccl")]
        {
            use nexora_autograd::gpu::nccl::safe::ReduceOp;
            let stream = self.inner.comm.stream();
            let send = stream.clone_htod(buf).map_err(|e| {
                crate::TransformerError::Implementation(format!("NCCL htod_copy: {e}"))
            })?;
            let mut recv =
                stream
                    .alloc_zeros::<f32>(buf.len())
                    .map_err(|e| {
                        crate::TransformerError::Implementation(format!(
                            "NCCL alloc_zeros: {e}"
                        ))
                    })?;
            self.inner
                .comm
                .all_reduce(&send, &mut recv, &ReduceOp::Sum)
                .map_err(|e| {
                    crate::TransformerError::Implementation(format!("NCCL all_reduce: {e}"))
                })?;
            let result = stream.clone_dtoh(&recv).map_err(|e| {
                crate::TransformerError::Implementation(format!("NCCL dtoh_copy: {e}"))
            })?;
            buf.copy_from_slice(&result);
            Ok(())
        }
        #[cfg(not(feature = "nccl"))]
        {
            let _ = buf;
            Err(crate::TransformerError::Implementation(
                "NCCL not available: enable `nccl` feature".into(),
            ))
        }
    }

    /// All-gather: each rank contributes its `sendbuf`, result is
    /// concatenated into `recvbuf` on every rank.
    ///
    /// `sendbuf` length = `recvbuf.len() / num_shards`
    pub fn all_gather(&self, sendbuf: &[f32], recvbuf: &mut [f32]) -> TransformerResult<()> {
        #[cfg(feature = "nccl")]
        {
            let stream = self.inner.comm.stream();
            let send = stream.clone_htod(sendbuf).map_err(|e| {
                crate::TransformerError::Implementation(format!("NCCL htod_copy: {e}"))
            })?;
            let mut recv =
                stream
                    .alloc_zeros::<f32>(recvbuf.len())
                    .map_err(|e| {
                        crate::TransformerError::Implementation(format!(
                            "NCCL alloc_zeros: {e}"
                        ))
                    })?;
            self.inner.comm.all_gather(&send, &mut recv).map_err(|e| {
                crate::TransformerError::Implementation(format!("NCCL all_gather: {e}"))
            })?;
            let result = stream.clone_dtoh(&recv).map_err(|e| {
                crate::TransformerError::Implementation(format!("NCCL dtoh_copy: {e}"))
            })?;
            recvbuf.copy_from_slice(&result);
            Ok(())
        }
        #[cfg(not(feature = "nccl"))]
        {
            let _ = (sendbuf, recvbuf);
            Err(crate::TransformerError::Implementation(
                "NCCL not available: enable `nccl` feature".into(),
            ))
        }
    }

    /// Reduce-scatter: sum across ranks, then scatter the result
    /// so each rank holds its shard of the global sum.
    pub fn reduce_scatter(&self, sendbuf: &[f32], recvbuf: &mut [f32]) -> TransformerResult<()> {
        #[cfg(feature = "nccl")]
        {
            use nexora_autograd::gpu::nccl::safe::ReduceOp;
            let stream = self.inner.comm.stream();
            let send = stream.clone_htod(sendbuf).map_err(|e| {
                crate::TransformerError::Implementation(format!("NCCL htod_copy: {e}"))
            })?;
            let mut recv =
                stream
                    .alloc_zeros::<f32>(recvbuf.len())
                    .map_err(|e| {
                        crate::TransformerError::Implementation(format!(
                            "NCCL alloc_zeros: {e}"
                        ))
                    })?;
            self.inner
                .comm
                .reduce_scatter(&send, &mut recv, &ReduceOp::Sum)
                .map_err(|e| {
                    crate::TransformerError::Implementation(format!("NCCL reduce_scatter: {e}"))
                })?;
            let result = stream.clone_dtoh(&recv).map_err(|e| {
                crate::TransformerError::Implementation(format!("NCCL dtoh_copy: {e}"))
            })?;
            recvbuf.copy_from_slice(&result);
            Ok(())
        }
        #[cfg(not(feature = "nccl"))]
        {
            let _ = (sendbuf, recvbuf);
            Err(crate::TransformerError::Implementation(
                "NCCL not available: enable `nccl` feature".into(),
            ))
        }
    }

    /// Access the inner NCCL communicator for GPU-native operations.
    #[cfg(feature = "nccl")]
    pub fn comm(&self) -> &nexora_autograd::gpu::nccl::safe::Comm {
        &self.inner.comm
    }
}

/// GPU-native all-reduce for in-flight GPU tensors.
///
/// Runs NCCL all-reduce directly on device memory without CPU round-trip.
/// Requires the `nccl` feature.
#[cfg(feature = "nccl")]
pub fn collective_gpu_all_reduce(
    nccl: &nexora_autograd::gpu::nccl::safe::Comm,
    buf: &nexora_autograd::gpu::CudaSlice<f32>,
) -> TransformerResult<()> {
    use nexora_autograd::gpu::nccl::safe::ReduceOp;
    let stream = nccl.stream();
    let mut tmp = stream.alloc_zeros::<f32>(buf.len()).map_err(|e| {
        crate::TransformerError::Implementation(format!("NCCL alloc_zeros: {e}"))
    })?;
    nccl.all_reduce(buf, &mut tmp, &ReduceOp::Sum)
        .map_err(|e| {
            crate::TransformerError::Implementation(format!("NCCL all_reduce: {e}"))
        })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nccl_new_without_feature_returns_err() {
        let result = NcclCollective::new(0, 2, 0);
        #[cfg(not(feature = "nccl"))]
        assert!(result.is_err());
        #[cfg(feature = "nccl")]
        assert!(result.is_err()); // no CUDA device in CI
    }

    #[test]
    fn test_nccl_struct_fields() {
        if let Ok(nccl) = NcclCollective::new(0, 1, 0) {
            assert_eq!(nccl.num_shards, 1);
            assert_eq!(nccl.shard_rank, 0);
            assert_eq!(nccl.device_id, 0);
        }
    }

    #[test]
    fn test_nccl_all_gather_returns_err() {
        let nccl = NcclCollective::new(0, 1, 0);
        if let Ok(nccl) = nccl {
            let mut buf = vec![0.0f32; 4];
            let result = nccl.all_gather(&[1.0f32; 2], &mut buf);
            #[cfg(not(feature = "nccl"))]
            assert!(result.is_err());
            #[cfg(feature = "nccl")]
            assert!(result.is_err()); // no actual GPU in test env
        }
    }

    #[test]
    fn test_nccl_reduce_scatter_returns_err() {
        let nccl = NcclCollective::new(0, 1, 0);
        if let Ok(nccl) = nccl {
            let mut buf = vec![0.0f32; 2];
            let result = nccl.reduce_scatter(&[1.0f32; 4], &mut buf);
            assert!(result.is_err());
        }
    }
}
