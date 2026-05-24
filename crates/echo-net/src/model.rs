//! EchoNetModel — orchestrator 9 blok Echo Net
//! mengimplementasikan autograd::Module untuk diferensiasi otomatis.

use ndarray::ArrayD;

use crate::DeepLearningError;
use crate::DLResult;
use nexora_autograd::Tensor;

use crate::{
    AdaptivePhaseSeparationStabilizer, DualEntropicResonanceRetrieval, EchoNetConfig, EchoNetState,
    HolographicWave, InverseSpectralCollapse, IterativeResonanceReasoner,
    MultiBandHolographicWriter, PersistentResonanceMemory, RecursiveHolographicCompression,
    SemanticSpectralEmbedding, TopKResonanceRouting,
};

/// Pipeline Echo Net lengkap dengan autograd.
/// Pipeline: SSE → APSS → MBHW → RHC → PRM → IRR → DERR → TKRR → ISC
///
/// All 9 blocks are registered as parameter groups. Blocks that currently use
/// raw ndarray (not autograd Tensor) for their internal operations return empty
/// parameter lists — gradient flow stops at those boundaries. To enable full
/// autograd, convert the internal ndarray ops in those blocks to use Tensor.
pub struct EchoNetModel {
    pub config: EchoNetConfig,
    pub state: EchoNetState,

    // 9 blok pipeline
    pub sse: SemanticSpectralEmbedding,
    pub apss: AdaptivePhaseSeparationStabilizer,
    pub mbhw: MultiBandHolographicWriter,
    pub rhc: RecursiveHolographicCompression,
    pub prm: PersistentResonanceMemory,
    pub irr: IterativeResonanceReasoner,
    pub derr: DualEntropicResonanceRetrieval,
    pub tkrr: TopKResonanceRouting,
    pub isc: InverseSpectralCollapse,

    // Tensor parameters (sync ke block internal ArrayD)
    sse_params: Vec<Tensor>,
    apss_params: Vec<Tensor>,
    mbhw_params: Vec<Tensor>,
    rhc_params: Vec<Tensor>,
    prm_params: Vec<Tensor>,
    irr_params: Vec<Tensor>,
    derr_params: Vec<Tensor>,
    tkrr_params: Vec<Tensor>,
    isc_params: Vec<Tensor>,
}

impl EchoNetModel {
    pub fn new(config: EchoNetConfig) -> DLResult<Self> {
        let embedding_dim = config.embedding_dim;
        let amplitude_dim = config.amplitude_dim;
        let phase_dim = config.phase_dim;
        let resonance_dim = config.resonance_dim;
        let vocab_size = config.vocab_size;
        let output_size = config.output_size;

        let sse = SemanticSpectralEmbedding::new(
            vocab_size,
            embedding_dim,
            amplitude_dim,
            phase_dim,
            resonance_dim,
            1024,
        )?;

        let apss = AdaptivePhaseSeparationStabilizer::new(embedding_dim, 0.5, 0.3, 0.5)?;

        let bands: Vec<crate::FrequencyBand> = config
            .band_frequencies
            .iter()
            .enumerate()
            .map(|(i, &freq)| crate::FrequencyBand {
                id: i,
                frequency_range: (freq * 0.5, freq * 2.0),
                kernel_size: config.kernel_size,
                description: format!("Band {}", i),
            })
            .collect();
        let mbhw = MultiBandHolographicWriter::new(bands, config.memory_size, 0.5, 0.3)?;

        let levels: Vec<crate::CompressionLevel> = (0..config.compression_levels)
            .map(|i| crate::CompressionLevel {
                level: i,
                compression_ratio: config.compression_ratio,
                window_size: 4 << i,
                description: format!("Level {}", i),
                target_features: (embedding_dim as f32
                    * (1.0
                        - config.compression_ratio * i as f32 / config.compression_levels as f32))
                    .max(8.0) as usize,
            })
            .collect();
        let rhc = RecursiveHolographicCompression::new(levels, 4, 0.5, 0.5)?;

        let prm = PersistentResonanceMemory::new(
            config.memory_size,
            config.decay_alpha,
            config.write_threshold,
            0.5,
        )?;

        let irr = IterativeResonanceReasoner::new(
            embedding_dim,
            config.reasoning_steps,
            config.reasoning_alpha,
            0.01,
        )?;

        let derr = DualEntropicResonanceRetrieval::new(
            config.energy_weight,
            config.entropy_weight,
            config.coherence_weight,
            0.5,
            config.memory_size,
        )?;

        let tkrr = TopKResonanceRouting::new(config.top_k, config.routing_threshold, 0.3, 0.1)?;

        let isc_cfg = crate::SpectralCollapseConfig {
            output_size,
            temperature: 1.0,
            collapse_strength: 1.0,
            phase_preservation: 0.5,
            amplitude_normalization: true,
            frequency_filtering: false,
            min_frequency: 0.0,
            max_frequency: 100.0,
        };
        let isc = InverseSpectralCollapse::new(embedding_dim, isc_cfg)?;

        // SSE params jadi Tensor (block 1)
        let sse_params = sse
            .get_parameters()
            .iter()
            .map(|&arr| {
                let t = Tensor::new(arr.clone().into_dyn());
                t.set_requires_grad(true);
                t
            })
            .collect();

        // APSS params (block 2) — trainable weights
        let apss_params = apss
            .get_parameters()
            .iter()
            .map(|arr| {
                let t = Tensor::new(arr.clone().into_dyn());
                t.set_requires_grad(true);
                t
            })
            .collect::<Vec<_>>();

        // MBHW params (block 3) — frequency_filters registered as trainable
        let mbhw_params = mbhw
            .get_parameters()
            .iter()
            .map(|arr| {
                let t = Tensor::new(arr.clone().into_dyn());
                t.set_requires_grad(true);
                t
            })
            .collect::<Vec<_>>();

        // RHC params (block 4) — feature_extractors registered as trainable
        let rhc_params = rhc
            .get_parameters()
            .iter()
            .map(|arr| {
                let t = Tensor::new(arr.clone().into_dyn());
                t.set_requires_grad(true);
                t
            })
            .collect::<Vec<_>>();

        // PRM params (block 5) — novelty_weights + resonance_kernel wrapped as Tensor
        let prm_params = prm
            .get_parameters()
            .iter()
            .map(|arr| {
                let t = Tensor::new(arr.clone().into_dyn());
                t.set_requires_grad(true);
                t
            })
            .collect::<Vec<_>>();

        // IRR params (block 6)
        let irr_params = vec![
            irr.get_query_weights(),
            irr.get_refinement_weights(),
            irr.get_output_weights(),
        ]
        .into_iter()
        .map(|t| {
            t.set_requires_grad(true);
            t
        })
        .collect();

        // DERR params (block 7) — trainable weights
        let derr_params = derr
            .get_parameters()
            .iter()
            .map(|arr| {
                let t = Tensor::new(arr.clone().into_dyn());
                t.set_requires_grad(true);
                t
            })
            .collect::<Vec<_>>();

        // TKRR params (block 8) — relevance_weights wrapped as Tensor
        let tkrr_params = tkrr
            .get_parameters()
            .iter()
            .map(|arr| {
                let t = Tensor::new(arr.clone().into_dyn());
                t.set_requires_grad(true);
                t
            })
            .collect::<Vec<_>>();

        // ISC params (block 9)
        let isc_params = vec![isc.get_output_weights(), isc.get_output_bias()]
            .into_iter()
            .map(|t| {
                t.set_requires_grad(true);
                t
            })
            .collect();

        Ok(Self {
            state: EchoNetState::new(&config)?,
            sse,
            apss,
            mbhw,
            rhc,
            prm,
            irr,
            derr,
            tkrr,
            isc,
            sse_params,
            apss_params,
            mbhw_params,
            rhc_params,
            prm_params,
            irr_params,
            derr_params,
            tkrr_params,
            isc_params,
            config,
        })
    }

    /// Sync Tensor params → block internal Array2
    fn sync_params_to_blocks(&mut self) {
        self.sse.set_parameters_from_tensors(&self.sse_params);

        // APSS (block 2) — no param sync needed

        // MBHW (block 3)
        self.mbhw.set_parameters_from_tensors(&self.mbhw_params);

        // RHC (block 4)
        self.rhc.set_parameters_from_tensors(&self.rhc_params);

        // PRM (block 5) — if params are registered, sync them
        self.prm.set_parameters_from_tensors(&self.prm_params);

        // IRR (block 6)
        if self.irr_params.len() >= 3 {
            self.irr.set_query_weights(&self.irr_params[0]);
            self.irr.set_refinement_weights(&self.irr_params[1]);
            self.irr.set_output_weights(&self.irr_params[2]);
        }

        // DERR (block 7) — no param sync needed

        // TKRR (block 8) — if params are registered, sync them
        self.tkrr.set_parameters_from_tensors(&self.tkrr_params);

        // ISC (block 9)
        self.isc.set_output_weights(&self.isc_params[0]);
        self.isc.set_output_bias(&self.isc_params[1]);
    }

    /// Forward pass lengkap melalui 9 blok
    pub fn forward(&mut self, token_ids: &[usize]) -> DLResult<Tensor> {
        self.sync_params_to_blocks();

        let positions: Vec<usize> = (0..token_ids.len()).collect();
        let timestamp = self.state.temporal_position;

        // 1. SSE
        let wave = self.sse.forward(token_ids, &positions)?;

        // 2. APSS
        let emb_flat: ArrayD<f32> = wave
            .amplitude
            .clone()
            .into_shape(wave.amplitude.len())
            .map_err(DeepLearningError::from)?
            .into_dyn();
        self.apss.forward(
            &mut HolographicWave {
                amplitude: wave.amplitude.clone(),
                phase: wave.phase.clone(),
                frequency: wave.frequency.clone(),
            },
            &emb_flat,
        )?;

        // 3. MBHW — returns Vec<Array2<Complex>>
        let band_memories = self.mbhw.forward(&wave, timestamp)?;

        // 4. RHC
        if let Ok(wave_flat) = wave.amplitude.clone().into_shape(wave.amplitude.len()) {
            if let Ok(wave_2d) = wave_flat.clone().into_shape((wave_flat.len(), 1)) {
                self.rhc.forward(&wave_2d, timestamp)?;
            }
        }

        // 5. PRM
        self.prm.forward(&wave, timestamp)?;

        // 6. IRR
        let flat_memory = if let Some(m) = band_memories.first() {
            let data: Vec<f32> = m.iter().map(|c| c.real + c.imag).collect();
            ArrayD::from_shape_vec(vec![data.len()], data).unwrap_or(ArrayD::zeros(vec![1]))
        } else {
            ArrayD::zeros(vec![1])
        };
        if let Ok(wave_amp_flat) = wave.amplitude.clone().into_shape(wave.amplitude.len()) {
            let wave_amp_dyn = wave_amp_flat.into_dyn();
            self.irr.forward(&wave_amp_dyn, &flat_memory)?;
        }

        // 7. DERR
        let candidates: Vec<ArrayD<f32>> = band_memories
            .iter()
            .map(|m| {
                let data: Vec<f32> = m.iter().map(|c| c.real + c.imag).collect();
                ArrayD::from_shape_vec(vec![data.len()], data).unwrap_or(ArrayD::zeros(vec![1]))
            })
            .collect();
        let retrieved = self.derr.forward(&wave, &candidates)?;

        // 8. TKRR
        let all_data = vec![flat_memory.clone(), retrieved.clone()];
        let routed = self.tkrr.forward(&wave, &all_data, &all_data)?;

        // 9. ISC — forward_tensor returns Tensor with gradient tracking
        //    through output_projection (matmul + bias + softmax)
        let out = self.isc.forward_tensor(
            &routed,
            timestamp,
            &self.isc_params[0],
            &self.isc_params[1],
        )?;

        self.state.temporal_position += token_ids.len();
        Ok(out)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let total = self.sse_params.len()
            + self.apss_params.len()
            + self.mbhw_params.len()
            + self.rhc_params.len()
            + self.prm_params.len()
            + self.irr_params.len()
            + self.derr_params.len()
            + self.tkrr_params.len()
            + self.isc_params.len();
        let mut params = Vec::with_capacity(total);
        params.extend(self.sse_params.iter().cloned());
        params.extend(self.apss_params.iter().cloned());
        params.extend(self.mbhw_params.iter().cloned());
        params.extend(self.rhc_params.iter().cloned());
        params.extend(self.prm_params.iter().cloned());
        params.extend(self.irr_params.iter().cloned());
        params.extend(self.derr_params.iter().cloned());
        params.extend(self.tkrr_params.iter().cloned());
        params.extend(self.isc_params.iter().cloned());
        params
    }

    pub fn reset(&mut self) {
        self.state.reset();
    }
}
