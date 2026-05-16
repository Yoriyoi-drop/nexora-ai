//! EchoNetModel — orchestrator 9 blok Echo Net
//! mengimplementasikan autograd::Module untuk diferensiasi otomatis.

use ndarray::ArrayD;

use crate::autograd::Tensor;
use crate::DLResult;

use super::{
    AdaptivePhaseSeparationStabilizer, DualEntropicResonanceRetrieval,
    EchoNetConfig, EchoNetState,
    HolographicWave, InverseSpectralCollapse, IterativeResonanceReasoner,
    MultiBandHolographicWriter, PersistentResonanceMemory,
    RecursiveHolographicCompression, SemanticSpectralEmbedding,
    TopKResonanceRouting,
};

/// Pipeline Echo Net lengkap dengan autograd.
/// Pipeline: SSE → APSS → MBHW → RHC → PRM → IRR → DERR → TKRR → ISC
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

    // Tensor parameters (disync ke block internal ArrayD)
    sse_params: Vec<Tensor>,
    irr_params: Vec<Tensor>,
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
            vocab_size, embedding_dim, amplitude_dim, phase_dim, resonance_dim, 1024,
        )?;

        let apss = AdaptivePhaseSeparationStabilizer::new(embedding_dim, 0.5, 0.3, 0.5)?;

        let bands: Vec<super::FrequencyBand> = config.band_frequencies.iter().enumerate().map(|(i, &freq)| {
            super::FrequencyBand {
                id: i,
                frequency_range: (freq * 0.5, freq * 2.0),
                kernel_size: config.kernel_size,
                description: format!("Band {}", i),
            }
        }).collect();
        let mbhw = MultiBandHolographicWriter::new(bands, config.memory_size, 0.5, 0.3)?;

        let levels: Vec<super::CompressionLevel> = (0..config.compression_levels).map(|i| {
            super::CompressionLevel {
                level: i,
                compression_ratio: config.compression_ratio,
                window_size: 4 << i,
                description: format!("Level {}", i),
                target_features: (embedding_dim as f32 * (1.0 - config.compression_ratio * i as f32 / config.compression_levels as f32)).max(8.0) as usize,
            }
        }).collect();
        let rhc = RecursiveHolographicCompression::new(levels, 4, 0.5, 0.5)?;

        let prm = PersistentResonanceMemory::new(
            config.memory_size, config.decay_alpha, config.write_threshold, 0.5,
        )?;

        let irr = IterativeResonanceReasoner::new(
            embedding_dim, config.reasoning_steps, config.reasoning_alpha, 0.01,
        )?;

        let derr = DualEntropicResonanceRetrieval::new(
            config.energy_weight, config.entropy_weight, config.coherence_weight, 0.5, config.memory_size,
        )?;

        let tkrr = TopKResonanceRouting::new(
            config.top_k, config.routing_threshold, 0.3, 0.1,
        )?;

        let isc_cfg = super::SpectralCollapseConfig {
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

        // SSE params jadi Tensor
        let sse_params = sse.get_parameters().iter().map(|&arr| {
            let t = Tensor::new(arr.clone().into_dyn());
            t.set_requires_grad(true);
            t
        }).collect();

        // IRR params jadi Tensor
        let irr_params = vec![
            irr.get_query_weights(),
            irr.get_refinement_weights(),
            irr.get_output_weights(),
        ].into_iter().map(|t| {
            t.set_requires_grad(true);
            t
        }).collect();

        // ISC params jadi Tensor
        let isc_params = vec![
            isc.get_output_weights(),
            isc.get_output_bias(),
        ].into_iter().map(|t| { t.set_requires_grad(true); t }).collect();

        Ok(Self {
            state: EchoNetState::new(&config)?,
            sse, apss, mbhw, rhc, prm, irr, derr, tkrr, isc,
            sse_params, irr_params, isc_params,
            config,
        })
    }

    /// Sync Tensor params → block internal Array2
    fn sync_params_to_blocks(&mut self) {
        self.sse.set_parameters_from_tensors(&self.sse_params);

        let irr_weights = [&self.irr_params[0], &self.irr_params[1], &self.irr_params[2]];
        self.irr.set_query_weights(irr_weights[0]);
        self.irr.set_refinement_weights(irr_weights[1]);
        self.irr.set_output_weights(irr_weights[2]);

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
        let emb_flat = wave.amplitude.clone().into_shape(wave.amplitude.len()).expect("reshape to 1D").into_dyn();
        self.apss.forward(&mut HolographicWave {
            amplitude: wave.amplitude.clone(),
            phase: wave.phase.clone(),
            frequency: wave.frequency.clone(),
        }, &emb_flat)?;

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
        let candidates: Vec<ArrayD<f32>> = band_memories.iter()
            .map(|m| {
                let data: Vec<f32> = m.iter().map(|c| c.real + c.imag).collect();
                ArrayD::from_shape_vec(vec![data.len()], data).unwrap_or(ArrayD::zeros(vec![1]))
            })
            .collect();
        let retrieved = self.derr.forward(&wave, &candidates)?;

        // 8. TKRR
        let all_data = vec![
            flat_memory.clone(),
            retrieved.clone(),
        ];
        let routed = self.tkrr.forward(&wave, &all_data, &all_data)?;

        // 9. ISC — returns Array1<f32> (sudah termasuk output_weights + bias + softmax)
        let logits_1d = self.isc.forward(&routed, timestamp)?;
        let logits_arr = ArrayD::from_shape_vec(vec![1, logits_1d.len()], logits_1d.to_vec()).expect("data length matches shape");

        // Bungkus di Tensor, dipecah dari computation graph via no_grad
        let out = Tensor::new(logits_arr);
        // out.softmax(1) — sudah di-softmax oleh ISC

        self.state.temporal_position += token_ids.len();
        Ok(out)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.sse_params.iter().cloned());
        params.extend(self.irr_params.iter().cloned());
        params.extend(self.isc_params.iter().cloned());
        params
    }

    pub fn reset(&mut self) {
        self.state.reset();
    }
}
