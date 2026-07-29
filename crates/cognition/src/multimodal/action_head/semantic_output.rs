use crate::multimodal::error::Result;
use crate::multimodal::types::*;
use rand::Rng;
use std::collections::HashMap;
use std::path::Path;

const VOCAB_SIZE: usize = 50257;

/// Vocabulary that maps token IDs to words.
/// Can load from a file (one word per line, line index = token ID)
/// or fall back to the common word hash + embedding lookup.
pub struct WordVocabulary {
    words: Vec<String>,
    common: HashMap<usize, String>,
}

impl WordVocabulary {
    pub fn new() -> Self {
        let mut common: HashMap<usize, String> = HashMap::new();
        let common_pairs = [
            (0_usize, "the"), (1, "a"), (2, "an"), (3, "is"), (4, "are"),
            (5, "was"), (6, "were"), (7, "be"), (8, "been"), (9, "being"),
            (10, "have"), (11, "has"), (12, "had"), (13, "do"), (14, "does"),
            (15, "did"), (16, "will"), (17, "would"), (18, "can"), (19, "could"),
            (20, "shall"), (21, "should"), (22, "may"), (23, "might"), (24, "must"),
            (25, "i"), (26, "you"), (27, "he"), (28, "she"), (29, "it"),
            (30, "we"), (31, "they"), (32, "me"), (33, "him"), (34, "her"),
            (35, "us"), (36, "them"), (37, "my"), (38, "your"), (39, "his"),
            (40, "its"), (41, "our"), (42, "their"), (43, "this"), (44, "that"),
            (45, "these"), (46, "those"), (47, "some"), (48, "any"), (49, "no"),
            (50, "all"), (51, "both"), (52, "each"), (53, "every"), (54, "few"),
            (55, "more"), (56, "most"), (57, "other"), (58, "such"), (59, "what"),
            (60, "which"), (61, "who"), (62, "whom"), (63, "when"), (64, "where"),
            (65, "why"), (66, "how"), (67, "and"), (68, "or"), (69, "but"),
            (70, "if"), (71, "because"), (72, "as"), (73, "until"), (74, "while"),
            (75, "of"), (76, "at"), (77, "by"), (78, "for"), (79, "with"),
            (80, "about"), (81, "between"), (82, "into"), (83, "through"), (84, "during"),
            (85, "before"), (86, "after"), (87, "above"), (88, "below"), (89, "to"),
            (90, "from"), (91, "up"), (92, "down"), (93, "in"), (94, "out"),
            (95, "on"), (96, "off"), (97, "over"), (98, "under"), (99, "again"),
        ];
        for (k, v) in common_pairs {
            common.insert(k, v.to_string());
        }

        Self { words: Vec::new(), common }
    }

    /// Load vocabulary from a file (one word per line, line number = token ID)
    pub fn load_from_file(path: impl AsRef<Path>) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let words: Vec<String> = content.lines().map(|l| l.trim().to_string()).collect();
        Ok(Self { words, common: HashMap::new() })
    }

    /// Lookup a token ID — first try file-based vocab, then common words, then hash-based fallback
    pub(crate) fn lookup(&self, token_id: usize, embedding: &EmbeddingLookup) -> String {
        // Try file-based vocabulary first
        if !self.words.is_empty() {
            if token_id < self.words.len() {
                let w = self.words[token_id].clone();
                if !w.is_empty() {
                    return w;
                }
            }
        }

        // Try common words
        if let Some(word) = self.common.get(&(token_id % 100)) {
            return word.clone();
        }

        // Hash-based fallback from embedding
        let emb = embedding.lookup(token_id);
        let hash: u64 = emb.iter().map(|v| v.to_bits() as u64).sum();
        let word_idx = hash % 500;
        let fallback_words = [
            "analyze", "generate", "process", "compute", "transform",
            "extract", "predict", "classify", "optimize", "evaluate",
            "integrate", "configure", "deploy", "synthesize", "aggregate",
            "parse", "render", "execute", "simulate", "calibrate",
        ];
        fallback_words[(word_idx as usize) % fallback_words.len()].to_string()
    }
}

pub(crate) struct EmbeddingLookup {
    table: Vec<Vec<f32>>,
    dim: usize,
}

impl EmbeddingLookup {
    fn new(dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / dim as f32).sqrt();
        let table = (0..VOCAB_SIZE)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 * scale - scale).collect())
            .collect();
        Self { table, dim }
    }

    fn lookup(&self, token_id: usize) -> Vec<f32> {
        let idx = token_id % VOCAB_SIZE;
        self.table[idx].clone()
    }
}

struct ImageDecoderMLP {
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
}

impl ImageDecoderMLP {
    fn new(latent_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale1 = (2.0 / latent_dim as f32).sqrt();
        let scale2 = (2.0 / hidden_dim as f32).sqrt();
        Self {
            w1: (0..latent_dim)
                .map(|_| (0..hidden_dim).map(|_| rng.gen::<f32>() * 2.0 * scale1 - scale1).collect())
                .collect(),
            b1: (0..hidden_dim).map(|_| 0.0).collect(),
            w2: (0..hidden_dim)
                .map(|_| (0..output_dim).map(|_| rng.gen::<f32>() * 2.0 * scale2 - scale2).collect())
                .collect(),
            b2: (0..output_dim).map(|_| 0.0).collect(),
        }
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let hidden_dim = self.b1.len();
        let output_dim = self.b2.len();
        let mut h = vec![0.0f32; hidden_dim];
        for i in 0..hidden_dim {
            let mut s = self.b1[i];
            for j in 0..x.len() {
                s += x[j] * self.w1[j][i];
            }
            let gelu = s * 0.5 * (1.0 + (s * 0.7978845608 * (1.0 + 0.044715 * s * s)).tanh());
            h[i] = gelu;
        }
        let mut out = vec![0.0f32; output_dim];
        for i in 0..output_dim {
            let mut s = self.b2[i];
            for j in 0..hidden_dim {
                s += h[j] * self.w2[j][i];
            }
            out[i] = s;
        }
        out
    }
}

pub struct SemanticOutputGenerator {
    _config: crate::multimodal::config::ActionConfig,
    text_generator: TextOutputGenerator,
    image_generator: ImageOutputGenerator,
    audio_generator: AudioOutputGenerator,
    video_generator: VideoOutputGenerator,
}

impl SemanticOutputGenerator {
    pub fn new(config: crate::multimodal::config::ActionConfig) -> Result<Self> {
        let embed_dim = 64usize;
        Ok(Self {
            text_generator: TextOutputGenerator::new(embed_dim)?,
            image_generator: ImageOutputGenerator::new(embed_dim)?,
            audio_generator: AudioOutputGenerator::new(embed_dim)?,
            video_generator: VideoOutputGenerator::new(embed_dim)?,
            _config: config,
        })
    }

    pub fn generate(
        &mut self,
        tokens: &[UnifiedToken],
        inputs: &MultiModalInputs,
    ) -> Result<SemanticOutputs> {
        let mut outputs = SemanticOutputs {
            text: None,
            image: None,
            audio: None,
            video: None,
        };

        if self.should_generate_text(tokens, inputs) {
            outputs.text = Some(self.text_generator.generate(tokens, inputs)?);
        }
        if self.should_generate_image(tokens, inputs) {
            outputs.image = Some(self.image_generator.generate(tokens, inputs)?);
        }
        if self.should_generate_audio(tokens, inputs) {
            outputs.audio = Some(self.audio_generator.generate(tokens, inputs)?);
        }
        if self.should_generate_video(tokens, inputs) {
            outputs.video = Some(self.video_generator.generate(tokens, inputs)?);
        }

        Ok(outputs)
    }

    fn should_generate_text(&self, tokens: &[UnifiedToken], inputs: &MultiModalInputs) -> bool {
        inputs.text.is_some() || tokens.iter().any(|t| t.modality == ModalityType::Text)
    }

    fn should_generate_image(&self, tokens: &[UnifiedToken], inputs: &MultiModalInputs) -> bool {
        inputs.image.is_some() || tokens.iter().any(|t| t.modality == ModalityType::Image)
    }

    fn should_generate_audio(&self, tokens: &[UnifiedToken], inputs: &MultiModalInputs) -> bool {
        inputs.audio.is_some() || tokens.iter().any(|t| t.modality == ModalityType::Audio)
    }

    fn should_generate_video(&self, tokens: &[UnifiedToken], inputs: &MultiModalInputs) -> bool {
        inputs.video.is_some() || tokens.iter().any(|t| t.modality == ModalityType::Video)
    }
}

#[derive(Debug, Clone)]
pub struct SemanticOutputs {
    pub text: Option<TextOutput>,
    pub image: Option<ImageOutput>,
    pub audio: Option<AudioOutput>,
    pub video: Option<VideoOutput>,
}

pub struct TextOutputGenerator {
    max_length: usize,
    _temperature: f32,
    embedding: EmbeddingLookup,
    vocabulary: WordVocabulary,
}

impl TextOutputGenerator {
    pub fn new(embed_dim: usize) -> Result<Self> {
        Ok(Self {
            max_length: 512,
            _temperature: 0.7,
            embedding: EmbeddingLookup::new(embed_dim),
            vocabulary: WordVocabulary::new(),
        })
    }

    /// Load vocabulary from file for better token-to-word mapping
    pub fn load_vocab(&mut self, path: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
        self.vocabulary = WordVocabulary::load_from_file(path)?;
        Ok(())
    }

    pub fn generate(
        &mut self,
        tokens: &[UnifiedToken],
        inputs: &MultiModalInputs,
    ) -> Result<TextOutput> {
        let text_tokens: Vec<_> = tokens
            .iter()
            .filter(|t| t.modality == ModalityType::Text)
            .collect();

        let mut generated_text = String::new();

        if let Some(ref text_input) = inputs.text {
            generated_text.push_str(&text_input.text);
            generated_text.push(' ');
        }

        for token in &text_tokens {
            let word = self.token_to_word(token.token_id)?;
            generated_text.push_str(&word);
            generated_text.push(' ');
        }

        generated_text = generated_text.trim().to_string();
        if generated_text.len() > self.max_length {
            generated_text.truncate(self.max_length - 3);
            generated_text.push_str("...");
        }

        let token_probs = if text_tokens.is_empty() {
            None
        } else {
            let probs: Vec<f32> = text_tokens.iter().map(|_| 0.8).collect();
            Some(probs)
        };

        Ok(TextOutput {
            text: generated_text,
            token_probs,
            confidence: 0.8,
        })
    }

    fn token_to_word(&self, token_id: usize) -> Result<String> {
        Ok(self.vocabulary.lookup(token_id, &self.embedding))
    }
}

pub struct ImageOutputGenerator {
    image_size: (usize, usize),
    channels: usize,
    decoder: ImageDecoderMLP,
}

impl ImageOutputGenerator {
    pub fn new(embed_dim: usize) -> Result<Self> {
        let latent_dim = embed_dim.max(64);
        let hidden_dim = latent_dim * 4;
        let output_dim = 16 * 16 * 3;
        Ok(Self {
            image_size: (64, 64),
            channels: 3,
            decoder: ImageDecoderMLP::new(latent_dim, hidden_dim, output_dim),
        })
    }

    pub fn generate(
        &mut self,
        tokens: &[UnifiedToken],
        _inputs: &MultiModalInputs,
    ) -> Result<ImageOutput> {
        let image_tokens: Vec<_> = tokens
            .iter()
            .filter(|t| t.modality == ModalityType::Image)
            .collect();

        let width = self.image_size.0;
        let height = self.image_size.1;
        let mut image_data = vec![0u8; width * height * self.channels];

        if !image_tokens.is_empty() {
            let mut latent = vec![0.0f32; image_tokens[0].embedding.len()];
            for token in &image_tokens {
                for (i, &v) in token.embedding.iter().enumerate() {
                    if i < latent.len() {
                        latent[i] += v;
                    }
                }
            }
            let n = image_tokens.len() as f32;
            if n > 0.0 {
                for v in &mut latent {
                    *v /= n;
                }
            }

            let pixels = self.decoder.forward(&latent);
            let patch_size = 16;
            let cols = width / patch_size;
            let rows = height / patch_size;
            let patches_per_row = 4;
            let patches_per_col = 4;

            for py in 0..rows.min(patches_per_col) {
                for px in 0..cols.min(patches_per_row) {
                    let patch_idx = py * patches_per_row + px;
                    for j in 0..patch_size {
                        for i in 0..patch_size {
                            if py * patch_size + j < height && px * patch_size + i < width {
                                for c in 0..3 {
                                    let pixel_idx = ((py * patch_size + j) * width + (px * patch_size + i)) * 3 + c;
                                    let patch_pixel_idx = patch_idx * (patch_size * patch_size * 3) + j * patch_size * 3 + i * 3 + c;
                                    let val = if patch_pixel_idx < pixels.len() {
                                        (pixels[patch_pixel_idx] * 127.0 + 128.0).clamp(0.0, 255.0) as u8
                                    } else {
                                        ((px * patch_size + i + py * patch_size + j) % 256) as u8
                                    };
                                    if pixel_idx < image_data.len() {
                                        image_data[pixel_idx] = val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if image_data.iter().all(|&b| b == 0) {
            for (i, pixel) in image_data.chunks_mut(3).enumerate() {
                let x = i % width;
                let y = i / width;
                pixel[0] = ((x * 7 + y * 3) % 256) as u8;
                pixel[1] = ((x * 5 + y * 11) % 256) as u8;
                pixel[2] = ((x * 13 + y * 2) % 256) as u8;
            }
        }

        Ok(ImageOutput {
            data: image_data,
            format: ImageFormat::PNG,
            width,
            height,
            description: Some("Generated image from multimodal tokens".to_string()),
        })
    }
}

pub struct AudioOutputGenerator {
    sample_rate: usize,
    duration: f32,
}

impl AudioOutputGenerator {
    pub fn new(_embed_dim: usize) -> Result<Self> {
        Ok(Self {
            sample_rate: 16000,
            duration: 2.0,
        })
    }

    pub fn generate(
        &mut self,
        tokens: &[UnifiedToken],
        _inputs: &MultiModalInputs,
    ) -> Result<AudioOutput> {
        let audio_tokens: Vec<_> = tokens
            .iter()
            .filter(|t| t.modality == ModalityType::Audio)
            .collect();

        let num_samples = (self.sample_rate as f32 * self.duration) as usize;
        let mut audio_data = vec![0.0f32; num_samples];

        if !audio_tokens.is_empty() {
            let mut embedding_sum = vec![0.0f32; audio_tokens[0].embedding.len()];
            for token in &audio_tokens {
                for (i, &v) in token.embedding.iter().enumerate() {
                    if i < embedding_sum.len() {
                        embedding_sum[i] += v;
                    }
                }
            }
            let nt = audio_tokens.len() as f32;
            if nt > 0.0 {
                for v in &mut embedding_sum {
                    *v /= nt;
                }
            }

            let fundamental: f32 = embedding_sum.iter().take(4).map(|v| v.abs()).sum::<f32>()
                * 50.0
                + 100.0;
            let harmonics: Vec<f32> = embedding_sum
                .chunks(8)
                .enumerate()
                .map(|(i, chunk)| {
                    let amp: f32 = chunk.iter().map(|v| v.abs()).sum::<f32>() / chunk.len() as f32;
                    amp * (0.5 / (i + 1) as f32)
                })
                .collect();

            for i in 0..num_samples {
                let t = i as f32 / self.sample_rate as f32;
                let mut sample = 0.0f32;
                for (h, &amp) in harmonics.iter().enumerate() {
                    let freq = fundamental * (h + 1) as f32;
                    sample += (2.0 * std::f32::consts::PI * freq * t).sin() * amp;
                }
                let envelope = 1.0 - (t / self.duration).min(1.0);
                audio_data[i] = sample * envelope * 0.3;
            }
        }

        Ok(AudioOutput {
            data: audio_data,
            sample_rate: self.sample_rate,
            duration: self.duration,
            transcription: Some("Generated audio from multimodal tokens".to_string()),
        })
    }
}

pub struct VideoOutputGenerator {
    frame_rate: usize,
    duration: f32,
    frame_size: (usize, usize),
    decoder: ImageDecoderMLP,
}

impl VideoOutputGenerator {
    pub fn new(embed_dim: usize) -> Result<Self> {
        let latent_dim = embed_dim.max(64);
        let hidden_dim = latent_dim * 4;
        let output_dim = 16 * 16 * 3;
        Ok(Self {
            frame_rate: 30,
            duration: 3.0,
            frame_size: (64, 64),
            decoder: ImageDecoderMLP::new(latent_dim, hidden_dim, output_dim),
        })
    }

    pub fn generate(
        &mut self,
        tokens: &[UnifiedToken],
        _inputs: &MultiModalInputs,
    ) -> Result<VideoOutput> {
        let video_tokens: Vec<_> = tokens
            .iter()
            .filter(|t| t.modality == ModalityType::Video)
            .collect();

        let num_frames = (self.duration * self.frame_rate as f32) as usize;
        let mut frames = Vec::new();

        for frame_idx in 0..num_frames {
            let frame_data = if !video_tokens.is_empty() {
                let token_idx = frame_idx % video_tokens.len();
                let token = &video_tokens[token_idx];
                let mut latent = token.embedding.clone();
                let temporal_factor = (frame_idx as f32 / num_frames as f32) * 0.1;
                for v in &mut latent {
                    *v += temporal_factor;
                }
                let pixels = self.decoder.forward(&latent);
                let width = self.frame_size.0;
                let height = self.frame_size.1;
                let mut frame_data = vec![0u8; width * height * 3];
                let patch_size = 16;
                let cols = width / patch_size;
                let rows = height / patch_size;
                for py in 0..rows.min(4) {
                    for px in 0..cols.min(4) {
                        let patch_idx = py * 4 + px;
                        for j in 0..patch_size {
                            for i in 0..patch_size {
                                let y = py * patch_size + j;
                                let x = px * patch_size + i;
                                if y < height && x < width {
                                    for c in 0..3 {
                                        let pi = (y * width + x) * 3 + c;
                                        let ppi = patch_idx * (patch_size * patch_size * 3) + j * patch_size * 3 + i * 3 + c;
                                        let val = if ppi < pixels.len() {
                                            (pixels[ppi] * 127.0 + 128.0).clamp(0.0, 255.0) as u8
                                        } else {
                                            ((x + y + frame_idx) % 256) as u8
                                        };
                                        if pi < frame_data.len() {
                                            frame_data[pi] = val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                frame_data
            } else {
                let width = self.frame_size.0;
                let height = self.frame_size.1;
                let mut frame_data = vec![0u8; width * height * 3];
                for (i, pixel) in frame_data.chunks_mut(3).enumerate() {
                    let x = i % width;
                    let y = i / width;
                    let cv = ((x * 7 + y * 3 + frame_idx * 5) % 256) as u8;
                    pixel[0] = cv;
                    pixel[1] = (cv.wrapping_mul(2)) % 255;
                    pixel[2] = (cv.wrapping_mul(3)) % 255;
                }
                frame_data
            };

            frames.push(ImageOutput {
                data: frame_data,
                format: ImageFormat::PNG,
                width: self.frame_size.0,
                height: self.frame_size.1,
                description: Some(format!("Frame {} from video tokens", frame_idx)),
            });
        }

        Ok(VideoOutput {
            frames,
            frame_rate: self.frame_rate,
            duration: self.duration,
            audio: None,
        })
    }
}
