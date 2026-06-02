use serde::{Deserialize, Serialize};

/// Konfigurasi kompresi konteks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Threshold token sebelum kompresi aktif
    pub threshold_tokens: usize,
    /// Target rasio kompresi (contoh: 0.05 = 100k → 5k)
    pub target_ratio: f32,
    /// Minimum token output setelah kompresi
    pub min_output_tokens: usize,
    /// Maximum token output setelah kompresi
    pub max_output_tokens: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            threshold_tokens: 2_000,
            target_ratio: 0.05,
            min_output_tokens: 256,
            max_output_tokens: 8_000,
        }
    }
}

/// Hasil kompresi
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionResult {
    pub original_tokens: usize,
    pub compressed_tokens: usize,
    pub ratio: f32,
    pub compressed_text: String,
    pub method: CompressionMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionMethod {
    Extractive,
    Truncate,
    Skip,
}

impl CompressionResult {
    pub fn savings_percent(&self) -> f32 {
        if self.original_tokens == 0 {
            return 0.0;
        }
        (1.0 - self.compressed_tokens as f32 / self.original_tokens as f32) * 100.0
    }
}

/// Context Compressor — memperkecil input sebelum dikirim ke Oracle
pub struct ContextCompressor {
    config: CompressionConfig,
    stats: CompressionStats,
}

impl ContextCompressor {
    pub fn new(config: CompressionConfig) -> Self {
        Self {
            config,
            stats: CompressionStats::default(),
        }
    }

    /// Perkiraan jumlah token dari teks (sederhana: split whitespace)
    fn estimate_tokens(text: &str) -> usize {
        text.split_whitespace().count()
    }

    /// Kompres teks berdasarkan konfigurasi
    pub fn compress(&mut self, text: &str) -> CompressionResult {
        let original_tokens = Self::estimate_tokens(text);

        if original_tokens <= self.config.threshold_tokens {
            self.stats.skipped += 1;
            return CompressionResult {
                original_tokens,
                compressed_tokens: original_tokens,
                ratio: 1.0,
                compressed_text: text.to_string(),
                method: CompressionMethod::Skip,
            };
        }

        let target_tokens = ((original_tokens as f32 * self.config.target_ratio) as usize)
            .max(self.config.min_output_tokens)
            .min(self.config.max_output_tokens)
            .min(original_tokens);

        let compressed = self.extractive_summarize(text, original_tokens, target_tokens);

        let compressed_tokens = Self::estimate_tokens(&compressed);
        let ratio = if original_tokens > 0 {
            compressed_tokens as f32 / original_tokens as f32
        } else {
            1.0
        };

        self.stats.compressed += 1;
        self.stats.total_savings += original_tokens - compressed_tokens;

        CompressionResult {
            original_tokens,
            compressed_tokens,
            ratio,
            compressed_text: compressed,
            method: CompressionMethod::Extractive,
        }
    }

    /// Ekstraktif summarization: ambil kalimat penting dari awal, tengah, akhir
    fn extractive_summarize(&self, text: &str, _original_tokens: usize, target_tokens: usize) -> String {
        let sentences: Vec<&str> = text
            .split(|c: char| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        if sentences.is_empty() {
            return text.chars().take(target_tokens * 5).collect();
        }

        let sentence_words: Vec<usize> = sentences.iter().map(|s| Self::estimate_tokens(s)).collect();
        let total_sentence_tokens: usize = sentence_words.iter().sum();
        if total_sentence_tokens == 0 {
            return String::new();
        }

        let mut selected = Vec::new();
        let mut budget = target_tokens;

        // Selalu ambil kalimat pertama (pembukaan)
        if !sentences.is_empty() && sentence_words[0] <= budget {
            selected.push(0);
            budget = budget.saturating_sub(sentence_words[0]);
        }

        // Ambil dari tengah (poin-poin penting)
        if sentences.len() > 3 && budget > 0 {
            let mid = sentences.len() / 2;
            for offset in 0..sentences.len().min(5) {
                let idx = mid + offset;
                if idx < sentences.len() && !selected.contains(&idx) && sentence_words[idx] <= budget {
                    selected.push(idx);
                    budget = budget.saturating_sub(sentence_words[idx]);
                }
                let idx2 = mid.saturating_sub(offset);
                if idx2 != mid && idx2 < sentences.len() && !selected.contains(&idx2) && sentence_words[idx2] <= budget {
                    selected.push(idx2);
                    budget = budget.saturating_sub(sentence_words[idx2]);
                }
            }
        }

        // Ambil dari akhir (kesimpulan)
        if sentences.len() > 1 && budget > 0 {
            let last = sentences.len() - 1;
            if !selected.contains(&last) && sentence_words[last] <= budget {
                selected.push(last);
            }
        }

        selected.sort();
        let result: String = selected
            .iter()
            .map(|&i| sentences[i])
            .collect::<Vec<&str>>()
            .join(". ");
        if result.is_empty() {
            sentences[0].to_string()
        } else {
            result + "."
        }
    }

    pub fn stats(&self) -> &CompressionStats {
        &self.stats
    }

    pub fn config(&self) -> &CompressionConfig {
        &self.config
    }

    pub fn reset_stats(&mut self) {
        self.stats = CompressionStats::default();
    }
}

impl Default for ContextCompressor {
    fn default() -> Self {
        Self::new(CompressionConfig::default())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    pub compressed: u64,
    pub skipped: u64,
    pub total_savings: usize,
}

impl Default for CompressionStats {
    fn default() -> Self {
        Self {
            compressed: 0,
            skipped: 0,
            total_savings: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_long_text(sentence_count: usize) -> String {
        let mut text = String::new();
        for i in 0..sentence_count {
            text.push_str(&format!("Kalimat nomor {} dalam dokumen panjang ini untuk testing kompresi konteks Oracle. ", i));
        }
        text
    }

    #[test]
    fn test_skip_short_text() {
        let mut compressor = ContextCompressor::default();
        let result = compressor.compress("Kalimat pendek.");
        assert!(matches!(result.method, CompressionMethod::Skip));
        assert_eq!(result.compressed_text, "Kalimat pendek.");
    }

    #[test]
    fn test_compress_long_text() {
        let mut compressor = ContextCompressor::new(CompressionConfig {
            threshold_tokens: 10,
            target_ratio: 0.3,
            min_output_tokens: 5,
            max_output_tokens: 100,
        });
        let text = make_long_text(50);
        let result = compressor.compress(&text);
        assert!(matches!(result.method, CompressionMethod::Extractive));
        assert!(result.compressed_tokens < result.original_tokens, "compressed >= original");
    }

    #[test]
    fn test_savings_percent() {
        let result = CompressionResult {
            original_tokens: 10000,
            compressed_tokens: 500,
            ratio: 0.05,
            compressed_text: "test".into(),
            method: CompressionMethod::Extractive,
        };
        let savings = result.savings_percent();
        assert!((savings - 95.0).abs() < 0.1, "expected 95% savings, got {}", savings);
    }

    #[test]
    fn test_stats_tracking() {
        let mut compressor = ContextCompressor::default();
        compressor.compress("pendek");
        compressor.compress(&make_long_text(400));
        let stats = compressor.stats();
        assert_eq!(stats.skipped, 1);
        assert_eq!(stats.compressed, 1);
    }

    #[test]
    fn test_compression_config_default() {
        let config = CompressionConfig::default();
        assert_eq!(config.threshold_tokens, 2_000);
        assert!((config.target_ratio - 0.05).abs() < 0.001);
    }
}
