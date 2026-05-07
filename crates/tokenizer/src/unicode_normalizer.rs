//! Unicode Normalizer - Rust implementation
//! 
//! Unicode normalization and text preprocessing

use std::collections::HashMap;
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// Unicode normalization forms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NormalizationForm {
    NFC,  // Normalization Form C (canonical composition)
    NFD,  // Normalization Form D (canonical decomposition)
    NFKC, // Normalization Form KC (compatibility composition)
    NFKD, // Normalization Form KD (compatibility decomposition)
}

/// Normalization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationConfig {
    pub form: NormalizationForm,
    pub preserve_case: bool,
    pub preserve_whitespace: bool,
    pub preserve_indent: bool,
    pub normalize_line_endings: bool,
    pub remove_control_chars: bool,
    pub custom_replacements: HashMap<char, String>,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        let mut custom_replacements = HashMap::new();
        custom_replacements.insert('"', "\"".to_string());
        custom_replacements.insert('\'', "'".to_string());
        custom_replacements.insert('\n', " ".to_string());
        custom_replacements.insert('\t', " ".to_string());
        
        Self {
            form: NormalizationForm::NFC,
            preserve_case: true,
            preserve_whitespace: false,
            preserve_indent: false,
            normalize_line_endings: true,
            remove_control_chars: true,
            custom_replacements,
        }
    }
}

/// Unicode normalizer
pub struct UnicodeNormalizer {
    config: NormalizationConfig,
}

impl UnicodeNormalizer {
    /// Create a new normalizer with default config
    pub fn new() -> Self {
        Self::with_config(NormalizationConfig::default())
    }
    
    /// Create a new normalizer with custom config
    pub fn with_config(config: NormalizationConfig) -> Self {
        Self { config }
    }
    
    /// Normalize text according to configuration
    pub fn normalize(&self, text: &str) -> Result<String> {
        let mut result = text.to_string();
        
        // Apply Unicode normalization
        result = self.apply_unicode_normalization(&result)?;
        
        // Apply custom replacements
        result = self.apply_custom_replacements(&result);
        
        // Normalize whitespace
        if !self.config.preserve_whitespace {
            result = self.normalize_whitespace(&result);
        }
        
        // Normalize line endings
        if self.config.normalize_line_endings {
            result = self.normalize_line_endings(&result);
        }
        
        // Remove control characters
        if self.config.remove_control_chars {
            result = self.remove_control_characters(&result);
        }
        
        // Apply case transformation
        if !self.config.preserve_case {
            result = result.to_lowercase();
        }
        
        Ok(result)
    }
    
    /// Apply Unicode normalization (simplified implementation)
    fn apply_unicode_normalization(&self, text: &str) -> Result<String> {
        // Note: This is a simplified implementation
        // Full Unicode normalization would require a library like unicode-normalization
        // For now, we'll do basic character-level normalization
        
        match self.config.form {
            NormalizationForm::NFC => {
                // Simplified NFC: just return the text as-is for now
                // In a full implementation, this would compose characters
                Ok(text.to_string())
            }
            NormalizationForm::NFD => {
                // Simplified NFD: just return the text as-is for now
                // In a full implementation, this would decompose characters
                Ok(text.to_string())
            }
            NormalizationForm::NFKC => {
                // Simplified NFKC: apply compatibility composition
                self.apply_compatibility_composition(text)
            }
            NormalizationForm::NFKD => {
                // Simplified NFKD: apply compatibility decomposition
                self.apply_compatibility_decomposition(text)
            }
        }
    }
    
    /// Apply compatibility composition (simplified)
    fn apply_compatibility_composition(&self, text: &str) -> Result<String> {
        let mut result = String::new();
        
        for ch in text.chars() {
            let normalized = self.normalize_compatibility_char(ch);
            result.push_str(&normalized);
        }
        
        Ok(result)
    }
    
    /// Apply compatibility decomposition (simplified)
    fn apply_compatibility_decomposition(&self, text: &str) -> Result<String> {
        let mut result = String::new();
        
        for ch in text.chars() {
            let normalized = self.decompose_compatibility_char(ch);
            result.push_str(&normalized);
        }
        
        Ok(result)
    }
    
    /// Normalize a single character for compatibility
    fn normalize_compatibility_char(&self, ch: char) -> String {
        // Simplified compatibility normalization
        match ch {
            // Common compatibility mappings
            '"' => "\"".to_string(),
            '\'' => "'".to_string(),
            '`' => "'".to_string(),
            '–' => "-".to_string(),      // en dash
            '—' => "-".to_string(),      // em dash
            '"' => "\"".to_string(),      // left double quote
            '"' => "\"".to_string(),      // right double quote
            '\'' => "'".to_string(),       // left single quote
            '\'' => "'".to_string(),       // right single quote
            '…' => "...".to_string(),    // ellipsis
            '©' => "(c)".to_string(),    // copyright
            '®' => "(r)".to_string(),    // registered
            '™' => "(tm)".to_string(),   // trademark
            _ => ch.to_string(),
        }
    }
    
    /// Decompose a single character for compatibility
    fn decompose_compatibility_char(&self, ch: char) -> String {
        // Simplified compatibility decomposition
        match ch {
            'á' => "a\u{0301}".to_string(),  // a + acute accent
            'é' => "e\u{0301}".to_string(),  // e + acute accent
            'í' => "i\u{0301}".to_string(),  // i + acute accent
            'ó' => "o\u{0301}".to_string(),  // o + acute accent
            'ú' => "u\u{0301}".to_string(),  // u + acute accent
            'à' => "a\u{0300}".to_string(),  // a + grave accent
            'è' => "e\u{0300}".to_string(),  // e + grave accent
            'ì' => "i\u{0300}".to_string(),  // i + grave accent
            'ò' => "o\u{0300}".to_string(),  // o + grave accent
            'ù' => "u\u{0300}".to_string(),  // u + grave accent
            'â' => "a\u{0302}".to_string(),  // a + circumflex
            'ê' => "e\u{0302}".to_string(),  // e + circumflex
            'î' => "i\u{0302}".to_string(),  // i + circumflex
            'ô' => "o\u{0302}".to_string(),  // o + circumflex
            'û' => "u\u{0302}".to_string(),  // u + circumflex
            'ä' => "a\u{0308}".to_string(),  // a + diaeresis
            'ë' => "e\u{0308}".to_string(),  // e + diaeresis
            'ï' => "i\u{0308}".to_string(),  // i + diaeresis
            'ö' => "o\u{0308}".to_string(),  // o + diaeresis
            'ü' => "u\u{0308}".to_string(),  // u + diaeresis
            'ã' => "a\u{0303}".to_string(),  // a + tilde
            'ñ' => "n\u{0303}".to_string(),  // n + tilde
            'õ' => "o\u{0303}".to_string(),  // o + tilde
            'ç' => "c\u{0327}".to_string(),  // c + cedilla
            _ => ch.to_string(),
        }
    }
    
    /// Apply custom character replacements
    fn apply_custom_replacements(&self, text: &str) -> String {
        let mut result = String::new();
        
        for ch in text.chars() {
            if let Some(replacement) = self.config.custom_replacements.get(&ch) {
                result.push_str(replacement);
            } else {
                result.push(ch);
            }
        }
        
        result
    }
    
    /// Normalize whitespace
    fn normalize_whitespace(&self, text: &str) -> String {
        let mut result = String::new();
        let mut prev_was_space = false;
        let mut in_indent = true;
        
        for (_i, ch) in text.chars().enumerate() {
            if ch.is_whitespace() {
                if self.config.preserve_indent && in_indent {
                    // Preserve leading whitespace for indentation
                    result.push(ch);
                } else if !prev_was_space {
                    // Replace multiple spaces with single space
                    result.push(' ');
                    prev_was_space = true;
                }
                // Skip additional spaces
            } else {
                result.push(ch);
                prev_was_space = false;
                in_indent = false;
                
                // Reset indent state after first non-whitespace in line
                if ch == '\n' {
                    in_indent = true;
                }
            }
        }
        
        result
    }
    
    /// Normalize line endings
    fn normalize_line_endings(&self, text: &str) -> String {
        // Convert all line endings to \n
        text.replace("\r\n", "\n")
            .replace('\r', "\n")
    }
    
    /// Remove control characters (except common ones)
    fn remove_control_characters(&self, text: &str) -> String {
        let mut result = String::new();
        
        for ch in text.chars() {
            if ch.is_control() {
                // Keep common control characters
                match ch {
                    '\n' | '\r' | '\t' => result.push(ch),
                    _ => {} // Skip other control characters
                }
            } else {
                result.push(ch);
            }
        }
        
        result
    }
    
    /// Get configuration
    pub fn config(&self) -> &NormalizationConfig {
        &self.config
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: NormalizationConfig) {
        self.config = config;
    }
    
    /// Add custom replacement
    pub fn add_custom_replacement(&mut self, from: char, to: String) {
        self.config.custom_replacements.insert(from, to);
    }
    
    /// Remove custom replacement
    pub fn remove_custom_replacement(&mut self, from: char) -> Option<String> {
        self.config.custom_replacements.remove(&from)
    }
}

/// Convenience functions
pub fn normalize_text(text: &str) -> Result<String> {
    let normalizer = UnicodeNormalizer::new();
    normalizer.normalize(text)
}

pub fn normalize_text_with_config(text: &str, config: NormalizationConfig) -> Result<String> {
    let normalizer = UnicodeNormalizer::with_config(config);
    normalizer.normalize(text)
}

pub fn normalize_nfc(text: &str) -> Result<String> {
    let config = NormalizationConfig {
        form: NormalizationForm::NFC,
        ..Default::default()
    };
    normalize_text_with_config(text, config)
}

pub fn normalize_nfd(text: &str) -> Result<String> {
    let config = NormalizationConfig {
        form: NormalizationForm::NFD,
        ..Default::default()
    };
    normalize_text_with_config(text, config)
}

pub fn normalize_nfkc(text: &str) -> Result<String> {
    let config = NormalizationConfig {
        form: NormalizationForm::NFKC,
        ..Default::default()
    };
    normalize_text_with_config(text, config)
}

pub fn normalize_nfkd(text: &str) -> Result<String> {
    let config = NormalizationConfig {
        form: NormalizationForm::NFKD,
        ..Default::default()
    };
    normalize_text_with_config(text, config)
}

/// Statistics about normalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationStats {
    pub original_length: usize,
    pub normalized_length: usize,
    pub characters_normalized: usize,
    pub whitespace_normalized: bool,
    pub line_endings_normalized: bool,
    pub control_chars_removed: usize,
}

impl UnicodeNormalizer {
    /// Normalize text with statistics
    pub fn normalize_with_stats(&self, text: &str) -> Result<(String, NormalizationStats)> {
        let original_length = text.len();
        let normalized = self.normalize(text)?;
        let normalized_length = normalized.len();
        
        let stats = NormalizationStats {
            original_length,
            normalized_length,
            characters_normalized: if original_length != normalized_length { 1 } else { 0 },
            whitespace_normalized: !self.config.preserve_whitespace,
            line_endings_normalized: self.config.normalize_line_endings,
            control_chars_removed: if self.config.remove_control_chars { 
                text.chars().filter(|c| c.is_control() && !matches!(c, '\n' | '\r' | '\t')).count()
            } else { 0 },
        };
        
        Ok((normalized, stats))
    }
}

impl Default for UnicodeNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_normalization() {
        let normalizer = UnicodeNormalizer::new();
        let result = normalizer.normalize("Hello, World!").unwrap();
        assert_eq!(result, "Hello, World!");
    }
    
    #[test]
    fn test_whitespace_normalization() {
        let config = NormalizationConfig {
            preserve_whitespace: false,
            preserve_indent: false,
            ..Default::default()
        };
        let normalizer = UnicodeNormalizer::with_config(config);
        
        let result = normalizer.normalize("  Hello   World  ").unwrap();
        assert_eq!(result, "Hello World");
    }
    
    #[test]
    fn test_indent_preservation() {
        let config = NormalizationConfig {
            preserve_whitespace: false,
            preserve_indent: true,
            ..Default::default()
        };
        let normalizer = UnicodeNormalizer::with_config(config);
        
        let result = normalizer.normalize("    Hello\n        World").unwrap();
        assert_eq!(result, "    Hello\n        World");
    }
    
    #[test]
    fn test_case_normalization() {
        let config = NormalizationConfig {
            preserve_case: false,
            ..Default::default()
        };
        let normalizer = UnicodeNormalizer::with_config(config);
        
        let result = normalizer.normalize("HELLO WORLD").unwrap();
        assert_eq!(result, "hello world");
    }
    
    #[test]
    fn test_custom_replacements() {
        let mut config = NormalizationConfig::default();
        config.custom_replacements.insert('@', "[at]".to_string());
        config.custom_replacements.insert('.', "[dot]".to_string());
        
        let normalizer = UnicodeNormalizer::with_config(config);
        let result = normalizer.normalize("user@example.com").unwrap();
        assert_eq!(result, "user[at]example[dot]com");
    }
    
    #[test]
    fn test_line_ending_normalization() {
        let config = NormalizationConfig {
            normalize_line_endings: true,
            ..Default::default()
        };
        let normalizer = UnicodeNormalizer::with_config(config);
        
        let result = normalizer.normalize("Line1\r\nLine2\rLine3\n").unwrap();
        assert_eq!(result, "Line1\nLine2\nLine3\n");
    }
    
    #[test]
    fn test_control_character_removal() {
        let config = NormalizationConfig {
            remove_control_chars: true,
            ..Default::default()
        };
        let normalizer = UnicodeNormalizer::with_config(config);
        
        let result = normalizer.normalize("Hello\u{0001}World\u{0002}Test").unwrap();
        assert_eq!(result, "HelloWorldTest");
    }
    
    #[test]
    fn test_compatibility_normalization() {
        let config = NormalizationConfig {
            form: NormalizationForm::NFKC,
            ..Default::default()
        };
        let normalizer = UnicodeNormalizer::with_config(config);
        
        let result = normalizer.normalize("Hello—World").unwrap();
        assert_eq!(result, "Hello-World");
    }
    
    #[test]
    fn test_decomposition_normalization() {
        let config = NormalizationConfig {
            form: NormalizationForm::NFKD,
            ..Default::default()
        };
        let normalizer = UnicodeNormalizer::with_config(config);
        
        let result = normalizer.normalize("café").unwrap();
        // Should decompose é to e + ´
        assert!(result.contains("e"));
    }
    
    #[test]
    fn test_convenience_functions() {
        let result = normalize_text("Hello, World!").unwrap();
        assert_eq!(result, "Hello, World!");
        
        let result = normalize_nfc("Hello, World!").unwrap();
        assert_eq!(result, "Hello, World!");
    }
    
    #[test]
    fn test_normalization_stats() {
        let normalizer = UnicodeNormalizer::new();
        let (result, stats) = normalizer.normalize_with_stats("Hello, World!").unwrap();
        
        assert_eq!(result, "Hello, World!");
        assert_eq!(stats.original_length, 13);
        assert_eq!(stats.normalized_length, 13);
        assert_eq!(stats.characters_normalized, 0);
    }
    
    #[test]
    fn test_custom_replacement_management() {
        let mut normalizer = UnicodeNormalizer::new();
        
        normalizer.add_custom_replacement('x', "y".to_string());
        let result = normalizer.normalize("text").unwrap();
        assert_eq!(result, "teyt");
        
        let removed = normalizer.remove_custom_replacement('x');
        assert_eq!(removed, Some("y".to_string()));
        
        let result = normalizer.normalize("text").unwrap();
        assert_eq!(result, "text");
    }
}
