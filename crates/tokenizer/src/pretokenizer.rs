//! Pre-tokenizer - Rust implementation
//! 
//! Pre-tokenization logic untuk breaking text into manageable pieces

use std::collections::HashMap;
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// Pre-tokenized text piece
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreTokenizedPiece {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub piece_type: PieceType,
}

/// Types of pre-tokenized pieces
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PieceType {
    Word,           // Regular word
    Punctuation,    // Punctuation marks
    Whitespace,     // Space, tab, newline
    Number,         // Numeric values
    String,         // Quoted strings
    Comment,        // Code comments
    Operator,       // Multi-character operators
    Unknown,        // Unrecognized
}

/// Pre-tokenized result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreTokenized {
    pub pieces: Vec<PreTokenizedPiece>,
    pub original: String,
    pub total_length: usize,
}

impl PreTokenized {
    pub fn new(original: String) -> Self {
        let total_length = original.len();
        Self {
            pieces: Vec::new(),
            original,
            total_length,
        }
    }
    
    pub fn add_piece(&mut self, text: String, start: usize, end: usize, piece_type: PieceType) {
        self.pieces.push(PreTokenizedPiece {
            text,
            start,
            end,
            piece_type,
        });
    }
    
    pub fn get_piece_lengths(&self) -> Vec<usize> {
        self.pieces.iter().map(|p| p.text.len()).collect()
    }
    
    pub fn get_piece_texts(&self) -> Vec<&str> {
        self.pieces.iter().map(|p| p.text.as_str()).collect()
    }
}

/// Pre-tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreTokenizerConfig {
    pub preserve_whitespace: bool,
    pub preserve_case: bool,
    pub split_numbers: bool,
    pub split_strings: bool,
    pub split_comments: bool,
    pub merge_operators: bool,
    pub custom_patterns: HashMap<String, PieceType>,
}

impl Default for PreTokenizerConfig {
    fn default() -> Self {
        let mut custom_patterns = HashMap::new();
        custom_patterns.insert("->".to_string(), PieceType::Operator);
        custom_patterns.insert("::".to_string(), PieceType::Operator);
        custom_patterns.insert("**".to_string(), PieceType::Operator);
        
        Self {
            preserve_whitespace: false,
            preserve_case: true,
            split_numbers: true,
            split_strings: true,
            split_comments: true,
            merge_operators: true,
            custom_patterns,
        }
    }
}

/// Pre-tokenizer for breaking text into pieces
pub struct PreTokenizer {
    config: PreTokenizerConfig,
    multi_char_operators: Vec<String>,
}

impl PreTokenizer {
    pub fn new(config: PreTokenizerConfig) -> Self {
        let mut multi_char_operators = vec![
            "->".to_string(), "::".to_string(), "**".to_string(), "//".to_string(), "/*".to_string(), "*/".to_string(), "<=".to_string(), ">=".to_string(), "!=".to_string(), "==".to_string(),
            "&&".to_string(), "||".to_string(), "<<".to_string(), ">>".to_string(), "+=".to_string(), "-=".to_string(), "*=".to_string(), "/=".to_string(), "%=".to_string(),
            "&=".to_string(), "|=".to_string(), "^=".to_string(), "++".to_string(), "--".to_string(), "...".to_string(), "=>".to_string(), ":::".to_string(), "??".to_string(),
        ];
        
        // Add custom patterns
        for pattern in config.custom_patterns.keys() {
            if pattern.len() > 1 && !multi_char_operators.contains(&pattern) {
                multi_char_operators.push(pattern.clone());
            }
        }
        
        // Sort by length (longer patterns first)
        multi_char_operators.sort_by(|a, b| b.len().cmp(&a.len()));
        
        Self {
            config,
            multi_char_operators,
        }
    }
    
    pub fn with_default_config() -> Self {
        Self::new(PreTokenizerConfig::default())
    }
    
    /// Pre-tokenize input text
    pub fn pretokenize(&self, text: &str) -> Result<PreTokenized> {
        let mut result = PreTokenized::new(text.to_string());
        let mut i = 0;
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        
        while i < n {
            let (piece, next_i) = self.extract_next_piece(&chars, i, n)?;
            
            if !piece.text.is_empty() {
                result.add_piece(piece.text, piece.start, piece.end, piece.piece_type);
            }
            
            i = next_i;
        }
        
        Ok(result)
    }
    
    /// Extract the next piece from the character array
    fn extract_next_piece(&self, chars: &[char], start: usize, n: usize) -> Result<(PreTokenizedPiece, usize)> {
        let i = start;
        
        if i >= n {
            return Ok((PreTokenizedPiece {
                text: String::new(),
                start: i,
                end: i,
                piece_type: PieceType::Unknown,
            }, i));
        }
        
        let c = chars[i];
        
        // Handle whitespace
        if c.is_whitespace() {
            let mut end = i + 1;
            while end < n && chars[end].is_whitespace() {
                end += 1;
            }
            
            let text: String = chars[i..end].iter().collect();
            return Ok((PreTokenizedPiece {
                text,
                start: i,
                end,
                piece_type: PieceType::Whitespace,
            }, end));
        }
        
        // Handle strings
        if self.config.split_strings && (c == '"' || c == '\'') {
            return self.extract_string(chars, i, n);
        }
        
        // Handle comments
        if self.config.split_comments && c == '/' {
            return self.extract_comment(chars, i, n);
        }
        
        // Handle multi-character operators
        if self.config.merge_operators && self.is_operator_char(c) {
            return self.extract_operator(chars, i, n);
        }
        
        // Handle numbers
        if self.config.split_numbers && self.is_number_char(c) {
            return self.extract_number(chars, i, n);
        }
        
        // Handle punctuation
        if self.is_punctuation(c) {
            return Ok((PreTokenizedPiece {
                text: c.to_string(),
                start: i,
                end: i + 1,
                piece_type: PieceType::Punctuation,
            }, i + 1));
        }
        
        // Handle words
        if self.is_word_char(c) {
            return self.extract_word(chars, i, n);
        }
        
        // Unknown character
        Ok((PreTokenizedPiece {
            text: c.to_string(),
            start: i,
            end: i + 1,
            piece_type: PieceType::Unknown,
        }, i + 1))
    }
    
    /// Extract string literal
    fn extract_string(&self, chars: &[char], start: usize, n: usize) -> Result<(PreTokenizedPiece, usize)> {
        let quote_char = chars[start];
        let mut i = start + 1;
        let mut escaped = false;
        
        while i < n {
            let c = chars[i];
            
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == quote_char {
                i += 1; // Include closing quote
                break;
            }
            
            i += 1;
        }
        
        if i > n {
            // Unterminated string
            i = n;
        }
        
        let text: String = chars[start..i].iter().collect();
        Ok((PreTokenizedPiece {
            text,
            start,
            end: i,
            piece_type: PieceType::String,
        }, i))
    }
    
    /// Extract comment
    fn extract_comment(&self, chars: &[char], start: usize, n: usize) -> Result<(PreTokenizedPiece, usize)> {
        if start + 1 >= n {
            return Ok((PreTokenizedPiece {
                text: chars[start].to_string(),
                start,
                end: start + 1,
                piece_type: PieceType::Unknown,
            }, start + 1));
        }
        
        let next_char = chars[start + 1];
        
        if next_char == '/' {
            // Single line comment
            let mut i = start + 2;
            while i < n && chars[i] != '\n' {
                i += 1;
            }
            
            let text: String = chars[start..i].iter().collect();
            Ok((PreTokenizedPiece {
                text,
                start,
                end: i,
                piece_type: PieceType::Comment,
            }, i))
        } else if next_char == '*' {
            // Multi-line comment
            let mut i = start + 2;
            while i + 1 < n {
                if chars[i] == '*' && chars[i + 1] == '/' {
                    i += 2;
                    break;
                }
                i += 1;
            }
            
            let text: String = chars[start..i].iter().collect();
            Ok((PreTokenizedPiece {
                text,
                start,
                end: i,
                piece_type: PieceType::Comment,
            }, i))
        } else {
            // Just a slash
            Ok((PreTokenizedPiece {
                text: chars[start].to_string(),
                start,
                end: start + 1,
                piece_type: PieceType::Operator,
            }, start + 1))
        }
    }
    
    /// Extract multi-character operator
    fn extract_operator(&self, chars: &[char], start: usize, n: usize) -> Result<(PreTokenizedPiece, usize)> {
        // Try to match the longest possible operator
        for op in &self.multi_char_operators {
            let op_chars: Vec<char> = op.chars().collect();
            if start + op_chars.len() <= n {
                let mut matches = true;
                for (j, &op_char) in op_chars.iter().enumerate() {
                    if chars[start + j] != op_char {
                        matches = false;
                        break;
                    }
                }
                
                if matches {
                    return Ok((PreTokenizedPiece {
                        text: op.clone(),
                        start,
                        end: start + op_chars.len(),
                        piece_type: PieceType::Operator,
                    }, start + op_chars.len()));
                }
            }
        }
        
        // Single character operator
        Ok((PreTokenizedPiece {
            text: chars[start].to_string(),
            start,
            end: start + 1,
            piece_type: PieceType::Operator,
        }, start + 1))
    }
    
    /// Extract number
    fn extract_number(&self, chars: &[char], start: usize, n: usize) -> Result<(PreTokenizedPiece, usize)> {
        let mut i = start;
        
        // Handle sign
        if i < n && (chars[i] == '+' || chars[i] == '-') {
            i += 1;
        }
        
        // Handle digits before decimal point
        while i < n && chars[i].is_ascii_digit() {
            i += 1;
        }
        
        // Handle decimal point and digits after
        if i < n && chars[i] == '.' {
            i += 1;
            while i < n && chars[i].is_ascii_digit() {
                i += 1;
            }
        }
        
        // Handle exponent
        if i < n && (chars[i] == 'e' || chars[i] == 'E') {
            i += 1;
            if i < n && (chars[i] == '+' || chars[i] == '-') {
                i += 1;
            }
            while i < n && chars[i].is_ascii_digit() {
                i += 1;
            }
        }
        
        let text: String = chars[start..i].iter().collect();
        Ok((PreTokenizedPiece {
            text,
            start,
            end: i,
            piece_type: PieceType::Number,
        }, i))
    }
    
    /// Extract word
    fn extract_word(&self, chars: &[char], start: usize, n: usize) -> Result<(PreTokenizedPiece, usize)> {
        let mut i = start;
        
        while i < n && self.is_word_char(chars[i]) {
            i += 1;
        }
        
        let mut text: String = chars[start..i].iter().collect();
        
        // Convert to lowercase if not preserving case
        if !self.config.preserve_case {
            text = text.to_lowercase();
        }
        
        Ok((PreTokenizedPiece {
            text,
            start,
            end: i,
            piece_type: PieceType::Word,
        }, i))
    }
    
    /// Check if character is punctuation
    fn is_punctuation(&self, c: char) -> bool {
        matches!(c, 
            '(' | ')' | '{' | '}' | '[' | ']' | ';' | ',' |
            ':' | '.' | '!' | '?' | '=' | '+' | '-' | '*' |
            '/' | '%' | '&' | '|' | '^' | '~' | '<' | '>'
        )
    }
    
    /// Check if character can be part of operator
    fn is_operator_char(&self, c: char) -> bool {
        matches!(c,
            '=' | '!' | '<' | '>' | '+' | '-' | '*' | '/' |
            '&' | '|' | '^' | ':' | '.' | '~' | '%' | ';'
        )
    }
    
    /// Check if character can be part of number
    fn is_number_char(&self, c: char) -> bool {
        c.is_ascii_digit() || c == '.' || c == '+' || c == '-' || c == 'e' || c == 'E'
    }
    
    /// Check if character can be part of word
    fn is_word_char(&self, c: char) -> bool {
        c.is_alphabetic() || c == '_' || (c.is_ascii_digit() && !self.config.split_numbers)
    }
    
    /// Get configuration
    pub fn config(&self) -> &PreTokenizerConfig {
        &self.config
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: PreTokenizerConfig) {
        self.config = config.clone();
        
        // Rebuild multi-character operators list
        self.multi_char_operators = vec![
            "->".to_string(), "::".to_string(), "**".to_string(), "//".to_string(), "/*".to_string(), "*/".to_string(), "<=".to_string(), ">=".to_string(), "!=".to_string(), "==".to_string(),
            "&&".to_string(), "||".to_string(), "<<".to_string(), ">>".to_string(), "+=".to_string(), "-=".to_string(), "*=".to_string(), "/=".to_string(), "%=".to_string(),
            "&=".to_string(), "|=".to_string(), "^=".to_string(), "++".to_string(), "--".to_string(), "...".to_string(), "=>".to_string(), ":::".to_string(), "??".to_string(),
        ];
        
        for pattern in self.config.custom_patterns.keys() {
            if pattern.len() > 1 && !self.multi_char_operators.contains(&pattern) {
                self.multi_char_operators.push(pattern.clone());
            }
        }
        
        self.multi_char_operators.sort_by(|a, b| b.len().cmp(&a.len()));
    }
}

/// Convenience function for simple pre-tokenization
pub fn pretokenize(text: &str) -> Result<PreTokenized> {
    let tokenizer = PreTokenizer::with_default_config();
    tokenizer.pretokenize(text)
}

/// Convenience function for pre-tokenization with custom config
pub fn pretokenize_with_config(text: &str, config: PreTokenizerConfig) -> Result<PreTokenized> {
    let tokenizer = PreTokenizer::new(config);
    tokenizer.pretokenize(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_pretokenization() {
        let result = pretokenize("Hello, world!").unwrap();
        assert_eq!(result.pieces.len(), 4); // "Hello", ",", " ", "world", "!"
        
        let piece_types: Vec<PieceType> = result.pieces.iter().map(|p| p.piece_type).collect();
        assert_eq!(piece_types, vec![
            PieceType::Word,
            PieceType::Punctuation,
            PieceType::Whitespace,
            PieceType::Word,
            PieceType::Punctuation,
        ]);
    }
    
    #[test]
    fn test_number_extraction() {
        let result = pretokenize("The value is 42.5 and -10").unwrap();
        
        let numbers: Vec<&str> = result.pieces
            .iter()
            .filter(|p| p.piece_type == PieceType::Number)
            .map(|p| p.text.as_str())
            .collect();
        
        assert_eq!(numbers, vec!["42.5", "-10"]);
    }
    
    #[test]
    fn test_string_extraction() {
        let result = pretokenize(r#"print("Hello, \"world\"!")"#).unwrap();
        
        let strings: Vec<&str> = result.pieces
            .iter()
            .filter(|p| p.piece_type == PieceType::String)
            .map(|p| p.text.as_str())
            .collect();
        
        assert_eq!(strings, vec![r#""Hello, \"world\"""#]);
    }
    
    #[test]
    fn test_operator_extraction() {
        let result = pretokenize("a <= b && c >= d").unwrap();
        
        let operators: Vec<&str> = result.pieces
            .iter()
            .filter(|p| p.piece_type == PieceType::Operator)
            .map(|p| p.text.as_str())
            .collect();
        
        assert_eq!(operators, vec!["<=", "&&", ">="]);
    }
    
    #[test]
    fn test_comment_extraction() {
        let result = pretokenize("x = 5 // This is a comment\ny = 10").unwrap();
        
        let comments: Vec<&str> = result.pieces
            .iter()
            .filter(|p| p.piece_type == PieceType::Comment)
            .map(|p| p.text.as_str())
            .collect();
        
        assert_eq!(comments, vec!["// This is a comment"]);
    }
    
    #[test]
    fn test_case_preservation() {
        let config = PreTokenizerConfig {
            preserve_case: true,
            ..Default::default()
        };
        
        let result = pretokenize_with_config("Hello WORLD", config).unwrap();
        
        let words: Vec<&str> = result.pieces
            .iter()
            .filter(|p| p.piece_type == PieceType::Word)
            .map(|p| p.text.as_str())
            .collect();
        
        assert_eq!(words, vec!["Hello", "WORLD"]);
    }
    
    #[test]
    fn test_custom_patterns() {
        let mut custom_patterns = HashMap::new();
        custom_patterns.insert(":::".to_string(), PieceType::Operator);
        
        let config = PreTokenizerConfig {
            custom_patterns,
            ..Default::default()
        };
        
        let result = pretokenize_with_config("Module:::Function", config).unwrap();
        
        let operators: Vec<&str> = result.pieces
            .iter()
            .filter(|p| p.piece_type == PieceType::Operator)
            .map(|p| p.text.as_str())
            .collect();
        
        assert_eq!(operators, vec![":::"]);
    }
}
