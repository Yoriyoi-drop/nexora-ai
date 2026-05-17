//! String utilities untuk Nexora

use regex::Regex;
use anyhow::Result;
use std::collections::HashMap;

pub struct StringUtils;

impl StringUtils {
    /// Truncate string dengan ellipsis
    pub fn truncate(s: &str, max_len: usize) -> String {
        if s.len() <= max_len {
            s.to_string()
        } else {
            format!("{}...", &s[..max_len.saturating_sub(3)])
        }
    }
    
    /// Check if string is empty or only whitespace
    pub fn is_empty_or_whitespace(s: &str) -> bool {
        s.trim().is_empty()
    }
    
    /// Clean and normalize string
    pub fn clean(s: &str) -> String {
        s.trim()
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
    }
    
    /// Remove extra whitespace
    pub fn normalize_whitespace(s: &str) -> String {
        let re = Regex::new(r"\s+").expect("valid regex pattern");
        re.replace_all(s.trim(), " ").to_string()
    }
    
    /// Calculate similarity between two strings (Jaccard similarity)
    pub fn calculate_similarity(s1: &str, s2: &str) -> Result<f64> {
        let words1: std::collections::HashSet<&str> = s1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = s2.split_whitespace().collect();
        
        if words1.is_empty() && words2.is_empty() {
            return Ok(1.0);
        }
        
        if words1.is_empty() || words2.is_empty() {
            return Ok(0.0);
        }
        
        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();
        
        Ok(intersection as f64 / union as f64)
    }
    
    /// Calculate Levenshtein distance
    pub fn levenshtein_distance(s1: &str, s2: &str) -> usize {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        let len1 = chars1.len();
        let len2 = chars2.len();
        
        if len1 == 0 {
            return len2;
        }
        if len2 == 0 {
            return len1;
        }
        
        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];
        
        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }
        
        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
                matrix[i][j] = std::cmp::min(
                    std::cmp::min(
                        matrix[i - 1][j] + 1,      // deletion
                        matrix[i][j - 1] + 1       // insertion
                    ),
                    matrix[i - 1][j - 1] + cost  // substitution
                );
            }
        }
        
        matrix[len1][len2]
    }
    
    /// Calculate normalized Levenshtein similarity
    pub fn levenshtein_similarity(s1: &str, s2: &str) -> f64 {
        let distance = Self::levenshtein_distance(s1, s2);
        let max_len = s1.len().max(s2.len());
        
        if max_len == 0 {
            1.0
        } else {
            1.0 - (distance as f64 / max_len as f64)
        }
    }
    
    /// Extract keywords from string
    pub fn extract_keywords(text: &str, min_length: usize) -> Vec<String> {
        let re = Regex::new(r"\b[a-zA-Z]+\b").expect("valid regex pattern");
        re.find_iter(text)
            .map(|m| m.as_str().to_lowercase())
            .filter(|word| word.len() >= min_length)
            .collect()
    }
    
    /// Count word frequency
    pub fn word_frequency(text: &str) -> HashMap<String, usize> {
        let words = Self::extract_keywords(text, 1);
        let mut freq = HashMap::with_capacity(words.len());
        
        for word in words {
            *freq.entry(word).or_insert(0) += 1;
        }
        
        freq
    }
    
    /// Check if string contains pattern
    pub fn contains_pattern(text: &str, pattern: &str) -> Result<bool> {
        let re = Regex::new(pattern)?;
        Ok(re.is_match(text))
    }
    
    /// Extract all matches for pattern
    pub fn extract_matches(text: &str, pattern: &str) -> Result<Vec<String>> {
        let re = Regex::new(pattern)?;
        Ok(re.find_iter(text).map(|m| m.as_str().to_string()).collect())
    }
    
    /// Replace all occurrences of pattern
    pub fn replace_pattern(text: &str, pattern: &str, replacement: &str) -> Result<String> {
        let re = Regex::new(pattern)?;
        Ok(re.replace_all(text, replacement).to_string())
    }
    
    /// Split string into chunks of specified size
    pub fn chunk_string(s: &str, chunk_size: usize) -> Vec<String> {
        s.chars()
            .collect::<Vec<char>>()
            .chunks(chunk_size)
            .map(|chunk| chunk.iter().collect::<String>())
            .collect()
    }
    
    /// Count Unicode characters (not bytes)
    pub fn count_chars(s: &str) -> usize {
        s.chars().count()
    }
    
    /// Count words
    pub fn count_words(s: &str) -> usize {
        s.split_whitespace().count()
    }
    
    /// Count sentences (simple heuristic)
    pub fn count_sentences(s: &str) -> usize {
        let re = Regex::new(r"[.!?]+").expect("valid regex pattern");
        re.find_iter(s).count()
    }
    
    /// Generate slug from string
    pub fn slugify(s: &str) -> String {
        let cleaned = Self::clean(s);
        let re = Regex::new(r"\s+").expect("valid regex pattern");
        let slug = re.replace_all(&cleaned, "-");
        slug.trim_matches('-').to_string()
    }
    
    /// Generate random string
    pub fn random_string(length: usize) -> String {
        use uuid::Uuid;
        Uuid::new_v4().to_string().replace('-', "")[..length.min(32)].to_string()
    }
    
    /// Mask sensitive information
    pub fn mask_sensitive(s: &str, visible_chars: usize) -> String {
        if s.len() <= visible_chars {
            s.to_string()
        } else {
            let visible = &s[..visible_chars];
            let masked = "*".repeat(s.len() - visible_chars);
            format!("{}{}", visible, masked)
        }
    }
    
    /// Validate email format
    pub fn is_valid_email(email: &str) -> bool {
        let re = Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").expect("valid regex pattern");
        re.is_match(email)
    }
    
    /// Validate URL format
    pub fn is_valid_url(url: &str) -> bool {
        let re = Regex::new(r"^https?://[^\s/$.?#].[^\s]*$").expect("valid regex pattern");
        re.is_match(url)
    }
    
    /// Extract domain from URL
    pub fn extract_domain(url: &str) -> Option<String> {
        let re = Regex::new(r"https?://([^/]+)").expect("valid regex pattern");
        re.captures(url).map(|caps| caps[1].to_string())
    }
    
    /// Convert to title case
    pub fn to_title_case(s: &str) -> String {
        s.split_whitespace()
            .map(|word| {
                let mut chars: Vec<char> = word.chars().collect();
                if !chars.is_empty() {
                    if let Some(first_char) = chars.get_mut(0) {
                        *first_char = first_char.to_uppercase().next().unwrap_or(*first_char);
                    }
                    for c in chars.iter_mut().skip(1) {
                        *c = c.to_lowercase().next().unwrap_or(*c);
                    }
                }
                chars.into_iter().collect()
            })
            .collect::<Vec<String>>()
            .join(" ")
    }
    
    /// Pad string to specified length
    pub fn pad_string(s: &str, length: usize, pad_char: char) -> String {
        if s.len() >= length {
            s.to_string()
        } else {
            let padding = pad_char.to_string().repeat(length - s.len());
            format!("{}{}", s, padding)
        }
    }
    
    /// Check if string is palindrome
    pub fn is_palindrome(s: &str) -> bool {
        let cleaned: String = s.chars().filter(|c| c.is_alphanumeric()).collect();
        cleaned == cleaned.chars().rev().collect::<String>()
    }
    
    /// Reverse string
    pub fn reverse_string(s: &str) -> String {
        s.chars().rev().collect()
    }
    
    /// Capitalize first letter
    pub fn capitalize(s: &str) -> String {
        let mut chars: Vec<char> = s.chars().collect();
        if !chars.is_empty() {
            if let Some(first_char) = chars.get_mut(0) {
                *first_char = first_char.to_uppercase().next().unwrap_or(*first_char);
            }
            for c in chars.iter_mut().skip(1) {
                *c = c.to_lowercase().next().unwrap_or(*c);
            }
        }
        chars.into_iter().collect()
    }
    
    /// Check if string contains only ASCII characters
    pub fn is_ascii(s: &str) -> bool {
        s.is_ascii()
    }
    
    /// Convert to safe filename
    pub fn to_safe_filename(s: &str) -> String {
        let re = Regex::new(r"[^a-zA-Z0-9._-]").expect("valid regex pattern");
        let safe = re.replace_all(s, "_");
        safe.trim_matches('_').to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_string_utils() {
        // Test truncate
        assert_eq!(StringUtils::truncate("hello world", 5), "he...");
        assert_eq!(StringUtils::truncate("hi", 10), "hi");
        
        // Test clean
        assert_eq!(StringUtils::clean("  Hello, World!  "), "hello world");
        
        // Test similarity
        let sim = StringUtils::calculate_similarity("hello world", "hello there").expect("valid regex pattern");
        assert!(sim > 0.0 && sim < 1.0);
        
        // Test levenshtein
        let dist = StringUtils::levenshtein_distance("kitten", "sitting");
        assert_eq!(dist, 3);
        
        // Test slugify
        assert_eq!(StringUtils::slugify("Hello World Test"), "hello-world-test");
        
        // Test email validation
        assert!(StringUtils::is_valid_email("test@example.com"));
        assert!(!StringUtils::is_valid_email("invalid-email"));
        
        // Test palindrome
        assert!(StringUtils::is_palindrome("racecar"));
        assert!(!StringUtils::is_palindrome("hello"));
        
        // Test capitalize
        assert_eq!(StringUtils::capitalize("hello world"), "Hello world");
    }
}
