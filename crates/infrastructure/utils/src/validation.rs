//! Validation utilities untuk Nexora

use regex::Regex;
use anyhow::Result;
use std::collections::HashSet;

pub struct ValidationUtils;

impl ValidationUtils {
    /// Validate text input
    pub fn validate_text(text: &str) -> Result<()> {
        if text.trim().is_empty() {
            return Err(anyhow::anyhow!("Text cannot be empty"));
        }
        
        if text.len() > 10000 {
            return Err(anyhow::anyhow!("Text too long (max 10000 characters)"));
        }
        
        Ok(())
    }
    
    /// Validate email address
    pub fn validate_email(email: &str) -> Result<()> {
        if email.trim().is_empty() {
            return Err(anyhow::anyhow!("Email cannot be empty"));
        }
        
        let email_regex = Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")?;
        if !email_regex.is_match(email) {
            return Err(anyhow::anyhow!("Invalid email format"));
        }
        
        if email.len() > 254 {
            return Err(anyhow::anyhow!("Email too long (max 254 characters)"));
        }
        
        Ok(())
    }
    
    /// Validate URL
    pub fn validate_url(url: &str) -> Result<()> {
        if url.trim().is_empty() {
            return Err(anyhow::anyhow!("URL cannot be empty"));
        }
        
        let url_regex = Regex::new(r"^https?://[^\s/$.?#].[^\s]*$")?;
        if !url_regex.is_match(url) {
            return Err(anyhow::anyhow!("Invalid URL format"));
        }
        
        if url.len() > 2048 {
            return Err(anyhow::anyhow!("URL too long (max 2048 characters)"));
        }
        
        Ok(())
    }
    
    /// Validate phone number (international format)
    pub fn validate_phone(phone: &str) -> Result<()> {
        if phone.trim().is_empty() {
            return Err(anyhow::anyhow!("Phone number cannot be empty"));
        }
        
        let phone_regex = Regex::new(r"^\+?[1-9]\d{1,14}$")?;
        let clean_phone = phone.chars().filter(|c| c.is_ascii_digit() || *c == '+').collect::<String>();
        
        if !phone_regex.is_match(&clean_phone) {
            return Err(anyhow::anyhow!("Invalid phone number format"));
        }
        
        Ok(())
    }
    
    /// Validate username
    pub fn validate_username(username: &str) -> Result<()> {
        if username.trim().is_empty() {
            return Err(anyhow::anyhow!("Username cannot be empty"));
        }
        
        if username.len() < 3 {
            return Err(anyhow::anyhow!("Username too short (min 3 characters)"));
        }
        
        if username.len() > 50 {
            return Err(anyhow::anyhow!("Username too long (max 50 characters)"));
        }
        
        let username_regex = Regex::new(r"^[a-zA-Z0-9_-]+$")?;
        if !username_regex.is_match(username) {
            return Err(anyhow::anyhow!("Username can only contain letters, numbers, underscores, and hyphens"));
        }
        
        if username.starts_with('_') || username.starts_with('-') {
            return Err(anyhow::anyhow!("Username cannot start with underscore or hyphen"));
        }
        
        Ok(())
    }
    
    /// Validate password strength
    pub fn validate_password(password: &str) -> Result<()> {
        if password.len() < 8 {
            return Err(anyhow::anyhow!("Password too short (min 8 characters)"));
        }
        
        if password.len() > 128 {
            return Err(anyhow::anyhow!("Password too long (max 128 characters)"));
        }
        
        let has_uppercase = password.chars().any(|c| c.is_uppercase());
        let has_lowercase = password.chars().any(|c| c.is_lowercase());
        let has_digit = password.chars().any(|c| c.is_ascii_digit());
        let has_special = password.chars().any(|c| "!@#$%^&*()_+-=[]{}|;:,.<>?".contains(c));
        
        if !has_uppercase {
            return Err(anyhow::anyhow!("Password must contain at least one uppercase letter"));
        }
        
        if !has_lowercase {
            return Err(anyhow::anyhow!("Password must contain at least one lowercase letter"));
        }
        
        if !has_digit {
            return Err(anyhow::anyhow!("Password must contain at least one digit"));
        }
        
        if !has_special {
            return Err(anyhow::anyhow!("Password must contain at least one special character"));
        }
        
        Ok(())
    }
    
    /// Validate API key format
    pub fn validate_api_key(api_key: &str) -> Result<()> {
        if api_key.trim().is_empty() {
            return Err(anyhow::anyhow!("API key cannot be empty"));
        }
        
        if api_key.len() < 16 {
            return Err(anyhow::anyhow!("API key too short (min 16 characters)"));
        }
        
        if api_key.len() > 256 {
            return Err(anyhow::anyhow!("API key too long (max 256 characters)"));
        }
        
        let api_key_regex = Regex::new(r"^[a-zA-Z0-9_-]+$")?;
        if !api_key_regex.is_match(api_key) {
            return Err(anyhow::anyhow!("API key can only contain letters, numbers, underscores, and hyphens"));
        }
        
        Ok(())
    }
    
    /// Validate JSON string
    pub fn validate_json(json_str: &str) -> Result<()> {
        if json_str.trim().is_empty() {
            return Err(anyhow::anyhow!("JSON cannot be empty"));
        }
        
        serde_json::from_str::<serde_json::Value>(json_str)
            .map_err(|e| anyhow::anyhow!("Invalid JSON: {}", e))?;
        
        Ok(())
    }
    
    /// Validate date string (ISO 8601)
    pub fn validate_date(date_str: &str) -> Result<()> {
        if date_str.trim().is_empty() {
            return Err(anyhow::anyhow!("Date cannot be empty"));
        }
        
        chrono::DateTime::parse_from_rfc3339(date_str)
            .map_err(|e| anyhow::anyhow!("Invalid date format: {}", e))?;
        
        Ok(())
    }
    
    /// Validate UUID
    pub fn validate_uuid(uuid_str: &str) -> Result<()> {
        if uuid_str.trim().is_empty() {
            return Err(anyhow::anyhow!("UUID cannot be empty"));
        }
        
        uuid::Uuid::parse_str(uuid_str)
            .map_err(|e| anyhow::anyhow!("Invalid UUID format: {}", e))?;
        
        Ok(())
    }
    
    /// Validate IP address (IPv4 or IPv6)
    pub fn validate_ip_address(ip: &str) -> Result<()> {
        if ip.trim().is_empty() {
            return Err(anyhow::anyhow!("IP address cannot be empty"));
        }
        
        if ip.parse::<std::net::Ipv4Addr>().is_ok() || ip.parse::<std::net::Ipv6Addr>().is_ok() {
            Ok(())
        } else {
            Err(anyhow::anyhow!("Invalid IP address format"))
        }
    }
    
    /// Validate domain name
    pub fn validate_domain(domain: &str) -> Result<()> {
        if domain.trim().is_empty() {
            return Err(anyhow::anyhow!("Domain cannot be empty"));
        }
        
        if domain.len() > 253 {
            return Err(anyhow::anyhow!("Domain too long (max 253 characters)"));
        }
        
        let domain_regex = Regex::new(r"^[a-zA-Z0-9.-]+$")?;
        if !domain_regex.is_match(domain) {
            return Err(anyhow::anyhow!("Domain contains invalid characters"));
        }
        
        if domain.starts_with('.') || domain.starts_with('-') {
            return Err(anyhow::anyhow!("Domain cannot start with dot or hyphen"));
        }
        
        if domain.ends_with('.') || domain.ends_with('-') {
            return Err(anyhow::anyhow!("Domain cannot end with dot or hyphen"));
        }
        
        // Check if domain has valid TLD
        let parts: Vec<&str> = domain.split('.').collect();
        if parts.len() < 2 {
            return Err(anyhow::anyhow!("Domain must have at least two parts (e.g., example.com)"));
        }
        
        let tld = parts.last().expect("domain has at least two parts");
        if tld.len() < 2 {
            return Err(anyhow::anyhow!("Top-level domain too short"));
        }
        
        Ok(())
    }
    
    /// Validate file path
    pub fn validate_file_path(path: &str) -> Result<()> {
        if path.trim().is_empty() {
            return Err(anyhow::anyhow!("File path cannot be empty"));
        }
        
        if path.len() > 4096 {
            return Err(anyhow::anyhow!("File path too long (max 4096 characters)"));
        }
        
        // Check for invalid characters
        let invalid_chars = ['<', '>', ':', '"', '|', '?', '*'];
        for c in invalid_chars.iter() {
            if path.contains(*c) {
                return Err(anyhow::anyhow!("File path contains invalid character: {}", c));
            }
        }
        
        Ok(())
    }
    
    /// Validate number range
    pub fn validate_number_range(num: f64, min: f64, max: f64) -> Result<()> {
        if num < min || num > max {
            return Err(anyhow::anyhow!("Number {} is outside valid range [{}, {}]", num, min, max));
        }
        Ok(())
    }
    
    /// Validate integer range
    pub fn validate_int_range(num: i64, min: i64, max: i64) -> Result<()> {
        if num < min || num > max {
            return Err(anyhow::anyhow!("Integer {} is outside valid range [{}, {}]", num, min, max));
        }
        Ok(())
    }
    
    /// Validate string length
    pub fn validate_string_length(s: &str, min: usize, max: usize) -> Result<()> {
        let len = s.len();
        if len < min {
            return Err(anyhow::anyhow!("String too short (min {} characters)", min));
        }
        if len > max {
            return Err(anyhow::anyhow!("String too long (max {} characters)", max));
        }
        Ok(())
    }
    
    /// Validate that string contains only allowed characters
    pub fn validate_allowed_chars(s: &str, allowed: &str) -> Result<()> {
        let allowed_set: HashSet<char> = allowed.chars().collect();
        for c in s.chars() {
            if !allowed_set.contains(&c) {
                return Err(anyhow::anyhow!("String contains invalid character: {}", c));
            }
        }
        Ok(())
    }
    
    /// Validate that string doesn't contain forbidden characters
    pub fn validate_forbidden_chars(s: &str, forbidden: &str) -> Result<()> {
        let forbidden_set: HashSet<char> = forbidden.chars().collect();
        for c in s.chars() {
            if forbidden_set.contains(&c) {
                return Err(anyhow::anyhow!("String contains forbidden character: {}", c));
            }
        }
        Ok(())
    }
    
    /// Validate that string matches pattern
    pub fn validate_pattern(s: &str, pattern: &str) -> Result<()> {
        let regex = Regex::new(pattern)?;
        if !regex.is_match(s) {
            return Err(anyhow::anyhow!("String does not match required pattern"));
        }
        Ok(())
    }
    
    /// Validate that string is alphanumeric
    pub fn validate_alphanumeric(s: &str) -> Result<()> {
        if !s.chars().all(|c| c.is_alphanumeric()) {
            return Err(anyhow::anyhow!("String must be alphanumeric"));
        }
        Ok(())
    }
    
    /// Validate that string is numeric
    pub fn validate_numeric(s: &str) -> Result<()> {
        if !s.chars().all(|c| c.is_ascii_digit()) {
            return Err(anyhow::anyhow!("String must be numeric"));
        }
        Ok(())
    }
    
    /// Validate that string is alphabetic
    pub fn validate_alphabetic(s: &str) -> Result<()> {
        if !s.chars().all(|c| c.is_alphabetic()) {
            return Err(anyhow::anyhow!("String must be alphabetic"));
        }
        Ok(())
    }
    
    /// Validate that string is lowercase
    pub fn validate_lowercase(s: &str) -> Result<()> {
        if !s.chars().all(|c| c.is_lowercase()) {
            return Err(anyhow::anyhow!("String must be lowercase"));
        }
        Ok(())
    }
    
    /// Validate that string is uppercase
    pub fn validate_uppercase(s: &str) -> Result<()> {
        if !s.chars().all(|c| c.is_uppercase()) {
            return Err(anyhow::anyhow!("String must be uppercase"));
        }
        Ok(())
    }
    
    /// Validate that string contains no whitespace
    pub fn validate_no_whitespace(s: &str) -> Result<()> {
        if s.chars().any(|c| c.is_whitespace()) {
            return Err(anyhow::anyhow!("String cannot contain whitespace"));
        }
        Ok(())
    }
    
    /// Validate that string is not empty after trimming
    pub fn validate_not_empty(s: &str) -> Result<()> {
        if s.trim().is_empty() {
            return Err(anyhow::anyhow!("String cannot be empty"));
        }
        Ok(())
    }
    
    /// Validate that string is within allowed values
    pub fn validate_allowed_values<T: std::fmt::Display + PartialEq>(s: &str, allowed: &[T]) -> Result<()> {
        for value in allowed {
            if format!("{}", value) == s {
                return Ok(());
            }
        }
        Err(anyhow::anyhow!("Value '{}' is not in allowed list", s))
    }
    
    /// Validate that string is a valid color hex code
    pub fn validate_color_hex(color: &str) -> Result<()> {
        let color_regex = Regex::new(r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$")?;
        if !color_regex.is_match(color) {
            return Err(anyhow::anyhow!("Invalid color hex format"));
        }
        Ok(())
    }
    
    /// Validate that string is a valid credit card number (Luhn algorithm)
    pub fn validate_credit_card(card_number: &str) -> Result<()> {
        let clean_number = card_number.chars().filter(|c| c.is_ascii_digit()).collect::<String>();
        
        if clean_number.len() < 13 || clean_number.len() > 19 {
            return Err(anyhow::anyhow!("Invalid credit card number length"));
        }
        
        let mut sum = 0;
        let mut double = false;
        
        for c in clean_number.chars().rev() {
            let mut digit = c.to_digit(10).expect("valid digit character") as u32;
            
            if double {
                digit *= 2;
                if digit > 9 {
                    digit -= 9;
                }
            }
            
            sum += digit;
            double = !double;
        }
        
        if sum % 10 != 0 {
            return Err(anyhow::anyhow!("Invalid credit card number"));
        }
        
        Ok(())
    }
    
    /// Validate that string is a valid timezone
    pub fn validate_timezone(timezone: &str) -> Result<()> {
        // This is a simplified validation - in production, you'd want to use a proper timezone library
        let timezone_regex = Regex::new(r"^[A-Za-z_]+/[A-Za-z_]+$")?;
        if !timezone_regex.is_match(timezone) {
            return Err(anyhow::anyhow!("Invalid timezone format"));
        }
        Ok(())
    }
    
    /// Validate that string is a valid language code (ISO 639-1)
    pub fn validate_language_code(lang_code: &str) -> Result<()> {
        let lang_regex = Regex::new(r"^[a-z]{2}(-[A-Z]{2})?$")?;
        if !lang_regex.is_match(lang_code) {
            return Err(anyhow::anyhow!("Invalid language code format"));
        }
        Ok(())
    }
    
    /// Validate that string is a valid currency code (ISO 4217)
    pub fn validate_currency_code(currency_code: &str) -> Result<()> {
        let currency_regex = Regex::new(r"^[A-Z]{3}$")?;
        if !currency_regex.is_match(currency_code) {
            return Err(anyhow::anyhow!("Invalid currency code format"));
        }
        Ok(())
    }
    
    /// Validate that string is a valid country code (ISO 3166-1 alpha-2)
    pub fn validate_country_code(country_code: &str) -> Result<()> {
        let country_regex = Regex::new(r"^[A-Z]{2}$")?;
        if !country_regex.is_match(country_code) {
            return Err(anyhow::anyhow!("Invalid country code format"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_validation_utils() {
        // Test text validation
        assert!(ValidationUtils::validate_text("Hello world").is_ok());
        assert!(ValidationUtils::validate_text("").is_err());
        
        // Test email validation
        assert!(ValidationUtils::validate_email("test@example.com").is_ok());
        assert!(ValidationUtils::validate_email("invalid-email").is_err());
        
        // Test URL validation
        assert!(ValidationUtils::validate_url("https://example.com").is_ok());
        assert!(ValidationUtils::validate_url("invalid-url").is_err());
        
        // Test username validation
        assert!(ValidationUtils::validate_username("user123").is_ok());
        assert!(ValidationUtils::validate_username("_invalid").is_err());
        assert!(ValidationUtils::validate_username("us").is_err()); // too short
        
        // Test password validation
        assert!(ValidationUtils::validate_password("Password123!").is_ok());
        assert!(ValidationUtils::validate_password("weak").is_err());
        
        // Test JSON validation
        assert!(ValidationUtils::validate_json(r#"{"key": "value"}"#).is_ok());
        assert!(ValidationUtils::validate_json("invalid json").is_err());
        
        // Test UUID validation
        assert!(ValidationUtils::validate_uuid("550e8400-e29b-41d4-a716-446655440000").is_ok());
        assert!(ValidationUtils::validate_uuid("invalid-uuid").is_err());
        
        // Test IP address validation
        assert!(ValidationUtils::validate_ip_address("192.168.1.1").is_ok());
        assert!(ValidationUtils::validate_ip_address("invalid-ip").is_err());
        
        // Test domain validation
        assert!(ValidationUtils::validate_domain("example.com").is_ok());
        assert!(ValidationUtils::validate_domain("invalid..domain").is_err());
        
        // Test number range validation
        assert!(ValidationUtils::validate_number_range(5.0, 1.0, 10.0).is_ok());
        assert!(ValidationUtils::validate_number_range(15.0, 1.0, 10.0).is_err());
        
        // Test string length validation
        assert!(ValidationUtils::validate_string_length("hello", 3, 10).is_ok());
        assert!(ValidationUtils::validate_string_length("hi", 3, 10).is_err());
        
        // Test pattern validation
        assert!(ValidationUtils::validate_pattern("abc123", r"^[a-z]{3}[0-9]{3}$").is_ok());
        assert!(ValidationUtils::validate_pattern("invalid", r"^[a-z]{3}[0-9]{3}$").is_err());
        
        // Test alphanumeric validation
        assert!(ValidationUtils::validate_alphanumeric("abc123").is_ok());
        assert!(ValidationUtils::validate_alphanumeric("abc-123").is_err());
        
        // Test credit card validation
        assert!(ValidationUtils::validate_credit_card("4539 1488 0343 6467").is_ok());
        assert!(ValidationUtils::validate_credit_card("1234567890123456").is_err());
        
        // Test color hex validation
        assert!(ValidationUtils::validate_color_hex("#FF0000").is_ok());
        assert!(ValidationUtils::validate_color_hex("FF0000").is_err());
        
        // Test language code validation
        assert!(ValidationUtils::validate_language_code("en").is_ok());
        assert!(ValidationUtils::validate_language_code("en-US").is_ok());
        assert!(ValidationUtils::validate_language_code("invalid").is_err());
        
        // Test currency code validation
        assert!(ValidationUtils::validate_currency_code("USD").is_ok());
        assert!(ValidationUtils::validate_currency_code("usd").is_err());
        
        // Test country code validation
        assert!(ValidationUtils::validate_country_code("US").is_ok());
        assert!(ValidationUtils::validate_country_code("USA").is_err());
    }
}
