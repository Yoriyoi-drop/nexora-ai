//! Security module for Nexora AI system

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;
use tracing::warn;

static RE_HTML_TAG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<[^>]*>").expect("valid HTML tag regex"));

static RE_DANGEROUS_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    let patterns: &[&str] = &[
        r"-----BEGIN.*PRIVATE KEY-----",
        r"-----BEGIN CERTIFICATE-----",
        r"ghp_",
        r"gho_",
        r"ghu_",
        r"ghs_",
        r"ghr_",
        r"AKIA",
        r"xox[baprs]-",
    ];
    patterns.iter().filter_map(|p| Regex::new(p).ok()).collect()
});

static MALICIOUS_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    let patterns: &[(&str, &str)] = &[
        (r"<script[^>]*>.*?</script>", "script tag"),
        (r"javascript:", "javascript: protocol"),
        (r"eval\s*\(", "eval() call"),
        (r"exec\s*\(", "exec() call"),
        (r"system\s*\(", "system() call"),
        (r"__import__", "__import__ statement"),
        (r"subprocess\.", "subprocess module"),
        (r"os\.", "os module access"),
        (r"require\s*\(", "require() call"),
        (r"import\s+.*from", "import-from statement"),
    ];
    patterns
        .iter()
        .filter_map(|(pattern, name)| {
            match Regex::new(pattern) {
                Ok(re) => Some(re),
                Err(e) => {
                    tracing::error!("Failed to compile security regex '{}': {}. This rule will be SILENTLY DISABLED!", name, e);
                    // DO NOT silently pass a no-op regex — security rules should fail loud
                    None
                }
            }
        })
        .collect()
});

static PATH_TRAVERSAL_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    let patterns: &[(&str, &str)] = &[
        (r"\.\.[/\\]", "parent directory traversal"),
        (r"[/\\]\.\.[/\\]", "path-embedded parent dir"),
        (r"%2e%2f", "URL-encoded ../ (%2e%2f)"),
        (r"%2e%5c", "URL-encoded ..\\ (%2e%5c)"),
    ];
    patterns
        .iter()
        .filter_map(|(pattern, name)| match Regex::new(pattern) {
            Ok(re) => Some(re),
            Err(e) => {
                tracing::error!("Failed to compile path traversal regex '{}': {}. This rule will be SILENTLY DISABLED!", name, e);
                None
            }
        })
        .collect()
});

static COMMAND_INJECTION_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    let patterns: &[(&str, &str)] = &[
        (r"[;&|`$()]", "shell metacharacters"),
        (r"(rm|del|format|shutdown|reboot)", "destructive commands"),
        (r"(sudo|su|doas)", "privilege escalation commands"),
        (r"(curl|wget|nc|netcat)", "network tool commands"),
        (r"(chmod|chown|chgrp)", "file permission commands"),
    ];
    patterns
        .iter()
        .filter_map(|(pattern, name)| match Regex::new(pattern) {
            Ok(re) => Some(re),
            Err(e) => {
                tracing::error!("Failed to compile command injection regex '{}': {}. This rule will be SILENTLY DISABLED!", name, e);
                None
            }
        })
        .collect()
});

/// Security configuration and validation
pub struct SecurityConfig {
    /// Maximum allowed input length for text generation
    pub max_input_length: usize,
    /// Maximum allowed output length for text generation
    pub max_output_length: usize,
    /// Blocked patterns in input (for security)
    pub blocked_patterns: Vec<String>,
    /// Allowed file extensions for file operations
    pub allowed_extensions: HashSet<String>,
    /// Rate limiting configuration
    pub rate_limit: RateLimitConfig,
}

/// Rate limiting configuration
pub struct RateLimitConfig {
    /// Maximum requests per minute
    pub requests_per_minute: u32,
    /// Maximum requests per hour
    pub requests_per_hour: u32,
    /// Maximum requests per day
    pub requests_per_day: u32,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        let mut allowed_extensions = HashSet::new();
        allowed_extensions.insert("rs".to_string());
        allowed_extensions.insert("py".to_string());
        allowed_extensions.insert("js".to_string());
        allowed_extensions.insert("ts".to_string());
        allowed_extensions.insert("java".to_string());
        allowed_extensions.insert("cpp".to_string());
        allowed_extensions.insert("c".to_string());
        allowed_extensions.insert("h".to_string());
        allowed_extensions.insert("txt".to_string());
        allowed_extensions.insert("md".to_string());
        allowed_extensions.insert("json".to_string());
        allowed_extensions.insert("toml".to_string());
        allowed_extensions.insert("yaml".to_string());
        allowed_extensions.insert("yml".to_string());

        Self {
            max_input_length: 10000,
            max_output_length: 5000,
            blocked_patterns: vec![
                r"<script[^>]*>.*?</script>".to_string(), // XSS
                r"javascript:".to_string(),               // JS injection
                r"eval\s*\(".to_string(),                 // Code execution
                r"exec\s*\(".to_string(),                 // Shell execution
                r"system\s*\(".to_string(),               // System calls
                r"__import__".to_string(),                // Python import
                r"subprocess\.".to_string(),              // Python subprocess
                r"os\.".to_string(),                      // Python OS module
                r"require\s*\(".to_string(),              // Node require
                r"import\s+.*from".to_string(),           // Python import
            ],
            allowed_extensions,
            rate_limit: RateLimitConfig {
                requests_per_minute: 60,
                requests_per_hour: 1000,
                requests_per_day: 10000,
            },
        }
    }
}

/// Security validator for input and operations
pub struct SecurityValidator {
    config: SecurityConfig,
}

impl SecurityValidator {
    pub fn new(config: SecurityConfig) -> Self {
        Self { config }
    }

    /// Validate input for security issues
    pub fn validate_input(&self, input: &str) -> Result<(), SecurityError> {
        // Check input length
        if input.len() > self.config.max_input_length {
            return Err(SecurityError::InputTooLong(
                input.len(),
                self.config.max_input_length,
            ));
        }

        // Check for malicious patterns (compile once, cache in config)
    for pattern in &self.config.blocked_patterns {
        // Try regex match first; fall back to substring match
        if let Ok(re) = regex::Regex::new(pattern) {
            if re.is_match(input) {
                return Err(SecurityError::BlockedPattern(pattern.clone()));
            }
        } else if input.contains(pattern) {
            return Err(SecurityError::BlockedPattern(pattern.clone()));
        }
    }

        // Check for regex patterns
        for regex in MALICIOUS_PATTERNS.iter() {
            if regex.is_match(input) {
                return Err(SecurityError::MaliciousContent(regex.as_str().to_string()));
            }
        }

        // Check for path traversal
        for regex in PATH_TRAVERSAL_PATTERNS.iter() {
            if regex.is_match(input) {
                return Err(SecurityError::PathTraversal);
            }
        }

        // Check for command injection
        for regex in COMMAND_INJECTION_PATTERNS.iter() {
            if regex.is_match(input) {
                return Err(SecurityError::CommandInjection(regex.as_str().to_string()));
            }
        }

        Ok(())
    }

    /// Validate file path for security issues
    pub fn validate_file_path(&self, path: &str) -> Result<(), SecurityError> {
        // Check for path traversal
        for regex in PATH_TRAVERSAL_PATTERNS.iter() {
            if regex.is_match(path) {
                return Err(SecurityError::PathTraversal);
            }
        }

        // Check file extension
        if let Some(extension) = std::path::Path::new(path).extension() {
            if let Some(ext_str) = extension.to_str() {
                if !self.config.allowed_extensions.contains(ext_str) {
                    return Err(SecurityError::InvalidExtension(ext_str.to_string()));
                }
            }
        }

        Ok(())
    }

    /// Sanitize input by removing potentially dangerous content
    pub fn sanitize_input(&self, input: &str) -> String {
        let mut sanitized = input.to_string();

        // Remove HTML tags
        sanitized = RE_HTML_TAG.replace_all(&sanitized, "").to_string();

        // Remove potentially dangerous characters
        let dangerous_chars = vec!['<', '>', '"', '\'', '&', '`', '$', '|', ';'];
        for char in dangerous_chars {
            sanitized = sanitized.replace(char, "");
        }

        // Limit length
        if sanitized.len() > self.config.max_input_length {
            sanitized.truncate(self.config.max_input_length);
        }

        sanitized
    }

    /// Check rate limit (simplified implementation)
    pub fn check_rate_limit(
        &self,
        current_requests: u32,
        window_minutes: u32,
    ) -> Result<(), SecurityError> {
        let allowed = match window_minutes {
            0..=1 => self.config.rate_limit.requests_per_minute,
            2..=60 => self.config.rate_limit.requests_per_hour,
            _ => self.config.rate_limit.requests_per_day,
        };

        if current_requests > allowed {
            warn!(
                "Rate limit exceeded: {} > {} in {} minutes",
                current_requests, allowed, window_minutes
            );
            return Err(SecurityError::RateLimitExceeded(current_requests, allowed));
        }

        Ok(())
    }
}

/// Security error types
#[derive(Debug, thiserror::Error)]
pub enum SecurityError {
    #[error("Input too long: {0} > {1} characters")]
    InputTooLong(usize, usize),

    #[error("Blocked pattern detected: {0}")]
    BlockedPattern(String),

    #[error("Malicious content detected: {0}")]
    MaliciousContent(String),

    #[error("Path traversal attempt detected")]
    PathTraversal,

    #[error("Command injection attempt detected: {0}")]
    CommandInjection(String),

    #[error("Invalid file extension: {0}")]
    InvalidExtension(String),

    #[error("Rate limit exceeded: {0} > {1}")]
    RateLimitExceeded(u32, u32),
}

/// Security utilities
pub struct SecurityUtils;

impl SecurityUtils {
    /// Generate a secure random token using cryptographically secure RNG
    pub fn generate_secure_token(length: usize) -> String {
        use rand::rngs::OsRng;
        use rand::Rng;
        let charset: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

        (0..length)
            .map(|_| charset[OsRng.gen_range(0..charset.len())] as char)
            .collect()
    }

    /// Hash a password using Argon2id (OWASP-recommended)
    /// Returns PHC string: $argon2id$v=19$m=19456,t=2,p=1$<salt>$<hash>
    pub fn hash_password(password: &str) -> Result<String, argon2::password_hash::Error> {
        use argon2::password_hash::{rand_core::OsRng, SaltString};
        use argon2::PasswordHasher;
        let salt = SaltString::generate(&mut OsRng);
        let config = argon2::Argon2::default();
        config
            .hash_password(password.as_bytes(), &salt)
            .map(|hash| hash.to_string())
    }

    /// Verify a password against an Argon2id PHC string
    pub fn verify_password(
        password: &str,
        hash: &str,
    ) -> Result<bool, argon2::password_hash::Error> {
        use argon2::PasswordHash;
        use argon2::PasswordVerifier;
        let parsed_hash = PasswordHash::new(hash)?;
        Ok(argon2::Argon2::default()
            .verify_password(password.as_bytes(), &parsed_hash)
            .is_ok())
    }

    /// Check if a string contains potentially dangerous content
    /// Blocks only explicit secret patterns like "BEGIN PRIVATE KEY",
    /// assignment patterns like "password = ...", or API key regexes.
    /// Normal conversation mentioning "password", "secret", etc. is NOT blocked.
    pub fn is_dangerous_content(content: &str) -> bool {
        let content_lower = content.to_lowercase();

        // Check for actual secrets being disclosed (assignment patterns)
        let assignment_patterns = ["password=", "password :", "password:", "passwd=", "pwd="];
        if assignment_patterns
            .iter()
            .any(|p| content_lower.contains(p))
        {
            // Only flag if there's an actual value after the assignment
            for p in &assignment_patterns {
                if let Some(idx) = content_lower.find(p) {
                    let after = &content_lower[idx + p.len()..];
                    if after.len() > 3 && !after.starts_with("***") {
                        return true;
                    }
                }
            }
        }

        RE_DANGEROUS_PATTERNS.iter().any(|r| r.is_match(content))
    }

    /// Escape HTML content
    pub fn escape_html(content: &str) -> String {
        content
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&#x27;")
            .replace('/', "&#x2F;")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_validation() {
        let config = SecurityConfig::default();
        let validator = SecurityValidator::new(config);

        // Test valid input
        assert!(validator.validate_input("Hello, world!").is_ok());

        // Test input too long
        let long_input = "a".repeat(10001);
        assert!(validator.validate_input(&long_input).is_err());

        // Test malicious patterns
        assert!(validator
            .validate_input("<script>alert('xss')</script>")
            .is_err());
        assert!(validator.validate_input("eval(malicious_code())").is_err());
        assert!(validator.validate_input("../etc/passwd").is_err());
    }

    #[test]
    fn test_file_path_validation() {
        let config = SecurityConfig::default();
        let validator = SecurityValidator::new(config);

        // Test valid file path
        assert!(validator.validate_file_path("src/main.rs").is_ok());

        // Test path traversal
        assert!(validator.validate_file_path("../../../etc/passwd").is_err());

        // Test invalid extension
        assert!(validator.validate_file_path("malware.exe").is_err());
    }

    #[test]
    fn test_input_sanitization() {
        let config = SecurityConfig::default();
        let validator = SecurityValidator::new(config);

        let input = "<script>alert('xss')</script>";
        let sanitized = validator.sanitize_input(input);

        assert!(!sanitized.contains('<'));
        assert!(!sanitized.contains('>'));
        assert!(!sanitized.contains("script"));
    }

    #[test]
    fn test_security_utilities() {
        let token = SecurityUtils::generate_secure_token(32);
        assert_eq!(token.len(), 32);

        let hash = SecurityUtils::hash_password("password").unwrap();
        assert!(SecurityUtils::verify_password("password", &hash).unwrap());
        assert!(!SecurityUtils::verify_password("wrong", &hash).unwrap());

        let not_dangerous =
            SecurityUtils::is_dangerous_content("This mentions password in conversation");
        assert!(!not_dangerous);
        let dangerous = SecurityUtils::is_dangerous_content("password=supersecret123");
        assert!(dangerous);
        let dangerous2 = SecurityUtils::is_dangerous_content("-----BEGIN RSA PRIVATE KEY-----");
        assert!(dangerous2);

        let escaped = SecurityUtils::escape_html("<script>alert('xss')</script>");
        assert!(!escaped.contains('<'));
        assert!(!escaped.contains('>'));
    }
}
