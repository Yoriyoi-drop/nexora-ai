//! Security Validation
//! 
//! Security validation rules and checks for configuration.

use std::collections::HashMap;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use serde_json::Value;
use regex::Regex;
use tracing::warn;

use crate::{ValidationResult, ValidationError, ValidationWarning, ValidationInfo, ErrorSeverity};

/// Security validation rule
#[derive(Debug, Clone)]
pub struct SecurityRule {
    name: String,
    description: String,
    validator: Box<dyn SecurityValidator>,
    severity: ErrorSeverity,
}

/// Security validator trait
pub trait SecurityValidator {
    fn validate(&self, config: &Value) -> Result<Vec<ValidationError>>;
    fn name(&self) -> &str;
}

/// Security validator
pub struct SecurityValidatorImpl {
    rules: Vec<SecurityRule>,
}

impl SecurityValidatorImpl {
    pub fn new() -> Self {
        Self {
            rules: vec![
                SecurityRule {
                    name: "database_ssl".to_string(),
                    description: "Validate database SSL configuration".to_string(),
                    validator: Box::new(DatabaseSSLValidator),
                    severity: ErrorSeverity::Warning,
                },
                SecurityRule {
                    name: "api_key_strength".to_string(),
                    description: "Validate API key strength".to_string(),
                    validator: Box::new(APIKeyStrengthValidator),
                    severity: ErrorSeverity::Error,
                },
                SecurityRule {
                    name: "allowed_hosts".to_string(),
                    description: "Validate allowed hosts configuration".to_string(),
                    validator: Box::new(AllowedHostsValidator),
                    severity: ErrorSeverity::Error,
                },
                SecurityRule {
                    name: "encryption_settings".to_string(),
                    description: "Validate encryption settings".to_string(),
                    validator: Box::new(EncryptionSettingsValidator),
                    severity: ErrorSeverity::Warning,
                },
                SecurityRule {
                    name: "authentication_config".to_string(),
                    description: "Validate authentication configuration".to_string(),
                    validator: Box::new(AuthenticationConfigValidator),
                    severity: ErrorSeverity::Error,
                },
            ],
        }
    }
    
    /// Validate configuration with security rules
    pub fn validate_config(&self, config: &Value, rules: &[SecurityRule]) -> Result<ValidationResult> {
        let mut result = ValidationResult {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            info: Vec::new(),
        };
        
        for rule in rules {
            let validation_errors = rule.validator.validate(config)?;
            
            for error in validation_errors {
                match error.severity {
                    ErrorSeverity::Critical | ErrorSeverity::Error => {
                        result.errors.push(error);
                        result.valid = false;
                    },
                    ErrorSeverity::Warning => {
                        result.warnings.push(ValidationWarning {
                            field: error.field.clone(),
                            message: error.message.clone(),
                            code: error.code.clone(),
                        });
                    },
                    ErrorSeverity::Info => {
                        result.info.push(ValidationInfo {
                            field: error.field.clone(),
                            message: error.message.clone(),
                            code: error.code.clone(),
                        });
                    }
                }
            }
        }
        
        Ok(result)
    }
}

/// Database SSL validator
struct DatabaseSSLValidator;

impl SecurityValidator for DatabaseSSLValidator {
    fn validate(&self, config: &Value) -> Result<Vec<ValidationError>> {
        let mut errors = Vec::new();
        
        if let Some(database) = config.get("database") {
            if let Some(database_obj) = database.as_object() {
                // Check if SSL is enabled for production
                if let Some(ssl_mode) = database_obj.get("ssl_mode") {
                    if let Some(mode) = ssl_mode.as_str() {
                        if mode == "disable" {
                            errors.push(ValidationError {
                                field: "database.ssl_mode".to_string(),
                                message: "SSL is disabled in database configuration".to_string(),
                                code: "SSL_DISABLED".to_string(),
                                severity: ErrorSeverity::Warning,
                            });
                        }
                    }
                }
                
                // Check for SSL certificate configuration
                if let Some(ssl_cert) = database_obj.get("ssl_cert") {
                    if ssl_cert.is_null() {
                        errors.push(ValidationError {
                            field: "database.ssl_cert".to_string(),
                            message: "SSL certificate not configured".to_string(),
                            code: "SSL_CERT_MISSING".to_string(),
                            severity: ErrorSeverity::Warning,
                        });
                    }
                }
                
                // Check for SSL key configuration
                if let Some(ssl_key) = database_obj.get("ssl_key") {
                    if ssl_key.is_null() {
                        errors.push(ValidationError {
                            field: "database.ssl_key".to_string(),
                            message: "SSL key not configured".to_string(),
                            code: "SSL_KEY_MISSING".to_string(),
                            severity: ErrorSeverity::Warning,
                        });
                    }
                }
            }
        }
        
        Ok(errors)
    }
    
    fn name(&self) -> &str {
        "DatabaseSSLValidator"
    }
}

/// API key strength validator
struct APIKeyStrengthValidator;

impl SecurityValidator for APIKeyStrengthValidator {
    fn validate(&self, config: &Value) -> Result<Vec<ValidationError>> {
        let mut errors = Vec::new();
        
        if let Some(api) = config.get("api") {
            if let Some(api_obj) = api.as_object() {
                // Check API key length
                if let Some(api_key) = api_obj.get("key") {
                    if let Some(key) = api_key.as_str() {
                        if key.len() < 32 {
                            errors.push(ValidationError {
                                field: "api.key".to_string(),
                                message: "API key is too short (minimum 32 characters)".to_string(),
                                code: "API_KEY_TOO_SHORT".to_string(),
                                severity: ErrorSeverity::Error,
                            });
                        }
                        
                        if key.len() > 512 {
                            errors.push(ValidationError {
                                field: "api.key".to_string(),
                                message: "API key is too long (maximum 512 characters)".to_string(),
                                code: "API_KEY_TOO_LONG".to_string(),
                                severity: ErrorSeverity::Warning,
                            });
                        }
                        
                        // Check key complexity
                        if !self.is_complex_key(key) {
                            errors.push(ValidationError {
                                field: "api.key".to_string(),
                                message: "API key lacks complexity (should include uppercase, lowercase, numbers, and special characters)".to_string(),
                                code: "API_KEY_NOT_COMPLEX".to_string(),
                                severity: ErrorSeverity::Error,
                            });
                        }
                    }
                }
                
                // Check if API key is hardcoded
                if let Some(key) = api_obj.get("key") {
                    if let Some(key_str) = key.as_str() {
                        if key_str == "your-api-key-here" || key_str == "test-key" {
                            errors.push(ValidationError {
                                field: "api.key".to_string(),
                                message: "API key appears to be a default/test value".to_string(),
                                code: "API_KEY_DEFAULT_VALUE".to_string(),
                                severity: ErrorSeverity::Critical,
                            });
                        }
                    }
                }
            }
        }
        
        Ok(errors)
    }
    
    fn name(&self) -> &str {
        "APIKeyStrengthValidator"
    }
    
    fn is_complex_key(&self, key: &str) -> bool {
        let has_uppercase = key.chars().any(|c| c.is_uppercase());
        let has_lowercase = key.chars().any(|c| c.is_lowercase());
        let has_numbers = key.chars().any(|c| c.is_numeric());
        let has_special = key.chars().any(|c| !c.is_alphanumeric());
        
        let complexity_score = [has_uppercase, has_lowercase, has_numbers, has_special]
            .iter()
            .map(|&has| if has { 1 } else { 0 })
            .sum::<i32>();
        
        complexity_score >= 3
    }
}

/// Allowed hosts validator
struct AllowedHostsValidator;

impl SecurityValidator for AllowedHostsValidator {
    fn validate(&self, config: &Value) -> Result<Vec<ValidationError>> {
        let mut errors = Vec::new();
        
        if let Some(server) = config.get("server") {
            if let Some(server_obj) = server.as_object() {
                if let Some(host) = server_obj.get("host") {
                    if let Some(host_str) = host.as_str() {
                        // Check for localhost in production
                        if host_str == "localhost" || host_str == "127.0.0.1" {
                            errors.push(ValidationError {
                                field: "server.host".to_string(),
                                message: "Localhost detected - not suitable for production".to_string(),
                                code: "LOCALHOST_HOST".to_string(),
                                severity: ErrorSeverity::Warning,
                            });
                        }
                        
                        // Check for wildcard hosts
                        if host_str.starts_with("*.") {
                            errors.push(ValidationError {
                                field: "server.host".to_string(),
                                message: "Wildcard host detected - potential security risk".to_string(),
                                code: "WILDCARD_HOST".to_string(),
                                severity: ErrorSeverity::Error,
                            });
                        }
                        
                        // Check for HTTP instead of HTTPS
                        if let Some(ssl) = server_obj.get("ssl") {
                            if let Some(ssl_enabled) = ssl.as_bool() {
                                if !ssl_enabled {
                                    errors.push(ValidationError {
                                        field: "server.ssl".to_string(),
                                        message: "SSL/TLS is disabled - insecure communication".to_string(),
                                        code: "SSL_DISABLED".to_string(),
                                        severity: ErrorSeverity::Critical,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(errors)
    }
    
    fn name(&self) -> &str {
        "AllowedHostsValidator"
    }
}

/// Encryption settings validator
struct EncryptionSettingsValidator;

impl SecurityValidator for EncryptionSettingsValidator {
    fn validate(&self, config: &Value) -> Result<Vec<ValidationError>> {
        let mut errors = Vec::new();
        
        if let Some(encryption) = config.get("encryption") {
            if let Some(encryption_obj) = encryption.as_object() {
                // Check encryption algorithm
                if let Some(algorithm) = encryption_obj.get("algorithm") {
                    if let Some(algo) = algorithm.as_str() {
                        match algo {
                            "des" | "rc4" | "md5" | "sha1" => {
                                errors.push(ValidationError {
                                    field: "encryption.algorithm".to_string(),
                                    message: format!("Weak encryption algorithm detected: {}", algo),
                                    code: "WEAK_ENCRYPTION".to_string(),
                                    severity: ErrorSeverity::Error,
                                });
                            },
                            _ => {} // Strong algorithms are OK
                        }
                    }
                }
                
                // Check key length
                if let Some(key_size) = encryption_obj.get("key_size") {
                    if let Some(size) = key_size.as_u64() {
                        if size < 128 {
                            errors.push(ValidationError {
                                field: "encryption.key_size".to_string(),
                                message: format!("Encryption key size too small: {} bits (minimum 128)", size),
                                code: "KEY_SIZE_TOO_SMALL".to_string(),
                                severity: ErrorSeverity::Error,
                            });
                        }
                    }
                }
                
                // Check for deprecated encryption modes
                if let Some(mode) = encryption_obj.get("mode") {
                    if let Some(mode_str) = mode.as_str() {
                        if mode_str == "ecb" {
                            errors.push(ValidationError {
                                field: "encryption.mode".to_string(),
                                message: "ECB mode is deprecated and insecure".to_string(),
                                code: "DEPRECATED_MODE".to_string(),
                                severity: ErrorSeverity::Error,
                            });
                        }
                    }
                }
            }
        }
        
        Ok(errors)
    }
    
    fn name(&self) -> &str {
        "EncryptionSettingsValidator"
    }
}

/// Authentication configuration validator
struct AuthenticationConfigValidator;

impl SecurityValidator for AuthenticationConfigValidator {
    fn validate(&self, config: &Value) -> Result<Vec<ValidationError>> {
        let mut errors = Vec::new();
        
        if let Some(auth) = config.get("authentication") {
            if let Some(auth_obj) = auth.as_object() {
                // Check password policy
                if let Some(password_policy) = auth_obj.get("password_policy") {
                    if let Some(policy_obj) = password_policy.as_object() {
                        // Check minimum password length
                        if let Some(min_length) = policy_obj.get("min_length") {
                            if let Some(length) = min_length.as_u64() {
                                if length < 8 {
                                    errors.push(ValidationError {
                                        field: "authentication.password_policy.min_length".to_string(),
                                        message: format!("Password minimum length too small: {} (minimum 8)", length),
                                        code: "PASSWORD_MIN_LENGTH".to_string(),
                                        severity: ErrorSeverity::Error,
                                    });
                                }
                            }
                        }
                        
                        // Check if password complexity is required
                        if let Some(require_complexity) = policy_obj.get("require_complexity") {
                            if let Some(complexity) = require_complexity.as_bool() {
                                if !complexity {
                                    errors.push(ValidationError {
                                        field: "authentication.password_policy.require_complexity".to_string(),
                                        message: "Password complexity should be required".to_string(),
                                        code: "PASSWORD_COMPLEXITY_REQUIRED".to_string(),
                                        severity: ErrorSeverity::Warning,
                                    });
                                }
                            }
                        }
                    }
                }
                
                // Check session timeout
                if let Some(session) = auth_obj.get("session") {
                    if let Some(session_obj) = session.as_object() {
                        if let Some(timeout) = session_obj.get("timeout") {
                            if let Some(timeout_minutes) = timeout.as_u64() {
                                if timeout_minutes > 1440 { // 24 hours
                                    errors.push(ValidationError {
                                        field: "authentication.session.timeout".to_string(),
                                        message: format!("Session timeout too long: {} minutes (maximum 1440)", timeout_minutes),
                                        code: "SESSION_TIMEOUT_TOO_LONG".to_string(),
                                        severity: ErrorSeverity::Warning,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(errors)
    }
    
    fn name(&self) -> &str {
        "AuthenticationConfigValidator"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_security_validator_impl_new() {
        let validator = SecurityValidatorImpl::new();
        // Should have 5 default rules
        assert_eq!(validator.rules.len(), 5);
    }

    #[test]
    fn test_database_ssl_no_database_config() {
        let validator = DatabaseSSLValidator;
        let config = json!({});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_database_ssl_disabled() {
        let validator = DatabaseSSLValidator;
        let config = json!({"database": {"ssl_mode": "disable"}});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.iter().any(|e| e.code == "SSL_DISABLED"));
    }

    #[test]
    fn test_database_ssl_enabled() {
        let validator = DatabaseSSLValidator;
        let config = json!({"database": {"ssl_mode": "require"}});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_database_ssl_missing_cert() {
        let validator = DatabaseSSLValidator;
        let config = json!({"database": {"ssl_cert": null}});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.iter().any(|e| e.code == "SSL_CERT_MISSING"));
    }

    #[test]
    fn test_database_ssl_cert_present() {
        let validator = DatabaseSSLValidator;
        let config = json!({"database": {"ssl_cert": "/path/to/cert.pem"}});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_api_key_strength_no_api_config() {
        let validator = APIKeyStrengthValidator;
        let config = json!({});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_api_key_too_short() {
        let validator = APIKeyStrengthValidator;
        let config = json!({"api": {"key": "short"}});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.iter().any(|e| e.code == "API_KEY_TOO_SHORT"));
    }

    #[test]
    fn test_api_key_default_value() {
        let validator = APIKeyStrengthValidator;
        let config = json!({"api": {"key": "test-key"}});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.iter().any(|e| e.code == "API_KEY_DEFAULT_VALUE"));
    }

    #[test]
    fn test_api_key_strong() {
        let validator = APIKeyStrengthValidator;
        let config = json!({"api": {"key": "aB3$xYz1!qR7#mN9@pL2&kD5*"}});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_api_key_not_complex() {
        let validator = APIKeyStrengthValidator;
        let config = json!({"api": {"key": "abcdefghijklmnopqrstuvwxyz123456"}});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.iter().any(|e| e.code == "API_KEY_NOT_COMPLEX"));
    }

    #[test]
    fn test_is_complex_key() {
        let validator = APIKeyStrengthValidator;
        assert!(validator.is_complex_key("aB3$xYz1!"));
        assert!(!validator.is_complex_key("abcdefgh"));
        assert!(validator.is_complex_key("ABCdef123"));
        assert!(!validator.is_complex_key("12345678"));
    }

    #[test]
    fn test_allowed_hosts_no_server_config() {
        let validator = AllowedHostsValidator;
        let config = json!({});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_allowed_hosts_localhost() {
        let validator = AllowedHostsValidator;
        let config = json!({"server": {"host": "127.0.0.1"}});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.iter().any(|e| e.code == "LOCALHOST_HOST"));
    }

    #[test]
    fn test_allowed_hosts_wildcard() {
        let validator = AllowedHostsValidator;
        let config = json!({"server": {"host": "*.example.com"}});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.iter().any(|e| e.code == "WILDCARD_HOST"));
    }

    #[test]
    fn test_allowed_hosts_ssl_disabled() {
        let validator = AllowedHostsValidator;
        let config = json!({"server": {"host": "example.com", "ssl": false}});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.iter().any(|e| e.code == "SSL_DISABLED"));
    }

    #[test]
    fn test_allowed_hosts_production_ok() {
        let validator = AllowedHostsValidator;
        let config = json!({"server": {"host": "api.example.com", "ssl": true}});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_encryption_settings_no_config() {
        let validator = EncryptionSettingsValidator;
        let config = json!({});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_encryption_weak_algorithm() {
        let validator = EncryptionSettingsValidator;
        let config = json!({"encryption": {"algorithm": "des"}});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.iter().any(|e| e.code == "WEAK_ENCRYPTION"));
    }

    #[test]
    fn test_encryption_strong_algorithm() {
        let validator = EncryptionSettingsValidator;
        let config = json!({"encryption": {"algorithm": "aes-256-gcm"}});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_encryption_key_too_small() {
        let validator = EncryptionSettingsValidator;
        let config = json!({"encryption": {"key_size": 64}});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.iter().any(|e| e.code == "KEY_SIZE_TOO_SMALL"));
    }

    #[test]
    fn test_encryption_key_ok() {
        let validator = EncryptionSettingsValidator;
        let config = json!({"encryption": {"key_size": 256}});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_encryption_deprecated_mode() {
        let validator = EncryptionSettingsValidator;
        let config = json!({"encryption": {"mode": "ecb"}});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.iter().any(|e| e.code == "DEPRECATED_MODE"));
    }

    #[test]
    fn test_authentication_no_config() {
        let validator = AuthenticationConfigValidator;
        let config = json!({});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_authentication_short_password() {
        let validator = AuthenticationConfigValidator;
        let config = json!({"authentication": {"password_policy": {"min_length": 4}}});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.iter().any(|e| e.code == "PASSWORD_MIN_LENGTH"));
    }

    #[test]
    fn test_authentication_password_length_ok() {
        let validator = AuthenticationConfigValidator;
        let config = json!({"authentication": {"password_policy": {"min_length": 12}}});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_authentication_complexity_not_required() {
        let validator = AuthenticationConfigValidator;
        let config = json!({"authentication": {"password_policy": {"require_complexity": false}}});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.iter().any(|e| e.code == "PASSWORD_COMPLEXITY_REQUIRED"));
    }

    #[test]
    fn test_authentication_session_too_long() {
        let validator = AuthenticationConfigValidator;
        let config = json!({"authentication": {"session": {"timeout": 9999}}});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.iter().any(|e| e.code == "SESSION_TIMEOUT_TOO_LONG"));
    }

    #[test]
    fn test_authentication_session_ok() {
        let validator = AuthenticationConfigValidator;
        let config = json!({"authentication": {"session": {"timeout": 60}}});
        let errors = validator.validate(&config).unwrap();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_security_validators_have_names() {
        assert_eq!(DatabaseSSLValidator.name(), "DatabaseSSLValidator");
        assert_eq!(APIKeyStrengthValidator.name(), "APIKeyStrengthValidator");
        assert_eq!(AllowedHostsValidator.name(), "AllowedHostsValidator");
        assert_eq!(EncryptionSettingsValidator.name(), "EncryptionSettingsValidator");
        assert_eq!(AuthenticationConfigValidator.name(), "AuthenticationConfigValidator");
    }
}
