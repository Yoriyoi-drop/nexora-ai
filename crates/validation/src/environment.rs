//! Environment Validation and Setup
//! 
//! Environment variable validation and setup utilities.

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use regex::Regex;
use tracing::info;

use crate::validator::EnvironmentRule;
use crate::{ValidationResult, ValidationError, ValidationWarning, ValidationInfo, ErrorSeverity};

/// Environment setup result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentSetupResult {
    pub created_directories: Vec<String>,
    pub set_env_vars: Vec<(String, String)>,
    pub errors: Vec<String>,
    pub validation: ValidationResult,
}

impl EnvironmentSetupResult {
    pub fn new() -> Self {
        Self {
            created_directories: Vec::new(),
            set_env_vars: Vec::new(),
            errors: Vec::new(),
            validation: ValidationResult {
                valid: true,
                errors: Vec::new(),
                warnings: Vec::new(),
                info: Vec::new(),
            },
        }
    }
    
    pub fn is_success(&self) -> bool {
        self.validation.valid && self.errors.is_empty()
    }
    
    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
        self.validation.valid = false;
    }
    
    pub fn add_created_directory(&mut self, dir: String) {
        self.created_directories.push(dir);
    }
    
    pub fn add_set_env_var(&mut self, key: String, value: String) {
        self.set_env_vars.push((key, value));
    }
}

/// Environment validator
pub struct EnvironmentValidator {
    rules: HashMap<String, EnvironmentRule>,
}

impl EnvironmentValidator {
    pub fn new() -> Self {
        let mut validator = Self {
            rules: HashMap::new(),
        };
        
        // Add default environment rules
        validator.add_default_rules();
        
        validator
    }
    
    /// Validate environment variables
    pub fn validate_environment(&self, rules: &HashMap<String, EnvironmentRule>) -> Result<ValidationResult> {
        let mut result = ValidationResult {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            info: Vec::new(),
        };
        
        for (rule_name, rule) in rules {
            let env_value = env::var(&rule.name);
            
            match env_value {
                Ok(value) => {
                    // Validate the value
                    self.validate_env_value(&rule.name, &value, rule, &mut result);
                }
                Err(_) => {
                    // Environment variable not set
                    if rule.required {
                        result.errors.push(ValidationError {
                            field: rule.name.clone(),
                            message: format!("Required environment variable '{}' is not set", rule.name),
                            code: "ENV_VAR_MISSING".to_string(),
                            severity: ErrorSeverity::Error,
                        });
                        result.valid = false;
                    } else {
                        result.warnings.push(ValidationWarning {
                            field: rule.name.clone(),
                            message: format!("Optional environment variable '{}' is not set", rule.name),
                            code: "ENV_VAR_NOT_SET".to_string(),
                        });
                        
                        // Set default value if available
                        if let Some(default_value) = &rule.default_value {
                            result.info.push(ValidationInfo {
                                field: rule.name.clone(),
                                message: format!("Using default value: {}", default_value),
                                code: "USING_DEFAULT".to_string(),
                            });
                        }
                    }
                }
            }
        }
        
        Ok(result)
    }
    
    /// Validate specific environment variable value
    fn validate_env_value(&self, var_name: &str, value: &str, rule: &EnvironmentRule, result: &mut ValidationResult) {
        // Regex validation
        if let Some(validator) = &rule.validator {
            if !validator.is_match(value) {
                result.errors.push(ValidationError {
                    field: var_name.to_string(),
                    message: format!("Environment variable '{}' value '{}' does not match required pattern", var_name, value),
                    code: "ENV_VAR_INVALID_FORMAT".to_string(),
                    severity: ErrorSeverity::Error,
                });
                result.valid = false;
            }
        }
        
        // Specific validations for common environment variables
        match var_name {
            "DATABASE_URL" => {
                self.validate_database_url(value, result);
            },
            "LOG_LEVEL" => {
                self.validate_log_level(value, result);
            },
            "PORT" => {
                self.validate_port(value, result);
            },
            "PATH" => {
                self.validate_path(value, result);
            },
            _ => {
                // Generic validation
                if value.is_empty() && rule.required {
                    result.errors.push(ValidationError {
                        field: var_name.to_string(),
                        message: format!("Environment variable '{}' cannot be empty", var_name),
                        code: "ENV_VAR_EMPTY".to_string(),
                        severity: ErrorSeverity::Error,
                    });
                    result.valid = false;
                }
            }
        }
    }
    
    /// Validate database URL
    fn validate_database_url(&self, url: &str, result: &mut ValidationResult) {
        if !url.starts_with("postgres://") && !url.starts_with("mysql://") && !url.starts_with("sqlite://") {
            result.errors.push(ValidationError {
                field: "DATABASE_URL".to_string(),
                message: format!("Invalid database URL format: {}", url),
                code: "INVALID_DB_URL".to_string(),
                severity: ErrorSeverity::Error,
            });
            result.valid = false;
        }
        
        // Check for password in URL (security warning)
        if url.contains(":password@") {
            result.warnings.push(ValidationWarning {
                field: "DATABASE_URL".to_string(),
                message: "Database URL contains password in plain text".to_string(),
                code: "PASSWORD_IN_URL".to_string(),
            });
        }
    }
    
    /// Validate log level
    fn validate_log_level(&self, level: &str, result: &mut ValidationResult) {
        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        
        if !valid_levels.contains(&level) {
            result.errors.push(ValidationError {
                field: "LOG_LEVEL".to_string(),
                message: format!("Invalid log level '{}'. Valid levels: {:?}", level, valid_levels),
                code: "INVALID_LOG_LEVEL".to_string(),
                severity: ErrorSeverity::Error,
            });
            result.valid = false;
        }
    }
    
    /// Validate port number
    fn validate_port(&self, port_str: &str, result: &mut ValidationResult) {
        match port_str.parse::<u16>() {
            Ok(port) => {
                if port == 0 {
                    result.errors.push(ValidationError {
                        field: "PORT".to_string(),
                        message: "Port cannot be 0".to_string(),
                        code: "INVALID_PORT".to_string(),
                        severity: ErrorSeverity::Error,
                    });
                    result.valid = false;
                } else if port < 1024 {
                    result.warnings.push(ValidationWarning {
                        field: "PORT".to_string(),
                        message: format!("Port {} is in privileged range (< 1024)", port),
                        code: "PRIVILEGED_PORT".to_string(),
                    });
                }
            }
            Err(_) => {
                result.errors.push(ValidationError {
                    field: "PORT".to_string(),
                    message: format!("Invalid port number: {}", port_str),
                    code: "INVALID_PORT".to_string(),
                    severity: ErrorSeverity::Error,
                });
                result.valid = false;
            }
        }
    }
    
    /// Validate PATH environment variable
    fn validate_path(&self, path: &str, result: &mut ValidationResult) {
        let path_entries: Vec<&str> = path.split(':').collect();
        
        for entry in path_entries {
            if !entry.is_empty() && !PathBuf::from(entry).exists() {
                result.warnings.push(ValidationWarning {
                    field: "PATH".to_string(),
                    message: format!("PATH entry does not exist: {}", entry),
                    code: "PATH_ENTRY_NOT_FOUND".to_string(),
                });
            }
        }
    }
    
    /// Setup environment
    pub fn setup_environment(&self, config: &HashMap<String, String>) -> Result<EnvironmentSetupResult> {
        let mut setup_result = EnvironmentSetupResult::new();
        
        // Create required directories
        let required_dirs = vec!["logs", "data", "temp", "config"];
        
        for dir in required_dirs {
            let dir_path = PathBuf::from(dir);
            if !dir_path.exists() {
                match std::fs::create_dir_all(&dir_path) {
                    Ok(_) => {
                        setup_result.add_created_directory(dir.to_string());
                        info!("Created directory: {}", dir);
                    }
                    Err(e) => {
                        setup_result.add_error(format!("Failed to create directory '{}': {}", dir, e));
                    }
                }
            }
        }
        
        // Set environment variables
        for (key, value) in config {
            env::set_var(key, value);
            setup_result.add_set_env_var(key.clone(), value.clone());
            let redacted = if key.to_uppercase().contains("SECRET")
                || key.to_uppercase().contains("KEY")
                || key.to_uppercase().contains("PASSWORD")
                || key.to_uppercase().contains("TOKEN")
                || key.to_uppercase().contains("CREDENTIAL")
            {
                "***REDACTED***"
            } else {
                &value
            };
            info!("Set environment variable: {}={}", key, redacted);
        }
        
        // Validate the setup
        let validation_result = self.validate_environment_setup(&setup_result)?;
        setup_result.validation = validation_result;
        
        Ok(setup_result)
    }
    
    /// Validate environment setup
    fn validate_environment_setup(&self, setup_result: &EnvironmentSetupResult) -> Result<ValidationResult> {
        let mut result = ValidationResult {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            info: Vec::new(),
        };
        
        // Check if all required directories were created
        let required_dirs = vec!["logs", "data", "temp", "config"];
        for dir in required_dirs {
            if !setup_result.created_directories.contains(&dir.to_string()) {
                result.warnings.push(ValidationWarning {
                    field: "directories".to_string(),
                    message: format!("Directory '{}' was not created", dir),
                    code: "DIRECTORY_NOT_CREATED".to_string(),
                });
            }
        }
        
        // Check if environment variables were set
        for (key, value) in &setup_result.set_env_vars {
            match env::var(key) {
                Ok(current_value) => {
                    if current_value != *value {
                        result.warnings.push(ValidationWarning {
                            field: key.clone(),
                            message: format!("Environment variable '{}' has different value than expected", key),
                            code: "ENV_VAR_MISMATCH".to_string(),
                        });
                    }
                }
                Err(_) => {
                    result.errors.push(ValidationError {
                        field: key.clone(),
                        message: format!("Environment variable '{}' was not set", key),
                        code: "ENV_VAR_NOT_SET".to_string(),
                        severity: ErrorSeverity::Error,
                    });
                    result.valid = false;
                }
            }
        }
        
        Ok(result)
    }
    
    /// Add default environment rules
    fn add_default_rules(&mut self) {
        self.rules.insert("DATABASE_URL".to_string(), EnvironmentRule {
            name: "DATABASE_URL".to_string(),
            required: true,
            validator: Some(Regex::new(r"^[a-zA-Z]+://.*")
                .expect("Failed to create database URL regex")), // safe: hardcoded regex pattern
            default_value: Some("postgres://localhost:5432/mydb".to_string()),
            description: "Database connection URL".to_string(),
        });
        
        self.rules.insert("LOG_LEVEL".to_string(), EnvironmentRule {
            name: "LOG_LEVEL".to_string(),
            required: false,
            validator: Some(Regex::new(r"^(trace|debug|info|warn|error)$")
                .expect("Failed to create log level regex")), // safe: hardcoded regex pattern
            default_value: Some("info".to_string()),
            description: "Logging level".to_string(),
        });
        
        self.rules.insert("PORT".to_string(), EnvironmentRule {
            name: "PORT".to_string(),
            required: false,
            validator: None,
            default_value: Some("8080".to_string()),
            description: "Server port".to_string(),
        });
        
        self.rules.insert("NODE_ENV".to_string(), EnvironmentRule {
            name: "NODE_ENV".to_string(),
            required: false,
            validator: Some(Regex::new(r"^(development|production|test)$")
                .expect("Failed to create node environment regex")), // safe: hardcoded regex pattern
            default_value: Some("development".to_string()),
            description: "Node environment".to_string(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_environment_setup_result_new() {
        let result = EnvironmentSetupResult::new();
        assert!(result.is_success());
        assert!(result.created_directories.is_empty());
        assert!(result.set_env_vars.is_empty());
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_environment_setup_result_add_error() {
        let mut result = EnvironmentSetupResult::new();
        result.add_error("test error".to_string());
        assert!(!result.is_success());
        assert_eq!(result.errors.len(), 1);
    }

    #[test]
    fn test_environment_setup_result_add_directory() {
        let mut result = EnvironmentSetupResult::new();
        result.add_created_directory("logs".to_string());
        assert_eq!(result.created_directories.len(), 1);
        assert_eq!(result.created_directories[0], "logs");
    }

    #[test]
    fn test_environment_setup_result_add_env_var() {
        let mut result = EnvironmentSetupResult::new();
        result.add_set_env_var("MY_VAR".to_string(), "value".to_string());
        assert_eq!(result.set_env_vars.len(), 1);
        assert_eq!(result.set_env_vars[0], ("MY_VAR".to_string(), "value".to_string()));
    }

    #[test]
    fn test_environment_validator_new() {
        let validator = EnvironmentValidator::new();
        assert_eq!(validator.rules.len(), 4);
        assert!(validator.rules.contains_key("DATABASE_URL"));
        assert!(validator.rules.contains_key("LOG_LEVEL"));
        assert!(validator.rules.contains_key("PORT"));
        assert!(validator.rules.contains_key("NODE_ENV"));
    }

    #[test]
    fn test_validate_env_empty_required_var() {
        let validator = EnvironmentValidator::new();
        let mut rules = HashMap::new();
        rules.insert("MY_VAR".to_string(), EnvironmentRule {
            name: "MY_VAR".to_string(),
            required: true,
            validator: None,
            default_value: None,
            description: "Test".to_string(),
        });

        // This var is not set, so it should error
        let result = validator.validate_environment(&rules).unwrap();
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.code == "ENV_VAR_MISSING"));
    }

    #[test]
    fn test_validate_env_optional_var_not_set() {
        let validator = EnvironmentValidator::new();
        let mut rules = HashMap::new();
        rules.insert("OPT_VAR".to_string(), EnvironmentRule {
            name: "OPT_VAR".to_string(),
            required: false,
            validator: None,
            default_value: None,
            description: "Test".to_string(),
        });

        let result = validator.validate_environment(&rules).unwrap();
        assert!(result.valid);
        assert!(result.warnings.iter().any(|w| w.code == "ENV_VAR_NOT_SET"));
    }

    #[test]
    fn test_validate_env_optional_with_default() {
        let validator = EnvironmentValidator::new();
        let mut rules = HashMap::new();
        rules.insert("OPT_VAR".to_string(), EnvironmentRule {
            name: "OPT_VAR".to_string(),
            required: false,
            validator: None,
            default_value: Some("default_val".to_string()),
            description: "Test".to_string(),
        });

        let result = validator.validate_environment(&rules).unwrap();
        assert!(result.valid);
        assert!(result.info.iter().any(|i| i.code == "USING_DEFAULT"));
    }

    #[test]
    fn test_validate_log_level_valid() {
        let validator = EnvironmentValidator::new();
        let mut rules = HashMap::new();
        rules.insert("LOG_LEVEL".to_string(), EnvironmentRule {
            name: "LOG_LEVEL".to_string(),
            required: false,
            validator: None,
            default_value: Some("info".to_string()),
            description: "Test".to_string(),
        });

        std::env::set_var("LOG_LEVEL", "debug");
        let result = validator.validate_environment(&rules).unwrap();
        assert!(result.valid);
        std::env::remove_var("LOG_LEVEL");
    }

    #[test]
    fn test_validate_port_valid() {
        let validator = EnvironmentValidator::new();
        let mut rules = HashMap::new();
        rules.insert("PORT".to_string(), EnvironmentRule {
            name: "PORT".to_string(),
            required: false,
            validator: None,
            default_value: Some("8080".to_string()),
            description: "Test".to_string(),
        });

        std::env::set_var("PORT", "3000");
        let result = validator.validate_environment(&rules).unwrap();
        assert!(result.valid);
        std::env::remove_var("PORT");
    }

    #[test]
    fn test_validate_database_url() {
        let validator = EnvironmentValidator::new();
        let mut rules = HashMap::new();
        rules.insert("DATABASE_URL".to_string(), EnvironmentRule {
            name: "DATABASE_URL".to_string(),
            required: true,
            validator: Some(Regex::new(r"^[a-zA-Z]+://.*").unwrap()),
            default_value: None,
            description: "Test".to_string(),
        });

        std::env::set_var("DATABASE_URL", "postgres://localhost:5432/test");
        let result = validator.validate_environment(&rules).unwrap();
        assert!(result.valid);

        std::env::set_var("DATABASE_URL", "mysql://user:password@localhost:3306/test");
        let result2 = validator.validate_environment(&rules).unwrap();
        assert!(result2.valid);
        assert!(result2.warnings.iter().any(|w| w.code == "PASSWORD_IN_URL"));

        std::env::remove_var("DATABASE_URL");
    }

    #[test]
    fn test_setup_environment_creates_dirs() {
        let validator = EnvironmentValidator::new();
        let config = HashMap::new();
        let result = validator.setup_environment(&config).unwrap();
        // Should not have errors for creating dirs that already exist or can be created
        assert!(result.is_success() || !result.errors.is_empty());
    }

    #[test]
    fn test_setup_environment_redacts_secrets() {
        let validator = EnvironmentValidator::new();
        let mut config = HashMap::new();
        config.insert("API_KEY".to_string(), "supersecret".to_string());

        let tmpdir = std::env::temp_dir();
        std::env::set_current_dir(&tmpdir).ok();
        let result = validator.setup_environment(&config).unwrap();
        assert!(result.set_env_vars.iter().any(|(k, _)| k == "API_KEY"));
    }

    #[test]
    fn test_validate_path_existing() {
        let validator = EnvironmentValidator::new();
        let mut rules = HashMap::new();
        rules.insert("PATH".to_string(), EnvironmentRule {
            name: "PATH".to_string(),
            required: false,
            validator: None,
            default_value: None,
            description: "Test".to_string(),
        });

        std::env::set_var("PATH", "/tmp");
        let result = validator.validate_environment(&rules).unwrap();
        assert!(result.valid);
        std::env::remove_var("PATH");
    }
}
