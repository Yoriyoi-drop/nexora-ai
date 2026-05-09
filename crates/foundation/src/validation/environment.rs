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

use super::{ValidationResult, ValidationError, ValidationWarning, ValidationInfo, ErrorSeverity};

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

/// Environment rule
#[derive(Debug, Clone)]
struct EnvironmentRule {
    name: String,
    required: bool,
    validator: Option<Regex>,
    default_value: Option<String>,
    description: String,
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
            match env::set_var(key, value) {
                Ok(_) => {
                    setup_result.add_set_env_var(key.clone(), value);
                    info!("Set environment variable: {}={}", key, value);
                }
                Err(e) => {
                    setup_result.add_error(format!("Failed to set environment variable '{}': {}", key, e));
                }
            }
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
                .map_err(|e| anyhow::anyhow!("Failed to create database URL regex: {}", e))?),
            default_value: Some("postgres://localhost:5432/mydb".to_string()),
            description: "Database connection URL".to_string(),
        });
        
        self.rules.insert("LOG_LEVEL".to_string(), EnvironmentRule {
            name: "LOG_LEVEL".to_string(),
            required: false,
            validator: Some(Regex::new(r"^(trace|debug|info|warn|error)$")
                .map_err(|e| anyhow::anyhow!("Failed to create log level regex: {}", e))?),
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
                .map_err(|e| anyhow::anyhow!("Failed to create node environment regex: {}", e))?),
            default_value: Some("development".to_string()),
            description: "Node environment".to_string(),
        });
    }
}
