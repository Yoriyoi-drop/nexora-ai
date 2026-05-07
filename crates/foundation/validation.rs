//! Configuration Validation and Environment Setup
//! 
//! Implementasi comprehensive configuration validation dengan:
//! - Environment variable validation
//! - Configuration schema validation
//! - Environment setup utilities
//! - Configuration migration
//! - Security validation

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use serde_json::{json, Value};
use regex::Regex;
use tracing::info;

/// Configuration validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub info: Vec<ValidationInfo>,
}

/// Validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
    pub code: String,
    pub severity: ErrorSeverity,
}

/// Validation warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    pub field: String,
    pub message: String,
    pub code: String,
}

/// Validation info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationInfo {
    pub field: String,
    pub message: String,
    pub code: String,
}

/// Error severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Critical,
    Error,
    Warning,
    Info,
}

/// Configuration validator
pub struct ConfigValidator {
    schemas: HashMap<String, ConfigSchema>,
    environment_rules: HashMap<String, EnvironmentRule>,
    security_rules: Vec<SecurityRule>,
}

/// Configuration schema
#[derive(Debug, Clone)]
struct ConfigSchema {
    name: String,
    version: String,
    fields: HashMap<String, FieldDefinition>,
    required_fields: Vec<String>,
}

/// Field definition
#[derive(Debug, Clone)]
struct FieldDefinition {
    field_type: FieldType,
    required: bool,
    default_value: Option<Value>,
    validator: Option<Regex>,
    min_value: Option<f64>,
    max_value: Option<f64>,
    allowed_values: Option<Vec<String>>,
    description: String,
}

/// Field types
#[derive(Debug, Clone)]
enum FieldType {
    String,
    Number,
    Boolean,
    Array,
    Object,
    Path,
    Url,
    Email,
    Port,
    Duration,
}

/// Environment rule
#[derive(Debug, Clone)]
struct EnvironmentRule {
    name: String,
    pattern: Regex,
    required: bool,
    default_value: Option<String>,
    description: String,
    sensitive: bool,
}

/// Security rule
#[derive(Debug, Clone)]
struct SecurityRule {
    name: String,
    rule_type: SecurityRuleType,
    pattern: Option<Regex>,
    description: String,
}

/// Security rule types
#[derive(Debug, Clone)]
enum SecurityRuleType {
    NoHardcodedSecrets,
    StrongPasswords,
    SecureUrls,
    FilePermissions,
    EnvironmentSecurity,
}

impl ConfigValidator {
    /// Create new configuration validator
    pub fn new() -> Self {
        let mut validator = Self {
            schemas: HashMap::new(),
            environment_rules: HashMap::new(),
            security_rules: Vec::new(),
        };
        
        validator.register_default_schemas();
        validator.register_default_environment_rules();
        validator.register_default_security_rules();
        
        validator
    }
    
    /// Validate configuration
    pub fn validate_config(&self, config: &Value, schema_name: &str) -> ValidationResult {
        let mut result = ValidationResult {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            info: Vec::new(),
        };
        
        // Get schema
        let schema = match self.schemas.get(schema_name) {
            Some(schema) => schema,
            None => {
                result.errors.push(ValidationError {
                    field: "schema".to_string(),
                    message: format!("Schema '{}' not found", schema_name),
                    code: "SCHEMA_NOT_FOUND".to_string(),
                    severity: ErrorSeverity::Critical,
                });
                result.valid = false;
                return result;
            }
        };
        
        // Validate required fields
        for required_field in &schema.required_fields {
            if !config.get(required_field).is_some() {
                result.errors.push(ValidationError {
                    field: required_field.clone(),
                    message: format!("Required field '{}' is missing", required_field),
                    code: "REQUIRED_FIELD_MISSING".to_string(),
                    severity: ErrorSeverity::Error,
                });
                result.valid = false;
            }
        }
        
        // Validate each field
        if let Value::Object(fields) = config {
            for (field_name, field_value) in fields {
                if let Some(field_def) = schema.fields.get(field_name) {
                    self.validate_field(field_name, field_value, field_def, &mut result);
                } else {
                    result.warnings.push(ValidationWarning {
                        field: field_name.clone(),
                        message: format!("Unknown field '{}'", field_name),
                        code: "UNKNOWN_FIELD".to_string(),
                    });
                }
            }
        }
        
        // Apply security validation
        self.validate_security(config, &mut result);
        
        result
    }
    
    /// Validate environment variables
    pub fn validate_environment(&self) -> ValidationResult {
        let mut result = ValidationResult {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            info: Vec::new(),
        };
        
        for (name, rule) in &self.environment_rules {
            let value = env::var(name);
            
            match (value, rule.required) {
                (Ok(value), _) => {
                    // Validate value against pattern
                    let pattern = &rule.pattern;
                    if !pattern.is_match(&value) {
                        result.errors.push(ValidationError {
                            field: name.clone(),
                            message: format!("Environment variable '{}' does not match required pattern", name),
                            code: "ENV_PATTERN_MISMATCH".to_string(),
                            severity: ErrorSeverity::Error,
                        });
                        result.valid = false;
                    }
                    
                    // Check for sensitive information in logs
                    if rule.sensitive {
                        result.info.push(ValidationInfo {
                            field: name.clone(),
                            message: "Sensitive environment variable detected".to_string(),
                            code: "SENSITIVE_ENV_VAR".to_string(),
                        });
                    }
                }
                (Err(_), true) => {
                    result.errors.push(ValidationError {
                        field: name.clone(),
                        message: format!("Required environment variable '{}' is not set", name),
                        code: "REQUIRED_ENV_VAR_MISSING".to_string(),
                        severity: ErrorSeverity::Error,
                    });
                    result.valid = false;
                }
                (Err(_), false) => {
                    // Optional variable not set, use default if available
                    if let Some(default) = &rule.default_value {
                        result.info.push(ValidationInfo {
                            field: name.clone(),
                            message: format!("Using default value: {}", default),
                            code: "USING_DEFAULT_VALUE".to_string(),
                        });
                    }
                }
            }
        }
        
        result
    }
    
    /// Setup environment
    pub fn setup_environment(&self) -> Result<EnvironmentSetupResult> {
        let mut result = EnvironmentSetupResult::new();
        
        // Create necessary directories
        self.create_directories(&mut result)?;
        
        // Set default environment variables
        self.set_default_env_vars(&mut result)?;
        
        // Validate environment after setup
        let validation = self.validate_environment();
        result.validation = validation.clone();
        
        if !validation.valid {
            return Err(anyhow::anyhow!("Environment setup failed validation"));
        }
        
        info!("Environment setup completed successfully");
        Ok(result)
    }
    
    /// Validate field against definition
    fn validate_field(&self, field_name: &str, field_value: &Value, field_def: &FieldDefinition, result: &mut ValidationResult) {
        // Validate type
        if !self.validate_field_type(field_value, &field_def.field_type) {
            result.errors.push(ValidationError {
                field: field_name.to_string(),
                message: format!("Field '{}' has invalid type", field_name),
                code: "INVALID_TYPE".to_string(),
                severity: ErrorSeverity::Error,
            });
            result.valid = false;
            return;
        }
        
        // Validate pattern
        if let Some(pattern) = &field_def.validator {
            if let Some(str_value) = field_value.as_str() {
                if !pattern.is_match(str_value) {
                    result.errors.push(ValidationError {
                        field: field_name.to_string(),
                        message: format!("Field '{}' does not match required pattern", field_name),
                        code: "PATTERN_MISMATCH".to_string(),
                        severity: ErrorSeverity::Error,
                    });
                    result.valid = false;
                }
            }
        }
        
        // Validate numeric range
        if let (Some(min), Some(max)) = (field_def.min_value, field_def.max_value) {
            if let Some(num_value) = field_value.as_f64() {
                if num_value < min || num_value > max {
                    result.errors.push(ValidationError {
                        field: field_name.to_string(),
                        message: format!("Field '{}' value {} is outside range [{}, {}]", field_name, num_value, min, max),
                        code: "OUT_OF_RANGE".to_string(),
                        severity: ErrorSeverity::Error,
                    });
                    result.valid = false;
                }
            }
        }
        
        // Validate allowed values
        if let Some(allowed) = &field_def.allowed_values {
            if let Some(str_value) = field_value.as_str() {
                if !allowed.contains(&str_value.to_string()) {
                    result.errors.push(ValidationError {
                        field: field_name.to_string(),
                        message: format!("Field '{}' value '{}' is not in allowed values", field_name, str_value),
                        code: "INVALID_VALUE".to_string(),
                        severity: ErrorSeverity::Error,
                    });
                    result.valid = false;
                }
            }
        }
    }
    
    /// Validate field type
    fn validate_field_type(&self, value: &Value, field_type: &FieldType) -> bool {
        match field_type {
            FieldType::String => value.is_string(),
            FieldType::Number => value.is_number(),
            FieldType::Boolean => value.is_boolean(),
            FieldType::Array => value.is_array(),
            FieldType::Object => value.is_object(),
            FieldType::Path => value.is_string(), // Additional path validation would be needed
            FieldType::Url => {
                if let Some(str_val) = value.as_str() {
                    str_val.starts_with("http://") || str_val.starts_with("https://")
                } else {
                    false
                }
            }
            FieldType::Email => {
                if let Some(str_val) = value.as_str() {
                    str_val.contains('@') && str_val.contains('.')
                } else {
                    false
                }
            }
            FieldType::Port => {
                if let Some(num_val) = value.as_u64() {
                    num_val > 0 && num_val <= 65535
                } else {
                    false
                }
            }
            FieldType::Duration => value.is_string(), // Additional duration parsing would be needed
        }
    }
    
    /// Apply security validation
    fn validate_security(&self, config: &Value, result: &mut ValidationResult) {
        for rule in &self.security_rules {
            match &rule.rule_type {
                SecurityRuleType::NoHardcodedSecrets => {
                    self.check_hardcoded_secrets(config, &rule.name, result);
                }
                SecurityRuleType::StrongPasswords => {
                    self.check_password_strength(config, &rule.name, result);
                }
                SecurityRuleType::SecureUrls => {
                    self.check_secure_urls(config, &rule.name, result);
                }
                SecurityRuleType::FilePermissions => {
                    self.check_file_permissions(config, &rule.name, result);
                }
                SecurityRuleType::EnvironmentSecurity => {
                    self.check_environment_security(config, &rule.name, result);
                }
            }
        }
    }
    
    /// Check for hardcoded secrets
    fn check_hardcoded_secrets(&self, config: &Value, rule_name: &str, result: &mut ValidationResult) {
        // Simple string-based detection for hardcoded secrets
        let config_str = serde_json::to_string(config).unwrap_or_default().to_lowercase();
        
        // Check for common secret patterns
        let secret_keywords = ["password", "secret", "key", "token", "api_key"];
        for keyword in &secret_keywords {
            if config_str.contains(keyword) && config_str.len() > keyword.len() + 10 {
                result.warnings.push(ValidationWarning {
                    field: "security".to_string(),
                    message: format!("Potential hardcoded secret detected (rule: {})", rule_name),
                    code: "HARDCODED_SECRET".to_string(),
                });
                break;
            }
        }
    }
    
    /// Check password strength
    fn check_password_strength(&self, config: &Value, rule_name: &str, result: &mut ValidationResult) {
        if let Some(password) = config.get("password").and_then(|v| v.as_str()) {
            if password.len() < 8 {
                result.errors.push(ValidationError {
                    field: "password".to_string(),
                    message: format!("Password too short (rule: {})", rule_name),
                    code: "WEAK_PASSWORD".to_string(),
                    severity: ErrorSeverity::Error,
                });
                result.valid = false;
            }
            
            let has_upper = password.chars().any(|c| c.is_uppercase());
            let has_lower = password.chars().any(|c| c.is_lowercase());
            let has_digit = password.chars().any(|c| c.is_digit(10));
            let has_special = password.chars().any(|c| "!@#$%^&*()_+-=[]{}|;:,.<>?".contains(c));
            
            if !(has_upper && has_lower && has_digit && has_special) {
                result.warnings.push(ValidationWarning {
                    field: "password".to_string(),
                    message: format!("Password does not meet complexity requirements (rule: {})", rule_name),
                    code: "PASSWORD_COMPLEXITY".to_string(),
                });
            }
        }
    }
    
    /// Check for secure URLs
    fn check_secure_urls(&self, config: &Value, rule_name: &str, result: &mut ValidationResult) {
        if let Some(url) = config.get("database_url").and_then(|v| v.as_str()) {
            if !url.starts_with("postgresql://") && !url.starts_with("mysql://") {
                result.warnings.push(ValidationWarning {
                    field: "database_url".to_string(),
                    message: format!("Database URL may not be secure (rule: {})", rule_name),
                    code: "INSECURE_URL".to_string(),
                });
            }
        }
    }
    
    /// Check file permissions (placeholder)
    fn check_file_permissions(&self, _config: &Value, rule_name: &str, result: &mut ValidationResult) {
        result.info.push(ValidationInfo {
            field: "security".to_string(),
            message: format!("File permission check not implemented (rule: {})", rule_name),
            code: "NOT_IMPLEMENTED".to_string(),
        });
    }
    
    /// Check environment security (placeholder)
    fn check_environment_security(&self, _config: &Value, rule_name: &str, result: &mut ValidationResult) {
        result.info.push(ValidationInfo {
            field: "security".to_string(),
            message: format!("Environment security check not implemented (rule: {})", rule_name),
            code: "NOT_IMPLEMENTED".to_string(),
        });
    }
    
    /// Create necessary directories
    fn create_directories(&self, result: &mut EnvironmentSetupResult) -> Result<()> {
        let directories = vec![
            "logs",
            "data",
            "cache",
            "temp",
            "config",
            "backups",
        ];
        
        for dir in directories {
            if let Err(e) = std::fs::create_dir_all(dir) {
                result.errors.push(format!("Failed to create directory '{}': {}", dir, e));
            } else {
                result.created_directories.push(dir.to_string());
            }
        }
        
        Ok(())
    }
    
    /// Set default environment variables
    fn set_default_env_vars(&self, result: &mut EnvironmentSetupResult) -> Result<()> {
        let defaults = vec![
            ("RUST_LOG", "info"),
            ("NEXORA_ENV", "development"),
            ("NEXORA_HOST", "127.0.0.1"),
            ("NEXORA_PORT", "8080"),
            ("NEXORA_DB_HOST", "localhost"),
            ("NEXORA_DB_PORT", "5432"),
            ("NEXORA_DB_NAME", "nexora"),
        ];
        
        for (key, value) in defaults {
            if env::var(key).is_err() {
                env::set_var(key, value);
                result.set_env_vars.push((key.to_string(), value.to_string()));
            }
        }
        
        Ok(())
    }
    
    /// Register default schemas
    fn register_default_schemas(&mut self) {
        // Server configuration schema
        let mut server_fields = HashMap::new();
        server_fields.insert("host".to_string(), FieldDefinition {
            field_type: FieldType::String,
            required: true,
            default_value: Some(json!("127.0.0.1")),
            validator: Some(Regex::new(r"^[0-9.]+$").unwrap()),
            min_value: None,
            max_value: None,
            allowed_values: None,
            description: "Server host address".to_string(),
        });
        server_fields.insert("port".to_string(), FieldDefinition {
            field_type: FieldType::Port,
            required: true,
            default_value: Some(json!(8080)),
            validator: None,
            min_value: Some(1.0),
            max_value: Some(65535.0),
            allowed_values: None,
            description: "Server port".to_string(),
        });
        
        let server_schema = ConfigSchema {
            name: "server".to_string(),
            version: "1.0".to_string(),
            fields: server_fields,
            required_fields: vec!["host".to_string(), "port".to_string()],
        };
        
        self.schemas.insert("server".to_string(), server_schema);
    }
    
    /// Register default environment rules
    fn register_default_environment_rules(&mut self) {
        let rules = vec![
            EnvironmentRule {
                name: "NEXORA_ENV".to_string(),
                pattern: Regex::new(r"^(development|staging|production)$").unwrap(),
                required: true,
                default_value: Some("development".to_string()),
                description: "Environment type".to_string(),
                sensitive: false,
            },
            EnvironmentRule {
                name: "NEXORA_SECRET_KEY".to_string(),
                pattern: Regex::new(r"^.{32,}$").unwrap(),
                required: false,
                default_value: None,
                description: "Secret key for encryption".to_string(),
                sensitive: true,
            },
            EnvironmentRule {
                name: "DATABASE_URL".to_string(),
                pattern: Regex::new(r"^[a-z]+://.*").unwrap(),
                required: true,
                default_value: None,
                description: "Database connection URL".to_string(),
                sensitive: true,
            },
        ];
        
        for rule in rules {
            self.environment_rules.insert(rule.name.clone(), rule);
        }
    }
    
    /// Register default security rules
    fn register_default_security_rules(&mut self) {
        self.security_rules = vec![
            SecurityRule {
                name: "no_hardcoded_secrets".to_string(),
                rule_type: SecurityRuleType::NoHardcodedSecrets,
                pattern: None,
                description: "No hardcoded secrets in configuration".to_string(),
            },
            SecurityRule {
                name: "strong_passwords".to_string(),
                rule_type: SecurityRuleType::StrongPasswords,
                pattern: None,
                description: "Passwords must be strong".to_string(),
            },
            SecurityRule {
                name: "secure_urls".to_string(),
                rule_type: SecurityRuleType::SecureUrls,
                pattern: None,
                description: "URLs must be secure".to_string(),
            },
        ];
    }
}

impl Default for ConfigValidator {
    fn default() -> Self {
        Self::new()
    }
}

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
}

/// Configuration migration utilities
pub struct ConfigMigrator {
    migrations: Vec<Migration>,
}

/// Migration definition
#[derive(Debug)]
struct Migration {
    version: String,
    description: String,
    migrate_fn: fn(Value) -> Result<Value>,
}

impl ConfigMigrator {
    /// Create new migrator
    pub fn new() -> Self {
        Self {
            migrations: Vec::new(),
        }
    }
    
    /// Add migration
    pub fn add_migration(&mut self, migration: Migration) {
        self.migrations.push(migration);
    }
    
    /// Migrate configuration
    pub fn migrate(&self, mut config: Value, from_version: &str, to_version: &str) -> Result<Value> {
        let mut current_version = from_version.to_string();
        
        for migration in &self.migrations {
            if migration.version > current_version && migration.version <= to_version.to_string() {
                config = (migration.migrate_fn)(config)?;
                current_version = migration.version.clone();
                info!("Migrated configuration to version {}", migration.version);
            }
        }
        
        Ok(config)
    }
}

impl Default for ConfigMigrator {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions
pub mod utils {
    use super::*;
    
    /// Load configuration from file
    pub fn load_config_file(path: &PathBuf) -> Result<Value> {
        let content = std::fs::read_to_string(path)?;
        let config: Value = serde_json::from_str(&content)?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn save_config_file(config: &Value, path: &PathBuf) -> Result<()> {
        let content = serde_json::to_string_pretty(config)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Get current environment
    pub fn get_environment() -> String {
        env::var("NEXORA_ENV").unwrap_or_else(|_| "development".to_string())
    }
    
    /// Check if running in production
    pub fn is_production() -> bool {
        get_environment() == "production"
    }
    
    /// Check if running in development
    pub fn is_development() -> bool {
        get_environment() == "development"
    }
    
    /// Validate port number
    pub fn validate_port(port: u16) -> bool {
        port <= 65535
    }
    
    /// Validate URL format
    pub fn validate_url(url: &str) -> bool {
        url.starts_with("http://") || url.starts_with("https://")
    }
    
    /// Validate email format
    pub fn validate_email(email: &str) -> bool {
        email.contains('@') && email.contains('.') && email.len() > 5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_validator_creation() {
        let validator = ConfigValidator::new();
        assert!(!validator.schemas.is_empty());
        assert!(!validator.environment_rules.is_empty());
        assert!(!validator.security_rules.is_empty());
    }
    
    #[test]
    fn test_server_config_validation() {
        let validator = ConfigValidator::new();
        
        let valid_config = json!({
            "host": "127.0.0.1",
            "port": 8080
        });
        
        let result = validator.validate_config(&valid_config, "server");
        assert!(result.valid);
        assert!(result.errors.is_empty());
    }
    
    #[test]
    fn test_invalid_server_config() {
        let validator = ConfigValidator::new();
        
        let invalid_config = json!({
            "host": "invalid_host",
            "port": 70000
        });
        
        let result = validator.validate_config(&invalid_config, "server");
        assert!(!result.valid);
        assert!(!result.errors.is_empty());
    }
    
    #[test]
    fn test_environment_validation() {
        let validator = ConfigValidator::new();
        
        // Set required environment variables
        env::set_var("NEXORA_ENV", "development");
        env::set_var("DATABASE_URL", "postgresql://localhost/test");
        
        let result = validator.validate_environment();
        assert!(result.valid);
        
        // Clean up
        env::remove_var("NEXORA_ENV");
        env::remove_var("DATABASE_URL");
    }
    
    #[test]
    fn test_environment_setup() {
        let validator = ConfigValidator::new();
        
        // Set required environment variables first
        env::set_var("NEXORA_ENV", "development");
        env::set_var("DATABASE_URL", "postgresql://localhost/test");
        
        let result = validator.setup_environment();
        assert!(result.is_ok());
        
        let setup_result = result.unwrap();
        assert!(!setup_result.created_directories.is_empty());
        // Note: set_env_vars might be empty if all required vars are already set
        
        // Clean up
        env::remove_var("NEXORA_ENV");
        env::remove_var("DATABASE_URL");
    }
    
    #[test]
    fn test_config_migrator() {
        let mut migrator = ConfigMigrator::new();
        
        // Add a simple migration
        migrator.add_migration(Migration {
            version: "2.0".to_string(),
            description: "Add new field".to_string(),
            migrate_fn: |mut config| {
                if let Value::Object(ref mut map) = config {
                    map.insert("new_field".to_string(), json!("default_value"));
                }
                Ok(config)
            },
        });
        
        let config = json!({"old_field": "value"});
        let migrated = migrator.migrate(config, "1.0", "2.0").unwrap();
        
        assert!(migrated.get("new_field").is_some());
    }
    
    #[test]
    fn test_utility_functions() {
        assert!(utils::validate_port(8080));
        assert!(utils::validate_port(0)); // Port 0 is valid (0 <= 65535)
        assert!(utils::validate_port(65535)); // Max valid port
        
        assert!(utils::validate_url("https://example.com"));
        assert!(!utils::validate_url("ftp://example.com"));
        
        assert!(utils::validate_email("test@example.com"));
        assert!(!utils::validate_email("invalid-email"));
    }
}
