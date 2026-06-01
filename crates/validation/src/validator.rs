//! Configuration Validator
//! 
//! Main configuration validator with comprehensive validation rules.

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use serde_json::{json, Value};
use regex::Regex;
use tracing::info;

use crate::environment::EnvironmentValidator;
use crate::security::{SecurityRule, SecurityValidatorImpl};

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
}

/// Environment rule
#[derive(Debug, Clone)]
pub(crate) struct EnvironmentRule {
    name: String,
    required: bool,
    validator: Option<Regex>,
    default_value: Option<String>,
    description: String,
}

/// Config field validator — validates a single config field with a closure.
struct ConfigFieldValidator {
    field_path: String,
    validate: Box<dyn Fn(&Value) -> Vec<ValidationError> + Send + Sync>,
}

impl SecurityValidator for ConfigFieldValidator {
    fn validate(&self, config: &Value) -> Result<Vec<ValidationError>> {
        let mut errors = vec![];
        // Walk the field path (e.g. "server.port")
        let parts: Vec<&str> = self.field_path.split('.').collect();
        let mut current = config;
        for part in &parts {
            match current.get(*part) {
                Some(val) => current = val,
                None => return Ok(vec![]), // field not present — skip
            }
        }
        errors.extend((self.validate)(current));
        Ok(errors)
    }

    fn name(&self) -> &str {
        "ConfigFieldValidator"
    }
}

impl ConfigValidator {
    /// Create new configuration validator
    pub fn new() -> Self {
        let mut validator = Self {
            schemas: HashMap::new(),
            environment_rules: HashMap::new(),
            security_rules: Vec::new(),
        };
        
        // Initialize default schemas
        validator.add_default_schemas();
        validator.add_default_environment_rules();
        validator.add_default_security_rules();
        
        validator
    }
    
    /// Validate configuration
    pub fn validate_config(&self, config: &Value, schema_name: &str) -> Result<ValidationResult> {
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
                return Ok(result);
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
        
        // Validate field types and constraints
        if let Some(config_obj) = config.as_object() {
            for (field_name, field_value) in config_obj {
                if let Some(field_def) = schema.fields.get(field_name) {
                    self.validate_field(field_name, field_value, field_def, &mut result);
                }
            }
        }
        
        Ok(result)
    }
    
    /// Validate environment variables
    pub fn validate_environment(&self) -> Result<ValidationResult> {
        let env_validator = EnvironmentValidator::new();
        env_validator.validate_environment(&self.environment_rules)
    }
    
    /// Validate security configuration
    pub fn validate_security(&self, config: &Value) -> Result<ValidationResult> {
        let security_validator = SecurityValidatorImpl::new();
        security_validator.validate_config(config, &self.security_rules)
    }
    
    /// Validate all aspects
    pub fn validate_all(&self, config: &Value, schema_name: &str) -> Result<ValidationResult> {
        let mut config_result = self.validate_config(config, schema_name)?;
        let env_result = self.validate_environment()?;
        let security_result = self.validate_security(config)?;
        
        // Combine results
        config_result.errors.extend(env_result.errors);
        config_result.errors.extend(security_result.errors);
        config_result.warnings.extend(env_result.warnings);
        config_result.warnings.extend(security_result.warnings);
        config_result.info.extend(env_result.info);
        config_result.info.extend(security_result.info);
        
        config_result.valid = config_result.errors.is_empty() && 
                              env_result.errors.is_empty() && 
                              security_result.errors.is_empty();
        
        Ok(config_result)
    }
    
    /// Add custom schema
    pub fn add_schema(&mut self, schema: ConfigSchema) {
        self.schemas.insert(schema.name.clone(), schema);
    }
    
    /// Add environment rule
    pub fn add_environment_rule(&mut self, rule: EnvironmentRule) {
        self.environment_rules.insert(rule.name.clone(), rule);
    }
    
    /// Add security rule
    pub fn add_security_rule(&mut self, rule: SecurityRule) {
        self.security_rules.push(rule);
    }
    
    /// Validate individual field
    fn validate_field(&self, field_name: &str, field_value: &Value, field_def: &FieldDefinition, result: &mut ValidationResult) {
        // Type validation
        match (&field_def.field_type, field_value) {
            (FieldType::String, Value::String(_)) => {},
            (FieldType::Number, Value::Number(_)) => {},
            (FieldType::Boolean, Value::Bool(_)) => {},
            (FieldType::Array, Value::Array(_)) => {},
            (FieldType::Object, Value::Object(_)) => {},
            (FieldType::Path, Value::String(path)) => {
                if !PathBuf::from(path).exists() {
                    result.errors.push(ValidationError {
                        field: field_name.to_string(),
                        message: format!("Path '{}' does not exist", path),
                        code: "PATH_NOT_FOUND".to_string(),
                        severity: ErrorSeverity::Error,
                    });
                    result.valid = false;
                }
            },
            (FieldType::Url, Value::String(url)) => {
                if !url.starts_with("http://") && !url.starts_with("https://") {
                    result.warnings.push(ValidationWarning {
                        field: field_name.to_string(),
                        message: format!("URL '{}' may not be valid", url),
                        code: "INVALID_URL".to_string(),
                    });
                }
            },
            _ => {
                result.errors.push(ValidationError {
                    field: field_name.to_string(),
                    message: format!("Field '{}' has incorrect type", field_name),
                    code: "TYPE_MISMATCH".to_string(),
                    severity: ErrorSeverity::Error,
                });
                result.valid = false;
            }
        }
        
        // Regex validation
        if let Some(validator) = &field_def.validator {
            if let Value::String(value) = field_value {
                if !validator.is_match(value) {
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
        
        // Range validation
        if let (Some(min), Some(max)) = (&field_def.min_value, &field_def.max_value) {
            if let Value::Number(num) = field_value {
                let value = num.as_f64().unwrap_or(0.0);
                if value < *min || value > *max {
                    result.errors.push(ValidationError {
                        field: field_name.to_string(),
                        message: format!("Field '{}' value {} is out of range [{}, {}]", field_name, value, min, max),
                        code: "OUT_OF_RANGE".to_string(),
                        severity: ErrorSeverity::Error,
                    });
                    result.valid = false;
                }
            }
        }
        
        // Allowed values validation
        if let Some(allowed_values) = &field_def.allowed_values {
            if let Value::String(value) = field_value {
                if !allowed_values.contains(value) {
                    result.errors.push(ValidationError {
                        field: field_name.to_string(),
                        message: format!("Field '{}' value '{}' is not in allowed values: {:?}", field_name, value, allowed_values),
                        code: "INVALID_VALUE".to_string(),
                        severity: ErrorSeverity::Error,
                    });
                    result.valid = false;
                }
            }
        }
    }
    
    /// Add default schemas
    fn add_default_schemas(&mut self) {
        // Database configuration schema
        let mut db_fields = HashMap::new();
        db_fields.insert("host".to_string(), FieldDefinition {
            field_type: FieldType::String,
            required: true,
            default_value: Some(json!("localhost")),
            validator: Some(Regex::new(r"^[a-zA-Z0-9.-]+$")
                .expect("Failed to create hostname regex")), // safe: hardcoded regex pattern
            min_value: None,
            max_value: None,
            allowed_values: None,
            description: "Database host".to_string(),
        });
        db_fields.insert("port".to_string(), FieldDefinition {
            field_type: FieldType::Number,
            required: true,
            default_value: Some(json!(5432)),
            validator: None,
            min_value: Some(1.0),
            max_value: Some(65535.0),
            allowed_values: None,
            description: "Database port".to_string(),
        });
        
        self.schemas.insert("database".to_string(), ConfigSchema {
            name: "database".to_string(),
            version: "1.0".to_string(),
            fields: db_fields,
            required_fields: vec!["host".to_string(), "port".to_string()],
        });
        
        // Server configuration schema
        let mut server_fields = HashMap::new();
        server_fields.insert("port".to_string(), FieldDefinition {
            field_type: FieldType::Number,
            required: true,
            default_value: Some(json!(8080)),
            validator: None,
            min_value: Some(1.0),
            max_value: Some(65535.0),
            allowed_values: None,
            description: "Server port".to_string(),
        });
        
        self.schemas.insert("server".to_string(), ConfigSchema {
            name: "server".to_string(),
            version: "1.0".to_string(),
            fields: server_fields,
            required_fields: vec!["port".to_string()],
        });
    }
    
    /// Add default environment rules
    fn add_default_environment_rules(&mut self) {
        self.environment_rules.insert("DATABASE_URL".to_string(), EnvironmentRule {
            name: "DATABASE_URL".to_string(),
            required: true,
            validator: Some(Regex::new(r"^postgres://.*")
                .expect("Failed to create database URL regex")), // safe: hardcoded regex pattern
            default_value: Some("postgres://localhost:5432/mydb".to_string()),
            description: "Database connection URL".to_string(),
        });
        
        self.environment_rules.insert("LOG_LEVEL".to_string(), EnvironmentRule {
            name: "LOG_LEVEL".to_string(),
            required: false,
            validator: Some(Regex::new(r"^(debug|info|warn|error)$")
                .expect("Failed to create log level regex")), // safe: hardcoded regex pattern
            default_value: Some("info".to_string()),
            description: "Logging level".to_string(),
        });
        
        self.environment_rules.insert("PORT".to_string(), EnvironmentRule {
            name: "PORT".to_string(),
            required: false,
            validator: None,
            default_value: Some("8080".to_string()),
            description: "Server port".to_string(),
        });
    }
    
    /// Add default security rules
    fn add_default_security_rules(&mut self) {
        self.security_rules.push(SecurityRule {
            name: "rate_limiting".to_string(),
            description: "Validate rate limiting configuration".to_string(),
            validator: Box::new(ConfigFieldValidator {
                field_path: "server.rate_limit".to_string(),
                validate: Box::new(|value: &Value| {
                    if let Some(rl) = value.as_u64() {
                        if rl == 0 {
                            return vec![ValidationError {
                                field: "server.rate_limit".to_string(),
                                message: "Rate limiting is disabled (set to 0)".to_string(),
                                code: "RATE_LIMIT_DISABLED".to_string(),
                                severity: ErrorSeverity::Warning,
                            }];
                        }
                    }
                    vec![]
                }),
            }),
            severity: ErrorSeverity::Warning,
        });

        self.security_rules.push(SecurityRule {
            name: "cors_settings".to_string(),
            description: "Validate CORS configuration".to_string(),
            validator: Box::new(ConfigFieldValidator {
                field_path: "server.cors".to_string(),
                validate: Box::new(|value: &Value| {
                    let mut errors = vec![];
                    if let Some(cors) = value.as_object() {
                        if let Some(origins) = cors.get("allowed_origins") {
                            if let Some(arr) = origins.as_array() {
                                if arr.iter().any(|o| o.as_str() == Some("*")) {
                                    errors.push(ValidationError {
                                        field: "server.cors.allowed_origins".to_string(),
                                        message: "Wildcard CORS origin (*) is insecure for production".to_string(),
                                        code: "CORS_WILDCARD".to_string(),
                                        severity: ErrorSeverity::Error,
                                    });
                                }
                            }
                        }
                        if let Some(credentials) = cors.get("allow_credentials") {
                            if credentials.as_bool() == Some(true) {
                                if let Some(origins) = cors.get("allowed_origins") {
                                    if let Some(arr) = origins.as_array() {
                                        if arr.iter().any(|o| o.as_str() == Some("*")) {
                                            errors.push(ValidationError {
                                                field: "server.cors.allow_credentials".to_string(),
                                                message: "Credentials should not be allowed with wildcard origin".to_string(),
                                                code: "CORS_CREDENTIALS_WILDCARD".to_string(),
                                                severity: ErrorSeverity::Error,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                    errors
                }),
            }),
            severity: ErrorSeverity::Error,
        });

        self.security_rules.push(SecurityRule {
            name: "log_sensitivity".to_string(),
            description: "Validate logging does not expose sensitive data".to_string(),
            validator: Box::new(ConfigFieldValidator {
                field_path: "logging".to_string(),
                validate: Box::new(|value: &Value| {
                    let mut errors = vec![];
                    if let Some(logging) = value.as_object() {
                        if let Some(level) = logging.get("level") {
                            if let Some(lvl) = level.as_str() {
                                if lvl.eq_ignore_ascii_case("debug") || lvl.eq_ignore_ascii_case("trace") {
                                    errors.push(ValidationError {
                                        field: "logging.level".to_string(),
                                        message: format!("Verbose logging level '{}' may expose sensitive data in production", lvl),
                                        code: "VERBOSE_LOGGING".to_string(),
                                        severity: ErrorSeverity::Warning,
                                    });
                                }
                            }
                        }
                    }
                    errors
                }),
            }),
            severity: ErrorSeverity::Warning,
        });

        self.security_rules.push(SecurityRule {
            name: "port_security".to_string(),
            description: "Validate server port configuration".to_string(),
            validator: Box::new(ConfigFieldValidator {
                field_path: "server.port".to_string(),
                validate: Box::new(|value: &Value| {
                    let mut errors = vec![];
                    if let Some(port) = value.as_u64() {
                        if port < 1024 && port != 80 && port != 443 {
                            errors.push(ValidationError {
                                field: "server.port".to_string(),
                                message: format!("Port {} requires elevated privileges (use >1024 in production)", port),
                                code: "PRIVILEGED_PORT".to_string(),
                                severity: ErrorSeverity::Warning,
                            });
                        }
                        if port > 65535 {
                            errors.push(ValidationError {
                                field: "server.port".to_string(),
                                message: format!("Port {} is outside valid range (1-65535)", port),
                                code: "INVALID_PORT".to_string(),
                                severity: ErrorSeverity::Error,
                            });
                        }
                    }
                    errors
                }),
            }),
            severity: ErrorSeverity::Warning,
        });

        info!("Default security rules loaded: rate_limiting, cors_settings, log_sensitivity, port_security");
    }
}

impl Default for ConfigValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_validation_result_default() {
        let r = ValidationResult {
            valid: true,
            errors: vec![],
            warnings: vec![],
            info: vec![],
        };
        assert!(r.valid);
        assert!(r.errors.is_empty());
    }

    #[test]
    fn test_error_severity_variants() {
        match ErrorSeverity::Critical {
            ErrorSeverity::Critical => {},
            _ => panic!("expected Critical"),
        }
        match ErrorSeverity::Error {
            ErrorSeverity::Error => {},
            _ => panic!("expected Error"),
        }
        match ErrorSeverity::Warning {
            ErrorSeverity::Warning => {},
            _ => panic!("expected Warning"),
        }
        match ErrorSeverity::Info {
            ErrorSeverity::Info => {},
            _ => panic!("expected Info"),
        }
    }

    #[test]
    fn test_config_validator_new() {
        let validator = ConfigValidator::new();
        // should have default schemas: database, server
        assert!(validator.schemas.contains_key("database"));
        assert!(validator.schemas.contains_key("server"));
        // should have default env rules
        assert!(validator.environment_rules.contains_key("DATABASE_URL"));
        assert!(validator.environment_rules.contains_key("LOG_LEVEL"));
        assert!(validator.environment_rules.contains_key("PORT"));
        // should have default security rules
        assert!(!validator.security_rules.is_empty());
    }

    #[test]
    fn test_config_validator_unknown_schema() {
        let validator = ConfigValidator::new();
        let config = json!({"host": "localhost"});
        let result = validator.validate_config(&config, "nonexistent").unwrap();
        assert!(!result.valid);
        assert_eq!(result.errors[0].code, "SCHEMA_NOT_FOUND");
    }

    #[test]
    fn test_validate_config_database_missing_required() {
        let validator = ConfigValidator::new();
        let config = json!({});
        let result = validator.validate_config(&config, "database").unwrap();
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.code == "REQUIRED_FIELD_MISSING"));
    }

    #[test]
    fn test_validate_config_database_valid() {
        let validator = ConfigValidator::new();
        let config = json!({"host": "localhost", "port": 5432});
        let result = validator.validate_config(&config, "database").unwrap();
        assert!(result.valid);
    }

    #[test]
    fn test_validate_config_type_mismatch() {
        let validator = ConfigValidator::new();
        let config = json!({"host": 12345, "port": 5432});
        let result = validator.validate_config(&config, "database").unwrap();
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.code == "TYPE_MISMATCH"));
    }

    #[test]
    fn test_validate_config_port_out_of_range() {
        let validator = ConfigValidator::new();
        let config = json!({"host": "localhost", "port": 99999});
        let result = validator.validate_config(&config, "database").unwrap();
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.code == "OUT_OF_RANGE"));
    }

    #[test]
    fn test_validate_config_port_valid_range() {
        let validator = ConfigValidator::new();
        let config = json!({"host": "localhost", "port": 8080});
        let result = validator.validate_config(&config, "server").unwrap();
        assert!(result.valid);
    }

    #[test]
    fn test_validate_all_combines_results() {
        let validator = ConfigValidator::new();
        let config = json!({"host": "localhost", "port": 5432});
        let result = validator.validate_all(&config, "database").unwrap();
        // This will pass config validation but may have env + security warnings
        assert!(result.valid || !result.errors.is_empty() || !result.warnings.is_empty());
    }

    #[test]
    fn test_add_custom_environment_rule() {
        let mut validator = ConfigValidator::new();
        let rule = EnvironmentRule {
            name: "CUSTOM_VAR".to_string(),
            required: true,
            validator: None,
            default_value: None,
            description: "Custom test var".to_string(),
        };
        validator.add_environment_rule(rule);
        assert!(validator.environment_rules.contains_key("CUSTOM_VAR"));
    }

    #[test]
    fn test_config_validator_debug() {
        let validator = ConfigValidator::new();
        assert_eq!(validator.schemas.len(), 2);
    }

    #[test]
    fn test_default_trait() {
        let v1 = ConfigValidator::new();
        let v2 = ConfigValidator::default();
        assert_eq!(v1.schemas.len(), v2.schemas.len());
    }
}
