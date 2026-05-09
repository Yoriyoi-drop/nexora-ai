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

use super::{environment::EnvironmentValidator, security::SecurityValidator};

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
struct EnvironmentRule {
    name: String,
    required: bool,
    validator: Option<Regex>,
    default_value: Option<String>,
    description: String,
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
        let security_validator = SecurityValidator::new();
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
                .map_err(|e| anyhow::anyhow!("Failed to create hostname regex: {}", e))?),
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
                .map_err(|e| anyhow::anyhow!("Failed to create database URL regex: {}", e))?),
            default_value: Some("postgres://localhost:5432/mydb".to_string()),
            description: "Database connection URL".to_string(),
        });
        
        self.environment_rules.insert("LOG_LEVEL".to_string(), EnvironmentRule {
            name: "LOG_LEVEL".to_string(),
            required: false,
            validator: Some(Regex::new(r"^(debug|info|warn|error)$")
                .map_err(|e| anyhow::anyhow!("Failed to create log level regex: {}", e))?),
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
        // This will be implemented in the security module
        info!("Default security rules loaded");
    }
}

impl Default for ConfigValidator {
    fn default() -> Self {
        Self::new()
    }
}
