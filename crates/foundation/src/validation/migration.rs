//! Configuration Migration Utilities
//! 
//! Configuration migration and versioning utilities.

use std::collections::HashMap;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use serde_json::Value;
use regex::Regex;
use tracing::info;

/// Configuration migration utilities
pub struct ConfigMigrator {
    migrations: Vec<Migration>,
}

/// Migration definition
#[derive(Debug, Clone)]
struct Migration {
    version: String,
    description: String,
    migrate_fn: fn(Value) -> Result<Value>,
}

impl ConfigMigrator {
    /// Create new migrator
    pub fn new() -> Self {
        Self {
            migrations: vec![
                Migration {
                    version: "1.0.0".to_string(),
                    description: "Initial version".to_string(),
                    migrate_fn: Self::migrate_1_0_0,
                },
                Migration {
                    version: "1.1.0".to_string(),
                    description: "Add database SSL options".to_string(),
                    migrate_fn: Self::migrate_1_1_0,
                },
                Migration {
                    version: "1.2.0".to_string(),
                    description: "Add server timeout configuration".to_string(),
                    migrate_fn: Self::migrate_1_2_0,
                },
                Migration {
                    version: "2.0.0".to_string(),
                    description: "Restructure configuration format".to_string(),
                    migrate_fn: Self::migrate_2_0_0,
                },
            ],
        }
    }
    
    /// Migrate configuration to latest version
    pub fn migrate_to_latest(&self, mut config: Value) -> Result<Value> {
        let current_version = self.get_config_version(&config)?;
        
        for migration in &self.migrations {
            if self.should_migrate(&current_version, &migration.version)? {
                info!("Migrating config from version {} to {}", current_version, migration.version);
                config = (migration.migrate_fn)(config)?;
                self.update_config_version(&mut config, &migration.version)?;
            }
        }
        
        Ok(config)
    }
    
    /// Migrate configuration to specific version
    pub fn migrate_to_version(&self, mut config: Value, target_version: &str) -> Result<Value> {
        let current_version = self.get_config_version(&config)?;
        
        for migration in &self.migrations {
            if self.should_migrate(&current_version, &migration.version)? &&
               self.should_migrate(&migration.version, target_version)? {
                info!("Migrating config from version {} to {}", current_version, migration.version);
                config = (migration.migrate_fn)(config)?;
                self.update_config_version(&mut config, &migration.version)?;
            }
        }
        
        Ok(config)
    }
    
    /// Get configuration version
    fn get_config_version(&self, config: &Value) -> Result<String> {
        match config.get("version") {
            Some(Value::String(version)) => Ok(version.clone()),
            _ => Ok("1.0.0".to_string()), // Default version
        }
    }
    
    /// Check if migration should be applied
    fn should_migrate(&self, current: &str, target: &str) -> Result<bool> {
        let current_parts: Vec<u32> = current.split('.').filter_map(|s| s.parse().ok()).collect();
        let target_parts: Vec<u32> = target.split('.').filter_map(|s| s.parse().ok()).collect();
        
        if current_parts.len() != 3 || target_parts.len() != 3 {
            return Ok(false);
        }
        
        // Compare version numbers
        for i in 0..3 {
            if current_parts[i] < target_parts[i] {
                return Ok(true);
            } else if current_parts[i] > target_parts[i] {
                return Ok(false);
            }
        }
        
        Ok(false) // Same version
    }
    
    /// Update configuration version
    fn update_config_version(&self, config: &mut Value, version: &str) -> Result<()> {
        if let Some(config_obj) = config.as_object_mut() {
            config_obj.insert("version".to_string(), Value::String(version.to_string()));
        }
        Ok(())
    }
    
    /// Validate migration compatibility
    pub fn validate_migration_path(&self, from_version: &str, to_version: &str) -> Result<bool> {
        // Check if migration path exists
        let mut current_version = from_version.to_string();
        let mut can_migrate = true;
        
        for migration in &self.migrations {
            if self.should_migrate(&current_version, &migration.version)? {
                if self.should_migrate(&migration.version, to_version)? {
                    current_version = migration.version.clone();
                } else {
                    // Migration path stops here
                    break;
                }
            }
        }
        
        // Check if we reached the target version
        can_migrate = self.should_migrate(&current_version, to_version)?;
        
        Ok(can_migrate)
    }
    
    /// Get available migration versions
    pub fn get_migration_versions(&self) -> Vec<String> {
        self.migrations.iter().map(|m| m.version.clone()).collect()
    }
    
    /// Get migration description
    pub fn get_migration_description(&self, version: &str) -> Option<String> {
        self.migrations.iter()
            .find(|m| m.version == version)
            .map(|m| m.description.clone())
    }
    
    // Migration functions
    fn migrate_1_0_0(config: Value) -> Result<Value> {
        // No migration needed for initial version
        Ok(config)
    }
    
    fn migrate_1_1_0(mut config: Value) -> Result<Value> {
        // Add database SSL options
        if let Some(database) = config.get_mut("database") {
            if let Some(database_obj) = database.as_object_mut() {
                // Add SSL options with defaults
                database_obj.insert("ssl_mode".to_string(), Value::String("prefer".to_string()));
                database_obj.insert("ssl_cert".to_string(), Value::Null);
                database_obj.insert("ssl_key".to_string(), Value::Null);
                database_obj.insert("ssl_ca".to_string(), Value::Null);
            }
        }
        
        Ok(config)
    }
    
    fn migrate_1_2_0(mut config: Value) -> Result<Value> {
        // Add server timeout configuration
        if let Some(server) = config.get_mut("server") {
            if let Some(server_obj) = server.as_object_mut() {
                // Add timeout options with defaults
                server_obj.insert("timeout_seconds".to_string(), Value::Number(serde_json::Number::from(30)));
                server_obj.insert("read_timeout".to_string(), Value::Number(serde_json::Number::from(10)));
                server_obj.insert("write_timeout".to_string(), Value::Number(serde_json::Number::from(10)));
            }
        }
        
        Ok(config)
    }
    
    fn migrate_2_0_0(mut config: Value) -> Result<Value> {
        // Restructure configuration format
        let mut new_config = serde_json::Map::new();
        
        // Migrate database configuration
        if let Some(database) = config.get("database") {
            let mut new_database = serde_json::Map::new();
            new_database.insert("connection".to_string(), database.clone());
            new_database.insert("pool_size".to_string(), Value::Number(serde_json::Number::from(10)));
            new_database.insert("max_connections".to_string(), Value::Number(serde_json::Number::from(100)));
            new_config.insert("database".to_string(), Value::Object(new_database));
        }
        
        // Migrate server configuration
        if let Some(server) = config.get("server") {
            let mut new_server = serde_json::Map::new();
            new_server.insert("http".to_string(), server.clone());
            new_server.insert("grpc".to_string(), Value::Null);
            new_config.insert("server".to_string(), Value::Object(new_server));
        }
        
        // Migrate logging configuration
        if let Some(logging) = config.get("logging") {
            new_config.insert("logging".to_string(), logging.clone());
        } else {
            // Add default logging configuration
            let mut default_logging = serde_json::Map::new();
            default_logging.insert("level".to_string(), Value::String("info".to_string()));
            default_logging.insert("format".to_string(), Value::String("json".to_string()));
            new_config.insert("logging".to_string(), Value::Object(default_logging));
        }
        
        // Add new version
        new_config.insert("version".to_string(), Value::String("2.0.0".to_string()));
        
        Ok(Value::Object(new_config))
    }
    
    /// Generate migration plan
    pub fn generate_migration_plan(&self, from_version: &str, to_version: &str) -> Result<MigrationPlan> {
        let mut steps = Vec::new();
        let mut current_version = from_version.to_string();
        
        for migration in &self.migrations {
            if self.should_migrate(&current_version, &migration.version)? &&
               self.should_migrate(&migration.version, to_version)? {
                steps.push(MigrationStep {
                    from_version: current_version.clone(),
                    to_version: migration.version.clone(),
                    description: migration.description.clone(),
                });
                current_version = migration.version.clone();
            }
        }
        
        Ok(MigrationPlan {
            from_version: from_version.to_string(),
            to_version: to_version.to_string(),
            steps,
            total_steps: steps.len(),
        })
    }
}

impl Default for ConfigMigrator {
    fn default() -> Self {
        Self::new()
    }
}

/// Migration plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPlan {
    pub from_version: String,
    pub to_version: String,
    pub steps: Vec<MigrationStep>,
    pub total_steps: usize,
}

/// Migration step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStep {
    pub from_version: String,
    pub to_version: String,
    pub description: String,
}
