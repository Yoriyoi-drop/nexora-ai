//! Secure Credential Management
//! 
//! This module provides secure credential storage and retrieval
//! without hardcoding sensitive information

use std::env;
use std::fs;
use std::path::Path;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use secrecy::{Secret, ExposeSecret};

/// Secure database credentials
#[derive(Debug, Clone, Serialize)]
pub struct DatabaseCredentials {
    pub host: String,
    pub port: u16,
    pub database: String,
    pub username: String,
    #[serde(skip)]
    pub password: Secret<String>,
    pub ssl_mode: SslMode,
}

impl<'de> Deserialize<'de> for DatabaseCredentials {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, Visitor, MapAccess};
        use std::fmt;
        
        struct DatabaseCredentialsVisitor;
        
        impl<'de> Visitor<'de> for DatabaseCredentialsVisitor {
            type Value = DatabaseCredentials;
            
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct DatabaseCredentials")
            }
            
            fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut host = None;
                let mut port = None;
                let mut database = None;
                let mut username = None;
                let mut password = None;
                let mut ssl_mode = None;
                
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "host" => {
                            if host.is_some() {
                                return Err(de::Error::duplicate_field("host"));
                            }
                            host = Some(map.next_value()?);
                        }
                        "port" => {
                            if port.is_some() {
                                return Err(de::Error::duplicate_field("port"));
                            }
                            port = Some(map.next_value()?);
                        }
                        "database" => {
                            if database.is_some() {
                                return Err(de::Error::duplicate_field("database"));
                            }
                            database = Some(map.next_value()?);
                        }
                        "username" => {
                            if username.is_some() {
                                return Err(de::Error::duplicate_field("username"));
                            }
                            username = Some(map.next_value()?);
                        }
                        "password" => {
                            if password.is_some() {
                                return Err(de::Error::duplicate_field("password"));
                            }
                            password = Some(map.next_value::<String>()?);
                        }
                        "ssl_mode" => {
                            if ssl_mode.is_some() {
                                return Err(de::Error::duplicate_field("ssl_mode"));
                            }
                            ssl_mode = Some(map.next_value()?);
                        }
                        _ => {
                            // Ignore unknown fields
                            let _: serde::de::IgnoredAny = map.next_value()?;
                        }
                    }
                }
                
                let host = host.ok_or_else(|| de::Error::missing_field("host"))?;
                let port = port.ok_or_else(|| de::Error::missing_field("port"))?;
                let database = database.ok_or_else(|| de::Error::missing_field("database"))?;
                let username = username.ok_or_else(|| de::Error::missing_field("username"))?;
                let ssl_mode = ssl_mode.unwrap_or(SslMode::Prefer);
                
                Ok(DatabaseCredentials {
                    host,
                    port,
                    database,
                    username,
                    password: password.map(Secret::new).unwrap_or_else(|| Secret::new(String::new())),
                    ssl_mode,
                })
            }
        }
        
        deserializer.deserialize_map(DatabaseCredentialsVisitor)
    }
}

/// SSL Mode for database connections
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SslMode {
    Disable,
    Allow,
    Prefer,
    Require,
}

/// Credential manager for secure storage
pub struct CredentialManager {
    config_path: Option<String>,
}

impl CredentialManager {
    /// Create new credential manager
    pub fn new() -> Self {
        Self {
            config_path: None,
        }
    }
    
    /// Create credential manager with custom config path
    pub fn with_config_path<P: AsRef<Path>>(path: P) -> Self {
        Self {
            config_path: Some(path.as_ref().to_string_lossy().into_owned()),
        }
    }
    
    /// Load database credentials from environment variables or config file
    pub fn load_database_credentials(&self) -> Result<DatabaseCredentials> {
        // Try environment variables first (most secure)
        if let Ok(creds) = self.load_from_env() {
            return Ok(creds);
        }
        
        // Try config file
        if let Some(config_path) = &self.config_path {
            if let Ok(creds) = self.load_from_file(config_path) {
                return Ok(creds);
            }
        }
        
        // Try default config locations
        for path in self.get_default_config_paths() {
            if let Ok(creds) = self.load_from_file(&path) {
                return Ok(creds);
            }
        }
        
        Err(anyhow!("No database credentials found in environment variables or config files"))
    }
    
    /// Load credentials from environment variables
    fn load_from_env(&self) -> Result<DatabaseCredentials> {
        let host = env::var("NEXORA_DB_HOST")
            .or_else(|_| env::var("DB_HOST"))
            .unwrap_or_else(|_| "localhost".to_string());
            
        let port = env::var("NEXORA_DB_PORT")
            .or_else(|_| env::var("DB_PORT"))
            .unwrap_or_else(|_| "5432".to_string())
            .parse::<u16>()
            .map_err(|_| anyhow!("Invalid DB port in environment"))?;
            
        let database = env::var("NEXORA_DB_NAME")
            .or_else(|_| env::var("DB_NAME"))
            .map_err(|_| anyhow!("Database name not found in environment"))?;
            
        let username = env::var("NEXORA_DB_USER")
            .or_else(|_| env::var("DB_USER"))
            .map_err(|_| anyhow!("Database username not found in environment"))?;
            
        let password = env::var("NEXORA_DB_PASSWORD")
            .or_else(|_| env::var("DB_PASSWORD"))
            .map_err(|_| anyhow!("Database password not found in environment"))?;
            
        let ssl_mode_str = env::var("NEXORA_DB_SSL_MODE")
            .or_else(|_| env::var("DB_SSL_MODE"))
            .unwrap_or_else(|_| "prefer".to_string());
            
        let ssl_mode = match ssl_mode_str.to_lowercase().as_str() {
            "disable" => SslMode::Disable,
            "allow" => SslMode::Allow,
            "prefer" => SslMode::Prefer,
            "require" => SslMode::Require,
            _ => SslMode::Prefer,
        };
        
        Ok(DatabaseCredentials {
            host,
            port,
            database,
            username,
            password: Secret::new(password),
            ssl_mode,
        })
    }
    
    /// Load credentials from config file
    fn load_from_file(&self, path: &str) -> Result<DatabaseCredentials> {
        if !Path::new(path).exists() {
            return Err(anyhow!("Config file not found: {}", path));
        }
        
        let content = fs::read_to_string(path)
            .map_err(|e| anyhow!("Failed to read config file: {}", e))?;
            
        // Try JSON format first
        if path.ends_with(".json") {
            let creds: DatabaseConfigFile = serde_json::from_str(&content)
                .map_err(|e| anyhow!("Failed to parse JSON config: {}", e))?;
            return Ok(creds.into());
        }
        
        // Try YAML format
        if path.ends_with(".yaml") || path.ends_with(".yml") {
            let creds: DatabaseConfigFile = serde_yaml::from_str(&content)
                .map_err(|e| anyhow!("Failed to parse YAML config: {}", e))?;
            return Ok(creds.into());
        }
        
        // Try TOML format
        if path.ends_with(".toml") {
            let creds: DatabaseConfigFile = toml::from_str(&content)
                .map_err(|e| anyhow!("Failed to parse TOML config: {}", e))?;
            return Ok(creds.into());
        }
        
        Err(anyhow!("Unsupported config file format: {}", path))
    }
    
    /// Get default config file paths
    fn get_default_config_paths(&self) -> Vec<String> {
        let home_dir = env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        let config_dir = env::var("XDG_CONFIG_HOME")
            .unwrap_or_else(|_| format!("{}/.config", home_dir));
        
        vec![
            format!("{}/nexora/database.json", config_dir),
            format!("{}/nexora/database.yaml", config_dir),
            format!("{}/nexora/database.toml", config_dir),
            format!("{}/.nexora/database.json", home_dir),
            "/etc/nexora/database.json".to_string(),
            "./database.json".to_string(),
        ]
    }
    
    /// Create example config file
    pub fn create_example_config<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let example_config = DatabaseConfigFile {
            database: DatabaseCredentials {
                host: "localhost".to_string(),
                port: 5432,
                database: "nexora".to_string(),
                username: "nexora_user".to_string(),
                password: Secret::new("your_password_here".to_string()),
                ssl_mode: SslMode::Prefer,
            },
        };
        
        let path_str = path.as_ref().to_string_lossy();
        let content = if path_str.ends_with(".json") {
            serde_json::to_string_pretty(&example_config)?
        } else if path_str.ends_with(".yaml") || path_str.ends_with(".yml") {
            serde_yaml::to_string(&example_config)?
        } else if path_str.ends_with(".toml") {
            toml::to_string_pretty(&example_config)?
        } else {
            return Err(anyhow!("Unsupported file format. Use .json, .yaml, or .toml"));
        };
        
        // Create parent directory if needed
        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent)?;
        }
        
        fs::write(&path, content)?;
        println!("Example config created at: {}", path.as_ref().display());
        println!("Please edit the file and set your actual credentials.");
        
        Ok(())
    }
}

/// Configuration file structure
#[derive(Debug, Serialize, Deserialize)]
struct DatabaseConfigFile {
    database: DatabaseCredentials,
}

impl From<DatabaseConfigFile> for DatabaseCredentials {
    fn from(config: DatabaseConfigFile) -> Self {
        config.database
    }
}

impl DatabaseCredentials {
    /// Build connection string (without password for logging)
    pub fn build_connection_string_safe(&self) -> String {
        format!(
            "host={} port={} dbname={} user={} sslmode={}",
            self.host,
            self.port,
            self.database,
            self.username,
            match self.ssl_mode {
                SslMode::Disable => "disable",
                SslMode::Allow => "allow",
                SslMode::Prefer => "prefer",
                SslMode::Require => "require",
            }
        )
    }
    
    /// Build complete connection string (with password)
    pub fn build_connection_string(&self) -> String {
        format!(
            "{} password={}",
            self.build_connection_string_safe(),
            self.password.expose_secret()
        )
    }
    
    /// Get safe credentials for logging (without password)
    pub fn to_safe_string(&self) -> String {
        format!(
            "DatabaseCredentials {{ host: {}, port: {}, database: {}, username: {}, ssl_mode: {:?} }}",
            self.host, self.port, self.database, self.username, self.ssl_mode
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    
    #[test]
    fn test_load_from_env() {
        // Set test environment variables
        env::set_var("NEXORA_DB_HOST", "testhost");
        env::set_var("NEXORA_DB_PORT", "5432");
        env::set_var("NEXORA_DB_NAME", "testdb");
        env::set_var("NEXORA_DB_USER", "testuser");
        env::set_var("NEXORA_DB_PASSWORD", "testpass");
        env::set_var("NEXORA_DB_SSL_MODE", "require");
        
        let manager = CredentialManager::new();
        let creds = manager.load_from_env().unwrap();
        
        assert_eq!(creds.host, "testhost");
        assert_eq!(creds.port, 5432);
        assert_eq!(creds.database, "testdb");
        assert_eq!(creds.username, "testuser");
        assert_eq!(creds.password.expose_secret(), "testpass");
        assert!(matches!(creds.ssl_mode, SslMode::Require));
        
        // Clean up
        env::remove_var("NEXORA_DB_HOST");
        env::remove_var("NEXORA_DB_PORT");
        env::remove_var("NEXORA_DB_NAME");
        env::remove_var("NEXORA_DB_USER");
        env::remove_var("NEXORA_DB_PASSWORD");
        env::remove_var("NEXORA_DB_SSL_MODE");
    }
    
    #[test]
    fn test_connection_string_safe() {
        let creds = DatabaseCredentials {
            host: "localhost".to_string(),
            port: 5432,
            database: "testdb".to_string(),
            username: "user".to_string(),
            password: Secret::new("secret".to_string()),
            ssl_mode: SslMode::Prefer,
        };
        
        let safe_string = creds.build_connection_string_safe();
        assert!(!safe_string.contains("secret"));
        assert!(safe_string.contains("host=localhost"));
    }
}
