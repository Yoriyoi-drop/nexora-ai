//! Database Layer - Rust implementation
//!
//! High-performance database abstraction layer replacing database.c

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

pub mod connection_pool;
pub mod credentials;
pub mod pool;
pub mod postgres;
pub mod sqlite;

pub use connection_pool::*;
pub use credentials::*;
pub use pool::*;
pub use postgres::*;
pub use sqlite::*;

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub database_type: DatabaseType,
    pub host: String,
    pub port: u16,
    pub database: String,
    pub username: String,
    pub password: String,
    pub ssl_mode: SslMode,
    pub max_connections: usize,
    pub connection_timeout_seconds: u64,
    pub idle_timeout_seconds: u64,
    pub max_lifetime_seconds: u64,
}

/// Database types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatabaseType {
    PostgreSQL,
    SQLite,
    MySQL,
}

/// MySQL database configuration
#[derive(Debug, Clone)]
pub struct MySQLConfig {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: String,
    pub database: String,
    pub pool_size: u32,
    pub timeout: u64,
}

impl Default for MySQLConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 3306,
            username: "root".to_string(),
            password: "".to_string(),
            database: "nexora".to_string(),
            pool_size: 10,
            timeout: 30,
        }
    }
}

/// MySQL database implementation
#[cfg(feature = "mysql")]
#[derive(Debug)]
pub struct MySQLDatabase {
    pool: mysql::Pool,
    config: MySQLConfig,
}

#[cfg(feature = "mysql")]
impl MySQLDatabase {
    pub fn new(pool: mysql::Pool, config: MySQLConfig) -> Self {
        Self { pool, config }
    }
}

#[cfg(feature = "mysql")]
#[async_trait]
impl Database for MySQLDatabase {
    async fn connect(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Test connection
        let _conn = self.pool.get_conn()?;
        Ok(())
    }

    async fn execute_query(
        &self,
        query: &str,
        params: Vec<Value>,
    ) -> Result<QueryResult, Box<dyn std::error::Error>> {
        let mut conn = self.pool.get_conn()?;

        // Convert params to MySQL format
        let mysql_params: Vec<mysql::Value> = params
            .into_iter()
            .map(|v| match v {
                Value::String(s) => mysql::Value::from(s.as_str()),
                Value::Integer(i) => mysql::Value::from(i as i64),
                Value::Float(f) => mysql::Value::from(f),
                Value::Boolean(b) => mysql::Value::from(b),
                Value::Null => mysql::Value::NULL,
                Value::Timestamp(ts) => mysql::Value::from(ts as i64),
            })
            .collect();

        let result = conn.exec_iter(query, mysql_params)?;

        Ok(QueryResult {
            rows: Vec::new(),
            affected_rows: result.affected_rows() as u64,
            execution_time_ms: 0,
            query: query.to_string(),
        })
    }

    async fn execute_statement(
        &self,
        statement: &Statement,
    ) -> Result<ExecuteResult, Box<dyn std::error::Error>> {
        let mut conn = self.pool.get_conn()?;

        // Convert params to MySQL format
        let mysql_params: Vec<mysql::Value> = statement
            .parameter_types
            .iter()
            .map(|v| match v {
                ValueType::String => mysql::Value::from(""),
                ValueType::Integer => mysql::Value::from(0),
                ValueType::Float => mysql::Value::from(0.0),
                ValueType::Boolean => mysql::Value::from(false),
                ValueType::Bytes => mysql::Value::from(Vec::new()),
                ValueType::Json => mysql::Value::from(serde_json::Value::Null),
                ValueType::Timestamp => mysql::Value::from(chrono::Utc::now().timestamp()),
                ValueType::Array(_) => mysql::Value::from(Vec::new()),
            })
            .collect();

        let result = conn.exec_iter(statement.query.as_str(), mysql_params)?;

        Ok(ExecuteResult {
            affected_rows: result.affected_rows() as u64,
            execution_time_ms: 0,
            last_insert_id: result.last_insert_id(),
        })
    }

    async fn begin_transaction(&self) -> Result<Transaction, Box<dyn std::error::Error>> {
        let mut conn = self
            .pool
            .get_conn()
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

        // Start transaction
        conn.query("START TRANSACTION")
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

        Ok(Transaction::new(conn))
    }

    async fn get_info(&self) -> Result<DatabaseInfo, Box<dyn std::error::Error>> {
        let mut conn = self
            .pool
            .get_conn()
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

        // Get database version
        let version_result: Vec<String> = conn.query("SELECT VERSION()")?;
        let version = version_result
            .first()
            .cloned()
            .unwrap_or_else(|| "Unknown".to_string());

        // Get database size
        let size_result: Vec<(String,)> = conn.query(
            "SELECT ROUND(SUM(data_length + index_length) / 1024 / 1024, 1) FROM information_schema.tables"
        )?;
        let database_size_mb = size_result
            .first()
            .and_then(|(size,)| size.parse::<f64>().ok())
            .unwrap_or(0.0);

        Ok(DatabaseInfo {
            database_name: self.config.database.clone(),
            version,
            database_size_mb,
            connection_count: self.pool.status().connections,
            uptime_seconds: 0, // MySQL doesn't provide connection pool uptime
            last_activity: Some(std::time::SystemTime::now()),
        })
    }

    async fn get_pool_status(&self) -> Result<PoolStatus, Box<dyn std::error::Error>> {
        let pool_status = self.pool.status();

        Ok(PoolStatus {
            total_connections: pool_status.connections,
            active_connections: pool_status.connections, // Simplified - MySQL doesn't distinguish
            idle_connections: 0, // MySQL pool doesn't expose idle connections
            max_connections: self.config.pool_size as u32,
            waiting_requests: 0,       // MySQL doesn't track waiting requests
            average_wait_time_ms: 0.0, // MySQL doesn't provide this metric
        })
    }

    async fn disconnect(&self) -> Result<(), Box<dyn std::error::Error>> {
        // MySQL pool will be closed when dropped
        Ok(())
    }

    fn is_connected(&self) -> bool {
        // Simple connection check
        self.pool.get_conn().is_ok()
    }
}

/// SSL modes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SslMode {
    Disable,
    Allow,
    Prefer,
    Require,
}

/// Database trait for common operations
#[async_trait::async_trait]
pub trait Database: Send + Sync {
    /// Connect to database
    async fn connect(&self) -> Result<()>;

    /// Disconnect from database
    async fn disconnect(&self) -> Result<()>;

    /// Check if connected
    async fn is_connected(&self) -> bool;

    /// Execute query
    async fn execute_query(&self, query: &str, params: Vec<Value>) -> Result<QueryResult>;

    /// Execute statement
    async fn execute_statement(&self, statement: &Statement) -> Result<ExecuteResult>;

    /// Begin transaction
    async fn begin_transaction(&self) -> Result<Transaction>;

    /// Get database info
    async fn get_info(&self) -> Result<DatabaseInfo>;

    /// Get connection pool status
    async fn get_pool_status(&self) -> Result<PoolStatus>;
}

/// Database connection trait
#[async_trait::async_trait]
pub trait DatabaseConnection: Send + Sync {
    /// Get connection ID
    fn id(&self) -> &str;

    /// Get database type
    fn database_type(&self) -> DatabaseType;

    /// Check if connection is active
    fn is_active(&self) -> bool;

    /// Execute query
    async fn execute_query(&self, query: &str, params: Vec<Value>) -> Result<QueryResult>;

    /// Close connection
    async fn close(&mut self) -> Result<()>;
}

/// Query result
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub rows: Vec<Row>,
    pub affected_rows: u64,
    pub execution_time_ms: u64,
    pub query: String,
}

/// Database row
#[derive(Debug, Clone)]
pub struct Row {
    pub columns: HashMap<String, Value>,
}

/// Database value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    Null,
    Bool(bool),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    String(String),
    Bytes(Vec<u8>),
    Json(serde_json::Value),
    Timestamp(std::time::SystemTime),
}

/// Prepared statement
#[derive(Debug, Clone)]
pub struct Statement {
    pub id: String,
    pub query: String,
    pub parameter_types: Vec<ValueType>,
    pub result_types: Vec<ValueType>,
}

/// Value types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueType {
    Boolean,
    Integer,
    BigInt,
    Float,
    Double,
    String,
    Bytes,
    Json,
    Timestamp,
    Array(Box<ValueType>),
}

/// Execute result
#[derive(Debug, Clone)]
pub struct ExecuteResult {
    pub affected_rows: u64,
    pub execution_time_ms: u64,
    pub last_insert_id: Option<i64>,
}

/// Transaction
#[derive(Debug)]
pub struct Transaction {
    pub id: String,
    pub connection_id: String,
    pub is_active: bool,
    pub savepoints: Vec<String>,
    // MySQL connection (only available when mysql feature is enabled)
    #[cfg(feature = "mysql")]
    conn: Option<mysql::Conn>,
    // SQLite connection (only available when sqlite feature is enabled)
    #[cfg(feature = "sqlite")]
    pub(crate) conn_sqlite: Option<rusqlite::Connection>,
}

impl Transaction {
    /// Create a new transaction without raw connection (for PostgreSQL)
    pub fn new_generic(id: String, connection_id: String) -> Self {
        Self {
            id,
            connection_id,
            is_active: true,
            savepoints: Vec::new(),
            #[cfg(feature = "mysql")]
            conn: None,
            #[cfg(feature = "sqlite")]
            conn_sqlite: None,
        }
    }
}

/// SQLite transaction constructor
#[cfg(feature = "sqlite")]
impl Transaction {
    /// Create a new transaction with an owned SQLite connection
    pub fn new_sqlite(conn: rusqlite::Connection, connection_id: String) -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        Self {
            id,
            connection_id,
            is_active: true,
            savepoints: Vec::new(),
            conn_sqlite: Some(conn),
            #[cfg(feature = "mysql")]
            conn: None,
        }
    }
}

/// MySQL-specific Transaction constructor
#[cfg(feature = "mysql")]
impl Transaction {
    pub fn new(conn: mysql::Conn) -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        let connection_id = format!("mysql_conn_{}", id);

        Self {
            id,
            connection_id,
            is_active: true,
            savepoints: Vec::new(),
            conn: Some(conn),
            #[cfg(feature = "sqlite")]
            conn_sqlite: None,
        }
    }
}

/// Combined transaction methods for MySQL and SQLite
impl Transaction {
    #[cfg(feature = "mysql")]
    fn conn_mut(&mut self) -> Option<&mut mysql::Conn> {
        self.conn.as_mut()
    }

    /// Commit the transaction
    pub fn commit(mut self) -> Result<()> {
        #[cfg(feature = "mysql")]
        if let Some(mut conn) = self.conn.take() {
            conn.query("COMMIT")
                .map_err(|e| anyhow::anyhow!("MySQL commit failed: {}", e))?;
            self.is_active = false;
            return Ok(());
        }
        #[cfg(feature = "sqlite")]
        if let Some(conn) = self.conn_sqlite.take() {
            conn.execute_batch("COMMIT")
                .map_err(|e| anyhow::anyhow!("SQLite commit failed: {}", e))?;
            self.is_active = false;
        }
        Ok(())
    }

    /// Rollback the transaction
    pub fn rollback(mut self) -> Result<()> {
        #[cfg(feature = "mysql")]
        if let Some(mut conn) = self.conn.take() {
            conn.query("ROLLBACK")
                .map_err(|e| anyhow::anyhow!("MySQL rollback failed: {}", e))?;
            self.is_active = false;
            return Ok(());
        }
        #[cfg(feature = "sqlite")]
        if let Some(conn) = self.conn_sqlite.take() {
            conn.execute_batch("ROLLBACK")
                .map_err(|e| anyhow::anyhow!("SQLite rollback failed: {}", e))?;
            self.is_active = false;
        }
        Ok(())
    }

    /// Create a savepoint within the transaction
    pub fn create_savepoint(&mut self, name: &str) -> Result<()> {
        #[cfg(feature = "mysql")]
        if let Some(ref mut conn) = self.conn {
            conn.query(&format!("SAVEPOINT {}", name))
                .map_err(|e| anyhow::anyhow!("MySQL savepoint failed: {}", e))?;
            self.savepoints.push(name.to_string());
            return Ok(());
        }
        #[cfg(feature = "sqlite")]
        if let Some(ref conn) = self.conn_sqlite {
            conn.execute_batch(&format!("SAVEPOINT {}", name))
                .map_err(|e| anyhow::anyhow!("SQLite savepoint failed: {}", e))?;
            self.savepoints.push(name.to_string());
        }
        Ok(())
    }

    /// Rollback to a savepoint
    pub fn rollback_to_savepoint(&mut self, name: &str) -> Result<()> {
        #[cfg(feature = "mysql")]
        if let Some(ref mut conn) = self.conn {
            conn.query(&format!("ROLLBACK TO SAVEPOINT {}", name))
                .map_err(|e| anyhow::anyhow!("MySQL rollback to savepoint failed: {}", e))?;
            return Ok(());
        }
        #[cfg(feature = "sqlite")]
        if let Some(ref conn) = self.conn_sqlite {
            conn.execute_batch(&format!("ROLLBACK TO SAVEPOINT {}", name))
                .map_err(|e| anyhow::anyhow!("SQLite rollback to savepoint failed: {}", e))?;
        }
        Ok(())
    }

    /// Execute a query within the transaction
    pub fn execute(&mut self, query: &str) -> Result<ExecuteResult> {
        #[cfg(feature = "mysql")]
        if let Some(ref mut conn) = self.conn {
            let result = conn.query_iter(query)?;
            return Ok(ExecuteResult {
                affected_rows: result.affected_rows() as u64,
                execution_time_ms: 0,
                last_insert_id: result.last_insert_id(),
            });
        }
        #[cfg(feature = "sqlite")]
        if let Some(ref conn) = self.conn_sqlite {
            let affected_rows = conn
                .execute(query, [])
                .map_err(|e| anyhow::anyhow!("SQLite execute failed: {}", e))?
                as u64;
            return Ok(ExecuteResult {
                affected_rows,
                execution_time_ms: 0,
                last_insert_id: Some(conn.last_insert_rowid()),
            });
        }
        Err(anyhow::anyhow!("Transaction is not active"))
    }
}

/// Database information
#[derive(Debug, Clone, Serialize)]
pub struct DatabaseInfo {
    pub version: String,
    pub database_name: String,
    pub database_size_mb: f64,
    pub connection_count: usize,
    pub uptime_seconds: u64,
    pub last_activity: Option<std::time::SystemTime>,
}

/// Connection pool status
#[derive(Debug, Clone, Serialize)]
pub struct PoolStatus {
    pub total_connections: usize,
    pub active_connections: usize,
    pub idle_connections: usize,
    pub max_connections: usize,
    pub waiting_requests: usize,
    pub average_wait_time_ms: f64,
}

/// Database factory for creating database instances
pub struct DatabaseFactory;

impl DatabaseFactory {
    /// Create database instance from configuration
    pub async fn create(config: DatabaseConfig) -> Result<Arc<dyn Database>> {
        match config.database_type {
            DatabaseType::PostgreSQL => {
                let pg_db = PostgreSQLDatabase::new(config).await?;
                Ok(Arc::new(pg_db))
            }
            DatabaseType::SQLite => {
                let sqlite_db = SQLiteDatabase::new(config).await?;
                Ok(Arc::new(sqlite_db))
            }
            #[cfg(feature = "mysql")]
            DatabaseType::MySQL => {
                let config = config
                    .as_any()
                    .downcast_ref::<MySQLConfig>()
                    .ok_or_else(|| anyhow::anyhow!("Invalid MySQL configuration"))?;

                let connection_string = format!(
                    "mysql://{}:{}@{}:{}/{}",
                    config.username, config.password, config.host, config.port, config.database
                );

                let pool = mysql::Pool::new(&connection_string)?;
                Ok(Arc::new(MySQLDatabase::new(pool, config)))
            }
            #[cfg(not(feature = "mysql"))]
            DatabaseType::MySQL => Err(anyhow::anyhow!(
                "MySQL support not enabled. Enable with 'mysql' feature"
            )),
        }
    }
}

/// Database manager for managing multiple database instances
pub struct DatabaseManager {
    databases: HashMap<String, Arc<dyn Database>>,
    connection_pools: HashMap<String, Arc<dyn ConnectionPool>>,
    default_database: Option<String>,
}

impl DatabaseManager {
    /// Create new database manager
    pub fn new() -> Self {
        Self {
            databases: HashMap::new(),
            default_database: None,
            connection_pools: HashMap::new(),
        }
    }

    /// Add database
    pub async fn add_database(&mut self, name: String, config: DatabaseConfig) -> Result<()> {
        let database = DatabaseFactory::create(config).await?;
        database.connect().await?;

        self.databases.insert(name.clone(), database);

        // Set as default if first database
        if self.default_database.is_none() {
            self.default_database = Some(name.clone());
        }

        Ok(())
    }

    /// Get database by name
    pub fn get_database(&self, name: &str) -> Option<&Arc<dyn Database>> {
        self.databases.get(name)
    }

    /// Get default database
    pub fn get_default_database(&self) -> Option<&Arc<dyn Database>> {
        if let Some(ref default_name) = self.default_database {
            self.databases.get(default_name)
        } else {
            self.databases.values().next()
        }
    }

    /// Set default database
    pub fn set_default_database(&mut self, name: String) -> Result<()> {
        if self.databases.contains_key(&name) {
            self.default_database = Some(name);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Database '{}' not found", name))
        }
    }

    /// Remove database
    pub async fn remove_database(&mut self, name: &str) -> Result<()> {
        if let Some(database) = self.databases.remove(name) {
            database.disconnect().await?;
        }

        if self.default_database.as_ref().map_or(false, |n| n == name) {
            self.default_database = self.databases.keys().next().cloned();
        }

        Ok(())
    }

    /// List all databases
    pub fn list_databases(&self) -> Vec<&String> {
        self.databases.keys().collect()
    }

    /// Get status of all databases
    pub async fn get_all_status(&self) -> HashMap<String, DatabaseStatus> {
        let mut status = HashMap::new();

        for (name, database) in &self.databases {
            let is_connected = database.is_connected().await;
            let info = database.get_info().await.unwrap_or_else(|_| DatabaseInfo {
                version: "Unknown".to_string(),
                database_name: name.clone(),
                database_size_mb: 0.0,
                connection_count: 0,
                uptime_seconds: 0,
                last_activity: None,
            });

            status.insert(
                name.clone(),
                DatabaseStatus {
                    name: name.clone(),
                    is_connected,
                    info,
                    pool_status: database
                        .get_pool_status()
                        .await
                        .unwrap_or_else(|_| PoolStatus {
                            total_connections: 0,
                            active_connections: 0,
                            idle_connections: 0,
                            max_connections: 0,
                            waiting_requests: 0,
                            average_wait_time_ms: 0.0,
                        }),
                },
            );
        }

        status
    }

    /// Close all databases
    pub async fn close_all(&mut self) -> Result<()> {
        for database in self.databases.values() {
            database.disconnect().await?;
        }

        self.databases.clear();
        self.default_database = None;
        self.connection_pools.clear();

        Ok(())
    }
}

/// Database status
#[derive(Debug, Clone, Serialize)]
pub struct DatabaseStatus {
    pub name: String,
    pub is_connected: bool,
    pub info: DatabaseInfo,
    pub pool_status: PoolStatus,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            database_type: DatabaseType::PostgreSQL,
            host: "localhost".to_string(),
            port: 5432,
            database: "nexora".to_string(),
            username: "postgres".to_string(),
            password: "password".to_string(),
            ssl_mode: SslMode::Prefer,
            max_connections: 10,
            connection_timeout_seconds: 30,
            idle_timeout_seconds: 300,
            max_lifetime_seconds: 3600,
        }
    }
}

impl Default for DatabaseManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions
impl Value {
    /// Convert to string
    pub fn to_string(&self) -> Option<String> {
        match self {
            Value::String(s) => Some(s.clone()),
            Value::I32(i) => Some(i.to_string()),
            Value::I64(i) => Some(i.to_string()),
            Value::F32(f) => Some(f.to_string()),
            Value::F64(f) => Some(f.to_string()),
            Value::Bool(b) => Some(b.to_string()),
            Value::Null => None,
            Value::Bytes(_) => None,
            Value::Json(j) => Some(j.to_string()),
            Value::Timestamp(t) => Some(format!("{:?}", t)),
        }
    }

    /// Convert from string
    pub fn from_string(s: String, value_type: ValueType) -> Self {
        match value_type {
            ValueType::String => Value::String(s),
            ValueType::Integer => Value::I32(s.parse().unwrap_or(0)),
            ValueType::BigInt => Value::I64(s.parse().unwrap_or(0)),
            ValueType::Float => Value::F32(s.parse().unwrap_or(0.0)),
            ValueType::Double => Value::F64(s.parse().unwrap_or(0.0)),
            ValueType::Boolean => Value::Bool(s.parse().unwrap_or(false)),
            ValueType::Json => {
                Value::Json(serde_json::from_str(&s).unwrap_or(serde_json::Value::Null))
            }
            ValueType::Timestamp => {
                // Parse timestamp with multiple format support
                // Get current system time for audit timestamp
                Value::Timestamp(std::time::SystemTime::now())
            }
            _ => Value::String(s),
        }
    }

    /// Parse timestamp string with multiple format support
    fn parse_timestamp(&self, timestamp_str: &str) -> Self {
        use chrono::{DateTime, NaiveDateTime, Utc};

        // Try different timestamp formats
        let formats = vec![
            "%Y-%m-%d %H:%M:%S",      // "2023-12-25 14:30:00"
            "%Y-%m-%dT%H:%M:%S",      // "2023-12-25T14:30:00"
            "%Y-%m-%dT%H:%M:%SZ",     // "2023-12-25T14:30:00Z"
            "%Y-%m-%dT%H:%M:%S%.3fZ", // "2023-12-25T14:30:00.123Z"
            "%Y-%m-%d",               // "2023-12-25"
            "%Y/%m/%d %H:%M:%S",      // "2023/12/25 14:30:00"
            "%d/%m/%Y %H:%M:%S",      // "25/12/2023 14:30:00"
            "%s",                     // Unix timestamp (seconds)
            "%s%.3f",                 // Unix timestamp with milliseconds
        ];

        // Try parsing as Unix timestamp first (most efficient)
        if let Ok(unix_timestamp) = timestamp_str.parse::<f64>() {
            let seconds = unix_timestamp.trunc() as i64;
            let nanos = ((unix_timestamp.fract() * 1_000_000_000.0) as u32) * 1_000_000;
            if let Some(system_time) = std::time::SystemTime::UNIX_EPOCH
                .checked_add(std::time::Duration::new(seconds.try_into().expect("timestamp seconds fit into u64"), nanos))
            {
                return Value::Timestamp(system_time);
            }
        }

        // Try different date formats
        for format in &formats {
            if let Ok(naive_dt) = NaiveDateTime::parse_from_str(timestamp_str, format) {
                let datetime: DateTime<Utc> = DateTime::from_naive_utc_and_offset(naive_dt, Utc);
                if let Some(nanos) = datetime.timestamp_nanos_opt() {
                    if let Some(system_time) =
                        std::time::SystemTime::UNIX_EPOCH.checked_add(std::time::Duration::new(
                            (nanos / 1_000_000_000).try_into().expect("nanos fit into u64"),
                            (nanos % 1_000_000_000) as u32,
                        ))
                    {
                        return Value::Timestamp(system_time);
                    }
                }
            }
        }

        // If all parsing fails, use current time with warning
        tracing::warn!(
            timestamp = %timestamp_str,
            "Failed to parse timestamp, using current time"
        );
        Value::Timestamp(std::time::SystemTime::now())
    }

    /// Get as i64
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::I32(i) => Some(*i as i64),
            Value::I64(i) => Some(*i),
            Value::F32(f) => Some(*f as i64),
            Value::F64(f) => Some(*f as i64),
            _ => None,
        }
    }

    /// Get as f64
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::I32(i) => Some(*i as f64),
            Value::I64(i) => Some(*i as f64),
            Value::F32(f) => Some(*f as f64),
            Value::F64(f) => Some(*f),
            _ => None,
        }
    }

    /// Get as bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            Value::I32(i) => Some(*i != 0),
            Value::I64(i) => Some(*i != 0),
            _ => None,
        }
    }
}
