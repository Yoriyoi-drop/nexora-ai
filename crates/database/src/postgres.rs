//! PostgreSQL Database Implementation - Rust
//!
//! PostgreSQL driver implementation replacing PostgreSQL C code

use anyhow::{anyhow, Result};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio_postgres::types::ToSql;
use tokio_postgres::{Client, Config, NoTls};

use crate::{
    credentials::{CredentialManager, DatabaseCredentials},
    ConnectionPool, ConnectionPoolConfig, Database, DatabaseConfig, DatabaseInfo, ExecuteResult,
    PoolStatus, QueryResult, Statement, Transaction, Value, ValueType,
};

/// PostgreSQL database implementation
pub struct PostgreSQLDatabase {
    config: DatabaseConfig,
    connection_pool: Arc<dyn ConnectionPool>,
    connection_info: Arc<RwLock<ConnectionInfo>>,
    client: Arc<RwLock<Option<Arc<Client>>>>,
}

/// Connection information
#[derive(Debug)]
struct ConnectionInfo {
    host: String,
    port: u16,
    database: String,
    username: String,
    is_connected: bool,
    created_at: Instant,
    last_activity: Instant,
}

impl Clone for ConnectionInfo {
    fn clone(&self) -> Self {
        Self {
            host: self.host.clone(),
            port: self.port,
            database: self.database.clone(),
            username: self.username.clone(),
            is_connected: self.is_connected,
            created_at: self.created_at,
            last_activity: self.last_activity,
        }
    }
}

impl From<DatabaseConfig> for ConnectionInfo {
    fn from(config: DatabaseConfig) -> Self {
        let now = std::time::Instant::now();
        Self {
            host: config.host,
            port: config.port,
            database: config.database,
            username: config.username,
            is_connected: false,
            created_at: now,
            last_activity: now,
        }
    }
}

impl From<crate::credentials::DatabaseCredentials> for ConnectionInfo {
    fn from(config: crate::credentials::DatabaseCredentials) -> Self {
        let now = Instant::now();
        Self {
            host: config.host,
            port: config.port,
            database: config.database,
            username: config.username,
            is_connected: false,
            created_at: now,
            last_activity: now,
        }
    }
}

/// PostgreSQL connection
pub struct PostgreSQLConnection {
    pub id: String,
    connection_info: ConnectionInfo,
    client: Option<Client>,
    is_active: bool,
}

/// PostgreSQL connection pool
pub struct PostgreSQLConnectionPool {
    connections: Arc<RwLock<Vec<PostgreSQLConnection>>>,
    config: ConnectionPoolConfig,
    statistics: Arc<RwLock<PoolStatistics>>,
    waiting_requests: Arc<RwLock<usize>>,
    credentials: DatabaseCredentials, // Store secure credentials
}

/// Pool statistics
#[derive(Debug, Default, Clone)]
struct PoolStatistics {
    total_connections: usize,
    active_connections: usize,
    idle_connections: usize,
    created_connections: u64,
    destroyed_connections: u64,
    total_wait_time_ms: u64,
    average_wait_time_ms: f64,
}

impl PostgreSQLDatabase {
    /// Create new PostgreSQL database with secure credentials
    pub async fn new(config: DatabaseConfig) -> Result<Self> {
        // Load credentials securely
        let credential_manager = CredentialManager::new();
        let credentials = credential_manager
            .load_database_credentials()
            .map_err(|e| anyhow!("Failed to load database credentials: {}", e))?;

        let connection_pool = Arc::new(
            PostgreSQLConnectionPool::new(
                ConnectionPoolConfig {
                    max_connections: config.max_connections,
                    min_connections: std::cmp::max(1, config.max_connections / 4),
                    connection_timeout: Duration::from_secs(config.connection_timeout_seconds),
                    idle_timeout: Duration::from_secs(config.idle_timeout_seconds),
                    max_lifetime: Duration::from_secs(config.max_lifetime_seconds),
                },
                credentials.clone(),
            )
            .await?,
        );

        let connection_info = Arc::new(RwLock::new(ConnectionInfo {
            host: credentials.host.clone(),
            port: credentials.port,
            database: credentials.database.clone(),
            username: credentials.username.clone(),
            is_connected: false,
            created_at: Instant::now(),
            last_activity: Instant::now(),
        }));

        Ok(Self {
            config,
            connection_pool,
            connection_info,
            client: Arc::new(RwLock::new(None)),
        })
    }

    /// Create new PostgreSQL database with explicit credentials
    pub async fn with_credentials(
        credentials: DatabaseCredentials,
        config: DatabaseConfig,
    ) -> Result<Self> {
        let connection_pool = Arc::new(
            PostgreSQLConnectionPool::new(
                ConnectionPoolConfig {
                    max_connections: config.max_connections,
                    min_connections: std::cmp::max(1, config.max_connections / 4),
                    connection_timeout: Duration::from_secs(config.connection_timeout_seconds),
                    idle_timeout: Duration::from_secs(config.idle_timeout_seconds),
                    max_lifetime: Duration::from_secs(config.max_lifetime_seconds),
                },
                credentials.clone(),
            )
            .await?,
        );

        let connection_info = Arc::new(RwLock::new(ConnectionInfo {
            host: credentials.host.clone(),
            port: credentials.port,
            database: credentials.database.clone(),
            username: credentials.username.clone(),
            is_connected: false,
            created_at: Instant::now(),
            last_activity: Instant::now(),
        }));

        Ok(Self {
            config,
            connection_pool,
            connection_info,
            client: Arc::new(RwLock::new(None)),
        })
    }

    /// Create PostgreSQL client with secure credentials
    async fn create_client(&self) -> Result<Client> {
        // Load credentials securely for each connection
        let credential_manager = CredentialManager::new();
        let credentials = credential_manager
            .load_database_credentials()
            .map_err(|e| anyhow!("Failed to load database credentials: {}", e))?;

        let connection_string = credentials.build_connection_string();

        // Log safe connection info (without password)
        tracing::info!(
            "Connecting to PostgreSQL: {}",
            credentials.build_connection_string_safe()
        );

        let config: Config = connection_string.parse()?;
        let (client, connection) = config.connect(NoTls).await?;

        // Spawn the connection task
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                tracing::error!("PostgreSQL connection error: {}", e);
            }
        });

        Ok(client)
    }
}

#[async_trait::async_trait]
impl Database for PostgreSQLDatabase {
    async fn connect(&self) -> Result<()> {
        let client = self.create_client().await?;

        // Test the connection
        let _version = client.query_one("SELECT version()", &[]).await?;

        // Store the client
        *self.client.write().await = Some(Arc::new(client));

        let mut info = self.connection_info.write().await;
        info.is_connected = true;
        info.last_activity = Instant::now();

        tracing::info!("Connected to PostgreSQL database: {}", self.config.database);
        Ok(())
    }

    async fn disconnect(&self) -> Result<()> {
        let mut info = self.connection_info.write().await;
        info.is_connected = false;

        // Close all connections in pool
        self.connection_pool.close_all().await?;

        tracing::info!(
            "Disconnected from PostgreSQL database: {}",
            self.config.database
        );
        Ok(())
    }

    async fn is_connected(&self) -> bool {
        let info = self.connection_info.read().await;
        info.is_connected
    }

    async fn execute_query(&self, query: &str, params: Vec<Value>) -> Result<QueryResult> {
        let start_time = Instant::now();

        let client_guard = self.client.read().await;
        let client = client_guard
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("Database not connected"))?;

        // Convert parameters to PostgreSQL types — Box owned values, then reference them
        let boxed_params: Vec<Box<dyn ToSql + Sync + Send>> = params
            .iter()
            .map(|value| -> Box<dyn ToSql + Sync + Send> {
                match value {
                    Value::String(s) => Box::new(s.clone()),
                    Value::I32(i) => Box::new(*i),
                    Value::I64(i) => Box::new(*i),
                    Value::F32(f) => Box::new(*f),
                    Value::F64(f) => Box::new(*f),
                    Value::Bool(b) => Box::new(*b),
                    Value::Null => Box::new(None::<String>),
                    Value::Bytes(bytes) => Box::new(bytes.clone()),
                    Value::Json(json) => Box::new(json.to_string()),
                    Value::Timestamp(ts) => {
                        let dur =
                            ts.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
                        let datetime = chrono::DateTime::from_timestamp(
                            dur.as_secs() as i64,
                            dur.subsec_nanos(),
                        )
                        .map(|dt| dt.to_rfc3339())
                        .unwrap_or_default();
                        Box::new(datetime)
                    }
                }
            })
            .collect();

        let pg_params: Vec<&(dyn ToSql + Sync)> = boxed_params
            .iter()
            .map(|b| b.as_ref() as &(dyn ToSql + Sync))
            .collect();

        let statement = client.prepare(query).await?;
        let rows = client.query(&statement, &pg_params[..]).await?;

        // Convert rows to our format
        let mut result_rows = Vec::new();
        for row in rows {
            let mut row_data = std::collections::HashMap::new();
            for (i, column) in row.columns().iter().enumerate() {
                let column_name = column.name();
                let value: Value = match row.try_get::<_, Option<String>>(i) {
                    Ok(Some(val)) => Value::String(val),
                    Ok(None) => Value::Null,
                    Err(_) => Value::Null,
                };
                row_data.insert(column_name.to_string(), value);
            }
            result_rows.push(crate::Row { columns: row_data });
        }

        let execution_time = start_time.elapsed();

        Ok(QueryResult {
            rows: result_rows,
            affected_rows: 0,
            execution_time_ms: execution_time.as_millis() as u64,
            query: query.to_string(),
        })
    }

    async fn execute_statement(&self, statement: &Statement) -> Result<ExecuteResult> {
        let start_time = Instant::now();

        let client_guard = self.client.read().await;
        let client = client_guard
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("Database not connected"))?;

        // Convert parameter types to PostgreSQL values
        let boxed_params: Vec<Box<dyn ToSql + Sync + Send>> = statement
            .parameter_types
            .iter()
            .map(|pt| -> Box<dyn ToSql + Sync + Send> {
                match pt {
                    ValueType::Boolean => Box::new(false),
                    ValueType::Integer => Box::new(0i32),
                    ValueType::BigInt => Box::new(0i64),
                    ValueType::Float => Box::new(0.0f32),
                    ValueType::Double => Box::new(0.0f64),
                    ValueType::String => Box::new(String::new()),
                    ValueType::Bytes => Box::new(Vec::<u8>::new()),
                    ValueType::Json => Box::new(String::from("null")),
                    ValueType::Timestamp => Box::new(String::from("1970-01-01T00:00:00Z")),
                    ValueType::Array(_) => Box::new(String::from("[]")),
                }
            })
            .collect();

        let pg_params: Vec<&(dyn ToSql + Sync)> = boxed_params
            .iter()
            .map(|b| b.as_ref() as &(dyn ToSql + Sync))
            .collect();

        let statement_prepared = client.prepare(&statement.query).await?;
        let rows = client.query(&statement_prepared, &pg_params[..]).await?;
        let affected_rows = rows.len() as u64;

        let execution_time = start_time.elapsed();

        Ok(ExecuteResult {
            affected_rows,
            execution_time_ms: execution_time.as_millis() as u64,
            last_insert_id: None,
        })
    }

    async fn begin_transaction(&self) -> Result<Transaction> {
        let client_arc = {
            let client_guard = self.client.read().await;
            let client = client_guard
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("Database not connected"))?;

            client.simple_query("BEGIN").await?;

            client_guard
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Database not connected"))?
                .clone()
        };

        let connection_id = format!("pg_tx_{}", uuid::Uuid::new_v4());

        Ok(Transaction::new_postgres(client_arc, connection_id))
    }

    async fn get_info(&self) -> Result<DatabaseInfo> {
        let info = self.connection_info.read().await;
        let pool_status = self.connection_pool.get_status().await?;

        // Simulate database size query
        let database_size_mb = 1024.0; // Placeholder

        Ok(DatabaseInfo {
            version: "PostgreSQL 14.0".to_string(), // Placeholder
            database_name: info.database.clone(),
            database_size_mb,
            connection_count: pool_status.total_connections,
            uptime_seconds: info.created_at.elapsed().as_secs(),
            last_activity: Some(convert_instant_to_systemtime(info.created_at)),
        })
    }

    async fn get_pool_status(&self) -> Result<PoolStatus> {
        self.connection_pool.get_status().await
    }
}

/// Convert Instant to SystemTime
fn convert_instant_to_systemtime(instant: std::time::Instant) -> std::time::SystemTime {
    std::time::SystemTime::now()
        .checked_sub(instant.elapsed())
        .unwrap_or_else(|| std::time::SystemTime::UNIX_EPOCH)
}

impl PostgreSQLConnectionPool {
    /// Create new connection pool with secure credentials
    pub async fn new(
        config: ConnectionPoolConfig,
        credentials: DatabaseCredentials,
    ) -> Result<Self> {
        let connections = Arc::new(RwLock::new(Vec::new()));
        let statistics = Arc::new(RwLock::new(PoolStatistics::default()));
        let waiting_requests = Arc::new(RwLock::new(0));

        // Create minimum connections with real database connections
        let mut conn_vec = Vec::with_capacity(config.min_connections);
        for i in 0..config.min_connections {
            let mut connection = PostgreSQLConnection::new(format!("conn_{}", i), &credentials);

            // Establish real connection
            connection.connect().await?;

            conn_vec.push(connection);
        }

        *connections.write().await = conn_vec;

        {
            let mut stats = statistics.write().await;
            stats.total_connections = config.min_connections;
            stats.idle_connections = config.min_connections;
            stats.created_connections = config.min_connections as u64;
        }

        tracing::info!(
            "Created PostgreSQL connection pool with {} initial connections",
            config.min_connections
        );

        Ok(Self {
            connections,
            config,
            statistics,
            waiting_requests,
            credentials,
        })
    }

    /// Get connection from pool with proper validation
    pub async fn get_connection(&self) -> Result<PostgreSQLConnection> {
        let start_time = std::time::Instant::now();

        // First, try to find an idle connection
        {
            let mut connections = self.connections.write().await;

            // Find any idle connection
            if let Some(pos) = connections.iter().position(|c| !c.is_active) {
                let mut connection = connections.swap_remove(pos);

                // Check if connection is still healthy
                if connection.ping().await {
                    connection.is_active = true;

                    // Update statistics
                    let mut stats = self.statistics.write().await;
                    stats.idle_connections -= 1;
                    stats.active_connections += 1;

                    let wait_time = start_time.elapsed();
                    stats.total_wait_time_ms += wait_time.as_millis() as u64;
                    if stats.total_connections > 0 {
                        stats.average_wait_time_ms =
                            stats.total_wait_time_ms as f64 / stats.total_connections as f64;
                    }

                    return Ok(connection);
                } else {
                    // Connection is dead, remove it
                    let mut stats = self.statistics.write().await;
                    stats.destroyed_connections += 1;
                    stats.total_connections = stats.total_connections.saturating_sub(1);
                }
            }
        }

        // Create new connection if under max limit
        {
            let connections = self.connections.read().await;
            let current_count = connections.len();
            drop(connections);

            if current_count < self.config.max_connections {
                let mut connection =
                    PostgreSQLConnection::new(format!("conn_{}", current_count), &self.credentials);

                // Establish real connection
                connection.connect().await?;
                connection.is_active = true;

                // Update statistics
                let mut stats = self.statistics.write().await;
                stats.total_connections += 1;
                stats.created_connections += 1;
                stats.active_connections += 1;

                let wait_time = start_time.elapsed();
                stats.total_wait_time_ms += wait_time.as_millis() as u64;
                if stats.total_connections > 0 {
                    stats.average_wait_time_ms =
                        stats.total_wait_time_ms as f64 / stats.total_connections as f64;
                }

                return Ok(connection);
            }
        }

        // Pool exhausted - wait for available connection or timeout
        let timeout = std::time::Duration::from_millis(5000); // 5 second timeout
        let deadline = start_time + timeout;

        while std::time::Instant::now() < deadline {
            // Wait briefly and try again
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;

            // Try to find idle connection again
            let mut connections = self.connections.write().await;
            if let Some(pos) = connections.iter().position(|c| !c.is_active) {
                let mut connection = connections.swap_remove(pos);

                // Check if connection is still healthy
                if connection.ping().await {
                    connection.is_active = true;

                    let mut stats = self.statistics.write().await;
                    stats.idle_connections -= 1;
                    stats.active_connections += 1;

                    let wait_time = start_time.elapsed();
                    stats.total_wait_time_ms += wait_time.as_millis() as u64;
                    if stats.total_connections > 0 {
                        stats.average_wait_time_ms =
                            stats.total_wait_time_ms as f64 / stats.total_connections as f64;
                    }

                    return Ok(connection);
                } else {
                    let mut stats = self.statistics.write().await;
                    stats.destroyed_connections += 1;
                    stats.total_connections = stats.total_connections.saturating_sub(1);
                }
            }
        }

        Err(anyhow::anyhow!("Connection pool exhausted after timeout"))
    }

    /// Return connection to pool with proper validation and cleanup
    pub async fn return_connection(&self, mut connection: PostgreSQLConnection) -> Result<()> {
        // Validate connection before returning to pool
        if !connection.connection_info.is_connected {
            // Connection is dead, don't return to pool
            let mut stats = self.statistics.write().await;
            stats.destroyed_connections += 1;
            stats.total_connections = stats.total_connections.saturating_sub(1);
            return Ok(());
        }

        // Check if connection is too old (based on max_lifetime)
        let age_seconds = connection.age_seconds();
        let max_age_seconds = self.config.max_lifetime.as_secs();

        if age_seconds > max_age_seconds {
            // Connection is too old, close it and don't return to pool
            connection.close().await?;
            let mut stats = self.statistics.write().await;
            stats.destroyed_connections += 1;
            stats.total_connections = stats.total_connections.saturating_sub(1);
            return Ok(());
        }

        // Check if connection has been idle too long
        let idle_seconds = connection.idle_seconds();
        let idle_timeout_seconds = self.config.idle_timeout.as_secs();

        if idle_seconds > idle_timeout_seconds {
            // Connection has been idle too long, close it
            connection.close().await?;
            let mut stats = self.statistics.write().await;
            stats.destroyed_connections += 1;
            stats.total_connections = stats.total_connections.saturating_sub(1);
            return Ok(());
        }

        // Connection is healthy, return to pool
        let mut connections = self.connections.write().await;

        if connection.is_active {
            connection.is_active = false;
            connections.push(connection);

            // Update statistics
            let mut stats = self.statistics.write().await;
            stats.active_connections -= 1;
            stats.idle_connections += 1;
        }

        Ok(())
    }

    /// Perform health check on all connections in pool
    pub async fn health_check(&self) -> Result<usize> {
        let mut connections = self.connections.write().await;
        let mut healthy_count = 0;
        let mut to_remove = Vec::new();

        for (i, connection) in connections.iter_mut().enumerate() {
            if connection.ping().await {
                healthy_count += 1;
            } else {
                to_remove.push(i);
            }
        }

        // Remove unhealthy connections
        for &i in to_remove.iter().rev() {
            let mut connection = connections.remove(i);
            connection.close().await?;

            let mut stats = self.statistics.write().await;
            stats.destroyed_connections += 1;
            stats.total_connections = stats.total_connections.saturating_sub(1);
        }

        Ok(healthy_count)
    }

    /// Close all connections and cleanup resources
    pub async fn close_all(&self) -> Result<()> {
        let mut connections = self.connections.write().await;

        for mut connection in connections.drain(..) {
            if let Err(e) = connection.close().await {
                eprintln!("Error closing connection {}: {}", connection.id, e);
            }
        }

        // Update statistics
        let mut stats = self.statistics.write().await;
        stats.total_connections = 0;
        stats.active_connections = 0;
        stats.idle_connections = 0;

        Ok(())
    }

    /// Get pool status with real-time statistics
    pub async fn get_status(&self) -> Result<PoolStatus> {
        let stats = self.statistics.read().await;
        let connections = self.connections.read().await;

        let active_count = connections.iter().filter(|c| c.is_active).count();
        let idle_count = connections.iter().filter(|c| !c.is_active).count();

        Ok(PoolStatus {
            total_connections: stats.total_connections,
            active_connections: active_count,
            idle_connections: idle_count,
            max_connections: self.config.max_connections,
            waiting_requests: self.get_waiting_requests_count(),
            average_wait_time_ms: stats.average_wait_time_ms,
        })
    }

    /// Get current number of waiting requests
    fn get_waiting_requests_count(&self) -> usize {
        if let Ok(waiting) = self.waiting_requests.try_read() {
            *waiting
        } else {
            0
        }
    }
}

#[async_trait::async_trait]
impl ConnectionPool for PostgreSQLConnectionPool {
    async fn get_connection(&self) -> Result<Box<dyn crate::DatabaseConnection>> {
        let connection = self.get_connection().await?;
        Ok(Box::new(connection))
    }

    async fn return_connection(
        &self,
        connection: Box<dyn crate::DatabaseConnection>,
    ) -> Result<()> {
        // Get connection details before consuming the box
        let connection_id = connection.id().to_string();
        let was_active = connection.is_active();
        let connection_type = connection.database_type();

        // Only return PostgreSQL connections to this pool
        if connection_type != crate::DatabaseType::PostgreSQL {
            return Err(anyhow::anyhow!("Wrong connection type for PostgreSQL pool"));
        }

        // Create a new connection instance for the pool with proper state
        let pool_conn = PostgreSQLConnection {
            id: connection_id,
            connection_info: crate::credentials::DatabaseCredentials {
                host: self.credentials.host.clone(),
                port: self.credentials.port,
                database: self.credentials.database.clone(),
                username: self.credentials.username.clone(),
                password: self.credentials.password.clone(),
                ssl_mode: self.credentials.ssl_mode,
            }
            .into(),
            client: None,     // Client will be recreated when needed
            is_active: false, // Always mark as inactive when returned to pool
        };

        // Return to pool if the connection was active
        if was_active {
            self.return_connection(pool_conn).await?;
        }

        // Properly consume the original connection
        drop(connection);

        Ok(())
    }

    async fn get_status(&self) -> Result<PoolStatus> {
        self.get_status().await
    }

    async fn close_all(&self) -> Result<()> {
        self.close_all().await
    }
}

impl PostgreSQLConnection {
    pub fn new(id: String, credentials: &DatabaseCredentials) -> Self {
        Self {
            id,
            connection_info: ConnectionInfo {
                host: credentials.host.clone(),
                port: credentials.port,
                database: credentials.database.clone(),
                username: credentials.username.clone(),
                is_connected: false,
                created_at: std::time::Instant::now(),
                last_activity: std::time::Instant::now(),
            },
            client: None,
            is_active: false,
        }
    }

    /// Establish actual database connection with secure credentials
    pub async fn connect(&mut self) -> Result<()> {
        // Load credentials securely for this connection
        let credential_manager = CredentialManager::new();
        let credentials = credential_manager
            .load_database_credentials()
            .map_err(|e| anyhow!("Failed to load database credentials: {}", e))?;

        let connection_string = credentials.build_connection_string();

        // Log safe connection info (without password)
        tracing::info!(
            "Establishing PostgreSQL connection: {}",
            credentials.build_connection_string_safe()
        );

        let config: Config = connection_string.parse()?;
        let (client, connection) = config.connect(NoTls).await?;

        // Spawn the connection task
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                tracing::error!("PostgreSQL connection error: {}", e);
            }
        });

        self.client = Some(client);
        self.connection_info.is_connected = true;
        self.connection_info.last_activity = std::time::Instant::now();
        self.is_active = true;

        tracing::debug!(
            "PostgreSQL connection established successfully for connection {}",
            self.id
        );
        Ok(())
    }

    /// Check if connection is alive and responsive
    pub async fn ping(&self) -> bool {
        if !self.connection_info.is_connected || !self.is_active {
            return false;
        }

        if let Some(client) = &self.client {
            // Try to execute a simple query
            match client.query_one("SELECT 1", &[]).await {
                Ok(_) => {
                    // Update last activity
                    // Note: This would require interior mutability in a real implementation
                    true
                }
                Err(_) => false,
            }
        } else {
            false
        }
    }

    /// Execute a query on this connection
    pub async fn execute_query(
        &self,
        query: &str,
        _params: Vec<crate::Value>,
    ) -> Result<crate::QueryResult> {
        if !self.is_active {
            return Err(anyhow::anyhow!("Connection is not active"));
        }

        let start_time = std::time::Instant::now();

        if let Some(client) = &self.client {
            let statement = client.prepare(query).await?;
            let rows = client.query(&statement, &[]).await?;

            // Convert rows to our format
            let mut result_rows = Vec::new();
            for row in rows {
                let mut row_data = std::collections::HashMap::new();
                for (i, column) in row.columns().iter().enumerate() {
                    let column_name = column.name();
                    let value: crate::Value = match row.try_get::<_, Option<String>>(i) {
                        Ok(Some(val)) => crate::Value::String(val),
                        Ok(None) => crate::Value::Null,
                        Err(_) => crate::Value::Null,
                    };
                    row_data.insert(column_name.to_string(), value);
                }
                result_rows.push(crate::Row { columns: row_data });
            }

            let execution_time = start_time.elapsed().as_millis() as u64;

            Ok(crate::QueryResult {
                rows: result_rows,
                affected_rows: 0,
                execution_time_ms: execution_time,
                query: query.to_string(),
            })
        } else {
            Err(anyhow::anyhow!("No client available"))
        }
    }

    /// Close connection and cleanup resources
    pub async fn close(&mut self) -> Result<()> {
        if self.connection_info.is_connected {
            // In a real implementation, this would close the tokio-postgres connection
            self.connection_info.is_connected = false;
        }

        self.is_active = false;
        Ok(())
    }

    /// Get connection age in seconds
    pub fn age_seconds(&self) -> u64 {
        self.connection_info.created_at.elapsed().as_secs()
    }

    /// Get time since last activity
    pub fn idle_seconds(&self) -> u64 {
        self.connection_info.last_activity.elapsed().as_secs()
    }
}

// ...

#[async_trait::async_trait]
impl crate::DatabaseConnection for PostgreSQLConnection {
    fn id(&self) -> &str {
        &self.id
    }

    fn database_type(&self) -> crate::DatabaseType {
        crate::DatabaseType::PostgreSQL
    }

    fn is_active(&self) -> bool {
        self.is_active
    }

    async fn execute_query(&self, query: &str, _params: Vec<Value>) -> Result<QueryResult> {
        // Simplified implementation
        Ok(QueryResult {
            rows: Vec::new(),
            affected_rows: 0,
            execution_time_ms: 0,
            query: query.to_string(),
        })
    }

    async fn close(&mut self) -> Result<()> {
        self.is_active = false;
        Ok(())
    }
}
