//! SQLite Database Implementation - Rust
//!
//! SQLite driver implementation using rusqlite. Feature-gated behind `sqlite`.
//! When the feature is not active, all operations return errors.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
#[cfg(feature = "sqlite")]
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use crate::{
    ConnectionPool, ConnectionPoolConfig, Database, DatabaseConfig, DatabaseInfo, DatabaseType,
    ExecuteResult, PoolStatus, QueryResult, Row, Statement, Transaction, Value,
};

/// SQLite database implementation
pub struct SQLiteDatabase {
    config: DatabaseConfig,
    connection_pool: Arc<dyn ConnectionPool>,
    sqlite_pool: Arc<SQLiteConnectionPool>,
    connection_info: Arc<RwLock<ConnectionInfo>>,
}

/// Connection information
#[derive(Debug, Clone)]
struct ConnectionInfo {
    database_path: String,
    is_connected: bool,
    created_at: Instant,
    last_activity: Instant,
}

/// SQLite connection
pub struct SQLiteConnection {
    id: String,
    connection_info: ConnectionInfo,
    is_active: bool,
    transaction_depth: u32,
    #[cfg(feature = "sqlite")]
    conn: Option<Arc<Mutex<rusqlite::Connection>>>,
}

/// SQLite connection pool
pub struct SQLiteConnectionPool {
    connections: Arc<RwLock<Vec<SQLiteConnection>>>,
    config: ConnectionPoolConfig,
    statistics: Arc<RwLock<PoolStatistics>>,
    waiting_requests: Arc<RwLock<usize>>,
    database_path: String,
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
    connection_errors: u64,
}

// ---------------------------------------------------------------------------
// Value conversion helpers (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "sqlite")]
fn value_to_rusqlite(value: &Value) -> rusqlite::types::Value {
    match value {
        Value::Null => rusqlite::types::Value::Null,
        Value::Bool(b) => rusqlite::types::Value::Integer(if *b { 1 } else { 0 }),
        Value::I32(i) => rusqlite::types::Value::Integer(*i as i64),
        Value::I64(i) => rusqlite::types::Value::Integer(*i),
        Value::F32(f) => rusqlite::types::Value::Real(*f as f64),
        Value::F64(f) => rusqlite::types::Value::Real(*f),
        Value::String(s) => rusqlite::types::Value::Text(s.clone()),
        Value::Bytes(b) => rusqlite::types::Value::Blob(b.clone()),
        Value::Json(j) => rusqlite::types::Value::Text(j.to_string()),
        Value::Timestamp(t) => {
            let dur = t.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
            let ts = chrono::DateTime::from_timestamp(dur.as_secs() as i64, dur.subsec_nanos())
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_default();
            rusqlite::types::Value::Text(ts)
        }
    }
}

#[cfg(feature = "sqlite")]
fn rusqlite_value_to_value(val: &rusqlite::types::Value) -> Value {
    match val {
        rusqlite::types::Value::Null => Value::Null,
        rusqlite::types::Value::Integer(i) => Value::I64(*i),
        rusqlite::types::Value::Real(f) => Value::F64(*f),
        rusqlite::types::Value::Text(s) => Value::String(s.clone()),
        rusqlite::types::Value::Blob(b) => Value::Bytes(b.clone()),
    }
}

#[cfg(feature = "sqlite")]
fn prepare_params(params: &[Value]) -> Vec<rusqlite::types::Value> {
    params.iter().map(value_to_rusqlite).collect()
}

// ---------------------------------------------------------------------------
// SQLiteDatabase
// ---------------------------------------------------------------------------

impl SQLiteDatabase {
    /// Create new SQLite database
    pub async fn new(config: DatabaseConfig) -> Result<Self> {
        let database_path = format!("{}.db", config.database);

        let sqlite_pool = Arc::new(
            SQLiteConnectionPool::new(
                ConnectionPoolConfig {
                    max_connections: config.max_connections,
                    min_connections: std::cmp::max(1, config.max_connections / 4),
                    connection_timeout: Duration::from_secs(config.connection_timeout_seconds),
                    idle_timeout: Duration::from_secs(config.idle_timeout_seconds),
                    max_lifetime: Duration::from_secs(config.max_lifetime_seconds),
                },
                &database_path,
            )
            .await?,
        );

        let connection_pool: Arc<dyn ConnectionPool> = sqlite_pool.clone();

        let connection_info = Arc::new(RwLock::new(ConnectionInfo {
            database_path,
            is_connected: false,
            created_at: Instant::now(),
            last_activity: Instant::now(),
        }));

        Ok(Self {
            config,
            connection_pool,
            sqlite_pool,
            connection_info,
        })
    }

    async fn get_database_path(&self) -> String {
        let info = self.connection_info.read().await;
        info.database_path.clone()
    }
}

#[async_trait::async_trait]
impl Database for SQLiteDatabase {
    async fn connect(&self) -> Result<()> {
        let database_path = self.get_database_path().await;

        #[cfg(feature = "sqlite")]
        {
            let conn = rusqlite::Connection::open(&database_path).map_err(|e| {
                anyhow!("Failed to open SQLite database '{}': {}", database_path, e)
            })?;

            conn.execute_batch("SELECT 1")
                .map_err(|e| anyhow!("SQLite connection test failed: {}", e))?;

            drop(conn);

            let mut info = self.connection_info.write().await;
            info.is_connected = true;
            info.last_activity = Instant::now();

            tracing::info!("Connected to SQLite database: {}", database_path);
            return Ok(());
        }

        #[cfg(not(feature = "sqlite"))]
        Err(anyhow!(
            "SQLite support not enabled. Enable with 'sqlite' feature"
        ))
    }

    async fn disconnect(&self) -> Result<()> {
        let mut info = self.connection_info.write().await;
        info.is_connected = false;

        self.connection_pool.close_all().await?;

        tracing::info!("Disconnected from SQLite database");
        Ok(())
    }

    async fn is_connected(&self) -> bool {
        let info = self.connection_info.read().await;
        info.is_connected
    }

    async fn execute_query(&self, query: &str, params: Vec<Value>) -> Result<QueryResult> {
        let start_time = Instant::now();

        #[cfg(feature = "sqlite")]
        {
            let connection = self.sqlite_pool.get_connection_inner().await?;
            let query_owned = query.to_string();
            let params_owned = params;

            let result = tokio::task::spawn_blocking(move || {
                let conn_guard = connection
                    .conn
                    .as_ref()
                    .ok_or_else(|| anyhow!("Connection not open"))?;
                let conn = conn_guard
                    .lock()
                    .map_err(|e| anyhow!("Mutex poisoned: {}", e))?;

                let rusqlite_params = prepare_params(&params_owned);
                let param_refs: Vec<&dyn rusqlite::types::ToSql> = rusqlite_params
                    .iter()
                    .map(|v| v as &dyn rusqlite::types::ToSql)
                    .collect();

                let mut stmt = conn.prepare(&query_owned)?;
                let column_count = stmt.column_count();
                let column_names: Vec<String> = (0..column_count)
                    .map(|i| stmt.column_name(i).unwrap_or("?").to_string())
                    .collect();

                let rows = stmt.query_map(param_refs.as_slice(), |row| {
                    let mut row_data = HashMap::new();
                    for i in 0..column_count {
                        let col_name = column_names[i].clone();
                        let raw: rusqlite::types::Value = row.get_unwrap(i);
                        row_data.insert(col_name, rusqlite_value_to_value(&raw));
                    }
                    Ok(Row { columns: row_data })
                })?;

                let mut result_rows = Vec::new();
                for row in rows {
                    result_rows.push(row?);
                }

                let affected_rows = conn.changes() as u64;

                Ok::<_, anyhow::Error>((result_rows, affected_rows))
            })
            .await??;

            let execution_time = start_time.elapsed();

            return Ok(QueryResult {
                rows: result.0,
                affected_rows: result.1,
                execution_time_ms: execution_time.as_millis() as u64,
                query: query.to_string(),
            });
        }

        #[cfg(not(feature = "sqlite"))]
        Err(anyhow!(
            "SQLite support not enabled. Enable with 'sqlite' feature"
        ))
    }

    async fn execute_statement(&self, statement: &Statement) -> Result<ExecuteResult> {
        let start_time = Instant::now();

        #[cfg(feature = "sqlite")]
        {
            let connection = self.sqlite_pool.get_connection_inner().await?;
            let query = statement.query.clone();

            let result = tokio::task::spawn_blocking(move || {
                let conn_guard = connection
                    .conn
                    .as_ref()
                    .ok_or_else(|| anyhow!("Connection not open"))?;
                let conn = conn_guard
                    .lock()
                    .map_err(|e| anyhow!("Mutex poisoned: {}", e))?;

                let affected_rows = conn.execute(&query, [])? as u64;
                let last_id = conn.last_insert_rowid();

                Ok::<_, anyhow::Error>((affected_rows, last_id))
            })
            .await??;

            let execution_time = start_time.elapsed();

            return Ok(ExecuteResult {
                affected_rows: result.0,
                execution_time_ms: execution_time.as_millis() as u64,
                last_insert_id: Some(result.1),
            });
        }

        #[cfg(not(feature = "sqlite"))]
        Err(anyhow!(
            "SQLite support not enabled. Enable with 'sqlite' feature"
        ))
    }

    async fn begin_transaction(&self) -> Result<Transaction> {
        #[cfg(feature = "sqlite")]
        {
            let mut connection = self.sqlite_pool.get_connection_inner().await?;
            let connection_id = connection.id.clone();

            let conn_arc = connection
                .conn
                .take()
                .ok_or_else(|| anyhow!("Connection not open"))?;

            self.sqlite_pool.mark_connection_consumed().await;

            let conn = Arc::try_unwrap(conn_arc)
                .map_err(|_| anyhow!("Connection owned by multiple references"))?
                .into_inner()
                .map_err(|e| anyhow!("Mutex poisoned: {}", e))?;

            conn.execute_batch("BEGIN TRANSACTION")
                .map_err(|e| anyhow!("Failed to begin transaction: {}", e))?;

            return Ok(Transaction::new_sqlite(conn, connection_id));
        }

        #[cfg(not(feature = "sqlite"))]
        Err(anyhow!(
            "SQLite support not enabled. Enable with 'sqlite' feature"
        ))
    }

    async fn get_info(&self) -> Result<DatabaseInfo> {
        let info = self.connection_info.read().await;
        let pool_status = self.connection_pool.get_status().await?;
        let database_path = info.database_path.clone();
        let created_at = info.created_at;
        drop(info);

        #[cfg(feature = "sqlite")]
        {
            let path = database_path.clone();
            let (version, size_mb) = tokio::task::spawn_blocking(move || {
                let conn = rusqlite::Connection::open(&path)
                    .map_err(|e| anyhow!("Failed to open database: {}", e))?;

                let version: String = conn
                    .query_row("SELECT sqlite_version()", [], |row| row.get(0))
                    .unwrap_or_else(|_| "unknown".to_string());

                let page_count: f64 = conn
                    .query_row("PRAGMA page_count", [], |row| row.get(0))
                    .unwrap_or(0.0);

                let page_size: f64 = conn
                    .query_row("PRAGMA page_size", [], |row| row.get(0))
                    .unwrap_or(4096.0);

                let size_mb = (page_count * page_size) / (1024.0 * 1024.0);

                Ok::<_, anyhow::Error>((version, size_mb))
            })
            .await??;

            return Ok(DatabaseInfo {
                version: format!("SQLite {}", version),
                database_name: database_path,
                database_size_mb: size_mb,
                connection_count: pool_status.total_connections,
                uptime_seconds: created_at.elapsed().as_secs(),
                last_activity: Some(convert_instant_to_systemtime(created_at)),
            });
        }

        #[cfg(not(feature = "sqlite"))]
        Ok(DatabaseInfo {
            version: "SQLite (not enabled)".to_string(),
            database_name: database_path,
            database_size_mb: 0.0,
            connection_count: pool_status.total_connections,
            uptime_seconds: created_at.elapsed().as_secs(),
            last_activity: Some(convert_instant_to_systemtime(created_at)),
        })
    }

    async fn get_pool_status(&self) -> Result<PoolStatus> {
        self.connection_pool.get_status().await
    }
}

fn convert_instant_to_systemtime(instant: std::time::Instant) -> std::time::SystemTime {
    std::time::SystemTime::now()
        .checked_sub(instant.elapsed())
        .unwrap_or_else(|| std::time::SystemTime::UNIX_EPOCH)
}

// ---------------------------------------------------------------------------
// SQLiteConnection
// ---------------------------------------------------------------------------

impl SQLiteConnection {
    #[cfg(feature = "sqlite")]
    pub async fn new(id: String, database_path: &str) -> Result<Self> {
        let conn = rusqlite::Connection::open(database_path)
            .map_err(|e| anyhow!("Failed to open SQLite '{}': {}", database_path, e))?;

        Ok(Self {
            id,
            connection_info: ConnectionInfo {
                database_path: database_path.to_string(),
                is_connected: true,
                created_at: Instant::now(),
                last_activity: Instant::now(),
            },
            is_active: true,
            transaction_depth: 0,
            conn: Some(Arc::new(Mutex::new(conn))),
        })
    }

    #[cfg(not(feature = "sqlite"))]
    pub async fn new(_id: String, _database_path: &str) -> Result<Self> {
        Err(anyhow!(
            "SQLite support not enabled. Enable with 'sqlite' feature"
        ))
    }

    #[cfg(feature = "sqlite")]
    pub async fn ping(&self) -> bool {
        let guard = match self.conn.as_ref() {
            Some(m) => match m.lock() {
                Ok(g) => g,
                Err(_) => return false,
            },
            None => return false,
        };
        guard.execute_batch("SELECT 1").is_ok()
    }

    #[cfg(not(feature = "sqlite"))]
    pub async fn ping(&self) -> bool {
        false
    }

    pub async fn close(&mut self) -> Result<()> {
        self.is_active = false;
        #[cfg(feature = "sqlite")]
        {
            self.conn = None;
        }
        Ok(())
    }

    #[cfg(feature = "sqlite")]
    pub async fn begin_transaction(&mut self) -> Result<()> {
        if self.transaction_depth == 0 {
            let guard = self
                .conn
                .as_ref()
                .ok_or_else(|| anyhow!("Connection not open"))?;
            let conn = guard.lock().map_err(|e| anyhow!("Mutex poisoned: {}", e))?;
            conn.execute_batch("BEGIN TRANSACTION")?;
        }
        self.transaction_depth += 1;
        Ok(())
    }

    #[cfg(not(feature = "sqlite"))]
    pub async fn begin_transaction(&mut self) -> Result<()> {
        Err(anyhow!(
            "SQLite support not enabled. Enable with 'sqlite' feature"
        ))
    }

    #[cfg(feature = "sqlite")]
    pub async fn commit_transaction(&mut self) -> Result<()> {
        if self.transaction_depth == 1 {
            let guard = self
                .conn
                .as_ref()
                .ok_or_else(|| anyhow!("Connection not open"))?;
            let conn = guard.lock().map_err(|e| anyhow!("Mutex poisoned: {}", e))?;
            conn.execute_batch("COMMIT")?;
        }
        if self.transaction_depth > 0 {
            self.transaction_depth -= 1;
        }
        Ok(())
    }

    #[cfg(not(feature = "sqlite"))]
    pub async fn commit_transaction(&mut self) -> Result<()> {
        Err(anyhow!(
            "SQLite support not enabled. Enable with 'sqlite' feature"
        ))
    }

    #[cfg(feature = "sqlite")]
    pub async fn rollback_transaction(&mut self) -> Result<()> {
        if self.transaction_depth == 1 {
            let guard = self
                .conn
                .as_ref()
                .ok_or_else(|| anyhow!("Connection not open"))?;
            let conn = guard.lock().map_err(|e| anyhow!("Mutex poisoned: {}", e))?;
            conn.execute_batch("ROLLBACK")?;
        }
        if self.transaction_depth > 0 {
            self.transaction_depth -= 1;
        }
        Ok(())
    }

    #[cfg(not(feature = "sqlite"))]
    pub async fn rollback_transaction(&mut self) -> Result<()> {
        Err(anyhow!(
            "SQLite support not enabled. Enable with 'sqlite' feature"
        ))
    }
}

// ---------------------------------------------------------------------------
// SQLiteConnectionPool
// ---------------------------------------------------------------------------

impl SQLiteConnectionPool {
    pub async fn new(config: ConnectionPoolConfig, database_path: &str) -> Result<Self> {
        let connections = Arc::new(RwLock::new(Vec::new()));
        let statistics = Arc::new(RwLock::new(PoolStatistics::default()));
        let waiting_requests = Arc::new(RwLock::new(0));

        let mut conn_vec = Vec::new();
        for i in 0..config.min_connections {
            match SQLiteConnection::new(format!("sqlite_conn_{}", i), database_path).await {
                Ok(conn) => conn_vec.push(conn),
                Err(e) => {
                    tracing::warn!("Failed to create initial SQLite connection: {}", e);
                }
            }
        }

        *connections.write().await = conn_vec;

        {
            let mut stats = statistics.write().await;
            stats.total_connections = config.min_connections;
            stats.active_connections = 0;
            stats.idle_connections = config.min_connections;
        }

        Ok(Self {
            connections,
            config,
            statistics,
            waiting_requests,
            database_path: database_path.to_string(),
        })
    }

    pub async fn get_connection_inner(&self) -> Result<SQLiteConnection> {
        let deadline = Instant::now() + self.config.connection_timeout;

        loop {
            let start_time = Instant::now();
            *self.waiting_requests.write().await += 1;

            {
                let mut connections = self.connections.write().await;

                if let Some(pos) = connections.iter().position(|c| !c.is_active) {
                    let conn = connections.swap_remove(pos);

                    #[cfg(feature = "sqlite")]
                    if !conn.ping().await {
                        let mut stats = self.statistics.write().await;
                        stats.destroyed_connections += 1;
                        stats.total_connections = stats.total_connections.saturating_sub(1);
                        *self.waiting_requests.write().await -= 1;
                        // Continue loop to retry
                        tokio::time::sleep(Duration::from_millis(10)).await;
                        continue;
                    }

                    {
                        let mut stats = self.statistics.write().await;
                        stats.idle_connections = stats.idle_connections.saturating_sub(1);
                        stats.active_connections += 1;
                        let wait_time = start_time.elapsed();
                        stats.total_wait_time_ms += wait_time.as_millis() as u64;
                        stats.average_wait_time_ms = if stats.total_connections > 0 {
                            stats.total_wait_time_ms as f64 / stats.total_connections as f64
                        } else {
                            0.0
                        };
                    }
                    *self.waiting_requests.write().await -= 1;
                    return Ok(conn);
                }
            }

            if let Ok(connections) = self.connections.try_read() {
                if connections.len() < self.config.max_connections {
                    drop(connections);
                    let new_id = format!("sqlite_conn_{}", uuid::Uuid::new_v4());
                    match SQLiteConnection::new(new_id, &self.database_path).await {
                        Ok(conn) => {
                            {
                                let mut stats = self.statistics.write().await;
                                stats.total_connections += 1;
                                stats.created_connections += 1;
                                stats.active_connections += 1;
                                let wait_time = start_time.elapsed();
                                stats.total_wait_time_ms += wait_time.as_millis() as u64;
                                stats.average_wait_time_ms = if stats.total_connections > 0 {
                                    stats.total_wait_time_ms as f64 / stats.total_connections as f64
                                } else {
                                    0.0
                                };
                            }
                            *self.waiting_requests.write().await -= 1;
                            return Ok(conn);
                        }
                        Err(e) => {
                            let mut stats = self.statistics.write().await;
                            stats.connection_errors += 1;
                            *self.waiting_requests.write().await -= 1;
                            return Err(anyhow!("Failed to create SQLite connection: {}", e));
                        }
                    }
                }
            }

            if Instant::now() >= deadline {
                *self.waiting_requests.write().await -= 1;
                return Err(anyhow!("SQLite connection pool exhausted"));
            }

            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    pub async fn return_connection(&self, mut connection: SQLiteConnection) -> Result<()> {
        let mut connections = self.connections.write().await;

        if connection.is_active {
            connection.is_active = false;
            connections.push(connection);

            let mut stats = self.statistics.write().await;
            stats.active_connections = stats.active_connections.saturating_sub(1);
            stats.idle_connections += 1;
        }

        Ok(())
    }

    pub async fn mark_connection_consumed(&self) {
        let mut stats = self.statistics.write().await;
        stats.active_connections = stats.active_connections.saturating_sub(1);
        stats.total_connections = stats.total_connections.saturating_sub(1);
    }

    pub async fn close_all(&self) -> Result<()> {
        let mut connections = self.connections.write().await;

        for mut conn in connections.drain(..) {
            if let Err(e) = conn.close().await {
                tracing::error!("Error closing SQLite connection {}: {}", conn.id, e);
            }
        }

        let mut stats = self.statistics.write().await;
        stats.total_connections = 0;
        stats.active_connections = 0;
        stats.idle_connections = 0;

        Ok(())
    }

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

    fn get_waiting_requests_count(&self) -> usize {
        if let Ok(waiting) = self.waiting_requests.try_read() {
            *waiting
        } else {
            0
        }
    }
}

#[async_trait::async_trait]
impl ConnectionPool for SQLiteConnectionPool {
    async fn get_connection(&self) -> Result<Box<dyn crate::DatabaseConnection>> {
        let conn = self.get_connection_inner().await?;
        Ok(Box::new(conn))
    }

    async fn return_connection(
        &self,
        connection: Box<dyn crate::DatabaseConnection>,
    ) -> Result<()> {
        let was_active = connection.is_active();
        let id = connection.id().to_string();
        let db_type = connection.database_type();

        if db_type != crate::DatabaseType::SQLite {
            return Err(anyhow!("Wrong connection type for SQLite pool"));
        }

        if was_active {
            let mut connections = self.connections.write().await;
            let stat = SQLiteConnection {
                id,
                connection_info: ConnectionInfo {
                    database_path: self.database_path.clone(),
                    is_connected: false,
                    created_at: Instant::now(),
                    last_activity: Instant::now(),
                },
                is_active: false,
                transaction_depth: 0,
                #[cfg(feature = "sqlite")]
                conn: None,
            };
            connections.push(stat);
            let mut stats = self.statistics.write().await;
            stats.active_connections = stats.active_connections.saturating_sub(1);
            stats.idle_connections += 1;
        }

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

// ---------------------------------------------------------------------------
// DatabaseConnection trait impl
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
impl crate::DatabaseConnection for SQLiteConnection {
    fn id(&self) -> &str {
        &self.id
    }

    fn database_type(&self) -> DatabaseType {
        DatabaseType::SQLite
    }

    fn is_active(&self) -> bool {
        self.is_active
    }

    async fn execute_query(&self, query: &str, params: Vec<Value>) -> Result<QueryResult> {
        let start_time = Instant::now();

        #[cfg(feature = "sqlite")]
        {
            let query_owned = query.to_string();
            let params_owned = params;
            let conn_arc = self
                .conn
                .as_ref()
                .ok_or_else(|| anyhow!("Connection not open"))?
                .clone();

            let result = tokio::task::spawn_blocking(move || {
                let conn = conn_arc
                    .lock()
                    .map_err(|e| anyhow!("Mutex poisoned: {}", e))?;

                let rusqlite_params = prepare_params(&params_owned);
                let param_refs: Vec<&dyn rusqlite::types::ToSql> = rusqlite_params
                    .iter()
                    .map(|v| v as &dyn rusqlite::types::ToSql)
                    .collect();

                let mut stmt = conn.prepare(&query_owned)?;
                let column_count = stmt.column_count();
                let column_names: Vec<String> = (0..column_count)
                    .map(|i| stmt.column_name(i).unwrap_or("?").to_string())
                    .collect();

                let rows = stmt.query_map(param_refs.as_slice(), |row| {
                    let mut row_data = HashMap::new();
                    for i in 0..column_count {
                        let col_name = column_names[i].clone();
                        let raw: rusqlite::types::Value = row.get_unwrap(i);
                        row_data.insert(col_name, rusqlite_value_to_value(&raw));
                    }
                    Ok(Row { columns: row_data })
                })?;

                let mut result_rows = Vec::new();
                for row in rows {
                    result_rows.push(row?);
                }

                let affected_rows = conn.changes() as u64;

                Ok::<_, anyhow::Error>(QueryResult {
                    rows: result_rows,
                    affected_rows,
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                    query: query_owned,
                })
            })
            .await??;

            return Ok(result);
        }

        #[cfg(not(feature = "sqlite"))]
        Err(anyhow!(
            "SQLite support not enabled. Enable with 'sqlite' feature"
        ))
    }

    async fn close(&mut self) -> Result<()> {
        self.is_active = false;
        Ok(())
    }
}
