//! Database Connection Pool with Advanced Features
//!
//! Implementasi connection pooling dengan query optimization dan monitoring

#![cfg(feature = "sqlx")]

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json;
use sqlx::{postgres::PgPoolOptions, Column, PgPool, Row};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tokio::task::JoinHandle;

mod serde_timestamp {
    use serde::{Deserializer, Serializer};
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(instant: &Instant, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let now = Instant::now();
        let since_epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let instant_as_system = since_epoch
            .checked_sub(now.duration_since(*instant))
            .unwrap_or(Duration::ZERO);
        serializer.serialize_u64(instant_as_system.as_secs())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Instant, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Deserialize;
        let secs: u64 = Deserialize::deserialize(deserializer)?;
        let now = Instant::now();
        let system_now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let elapsed_since_epoch = Duration::from_secs(secs);
        let age = system_now.saturating_sub(elapsed_since_epoch);
        Ok(now - age)
    }
}
use std::collections::HashMap;
use tracing::{debug, error, info, warn};

use nexora_common::retry::RetryConfig;

/// Database connection pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    /// Database URL
    pub database_url: String,
    /// Maximum number of connections
    pub max_connections: u32,
    /// Minimum number of connections
    pub min_connections: u32,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Idle timeout
    pub idle_timeout: Duration,
    /// Max lifetime for connections
    pub max_lifetime: Option<Duration>,
    /// Test query for connection health
    pub test_query: String,
    /// Enable query logging
    pub enable_query_logging: bool,
    /// Slow query threshold
    pub slow_query_threshold: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            database_url: std::env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://localhost/nexora".to_string()),
            max_connections: std::env::var("DB_MAX_CONNECTIONS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(20),
            min_connections: std::env::var("DB_MIN_CONNECTIONS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(5),
            connect_timeout: Duration::from_secs(
                std::env::var("DB_CONNECT_TIMEOUT")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(30),
            ),
            idle_timeout: Duration::from_secs(
                std::env::var("DB_IDLE_TIMEOUT")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(600),
            ),
            max_lifetime: std::env::var("DB_MAX_LIFETIME")
                .ok()
                .and_then(|v| v.parse().ok())
                .map(Duration::from_secs)
                .or(Some(Duration::from_secs(1800))),
            test_query: std::env::var("DB_TEST_QUERY").unwrap_or_else(|_| "SELECT 1".to_string()),
            enable_query_logging: std::env::var("DB_ENABLE_QUERY_LOGGING")
                .ok()
                .map(|v| v == "true" || v == "1")
                .unwrap_or(true),
            slow_query_threshold: Duration::from_millis(
                std::env::var("DB_SLOW_QUERY_THRESHOLD_MS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(500),
            ),
        }
    }
}

/// Advanced database pool with monitoring and optimization
pub struct DatabasePool {
    /// SQLx connection pool
    pool: PgPool,
    /// Pool configuration
    config: PoolConfig,
    /// Pool statistics
    stats: Arc<RwLock<PoolStatistics>>,
    /// Query cache
    query_cache: Arc<RwLock<HashMap<String, CachedQuery>>>,
    /// Slow query logger
    slow_query_logger: Arc<Mutex<Vec<SlowQueryRecord>>>,
    /// Health checker
    health_checker: Arc<HealthChecker>,
    /// Tracked background task handles for supervision
    background_handles: Arc<std::sync::Mutex<Vec<JoinHandle<()>>>>,
}

/// Pool statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolStatistics {
    /// Total connections
    pub total_connections: u32,
    /// Active connections
    pub active_connections: u32,
    /// Idle connections
    pub idle_connections: u32,
    /// Total queries executed
    pub total_queries: u64,
    /// Queries per second
    pub queries_per_second: f64,
    /// Average query time
    pub avg_query_time: Duration,
    /// Slow queries count
    pub slow_queries: u64,
    /// Failed queries count
    pub failed_queries: u64,
    /// Connection errors count
    pub connection_errors: u64,
    /// Last reset time
    #[serde(with = "serde_timestamp")]
    pub last_reset: Instant,
}

impl Default for PoolStatistics {
    fn default() -> Self {
        Self {
            total_connections: 0,
            active_connections: 0,
            idle_connections: 0,
            total_queries: 0,
            queries_per_second: 0.0,
            avg_query_time: Duration::ZERO,
            slow_queries: 0,
            failed_queries: 0,
            connection_errors: 0,
            last_reset: Instant::now(),
        }
    }
}

/// Cached query result — stores serialized JSON rows
#[derive(Debug)]
struct CachedQuery {
    query_hash: u64,
    rows_json: Vec<serde_json::Value>,
    timestamp: Instant,
    ttl: Duration,
}

/// Slow query record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlowQueryRecord {
    pub query: String,
    pub duration: Duration,
    #[serde(with = "serde_timestamp")]
    pub timestamp: Instant,
    pub parameters: Option<serde_json::Value>,
    pub error: Option<String>,
}

/// Health checker for connections
struct HealthChecker {
    pub(crate) test_query: String,
    interval: Duration,
    last_check: Arc<Mutex<Instant>>,
    is_healthy: Arc<RwLock<bool>>,
}

impl DatabasePool {
    /// Create new database pool
    pub async fn new(config: PoolConfig) -> Result<Self> {
        info!(
            "Creating database pool with max {} connections",
            config.max_connections
        );

        // Create SQLx pool with retry
        let pool = retry_connect(&config, 3).await?;

        // Initialize components
        let stats = Arc::new(RwLock::new(PoolStatistics::default()));
        let query_cache = Arc::new(RwLock::new(HashMap::new()));
        let slow_query_logger = Arc::new(Mutex::new(Vec::new()));

        let health_checker = Arc::new(HealthChecker {
            test_query: config.test_query.clone(),
            interval: Duration::from_secs(30),
            last_check: Arc::new(Mutex::new(Instant::now())),
            is_healthy: Arc::new(RwLock::new(true)),
        });

        let db_pool = Self {
            pool,
            config,
            stats,
            query_cache,
            slow_query_logger,
            health_checker,
            background_handles: Arc::new(std::sync::Mutex::new(Vec::new())),
        };

        // Start background tasks
        db_pool.start_background_tasks().await?;

        info!("Database pool created successfully");
        Ok(db_pool)
    }

    /// Execute query with monitoring and caching
    pub async fn execute_query(&self, query: &str) -> Result<Vec<serde_json::Value>> {
        let start_time = Instant::now();

        // Check cache for SELECT queries
        if self.is_select_query(query) {
            if let Some(cached) = self.check_query_cache(query).await? {
                debug!("Query cache hit for: {}", query);
                return Ok(cached);
            }
        }

        // Execute query with retry
        let pool = self.pool.clone();
        let query_string = query.to_string();
        let result = RetryConfig::default()
            .retry(|| {
                let pool = pool.clone();
                let query_string = query_string.clone();
                async move { sqlx::query(&query_string).fetch_all(&pool).await }
            })
            .await;

        let duration = start_time.elapsed();

        match result {
            Ok(rows) => {
                // Update statistics
                self.update_query_stats(duration, false).await;

                // Log slow queries
                if duration > self.config.slow_query_threshold {
                    self.log_slow_query(query, duration, None, None).await;
                }

                // Convert PgRow to JSON for return value
                let json_rows: Vec<serde_json::Value> = rows
                    .iter()
                    .map(|row| {
                        let cols: Vec<String> =
                            row.columns().iter().map(|c| c.name().to_string()).collect();
                        let mut map = serde_json::Map::new();
                        for col in cols {
                            let val: serde_json::Value =
                                row.try_get(col.as_str()).unwrap_or(serde_json::Value::Null);
                            map.insert(col, val);
                        }
                        serde_json::Value::Object(map)
                    })
                    .collect();

                // Cache SELECT queries
                if self.is_select_query(query) {
                    self.cache_query(query, &rows).await?;
                }

                Ok(json_rows)
            }
            Err(e) => {
                // Update error statistics
                self.update_query_stats(duration, true).await;

                // Log failed query
                self.log_slow_query(query, duration, Some(e.to_string()), None)
                    .await;

                Err(anyhow!("Query execution failed: {}", e))
            }
        }
    }

    /// Execute query with prepared statement (legacy)
    #[deprecated(note = "Use direct sqlx::query().bind() instead for parameterized queries")]
    pub async fn execute_prepared(&self, query: &str) -> Result<Vec<sqlx::postgres::PgRow>> {
        warn!("execute_prepared is deprecated - use sqlx::query().bind() instead to prevent SQL injection");
        let start_time = Instant::now();

        let query_builder = sqlx::query(query);
        let result = query_builder.fetch_all(&self.pool).await;
        let duration = start_time.elapsed();

        match result {
            Ok(rows) => {
                self.update_query_stats(duration, false).await;

                if duration > self.config.slow_query_threshold {
                    self.log_slow_query(query, duration, None, None).await;
                }

                Ok(rows)
            }
            Err(e) => {
                self.update_query_stats(duration, true).await;
                self.log_slow_query(query, duration, Some(e.to_string()), None)
                    .await;
                Err(anyhow!("Prepared query execution failed: {}", e))
            }
        }
    }

    /// Execute transaction
    pub async fn execute_transaction(&self) -> Result<()> {
        let start_time = Instant::now();

        let tx = self.pool.begin().await?;

        // Simple transaction for now
        tx.commit().await?;

        let duration = start_time.elapsed();
        if duration > self.config.slow_query_threshold {
            self.log_slow_query("TRANSACTION", duration, None, None)
                .await;
        }

        Ok(())
    }

    /// Check pool health
    pub async fn health_check(&self) -> Result<bool> {
        let is_healthy = self.health_checker.is_healthy.read().await;
        Ok(*is_healthy)
    }

    /// Get slow queries
    pub async fn get_slow_queries(&self, limit: usize) -> Vec<SlowQueryRecord> {
        let logger = self.slow_query_logger.lock().await;
        logger.iter().rev().take(limit).cloned().collect()
    }

    /// Clear query cache
    pub async fn clear_cache(&self) -> Result<()> {
        let mut cache = self.query_cache.write().await;
        cache.clear();
        info!("Query cache cleared");
        Ok(())
    }

    /// Optimize database
    pub async fn optimize(&self) -> Result<OptimizationReport> {
        info!("Starting database optimization");

        let mut report = OptimizationReport::new();

        // Analyze semua table dalam 1 query (JOIN) — fix N+1
        let all_stats = self.analyze_all_tables().await?;
        for (table, stats) in all_stats {
            report.add_table_stats(table, stats);
        }

        // Update statistics
        self.update_database_statistics().await?;

        // Check for unused indexes
        let unused_indexes = self.find_unused_indexes().await?;
        report.set_unused_indexes(unused_indexes);

        // Vacuum analyze
        self.vacuum_analyze().await?;

        info!("Database optimization completed");
        Ok(report)
    }

    /// Private helper methods

    fn is_select_query(&self, query: &str) -> bool {
        let query_lower = query.trim().to_lowercase();
        query_lower.starts_with("select") && !query_lower.contains("for update")
    }

    async fn check_query_cache(&self, query: &str) -> Result<Option<Vec<serde_json::Value>>> {
        let cache = self.query_cache.read().await;

        if let Some(cached) = cache.get(query) {
            if cached.timestamp.elapsed() < cached.ttl {
                debug!("Query cache hit for: {}", query);
                return Ok(Some(cached.rows_json.clone()));
            }
        }

        Ok(None)
    }

    /// Cache SELECT query results as serialized JSON values.
    /// Serializes PgRow to JSON for storage since PgRow doesn't implement Clone.
    async fn cache_query(&self, query: &str, rows: &[sqlx::postgres::PgRow]) -> Result<()> {
        let mut cache = self.query_cache.write().await;

        // LRU eviction if cache exceeds limit
        if cache.len() >= 1000 {
            let oldest_key = cache
                .iter()
                .min_by_key(|(_, v)| v.timestamp)
                .map(|(k, _)| k.clone());
            if let Some(key) = oldest_key {
                cache.remove(&key);
            }
        }

        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        let query_hash = hasher.finish();

        // Serialize rows to JSON for cache storage
        let rows_json: Vec<serde_json::Value> = rows
            .iter()
            .map(|row| {
                let cols: Vec<String> =
                    row.columns().iter().map(|c| c.name().to_string()).collect();
                let mut map = serde_json::Map::new();
                for col in cols {
                    let val: serde_json::Value =
                        row.try_get(col.as_str()).unwrap_or(serde_json::Value::Null);
                    map.insert(col, val);
                }
                serde_json::Value::Object(map)
            })
            .collect();

        cache.insert(
            query.to_string(),
            CachedQuery {
                query_hash,
                rows_json,
                timestamp: Instant::now(),
                ttl: Duration::from_secs(300), // 5 minutes TTL
            },
        );

        Ok(())
    }

    async fn update_query_stats(&self, duration: Duration, failed: bool) {
        let mut stats = self.stats.write().await;

        stats.total_queries += 1;

        if failed {
            stats.failed_queries += 1;
        }

        // Update average query time
        let total_time_ms = stats.avg_query_time.as_millis() as u64 * (stats.total_queries - 1)
            + duration.as_millis() as u64;
        stats.avg_query_time = Duration::from_millis(total_time_ms / stats.total_queries);

        // Update queries per second
        let elapsed = stats.last_reset.elapsed();
        if elapsed > Duration::ZERO {
            stats.queries_per_second = stats.total_queries as f64 / elapsed.as_secs_f64();
        }

        if duration > self.config.slow_query_threshold {
            stats.slow_queries += 1;
        }

        // Update connection stats
        let pool_size = self.pool.size();
        let idle = self.pool.num_idle();
        stats.total_connections = pool_size as u32;
        stats.idle_connections = idle as u32;
        stats.active_connections = stats.total_connections - stats.idle_connections;
    }

    async fn log_slow_query(
        &self,
        query: &str,
        duration: Duration,
        error: Option<String>,
        parameters: Option<serde_json::Value>,
    ) {
        if !self.config.enable_query_logging {
            return;
        }

        let record = SlowQueryRecord {
            query: query.to_string(),
            duration,
            timestamp: Instant::now(),
            parameters,
            error,
        };

        let mut logger = self.slow_query_logger.lock().await;
        logger.push(record);

        // Keep only last 1000 slow queries
        if logger.len() > 1000 {
            logger.remove(0);
        }

        warn!("Slow query detected: {} took {:?}", query, duration);
    }

    async fn start_background_tasks(&self) -> Result<()> {
        // Start health checker
        let health_checker = self.health_checker.clone();
        let pool = self.pool.clone();
        let test_query = self.config.test_query.clone();

        let handle = tokio::spawn(async move {
            let fut = std::panic::AssertUnwindSafe(async move {
                let mut interval = tokio::time::interval(health_checker.interval);

                loop {
                    interval.tick().await;

                    let start = Instant::now();
                    let result = sqlx::query_scalar::<_, i32>(&test_query)
                        .fetch_one(&pool)
                        .await;
                    let duration = start.elapsed();

                    let is_healthy = result.is_ok() && duration < Duration::from_secs(5);

                    let mut last_check = health_checker.last_check.lock().await;
                    *last_check = Instant::now();
                    drop(last_check);

                    let mut health_status = health_checker.is_healthy.write().await;
                    *health_status = is_healthy;
                    drop(health_status);

                    if !is_healthy {
                        error!("Database health check failed");
                    }
                }
            });
            if let Err(e) = futures::future::FutureExt::catch_unwind(fut).await {
                error!("Database health check loop panicked: {:?}", e);
            }
        });
        if let Ok(mut handles) = self.background_handles.lock() {
            handles.push(handle);
        }

        // Start statistics reset
        let stats = self.stats.clone();
        let handle2 = tokio::spawn(async move {
            let fut = std::panic::AssertUnwindSafe(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(60));

                loop {
                    interval.tick().await;

                    let mut stats_guard = stats.write().await;
                    stats_guard.last_reset = Instant::now();
                }
            });
            if let Err(e) = futures::future::FutureExt::catch_unwind(fut).await {
                error!("Database stats reset loop panicked: {:?}", e);
            }
        });
        if let Ok(mut handles) = self.background_handles.lock() {
            handles.push(handle2);
        }

        Ok(())
    }

    async fn get_table_names(&self) -> Result<Vec<String>> {
        let rows = self
            .execute_query("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            .await?;

        Ok(rows
            .iter()
            .filter_map(|row| row["tablename"].as_str().map(String::from))
            .collect())
    }

    /// Analyze semua tables dalam 1 JOIN query — fix N+1
    async fn analyze_all_tables(&self) -> Result<Vec<(String, TableStatistics)>> {
        let rows = self
            .execute_query(
                "SELECT p.tablename,
                        COALESCE(s.n_tup_ins, 0) as inserts,
                        COALESCE(s.n_tup_upd, 0) as updates,
                        COALESCE(s.n_tup_del, 0) as deletes,
                        COALESCE(s.n_live_tup, 0) as live_tuples,
                        COALESCE(s.n_dead_tup, 0) as dead_tuples
                 FROM pg_tables p
                 LEFT JOIN pg_stat_user_tables s ON s.tablename = p.tablename
                 WHERE p.schemaname = 'public'",
            )
            .await?;

        let stats = rows
            .iter()
            .filter_map(|row| {
                let table_name = row["tablename"].as_str().map(String::from);
                table_name.map(|name| {
                    let stats = TableStatistics {
                        table_name: name.clone(),
                        inserts: row["inserts"].as_i64().unwrap_or(0),
                        updates: row["updates"].as_i64().unwrap_or(0),
                        deletes: row["deletes"].as_i64().unwrap_or(0),
                        live_tuples: row["live_tuples"].as_i64().unwrap_or(0),
                        dead_tuples: row["dead_tuples"].as_i64().unwrap_or(0),
                    };
                    (name, stats)
                })
            })
            .collect();

        Ok(stats)
    }

    async fn analyze_table(&self, table: &str) -> Result<TableStatistics> {
        let query = "SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del, n_live_tup, n_dead_tup \
                     FROM pg_stat_user_tables WHERE tablename = $1";
        let row = sqlx::query(query)
            .bind(table)
            .fetch_optional(&self.pool)
            .await?;

        if let Some(row) = row {
            Ok(TableStatistics {
                table_name: table.to_string(),
                inserts: row
                    .try_get::<Option<i64>, _>("n_tup_ins")
                    .unwrap_or(None)
                    .unwrap_or(0),
                updates: row
                    .try_get::<Option<i64>, _>("n_tup_upd")
                    .unwrap_or(None)
                    .unwrap_or(0),
                deletes: row
                    .try_get::<Option<i64>, _>("n_tup_del")
                    .unwrap_or(None)
                    .unwrap_or(0),
                live_tuples: row
                    .try_get::<Option<i64>, _>("n_live_tup")
                    .unwrap_or(None)
                    .unwrap_or(0),
                dead_tuples: row
                    .try_get::<Option<i64>, _>("n_dead_tup")
                    .unwrap_or(None)
                    .unwrap_or(0),
            })
        } else {
            Ok(TableStatistics::default(table))
        }
    }

    async fn update_database_statistics(&self) -> Result<()> {
        self.execute_query("ANALYZE").await?;
        Ok(())
    }

    async fn find_unused_indexes(&self) -> Result<Vec<String>> {
        let rows = self
            .execute_query(
                "SELECT schemaname, tablename, indexname 
             FROM pg_stat_user_indexes 
             WHERE idx_scan = 0 
             AND schemaname = 'public'",
            )
            .await?;

        Ok(rows
            .iter()
            .map(|row| {
                let schema = row["schemaname"].as_str().unwrap_or("public").to_string();
                let table = row["tablename"].as_str().unwrap_or("unknown").to_string();
                let index = row["indexname"].as_str().unwrap_or("unknown").to_string();
                format!("{}.{}.{}", schema, table, index)
            })
            .collect())
    }

    async fn vacuum_analyze(&self) -> Result<()> {
        self.execute_query("VACUUM ANALYZE").await?;
        Ok(())
    }
}

/// Table statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableStatistics {
    pub table_name: String,
    pub inserts: i64,
    pub updates: i64,
    pub deletes: i64,
    pub live_tuples: i64,
    pub dead_tuples: i64,
}

impl TableStatistics {
    fn default(table: &str) -> Self {
        Self {
            table_name: table.to_string(),
            inserts: 0,
            updates: 0,
            deletes: 0,
            live_tuples: 0,
            dead_tuples: 0,
        }
    }
}

/// Optimization report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationReport {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub table_stats: HashMap<String, TableStatistics>,
    pub unused_indexes: Vec<String>,
    pub recommendations: Vec<String>,
}

impl OptimizationReport {
    pub fn new() -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            table_stats: HashMap::new(),
            unused_indexes: Vec::new(),
            recommendations: Vec::new(),
        }
    }

    pub fn add_table_stats(&mut self, table: String, stats: TableStatistics) {
        self.table_stats.insert(table, stats);
    }

    pub fn set_unused_indexes(&mut self, indexes: Vec<String>) {
        self.unused_indexes = indexes;

        // Generate recommendations
        if !self.unused_indexes.is_empty() {
            self.recommendations.push(format!(
                "Consider dropping {} unused indexes",
                self.unused_indexes.len()
            ));
        }

        // Check for tables with many dead tuples
        for (table, stats) in &self.table_stats {
            if stats.dead_tuples > stats.live_tuples * 2 {
                self.recommendations.push(format!(
                    "Table {} has many dead tuples, consider VACUUM",
                    table
                ));
            }
        }
    }
}

impl HealthChecker {
    pub(crate) async fn check_health(&self, pool: &PgPool) -> bool {
        let start = Instant::now();
        sqlx::query(&self.test_query).fetch_one(pool).await.is_ok()
            && start.elapsed() < Duration::from_secs(5)
    }
}

impl DatabasePool {
    /// Check pool health status. Returns true if the pool is responsive.
    pub async fn health_check(&self) -> bool {
        self.health_checker.check_health(&self.pool).await
    }
}

async fn retry_connect(config: &PoolConfig, max_retries: u32) -> Result<PgPool> {
    let mut last_err = anyhow::anyhow!("Connection failed after {} retries", max_retries);
    for attempt in 1..=max_retries {
        match PgPoolOptions::new()
            .max_connections(config.max_connections)
            .min_connections(config.min_connections)
            .acquire_timeout(config.connect_timeout)
            .idle_timeout(config.idle_timeout)
            .max_lifetime(config.max_lifetime)
            .test_before_acquire(true)
            .after_connect(|conn, _meta| {
                Box::pin(async move {
                    sqlx::query("SELECT 1").execute(conn).await?;
                    Ok(())
                })
            })
            .connect(&config.database_url)
            .await
        {
            Ok(pool) => return Ok(pool),
            Err(e) => {
                tracing::warn!(
                    "DB connection attempt {}/{} failed: {}",
                    attempt,
                    max_retries,
                    e
                );
                last_err = e.into();
                if attempt < max_retries {
                    tokio::time::sleep(tokio::time::Duration::from_millis(
                        100 * 2u64.pow(attempt - 1),
                    ))
                    .await;
                }
            }
        }
    }
    Err(last_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_config_default() {
        let config = PoolConfig::default();
        assert_eq!(config.max_connections, 20);
        assert_eq!(config.min_connections, 5);
    }

    #[test]
    fn test_pool_statistics_default() {
        let stats = PoolStatistics::default();
        assert_eq!(stats.total_queries, 0);
        assert_eq!(stats.failed_queries, 0);
    }

    #[test]
    fn test_table_statistics() {
        let stats = TableStatistics::default("test_table");
        assert_eq!(stats.table_name, "test_table");
        assert_eq!(stats.inserts, 0);
    }

    #[test]
    fn test_optimization_report() {
        let report = OptimizationReport::new();
        assert!(report.table_stats.is_empty());
        assert!(report.recommendations.is_empty());
    }
}
