//! Connection Pool - Rust implementation
//!
//! Generic connection pool for database connections

use anyhow::Result;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use crate::{DatabaseConnection, PoolStatus};

/// Connection pool configuration
#[derive(Debug, Clone)]
pub struct ConnectionPoolConfig {
    pub max_connections: usize,
    pub min_connections: usize,
    pub connection_timeout: Duration,
    pub idle_timeout: Duration,
    pub max_lifetime: Duration,
}

/// Generic connection pool trait
#[async_trait::async_trait]
pub trait ConnectionPool: Send + Sync {
    /// Get connection from pool
    async fn get_connection(&self) -> Result<Box<dyn DatabaseConnection>>;

    /// Return connection to pool
    async fn return_connection(&self, connection: Box<dyn DatabaseConnection>) -> Result<()>;

    /// Get pool status
    async fn get_status(&self) -> Result<PoolStatus>;

    /// Close all connections
    async fn close_all(&self) -> Result<()>;
}

/// Generic connection pool implementation
pub struct GenericConnectionPool<T> {
    connections: Arc<RwLock<Vec<PooledConnection<T>>>>,
    config: ConnectionPoolConfig,
    statistics: Arc<RwLock<PoolStatistics>>,
    connection_factory: Arc<dyn ConnectionFactory<T>>,
    waiting_requests: Arc<RwLock<usize>>, // Track number of waiting requests
}

/// Pooled connection wrapper
#[derive(Debug)]
struct PooledConnection<T> {
    connection: T,
    created_at: Instant,
    last_used: Instant,
    is_active: bool,
    usage_count: u64,
}

/// A ticket that preserves connection timing through the get/return cycle
#[derive(Debug)]
pub struct ConnectionTicket<T> {
    pub connection: T,
    pub created_at: Instant,
    pub usage_count: u64,
}

/// Connection factory trait
#[async_trait::async_trait]
pub trait ConnectionFactory<T>: Send + Sync {
    /// Create new connection
    async fn create_connection(&self) -> Result<T>;

    /// Validate connection
    async fn validate_connection(&self, connection: &T) -> bool;

    /// Close connection
    async fn close_connection(&self, connection: T) -> Result<()>;
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
    validation_failures: u64,
}

impl<T: Clone> GenericConnectionPool<T> {
    /// Create new generic connection pool
    pub async fn new<F>(config: ConnectionPoolConfig, factory: F) -> Result<Self>
    where
        F: ConnectionFactory<T> + 'static,
    {
        let connections = Arc::new(RwLock::new(Vec::new()));
        let statistics = Arc::new(RwLock::new(PoolStatistics::default()));
        let connection_factory = Arc::new(factory);
        let waiting_requests = Arc::new(RwLock::new(0));

        // Create minimum connections
        let mut conn_vec = Vec::with_capacity(config.min_connections);
        for _ in 0..config.min_connections {
            match connection_factory.create_connection().await {
                Ok(connection) => {
                    let pooled = PooledConnection {
                        connection,
                        created_at: Instant::now(),
                        last_used: Instant::now(),
                        is_active: false,
                        usage_count: 0,
                    };
                    conn_vec.push(pooled);
                }
                Err(e) => {
                    tracing::warn!("Failed to create initial connection: {}", e);
                }
            }
        }

        {
            *connections.write().await = conn_vec;
        }

        {
            let mut stats = statistics.write().await;
            stats.total_connections = connections.read().await.len();
            stats.idle_connections = connections.read().await.len();
        }

        Ok(Self {
            connections,
            config,
            statistics,
            connection_factory,
            waiting_requests,
        })
    }

    /// Get connection from pool
    pub async fn get_connection(&self) -> Result<ConnectionTicket<T>> {
        let start_time = Instant::now();

        // Increment waiting requests counter
        *self.waiting_requests.write().await += 1;

        // Try to get existing connection
        {
            let mut connections = self.connections.write().await;

            // Find idle connection
            if let Some(pos) = connections.iter().position(|c| !c.is_active) {
                let mut pooled = connections.swap_remove(pos);

                // Validate connection
                if self
                    .connection_factory
                    .validate_connection(&pooled.connection)
                    .await
                {
                    let created_at = pooled.created_at;
                    let usage_count = pooled.usage_count;

                    pooled.is_active = true;
                    pooled.last_used = Instant::now();
                    pooled.usage_count += 1;

                    // Update statistics
                    let mut stats = self.statistics.write().await;
                    stats.idle_connections -= 1;
                    stats.active_connections += 1;

                    // Decrement waiting requests counter
                    *self.waiting_requests.write().await -= 1;

                    let wait_time = start_time.elapsed();
                    stats.total_wait_time_ms += wait_time.as_millis() as u64;
                    stats.average_wait_time_ms = if stats.total_connections > 0 {
                        stats.total_wait_time_ms as f64 / stats.total_connections as f64
                    } else {
                        0.0
                    };

                    return Ok(ConnectionTicket {
                        connection: pooled.connection,
                        created_at,
                        usage_count,
                    });
                } else {
                    // Connection is invalid, remove it
                    if let Err(e) = self
                        .connection_factory
                        .close_connection(pooled.connection)
                        .await
                    {
                        tracing::error!("Failed to close invalid connection: {}", e);
                    }

                    let mut stats = self.statistics.write().await;
                    stats.validation_failures += 1;
                    stats.total_connections -= 1;
                }
            }
        }

        // Create new connection if under max
        {
            let connections = self.connections.read().await;
            if connections.len() < self.config.max_connections {
                drop(connections);

                match self.connection_factory.create_connection().await {
                    Ok(connection) => {
                        let created_at = Instant::now();
                        let pooled = PooledConnection {
                            connection,
                            created_at,
                            last_used: Instant::now(),
                            is_active: true,
                            usage_count: 1,
                        };

                        let mut connections = self.connections.write().await;
                        connections.push(pooled);

                        // Update statistics
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

                        // Decrement waiting requests counter
                        *self.waiting_requests.write().await -= 1;

                        let last = connections.last_mut()
                            .expect("connection was just added");
                        let conn_clone = last.connection.clone();
                        return Ok(ConnectionTicket {
                            connection: conn_clone,
                            created_at: last.created_at,
                            usage_count: last.usage_count,
                        });
                    }
                    Err(e) => {
                        let mut stats = self.statistics.write().await;
                        stats.connection_errors += 1;
                        return Err(anyhow::anyhow!("Failed to create connection: {}", e));
                    }
                }
            }
        }

        // Wait for available connection (with timeout)
        let timeout = self.config.connection_timeout;
        let elapsed = start_time.elapsed();

        if elapsed < timeout {
            tokio::time::sleep(Duration::from_millis(50)).await;
            return self.get_connection().await;
        }

        Err(anyhow::anyhow!(
            "Connection pool timeout after {:?}",
            elapsed
        ))
    }

    /// Return connection to pool with preserved creation time
    pub async fn return_connection(&self, ticket: ConnectionTicket<T>) -> Result<()> {
        let mut connections = self.connections.write().await;

        // Check if connection is still valid
        if self
            .connection_factory
            .validate_connection(&ticket.connection)
            .await
        {
            let pooled = PooledConnection {
                connection: ticket.connection,
                created_at: ticket.created_at,
                last_used: Instant::now(),
                is_active: false,
                usage_count: ticket.usage_count,
            };

            connections.push(pooled);

            // Update statistics
            let mut stats = self.statistics.write().await;
            stats.active_connections -= 1;
            stats.idle_connections += 1;
        } else {
            // Close invalid connection
            if let Err(e) = self.connection_factory.close_connection(ticket.connection).await {
                tracing::error!("Failed to close invalid connection: {}", e);
            }

            let mut stats = self.statistics.write().await;
            stats.validation_failures += 1;
            stats.total_connections -= 1;
        }

        Ok(())
    }

    /// Close all connections
    pub async fn close_all(&self) -> Result<()> {
        let mut connections = self.connections.write().await;

        for pooled in connections.drain(..) {
            if let Err(e) = self
                .connection_factory
                .close_connection(pooled.connection)
                .await
            {
                tracing::error!("Failed to close connection: {}", e);
            }
        }

        // Update statistics
        let mut stats = self.statistics.write().await;
        stats.total_connections = 0;
        stats.active_connections = 0;
        stats.idle_connections = 0;

        Ok(())
    }

    /// Get pool status
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
        // This is a synchronous method that reads the current value
        // In a real implementation, you might want to make this async
        // but for now we'll use a blocking read with a timeout

        if let Ok(waiting) = self.waiting_requests.try_read() {
            *waiting
        } else {
            // Fallback if lock is contended
            0
        }
    }

    /// Cleanup expired connections
    pub async fn cleanup_expired(&self) -> Result<usize> {
        let mut connections = self.connections.write().await;
        let mut removed_count = 0;

        // Collect expired connections first
        let expired_connections: Vec<_> = connections
            .iter()
            .filter(|pooled| {
                pooled.created_at.elapsed() > self.config.max_lifetime
                    || pooled.last_used.elapsed() > self.config.idle_timeout
            })
            .filter(|pooled| !pooled.is_active)
            .map(|pooled| pooled)
            .collect();

        // Close expired connections
        for pooled in expired_connections {
            if let Err(e) = self
                .connection_factory
                .close_connection(pooled.connection.clone())
                .await
            {
                tracing::error!("Failed to close expired connection: {}", e);
            }
            removed_count += 1;
        }

        // Remove expired connections
        connections.retain(|pooled| {
            !(pooled.created_at.elapsed() > self.config.max_lifetime
                || pooled.last_used.elapsed() > self.config.idle_timeout)
                || pooled.is_active
        });

        // Update statistics
        let mut stats = self.statistics.write().await;
        stats.total_connections = connections.len();
        stats.idle_connections = connections.iter().filter(|c| !c.is_active).count();
        stats.destroyed_connections += removed_count as u64;

        Ok(removed_count)
    }

    /// Maintain minimum connections
    pub async fn maintain_minimum(&self) -> Result<usize> {
        let mut connections = self.connections.write().await;
        let current_count = connections.len();
        let target_count = self.config.min_connections;

        if current_count < target_count {
            let mut created_count = 0;

            for _ in current_count..target_count {
                match self.connection_factory.create_connection().await {
                    Ok(connection) => {
                        let pooled = PooledConnection {
                            connection,
                            created_at: Instant::now(),
                            last_used: Instant::now(),
                            is_active: false,
                            usage_count: 0,
                        };
                        connections.push(pooled);
                        created_count += 1;
                    }
                    Err(e) => {
                        tracing::warn!("Failed to create maintenance connection: {}", e);
                        break;
                    }
                }
            }

            // Update statistics
            let mut stats = self.statistics.write().await;
            stats.total_connections = connections.len();
            stats.idle_connections = connections.iter().filter(|c| !c.is_active).count();
            stats.created_connections += created_count as u64;

            Ok(created_count)
        } else {
            Ok(0)
        }
    }
}

impl Default for ConnectionPoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 10,
            min_connections: 2,
            connection_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(300),
            max_lifetime: Duration::from_secs(3600),
        }
    }
}
