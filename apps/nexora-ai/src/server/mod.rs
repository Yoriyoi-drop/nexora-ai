use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tower::limit::ConcurrencyLimitLayer;
use tower_http::timeout::TimeoutLayer;
use tracing::info;

use crate::NexoraAI;

pub mod agent_handlers;
pub mod handlers;
pub mod router;
pub mod tls;

pub use crate::config::server::ServerConfig;
pub use handlers::*;
pub use router::create_router;
pub use tls::start_tls_server;
pub use tls::{load_rustls_pem_file, load_rustls_private_key};

pub struct NexoraServer {
    config: ServerConfig,
}

impl NexoraServer {
    pub fn new(config: ServerConfig) -> Self {
        Self { config }
    }

    pub async fn start(&self, nexora: Arc<NexoraAI>) -> Result<(), anyhow::Error> {
        let app = create_router(nexora, &self.config).await?;

        info!(
            "Enforcing connection limit ({}) and request timeout ({}s)",
            self.config.max_connections, self.config.request_timeout_seconds
        );
        let app = app
            .layer(ConcurrencyLimitLayer::new(self.config.max_connections))
            .layer(TimeoutLayer::new(Duration::from_secs(
                self.config.request_timeout_seconds,
            )));

        let addr: SocketAddr = format!("{}:{}", self.config.host, self.config.port).parse()?;
        let listener = TcpListener::bind(addr).await?;
        info!("Server listening on {}", addr);

        if self.config.enable_tls {
            start_tls_server(&self.config, listener, app).await?;
        } else {
            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown_signal())
                .await?;
        }
        Ok(())
    }
}

async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install signal handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
