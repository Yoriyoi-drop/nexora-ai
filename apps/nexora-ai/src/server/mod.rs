use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::info;

use crate::NexoraAI;

pub mod handlers;
pub mod router;
pub mod tls;

pub use crate::config::server::ServerConfig;
pub use tls::{load_rustls_pem_file, load_rustls_private_key};
pub use handlers::*;
pub use router::create_router;
pub use tls::start_tls_server;

pub struct NexoraServer {
    config: ServerConfig,
}

impl NexoraServer {
    pub fn new(config: ServerConfig) -> Self {
        Self { config }
    }

    pub async fn start(&self, nexora: Arc<NexoraAI>) -> Result<(), anyhow::Error> {
        let app = create_router(nexora, &self.config).await?;
        let addr: SocketAddr = format!("{}:{}", self.config.host, self.config.port).parse()?;
        let listener = TcpListener::bind(addr).await?;
        info!("Server listening on {}", addr);

        if self.config.enable_tls {
            start_tls_server(&self.config, listener, app).await?;
        } else {
            axum::serve(listener, app).await?;
        }
        Ok(())
    }
}
