use tracing::info;

pub mod config;
pub mod handlers;
pub mod router;
pub mod tls;

pub struct NexoraServer {
    config: ServerConfig,
}

impl NexoraServer {
    pub fn new(config: ServerConfig) -> Self {
        Self { config }
    }

    pub async fn start(&self) -> Result<(), anyhow::Error> {
        info!("Starting Nexora server on {}:{}", self.config.host, self.config.port);
        Ok(())
    }
}

pub use config::{ServerConfig, load_rustls_pem_file, load_rustls_private_key};
pub use handlers::*;
pub use router::create_router;
pub use tls::start_tls_server;
