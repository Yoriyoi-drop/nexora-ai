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
