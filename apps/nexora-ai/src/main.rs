//! Main entry point untuk Nexora AI
//! 
//! Aplikasi utama yang menjalankan Nexora AI system dengan Rust implementation

use nexora_ai::cli::Cli;
use nexora_ai::error::{NexoraError, NexoraResult};
use tracing::{error, info};
use clap::Parser;

#[tokio::main]
async fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_target(false)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    info!("🚀 Starting Nexora AI system...");

    // Parse command line arguments
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) => {
            eprintln!("❌ CLI parsing error: {}", e);
            std::process::exit(1);
        }
    };
    
    // Run the CLI application
    if let Err(e) = cli.run().await {
        error!("❌ Application error: {}", e);
        error!("Error code: {}, HTTP status: {}", e.error_code(), e.http_status());
        std::process::exit(1);
    }
    
    info!("✅ Nexora AI system shutdown gracefully");
}


#[cfg(test)]
mod tests {
    use super::*;
    use nexora_ai::cli::Commands;
    use clap::Parser;

    #[tokio::test]
    async fn test_main_function_parsing() {
        // Test basic CLI parsing in main context
        let args = vec!["nexora-cli", "health"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[tokio::test]
    async fn test_main_error_handling() {
        // Test invalid command
        let args = vec!["nexora-cli", "invalid_command"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_err());
    }

    #[tokio::test]
    async fn test_main_with_all_commands() {
        // Test parsing all available commands
        let test_commands = vec![
            vec!["nexora-cli", "health"],
            vec!["nexora-cli", "info"],
            vec!["nexora-cli", "start"],
        ];
        
        for args in &test_commands {
            let cli = Cli::try_parse_from(args);
            assert!(cli.is_ok(), "Command {:?} should parse successfully", args);
        }
    }

    #[test]
    fn test_main_structural_integrity() {
        // Test the structural integrity of main.rs
        let args = vec!["nexora-cli", "--config", "nexora.toml", "health"];
        let cli = Cli::try_parse_from(args).unwrap();
        assert!(matches!(cli.command, Commands::Health { .. }));
    }
}
