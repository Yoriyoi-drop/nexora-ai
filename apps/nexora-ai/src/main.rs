#[cfg(feature = "blas")]
extern crate blas_src;
#[cfg(feature = "blas")]
extern crate openblas_src;

use clap::Parser;
use nexora_ai::cli::Cli;
use tracing::{error, info};

#[tokio::main]
async fn main() {
    std::panic::set_hook(Box::new(|panic_info| {
        use std::io::Write;
        let _ = writeln!(std::io::stderr(), "PANIC: {}", panic_info);
        let bt = std::backtrace::Backtrace::force_capture();
        let _ = writeln!(std::io::stderr(), "BACKTRACE:\n{:?}", bt);
        let _ = std::io::stderr().flush();
    }));

    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) => {
            eprintln!("CLI parsing error: {}", e);
            std::process::exit(1);
        }
    };

    if let Err(e) = cli.run().await {
        error!("Application error: {}", e);
        error!(
            "Error code: {}, HTTP status: {}",
            e.error_code(),
            e.http_status()
        );
        std::process::exit(1);
    }

    info!("Nexora AI system shutdown gracefully");
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use nexora_ai::cli::Commands;

    #[tokio::test]
    async fn test_main_function_parsing() {
        let args = vec!["nexora-cli", "health"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[tokio::test]
    async fn test_main_error_handling() {
        let args = vec!["nexora-cli", "invalid_command"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_err());
    }

    #[tokio::test]
    async fn test_main_with_all_commands() {
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
        let args = vec!["nexora-cli", "--config", "nexora.toml", "health"];
        let cli = Cli::try_parse_from(args).expect("health command should parse successfully");
        assert!(matches!(cli.command, Commands::Health { .. }));
    }
}
