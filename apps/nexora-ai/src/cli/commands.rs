//! Command definitions for Nexora-AI CLI

use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Nexora AI - Advanced Language Model System
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Configuration file path
    #[arg(short, long, default_value = "nexora.toml")]
    #[arg(help = "Path to configuration file")]
    pub config: PathBuf,

    /// Log level (trace, debug, info, warn, error)
    #[arg(short, long, default_value = "info")]
    #[arg(help = "Set logging level")]
    pub log_level: String,

    /// Enable verbose output
    #[arg(short, long)]
    #[arg(help = "Enable verbose output")]
    pub verbose: bool,

    /// Subcommand to execute
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Start the Nexora AI server
    #[command(aliases = &["dev", "serve"])]
    Start {
        /// Host to bind to
        #[arg(short = 'H', long, default_value = "127.0.0.1")]
        host: String,

        /// Port to bind to
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Enable TLS
        #[arg(long)]
        tls: bool,

        /// TLS certificate path
        #[arg(long)]
        cert_path: Option<PathBuf>,

        /// TLS private key path
        #[arg(long)]
        key_path: Option<PathBuf>,
    },

    /// Process a single request
    Process {
        /// Input text to process
        #[arg(short, long)]
        input: String,

        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: String,

        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Generate text
    Generate {
        /// Prompt for text generation
        #[arg(short, long)]
        prompt: String,

        /// Maximum number of tokens to generate
        #[arg(short, long, default_value = "100")]
        max_tokens: usize,

        /// Temperature for generation (0.0-1.0)
        #[arg(short, long, default_value = "0.7")]
        temperature: f32,

        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Chat with the AI (interactive by default, use --message for one-shot)
    Chat {
        /// Message to send (non-interactive mode)
        #[arg(short, long)]
        message: Option<String>,

        /// Conversation ID for context
        #[arg(short, long)]
        conversation_id: Option<String>,

        /// Chat history file
        #[arg(long)]
        history_file: Option<PathBuf>,
    },

    /// Analyze code
    Analyze {
        /// Code file to analyze
        #[arg(short, long)]
        file: PathBuf,

        /// Programming language
        #[arg(short, long)]
        language: String,

        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: String,

        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Generate code
    Codegen {
        /// Description of code to generate
        #[arg(short, long)]
        description: String,

        /// Programming language
        #[arg(short, long)]
        language: String,

        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Train foundation model in-place (updates registered model weights)
    #[command(aliases = &["tf"])]
    TrainFoundation {
        /// Training data (.arrow or text)
        #[arg(short, long)]
        data: PathBuf,

        /// Model ID to train (default: omnis). Use "all" for all models.
        #[arg(short = 'm', long, default_value = "omnis")]
        model_id: String,

        /// Number of training steps
        #[arg(short = 's', long, default_value = "100")]
        steps: usize,

        /// Batch size
        #[arg(short = 'b', long, default_value = "4")]
        batch_size: usize,

        /// Learning rate
        #[arg(short = 'l', long, default_value = "0.01")]
        learning_rate: f32,

        /// Sequence length
        #[arg(short = 'S', long, default_value = "64")]
        seq_length: usize,

        /// Output checkpoint path (.safetensors)
        #[arg(short = 'o', long)]
        output: Option<PathBuf>,

        /// Validation data (optional)
        #[arg(short = 'V', long)]
        val_data: Option<PathBuf>,

        /// Train all models in parallel
        #[arg(short = 'P', long)]
        parallel: bool,

        /// Enable GPU acceleration (default: enabled). Use --gpu false for CPU.
        #[arg(short = 'g', long, default_value_t = true)]
        gpu: bool,
    },

    /// Collect dataset from web sources and save as .arrow
    #[command(aliases = &["cd"])]
    CollectData {
        /// Source providers (comma-separated: hackernews,wikipedia,reddit). Empty = all.
        #[arg(short, long, default_value = "")]
        sources: String,

        /// Max total samples to collect
        #[arg(short = 'm', long, default_value = "500")]
        max_samples: usize,

        /// Max shard size in MB (dataset split into shards)
        #[arg(long, default_value = "500")]
        max_shard_size_mb: usize,

        /// Output .arrow file or directory
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Load a trained checkpoint into a model
    #[command(aliases = &["lc"])]
    LoadCheckpoint {
        /// Model ID to load into (e.g. omnis, swift)
        #[arg(short, long)]
        model: String,

        /// Path to .safetensors checkpoint
        #[arg(short, long)]
        path: PathBuf,
    },

    /// Train a model
    #[command(aliases = &["t"])]
    Train {
        /// Training data file
        #[arg(short, long)]
        data: PathBuf,

        /// Model output path
        #[arg(short, long)]
        output: PathBuf,

        /// Tokenizer path (load or save)
        #[arg(short = 'T', long)]
        tokenizer: Option<PathBuf>,

        /// Number of epochs
        #[arg(short = 'e', long, default_value = "10")]
        epochs: usize,

        /// Batch size
        #[arg(short = 'b', long, default_value = "32")]
        batch_size: usize,

        /// Learning rate
        #[arg(short = 'l', long, default_value = "0.001")]
        learning_rate: f32,

        /// Enable GPU acceleration (default: enabled). Use --gpu false for CPU.
        #[arg(short = 'g', long, default_value_t = true)]
        gpu: bool,

        /// Sequence length for training (context window)
        #[arg(long, default_value = "128")]
        seq_length: usize,

        /// Resume from last checkpoint
        #[arg(short = 'R', long, default_value_t = false)]
        resume: bool,

        /// Model ID to train (default: all). Use specific name for single model.
        #[arg(short = 'm', long, default_value = "all")]
        model_id: String,

        /// Train all models in parallel
        #[arg(short = 'P', long)]
        parallel: bool,
    },

    /// Evaluate model
    #[command(aliases = &["e", "eval"])]
    Evaluate {
        /// Model path
        #[arg(short = 'm', long)]
        model: PathBuf,

        /// Test data file
        #[arg(short = 'd', long)]
        test_data: PathBuf,

        /// Tokenizer path (required for proper encoding)
        #[arg(short = 'T', long)]
        tokenizer: PathBuf,

        /// Output file for results
        #[arg(short = 'o', long)]
        output: Option<PathBuf>,
    },

    /// Show system information
    Info {
        /// Show performance metrics
        #[arg(long)]
        performance: bool,

        /// Show memory statistics
        #[arg(long)]
        memory: bool,

        /// Show model information
        #[arg(long)]
        models: bool,

        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: String,
    },

    /// Health check
    Health {
        /// Detailed health check
        #[arg(long)]
        detailed: bool,
    },

    /// Configuration management
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// Tokenizer operations
    Tokenizer {
        #[command(subcommand)]
        action: TokenizerAction,
    },

    /// Memory operations
    Memory {
        #[command(subcommand)]
        action: MemoryAction,
    },
}

#[derive(Subcommand, Debug)]
pub enum ConfigAction {
    /// Show current configuration
    Show,
    /// Validate configuration
    Validate,
    /// Generate default configuration
    Generate {
        /// Output file
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Update configuration
    Update {
        /// Key to update
        #[arg(short, long)]
        key: String,
        /// Value to set
        #[arg(short, long)]
        value: String,
    },
}

#[derive(Subcommand, Debug)]
pub enum TokenizerAction {
    /// Train new tokenizer
    Train {
        /// Training data file
        #[arg(short, long)]
        data: PathBuf,
        /// Output model path
        #[arg(short, long)]
        output: PathBuf,
        /// Vocabulary size
        #[arg(long, default_value = "50000")]
        vocab_size: usize,
        /// Minimum frequency
        #[arg(long, default_value = "2")]
        min_frequency: usize,
    },
    /// Test tokenizer
    Test {
        /// Text to tokenize
        #[arg(short, long)]
        text: String,
        /// Show detailed output
        #[arg(long)]
        detailed: bool,
    },
    /// Show tokenizer information
    Info {
        /// Model path
        #[arg(short, long)]
        model: PathBuf,
    },
}

#[derive(Subcommand, Debug)]
pub enum MemoryAction {
    /// Show memory statistics
    Stats {
        /// Show detailed statistics
        #[arg(long)]
        detailed: bool,
    },
    /// Clear memory
    Clear {
        /// Memory layer to clear (short, session, long, knowledge, all)
        #[arg(short, long, default_value = "all")]
        layer: String,
    },
    /// Export memory
    Export {
        /// Output file
        #[arg(short, long)]
        output: PathBuf,
        /// Format (json, csv)
        #[arg(short, long, default_value = "json")]
        format: String,
    },
    /// Import memory
    Import {
        /// Input file
        #[arg(short, long)]
        input: PathBuf,
        /// Format (json, csv)
        #[arg(short, long, default_value = "json")]
        format: String,
    },
}
