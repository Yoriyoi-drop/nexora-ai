//! Chat functionality for CLI

use anyhow::Result;
use std::path::PathBuf;
use tracing::info;

use crate::NexoraAI;

impl crate::cli::commands::Cli {
    /// Run chat command
    pub async fn run_chat(
        &self,
        nexora: &NexoraAI,
        interactive: bool,
        message: &Option<String>,
        conversation_id: &Option<String>,
        history_file: &Option<PathBuf>,
    ) -> Result<()> {
        if interactive {
            self.run_interactive_chat(nexora, conversation_id, history_file).await
        } else if let Some(msg) = message {
            let truncated = if msg.len() > 100 { format!("{} [truncated {} chars]", &msg[..100], msg.len()) } else { msg.clone() };
            info!("Chat message: {}", truncated);
            let response = nexora.chat(msg, conversation_id.clone()).await?;
            println!("{}", response);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Either --interactive or --message must be provided"))
        }
    }
    
    /// Run interactive chat
    async fn run_interactive_chat(
        &self,
        nexora: &NexoraAI,
        conversation_id: &Option<String>,
        history_file: &Option<PathBuf>,
    ) -> Result<()> {
        use std::io::{self, Write};
        
        println!("Nexora AI Interactive Chat");
        println!("Type 'exit' to quit, 'help' for commands");
        
        let mut history = Vec::new();
        
        loop {
            print!("> ");
            io::stdout().flush()?;
            
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim();
            
            match input {
                "exit" | "quit" => break,
                "help" => {
                    println!("Commands:");
                    println!("  exit/quit - Exit chat");
                    println!("  help - Show this help");
                    println!("  clear - Clear screen");
                    println!("  history - Show chat history");
                    println!("  Any other text - Send to AI");
                }
                "clear" => {
                    // Clear terminal
                    print!("\x1B[2J\x1B[1;1H");
                }
                "history" => {
                    for (i, (user_msg, ai_msg)) in history.iter().enumerate() {
                        println!("{}: {}", i * 2, user_msg);
                        println!("{}: {}", i * 2 + 1, ai_msg);
                    }
                }
                _ => {
                    let response = nexora.chat(input, conversation_id.clone()).await?;
                    println!("{}", response);
                    history.push((input.to_string(), response));
                }
            }
        }
        
        // Save history if file provided
        if let Some(file) = history_file {
            let history_json = serde_json::to_string_pretty(&history)?;
            std::fs::write(file, history_json)?;
        }
        
        Ok(())
    }
}
