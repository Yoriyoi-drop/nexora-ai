//! Special Tokens - Rust implementation
//! 
//! Management of special tokens for tokenizer

use std::collections::HashMap;
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// Special token IDs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpecialTokenID {
    Pad = 0,     // Padding token
    Unk = 1,     // Unknown token
    Bos = 2,     // Beginning of sequence
    Eos = 3,     // End of sequence
    Eod = 4,     // End of document
    
    // Code-related tokens
    CodeStart = 5,
    CodeEnd = 6,
    MathStart = 7,
    MathEnd = 8,
    DocStart = 9,
    DocEnd = 10,
    
    // Conversation tokens
    UserStart = 11,
    UserEnd = 12,
    AssistantStart = 13,
    AssistantEnd = 14,
    SystemStart = 15,
    SystemEnd = 16,
    
    // Task-specific tokens
    AnswerStart = 17,
    AnswerEnd = 18,
    BugStart = 19,
    BugEnd = 20,
    FixStart = 21,
    FixEnd = 22,
    DiffStart = 23,
    DiffEnd = 24,
    TracebackStart = 25,
    TracebackEnd = 26,
    ExplainStart = 27,
    ExplainEnd = 28,
}

impl SpecialTokenID {
    pub const COUNT: usize = 29;
    
    /// Get the string representation of this special token
    pub fn as_str(&self) -> &'static str {
        match self {
            SpecialTokenID::Pad => "<pad>",
            SpecialTokenID::Unk => "<unk>",
            SpecialTokenID::Bos => "<bos>",
            SpecialTokenID::Eos => "<eos>",
            SpecialTokenID::Eod => "<eod>",
            SpecialTokenID::CodeStart => "<code>",
            SpecialTokenID::CodeEnd => "</code>",
            SpecialTokenID::MathStart => "<math>",
            SpecialTokenID::MathEnd => "</math>",
            SpecialTokenID::DocStart => "<doc>",
            SpecialTokenID::DocEnd => "</doc>",
            SpecialTokenID::UserStart => "<user>",
            SpecialTokenID::UserEnd => "</user>",
            SpecialTokenID::AssistantStart => "<assistant>",
            SpecialTokenID::AssistantEnd => "</assistant>",
            SpecialTokenID::SystemStart => "<system>",
            SpecialTokenID::SystemEnd => "</system>",
            SpecialTokenID::AnswerStart => "<answer>",
            SpecialTokenID::AnswerEnd => "</answer>",
            SpecialTokenID::BugStart => "<bug>",
            SpecialTokenID::BugEnd => "</bug>",
            SpecialTokenID::FixStart => "<fix>",
            SpecialTokenID::FixEnd => "</fix>",
            SpecialTokenID::DiffStart => "<diff>",
            SpecialTokenID::DiffEnd => "</diff>",
            SpecialTokenID::TracebackStart => "<traceback>",
            SpecialTokenID::TracebackEnd => "</traceback>",
            SpecialTokenID::ExplainStart => "<explain>",
            SpecialTokenID::ExplainEnd => "</explain>",
        }
    }
    
    /// Check if this is a start token
    pub fn is_start(&self) -> bool {
        matches!(self,
            SpecialTokenID::Bos | SpecialTokenID::CodeStart | SpecialTokenID::MathStart |
            SpecialTokenID::DocStart | SpecialTokenID::UserStart | SpecialTokenID::AssistantStart |
            SpecialTokenID::SystemStart | SpecialTokenID::AnswerStart | SpecialTokenID::BugStart |
            SpecialTokenID::FixStart | SpecialTokenID::DiffStart | SpecialTokenID::TracebackStart |
            SpecialTokenID::ExplainStart
        )
    }
    
    /// Check if this is an end token
    pub fn is_end(&self) -> bool {
        matches!(self,
            SpecialTokenID::Eos | SpecialTokenID::Eod | SpecialTokenID::CodeEnd | SpecialTokenID::MathEnd |
            SpecialTokenID::DocEnd | SpecialTokenID::UserEnd | SpecialTokenID::AssistantEnd |
            SpecialTokenID::SystemEnd | SpecialTokenID::AnswerEnd | SpecialTokenID::BugEnd |
            SpecialTokenID::FixEnd | SpecialTokenID::DiffEnd | SpecialTokenID::TracebackEnd |
            SpecialTokenID::ExplainEnd
        )
    }
    
    /// Get the matching end token for a start token
    pub fn matching_end(&self) -> Option<SpecialTokenID> {
        match self {
            SpecialTokenID::CodeStart => Some(SpecialTokenID::CodeEnd),
            SpecialTokenID::MathStart => Some(SpecialTokenID::MathEnd),
            SpecialTokenID::DocStart => Some(SpecialTokenID::DocEnd),
            SpecialTokenID::UserStart => Some(SpecialTokenID::UserEnd),
            SpecialTokenID::AssistantStart => Some(SpecialTokenID::AssistantEnd),
            SpecialTokenID::SystemStart => Some(SpecialTokenID::SystemEnd),
            SpecialTokenID::AnswerStart => Some(SpecialTokenID::AnswerEnd),
            SpecialTokenID::BugStart => Some(SpecialTokenID::BugEnd),
            SpecialTokenID::FixStart => Some(SpecialTokenID::FixEnd),
            SpecialTokenID::DiffStart => Some(SpecialTokenID::DiffEnd),
            SpecialTokenID::TracebackStart => Some(SpecialTokenID::TracebackEnd),
            SpecialTokenID::ExplainStart => Some(SpecialTokenID::ExplainEnd),
            _ => None,
        }
    }
    
    /// Get the matching start token for an end token
    pub fn matching_start(&self) -> Option<SpecialTokenID> {
        match self {
            SpecialTokenID::CodeEnd => Some(SpecialTokenID::CodeStart),
            SpecialTokenID::MathEnd => Some(SpecialTokenID::MathStart),
            SpecialTokenID::DocEnd => Some(SpecialTokenID::DocStart),
            SpecialTokenID::UserEnd => Some(SpecialTokenID::UserStart),
            SpecialTokenID::AssistantEnd => Some(SpecialTokenID::AssistantStart),
            SpecialTokenID::SystemEnd => Some(SpecialTokenID::SystemStart),
            SpecialTokenID::AnswerEnd => Some(SpecialTokenID::AnswerStart),
            SpecialTokenID::BugEnd => Some(SpecialTokenID::BugStart),
            SpecialTokenID::FixEnd => Some(SpecialTokenID::FixStart),
            SpecialTokenID::DiffEnd => Some(SpecialTokenID::DiffStart),
            SpecialTokenID::TracebackEnd => Some(SpecialTokenID::TracebackStart),
            SpecialTokenID::ExplainEnd => Some(SpecialTokenID::ExplainStart),
            _ => None,
        }
    }
}

/// Special tokens manager
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    token_to_id: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    special_ids: [u32; SpecialTokenID::COUNT],
    next_id: u32,
}

impl SpecialTokens {
    /// Create a new special tokens manager
    pub fn new() -> Self {
        let mut token_to_id = HashMap::with_capacity(SpecialTokenID::COUNT);
        let mut id_to_token = HashMap::with_capacity(SpecialTokenID::COUNT);
        let mut special_ids = [0; SpecialTokenID::COUNT];
        
        // Initialize all special tokens
        for (i, _token_id) in (0..SpecialTokenID::COUNT).enumerate() {
            let special_token = Self::index_to_special_token(i);
            let token_str = special_token.as_str();
            
            token_to_id.insert(token_str.to_string(), i as u32);
            id_to_token.insert(i as u32, token_str.to_string());
            special_ids[i] = i as u32;
        }
        
        Self {
            token_to_id,
            id_to_token,
            special_ids,
            next_id: SpecialTokenID::COUNT as u32,
        }
    }
    
    /// Convert index to SpecialTokenID
    fn index_to_special_token(index: usize) -> SpecialTokenID {
        match index {
            0 => SpecialTokenID::Pad,
            1 => SpecialTokenID::Unk,
            2 => SpecialTokenID::Bos,
            3 => SpecialTokenID::Eos,
            4 => SpecialTokenID::Eod,
            5 => SpecialTokenID::CodeStart,
            6 => SpecialTokenID::CodeEnd,
            7 => SpecialTokenID::MathStart,
            8 => SpecialTokenID::MathEnd,
            9 => SpecialTokenID::DocStart,
            10 => SpecialTokenID::DocEnd,
            11 => SpecialTokenID::UserStart,
            12 => SpecialTokenID::UserEnd,
            13 => SpecialTokenID::AssistantStart,
            14 => SpecialTokenID::AssistantEnd,
            15 => SpecialTokenID::SystemStart,
            16 => SpecialTokenID::SystemEnd,
            17 => SpecialTokenID::AnswerStart,
            18 => SpecialTokenID::AnswerEnd,
            19 => SpecialTokenID::BugStart,
            20 => SpecialTokenID::BugEnd,
            21 => SpecialTokenID::FixStart,
            22 => SpecialTokenID::FixEnd,
            23 => SpecialTokenID::DiffStart,
            24 => SpecialTokenID::DiffEnd,
            25 => SpecialTokenID::TracebackStart,
            26 => SpecialTokenID::TracebackEnd,
            27 => SpecialTokenID::ExplainStart,
            28 => SpecialTokenID::ExplainEnd,
            _ => SpecialTokenID::Unk,
        }
    }
    
    /// Get the ID for a special token
    pub fn get_id(&self, token: &SpecialTokenID) -> u32 {
        self.special_ids[*token as usize]
    }
    
    /// Get the ID for a special token by string
    pub fn get_id_by_str(&self, token_str: &str) -> Option<u32> {
        self.token_to_id.get(token_str).copied()
    }
    
    /// Get the string representation of a special token by ID
    pub fn get_token_str(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }
    
    /// Get the string representation of a special token
    pub fn get_token(&self, token: &SpecialTokenID) -> &'static str {
        token.as_str()
    }
    
    /// Check if a token is a special token by ID
    pub fn is_special(&self, id: u32) -> bool {
        self.id_to_token.contains_key(&id)
    }
    
    /// Check if a token is a special token by string
    pub fn is_special_str(&self, token_str: &str) -> bool {
        self.token_to_id.contains_key(token_str)
    }
    
    /// Get all special token IDs
    pub fn get_all_ids(&self) -> [u32; SpecialTokenID::COUNT] {
        self.special_ids
    }
    
    /// Get all special token strings
    pub fn get_all_tokens(&self) -> Vec<&str> {
        (0..SpecialTokenID::COUNT)
            .map(|i| Self::index_to_special_token(i).as_str())
            .collect()
    }
    
    /// Add a custom special token
    pub fn add_custom_token(&mut self, token_str: &str) -> Result<u32> {
        if self.token_to_id.contains_key(token_str) {
            return Ok(self.token_to_id[token_str]);
        }
        
        let id = self.next_id;
        self.next_id += 1;
        
        self.token_to_id.insert(token_str.to_string(), id);
        self.id_to_token.insert(id, token_str.to_string());
        
        Ok(id)
    }
    
    /// Get the count of special tokens
    pub fn count(&self) -> usize {
        SpecialTokenID::COUNT
    }
    
    /// Get tokens by category
    pub fn get_conversation_tokens(&self) -> Vec<SpecialTokenID> {
        vec![
            SpecialTokenID::UserStart, SpecialTokenID::UserEnd,
            SpecialTokenID::AssistantStart, SpecialTokenID::AssistantEnd,
            SpecialTokenID::SystemStart, SpecialTokenID::SystemEnd,
        ]
    }
    
    /// Get code-related tokens
    pub fn get_code_tokens(&self) -> Vec<SpecialTokenID> {
        vec![
            SpecialTokenID::CodeStart, SpecialTokenID::CodeEnd,
            SpecialTokenID::MathStart, SpecialTokenID::MathEnd,
            SpecialTokenID::DocStart, SpecialTokenID::DocEnd,
        ]
    }
    
    /// Get task-related tokens
    pub fn get_task_tokens(&self) -> Vec<SpecialTokenID> {
        vec![
            SpecialTokenID::AnswerStart, SpecialTokenID::AnswerEnd,
            SpecialTokenID::BugStart, SpecialTokenID::BugEnd,
            SpecialTokenID::FixStart, SpecialTokenID::FixEnd,
            SpecialTokenID::DiffStart, SpecialTokenID::DiffEnd,
            SpecialTokenID::TracebackStart, SpecialTokenID::TracebackEnd,
            SpecialTokenID::ExplainStart, SpecialTokenID::ExplainEnd,
        ]
    }
    
    /// Get control tokens (bos, eos, etc.)
    pub fn get_control_tokens(&self) -> Vec<SpecialTokenID> {
        vec![
            SpecialTokenID::Pad, SpecialTokenID::Unk,
            SpecialTokenID::Bos, SpecialTokenID::Eos, SpecialTokenID::Eod,
        ]
    }
    
    /// Find matching pairs in a sequence of token IDs
    pub fn find_matching_pairs(&self, token_ids: &[u32]) -> Vec<(usize, usize)> {
        let mut pairs = Vec::with_capacity(token_ids.len());
        let mut stack = Vec::with_capacity(token_ids.len());
        
        for (i, &id) in token_ids.iter().enumerate() {
            if let Some(token_str) = self.get_token_str(id) {
                // Check if it's an end token
                if let Some(end_token) = self.get_id_by_str(token_str) {
                    if let Some(start_token) = Self::get_special_token_by_id(end_token).and_then(|t| t.matching_start()) {
                        // Look for matching start token in stack
                        if let Some(stack_pos) = stack.iter().rposition(|&(stack_id, stack_token)| {
                            stack_id == self.get_id(&start_token) as usize && stack_token == start_token
                        }) {
                            let (start_idx, _) = stack.remove(stack_pos);
                            pairs.push((start_idx, i));
                        }
                    }
                } else if let Some(start_token) = Self::get_special_token_by_id(id) {
                    if start_token.is_start() {
                        stack.push((i, start_token));
                    }
                }
            }
        }
        
        pairs
    }
    
    /// Get SpecialTokenID by ID
    fn get_special_token_by_id(id: u32) -> Option<SpecialTokenID> {
        if id < SpecialTokenID::COUNT as u32 {
            Some(Self::index_to_special_token(id as usize))
        } else {
            None
        }
    }
    
    /// Validate a sequence of token IDs for proper special token pairing
    pub fn validate_sequence(&self, token_ids: &[u32]) -> Result<()> {
        let pairs = self.find_matching_pairs(token_ids);
        let mut used_positions = std::collections::HashSet::with_capacity(pairs.len() * 2);
        
        for (start, end) in pairs {
            if used_positions.contains(&start) || used_positions.contains(&end) {
                return Err(anyhow::anyhow!("Overlapping special token pairs at positions {} and {}", start, end));
            }
            used_positions.insert(start);
            used_positions.insert(end);
        }
        
        // Check for unmatched start tokens
        for (i, &id) in token_ids.iter().enumerate() {
            if !used_positions.contains(&i) {
                if let Some(token) = Self::get_special_token_by_id(id) {
                    if token.is_start() {
                        return Err(anyhow::anyhow!("Unmatched start token at position {}: {}", i, token.as_str()));
                    }
                }
            }
        }
        
        Ok(())
    }
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions
pub fn get_special_token_id(token: &SpecialTokenID) -> u32 {
    let special_tokens = SpecialTokens::new();
    special_tokens.get_id(token)
}

pub fn get_special_token_str(token: &SpecialTokenID) -> &'static str {
    token.as_str()
}

pub fn is_special_token_str(token_str: &str) -> bool {
    let special_tokens = SpecialTokens::new();
    special_tokens.is_special_str(token_str)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_special_token_strings() {
        assert_eq!(SpecialTokenID::Bos.as_str(), "<bos>");
        assert_eq!(SpecialTokenID::Eos.as_str(), "<eos>");
        assert_eq!(SpecialTokenID::CodeStart.as_str(), "<code>");
        assert_eq!(SpecialTokenID::CodeEnd.as_str(), "</code>");
    }
    
    #[test]
    fn test_token_matching() {
        assert_eq!(SpecialTokenID::CodeStart.matching_end(), Some(SpecialTokenID::CodeEnd));
        assert_eq!(SpecialTokenID::CodeEnd.matching_start(), Some(SpecialTokenID::CodeStart));
        assert_eq!(SpecialTokenID::Bos.matching_end(), None);
        assert_eq!(SpecialTokenID::Eos.matching_start(), None);
    }
    
    #[test]
    fn test_special_tokens_manager() {
        let special_tokens = SpecialTokens::new();
        
        assert_eq!(special_tokens.get_id(&SpecialTokenID::Bos), 2);
        assert_eq!(special_tokens.get_token_str(2), Some("<bos>"));
        assert!(special_tokens.is_special(2));
        assert!(special_tokens.is_special_str("<bos>"));
        assert!(!special_tokens.is_special(1000));
        assert!(!special_tokens.is_special_str("regular_token"));
    }
    
    #[test]
    fn test_token_categories() {
        let special_tokens = SpecialTokens::new();
        
        let conv_tokens = special_tokens.get_conversation_tokens();
        assert!(conv_tokens.contains(&SpecialTokenID::UserStart));
        assert!(conv_tokens.contains(&SpecialTokenID::AssistantEnd));
        
        let code_tokens = special_tokens.get_code_tokens();
        assert!(code_tokens.contains(&SpecialTokenID::CodeStart));
        assert!(code_tokens.contains(&SpecialTokenID::MathEnd));
        
        let task_tokens = special_tokens.get_task_tokens();
        assert!(task_tokens.contains(&SpecialTokenID::BugStart));
        assert!(task_tokens.contains(&SpecialTokenID::FixEnd));
    }
    
    #[test]
    fn test_sequence_validation() {
        let special_tokens = SpecialTokens::new();
        
        // Valid sequence
        let valid_sequence = vec![
            special_tokens.get_id(&SpecialTokenID::Bos),
            special_tokens.get_id(&SpecialTokenID::CodeStart),
            100, // regular token
            special_tokens.get_id(&SpecialTokenID::CodeEnd),
            special_tokens.get_id(&SpecialTokenID::Eos),
        ];
        
        assert!(special_tokens.validate_sequence(&valid_sequence).is_ok());
        
        // Invalid sequence - unmatched start
        let invalid_sequence = vec![
            special_tokens.get_id(&SpecialTokenID::Bos),
            special_tokens.get_id(&SpecialTokenID::CodeStart),
            special_tokens.get_id(&SpecialTokenID::Eos),
        ];
        
        assert!(special_tokens.validate_sequence(&invalid_sequence).is_err());
    }
    
    #[test]
    fn test_custom_tokens() {
        let mut special_tokens = SpecialTokens::new();
        
        let custom_id = special_tokens.add_custom_token("<custom>").unwrap();
        assert!(special_tokens.is_special(custom_id));
        assert_eq!(special_tokens.get_token_str(custom_id), Some("<custom>"));
        
        // Adding same token should return same ID
        let same_id = special_tokens.add_custom_token("<custom>").unwrap();
        assert_eq!(custom_id, same_id);
    }
}
