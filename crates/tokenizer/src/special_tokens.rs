//! Special Tokens - Rust implementation
//! 
//! Management of special tokens for tokenizer

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
    
    pub fn try_from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::Pad),
            1 => Some(Self::Unk),
            2 => Some(Self::Bos),
            3 => Some(Self::Eos),
            4 => Some(Self::Eod),
            5 => Some(Self::CodeStart),
            6 => Some(Self::CodeEnd),
            7 => Some(Self::MathStart),
            8 => Some(Self::MathEnd),
            9 => Some(Self::DocStart),
            10 => Some(Self::DocEnd),
            11 => Some(Self::UserStart),
            12 => Some(Self::UserEnd),
            13 => Some(Self::AssistantStart),
            14 => Some(Self::AssistantEnd),
            15 => Some(Self::SystemStart),
            16 => Some(Self::SystemEnd),
            17 => Some(Self::AnswerStart),
            18 => Some(Self::AnswerEnd),
            19 => Some(Self::BugStart),
            20 => Some(Self::BugEnd),
            21 => Some(Self::FixStart),
            22 => Some(Self::FixEnd),
            23 => Some(Self::DiffStart),
            24 => Some(Self::DiffEnd),
            25 => Some(Self::TracebackStart),
            26 => Some(Self::TracebackEnd),
            27 => Some(Self::ExplainStart),
            28 => Some(Self::ExplainEnd),
            _ => None,
        }
    }
    
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
            SpecialTokenID::CodeStart | SpecialTokenID::MathStart |
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
    custom_token_to_id: std::collections::HashMap<String, u32>,
    custom_id_to_token: std::collections::HashMap<u32, String>,
    next_id: u32,
}

impl SpecialTokens {
    pub fn new() -> Self {
        Self {
            custom_token_to_id: std::collections::HashMap::new(),
            custom_id_to_token: std::collections::HashMap::new(),
            next_id: SpecialTokenID::COUNT as u32,
        }
    }
    
    fn builtin_id_by_str(s: &str) -> Option<u32> {
        match s {
            "<pad>" => Some(0),
            "<unk>" => Some(1),
            "<bos>" => Some(2),
            "<eos>" => Some(3),
            "<eod>" => Some(4),
            "<code>" => Some(5),
            "</code>" => Some(6),
            "<math>" => Some(7),
            "</math>" => Some(8),
            "<doc>" => Some(9),
            "</doc>" => Some(10),
            "<user>" => Some(11),
            "</user>" => Some(12),
            "<assistant>" => Some(13),
            "</assistant>" => Some(14),
            "<system>" => Some(15),
            "</system>" => Some(16),
            "<answer>" => Some(17),
            "</answer>" => Some(18),
            "<bug>" => Some(19),
            "</bug>" => Some(20),
            "<fix>" => Some(21),
            "</fix>" => Some(22),
            "<diff>" => Some(23),
            "</diff>" => Some(24),
            "<traceback>" => Some(25),
            "</traceback>" => Some(26),
            "<explain>" => Some(27),
            "</explain>" => Some(28),
            _ => None,
        }
    }
    
    fn builtin_str_by_id(id: u32) -> Option<&'static str> {
        match id {
            0 => Some("<pad>"),
            1 => Some("<unk>"),
            2 => Some("<bos>"),
            3 => Some("<eos>"),
            4 => Some("<eod>"),
            5 => Some("<code>"),
            6 => Some("</code>"),
            7 => Some("<math>"),
            8 => Some("</math>"),
            9 => Some("<doc>"),
            10 => Some("</doc>"),
            11 => Some("<user>"),
            12 => Some("</user>"),
            13 => Some("<assistant>"),
            14 => Some("</assistant>"),
            15 => Some("<system>"),
            16 => Some("</system>"),
            17 => Some("<answer>"),
            18 => Some("</answer>"),
            19 => Some("<bug>"),
            20 => Some("</bug>"),
            21 => Some("<fix>"),
            22 => Some("</fix>"),
            23 => Some("<diff>"),
            24 => Some("</diff>"),
            25 => Some("<traceback>"),
            26 => Some("</traceback>"),
            27 => Some("<explain>"),
            28 => Some("</explain>"),
            _ => None,
        }
    }
    
    pub fn get_id(&self, token: &SpecialTokenID) -> u32 {
        *token as u32
    }
    
    pub fn get_id_by_str(&self, token_str: &str) -> Option<u32> {
        Self::builtin_id_by_str(token_str)
            .or_else(|| self.custom_token_to_id.get(token_str).copied())
    }
    
    pub fn get_token_str(&self, id: u32) -> Option<&str> {
        Self::builtin_str_by_id(id)
            .or_else(|| self.custom_id_to_token.get(&id).map(|s| s.as_str()))
    }
    
    pub fn get_token(&self, token: &SpecialTokenID) -> &'static str {
        token.as_str()
    }
    
    pub fn is_special(&self, id: u32) -> bool {
        (id < SpecialTokenID::COUNT as u32) || self.custom_id_to_token.contains_key(&id)
    }
    
    pub fn is_special_str(&self, token_str: &str) -> bool {
        Self::builtin_id_by_str(token_str).is_some() || self.custom_token_to_id.contains_key(token_str)
    }
    
    pub fn get_all_ids(&self) -> [u32; SpecialTokenID::COUNT] {
        let mut ids = [0; SpecialTokenID::COUNT];
        for i in 0..SpecialTokenID::COUNT {
            ids[i] = i as u32;
        }
        ids
    }
    
    pub fn get_all_tokens(&self) -> Vec<&str> {
        (0..SpecialTokenID::COUNT)
            .map(|i| SpecialTokenID::try_from_id(i as u32).map(|t| t.as_str()).unwrap_or("<unk>"))
            .collect()
    }
    
    pub fn add_custom_token(&mut self, token_str: &str) -> Result<u32> {
        if let Some(id) = Self::builtin_id_by_str(token_str) {
            return Ok(id);
        }
        if let Some(&id) = self.custom_token_to_id.get(token_str) {
            return Ok(id);
        }
        
        let id = self.next_id;
        self.next_id += 1;
        
        self.custom_token_to_id.insert(token_str.to_string(), id);
        self.custom_id_to_token.insert(id, token_str.to_string());
        
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
            if let Some(special_token) = Self::get_special_token_by_id(id) {
                if let Some(start_token) = special_token.matching_start() {
                    // It's an end token — find matching start
                    if let Some(stack_pos) = stack.iter().rposition(|&(_, st)| st == start_token) {
                        let (start_idx, _) = stack.remove(stack_pos);
                        pairs.push((start_idx, i));
                    }
                } else if special_token.is_start() {
                    stack.push((i, special_token));
                }
            }
        }
        
        pairs
    }
    
    fn get_special_token_by_id(id: u32) -> Option<SpecialTokenID> {
        SpecialTokenID::try_from_id(id)
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
    *token as u32
}

pub fn get_special_token_str(token: &SpecialTokenID) -> &'static str {
    token.as_str()
}

pub fn is_special_token_str(token_str: &str) -> bool {
    SpecialTokens::builtin_id_by_str(token_str).is_some()
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
