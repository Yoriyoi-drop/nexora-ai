pub mod cache;
pub mod compression;
pub mod iterator;
pub mod loader;
pub mod manifest;
pub mod progress;
pub mod registry;
pub mod scanner;
pub mod schema;
pub mod shuffle;

pub use cache::*;
pub use compression::*;
pub use iterator::*;
pub use loader::*;
pub use manifest::*;
pub use progress::*;
pub use registry::*;
pub use scanner::*;
pub use schema::*;
pub use shuffle::*;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DatasetSplit {
    Train,
    Val,
    Test,
    Reinforcement,
    Synthetic,
    Instruction,
}

impl DatasetSplit {
    pub fn as_str(&self) -> &'static str {
        match self {
            DatasetSplit::Train => "train",
            DatasetSplit::Val => "val",
            DatasetSplit::Test => "test",
            DatasetSplit::Reinforcement => "reinforcement",
            DatasetSplit::Synthetic => "synthetic",
            DatasetSplit::Instruction => "instruction",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "train" => Some(DatasetSplit::Train),
            "val" | "validation" => Some(DatasetSplit::Val),
            "test" => Some(DatasetSplit::Test),
            "reinforcement" | "rl" => Some(DatasetSplit::Reinforcement),
            "synthetic" | "synth" => Some(DatasetSplit::Synthetic),
            "instruction" | "instruct" => Some(DatasetSplit::Instruction),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DatasetConfig {
    pub path: String,
    pub split: DatasetSplit,
    pub batch_size: usize,
    pub shuffle_buffer: usize,
    pub prefetch_batches: usize,
    pub num_workers: usize,
    pub seq_length: usize,
    pub resume: bool,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            path: String::new(),
            split: DatasetSplit::Train,
            batch_size: 8,
            shuffle_buffer: 10000,
            prefetch_batches: 4,
            num_workers: 2,
            seq_length: 128,
            resume: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_split_as_str() {
        assert_eq!(DatasetSplit::Train.as_str(), "train");
        assert_eq!(DatasetSplit::Val.as_str(), "val");
        assert_eq!(DatasetSplit::Test.as_str(), "test");
        assert_eq!(DatasetSplit::Instruction.as_str(), "instruction");
    }

    #[test]
    fn test_dataset_split_from_str() {
        assert_eq!(DatasetSplit::from_str("train"), Some(DatasetSplit::Train));
        assert_eq!(DatasetSplit::from_str("val"), Some(DatasetSplit::Val));
        assert_eq!(
            DatasetSplit::from_str("validation"),
            Some(DatasetSplit::Val)
        );
        assert_eq!(
            DatasetSplit::from_str("rl"),
            Some(DatasetSplit::Reinforcement)
        );
        assert_eq!(DatasetSplit::from_str("unknown"), None);
    }

    #[test]
    fn test_dataset_config_default() {
        let cfg = DatasetConfig::default();
        assert_eq!(cfg.batch_size, 8);
        assert_eq!(cfg.shuffle_buffer, 10000);
        assert_eq!(cfg.seq_length, 128);
        assert!(!cfg.resume);
        assert_eq!(cfg.split, DatasetSplit::Train);
    }

    #[test]
    fn test_dataset_split_debug() {
        assert_eq!(format!("{:?}", DatasetSplit::Train), "Train");
        assert_eq!(format!("{:?}", DatasetSplit::Synthetic), "Synthetic");
    }

    #[test]
    fn test_re_exports() {
        let _ = DatasetSplit::Train;
    }
}
