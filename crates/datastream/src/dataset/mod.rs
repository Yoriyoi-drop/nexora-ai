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
