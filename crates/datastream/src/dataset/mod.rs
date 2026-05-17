pub mod manifest;
pub mod scanner;
pub mod compression;
pub mod loader;
pub mod iterator;
pub mod shuffle;
pub mod schema;
pub mod cache;
pub mod progress;
pub mod registry;

pub use manifest::*;
pub use scanner::*;
pub use compression::*;
pub use loader::*;
pub use iterator::*;
pub use shuffle::*;
pub use schema::*;
pub use cache::*;
pub use progress::*;
pub use registry::*;

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
