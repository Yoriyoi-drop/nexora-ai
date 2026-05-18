//! NXR DataStream Filter Fabric
//!
//! DAG-based streaming data pipeline untuk AI training.
//!
//! ## Arsitektur

//!
//! ```text
//! Stream Sources → Intake Engine → Filter DAG → Intelligence Core → Delivery Layer
//! ```
//!
//! ## Filter DAG (bukan linear pipeline)
//!
//! Filter dijalankan sebagai **Directed Acyclic Graph**, bukan pipeline linear.
//! Ini memungkinkan eksekusi paralel dan routing dinamis.
//!
//! ## Contoh
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use nexora_datastream::*;
//! use nexora_datastream::filter::*;
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut graph = ExecutionGraph::new();
//!
//!     graph.add_node("length",  Arc::new(LengthFilter::default()), vec![], true, 1);
//!     graph.add_node("quality", Arc::new(QualityFilter::default()), vec!["length".into()], true, 2);
//!     graph.add_node("dedup",   Arc::new(DedupFilter::new()), vec!["quality".into()], false, 3);
//!
//!     graph.finalize();
//! }
//! ```

pub mod filter;
pub mod graph;
pub mod intake;
pub mod intelligence;
pub mod delivery;
pub mod types;
pub mod source;

#[cfg(feature = "arrow")]
pub mod arrow_reader;

#[cfg(feature = "arrow")]
pub mod dataset;

pub use filter::*;
pub use graph::*;
pub use intake::*;
pub use intelligence::*;
pub use delivery::*;
pub use types::*;
pub use source::*;

#[cfg(feature = "arrow")]
pub use dataset::*;

pub use filter::Filter as FilterTrait;

pub struct PipelineBuilder {
    graph: ExecutionGraph,
    intake: StreamIntakeEngine,
    intelligence: DatasetIntelligenceCore,
    delivery: TrainingDeliveryLayer,
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self {
            graph: ExecutionGraph::new(),
            intake: StreamIntakeEngine::default(),
            intelligence: DatasetIntelligenceCore::new(),
            delivery: TrainingDeliveryLayer::default(),
        }
    }

    pub fn with_graph(mut self, graph: ExecutionGraph) -> Self {
        self.graph = graph;
        self
    }

    pub fn with_intake(mut self, intake: StreamIntakeEngine) -> Self {
        self.intake = intake;
        self
    }

    pub fn with_intelligence(mut self, intelligence: DatasetIntelligenceCore) -> Self {
        self.intelligence = intelligence;
        self
    }

    pub fn with_delivery(mut self, delivery: TrainingDeliveryLayer) -> Self {
        self.delivery = delivery;
        self
    }

    pub fn add_filter(mut self, id: &str, filter: impl FilterTrait + 'static, depends_on: Vec<String>, concurrent: bool, priority: u8) -> Self {
        self.graph.add_node(id, std::sync::Arc::new(filter), depends_on, concurrent, priority);
        self
    }

    pub fn build(self) -> Pipeline {
        Pipeline {
            graph: self.graph,
            intake: self.intake,
            intelligence: self.intelligence,
            delivery: self.delivery,
            cancel_tx: None,
        }
    }
}

pub struct Pipeline {
    pub graph: ExecutionGraph,
    pub intake: StreamIntakeEngine,
    pub intelligence: DatasetIntelligenceCore,
    pub delivery: TrainingDeliveryLayer,
    cancel_tx: Option<tokio::sync::watch::Sender<bool>>,
}

impl Pipeline {
    pub fn builder() -> PipelineBuilder {
        PipelineBuilder::new()
    }

    pub fn new() -> Self {
        Self::builder().build()
    }

    pub async fn run(
        &mut self,
        samples: Vec<DataSample>,
        (cancel_tx, cancel_rx): (tokio::sync::watch::Sender<bool>, tokio::sync::watch::Receiver<bool>),
    ) -> Vec<graph::ExecutionResult> {
        self.cancel_tx = Some(cancel_tx);
        self.graph.finalize();
        let mut results = Vec::with_capacity(samples.len());

        for sample in samples {
            if *cancel_rx.borrow() {
                break;
            }

            let result = self.graph.execute(sample, cancel_rx.clone()).await;

            if let graph::ExecutionResult::Accepted { ref sample, .. } = result {
                let score = self.intelligence.score_sample(sample);
                self.intelligence.update_quality_distribution(score);
            }

            results.push(result);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::*;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_pipeline_basic_flow() {
        let mut graph = ExecutionGraph::new();
        graph.add_node("length", std::sync::Arc::new(LengthFilter::default()), vec![], true, 1);
        graph.add_node("quality", std::sync::Arc::new(QualityFilter::default()), vec!["length".into()], true, 2);
        graph.finalize();

        let sample = DataSample {
            id: Uuid::new_v4(),
            text: "The quick brown fox jumps over the lazy dog. This is a test sentence for the pipeline. We need enough words to pass the length filter and demonstrate quality.".to_string(),
            token_ids: None,
            metadata: std::collections::HashMap::new(),
            source: types::SourceInfo {
                name: "test".to_string(),
                url: None,
                trust_score: 0.5,
                category: types::SourceCategory::Other,
                fetch_timestamp: 0,
            },
            stats: types::SampleStats::default(),
            domains: vec![],
            score: None,
            curriculum_level: None,
        };

        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let result = graph.execute(sample, cancel_rx).await;
        assert!(result.is_accepted());
        drop(cancel_tx);
    }

    #[tokio::test]
    async fn test_toxicity_filter() {
        let filter = ToxicityFilter::default();
        let clean = DataSample {
            id: Uuid::new_v4(),
            text: "This is a clean and respectful sentence about technology and science.".to_string(),
            token_ids: None,
            metadata: std::collections::HashMap::new(),
            source: types::SourceInfo {
                name: "test".to_string(),
                url: None,
                trust_score: 0.9,
                category: types::SourceCategory::Academic,
                fetch_timestamp: 0,
            },
            stats: types::SampleStats::default(),
            domains: vec![],
            score: None,
            curriculum_level: None,
        };
        let result = filter.filter(&clean).await;
        assert!(result.passed);
    }
}
