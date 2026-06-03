use async_trait::async_trait;

pub mod curriculum;
pub mod custom;
pub mod dedup;
pub mod domain;
pub mod entropy;
pub mod ml_classifier;
pub mod language;
pub mod length;
pub mod perplexity;
#[cfg(feature = "prompt-injection")]
pub mod prompt_injection;
pub mod quality;
pub mod regex;
pub mod semantic_dedup;
pub mod token;
#[cfg(feature = "toxicity")]
pub mod toxicity;
pub mod traits;
pub mod trust_score;

pub use curriculum::CurriculumRanker;
pub use custom::CustomFilter;
pub use dedup::DedupFilter;
pub use domain::DomainClassifier;
pub use entropy::EntropyFilter;
pub use ml_classifier::MLClassifier;
pub use language::LanguageFilter;
pub use length::LengthFilter;
pub use perplexity::PerplexityFilter;
#[cfg(feature = "prompt-injection")]
pub use prompt_injection::PromptInjectionFilter;
pub use quality::QualityFilter;
pub use regex::RegexFilter;
pub use semantic_dedup::SemanticDedupFilter;
pub use token::TokenFilter;
#[cfg(feature = "toxicity")]
pub use toxicity::ToxicityFilter;
pub use traits::{Filter, ParallelFilter};
pub use trust_score::TrustScoreFilter;

#[async_trait]
impl ParallelFilter for curriculum::CurriculumRanker {}
#[async_trait]
impl ParallelFilter for custom::CustomFilter {}
#[async_trait]
impl ParallelFilter for dedup::DedupFilter {}
#[async_trait]
impl ParallelFilter for domain::DomainClassifier {}
#[async_trait]
impl ParallelFilter for entropy::EntropyFilter {}
#[async_trait]
impl ParallelFilter for language::LanguageFilter {}
#[async_trait]
impl ParallelFilter for length::LengthFilter {}
#[async_trait]
impl ParallelFilter for perplexity::PerplexityFilter {}
#[cfg(feature = "prompt-injection")]
#[async_trait]
impl ParallelFilter for prompt_injection::PromptInjectionFilter {}
#[async_trait]
impl ParallelFilter for quality::QualityFilter {}
#[async_trait]
impl ParallelFilter for regex::RegexFilter {}
#[async_trait]
impl ParallelFilter for semantic_dedup::SemanticDedupFilter {}
#[async_trait]
impl ParallelFilter for token::TokenFilter {}
#[cfg(feature = "toxicity")]
#[async_trait]
impl ParallelFilter for toxicity::ToxicityFilter {}
#[async_trait]
impl ParallelFilter for trust_score::TrustScoreFilter {}

#[cfg(test)]
mod tests {
    #[test]
    fn test_re_exports() {
        #[cfg(feature = "toxicity")]
        let _ = super::ToxicityFilter::default();
        #[cfg(feature = "prompt-injection")]
        let _ = super::PromptInjectionFilter::default();
        let _ = super::DedupFilter::default();
        let _ = super::QualityFilter::default();
        let _ = super::LengthFilter::default();
        let _ = super::PerplexityFilter::default();
        let _ = super::EntropyFilter::default();
        let _ = super::CurriculumRanker::default();
        let _ = super::DomainClassifier::default();
        let _ = super::LanguageFilter::default();
        let _ = super::RegexFilter::default();
        let _ = super::TokenFilter::default();
        let _ = super::TrustScoreFilter::default();
        let _ = super::SemanticDedupFilter::default();
        let _ = super::CustomFilter::new("test");
    }
}
