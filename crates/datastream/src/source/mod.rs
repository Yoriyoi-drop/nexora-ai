use std::collections::HashMap;
use async_trait::async_trait;
use chrono::Utc;

use crate::types::{SourceInfo, SourceCategory, DataSample, SampleStats};
use uuid::Uuid;

#[async_trait]
pub trait SourceProvider: Send + Sync {
    fn name(&self) -> &str;
    fn url(&self) -> &str;
    fn category(&self) -> SourceCategory;
    fn default_trust_score(&self) -> f64;
    fn description(&self) -> &str;

    fn source_info(&self) -> SourceInfo {
        SourceInfo {
            name: self.name().to_string(),
            url: Some(self.url().to_string()),
            trust_score: self.default_trust_score(),
            category: self.category(),
            fetch_timestamp: Utc::now().timestamp(),
        }
    }

    fn sample_data(&self) -> Vec<String>;

    async fn fetch_samples(&self) -> Vec<DataSample> {
        let source = self.source_info();
        self.sample_data().into_iter().map(|text| DataSample {
            id: Uuid::new_v4(),
            text,
            token_ids: None,
            metadata: HashMap::new(),
            source: source.clone(),
            stats: SampleStats::default(),
            domains: vec![],
            score: None,
            curriculum_level: None,
        }).collect()
    }
}

pub struct SourceRegistry {
    providers: HashMap<String, Box<dyn SourceProvider>>,
}

impl SourceRegistry {
    pub fn new() -> Self {
        Self { providers: HashMap::new() }
    }

    pub fn register(&mut self, provider: Box<dyn SourceProvider>) {
        self.providers.insert(provider.name().to_string(), provider);
    }

    pub fn get(&self, name: &str) -> Option<&dyn SourceProvider> {
        self.providers.get(name).map(|p| p.as_ref())
    }

    pub fn all(&self) -> Vec<&dyn SourceProvider> {
        self.providers.values().map(|p| p.as_ref()).collect()
    }

    pub fn names(&self) -> Vec<&str> {
        self.providers.keys().map(|s| s.as_str()).collect()
    }

    pub fn resolve(&self, names: &[String]) -> Vec<&dyn SourceProvider> {
        names.iter()
            .filter_map(|n| self.get(n))
            .collect()
    }

    pub fn build_default() -> Self {
        let mut reg = Self::new();
        reg.register(Box::new(DocumentationProvider));
        reg.register(Box::new(NewsProvider));
        reg.register(Box::new(StackOverflowProvider));
        reg.register(Box::new(RedditProvider));
        reg.register(Box::new(YouTubeProvider));
        reg.register(Box::new(PatentsProvider));
        reg.register(Box::new(GovernmentProvider));
        reg.register(Box::new(MedicalProvider));
        reg.register(Box::new(LegalProvider));
        reg.register(Box::new(EducationProvider));
        reg
    }
}

pub struct DocumentationProvider;
pub struct NewsProvider;
pub struct StackOverflowProvider;
pub struct RedditProvider;
pub struct YouTubeProvider;
pub struct PatentsProvider;
pub struct GovernmentProvider;
pub struct MedicalProvider;
pub struct LegalProvider;
pub struct EducationProvider;

macro_rules! impl_source_provider {
    ($name:ident, $name_str:expr, $url:expr, $cat:expr, $trust:expr, $desc:expr, $data:expr) => {
        #[async_trait]
        impl SourceProvider for $name {
            fn name(&self) -> &str { $name_str }
            fn url(&self) -> &str { $url }
            fn category(&self) -> SourceCategory { $cat }
            fn default_trust_score(&self) -> f64 { $trust }
            fn description(&self) -> &str { $desc }
            fn sample_data(&self) -> Vec<String> { $data }
        }
    };
}

impl_source_provider!(DocumentationProvider, "documentation", "https://docs.example.com",
    SourceCategory::Documentation, 0.85,
    "Technical documentation, API references, man pages, and guide-based learning materials",
    vec![
        "The `HashMap::insert` method inserts a key-value pair into the map. If the map already had a value for this key, the old value is returned.".to_string(),
        "To configure the server, edit the `config.toml` file and set the `port` parameter to your desired listening port number.".to_string(),
        "The gradient descent algorithm iteratively adjusts parameters to minimize the loss function by moving in the direction of steepest descent.".to_string(),
    ]
);

impl_source_provider!(NewsProvider, "news", "https://newsapi.org",
    SourceCategory::News, 0.70,
    "Curated news articles covering global events, technology, science, and business from reputable outlets",
    vec![
        "Scientists at CERN announced a breakthrough in quantum entanglement detection, enabling faster-than-light communication protocols in controlled environments.".to_string(),
        "The Federal Reserve raised interest rates by 25 basis points, citing persistent inflation and strong labor market data from the previous quarter.".to_string(),
        "A new renewable energy plant in Morocco will power over one million homes using concentrated solar power and advanced battery storage systems.".to_string(),
    ]
);

impl_source_provider!(StackOverflowProvider, "stackoverflow", "https://stackoverflow.com",
    SourceCategory::StackOverflow, 0.82,
    "Community-curated programming Q&A covering best practices, debugging patterns, and architectural decisions",
    vec![
        "Question: Why does my Rust program fail to compile with 'borrowed value does not live long enough'? Answer: The reference you're returning points to a local variable that goes out of scope at the end of the function.".to_string(),
        "Question: What is the difference between `unwrap()` and `expect()` in Rust? Answer: Both return the inner value for `Ok` or panic for `Err`, but `expect` lets you provide a custom panic message.".to_string(),
    ]
);

impl_source_provider!(RedditProvider, "reddit", "https://reddit.com",
    SourceCategory::Reddit, 0.45,
    "Community discussions across thousands of subreddits covering technology, science, and casual conversation",
    vec![
        "TIL that octopuses have three hearts, blue blood, and can change color in under a second. Two hearts pump blood to the gills, one pumps to the rest of the body.".to_string(),
        "Anyone else find that learning functional programming completely changed how you approach problems in imperative languages? Map/filter/reduce is now my default mental model.".to_string(),
    ]
);

impl_source_provider!(YouTubeProvider, "youtube", "https://youtube.com",
    SourceCategory::YouTube, 0.50,
    "Video transcriptions from educational, technical, and documentary content across diverse channels",
    vec![
        "Welcome back to another episode of Machine Learning 101. Today we're going to implement a transformer from scratch in PyTorch, starting with multi-headed attention.".to_string(),
        "In this video we'll build a REST API using Actix-web and SQLx. By the end you'll have a fully functional CRUD backend with PostgreSQL persistence.".to_string(),
    ]
);

impl_source_provider!(PatentsProvider, "patents", "https://patents.google.com",
    SourceCategory::Patents, 0.92,
    "Patent filings from USPTO, EPO, and WIPO covering technological innovations and prior art across all domains",
    vec![
        "A system and method for distributed ledger transaction validation using proof-of-stake consensus with sharded state storage and cross-shard communication channels.".to_string(),
        "Method for training neural networks using synthetic gradient prediction, wherein each layer computes approximate gradients without waiting for full backpropagation.".to_string(),
    ]
);

impl_source_provider!(GovernmentProvider, "government", "https://data.gov",
    SourceCategory::Government, 0.88,
    "Open government data portals providing structured datasets on demographics, economics, public health, and infrastructure",
    vec![
        "The 2024 census reports a population of 341.2 million, with a median age of 38.5 years and an urban population density of 93 people per square kilometer.".to_string(),
        "Annual energy consumption in the residential sector decreased by 3.2%, driven by improved insulation standards and widespread adoption of heat pump technology.".to_string(),
    ]
);

impl_source_provider!(MedicalProvider, "medical", "https://pubmed.ncbi.nlm.nih.gov",
    SourceCategory::Medical, 0.95,
    "Peer-reviewed biomedical literature from PubMed/MEDLINE covering clinical trials, case studies, and medical research",
    vec![
        "A randomized controlled trial of 1,204 patients evaluated the efficacy of mRNA-1273. A two-dose regimen demonstrated 94.1% efficacy against symptomatic infection.".to_string(),
        "Deep learning-based analysis of retinal fundus photographs can predict cardiovascular risk factors including age, blood pressure, and smoking status from images alone.".to_string(),
    ]
);

impl_source_provider!(LegalProvider, "legal", "https://law.cornell.edu",
    SourceCategory::Legal, 0.85,
    "Legal documents including court opinions, statutes, regulations, and contracts from US and international jurisdictions",
    vec![
        "The court held that the use of copyrighted material for training artificial intelligence models constitutes fair use when the purpose is transformative and the original work is not replicated in output.".to_string(),
        "Section 230 of the Communications Decency Act provides immunity for interactive computer services from liability arising from content posted by third-party users.".to_string(),
    ]
);

impl_source_provider!(EducationProvider, "education", "https://khanacademy.org",
    SourceCategory::Education, 0.88,
    "Curated educational content from Khan Academy, Coursera, edX spanning mathematics, science, humanities, and computer science",
    vec![
        "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides: a² + b² = c².".to_string(),
        "DNA replication begins at specific locations called origins of replication, where the double helix unwinds and each strand serves as a template for a new complementary strand.".to_string(),
    ]
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_builds() {
        let reg = SourceRegistry::build_default();
        assert_eq!(reg.all().len(), 10);
    }

    #[test]
    fn test_each_provider_has_data() {
        let reg = SourceRegistry::build_default();
        for provider in reg.all() {
            assert!(!provider.sample_data().is_empty(),
                "Provider '{}' has no sample data", provider.name());
        }
    }

    #[tokio::test]
    async fn test_fetch_samples() {
        let reg = SourceRegistry::build_default();
        let doc = reg.get("documentation").unwrap();
        let samples = doc.fetch_samples().await;
        assert!(!samples.is_empty());
        assert_eq!(samples[0].source.category, SourceCategory::Documentation);
    }

    #[test]
    fn test_registry_resolve() {
        let reg = SourceRegistry::build_default();
        let names = vec!["documentation".to_string(), "medical".to_string()];
        let resolved = reg.resolve(&names);
        assert_eq!(resolved.len(), 2);
    }

    #[test]
    fn test_source_info_fields() {
        let provider = DocumentationProvider;
        let info = provider.source_info();
        assert_eq!(info.category, SourceCategory::Documentation);
        assert!((info.trust_score - 0.85).abs() < 0.01);
        assert!(info.url.is_some());
    }
}
