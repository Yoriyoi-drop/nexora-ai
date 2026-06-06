use nexora_shared::base_model::NxrModelResult;

#[derive(Debug, Clone, Default)]
pub struct SynthPrimeRuntimeAgent;

impl SynthPrimeRuntimeAgent {
    pub fn new() -> Self {
        Self
    }

    /// Synthesize fragments into a cohesive result.
    ///
    /// FUTURE: Will delegate to foundation CausalLM with a synthesis prompt.
    /// The previous implementation derived coherence from unique-word ratio
    /// — that was not a real synthesis operation.
    pub fn synthesize(&self, _fragments: &[String]) -> NxrModelResult<String> {
        Ok("[synthesis will delegate to foundation model]".to_string())
    }
}
