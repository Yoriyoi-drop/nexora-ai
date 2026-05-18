
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HallucinationReport {
    pub risk_level: String,
    pub score: f32,
    pub action: String,
    pub disclaimer: Option<String>,
}

pub struct HallucinationIntegration {
    #[cfg(feature = "hallucination")]
    guard: Option<nexora_hallucination::HallucinationGuard>,
    enabled: bool,
}

impl HallucinationIntegration {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "hallucination")]
            guard: None,
            enabled: false,
        }
    }

    #[cfg(feature = "hallucination")]
    pub fn with_guard(guard: nexora_hallucination::HallucinationGuard) -> Self {
        Self {
            guard: Some(guard),
            enabled: true,
        }
    }

    pub fn enable(&mut self) {
        self.enabled = true;
        #[cfg(feature = "hallucination")]
        if self.guard.is_none() {
            self.guard = Some(nexora_hallucination::HallucinationGuard::new(
                nexora_hallucination::GuardConfig::default(),
            ));
        }
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn is_enabled(&self) -> bool {
        if !self.enabled {
            return false;
        }
        #[cfg(feature = "hallucination")]
        {
            return self.guard.is_some();
        }
        #[cfg(not(feature = "hallucination"))]
        {
            return true;
        }
    }

    pub async fn check_input(&self, text: &str, ctx: Option<&str>) -> Option<HallucinationReport> {
        if !self.is_enabled() {
            return None;
        }

        #[cfg(feature = "hallucination")]
        {
            let guard = self.guard.as_ref().unwrap();
            let result = guard.run_pipeline(text, ctx, None).await;
            return match result {
                Ok(r) => Some(HallucinationReport {
                    risk_level: format!("{:?}", r.risk_level),
                    score: r.score,
                    action: format!("{:?}", r.action),
                    disclaimer: match r.risk_level {
                        nexora_hallucination::RiskLevel::Medium => {
                            Some("Harap verifikasi informasi ini — please verify this information.".to_string())
                        }
                        nexora_hallucination::RiskLevel::High
                        | nexora_hallucination::RiskLevel::Critical => {
                            Some("Output ini memerlukan verifikasi sebelum digunakan.".to_string())
                        }
                        _ => None,
                    },
                }),
                Err(e) => {
                    tracing::warn!("Hallucination check failed: {e}");
                    None
                }
            };
        }

        #[cfg(not(feature = "hallucination"))]
        None
    }

    #[cfg(feature = "hallucination")]
    pub fn wrap_system_prompt(&self, base_prompt: &str, language: &str) -> String {
        let anti_prompt = if language == "id" {
            nexora_hallucination::SystemPromptManager::default_prompt_indonesian()
        } else {
            nexora_hallucination::SystemPromptManager::default_prompt()
        };
        format!("{}\n\n{}", base_prompt, anti_prompt)
    }

    #[cfg(not(feature = "hallucination"))]
    pub fn wrap_system_prompt(&self, base_prompt: &str, _language: &str) -> String {
        base_prompt.to_string()
    }
}

impl Default for HallucinationIntegration {
    fn default() -> Self {
        Self::new()
    }
}
