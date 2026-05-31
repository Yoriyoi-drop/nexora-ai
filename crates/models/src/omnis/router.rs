use crate::classifier_util;
use nexora_has_moe_ffn::Router;
use ndarray::Array2;
use std::sync::OnceLock;

pub const DOMAINS: [&str; 7] = [
    "math",
    "science",
    "code",
    "creative",
    "reasoning",
    "factual",
    "general",
];

static ROUTER: OnceLock<OmnisMoERouter> = OnceLock::new();

pub struct OmnisMoERouter {
    moe_router: Router,
    embed_table: Array2<f32>,
}

impl OmnisMoERouter {
    pub fn global() -> &'static Self {
        ROUTER.get_or_init(|| {
            Self {
                moe_router: Router::new(768, DOMAINS.len(), 2),
                embed_table: Array2::zeros((1, 768)),
            }
        })
    }

    pub fn init(embed_table: Array2<f32>) {
        let hidden_size = embed_table.shape()[1];
        ROUTER.set(Self {
            moe_router: Router::new(hidden_size, DOMAINS.len(), 2),
            embed_table,
        }).ok();
    }

    pub fn is_initialized(&self) -> bool {
        self.embed_table.shape()[0] > 1
    }

    pub fn predict(&self, token_ids: &[u32]) -> Vec<(String, f32)> {
        if token_ids.is_empty() || !self.is_initialized() {
            return vec![("general".to_string(), 1.0)];
        }

        let avg = classifier_util::embed_average(&self.embed_table, token_ids);
        let input = avg.clone().into_shape((1, avg.len())).expect("reshape 1D→2D with matching element count");
        let probs = self.moe_router.forward(&input);

        let mut results: Vec<_> = DOMAINS
            .iter()
            .enumerate()
            .map(|(i, d)| ((*d).to_string(), probs[[0, i]]))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

const EXPERT_PROMPTS: &[(&str, &str)] = &[
    ("math", "You are a mathematics expert. Solve step-by-step with clear reasoning."),
    ("science", "You are a science expert. Provide accurate, evidence-based explanations."),
    ("code", "You are a code expert. Write clean, idiomatic, well-documented code."),
    ("creative", "You are a creative writing expert. Be imaginative and engaging."),
    ("reasoning", "You are a logical reasoning expert. Analyze systematically from first principles."),
    ("factual", "You are a factual knowledge expert. Provide precise, well-sourced information."),
    ("general", "You are a general-purpose assistant. Respond helpfully and accurately."),
];

pub fn domain_system_prompt(domain: &str) -> &'static str {
    EXPERT_PROMPTS
        .iter()
        .find(|(d, _)| *d == domain)
        .map(|(_, p)| *p)
        .unwrap_or("You are a general-purpose assistant. Respond helpfully and accurately.")
}

pub fn detect_domains(text: &str, token_ids: &[u32]) -> Vec<(String, f32)> {
    let router = ROUTER.get().unwrap_or_else(OmnisMoERouter::global);
    if token_ids.is_empty() || !router.is_initialized() {
        return vec![("general".to_string(), 1.0)];
    }
    router.predict(token_ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn uninit() -> OmnisMoERouter {
        OmnisMoERouter {
            moe_router: Router::new(768, DOMAINS.len(), 2),
            embed_table: Array2::zeros((1, 768)),
        }
    }

    fn init_router(hidden: usize) -> OmnisMoERouter {
        let mut moe_router = Router::new(hidden, DOMAINS.len(), 2);
        moe_router.init_random();
        OmnisMoERouter {
            moe_router,
            embed_table: Array2::zeros((10, hidden)),
        }
    }

    #[test]
    fn test_global_not_initialized() {
        assert!(!OmnisMoERouter::global().is_initialized());
    }

    #[test]
    fn test_uninit_not_initialized() {
        assert!(!uninit().is_initialized());
    }

    #[test]
    fn test_init_initialized() {
        assert!(init_router(768).is_initialized());
    }

    #[test]
    fn test_predict_default_on_uninit() {
        assert_eq!(uninit().predict(&[1])[0].0, "general");
    }

    #[test]
    fn test_predict_empty_ids() {
        assert_eq!(init_router(768).predict(&[])[0].0, "general");
    }

    #[test]
    fn test_predict_returns_all_domains() {
        let r = init_router(768).predict(&[0, 1]);
        assert_eq!(r.len(), DOMAINS.len());
        let sum: f32 = r.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_detect_default_on_uninit() {
        assert_eq!(detect_domains("x", &[])[0].0, "general");
    }
}
