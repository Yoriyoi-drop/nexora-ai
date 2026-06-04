use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use nexora_foundation::shared::model_identity::NxrModelId;
use nexora_transformer::config::ModelTier;
use nexora_transformer::tier_vram_estimate_mb as compute_tier_vram;

// ── Intent Classification ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntentKind {
    CodeReview,
    CodeGenerate,
    Emotion,
    Creative,
    Security,
    Reasoning,
    Factual,
    Strategy,
    QuickQuery,
    Knowledge,
    General,
}

impl IntentKind {
    fn classify(text: &str) -> Self {
        let lower = text.to_lowercase();

        if lower.contains("debug")
            || lower.contains("review")
            || lower.contains("rust")
            || lower.contains("python")
            || lower.contains("compile")
            || lower.contains("refactor")
            || lower.contains("bug")
            || lower.contains("syntax")
            || lower.contains("code")
        {
            return if lower.contains("generate") || lower.contains("tulis") || lower.contains("buat")
            {
                IntentKind::CodeGenerate
            } else {
                IntentKind::CodeReview
            };
        }

        if lower.contains("sedih")
            || lower.contains("senang")
            || lower.contains("marah")
            || lower.contains("emosi")
            || lower.contains("feeling")
            || lower.contains("sad")
            || lower.contains("happy")
            || lower.contains("angry")
            || lower.contains("takut")
            || lower.contains("cemas")
        {
            return IntentKind::Emotion;
        }

        if lower.contains("puisi")
            || lower.contains("cerita")
            || lower.contains("gambar")
            || lower.contains("creative")
            || lower.contains("tulisan")
            || lower.contains("story")
            || lower.contains("poem")
            || lower.contains("narrative")
        {
            return IntentKind::Creative;
        }

        if lower.contains("security")
            || lower.contains("xss")
            || lower.contains("injection")
            || lower.contains("sql")
            || lower.contains("hack")
            || lower.contains("vulnerability")
            || lower.contains("crack")
            || lower.contains("threat")
        {
            return IntentKind::Security;
        }

        if lower.contains("strategi")
            || lower.contains("keputusan")
            || lower.contains("decision")
            || lower.contains("plan")
            || lower.contains("rencana")
            || lower.contains("analisis")
            || lower.contains("bisnis")
        {
            return IntentKind::Strategy;
        }

        if lower.contains("jam")
            || lower.contains("cuaca")
            || lower.contains("weather")
            || lower.contains("time")
            || lower.contains("sekarang")
            || lower.contains("tanggal")
            || lower.contains("date")
            || lower.contains("berapa")
        {
            return IntentKind::QuickQuery;
        }

        if lower.contains("sejarah")
            || lower.contains("data")
            || lower.contains("fakta")
            || lower.contains("informasi")
            || lower.contains("knowledge")
            || lower.contains("archive")
        {
            return IntentKind::Knowledge;
        }

        if lower.contains("jelaskan")
            || lower.contains("bagaimana")
            || lower.contains("apa itu")
            || lower.contains("mengapa")
            || lower.contains("kenapa")
            || lower.contains("explain")
            || lower.contains("why")
            || lower.contains("how")
            || lower.contains("what is")
            || lower.contains("teori")
            || lower.contains("konsep")
        {
            return IntentKind::Reasoning;
        }

        IntentKind::General
    }

    fn target_model(&self) -> NxrModelId {
        match self {
            IntentKind::CodeReview | IntentKind::CodeGenerate => NxrModelId::Vortex,
            IntentKind::Emotion => NxrModelId::Aether,
            IntentKind::Creative => NxrModelId::Spectra,
            IntentKind::Security => NxrModelId::Cipher,
            IntentKind::Strategy => NxrModelId::Axiom,
            IntentKind::QuickQuery => NxrModelId::Swift,
            IntentKind::Knowledge => NxrModelId::Kronos,
            IntentKind::Reasoning | IntentKind::Factual | IntentKind::General => NxrModelId::Omnis,
        }
    }

    fn target_tier(&self) -> ModelTier {
        match self {
            IntentKind::CodeReview | IntentKind::CodeGenerate | IntentKind::Emotion => {
                ModelTier::Apex
            }
            IntentKind::Creative | IntentKind::Security => ModelTier::Pro,
            IntentKind::Strategy => ModelTier::Ultra,
            IntentKind::QuickQuery => ModelTier::Edge,
            IntentKind::Knowledge => ModelTier::Core,
            IntentKind::Reasoning | IntentKind::Factual | IntentKind::General => ModelTier::Ultra,
        }
    }
}

// ── LRU Tracking (global, shared via static) ──────────────────────────────

/// LRU list: most recently used at the end. Max 5 tiers.
static TIER_LRU: OnceLock<Mutex<Vec<(ModelTier, Instant)>>> = OnceLock::new();
static TIER_HIT_COUNT: OnceLock<Mutex<Vec<(ModelTier, u64)>>> = OnceLock::new();

fn tier_lru() -> &'static Mutex<Vec<(ModelTier, Instant)>> {
    TIER_LRU.get_or_init(|| Mutex::new(Vec::new()))
}

fn tier_hit_count() -> &'static Mutex<Vec<(ModelTier, u64)>> {
    TIER_HIT_COUNT.get_or_init(|| {
        Mutex::new(vec![
            (ModelTier::Ultra, 0),
            (ModelTier::Apex, 0),
            (ModelTier::Pro, 0),
            (ModelTier::Core, 0),
            (ModelTier::Edge, 0),
        ])
    })
}

// ── Config ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TierRouterConfig {
    pub max_tiers_in_vram: usize,
    pub vram_budget_mb: u64,
    pub eviction_threshold: f32,
}

impl Default for TierRouterConfig {
    fn default() -> Self {
        Self {
            max_tiers_in_vram: 1,
            vram_budget_mb: 24000,
            eviction_threshold: 0.85,
        }
    }
}

// ── Result ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RouteResult {
    pub model_id: NxrModelId,
    pub tier: ModelTier,
    pub intent: IntentKind,
    pub confidence: f32,
}

// ── TierRouter ────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct TierRouter {
    config: TierRouterConfig,
}

impl TierRouter {
    pub fn new(config: TierRouterConfig) -> Self {
        Self { config }
    }

    pub fn route(&self, prompt: &str) -> RouteResult {
        let intent = IntentKind::classify(prompt);
        RouteResult {
            model_id: intent.target_model(),
            tier: intent.target_tier(),
            intent,
            confidence: 0.85,
        }
    }

    pub fn mark_tier_used(&self, tier: ModelTier) {
        if let Ok(mut lru) = tier_lru().lock() {
            lru.retain(|(t, _)| *t != tier);
            lru.push((tier, Instant::now()));
        }
        if let Ok(mut hits) = tier_hit_count().lock() {
            for (t, count) in hits.iter_mut() {
                if *t == tier {
                    *count += 1;
                    break;
                }
            }
        }
    }

    pub fn lru_eviction_candidate(&self) -> Option<ModelTier> {
        let lru = tier_lru().lock().ok()?;
        lru.first().map(|(t, _)| *t)
    }

    pub fn should_evict(&self, current_vram_mb: u64) -> bool {
        current_vram_mb as f32 > self.config.vram_budget_mb as f32 * self.config.eviction_threshold
    }

    pub fn loaded_tiers(&self) -> Vec<ModelTier> {
        tier_lru()
            .lock()
            .map(|lru| lru.iter().map(|(t, _)| *t).collect())
            .unwrap_or_default()
    }

    pub fn hit_counts(&self) -> Vec<(ModelTier, u64)> {
        tier_hit_count()
            .lock()
            .map(|h| h.clone())
            .unwrap_or_default()
    }

    pub fn config(&self) -> &TierRouterConfig {
        &self.config
    }

    /// VRAM estimasi per tier (dalam MB, sesuai konfigurasi model)
    pub fn vram_estimate_mb(&self, tier: ModelTier) -> u64 {
        compute_tier_vram(tier)
    }

    /// Total VRAM semua tier yang sedang terload
    pub fn total_vram_used_mb(&self) -> u64 {
        self.loaded_tiers()
            .iter()
            .map(|t| self.vram_estimate_mb(*t))
            .sum()
    }
}
