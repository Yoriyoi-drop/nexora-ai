use nexora_foundation::shared::model_identity::NxrModelId;

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
}

// ── Result ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RouteResult {
    pub model_id: NxrModelId,
    pub intent: IntentKind,
    pub confidence: f32,
}

// ── IntentRouter (formerly TierRouter) ─────────────────────────────────────

/// IntentRouter — simple intent classifier tanpa tier / VRAM tracking.
/// Hanya mapping: prompt → IntentKind → NxrModelId.
/// Satu shared backbone CausalLM dipakai semua model.
#[derive(Clone)]
pub struct IntentRouter;

impl IntentRouter {
    pub fn new() -> Self {
        Self
    }

    pub fn route(&self, prompt: &str) -> RouteResult {
        let intent = IntentKind::classify(prompt);
        RouteResult {
            model_id: intent.target_model(),
            intent,
            confidence: 0.85,
        }
    }

    /// Deteksi apakah prompt ini butuh multi-model debate
    pub fn requires_debate(&self, prompt: &str, intent: IntentKind) -> bool {
        let lower = prompt.to_lowercase();

        // Explicit debate keywords — selalu trigger debate
        let debate_keywords = [
            "debat", "diskusi", "musyawarah", "forum",
            "berunding", "brainstorm", "berdiskusi",
        ];
        if debate_keywords.iter().any(|k| lower.contains(k)) {
            return true;
        }

        // Comparison & evaluation keywords
        let compare_keywords = [
            "bandingkan", "perbedaan", "persamaan", "vs ",
            "versus", "kelebihan", "kekurangan", "pro kontra",
            "compare", "contrast", "difference", "similarity",
            "pro and cons", "trade-off", "alternatif",
        ];
        if compare_keywords.iter().any(|k| lower.contains(k)) {
            return true;
        }

        // Complex analysis keywords
        let analysis_keywords = [
            "analisis", "evaluasi", "kaji", "telaah",
            "review mendalam", "komprehensif",
            "multi-perspektif", "berbagai sudut pandang",
            "menurut para ahli", "expert opinion",
            "dampak", "implikasi", "rekomendasi",
        ];
        if analysis_keywords.iter().any(|k| lower.contains(k)) {
            return true;
        }

        // Decision-making keywords
        let decision_keywords = [
            "keputusan", "decision", "strategi", "strategy",
            "rencana", "planning", "solusi terbaik",
            "best approach", "recommendation",
        ];
        if decision_keywords.iter().any(|k| lower.contains(k)) {
            return true;
        }

        // Uncertainty markers — prompt butuh multiple perspectives
        let uncertainty_keywords = [
            "mungkin", "maybe", "perhaps", "tidak yakin",
            "uncertain", "ambiguous", "kompleks", "complex",
            "sulit", "difficult", "challenging", "kontroversial",
            "controversial", "berpotensi", "potential risk",
        ];
        if uncertainty_keywords.iter().any(|k| lower.contains(k)) {
            return true;
        }

        // Intent-based: Strategy, Reasoning, Security selalu butuh debate
        matches!(
            intent,
            IntentKind::Strategy
                | IntentKind::Reasoning
                | IntentKind::Security
        )
    }

    /// Suggest model participants for debate berdasarkan intent
    pub fn suggest_participants(&self, prompt: &str, primary: NxrModelId, intent: IntentKind) -> Vec<NxrModelId> {
        let lower = prompt.to_lowercase();
        let mut extra: Vec<NxrModelId> = Vec::new();

        // Code review + security overlap
        if lower.contains("security") || lower.contains("vulnerability")
            || lower.contains("xss") || lower.contains("injection")
        {
            extra.push(NxrModelId::Cipher);
        }

        // Code + creative overlap
        if lower.contains("ui") || lower.contains("design") || lower.contains("frontend")
            || lower.contains("interface") || lower.contains("ux")
        {
            extra.push(NxrModelId::Spectra);
        }

        // Complex reasoning + knowledge
        if lower.contains("sejarah") || lower.contains("data") || lower.contains("penelitian")
            || lower.contains("research") || lower.contains("historical")
        {
            extra.push(NxrModelId::Kronos);
        }

        // Human-centric + emotion
        if lower.contains("orang") || lower.contains("masyarakat") || lower.contains("social")
            || lower.contains("people") || lower.contains("human") || lower.contains("etika")
        {
            extra.push(NxrModelId::Aether);
        }

        // Self-improvement / iterative
        if lower.contains("improve") || lower.contains("optimasi") || lower.contains("refactor")
            || lower.contains("iteration") || lower.contains("iterasi")
        {
            extra.push(NxrModelId::Genesis);
        }

        // Multi-agent coordination
        if lower.contains("orchestrasi") || lower.contains("workflow") || lower.contains("pipeline")
            || lower.contains("multi-step") || lower.contains("complex task")
        {
            extra.push(NxrModelId::Nexum);
        }

        use crate::core::debate::suggest_participants_for_intent;
        let mut all = suggest_participants_for_intent(&intent, primary);
        for m in extra {
            if !all.contains(&m) {
                all.push(m);
            }
        }
        all
    }
}

// ── Backward compat alias ──────────────────────────────────────────────────

pub type TierRouter = IntentRouter;
pub type TierRouterConfig = ();
