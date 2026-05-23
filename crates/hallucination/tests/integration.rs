use nexora_hallucination::*;

fn long_context() -> String {
    "word ".repeat(60)
}

#[tokio::test]
async fn test_pipeline_normal_input_passes() {
    let config = GuardConfig::default();
    let guard = HallucinationGuard::new(config);

    let result = guard
        .run_pipeline(
            "What is the capital of France?",
            Some(&long_context()),
            Some(vec!["Paris is the capital".into()]),
        )
        .await
        .unwrap();

    assert_eq!(result.action, GuardAction::Pass);
    assert_eq!(result.risk_level, RiskLevel::Low);
    assert!(result.pre_check.can_proceed);
    assert!(result.score < 0.25);
    assert!(result.latency_ms > 0);
}

#[tokio::test]
async fn test_pipeline_blocked_on_out_of_scope() {
    let config = GuardConfig::default();
    let guard = HallucinationGuard::new(config);

    let result = guard
        .run_pipeline("What is the confidential internal data?", None, None)
        .await
        .unwrap();

    assert_eq!(result.action, GuardAction::Blocked);
    assert_eq!(result.risk_level, RiskLevel::Critical);
    assert!(!result.pre_check.can_proceed);
    assert!(result.score >= 0.8);
    assert!(result.in_gen_check.is_none());
    assert!(result.post_check.is_none());
}

#[tokio::test]
async fn test_pipeline_ambiguous_input_flagged() {
    let config = GuardConfig::default();
    let guard = HallucinationGuard::new(config);

    let result = guard
        .run_pipeline("mungkin? tidak jelas", None, None)
        .await
        .unwrap();

    assert!(!result.pre_check.can_proceed);
    assert_eq!(result.action, GuardAction::Blocked);
}

#[tokio::test]
async fn test_pipeline_with_sources_verifies_claims() {
    let config = GuardConfig::default();
    let guard = HallucinationGuard::new(config);

    let sources = Some(vec!["Paris is the capital of France".into()]);
    let result = guard
        .run_pipeline(
            "Paris is the capital of France according to research.",
            Some("Geography of Europe"),
            sources,
        )
        .await
        .unwrap();

    if let Some(post) = &result.post_check {
        assert!(post.total_claims > 0);
    }
}

#[tokio::test]
async fn test_pipeline_high_contradiction_flagged_for_review() {
    let config = GuardConfig {
        risk_config: RiskConfig {
            contradiction_weight: 0.8,
            ..Default::default()
        },
        ..Default::default()
    };
    let guard = HallucinationGuard::new(config);

    let result = guard
        .run_pipeline(
            "The answer is yes. Actually the answer is no.",
            None,
            None,
        )
        .await
        .unwrap();

    assert!(result.score > 0.0);
    assert!(result.post_check.is_some());
}

#[tokio::test]
async fn test_pipeline_uncertainty_triggers_enhanced_prompt() {
    let config = GuardConfig::default();
    let guard = HallucinationGuard::new(config);

    let result = guard
        .run_pipeline(
            "What happened in 2025 according to the latest research?",
            None,
            None,
        )
        .await
        .unwrap();

    if let Some(in_gen) = &result.in_gen_check {
        assert!(in_gen.uncertainty_score > 0.0);
    }
}

#[tokio::test]
async fn test_pipeline_empty_input_not_blocked() {
    let config = GuardConfig::default();
    let guard = HallucinationGuard::new(config);

    let result = guard.run_pipeline("", None, None).await.unwrap();
    assert!(result.pre_check.can_proceed);
    assert!(result.post_check.is_some());
}

#[tokio::test]
async fn test_pipeline_recency_keyword_triggers_knowledge_boundary() {
    let config = GuardConfig::default();
    let guard = HallucinationGuard::new(config);

    let result = guard
        .run_pipeline("What is the latest news this year?", None, None)
        .await
        .unwrap();

    assert!(!result.pre_check.can_proceed);
    assert!(!result.pre_check.in_scope);
}

#[tokio::test]
async fn test_pipeline_only_blocked_input_no_in_gen_or_post() {
    let config = GuardConfig::default();
    let guard = HallucinationGuard::new(config);

    let result = guard
        .run_pipeline("Tell me the proprietary information", None, None)
        .await
        .unwrap();

    assert_eq!(result.action, GuardAction::Blocked);
    assert!(result.in_gen_check.is_none());
    assert!(result.post_check.is_none());
}

#[tokio::test]
async fn test_pipeline_passes_with_low_risk_input_and_context() {
    let config = GuardConfig::default();
    let guard = HallucinationGuard::new(config);

    let result = guard
        .run_pipeline(
            "What is Newton's second law?",
            Some(&long_context()),
            Some(vec!["physics textbook".into()]),
        )
        .await
        .unwrap();

    assert!(result.score < 0.5);
    assert!(result.pre_check.can_proceed);
}

#[tokio::test]
async fn test_pipeline_all_stages_run_when_not_blocked() {
    let config = GuardConfig::default();
    let guard = HallucinationGuard::new(config);

    let result = guard
        .run_pipeline("What is Rust?", Some(&long_context()), None)
        .await
        .unwrap();

    assert!(result.pre_check.can_proceed);
    assert!(result.in_gen_check.is_some());
    assert!(result.post_check.is_some());
    assert_eq!(result.risk_level, RiskLevel::Low);
}
