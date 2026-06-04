use nexora_foundation::shared::model_identity::NxrModelId;
use nexora_models::foundation::transformer_config_for;

#[test]
fn test_transformer_config_all_models_have_sane_values() {
    for id in NxrModelId::all() {
        let config = transformer_config_for(id);
        assert!(config.hidden_size > 0, "hidden_size zero for {:?}", id);
        assert!(config.num_heads > 0, "num_heads zero for {:?}", id);
        assert!(config.num_kv_heads > 0, "num_kv_heads zero for {:?}", id);
        assert!(config.num_layers > 0, "num_layers zero for {:?}", id);
        assert!(config.max_seq_len > 0, "max_seq_len zero for {:?}", id);
        assert!(config.intermediate_size > 0, "intermediate_size zero for {:?}", id);
        assert!(config.num_kv_heads <= config.num_heads, "kv_heads > heads for {:?}", id);
    }
}

#[test]
fn test_transformer_config_tier_consistency() {
    let omnis = transformer_config_for(NxrModelId::Omnis);
    let axiom = transformer_config_for(NxrModelId::Axiom);
    let genesis = transformer_config_for(NxrModelId::Genesis);
    let nexum = transformer_config_for(NxrModelId::Nexum);
    let low = transformer_config_for(NxrModelId::Swift);

    // Omnis > Axiom > Genesis/Nexum > Low-tier
    assert!(omnis.hidden_size > axiom.hidden_size,
        "Omnis hidden ({}) should be > Axiom ({})", omnis.hidden_size, axiom.hidden_size);
    assert!(axiom.hidden_size >= genesis.hidden_size,
        "Axiom hidden ({}) should be >= Genesis ({})", axiom.hidden_size, genesis.hidden_size);
    assert_eq!(genesis.hidden_size, nexum.hidden_size,
        "Genesis and Nexum should have same hidden size");
    assert!(genesis.hidden_size > low.hidden_size,
        "Genesis hidden ({}) should be > Low-tier ({})", genesis.hidden_size, low.hidden_size);
}

#[test]
fn test_transformer_config_flagship_vs_edge() {
    let omnis = transformer_config_for(NxrModelId::Omnis);
    let swift = transformer_config_for(NxrModelId::Swift);

    assert_eq!(omnis.num_heads, 8, "Omnis should have 8 heads");
    assert_eq!(omnis.num_layers, 16, "Omnis should have 16 layers");
    assert_eq!(swift.num_heads, 4, "Swift should have 4 heads");
    assert_eq!(swift.num_layers, 3, "Swift should have 3 layers");
}

#[test]
fn test_transformer_config_vocab_size_positive() {
    let config = transformer_config_for(NxrModelId::Cipher);
    assert!(config.vocab_size > 0, "vocab_size should be positive");
    let omnis = transformer_config_for(NxrModelId::Omnis);
    assert_eq!(config.vocab_size, omnis.vocab_size, "all models should share vocab_size");
}

#[test]
fn test_transformer_config_different_tiers_different() {
    let all: Vec<NxrModelId> = NxrModelId::all();
    // Group by tier via hidden_size
    let by_tier: Vec<Vec<NxrModelId>> = {
        let mut groups: Vec<(usize, Vec<NxrModelId>)> = Vec::new();
        for &id in &all {
            let h = transformer_config_for(id).hidden_size;
            if let Some(g) = groups.iter_mut().find(|(k, _)| *k == h) {
                g.1.push(id);
            } else {
                groups.push((h, vec![id]));
            }
        }
        groups.into_iter().map(|(_, v)| v).collect()
    };

    // Same-tier models should have identical hidden_size + num_layers
    for group in &by_tier {
        for i in 1..group.len() {
            let a = transformer_config_for(group[0]);
            let b = transformer_config_for(group[i]);
            assert_eq!(a.hidden_size, b.hidden_size,
                "Same-tier {:?} vs {:?}: hidden mismatch", group[0], group[i]);
            assert_eq!(a.num_layers, b.num_layers,
                "Same-tier {:?} vs {:?}: layers mismatch", group[0], group[i]);
        }
    }

    // Different tiers should have different hidden_size or num_layers
    for i in 0..by_tier.len() {
        for j in (i + 1)..by_tier.len() {
            let a = transformer_config_for(by_tier[i][0]);
            let b = transformer_config_for(by_tier[j][0]);
            assert!(
                a.hidden_size != b.hidden_size || a.num_layers != b.num_layers,
                "Different tiers {:?} and {:?} have identical config (hidden={}, layers={})",
                by_tier[i][0], by_tier[j][0], a.hidden_size, a.num_layers
            );
        }
    }
}
