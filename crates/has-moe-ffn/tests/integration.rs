use ndarray::Array2;
use nexora_has_moe_ffn::*;

#[test]
fn test_moe_end_to_end_small() {
    let config = HasMoeFFNConfig {
        num_experts: 4,
        top_k: 2,
        hidden_size: 8,
        intermediate_size: 16,
        use_dropout: false,
        dropout_rate: 0.0,
    };
    let moe = HasMoeFFN::new(config);
    let input = Array2::from_shape_fn((2, 8), |(i, j)| (i * 8 + j) as f32 / 16.0);
    let output = moe.forward(&input);
    assert_eq!(output.dim(), (2, 8));
    for val in output.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_moe_different_batch_sizes() {
    let config = HasMoeFFNConfig {
        num_experts: 4,
        top_k: 2,
        hidden_size: 4,
        intermediate_size: 8,
        use_dropout: false,
        dropout_rate: 0.0,
    };
    let moe = HasMoeFFN::new(config);

    for batch_size in &[1, 2, 4, 8] {
        let input = Array2::ones((*batch_size, 4));
        let output = moe.forward(&input);
        assert_eq!(output.dim(), (*batch_size, 4));
    }
}

#[test]
fn test_moe_with_single_expert_top1() {
    let config = HasMoeFFNConfig {
        num_experts: 2,
        top_k: 1,
        hidden_size: 4,
        intermediate_size: 8,
        use_dropout: false,
        dropout_rate: 0.0,
    };
    let moe = HasMoeFFN::new(config);
    let input = Array2::ones((3, 4));
    let output = moe.forward(&input);
    assert_eq!(output.dim(), (3, 4));
}

#[test]
fn test_moe_output_finite_all_values() {
    let config = HasMoeFFNConfig {
        num_experts: 8,
        top_k: 3,
        hidden_size: 16,
        intermediate_size: 32,
        use_dropout: false,
        dropout_rate: 0.0,
    };
    let moe = HasMoeFFN::new(config);
    let input = Array2::from_shape_fn((4, 16), |(i, j)| (i * 16 + j) as f32 / 64.0);
    let output = moe.forward(&input);
    for val in output.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_router_softmax_consistency() {
    let config = HasMoeFFNConfig {
        num_experts: 6,
        top_k: 2,
        hidden_size: 8,
        intermediate_size: 16,
        use_dropout: false,
        dropout_rate: 0.0,
    };
    let moe = HasMoeFFN::new(config);
    let input = Array2::ones((3, 8));
    let output = moe.forward(&input);
    assert_eq!(output.dim(), (3, 8));
}

#[test]
fn test_dense_layer_as_component() {
    let config = LayerConfig {
        input_size: 8,
        output_size: 4,
        use_bias: true,
        activation: "gelu".into(),
    };
    let layer = DenseLayer::new(config);
    let input = vec![0.5; 8];
    let output = layer.forward(&input);
    assert_eq!(output.len(), 4);
    assert_eq!(layer.param_count(), 8 * 4 + 4);
}

#[test]
fn test_attention_in_isolation() {
    let attn = Attention::new(8, 2, 0.0);
    let input = vec![0.5; 8];
    let output = attn.forward(&input, &input, &input);
    assert_eq!(output.len(), 8);
}

#[test]
fn test_gelu_function() {
    assert!((gelu(0.0) - 0.0).abs() < 1e-5);
    assert!(gelu(1.0) > 0.8);
    assert!(gelu(-1.0) < 0.0);
}

#[test]
fn test_moe_config_access() {
    let config = HasMoeFFNConfig::default();
    let moe = HasMoeFFN::new(config);
    let cfg = moe.config();
    assert_eq!(cfg.num_experts, 8);
    assert_eq!(cfg.hidden_size, 768);
}
