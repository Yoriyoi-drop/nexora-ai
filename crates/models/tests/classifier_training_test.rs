use ndarray::{Array1, Array2};
use nexora_models::classifier_util;

fn make_classifier(embed_dim: usize, hidden: usize, num_classes: usize) -> (Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>) {
    let w1 = classifier_util::xavier_init(embed_dim, hidden);
    let b1 = Array1::zeros(hidden);
    let w2 = classifier_util::xavier_init(hidden, num_classes);
    let b2 = Array1::zeros(num_classes);
    (w1, b1, w2, b2)
}

fn make_data(n: usize, embed_dim: usize, num_classes: usize) -> (Array2<f32>, Array2<f32>) {
    let mut inputs = Array2::zeros((n, embed_dim));
    let mut labels = Array2::zeros((n, num_classes));
    for i in 0..n {
        inputs[[i, i % embed_dim]] = 1.0;
        labels[[i, i % num_classes]] = 1.0;
    }
    (inputs, labels)
}

#[test]
fn test_softmax_sums_to_one() {
    let x = Array1::from(vec![2.0, 1.0, 0.1, -1.0, 3.0]);
    let p = classifier_util::softmax(&x);
    let sum: f32 = p.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!(p.iter().all(|&v| v > 0.0));
}

#[test]
fn test_gelu_shape_preserved() {
    let x = Array1::from(vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
    let y = classifier_util::gelu(&x);
    assert_eq!(y.shape(), x.shape());
    assert!(y[2].abs() < 1e-6, "gelu(0) should be near 0");
    assert!(y[3] > 0.0, "gelu(positive) should be positive");
    assert!(y[4] > y[3], "gelu should increase after 0");
    assert!(y[5] > y[4], "gelu should increase after 0");
}

#[test]
fn test_xavier_init_shape() {
    let w = classifier_util::xavier_init(128, 64);
    assert_eq!(w.shape(), &[128, 64]);
}

#[test]
fn test_embed_average() {
    let mut embed = Array2::zeros((10, 4));
    embed[[0, 0]] = 2.0;
    embed[[0, 1]] = 4.0;
    embed[[0, 2]] = 6.0;
    embed[[0, 3]] = 8.0;
    let avg = classifier_util::embed_average(&embed, &[0]);
    assert!((avg[0] - 2.0).abs() < 1e-6);
    assert!((avg[1] - 4.0).abs() < 1e-6);
}

#[test]
fn test_classifier_weights_save_load() {
    let (w1, b1, w2, b2) = make_classifier(16, 8, 4);
    let tmp = std::env::temp_dir().join("test_clf_weights.json");
    let path = tmp.to_str().unwrap();

    classifier_util::save_classifier_weights(path, &w1, &b1, &w2, &b2).unwrap();
    let loaded = classifier_util::load_classifier_weights(path).unwrap();

    let w1_loaded = loaded.to_w1();
    let b1_loaded = loaded.to_b1();
    let w2_loaded = loaded.to_w2();
    let b2_loaded = loaded.to_b2();

    assert_eq!(w1.shape(), w1_loaded.shape());
    assert_eq!(b1.len(), b1_loaded.len());
    assert!(w1.iter().zip(w1_loaded.iter()).all(|(a, b)| (a - b).abs() < 1e-6));
    assert!(b1.iter().zip(b1_loaded.iter()).all(|(a, b)| (a - b).abs() < 1e-6));

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_train_classifier_sgd_finite_loss() {
    let num_classes = 3;
    let embed_dim = 8;
    let hidden = 16;
    let n = 10;

    let (mut w1, mut b1, mut w2, mut b2) = make_classifier(embed_dim, hidden, num_classes);
    let (inputs, labels) = make_data(n, embed_dim, num_classes);

    // Train
    let loss = classifier_util::train_classifier_sgd(
        &mut w1, &mut b1, &mut w2, &mut b2, &inputs, &labels, 0.1, 50,
    );
    assert!(loss.is_finite(), "loss should be finite, got {}", loss);
    // Loss over 50 steps with 10 samples: expect some reduction from random init
    assert!(loss > 0.0, "loss should be positive, got {}", loss);
}

#[test]
fn test_train_classifier_sgd_empty_data() {
    let (mut w1, mut b1, mut w2, mut b2) = make_classifier(8, 16, 3);
    let inputs = Array2::zeros((0, 8));
    let labels = Array2::zeros((0, 3));
    let loss = classifier_util::train_classifier_sgd(
        &mut w1, &mut b1, &mut w2, &mut b2, &inputs, &labels, 0.1, 10,
    );
    assert_eq!(loss, 0.0, "empty data should give 0 loss");
}
