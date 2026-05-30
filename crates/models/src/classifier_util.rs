use ndarray::{Array1, Array2};

pub fn xavier_init(rows: usize, cols: usize) -> Array2<f32> {
    let scale = (1.0 / rows as f32).sqrt();
    use rand::Rng;
    let mut rng = rand::thread_rng();
    Array2::from_shape_simple_fn((rows, cols), || rng.gen::<f32>() * 2.0 * scale - scale)
}

pub fn xavier_init_seeded(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let scale = (1.0 / rows as f32).sqrt();
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    Array2::from_shape_simple_fn((rows, cols), || rng.gen::<f32>() * 2.0 * scale - scale)
}

pub fn gelu(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| {
        0.5 * v * (1.0 + (v * 0.7978845608028654 * (1.0 + 0.044715 * v * v)).tanh())
    })
}

pub fn softmax(x: &Array1<f32>) -> Array1<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    Array1::from_iter(exps.into_iter().map(|e| e / sum))
}

pub fn embed_average(embed_table: &Array2<f32>, token_ids: &[u32]) -> Array1<f32> {
    let embed_dim = embed_table.shape()[1];
    let vocab = embed_table.shape()[0];
    if token_ids.is_empty() {
        return Array1::zeros(embed_dim);
    }
    let mut avg = Array1::zeros(embed_dim);
    let mut count = 0usize;
    for &tid in token_ids {
        let idx = (tid as usize).min(vocab.saturating_sub(1));
        let row = embed_table.row(idx);
        for j in 0..embed_dim {
            avg[j] += row[j];
        }
        count += 1;
    }
    if count > 0 {
        avg.mapv_inplace(|v| v / count as f32);
    }
    avg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xavier_init_seeded_shape() {
        let w = xavier_init_seeded(10, 5, 0);
        assert_eq!(w.shape(), &[10, 5]);
    }

    #[test]
    fn test_xavier_init_shape() {
        let w = xavier_init(10, 5);
        assert_eq!(w.shape(), &[10, 5]);
    }

    #[test]
    fn test_xavier_init_seeded_deterministic() {
        let a = xavier_init_seeded(10, 5, 42);
        let b = xavier_init_seeded(10, 5, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_xavier_init_seeded_different_seed() {
        let a = xavier_init_seeded(10, 5, 42);
        let b = xavier_init_seeded(10, 5, 99);
        assert_ne!(a, b);
    }

    #[test]
    fn test_xavier_init_range() {
        let w = xavier_init(100, 50);
        let scale = (1.0 / 100.0_f32).sqrt();
        for val in w.iter() {
            assert!(*val >= -scale && *val <= scale);
        }
    }

    #[test]
    fn test_gelu_shape() {
        let x = Array1::from(vec![0.0, 1.0, -1.0]);
        let y = gelu(&x);
        assert_eq!(y.len(), 3);
    }

    #[test]
    fn test_gelu_zero() {
        let y = gelu(&Array1::from(vec![0.0]));
        let expected = 0.0;
        assert!((y[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_gelu_monotonic_nonnegative() {
        let x = Array1::from(vec![-0.5, 0.0, 0.5, 1.0, 2.0]);
        let y = gelu(&x);
        for i in 1..y.len() {
            assert!(y[i] >= y[i - 1], "gelu must be monotonic for x >= -0.5");
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let x = Array1::from(vec![2.0, 1.0, 0.1]);
        let p = softmax(&x);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_shape_preserved() {
        let x = Array1::from(vec![0.5, -1.0, 3.0, 0.0]);
        let p = softmax(&x);
        assert_eq!(p.len(), 4);
    }

    #[test]
    fn test_embed_average_empty() {
        let embed = Array2::zeros((10, 4));
        let avg = embed_average(&embed, &[]);
        assert_eq!(avg.len(), 4);
        assert!(avg.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_embed_average_single_id() {
        let mut embed = Array2::zeros((3, 2));
        embed[[1, 0]] = 2.0;
        embed[[1, 1]] = 4.0;
        let avg = embed_average(&embed, &[1]);
        assert!((avg[0] - 2.0).abs() < 1e-6);
        assert!((avg[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_embed_average_multiple_ids() {
        let mut embed = Array2::zeros((3, 2));
        embed[[0, 0]] = 1.0;
        embed[[0, 1]] = 2.0;
        embed[[1, 0]] = 3.0;
        embed[[1, 1]] = 4.0;
        let avg = embed_average(&embed, &[0, 1]);
        assert!((avg[0] - 2.0).abs() < 1e-6);
        assert!((avg[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_embed_average_oob_clamp() {
        let mut embed = Array2::zeros((2, 2));
        embed[[1, 0]] = 5.0;
        embed[[1, 1]] = 6.0;
        let avg = embed_average(&embed, &[99]);
        assert!((avg[0] - 5.0).abs() < 1e-6);
        assert!((avg[1] - 6.0).abs() < 1e-6);
    }
}
