use ndarray::Array2;
use rand::Rng;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

static EMBEDDING_REGISTRY: OnceLock<RwLock<HashMap<(usize, usize, u64), Arc<Array2<f32>>>>> =
    OnceLock::new();

fn registry() -> &'static RwLock<HashMap<(usize, usize, u64), Arc<Array2<f32>>>> {
    EMBEDDING_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Resolve a shared embedding table by (vocab_size, hidden_size, seed).
/// If the table was already created, returns the cached `Arc<Array2<f32>>`.
/// Otherwise creates, caches, and returns it.
pub fn resolve_embedding(
    vocab_size: usize,
    hidden_size: usize,
    seed: u64,
) -> Arc<Array2<f32>> {
    let key = (vocab_size, hidden_size, seed);

    if let Ok(cache) = registry().read() {
        if let Some(embed) = cache.get(&key) {
            return Arc::clone(embed);
        }
    }

    let scale = (hidden_size as f32).sqrt().recip();
    let mut rng = rand::thread_rng();
    let arr = Array2::from_shape_fn((vocab_size, hidden_size), |_| {
        rng.gen::<f32>() * 2.0 * scale - scale
    });
    let shared = Arc::new(arr);

    if let Ok(mut cache) = registry().write() {
        let entry = cache.entry(key).or_insert_with(|| Arc::clone(&shared));
        return Arc::clone(entry);
    }

    shared
}

/// Number of cached embedding tables.
pub fn registry_size() -> usize {
    registry().read().map(|r| r.len()).unwrap_or(0)
}

/// Clear all cached embeddings — frees memory.
pub fn clear_embedding_registry() {
    if let Ok(mut cache) = registry().write() {
        cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_key_returns_same_pointer() {
        clear_embedding_registry();
        let a = resolve_embedding(100, 32, 42);
        let b = resolve_embedding(100, 32, 42);
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn test_different_key_returns_different() {
        clear_embedding_registry();
        let a = resolve_embedding(100, 32, 42);
        let b = resolve_embedding(200, 32, 42);
        assert!(!Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn test_different_seed_returns_different() {
        clear_embedding_registry();
        let a = resolve_embedding(100, 32, 42);
        let b = resolve_embedding(100, 32, 99);
        assert!(!Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn test_shape_is_correct() {
        let embed = resolve_embedding(50, 16, 0);
        assert_eq!(embed.dim(), (50, 16));
    }
}
