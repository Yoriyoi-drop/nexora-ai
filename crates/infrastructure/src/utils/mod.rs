use std::collections::HashMap;

pub fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        text.to_string()
    } else {
        format!("{}...", &text[..max_chars.saturating_sub(3)])
    }
}

pub fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

pub fn merge_maps<K: std::hash::Hash + Eq + Clone, V: Clone>(
    base: &HashMap<K, V>,
    override_map: &HashMap<K, V>,
) -> HashMap<K, V> {
    let mut result = base.clone();
    for (k, v) in override_map {
        result.insert(k.clone(), v.clone());
    }
    result
}

pub fn byte_size_human(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;
    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }
    format!("{:.2} {}", size, UNITS[unit_idx])
}

pub fn duration_human_ms(ms: u64) -> String {
    if ms < 1000 {
        format!("{}ms", ms)
    } else if ms < 60_000 {
        format!("{:.2}s", ms as f64 / 1000.0)
    } else {
        format!("{}m {}s", ms / 60_000, (ms % 60_000) / 1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_truncate_text_short() {
        assert_eq!(truncate_text("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_text_long() {
        let result = truncate_text("hello world this is long", 10);
        assert!(result.len() <= 10);
        assert!(result.ends_with("..."));
    }

    #[test]
    fn test_truncate_text_exact() {
        assert_eq!(truncate_text("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_text_empty() {
        assert_eq!(truncate_text("", 5), "");
    }

    #[test]
    fn test_sanitize_filename_keeps_valid() {
        assert_eq!(sanitize_filename("test-file_v1.2"), "test-file_v1.2");
    }

    #[test]
    fn test_sanitize_filename_replaces_invalid() {
        assert_eq!(sanitize_filename("hello/world:test"), "hello_world_test");
    }

    #[test]
    fn test_sanitize_filename_all_invalid() {
        assert_eq!(sanitize_filename("!!!@@@###"), "_________");
    }

    #[test]
    fn test_sanitize_filename_empty() {
        assert_eq!(sanitize_filename(""), "");
    }

    #[test]
    fn test_merge_maps_empty_base() {
        let base = HashMap::new();
        let mut overrides = HashMap::new();
        overrides.insert("key1".to_string(), "val1".to_string());
        let result = merge_maps(&base, &overrides);
        assert_eq!(result.len(), 1);
        assert_eq!(result.get("key1").unwrap(), "val1");
    }

    #[test]
    fn test_merge_maps_overrides() {
        let mut base = HashMap::new();
        base.insert("a".to_string(), "1".to_string());
        base.insert("b".to_string(), "2".to_string());
        let mut overrides = HashMap::new();
        overrides.insert("b".to_string(), "3".to_string());
        overrides.insert("c".to_string(), "4".to_string());
        let result = merge_maps(&base, &overrides);
        assert_eq!(result.get("a").unwrap(), "1");
        assert_eq!(result.get("b").unwrap(), "3");
        assert_eq!(result.get("c").unwrap(), "4");
    }

    #[test]
    fn test_merge_maps_distinct_keys() {
        let mut base = HashMap::new();
        base.insert("x".to_string(), "10".to_string());
        let mut overrides = HashMap::new();
        overrides.insert("y".to_string(), "20".to_string());
        let result = merge_maps(&base, &overrides);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_byte_size_human_bytes() {
        assert_eq!(byte_size_human(500), "500.00 B");
    }

    #[test]
    fn test_byte_size_human_kb() {
        let result = byte_size_human(2048);
        assert!(result.contains("KB"));
    }

    #[test]
    fn test_byte_size_human_mb() {
        let result = byte_size_human(5_242_880);
        assert!(result.contains("MB"));
    }

    #[test]
    fn test_byte_size_human_gb() {
        let result = byte_size_human(5_368_709_120);
        assert!(result.contains("GB"));
    }

    #[test]
    fn test_byte_size_human_zero() {
        assert_eq!(byte_size_human(0), "0.00 B");
    }

    #[test]
    fn test_duration_human_ms_under_1s() {
        assert_eq!(duration_human_ms(500), "500ms");
    }

    #[test]
    fn test_duration_human_ms_seconds() {
        assert_eq!(duration_human_ms(2500), "2.50s");
    }

    #[test]
    fn test_duration_human_ms_minutes() {
        assert_eq!(duration_human_ms(125_000), "2m 5s");
    }

    #[test]
    fn test_duration_human_ms_exact_second() {
        assert_eq!(duration_human_ms(1000), "1.00s");
    }

    #[test]
    fn test_duration_human_ms_exact_minute() {
        assert_eq!(duration_human_ms(60_000), "1m 0s");
    }

    #[test]
    fn test_duration_human_ms_zero() {
        assert_eq!(duration_human_ms(0), "0ms");
    }
}
