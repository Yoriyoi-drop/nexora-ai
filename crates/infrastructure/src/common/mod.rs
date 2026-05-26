use std::time::{SystemTime, UNIX_EPOCH};

pub fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub fn unix_timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

pub fn format_duration(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    format!("{}h {}m {}s", h, m, s)
}

pub fn get_process_memory_mb() -> f64 {
    let status = tokio::task::block_in_place(|| std::fs::read_to_string("/proc/self/status"));
    if let Ok(status) = status {
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    if let Ok(kb) = kb_str.parse::<f64>() {
                        return kb / 1024.0;
                    }
                }
            }
        }
    }
    0.0
}

pub fn get_cpu_usage_percent() -> f64 {
    let stat = tokio::task::block_in_place(|| std::fs::read_to_string("/proc/stat"));
    if let Ok(stat) = stat {
        if let Some(cpu_line) = stat.lines().next() {
            if cpu_line.starts_with("cpu ") {
                let parts: Vec<&str> = cpu_line.split_whitespace().collect();
                if parts.len() >= 8 {
                    let mut total_time = 0u64;
                    let mut idle_time = 0u64;
                    for (i, part) in parts.iter().enumerate().skip(1).take(7) {
                        if let Ok(time) = part.parse::<u64>() {
                            total_time += time;
                            if i == 3 {
                                idle_time = time;
                            }
                        }
                    }
                    if total_time > 0 {
                        return ((total_time - idle_time) as f64 / total_time as f64) * 100.0;
                    }
                }
            }
        }
    }
    0.0
}

pub fn get_load_average() -> (f64, f64, f64) {
    let load_str = tokio::task::block_in_place(|| std::fs::read_to_string("/proc/loadavg"));
    if let Ok(load_str) = load_str {
        let parts: Vec<&str> = load_str.split_whitespace().collect();
        if parts.len() >= 3 {
            let l1: f64 = parts[0].parse().unwrap_or(0.0);
            let l5: f64 = parts[1].parse().unwrap_or(0.0);
            let l15: f64 = parts[2].parse().unwrap_or(0.0);
            return (l1, l5, l15);
        }
    }
    (0.0, 0.0, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unix_timestamp_non_zero() {
        assert!(unix_timestamp() > 1_700_000_000);
    }

    #[test]
    fn test_unix_timestamp_millis_greater() {
        let secs = unix_timestamp();
        let millis = unix_timestamp_millis();
        assert!(millis > (secs as u128) * 1000);
    }

    #[test]
    fn test_format_duration_zero() {
        assert_eq!(format_duration(0), "0h 0m 0s");
    }

    #[test]
    fn test_format_duration_exact() {
        assert_eq!(format_duration(3661), "1h 1m 1s");
    }

    #[test]
    fn test_format_duration_one_hour() {
        assert_eq!(format_duration(3600), "1h 0m 0s");
    }

    #[test]
    fn test_format_duration_one_day() {
        assert_eq!(format_duration(86400), "24h 0m 0s");
    }

    #[test]
    fn test_get_process_memory_mb_type() {
        let mem: f64 = get_process_memory_mb();
        assert!(!mem.is_nan());
    }

    #[test]
    fn test_get_cpu_usage_type() {
        let cpu: f64 = get_cpu_usage_percent();
        assert!(!cpu.is_nan());
    }

    #[test]
    fn test_get_load_average_type() {
        let (l1, l5, l15): (f64, f64, f64) = get_load_average();
        assert!(!l1.is_nan());
        assert!(!l5.is_nan());
        assert!(!l15.is_nan());
    }
}
